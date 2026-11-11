"""Immutable supplemental parent-wall ledger; never edits the original SAA queue.

Queue receipts remain authoritative for Q. Each settled parent adds only
E=max(0,parent wall-new Q). Unsettled reservations prohibit another parent.
This component alone does not enable formal recovery dispatch or scientific PASS.
"""
from contextlib import contextmanager
import fcntl
import math
import json
import inspect
from pathlib import Path

from . import campaign_batch as batch
from .random_streams import canonical_hash
from .runtime.jobctl import observation, TERMINAL

ANCHOR='SAA_RECOVERY_EXTERNAL_LEDGER_V1'
RESERVATION='SAA_RECOVERY_PARENT_RESERVATION_V1'
SETTLEMENT='SAA_RECOVERY_PARENT_SETTLEMENT_V1'
SOURCE_UPGRADE_REFERENCE=None


def need(condition,message):
    if not condition:raise ValueError(message)


def read(reference):
    batch.verify_reference(reference)
    return batch.read_json(reference['path'])


def new(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
    return batch.file_reference(path)


def number(value, name):
    need(type(value) in (float,int) and math.isfinite(value) and value>=0, 'Invalid '+name)
    return value


def _snapshot(ref, queue_ref, *, inventory=False):
    state=read(ref);queue=read(queue_ref)
    root=Path(queue['queue_root'])
    need(Path(ref['path']).resolve()!=root/'state.json','Immutable queue snapshot required')
    need(state['request_reference']==queue_ref,'Foreign queue snapshot')
    refs=state['attempt_receipts']
    need(len({r['path'] for r in refs})==len(refs),'Duplicate queue receipt')
    _,spent=batch._history(state,queue_ref,root if inventory else None,batch._carryovers(queue))
    plan=queue['items'][0]['plan_id']
    need(set(spent)<={plan},'Foreign plan in queue ledger')
    return state,number(spent.get(plan,0.),'queue cumulative charge')


def _anchor(ref):
    value=read(ref)
    need(value['kind']==ANCHOR,'Wrong external ledger kind')
    root=Path(value['ledger_root'])
    need(root.is_absolute() and Path(ref['path'])==root/'anchor.json','Foreign ledger root')
    need(root.resolve()==root and not any(p.is_symlink() for p in root.iterdir()),'Ledger symlink aliases are forbidden')
    queue=read(value['queue_reference'])
    need(len(queue['items'])==1 and queue['items'][0]['plan_id']==value['plan_id'],'Ledger requires one fixed plan')
    need(root.is_relative_to(Path(value['campaign_root'])) and Path(queue['queue_root']).is_relative_to(Path(value['campaign_root'])), 'Cross-campaign ledger')
    need(value['budget_seconds']==queue['items'][0]['plan_budget_seconds'] and value['budget_seconds']>0,'Original plan cap differs')
    _,opening=_snapshot(value['initial_queue_snapshot_reference'],value['queue_reference'])
    need(opening==value['initial_queue_seconds'],'Original queue budget cannot reset')
    review=read(value['independent_review_reference'])
    need(review['kind']=='SAA_RECOVERY_LEDGER_ANCHOR_REVIEW_V1' and review['passed'] is True and
         review['anchor_payload_sha256']==canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'}),'Unreviewed original ledger anchor')
    current=batch.file_reference(__file__)
    if value['ledger_source_reference']!=current:
        upgrade=read(SOURCE_UPGRADE_REFERENCE)
        need(upgrade['kind']=='SAA_LEDGER_SOURCE_UPGRADE_V1' and upgrade['anchor_reference']==ref and
             upgrade['original_source_reference']==value['ledger_source_reference'] and
             upgrade['upgraded_source_reference']==current,'External ledger source upgrade differs')
        reviewed=read(upgrade['independent_review_reference'])
        need(reviewed['kind']=='SAA_LEDGER_SOURCE_UPGRADE_REVIEW_V1' and reviewed['passed'] is True and
             reviewed['upgrade_payload_sha256']==canonical_hash({k:v for k,v in upgrade.items() if k!='independent_review_reference'}),
             'Ledger source upgrade lacks independent review')
        batch.verify_reference(value['ledger_source_reference'])
    return value


@contextmanager
def _locked(root):
    with (root/'.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def _chain(anchor_ref):
    """Recompute all immutable arithmetic; filenames cannot hide an unsettled parent."""
    anchor=_anchor(anchor_ref);root=Path(anchor['ledger_root']);extra=0.;last=None
    claim=Path(anchor['campaign_root'])/'operations'/'saa-recovery-ledger-claims'/(canonical_hash(anchor['queue_reference'])+'.json')
    need(claim.exists() and read(batch.file_reference(claim))==dict(kind='SAA_RECOVERY_EXCLUSIVE_LEDGER_CLAIM_V1',queue_reference=anchor['queue_reference'],anchor_reference=anchor_ref),'Exclusive ledger claim missing or changed')
    rs=sorted(root.glob('reservation-*.json'));ss=sorted(root.glob('settlement-*.json'))
    need(len(ss)<=len(rs)<=len(ss)+1,'Unsettled or incomplete ledger chain')
    jobs=set();seen_receipts=set();expected_open=anchor['initial_queue_snapshot_reference']
    for i,rp in enumerate(rs,1):
        need(rp.name==f'reservation-{i:04d}.json','Reservation sequence changed')
        rr=batch.file_reference(rp);r=read(rr)
        need(r['kind']==RESERVATION and r['anchor_reference']==anchor_ref and r['previous_settlement_reference']==last,'Reservation chain changed')
        need(r['opening_queue_snapshot_reference']==expected_open,'Queue history between parents changed')
        need(r['job_id'] not in jobs,'Duplicate parent job');jobs.add(r['job_id'])
        _,q=_snapshot(expected_open,anchor['queue_reference'])
        need(r['opening_queue_seconds']==q and r['opening_extra_seconds']==extra and r['effective_spent_seconds']==q+extra,'Reservation accounting changed')
        _parent_contract(anchor,r)
        if i>len(ss):return anchor,extra,last,r,rr
        sp=root/f'settlement-{i:04d}.json';need(ss[i-1]==sp,'Settlement sequence changed')
        sr=batch.file_reference(sp);s=read(sr)
        expected=_settlement_values(anchor_ref,r,rr,s['terminal_job_result_reference'],s['closing_queue_snapshot_reference'],check_live=False)
        need(s==expected,'Settlement arithmetic or identity changed')
        for item in s['new_queue_receipt_references']:
            need(item['path'] not in seen_receipts,'Duplicate charged receipt');seen_receipts.add(item['path'])
        extra+=s['extra_parent_seconds'];last=sr;expected_open=s['closing_queue_snapshot_reference']
    return anchor,extra,last,None,None


def _parent_contract(anchor,r):
    manifest=read(r['parent_manifest_reference']);request=read(r['parent_job_request_reference'])
    job=Path(anchor['campaign_root'])/'jobs'/r['job_id']
    need(Path(r['parent_manifest_reference']['path'])==job/'manifest.json','Foreign parent manifest')
    need(manifest['job_id']==request['job_id']==r['job_id'] and Path(manifest['outroot'])==job.parent,'Parent job identity differs')
    for key in request:
        need(manifest[key]==request[key],'Parent request/manifest '+key+' differs')
    admission=read(r['admission_reference'])
    if admission.get('kind')=='SAA_L_SHAPED_PARENT_ADMISSION_V1':
        from .saa_l_shaped_parent import ledger_parent_contract
        return ledger_parent_contract(anchor,r,manifest,admission)
    if admission.get('kind')=='SAA_SAVED_DUAL_BOUND_ADMISSION_V1':
        return _bound_parent_contract(anchor,r,manifest,admission)
    need(admission['kind']=='SAA_VALIDATOR_RECOVERY_ADMISSION_V1' and admission['domain']==read(anchor['queue_reference'])['domain'],'Wrong recovery admission kind/domain')
    need(admission['queue_reference']==anchor['queue_reference'] and admission['queue_root']==read(anchor['queue_reference'])['queue_root'] and admission['parent_job_id']==r['job_id'] and Path(admission['parent_job_directory'])==job,'Foreign recovery admission')
    need(admission['opening_cumulative_seconds']==r['opening_queue_seconds'],'Admission reset queue opening')
    opening=read(r['opening_queue_snapshot_reference'])
    need(read(admission['baseline_ledger_reference'])==opening and admission['baseline_receipt_references']==opening['attempt_receipts'],'Admission baseline ledger differs')
    review=read(admission['independent_review_reference'])
    need(review['kind']=='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1' and review['passed'] is True and review['adapter_reference']==admission['adapter_reference'] and review['validator_reference']==admission['validator_reference'] and review['admission_payload_sha256']==canonical_hash({k:v for k,v in admission.items() if k!='independent_review_reference'}),'Recovery admission review differs')
    # The existing adapter remains the explicit numerical parent entry.
    expected=[admission['original_python'],admission['adapter_reference']['path'],'parent','--admission',r['admission_reference']['path'],'--expected-sha256',r['admission_reference']['sha256']]
    need(manifest['argv']==expected,'Parent actual argv differs from reviewed adapter')
    batch.verify_reference(admission['adapter_reference']);batch.verify_reference(admission['validator_reference'])
    timeout=number(manifest['timeout_seconds'],'parent timeout');grace=number(manifest['grace_seconds'],'parent grace')
    need(timeout>0 and timeout+grace<=anchor['budget_seconds']-r['effective_spent_seconds'],'Parent timeout plus grace exceeds effective remaining budget')
    need(timeout<=read(anchor['queue_reference'])['job_wall_seconds'],'Parent exceeds original queue cap')


def _bound_parent_contract(anchor,r,manifest,admission):
    """Explicit bound-only parent; unchanged queue, same append-only debt chain."""
    queue=read(anchor['queue_reference']);opening=read(r['opening_queue_snapshot_reference'])
    need(admission['domain']==queue['domain'] and admission['queue_reference']==anchor['queue_reference'] and
         admission['queue_root']==queue['queue_root'] and admission['parent_job_id']==r['job_id'] and
         Path(admission['parent_job_directory'])==Path(anchor['campaign_root'])/'jobs'/r['job_id'],'Foreign SAA bound-only parent')
    need(admission['opening_cumulative_seconds']==r['opening_queue_seconds'] and
         admission['opening_effective_seconds']==r['effective_spent_seconds'] and
         admission['expected_last_settlement_reference']==r['previous_settlement_reference'] and
         read(admission['baseline_ledger_reference'])==opening and admission['baseline_receipt_references']==opening['attempt_receipts'],
         'Bound-only admission reset original queue or external debt')
    review=read(admission['independent_review_reference'])
    need(review['kind']=='SAA_SAVED_DUAL_BOUND_PRODUCTION_REVIEW_V1' and review['passed'] is True and
         review['admission_payload_sha256']==canonical_hash({k:v for k,v in admission.items() if k!='independent_review_reference'}) and
         review['driver_reference']==admission['driver_reference'] and review['request_reference']==admission['request_reference'],
         'SAA bound-only production/source review differs')
    for key in ('driver_reference','request_reference'):batch.verify_reference(admission[key])
    expected=[admission['original_python'],admission['driver_reference']['path'],'--request',admission['request_reference']['path'],
              '--expected-sha256',admission['request_reference']['sha256'],'--admission',r['admission_reference']['path'],
              '--admission-sha256',r['admission_reference']['sha256']]
    need(manifest['argv']==expected,'Bound-only actual argv differs')
    request=read(admission['request_reference'])
    need(request['kind']=='S05_SAA_SAVED_DUAL_BOUND_ONLY_V1' and request['domain']==queue['domain'] and
         request['opening_effective_seconds']==r['effective_spent_seconds'] and request['solver_seconds']<=600 and
         request['solver_seconds']>0 and request['post_master_reserve_seconds']>=100,'Wrong SAA bound-only scientific/budget scope')
    timeout=number(manifest['timeout_seconds'],'bound parent timeout');grace=number(manifest['grace_seconds'],'bound parent grace')
    need(0<request['wall_seconds']<timeout<=1500 and timeout+grace<=anchor['budget_seconds']-r['effective_spent_seconds'] and
         manifest['threads']==1,'Bound-only parent exceeds original remaining allowance')


def claim_recovery_ledger(anchor_reference):
    """One append-only ledger per original queue, outside its frozen directory."""
    anchor=_anchor(anchor_reference)
    claims=Path(anchor['campaign_root'])/'operations'/'saa-recovery-ledger-claims'
    claims.mkdir(parents=True,exist_ok=True)
    need(not claims.is_symlink() and claims.resolve().is_relative_to(Path(anchor['campaign_root']).resolve()),'Foreign claim directory')
    path=claims/(canonical_hash(anchor['queue_reference'])+'.json')
    value=dict(kind='SAA_RECOVERY_EXCLUSIVE_LEDGER_CLAIM_V1',queue_reference=anchor['queue_reference'],anchor_reference=anchor_reference)
    try:new(path,value)
    except FileExistsError:need(read(batch.file_reference(path))==value,'Original queue already claimed by another external ledger')
    return batch.file_reference(path)


def reserve_recovery_parent(anchor_reference, *, parent_job_request_reference,
                            parent_manifest_reference, admission_reference,
                            opening_queue_snapshot_reference):
    """Call before dispatch into the adapter; record survives any pre-queue failure."""
    anchor=_anchor(anchor_reference);root=Path(anchor['ledger_root'])
    with _locked(root):
        claim_recovery_ledger(anchor_reference)
        anchor,extra,last,pending,_=_chain(anchor_reference)
        need(pending is None,'Unsettled recovery parent blocks restart')
        expected=anchor['initial_queue_snapshot_reference'] if last is None else read(last)['closing_queue_snapshot_reference']
        need(opening_queue_snapshot_reference==expected,'Opening snapshot must continue settled history')
        state,q=_snapshot(expected,anchor['queue_reference'],inventory=True)
        admission=read(admission_reference)
        if state.get('active') or state.get('halted'):
            ready=verify_queue_rearm(admission['queue_rearm_reference'],anchor_reference,last)
            need(ready['old_snapshot_reference']==expected,'Rearm does not start from latest closing snapshot')
            state=read(ready['ready_snapshot_reference'])
        need(not state.get('active') and not state.get('halted'),'Queue is active or halted')
        need(read(batch.file_reference(Path(read(anchor['queue_reference'])['queue_root'])/'state.json'))==state,'Live queue differs from immutable opening')
        manifest=read(parent_manifest_reference)
        r=dict(kind=RESERVATION,anchor_reference=anchor_reference,previous_settlement_reference=last,
          job_id=manifest['job_id'],parent_job_request_reference=parent_job_request_reference,
          parent_manifest_reference=parent_manifest_reference,admission_reference=admission_reference,
          opening_queue_snapshot_reference=expected,opening_queue_seconds=q,opening_extra_seconds=extra,
          effective_spent_seconds=q+extra)
        _parent_contract(anchor,r)
        for old in root.glob('reservation-*.json'):need(read(batch.file_reference(old))['job_id']!=r['job_id'],'Duplicate parent job')
        index=len(list(root.glob('reservation-*.json')))+1
        return new(root/f'reservation-{index:04d}.json',r)


def _settlement_values(anchor_ref,r,rr,terminal_ref,closing_ref,*,check_live):
    anchor=_anchor(anchor_ref);terminal=read(terminal_ref);manifest=read(r['parent_manifest_reference'])
    job=Path(r['parent_manifest_reference']['path']).parent
    need(Path(terminal_ref['path'])==job/'result.json' and terminal['job_id']==r['job_id'],'Wrong terminal parent')
    need(terminal.get('complete') is True and terminal['state'] in TERMINAL,'Parent is not terminal')
    if check_live:
        status=batch.read_json(job/'status.json');observed=observation(status)
        need(status['state']==terminal['state'] and not observed['blocks_new_launch'] and not observed.get('supervisor_alive') and not observed.get('command_alive'),'Parent still live or unreviewed')
    wall=number(terminal['elapsed_seconds'],'actual parent wall')
    opening,oq=_snapshot(r['opening_queue_snapshot_reference'],anchor['queue_reference'])
    closing,cq=_snapshot(closing_ref,anchor['queue_reference'],inventory=check_live)
    if check_live:need(batch.read_json(Path(read(anchor['queue_reference'])['queue_root'])/'state.json')==closing,'Closing snapshot differs from live queue')
    prefix=opening['attempt_receipts'];refs=closing['attempt_receipts']
    need(refs[:len(prefix)]==prefix and len(refs)>=len(prefix),'Original queue receipt prefix changed')
    newrefs=refs[len(prefix):]
    oldfinals=opening.get('finalization_receipts',[]);newfinals=closing.get('finalization_receipts',[])
    need(newfinals[:len(oldfinals)]==oldfinals,'Original finalization prefix changed')
    permitted={str(Path(x['path']).parent/'finalization.json') for x in newrefs}
    need(all(x['path'] in permitted for x in newfinals[len(oldfinals):]),'Closing allocation is not from a new parent receipt')
    for ref in newrefs:
        receipt=read(ref);inv=read(receipt['invocation_reference']);parent=inv['parent_job']
        need(parent['job_id']==r['job_id'] and parent['manifest_reference']==r['parent_manifest_reference'],'New receipt belongs to another parent')
    need(cq>=oq,'Queue cumulative accounting decreased')
    dq=cq-oq;extra=max(0.,wall-dq);effective=cq+r['opening_extra_seconds']+extra
    return dict(kind=SETTLEMENT,reservation_reference=rr,terminal_job_result_reference=terminal_ref,
      closing_queue_snapshot_reference=closing_ref,new_queue_receipt_references=newrefs,
      actual_parent_wall_seconds=wall,new_queue_charge_seconds=dq,extra_parent_seconds=extra,
      closing_queue_seconds=cq,cumulative_extra_seconds=r['opening_extra_seconds']+extra,
      effective_spent_seconds=effective,budget_seconds=anchor['budget_seconds'],
      budget_within_limit=effective<=anchor['budget_seconds'],scientific_acceptance=False,
      clock_scope='Original queue cumulative plus nonduplicated actual jobctl parent wall excess')


def settle_recovery_parent(reservation_reference, terminal_job_result_reference, closing_queue_snapshot_reference):
    r=read(reservation_reference);anchor=_anchor(r['anchor_reference']);root=Path(anchor['ledger_root'])
    with _locked(root):
        _chain(r['anchor_reference'])
        number_text=Path(reservation_reference['path']).stem.removeprefix('reservation-')
        need(Path(reservation_reference['path'])==root/f'reservation-{number_text}.json','Foreign reservation path')
        path=root/f'settlement-{number_text}.json'
        expected=_settlement_values(r['anchor_reference'],r,reservation_reference,terminal_job_result_reference,closing_queue_snapshot_reference,check_live=not path.exists())
        if path.exists():need(read(batch.file_reference(path))==expected,'Repeated terminal settlement changed')
        else:new(path,expected)
        return batch.file_reference(path)


def verify_settled_recovery_evidence(anchor_reference, settlement_reference, execution_reference):
    """Necessary supplementary gate for candidate/final acceptance, never sufficient."""
    anchor,extra,last,pending,_=_chain(anchor_reference)
    need(pending is None and last==settlement_reference,'Candidate/final requires latest settled ledger')
    s=read(last);need(s['budget_within_limit'],'Effective plan budget exceeded')
    r=read(s['reservation_reference']);execution=read(execution_reference)
    need(execution['kind']=='SAA_VALIDATOR_RECOVERY_EXECUTION_V1' and execution['admission_reference']==r['admission_reference'],'Execution/settled admission differs')
    need(execution['role'] in ('parent','child'),'Wrong execution role')
    parent=execution if execution['role']=='parent' else read(execution['parent_execution_reference'])
    need(parent['parent_job']['job_id']==r['job_id'] and parent['parent_job']['manifest_reference']==r['parent_manifest_reference'],'Execution/settled parent differs')
    return dict(kind='SAA_RECOVERY_SETTLED_BUDGET_GATE_V1',passed=True,anchor_reference=anchor_reference,
      settlement_reference=settlement_reference,execution_reference=execution_reference,
      ledger_source_reference=anchor['ledger_source_reference'],
      effective_spent_seconds=s['effective_spent_seconds'],remaining_seconds=anchor['budget_seconds']-s['effective_spent_seconds'],
      scientific_acceptance=False)


def verify_recovery_candidate_budget(anchor_reference, settlement_reference, candidate_reference,
                                     *, final_solver_record_reference=None, execution_result_validator=None):
    """Bind exact new candidate/final record to the settled parent, supplementary only.

    This must accompany (not replace) the original numerical and final-record gates.
    The existing adapter's formal guard is deliberately still effective here.
    """
    candidate=read(candidate_reference)
    execution=read(candidate['validation_recovery_execution_reference'])
    admission=read(execution['admission_reference'])
    if execution_result_validator is None:
        need(admission['domain']=='development','Formal candidate requires explicitly loaded reviewed execution validator')
        from .saa_validator_recovery import validate_execution_result
        execution_result_validator=validate_execution_result
    source=inspect.getsourcefile(execution_result_validator)
    need(source is not None and batch.file_reference(source)==admission['adapter_reference']
         and execution_result_validator.__name__=='validate_execution_result'
         and execution_result_validator.__globals__.get('validate_execution_result') is execution_result_validator
         and batch.file_reference(execution_result_validator.__globals__['__file__'])==admission['adapter_reference'],
         'Execution validator source or callable differs')
    gate=verify_settled_recovery_evidence(anchor_reference,settlement_reference,
                                        candidate['validation_recovery_execution_reference'])
    anchor=_anchor(anchor_reference);s=read(settlement_reference);matches=[]
    need(candidate['plan_id']==anchor['plan_id'] and candidate['method_id']=='SAA','Foreign recovery candidate')
    for ref in s['new_queue_receipt_references']:
        receipt=read(ref);result=read(receipt['result_reference'])
        numerical=result.get('numerical_audit_reference')
        if numerical is not None and read(numerical).get('candidate_record_reference')==candidate_reference:
            execution_result_validator(receipt['result_reference'],receipt['invocation_reference'])
            need(result['validation_recovery_execution_reference']==candidate['validation_recovery_execution_reference'],'Candidate/result execution receipt differs')
            matches.append(ref)
    need(len(matches)==1,'Candidate not uniquely produced by the settled parent')
    if final_solver_record_reference is not None:
        final=read(final_solver_record_reference)
        need(final['candidate_record_reference']==candidate_reference and final['receipt_reference']==matches[0]
             and final['plan_id']==anchor['plan_id'],'Final record is not the settled candidate')
        # _history already checks the finalization seal and its raw queue charge;
        # identify this exact solver record in that immutable original seal.
        closing=read(s['closing_queue_snapshot_reference'])
        finals=[read(r) for r in closing.get('finalization_receipts',[]) if read(r)['receipt_reference']==matches[0]]
        need(len(finals)==1 and finals[0]['references']['solver_record']==final_solver_record_reference,'Final record missing from original seal')
    return dict(gate,candidate_reference=candidate_reference,receipt_reference=matches[0],
                final_solver_record_reference=final_solver_record_reference)


def verify_queue_rearm(reference,anchor_reference,expected_last):
    """Only clear dead terminal operational markers; never change evidence or Q."""
    value=read(reference)
    need(set(value)=={'kind','anchor_reference','settlement_reference','old_snapshot_reference','ready_snapshot_reference',
          'terminal_result_reference','terminal_manifest_reference','independent_review_reference'} and
          value['kind']=='SAA_SETTLED_QUEUE_REARM_V1' and value['anchor_reference']==anchor_reference and
          value['settlement_reference']==expected_last,'Queue rearm scope/head differs')
    review=read(value['independent_review_reference'])
    need(review['kind']=='SAA_SETTLED_QUEUE_REARM_REVIEW_V1' and review['passed'] is True and
          review['rearm_payload_sha256']==canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'}),
          'Queue rearm is not independently reviewed')
    settlement=read(expected_last);reservation=read(settlement['reservation_reference'])
    need(reservation['anchor_reference']==anchor_reference and settlement['terminal_job_result_reference']==value['terminal_result_reference'] and
          reservation['parent_manifest_reference']==value['terminal_manifest_reference'] and
          settlement['closing_queue_snapshot_reference']==value['old_snapshot_reference'], 'Rearm terminal/settlement binding differs')
    old=read(value['old_snapshot_reference']);ready=read(value['ready_snapshot_reference'])
    need(old.get('active') and old.get('halted')=='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW' and
          ready==dict(old,active=None,halted=None),'Rearm changed queue evidence beyond terminal markers')
    result=read(value['terminal_result_reference']);manifest=read(value['terminal_manifest_reference'])
    job=Path(value['terminal_manifest_reference']['path']).parent
    need(result.get('complete') is True and result['state'] in TERMINAL and result['job_id']==manifest['job_id'] and
          Path(value['terminal_result_reference']['path'])==job/'result.json','Rearm parent is not terminal')
    observed=observation(batch.read_json(job/'status.json'))
    need(not observed['blocks_new_launch'] and not observed.get('supervisor_alive') and not observed.get('command_alive') and
          not batch.identity_alive(old['active'].get('process')),'Rearm process is still alive')
    invocation=read(old['active']['invocation_reference'])
    need(invocation['parent_job']['manifest_reference']==value['terminal_manifest_reference'] and
          invocation['parent_job']['job_id']==result['job_id'],'Rearm active marker belongs to another parent')
    return value


def apply_queue_rearm(reference,anchor_reference):
    """Explicit reviewed administrative transition, preserving old bytes externally."""
    anchor,extra,last,pending,_=_chain(anchor_reference)
    need(pending is None,'Unsettled parent blocks rearm')
    value=verify_queue_rearm(reference,anchor_reference,last)
    root=Path(read(anchor['queue_reference'])['queue_root']);statepath=root/'state.json'
    with batch._lock(root):
        need(batch.file_reference(statepath)['sha256']==value['old_snapshot_reference']['sha256'] and
              batch.file_reference(statepath)['size_bytes']==value['old_snapshot_reference']['size_bytes'],
              'Live queue changed before rearm; no repeat apply')
        from .runtime.jobctl import atomic_json
        atomic_json(statepath,read(value['ready_snapshot_reference']))
    return batch.file_reference(statepath)
