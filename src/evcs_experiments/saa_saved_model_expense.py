"""Explicit first saved-model recovery parent, sunk expense and terminal gate.

This does not reinterpret a failed two-item queue as an old one-item ledger.
One deterministic claim survives every failure. It never rearms old queues.
"""
from __future__ import annotations
from contextlib import contextmanager
import argparse, ast, hashlib, importlib.util, inspect, json, math, os, sys, time
from pathlib import Path
from types import FunctionType
from evcs_experiments import campaign_batch as batch
from evcs_experiments.random_streams import canonical_hash

KIND='SAA_SAVED_MODEL_RECOVERY_ADMISSION_V1'
REVIEW='SAA_SAVED_MODEL_RECOVERY_DISPATCH_REVIEW_V1'
RESERVATION='SAA_SAVED_MODEL_RECOVERY_RESERVATION_V1'
EXECUTION='SAA_SAVED_MODEL_RECOVERY_EXECUTION_V1'

def need(ok,message):
    if not ok:raise ValueError(message)

def read(ref):
    batch.verify_reference(ref);return batch.read_json(ref['path'])

def new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return batch.file_reference(path)

def number(x):
    need(type(x) in (int,float) and math.isfinite(x) and x>=0,'Invalid expense number');return x

def verify_opening(fee_reference,isolation_decision_reference,original_worker_reference):
    f,d,w=read(fee_reference),read(isolation_decision_reference),read(original_worker_reference)
    need(f['kind']=='SAA_ASSEMBLED_MASTER_FAILED_PARENT_EXPENSE_V1' and f['passed'] is True and f['accounting_PASS'] is True and f['status']=='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION','Original failed expense is not accepted')
    need(d['kind']=='SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1' and d['old_queue_permanently_quarantined'] is True and d['original_queue_rearm_prohibited'] is True and d['terminal_fee_settlement_reference']==fee_reference,'Old mixed queue not quarantined')
    need(f['references']['worker_request']==original_worker_reference and w['method_id']=='SAA' and w['domain'] in ('formal','development'),'Wrong original SAA worker')
    q=read(f['original_queue_reference']);s=read(f['closing_snapshot_reference']);j=read(f['terminal_job_result_reference'])
    need(d['original_queue_reference']==f['original_queue_reference'] and d['original_queue_snapshot_reference']==f['closing_snapshot_reference'],'Isolation changes expense origin')
    need(len(q['items'])==2 and q['items'][0]['request_reference']==original_worker_reference and q['items'][0]['plan_id']==w['plan_id'],'Original mixed queue changed')
    need(s==batch.read_json(Path(q['queue_root'])/'state.json') and s['request_reference']==f['original_queue_reference'] and s['attempt_receipts']==[] and s['halted']=='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW','Quarantined queue changed or rearmed')
    need(j['complete'] is True and j['state']=='FAILED' and j['returncode']==2 and f['plan_id']==w['plan_id'],'Wrong failed parent')
    amount=number(f['SAA_charged_seconds']);cap=number(w['budget']['cumulative_wall_seconds'])
    need(0<amount<cap and amount==j['elapsed_seconds']==float(f['cumulative_accounted_seconds_decimal']) and f['SAA_remaining_seconds']==cap-amount and f['ARO_charged_seconds']==0 and f['ARO_no_invocation'] is True and f['ARO_no_worker_outputs'] is True and f['parent_charged_once'] is True and f['nested_seconds_added'] is False,'Original debt reset or double counted')
    need(q['items'][0]['plan_budget_seconds']==cap and float(f['opening_cumulative_seconds_decimal'])==0,'Original budget/opening differs')
    if w['domain']=='formal':need(w['plan_id']=='risk_model.helene_2024.SAA.baseline' and amount==2065.692102 and cap==7200,'Only current Helene formal debt is supported')
    return dict(reference=fee_reference,plan_id=w['plan_id'],accounted_seconds=amount,budget_seconds=cap,remaining_seconds=cap-amount,domain=w['domain'])

def _admission(ref):
    a=read(ref);need(a['kind']==KIND and a['domain'] in ('formal','development'),'Wrong recovery admission')
    need(a['expense_module_reference']==batch.file_reference(__file__),'Expense source changed')
    review=read(a['independent_review_reference'])
    need(review['kind']==REVIEW and review['passed'] is True and review['admission_payload_sha256']==canonical_hash({k:v for k,v in a.items() if k!='independent_review_reference'}) and all(review[k]==a[k] for k in ('expense_module_reference','worker_module_reference')),'Unreviewed recovery source/admission')
    for k in ('worker_module_reference','original_queue_source_reference'):batch.verify_reference(a[k])
    need(a['original_queue_source_reference']==batch.file_reference(batch.__file__),'Original queue source differs')
    policy=read(a['policy_reference']);opening=verify_opening(policy['fee_reference'],policy['isolation_decision_reference'],policy['original_worker_reference'])
    q,qr=batch.checked_request(a['queue_reference']['path']);w=read(a['worker_request_reference'])
    need(qr==a['queue_reference'] and len(q['items'])==1 and q['domain']==a['domain']==opening['domain']==w['domain'],'Recovery requires one explicit same-domain SAA item')
    item=q['items'][0]
    need(item['request_reference']==a['worker_request_reference'] and item['plan_id']==opening['plan_id']==w['plan_id'] and item['plan_budget_seconds']==opening['budget_seconds']==w['budget']['cumulative_wall_seconds'],'Recovery plan/budget changed')
    need(item.get('failure_carryover_reference')==w.get('failure_carryover_reference')==opening['reference'],'Explicit saved-model expense missing from queue/worker')
    need(q['parent_wall_policy']=='verified_parent_remaining_v1' and a['original_python']==sys.executable,'Parent clock or interpreter differs')
    need(Path(q['queue_root']).is_relative_to(Path(a['campaign_root'])) and Path(a['parent_execution_path']).is_relative_to(Path(a['campaign_root'])/'operations'),'Cross-campaign recovery')
    need(isinstance(a['parent_job_id'],str) and 0<number(a['parent_timeout_seconds']) and number(a['parent_grace_seconds'])+a['parent_timeout_seconds']<=opening['remaining_seconds'],'Parent exceeds original remaining expense')
    need(item['argv']==[a['original_python'],a['expense_module_reference']['path'],'child'],'Queue must dispatch exact explicit child entry')
    if a['domain']=='formal':need('development_entry_reference' not in a and all(not Path(a[k]['path']).is_relative_to(Path(batch.__file__).parent) for k in ('worker_module_reference','expense_module_reference')),'Formal cannot inject fixtures or change frozen package')
    return a,q,opening

def parent_argv(a,ref):return [a['original_python'],a['expense_module_reference']['path'],'parent','--admission',ref['path'],'--expected-sha256',ref['sha256']]

def actual_argv():
    p=Path('/proc/self/cmdline');return p.read_bytes().decode().rstrip('\0').split('\0') if p.exists() else [sys.executable,*sys.argv]

def claim_path(a,opening):
    fee=read(opening['reference'])
    key=canonical_hash(dict(terminal_job_result_reference=fee['terminal_job_result_reference'],plan_id=opening['plan_id']))
    return Path(a['campaign_root']).resolve()/'operations/saa-saved-model-recovery-claims'/(key+'.json')

def _same_parent(current,original):
    if any(current.get(k)!=original.get(k) for k in ('job_id','directory','manifest_reference')):return False
    a,b=current.get('wall_clock',{}),original.get('wall_clock',{})
    if a.get('clock_kind')!=b.get('clock_kind') or a.get('timeout_seconds')!=b.get('timeout_seconds'):return False
    left,right=a.get('supervisor',{}),b.get('supervisor',{})
    return all(left.get(k)==right.get(k) for k in ('pid','token','sid','pgid'))

def reserve_parent(admission_reference,parent_job):
    a,q,o=_admission(admission_reference);manifest=read(parent_job['manifest_reference']);jr=dict(job_id=a['parent_job_id'],argv=parent_argv(a,admission_reference),outroot=str(Path(a['campaign_root'])/'jobs'),threads=1,timeout_seconds=a['parent_timeout_seconds'],grace_seconds=a['parent_grace_seconds'])
    need(all(manifest[k]==v for k,v in jr.items()) and parent_job['job_id']==jr['job_id'] and parent_job['directory']==str(Path(a['campaign_root'])/'jobs'/jr['job_id']),'Actual parent differs')
    need(actual_argv()==jr['argv'] and os.environ.get('EVCS_JOB_ID')==jr['job_id'] and os.environ.get('EVCS_JOB_DIRECTORY')==parent_job['directory'],'Parent was not executed by admitted argv')
    clock=parent_job.get('wall_clock',{});need(clock.get('clock_kind')=='linux_supervisor_birth_boottime_v1','Original verified supervisor clock required')
    for status in (Path(a['campaign_root'])/'jobs').glob('*/status.json'):
        value=batch.read_json(status)
        if value['job_id']!=jr['job_id']:
            from evcs_experiments.runtime.jobctl import observation
            need(not observation(value)['blocks_new_launch'],'Another parent is active')
    path=claim_path(a,o)
    if not path.exists():
        # A launch that failed even before this claim still spent parent time.
        # A renamed admission cannot hide that earlier unreserved invocation.
        for manifest_path in (Path(a['campaign_root'])/'jobs').glob('*/manifest.json'):
            previous=batch.read_json(manifest_path);argv=previous.get('argv',[])
            if previous.get('job_id')==jr['job_id'] or len(argv)!=7 or argv[2:4]!=['parent','--admission'] or Path(argv[1]).name!=Path(__file__).name:continue
            prior_ref=batch.file_reference(argv[4]);need(prior_ref['sha256']==argv[6],'Prior saved-model parent admission changed; expense review required')
            prior_admission=read(prior_ref);prior_policy=read(prior_admission['policy_reference'])
            need(prior_policy['fee_reference']!=o['reference'],'Prior unreserved saved-model parent requires terminal expense review')
    path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.parent.is_symlink() and path.parent.resolve().is_relative_to(Path(a['campaign_root']).resolve()),'Aliased claim path')
    value=dict(kind=RESERVATION,admission_reference=admission_reference,policy_reference=a['policy_reference'],queue_reference=a['queue_reference'],worker_request_reference=a['worker_request_reference'],opening=o,parent_job=parent_job,expense_module_reference=a['expense_module_reference'],worker_module_reference=a['worker_module_reference'],actual_parent_argv=actual_argv(),process_identity=batch.process_identity(os.getpid()),parent_pid=os.getpid(),parent_execution_path=a['parent_execution_path'],development_entry_reference=a.get('development_entry_reference'))
    return new(path,value)

def verify_reservation(reservation_reference,*,invocation_reference=None,policy_reference=None):
    r=read(reservation_reference);need(r['kind']==RESERVATION,'Wrong saved-model reservation')
    a,q,o=_admission(r['admission_reference'])
    need(Path(reservation_reference['path'])==claim_path(a,o) and r['opening']==o and r['queue_reference']==a['queue_reference'] and r['worker_request_reference']==a['worker_request_reference'] and r['policy_reference']==a['policy_reference'],'Reservation/opening changed')
    need(r['expense_module_reference']==a['expense_module_reference'] and r['worker_module_reference']==a['worker_module_reference'] and r['actual_parent_argv']==parent_argv(a,r['admission_reference']),'Reservation source/argv changed')
    need(r.get('development_entry_reference')==a.get('development_entry_reference'),'Development fixture source differs from reservation')
    if policy_reference is not None:need(policy_reference==r['policy_reference'],'Foreign recovery policy')
    if invocation_reference is not None:
        execution=read(batch.file_reference(a['parent_execution_path']))
        need(execution.get('development_entry_reference')==a.get('development_entry_reference'),'Parent execution development fixture differs')
        need(execution['kind']==EXECUTION and execution['role']=='parent' and execution['admission_reference']==r['admission_reference'] and execution['reservation_reference']==reservation_reference and _same_parent(execution['parent_job'],r['parent_job']) and execution['actual_argv']==r['actual_parent_argv'] and execution['pid']==r['parent_pid'] and execution['process_identity']==r['process_identity'],'Parent execution proof differs')
        inv=read(invocation_reference)
        clock=execution['clock_binding'];observed=clock['observed_parent_clock']
        need(clock['original_queue_source_reference']==a['original_queue_source_reference'] and clock['replacement_count']==1 and clock['accounting_start_monotonic']==observed['observed_monotonic']-observed['elapsed_seconds'] and inv['attempt_started_monotonic']==clock['accounting_start_monotonic'],'Invocation did not include parent startup')
        need(inv['request_reference']==a['queue_reference'] and inv['item_request_reference']==a['worker_request_reference'] and inv['plan_id']==o['plan_id'] and _same_parent(inv['parent_job'],r['parent_job']),'Foreign invocation')
        need(inv['cumulative_plan_seconds']==o['accounted_seconds'] and inv['opening_carryover_seconds']==o['accounted_seconds'] and inv['failure_carryover_reference']==o['reference'] and inv['plan_remaining_seconds']==o['remaining_seconds'],'Invocation discarded original debt')
    return dict(**o,module_reference=batch.file_reference(__file__),admission_reference=r['admission_reference'],parent_execution_reference=(batch.file_reference(a['parent_execution_path']) if Path(a['parent_execution_path']).is_file() else None),reservation_reference=reservation_reference)

def _load_worker(a):
    name='evcs_experiments._explicit_saved_model_worker';ref=a['worker_module_reference'];batch.verify_reference(ref)
    module=sys.modules.get(name)
    if module is None:
        spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    need(batch.file_reference(module.__file__)==ref,'Loaded saved-model worker source differs');return module

@contextmanager
def _development_scope(a):
    """Explicit process-test fixture; the formal admission rejects this key.

    Numerical preparation uses the worker's existing private fixture type. The
    separately pinned context supplies only declared development process metadata
    on hosts without the production Linux supervisor clock.
    """
    if a['domain']=='formal':
        need('development_entry_reference' not in a,'Formal cannot inject a development entry')
        yield None
        return
    ref=a.get('development_entry_reference')
    need(ref is not None,'Actual development dispatch requires an explicit fixture source')
    batch.verify_reference(ref)
    name='evcs_experiments._saved_model_development_entry'
    need(name not in sys.modules,'Development entry already active')
    spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        from evcs_experiments.saa_planning_worker import _DevelopmentFixture
        fixture=module.fixture(a)
        need(isinstance(fixture,_DevelopmentFixture),'Development entry must return existing private SAA fixture')
        need(batch.file_reference(module.__file__)==ref,'Development fixture changed while loading')
        with module.process_metadata(a):
            yield fixture
    finally:
        sys.modules.pop(name,None)

@contextmanager
def _validator_scope(a):
    if a['domain']=='development' and read(a['policy_reference']).get('validator_reference') is None:
        yield
        return
    from evcs_experiments import saa_validator_recovery as installer
    policy=read(a['policy_reference']);source=policy['original_validator_installer_reference']
    batch.verify_reference(source);batch.verify_reference(policy['validator_reference'])
    original=read(policy['original_worker_reference'])
    need(source==batch.file_reference(installer.__file__) and source in original['planning_source_references'],'Original validator installer changed')
    need(policy['validator_reference']['sha256']=='f49ac95b15c1d45b58ee51ab852f863beb4c11096848550c6c4f05f8d394b20a','Wrong typed validator')
    restore=installer.install_validator(dict(validator_reference=policy['validator_reference']))
    try:yield
    finally:restore()

def _clock_binding(source_reference,clock):
    text=Path(source_reference['path']).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run_batch')
    source=''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]);old='started=time.monotonic();root=Path(outputroot).resolve();root.mkdir(parents=True,exist_ok=True)';replacement=old.replace('time.monotonic()','_saved_model_parent_start()')
    need(source.count(old)==1 and inspect.getsource(batch.run_batch)==source,'Original queue function changed')
    derived=source.replace(old,replacement);start=clock['observed_monotonic']-clock['elapsed_seconds'];need(number(clock['elapsed_seconds'])>=0 and math.isfinite(start) and start<=time.monotonic(),'Invalid parent clock')
    return derived,start,dict(original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),derived_function_sha256=hashlib.sha256(derived.encode()).hexdigest(),replacement_count=1,observed_parent_clock=clock,accounting_start_monotonic=start,original_queue_source_reference=source_reference)

def _run_parent(ref,fixture):
    a,q,o=_admission(ref);parent=batch._parent(q);reservation=reserve_parent(ref,parent)
    # Claim is durable before importing the new worker or checking saved model.
    worker=_load_worker(a);worker.verify_policy(a['policy_reference'],a['worker_request_reference'],_development_fixture=fixture)
    need(not Path(q['queue_root']).exists(),'Recovery first queue must be fresh; claim remains consumed')
    derived,start,clock=_clock_binding(a['original_queue_source_reference'],parent['wall_clock'])
    parent_ref=new(a['parent_execution_path'],dict(kind=EXECUTION,role='parent',admission_reference=ref,reservation_reference=reservation,parent_job=parent,actual_argv=actual_argv(),pid=os.getpid(),process_identity=batch.process_identity(os.getpid()),clock_binding=clock,validator_reference=read(a['policy_reference']).get('validator_reference'),original_validator_installer_reference=read(a['policy_reference']).get('original_validator_installer_reference'),development_entry_reference=a.get('development_entry_reference')))
    original_run,original_carry=batch.run_batch,batch._carryovers
    need('_saved_model_parent_start' not in batch.__dict__,'Queue clock already installed')
    used=False
    def start_once():
        nonlocal used
        need(not used,'Queue invocation already consumed');used=True;return start
    def carry(request):
        need(request==q,'Foreign queue attempted saved-model expense');verify_reservation(reservation);return {o['plan_id']:o}
    namespace={};exec(compile(derived,__file__+'::saved_model_queue_clock','exec'),namespace)
    batch.__dict__['_saved_model_parent_start']=start_once
    batch.run_batch=FunctionType(namespace['run_batch'].__code__,batch.__dict__)
    batch._carryovers=carry
    os.environ['EVCS_SAVED_MODEL_ADMISSION']=ref['path'];os.environ['EVCS_SAVED_MODEL_ADMISSION_SHA256']=ref['sha256']
    os.environ['EVCS_SAVED_MODEL_RESERVATION']=reservation['path'];os.environ['EVCS_SAVED_MODEL_RESERVATION_SHA256']=reservation['sha256']
    try:
        with _validator_scope(a):
            return batch.run_batch(a['queue_reference']['path'],q['queue_root'])
    finally:
        batch.run_batch=original_run;batch._carryovers=original_carry;batch.__dict__.pop('_saved_model_parent_start',None)

def _run_child(admission_reference,fixture):
    a,q,o=_admission(admission_reference);rp=os.environ['EVCS_SAVED_MODEL_RESERVATION'];rr=batch.file_reference(rp);need(rr['sha256']==os.environ['EVCS_SAVED_MODEL_RESERVATION_SHA256'],'Reservation changed before child')
    invref=batch.file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST']);need(invref['sha256']==os.environ['EVCS_BATCH_INVOCATION_SHA256'],'Invocation changed before child')
    verified=verify_reservation(rr,invocation_reference=invref,policy_reference=a['policy_reference']);r=read(rr);inv=read(invref)
    need(actual_argv()==inv['argv']==q['items'][0]['argv'] and os.getppid()==r['parent_pid'],'Child argv/parent differs')
    parent_identity=batch.process_identity(os.getppid());need(parent_identity is not None and all(parent_identity[k]==r['process_identity'][k] for k in ('pid','token','sid','pgid')),'Parent birth identity differs')
    worker=_load_worker(a)
    with _validator_scope(a):
        return worker.run_attempt(invref,policy_reference=a['policy_reference'],reservation_reference=rr,expense_module=sys.modules[__name__],_development_fixture=fixture)

def run_parent(ref):
    a,_,_=_admission(ref)
    with _development_scope(a) as fixture:
        return _run_parent(ref,fixture)

def run_child(ref):
    a,_,_=_admission(ref)
    with _development_scope(a) as fixture:
        return _run_child(ref,fixture)

def settle_parent(reservation_reference,terminal_reference,closing_snapshot_reference=None):
    r=read(reservation_reference);a,q,o=_admission(r['admission_reference']);verify_reservation(reservation_reference)
    terminal=read(terminal_reference);job=Path(r['parent_job']['directory'])
    need(Path(terminal_reference['path'])==job/'result.json' and terminal['job_id']==r['parent_job']['job_id'] and terminal['complete'] is True,'Wrong or incomplete terminal parent')
    from evcs_experiments.runtime.jobctl import observation,TERMINAL
    status=batch.read_json(job/'status.json');need(terminal['state'] in TERMINAL and status['state']==terminal['state'] and not observation(status)['blocks_new_launch'],'Terminal process remains active')
    root=Path(q['queue_root'])
    if closing_snapshot_reference is None:
        need(not root.exists(),'Existing queue requires an exact closing snapshot')
        closing=dict(attempt_receipts=[],active=None,halted='PARENT_FAILED_BEFORE_QUEUE');spent={o['plan_id']:o['accounted_seconds']}
    else:
        closing=read(closing_snapshot_reference);need(Path(closing_snapshot_reference['path'])!=root/'state.json' and closing==batch.read_json(root/'state.json'),'Exact immutable closing snapshot required')
        with _validator_scope(a):
            latest,spent=batch._history(closing,a['queue_reference'],root,opening={o['plan_id']:o})
        need(set(spent)=={o['plan_id']},'Foreign recovery charge')
    delta=number(spent[o['plan_id']]-o['accounted_seconds']);wall=number(terminal['elapsed_seconds']);total=o['accounted_seconds']+max(delta,wall)
    receipt_count=len(closing['attempt_receipts']);need(receipt_count<=1,'First saved-model reservation produced repeated attempts')
    for ref in closing['attempt_receipts']:
        receipt=read(ref);inv=read(receipt['invocation_reference']);verify_reservation(reservation_reference,invocation_reference=receipt['invocation_reference'],policy_reference=a['policy_reference'])
        need(_same_parent(inv['parent_job'],r['parent_job']),'Receipt belongs to another parent')
    accepted=receipt_count==1 and delta>=wall and total<=o['budget_seconds'] and not closing.get('active') and not closing.get('halted') and terminal['state']=='SUCCEEDED' and terminal['returncode']==0
    result=dict(kind='SAA_SAVED_MODEL_RECOVERY_TERMINAL_EXPENSE_V1',reservation_reference=reservation_reference,terminal_job_result_reference=terminal_reference,closing_snapshot_reference=closing_snapshot_reference,original_failed_parent_expense_reference=o['reference'],original_spent_seconds=o['accounted_seconds'],new_queue_charge_seconds=delta,actual_parent_seconds=wall,new_accounted_seconds=max(delta,wall),cumulative_spent_seconds=total,remaining_seconds=max(0.,o['budget_seconds']-total),uncovered_parent_seconds=max(0.,wall-delta),budget_within_limit=total<=o['budget_seconds'],passed=accepted,status='PASS_PARENT_COST_COVERAGE' if accepted else 'HOLD_FAILED_OR_UNCOVERED_PARENT_EXPENSE_RETAINED',automatic_resume_authorized=False,scientific_acceptance=False,formal_OOS=False)
    target=Path(reservation_reference['path']).with_name(Path(reservation_reference['path']).stem+'.terminal.json')
    if target.exists():need(read(batch.file_reference(target))==result,'Terminal expense already settled differently');return batch.file_reference(target)
    return new(target,result)

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['parent','child']);p.add_argument('--admission',type=Path);p.add_argument('--expected-sha256');args=p.parse_args();ref=batch.file_reference(args.admission or os.environ['EVCS_SAVED_MODEL_ADMISSION']);expected=args.expected_sha256 or os.environ['EVCS_SAVED_MODEL_ADMISSION_SHA256'];need(ref['sha256']==expected,'Admission hash changed')
    result=run_parent(ref) if args.role=='parent' else run_child(ref);print(json.dumps(result,allow_nan=False));return 0 if result.get('status') in ('COMPLETE','COMPLETE_BATCH','TIME_LIMIT','BUDGET_EXHAUSTED') else 2
if __name__=='__main__':raise SystemExit(main())
