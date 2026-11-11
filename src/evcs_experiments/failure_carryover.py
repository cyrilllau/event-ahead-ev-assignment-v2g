"""Explicit failed-pre-RMP cost settlement; never a checkpoint migration.

A caller must verify this receipt before dispatch and compare the newly prepared
problem before solving. This module does not launch work or alter old evidence.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .random_streams import canonical_hash
from .runtime.jobctl import observation

KIND='FAILED_PRE_RMP_CARRYOVER_V1'
ENGINE_FILES=('ccg.py','extensive.py','first_stage.py','recourse.py','probability.py',
 'probability_phase1.py','random_streams.py','lazy_recourses.py','zero_presolve.py','native_highs.py','native_gurobi.py')
ALLOWED_CHANGED_SOURCE={'failure_carryover.py','campaign_batch.py','formal_planning_worker.py','formal_physical_inputs.py'}


def _need(ok,message):
    if not ok:raise ValueError(message)


def reference(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)


def _verify(ref):
    _need(isinstance(ref,dict) and set(ref)=={'path','sha256','size_bytes'},'Malformed carryover reference')
    _need(Path(ref['path']).is_absolute() and reference(ref['path'])==ref,'Carryover reference changed')
    return Path(ref['path'])


def _read(ref):
    return json.loads(_verify(ref).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def _release(ref):
    p=_verify(ref);rows=_read(ref)['files'];root=p.parent;seen=set();sources={}
    for row in rows:
        q=(root/row['path']).resolve()
        _need(q.is_relative_to(root) and str(q) not in seen,'Release inventory path/duplicate')
        seen.add(str(q));a=reference(q)
        _need(a['sha256']==row['sha256'] and a['size_bytes']==row['size_bytes'],'Release member changed')
        prefix='src/evcs_experiments/'
        if row['path'].startswith(prefix) and row['path'].endswith('.py'):
            sources[row['path'][len(prefix):]]=row['sha256']
    _need(set(sources)=={str(p.relative_to(root/'src/evcs_experiments')) for p in (root/'src/evcs_experiments').rglob('*.py')}, 'Release Python inventory differs')
    _need(set(ENGINE_FILES)<=set(sources),'Incomplete engine source closure')
    return sources,canonical_hash({n:sources[n] for n in ENGINE_FILES})


def scientific_identity(identity):
    """Only the explicit adapter provenance map may differ across segments."""
    result=json.loads(json.dumps(identity))
    binding=result['lazy_input_binding']['binding']
    _need(isinstance(binding.pop('planning_source'),dict),'Missing planning source identity')
    return result


def verify_prepared_problem(verified,identity):
    _need(scientific_identity(identity)==scientific_identity(verified['old_problem_identity']),
          'Carryover scientific problem identity changed')


def _claim_path(record):
    result=Path(record['references']['job_result']['path'])
    _need(result.name=='result.json' and result.parents[1].name=='jobs','Unknown campaign job origin')
    key=canonical_hash(record['references']['job_result'])
    return result.parents[2]/'recovery_claims'/(key+'.json')


def verify_failure_carryover(ref,worker_request,queue_root,*,require_claim=True):
    record=_read(ref);refs=record['references']
    _need(record.get('schema_version')==1 and record.get('kind')==KIND,'Unknown carryover schema')
    data={k:_read(v) for k,v in refs.items()}
    _need(record['successor_queue_root']==str(Path(queue_root).resolve()) and
          record['successor_output_root']==worker_request['output_root'],'Carryover successor differs')
    _need(record['new_release_reference']==worker_request['release_reference'],'Carryover successor source differs')
    old_sources,old_engine=_release(record['old_release_reference'])
    new_sources,new_engine=_release(record['new_release_reference'])
    changed={n for n in set(old_sources)|set(new_sources) if old_sources.get(n)!=new_sources.get(n)}
    _need(set(old_sources)<=set(new_sources) and changed<=ALLOWED_CHANGED_SOURCE and old_engine==new_engine,'Carryover changes unapproved engine/source')
    _need(record['source_changes']=={n:dict(old=old_sources.get(n),new=new_sources.get(n)) for n in sorted(changed)},
          'Carryover source diff differs')
    old=data['worker_request'];q=data['queue_request'];job=data['job_result'];jm=data['job_manifest'];bs=data['batch_state'];run=data['batch_result'];inv=data['invocation'];failure=data['failure'];wf=data['worker_failure']
    for key in ('domain','spec_id','model_id','plan_id','target_id','method_id','implementation_method_mapping',
                'budget','protocol_reference','summary_reference','audit_reference','bank_acceptance_reference','frozen_plan_identity'):
        _need(old[key]==worker_request[key],'Carryover plan/input/budget changed: '+key)
    _need(_read(old['config_reference'])==_read(worker_request['config_reference']), 'Carryover scientific configuration changed')
    _need(old['domain']=='formal' and old['spec_id']=='S05' and old['budget']['cumulative_wall_seconds']==7200,
          'Carryover only supports the frozen formal S05 budget')
    _need(record['plan_id']==old['plan_id'],'Carryover plan differs')
    _need(old['release_reference']==record['old_release_reference'],'Predecessor source differs')
    _need(len(q['items'])==1 and q['items'][0]['plan_id']==old['plan_id'] and
          q['items'][0]['request_reference']==refs['worker_request'],'Predecessor must be one fixed plan')
    _need(q['queue_root']==str(Path(refs['batch_state']['path']).parent),'Predecessor queue origin differs')
    _need(bs['attempt_receipts']==[] and bs['halted']=='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW' and
          bs['active']['invocation_reference']==refs['invocation'] and bs['runs']==[refs['batch_result']],
          'Predecessor is not the one unresolved first failure')
    _need(inv['cumulative_plan_seconds']==0 and inv['previous_checkpoint'] is None and inv['previous_result_reference'] is None, 'Only first failed attempt may be settled')
    attempts=list(Path(q['queue_root']).glob('*__attempt_*'))
    _need(len(attempts)==1 and str(attempts[0])==inv['attempt_directory'], 'Predecessor attempt inventory differs')
    _need(bs['request_reference']==refs['queue_request']==run['request_reference']==inv['request_reference'],
          'Predecessor queue binding differs')
    _need(inv['item_request_reference']==refs['worker_request'] and inv['parent_job']['manifest_reference']==refs['job_manifest']
          and inv['parent_job']['job_id']==job['job_id']==jm['job_id'],'Predecessor job binding differs')
    _need('--request' in jm['argv'] and jm['argv'][jm['argv'].index('--request')+1]==refs['queue_request']['path'], 'Parent job queue command differs')
    _need(job.get('complete') is True and job.get('state')=='FAILED' and job.get('returncode')==2 and
          run['status']=='FAILED_VALIDITY_OR_WORKER' and run['unsettled_plan_seconds'] is True,
          'Unsettled/nonterminal failure cannot be imported')
    _need(not observation(job)['blocks_new_launch'],'Predecessor processes remain alive')
    _need(wf['status']=='FAILED_WORKER_REQUIRES_DIAGNOSIS' and wf['error']=='Formal feature semantic identity differs'
          and failure['error']=='Worker exited nonzero: 1','Unreviewed failure reason')
    amount=job['elapsed_seconds']
    _need(type(amount) in (float,int) and math.isfinite(amount) and 0<amount<7200,'Invalid settled duration')
    _need(record['accounted_seconds']==amount and amount>=run['elapsed_seconds']>=failure['elapsed_seconds']>=wf['cumulative_accounted_seconds'],
          'Failure cost reset/inconsistent nested clocks')
    _need(run['cumulative_plan_seconds']=={old['plan_id']:failure['elapsed_seconds']},'Failure plan cost differs')
    cp=data['engine_checkpoint'];s=cp['state'];driver=data['driver_state']
    _need(cp['sha256']==canonical_hash(s) and s['problem_sha256']==canonical_hash(s['identity']) and
          s['engine_sha256']==old_engine,'Original checkpoint identity changed')
    _need(s['trace']==[] and all(s[k] is None for k in ('pending','best','lower_bound_usd','upper_bound_usd')) and
          s['next_iteration']==1 and s['status']=='RUNNING' and s['phase1']['status']=='FEASIBLE_BASIC',
          'Only phase-I-only pre-RMP failure can use this settlement')
    _need(driver['status']=='FAILED_EXCEPTION' and driver['calls']==[],'Unexpected driver progress')
    _need(data['driver_request']['problem_identity']==s['identity']==data['raw_request']['problem_identity'],
          'Original request/checkpoint binding differs')
    root=Path(old['output_root']).resolve();inventory=data['plan_inventory'];seen=set()
    _need(isinstance(inventory,list) and inventory,'Complete raw/driver inventory required')
    for member in inventory:
        path=_verify(member);_need(path.is_relative_to(root) and str(path) not in seen,'Raw inventory duplicate/path')
        seen.add(str(path))
    _need(seen=={str(p.resolve()) for p in root.rglob('*') if p.is_file()},'Old raw/driver inventory changed')
    history=json.loads((root/'raw/history.json').read_text())
    _need(len(history)==1,'Unexpected raw solver events')
    event=_read(history[0]);_need(event['event']=='phase1' and event['iteration']==0,'Old RMP/q event cannot be imported')
    phase_refs=[a for a in event['artifacts'] if Path(a['path']).name=='solver.json']
    _need(len(phase_refs)==1 and _read(phase_refs[0])==s['phase1'], 'Phase-I raw evidence differs from checkpoint')
    for path in (root/'raw').rglob('record.json'):
        item=json.loads(path.read_text())
        for artifact in item['artifacts']:_verify(artifact)
    review=data['independent_review']
    _need(review['status']=='PASS_FAILURE_CARRYOVER_REVIEW' and review['job_result']==refs['job_result'] and
          review['old_release']==record['old_release_reference'] and review['new_release']==record['new_release_reference']
          and review['plan_id']==record['plan_id'],'Independent carryover review differs')
    if require_claim:
        claim=json.loads(_claim_path(record).read_text())
        _need(claim==dict(carryover_reference=ref,successor_queue_root=record['successor_queue_root'],
                          successor_output_root=record['successor_output_root']), 'Failure predecessor already claimed by another successor')
    return dict(reference=ref,accounted_seconds=amount,plan_id=record['plan_id'],engine_sha256=old_engine,
                old_problem_identity=s['identity'],successor_queue_root=record['successor_queue_root'])


def claim_failure_carryover(ref,worker_request,queue_root):
    """After independent review, atomically reserve this predecessor once.

Only creates a new campaign claim; never overwrites or unlocks old execution.
"""
    result=verify_failure_carryover(ref,worker_request,queue_root,require_claim=False)
    record=_read(ref);path=_claim_path(record);path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(carryover_reference=ref,successor_queue_root=record['successor_queue_root'],successor_output_root=record['successor_output_root'])
    with path.open('x') as handle:
        json.dump(value,handle,allow_nan=False,indent=2);handle.write('\n')
    return result
