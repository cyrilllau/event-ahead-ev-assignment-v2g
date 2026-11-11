"""Bounded serial work queue inside one jobctl job; no automatic relaunch.

A COMPLETE item means its declared work finished, never scientific spec PASS.
A worker reads EVCS_BATCH_ATTEMPT_REQUEST and writes its result_path. Results
bind EVCS_BATCH_INVOCATION_SHA256 and nonempty immutable artifact references.
TIME_LIMIT is resumable only with exit zero and a hashed checkpoint included in
those artifacts. Unexpected exits/timeouts halt for review, preserving partial
outputs. Workers inherit the parent job process group: jobctl retains cleanup
ownership. This module does not implement CCG, its evidence reserve, or audits.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from .runtime.jobctl import atomic_json, group_members, identity_alive, process_identity, utcnow

# Keep unresolved subprocess handles until the parent supervisor cleans up.
_UNSETTLED_PROCESSES = []


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def file_reference(path):
    path=Path(path).resolve(strict=True)
    if not path.is_file(): raise ValueError('Reference must name a file')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)


def verify_reference(ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256','size_bytes'}:
        raise ValueError('Reference requires path/sha256/size_bytes')
    if not Path(ref['path']).is_absolute() or file_reference(ref['path'])!=ref:
        raise ValueError('Frozen reference changed: '+str(ref.get('path')))
    return ref


def _positive(value,name,maximum=7200):
    if type(value) not in (float,int) or not math.isfinite(value) or not 0<value<=maximum:
        raise ValueError(name+' must be positive finite and <= '+str(maximum))


def _id(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',value)


def checked_request(path):
    ref=file_reference(path);request=read_json(path)
    if request.get('schema_version')!=1 or not all(_id(request.get(k)) for k in ('batch_id','spec_id','model_id')):
        raise ValueError('Invalid schema/batch/spec/model identity')
    if request.get('domain') not in ('development','formal'):raise ValueError('Explicit domain required')
    if not isinstance(request.get('queue_root'),str) or not Path(request['queue_root']).is_absolute():raise ValueError('Immutable absolute queue_root required')
    _positive(request.get('job_wall_seconds'),'job_wall_seconds')
    reserve=request.get('shutdown_reserve_seconds',10.)
    _positive(reserve,'shutdown_reserve_seconds')
    if reserve>=request['job_wall_seconds']:raise ValueError('Shutdown reserve consumes job budget')
    policy=request.get('parent_wall_policy')
    if policy not in (None,'verified_parent_remaining_v1'):
        raise ValueError('Unknown parent wall policy')
    if policy is not None and reserve>10:raise ValueError('Verified queue seal reserve cannot exceed original 10 seconds')
    sources=request.get('source_references')
    if not isinstance(sources,list) or not sources:raise ValueError('Frozen source references required')
    for source in sources:verify_reference(source)
    if file_reference(__file__) not in sources:raise ValueError('Queue implementation must be source-bound')
    verify_reference(request['budget_reference'])
    items=request.get('items')
    if not isinstance(items,list) or not items:raise ValueError('Nonempty fixed item list required')
    seen=set();plan_limits={}
    for item in items:
        if not all(_id(item.get(k)) for k in ('item_id','plan_id')) or item['item_id'] in seen:
            raise ValueError('Invalid/duplicate item identity')
        seen.add(item['item_id']);_positive(item.get('plan_budget_seconds'),'plan_budget_seconds')
        if plan_limits.setdefault(item['plan_id'],item['plan_budget_seconds'])!=item['plan_budget_seconds']:
            raise ValueError('A plan must share one cumulative budget across items')
        argv=item.get('argv')
        if (not isinstance(argv,list) or not argv or any(not isinstance(v,str) or '\0' in v for v in argv)
                or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file()):
            raise ValueError('argv requires explicit executable and argument list; no implicit shell')
        cwd=item.get('cwd')
        if not isinstance(cwd,str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():raise ValueError('Absolute cwd required')
        verify_reference(item['request_reference'])
        worker=read_json(item['request_reference']['path'])
        publication_policy=worker.get('canonical_export_policy')
        if publication_policy is not None:
            from .canonical_in_attempt import POLICY
            if publication_policy!=POLICY:raise ValueError('Unknown canonical export policy')
            publication_source=file_reference(Path(__file__).with_name('canonical_in_attempt.py'))
            if publication_source not in sources or publication_source not in worker.get('planning_source_references',[]):
                raise ValueError('Canonical publication implementation must be source-bound before launch')
            if worker.get('method_id') not in ('EA','OR','ARO','SAA') or (worker['method_id']!='SAA' and worker.get('numerical_audit') is None):
                raise ValueError('Canonical publication requires an audited core worker')
    request=dict(request,shutdown_reserve_seconds=reserve)
    if file_reference(path)!=ref:raise ValueError('Request changed while validating')
    return request,ref


@contextlib.contextmanager
def _lock(root):
    with (root/'.batch.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Batch already active; no second launch') from exc
        try:yield handle
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def _write_new(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return file_reference(path)


def _result(ref,invocation_ref):
    verify_reference(ref);verify_reference(invocation_ref)
    result=read_json(ref['path'])
    if result.get('invocation_sha256')!=invocation_ref['sha256']:raise ValueError('Worker result invocation binding differs')
    artifacts=result.get('artifacts')
    if not isinstance(artifacts,list) or not artifacts:raise ValueError('Worker must retain nonempty hashed artifacts')
    seen=set()
    for artifact in artifacts:
        verify_reference(artifact)
        if artifact['path'] in seen:raise ValueError('Duplicate artifact references')
        seen.add(artifact['path'])
    status=result.get('status')
    if status not in ('COMPLETE','TIME_LIMIT'):raise ValueError('Worker validity failure or unknown status: '+str(status))
    if status=='TIME_LIMIT':
        checkpoint=result.get('checkpoint')
        if result.get('resumable') is not True or checkpoint not in artifacts:
            raise ValueError('TIME_LIMIT requires an explicit immutable resumable checkpoint artifact')
        verify_reference(checkpoint)
    return result


def _finalization_charge(receipt_ref, result, prior, finalization_reference=None):
    if result.get('numerical_audit_reference') is None or result['status']!='COMPLETE':
        return None
    path=Path(receipt_ref['path']).parent/'finalization.json'
    if finalization_reference is None or finalization_reference['path']!=str(path):
        raise ValueError('Final record seal lacks immutable history reference')
    verify_reference(finalization_reference)
    value=read_json(path)
    if value['status']!='PASS_FINAL_RECORD_SEAL' or value['receipt_reference']!=receipt_ref:
        raise ValueError('Missing or foreign finalized plan seal')
    for ref in value['references'].values():verify_reference(ref)
    solver=read_json(value['references']['solver_record']['path'])
    budget=read_json(value['references']['budget']['path'])
    actual=value['actual_attempt_seconds_before_confirmation_write'];charged=value['charged_attempt_seconds']
    if not all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in (actual,charged)):
        raise ValueError('Invalid finalization clock')
    if (actual>charged or charged<read_json(receipt_ref['path'])['elapsed_seconds'] or
        solver['receipt_reference']!=receipt_ref or solver['plan_id']!=read_json(receipt_ref['path'])['plan_id'] or
        budget['plan_id']!=solver['plan_id'] or
        solver['actual_attempt_seconds_at_cutoff']+solver['final_seal_allocated_seconds']!=charged or
        prior+charged>budget['planning_budget']['cumulative_wall_seconds'] or
        solver['charged_attempt_seconds']!=charged or
        solver['prior_cumulative_wall_seconds']!=prior or
        solver['cumulative_wall_seconds']!=prior+charged or
        budget['cumulative_wall_seconds']!=prior+charged or
        not solver['final_budget_acceptance'] or
        solver['numerical_audit_reference']!=result['numerical_audit_reference']):
        raise ValueError('Finalized plan accounting differs')
    return charged


def _finalize_attempt(receipt_ref,result,prior,item,attempt,attempt_started,deadline):
    from .formal_record_settlement import settle_plan_record,finalize_settled_record
    request=read_json(item['request_reference']['path']);budget=request['budget']
    answer=settle_plan_record(receipt_reference=receipt_ref,numerical_audit_reference=result['numerical_audit_reference'],
        prior_cumulative_seconds=prior,planning_budget=budget)
    prepared=None
    policy=request.get('canonical_export_policy')
    if policy is not None:
        from .canonical_in_attempt import POLICY,prepare_publication
        if policy!=POLICY:raise ValueError('Unknown canonical export policy')
        prepared=prepare_publication(receipt_ref,answer,item['request_reference'],attempt/'canonical_record')
    cutoff=time.monotonic()-attempt_started;reserve=budget['shutdown_reserve_seconds']
    if time.monotonic()+reserve>deadline:raise RuntimeError('Insufficient parent time for final record seal')
    final=finalize_settled_record(answer,actual_attempt_seconds_at_cutoff=cutoff,
        final_seal_reserve_seconds=reserve,output_directory=attempt/'final_record')
    actual=time.monotonic()-attempt_started
    charged=final['solver_record']['charged_attempt_seconds']
    if actual>charged:raise RuntimeError('Final record seal exceeded declared allocation')
    final_ref=_write_new(attempt/'finalization.json',dict(status='PASS_FINAL_RECORD_SEAL',receipt_reference=receipt_ref,
        references=final['references'],actual_attempt_seconds_before_confirmation_write=actual,
        charged_attempt_seconds=charged,seal_allocation_seconds=reserve,
        allocation_is_measured_solver_time=False,formal_OOS=False,scientific_spec_pass=False))
    if time.monotonic()-attempt_started>charged:raise RuntimeError('Seal confirmation exceeded allocated budget')
    publication_ref=None
    if prepared is not None:
        from .canonical_in_attempt import complete_publication
        publication_ref=complete_publication(prepared,final_ref,attempt_started=attempt_started,
            charged_attempt_seconds=charged,deadline=deadline)
    return (charged,final_ref) if prepared is None else (charged,final_ref,publication_ref)


def _carryovers(request):
    """Validate a single reviewed predecessor; never infer a prior balance."""
    from .failure_carryover import verify_failure_carryover
    result={}
    for item in request['items']:
        worker=read_json(item['request_reference']['path'])
        ref=item.get('failure_carryover_reference')
        if ref!=worker.get('failure_carryover_reference'):
            raise ValueError('Queue/worker carryover binding differs')
        if ref is None:continue
        if len(request['items'])!=1 or item['plan_id'] in result:
            raise ValueError('Failure carryover supports one unique plan/item only')
        verify_reference(ref)
        carryover_kind=read_json(ref['path']).get('kind')
        if carryover_kind=='SAA_L_SHAPED_FAILED_PARENTS_CARRYOVER_V1':
            from .saa_l_shaped_failure_debt import verify_failure_debt
            if file_reference(Path(__file__).with_name('saa_l_shaped_failure_debt.py')) not in request['source_references']:
                raise ValueError('Decomposition debt validator must be source-bound')
            verified=verify_failure_debt(ref,worker,request['queue_root'])
        elif carryover_kind=='SAA_PRE_ASSEMBLY_FAILED_PARENT_CARRYOVER_V1':
            from .saa_failure_carryover import verify_saa_failure_carryover
            if worker.get('method_id')!='SAA':raise ValueError('SAA carryover requires SAA worker')
            if file_reference(Path(__file__).with_name('saa_failure_carryover.py')) not in request['source_references']:
                raise ValueError('SAA carryover validator must be source-bound')
            verified=verify_saa_failure_carryover(ref,worker,request['queue_root'])
        else:
            if file_reference(Path(__file__).with_name('failure_carryover.py')) not in request['source_references']:
                raise ValueError('Carryover validator must be source-bound')
            verified=verify_failure_carryover(ref,worker,request['queue_root'])
        if verified['plan_id']!=item['plan_id']:
            raise ValueError('Carryover plan/item differs')
        result[item['plan_id']]=verified
    return result


def _history(state,request_ref,root=None,opening=None):
    verify_reference(state['request_reference'])
    if state['request_reference']!=request_ref:raise ValueError('Queue request identity changed; existing evidence retained')
    if root is not None:
        existing={str(p.resolve()) for p in root.glob('*__attempt_*/receipt.json')}
        if existing!={r['path'] for r in state['attempt_receipts']}:raise ValueError('Receipt inventory changed; cumulative accounting cannot reset')
    opening=opening or {}
    latest={};spent={k:v['accounted_seconds'] for k,v in opening.items()}
    final_refs=state.get('finalization_receipts',[])
    if len({r['path'] for r in final_refs})!=len(final_refs):raise ValueError('Duplicate finalization reference')
    if root is not None and {r['path'] for r in final_refs}!={str(p) for p in root.glob('*__attempt_*/finalization.json')}:
        raise ValueError('Finalization inventory differs from immutable history')
    finals={r['path']:r for r in final_refs}
    publication_refs=state.get('canonical_publication_receipts',[])
    if len({r['path'] for r in publication_refs})!=len(publication_refs):raise ValueError('Duplicate canonical publication reference')
    if root is not None and {r['path'] for r in publication_refs}!={str(p) for p in root.glob('*__attempt_*/canonical_record/completion.json')}:
        raise ValueError('Canonical publication inventory differs from immutable history')
    publications={r['path']:r for r in publication_refs}
    for ref in state['attempt_receipts']:
        verify_reference(ref);receipt=read_json(ref['path'])
        if receipt['request_reference']!=request_ref:raise ValueError('Attempt request binding differs')
        elapsed=receipt['elapsed_seconds']
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid cumulative time receipt')
        if receipt.get('closing_accounting_policy') is not None:
            frozen=read_json(request_ref['path'])
            observed=receipt['elapsed_seconds'];reserve=receipt.get('closing_reserve_seconds')
            charged=receipt.get('charged_elapsed_seconds')
            if (receipt['closing_accounting_policy']!='bounded_queue_seal_v1' or
                    frozen.get('parent_wall_policy')!='verified_parent_remaining_v1' or
                    receipt['status']!='TIME_LIMIT' or type(observed) not in (int,float) or
                    not math.isfinite(observed) or observed<0 or
                    reserve!=frozen.get('shutdown_reserve_seconds',10.) or reserve>10 or
                    charged!=observed+reserve):
                raise ValueError('Queue closing accounting differs')
            elapsed=charged
        result=_result(receipt['result_reference'],receipt['invocation_reference'])
        invocation=read_json(receipt['invocation_reference']['path'])
        if (invocation['item_id'],invocation['plan_id'])!=(receipt['item_id'],receipt['plan_id']):raise ValueError('Attempt identity differs')
        carry=opening.get(receipt['plan_id'])
        expected_ref=None if carry is None else carry['reference']
        if (invocation.get('failure_carryover_reference')!=expected_ref or
                invocation.get('opening_carryover_seconds',0.)!=(0. if carry is None else carry['accounted_seconds']) or
                invocation['cumulative_plan_seconds']!=spent.get(receipt['plan_id'],0.)):
            raise ValueError('History carryover binding differs')
        previous=latest.get(receipt['item_id'])
        if previous and previous['result']['status']=='COMPLETE':raise ValueError('Completed item was rerun')
        latest[receipt['item_id']]=dict(receipt=receipt,result=result)
        charge=_finalization_charge(ref,result,spent.get(receipt['plan_id'],0.),
            finals.get(str(Path(ref['path']).parent/'finalization.json')))
        worker=read_json(invocation['item_request_reference']['path'])
        policy=worker.get('canonical_export_policy')
        publication=publications.get(str(Path(ref['path']).parent/'canonical_record/completion.json'))
        if policy is not None:
            from .canonical_in_attempt import POLICY,verify_publication
            if policy!=POLICY:raise ValueError('Unknown canonical export policy in history')
            if result['status']=='COMPLETE':
                if charge is None or publication is None:raise ValueError('Complete canonical attempt lacks counted publication')
                if (Path(ref['path']).parent/'failure.json').exists():raise ValueError('Canonical attempt has unresolved failure evidence')
                verify_publication(publication,receipt_reference=ref,
                    finalization_reference=finals[str(Path(ref['path']).parent/'finalization.json')],
                    worker_request_reference=invocation['item_request_reference'])
            elif publication is not None:raise ValueError('Incomplete attempt cannot have canonical publication')
        elif publication is not None:raise ValueError('Unrequested canonical publication')
        spent[receipt['plan_id']]=spent.get(receipt['plan_id'],0.)+(elapsed if charge is None else charge)
    return latest,spent


def _supervisor_remaining(manifest,status):
    """Conservative Linux parent clock: process birth precedes jobctl timer.

    Start ticks are rounded down by the kernel; CLOCK_BOOTTIME also charges
    suspended time. Neither can extend jobctl's monotonic timeout. A reused
    PID/boot or non-Linux identity is rejected, never replaced with wall time.
    """
    _positive(manifest.get('timeout_seconds'),'parent timeout_seconds')
    if status.get('state') not in ('LAUNCHING','RUNNING'):
        raise ValueError('Parent is not admitting active work')
    identity=status.get('supervisor')
    if not identity or not identity_alive(identity):raise ValueError('Parent supervisor is not alive')
    current=process_identity(identity['pid'])
    if (current is None or current['token']!=identity['token'] or
            current['sid']!=status.get('supervisor_sid') or current['pid']!=current['sid']):
        raise ValueError('Parent supervisor identity differs')
    token=current['token'].split(':')
    if len(token)!=3 or token[0]!='linux' or not hasattr(time,'CLOCK_BOOTTIME'):
        raise ValueError('Verified parent clock requires Linux boot/start ticks')
    ticks=int(token[2]);frequency=os.sysconf('SC_CLK_TCK')
    if ticks<0 or frequency<=0:raise ValueError('Invalid supervisor start clock')
    now=time.monotonic();boot=time.clock_gettime(time.CLOCK_BOOTTIME)
    elapsed=boot-ticks/frequency
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid parent elapsed clock')
    remaining=manifest['timeout_seconds']-elapsed
    return dict(clock_kind='linux_supervisor_birth_boottime_v1',supervisor=current,
                timeout_seconds=manifest['timeout_seconds'],elapsed_seconds=elapsed,
                observed_monotonic=now,remaining_seconds=remaining,
                conservative_deadline_monotonic=now+remaining)


def _parent(request):
    directory=os.environ.get('EVCS_JOB_DIRECTORY')
    if request['domain']=='formal' and not directory:raise ValueError('Formal queue must run inside one jobctl parent job')
    if not directory:
        if request.get('parent_wall_policy') is not None:raise ValueError('Verified parent policy requires jobctl parent')
        return None
    manifest=read_json(Path(directory)/'manifest.json');status=read_json(Path(directory)/'status.json')
    if (manifest['job_id']!=os.environ.get('EVCS_JOB_ID') or manifest.get('spec_id')!=request['spec_id']
            or (request.get('parent_wall_policy') is None and manifest['timeout_seconds']<request['job_wall_seconds'])):
        raise ValueError('Parent job/spec/budget differs')
    # Popen may reach this code before jobctl persists RUNNING; session identity
    # is already frozen before it launches its command.
    if os.getsid(0)!=status.get('supervisor_sid'):raise ValueError('Queue is outside parent supervisor session')
    result=dict(job_id=manifest['job_id'],directory=str(Path(directory).resolve()),manifest_reference=file_reference(Path(directory)/'manifest.json'))
    if request.get('parent_wall_policy')=='verified_parent_remaining_v1':
        clock=_supervisor_remaining(manifest,status)
        if clock['timeout_seconds']>request['job_wall_seconds']:
            raise ValueError('Parent timeout exceeds frozen queue maximum')
        result['wall_clock']=clock
    return result


def run_batch(request_file,outputroot):
    """Run each pending item at most once in this invocation; resume explicitly.

No solver budget is reset: plan_remaining_seconds subtracts immutable prior
attempt wall receipts, including failed-to-finish cooperative attempts. Shared
job time includes queue verification and bookkeeping. Cache verification and
other overhead are reported separately; cached results are never new solves.
"""
    started=time.monotonic();root=Path(outputroot).resolve();root.mkdir(parents=True,exist_ok=True)
    with _lock(root) as lock:
        request,request_ref=checked_request(request_file);parent=_parent(request)
        if Path(request['queue_root']).resolve()!=root:raise ValueError('Queue root differs from frozen request; budgets cannot reset')
        state_path=root/'state.json'
        if state_path.exists():state=read_json(state_path)
        else:
            if any(p.name!='.batch.lock' for p in root.iterdir()):raise ValueError('Nonempty queue without state requires review')
            state=dict(schema_version=1,request_reference=request_ref,attempt_receipts=[],active=None,halted=None,runs=[])
            atomic_json(state_path,state)
        if state.get('active') or state.get('halted'):
            active=state.get('active') or {}
            raise RuntimeError('Prior active/unreviewed execution blocks launch; alive='+str(identity_alive(active.get('process'))))
        carryovers=_carryovers(request)
        latest,spent=_history(state,request_ref,root,carryovers)
        expected={item['item_id'] for item in request['items']}
        if not set(latest)<=expected:raise ValueError('History contains undeclared items')
        run_id='run_'+uuid.uuid4().hex;reused=[];executed=[];run_attempt_seconds=0.
        deadline=started+request['job_wall_seconds']
        if parent is not None and 'wall_clock' in parent:
            deadline=min(deadline,parent['wall_clock']['conservative_deadline_monotonic'])
        deadline-=request['shutdown_reserve_seconds']
        run_dir=root/run_id;run_dir.mkdir()
        def finish(status,**extra):
            total=time.monotonic()-started
            result=dict(schema_version=1,status=status,batch_id=request['batch_id'],spec_id=request['spec_id'],
                model_id=request['model_id'],domain=request['domain'],request_reference=request_ref,parent_job=parent,
                reused_item_ids=reused,new_attempt_item_ids=executed,completed_item_ids=[i for i,v in latest.items() if v['result']['status']=='COMPLETE'],
                pending_item_ids=[i['item_id'] for i in request['items'] if i['item_id'] not in latest or latest[i['item_id']]['result']['status']!='COMPLETE'],
                cumulative_plan_seconds=spent,elapsed_seconds=total,attempt_wall_seconds=run_attempt_seconds,
                failure_carryover_references={k:v['reference'] for k,v in carryovers.items()},
                opening_carryover_seconds={k:v['accounted_seconds'] for k,v in carryovers.items()},
                verification_and_queue_overhead_seconds=max(0.,total-run_attempt_seconds),automatic_relaunch=False,
                scientific_acceptance=False,finished_at_utc=utcnow(),**extra)
            ref=_write_new(run_dir/'result.json',result);state['runs'].append(ref);atomic_json(state_path,state)
            return result
        accounting_cursor=started
        for item in request['items']:
            item_id=item['item_id'];plan_id=item['plan_id']
            previous=latest.get(item_id)
            if previous and previous['result']['status']=='COMPLETE':reused.append(item_id);continue
            verified_wall=request.get('parent_wall_policy')=='verified_parent_remaining_v1'
            attempt_started=(accounting_cursor if verified_wall else
                             started if carryovers and not executed else time.monotonic())
            # Verify every old output before any new process can overwrite it.
            checked_request(request_file);latest,spent=_history(state,request_ref,root,carryovers)
            plan_remaining=item['plan_budget_seconds']-spent.get(plan_id,0.)
            remaining=min(plan_remaining-(time.monotonic()-attempt_started)-request['shutdown_reserve_seconds'],deadline-time.monotonic())
            if remaining<=0:
                if verified_wall:
                    # No worker receipt exists: retain the real preprocessing
                    # charge and refuse another run until explicitly settled.
                    cost=time.monotonic()-attempt_started
                    state['halted']='PRE_WORKER_COST_REQUIRES_SETTLEMENT';atomic_json(state_path,state)
                    spent[plan_id]=spent.get(plan_id,0.)+cost
                    return finish('BUDGET_EXHAUSTED',next_item_id=item_id,unsettled_plan_seconds=True,
                        pre_worker_cost=dict(plan_id=plan_id,elapsed_seconds=cost,
                            accounting_start_monotonic=attempt_started,
                            reason='Original parent/plan allowance cannot admit a worker with shutdown reserve'))
                return finish('BUDGET_EXHAUSTED',next_item_id=item_id)
            worker_deadline=time.monotonic()+remaining
            attempt=root/(item_id+'__attempt_'+str(sum(read_json(r['path'])['item_id']==item_id for r in state['attempt_receipts'])+1).zfill(4))
            attempt.mkdir()
            invocation=dict(schema_version=1,batch_id=request['batch_id'],spec_id=request['spec_id'],model_id=request['model_id'],domain=request['domain'],
                item_id=item_id,plan_id=plan_id,item_request_reference=item['request_reference'],request_reference=request_ref,
                remaining_seconds=remaining,plan_remaining_seconds=plan_remaining,cumulative_plan_seconds=spent.get(plan_id,0.),
                failure_carryover_reference=None if plan_id not in carryovers else carryovers[plan_id]['reference'],
                opening_carryover_seconds=0. if plan_id not in carryovers else carryovers[plan_id]['accounted_seconds'],
                attempt_started_monotonic=attempt_started,worker_deadline_monotonic=worker_deadline,
                plan_budget_seconds=item['plan_budget_seconds'],previous_checkpoint=None if not previous else previous['result'].get('checkpoint'),
                previous_result_reference=None if not previous else previous['receipt']['result_reference'],
                attempt_directory=str(attempt),result_path=str(attempt/'worker_result.json'),parent_job=parent,
                argv=item['argv'],cwd=item['cwd'],created_at_utc=utcnow())
            invocation_ref=_write_new(attempt/'invocation.json',invocation)
            state['active']=dict(item_id=item_id,plan_id=plan_id,invocation_reference=invocation_ref,process=None,started_at_utc=utcnow())
            atomic_json(state_path,state)
            env=os.environ.copy();env.update(EVCS_BATCH_ATTEMPT_REQUEST=invocation_ref['path'],EVCS_BATCH_INVOCATION_SHA256=invocation_ref['sha256'])
            executed.append(item_id);process=None
            baseline_group={(p['pid'],p['token']) for p in group_members(os.getpgrp(),os.getsid(0))}
            try:
                with (attempt/'stdout.log').open('xb') as stdout,(attempt/'stderr.log').open('xb') as stderr:
                    # No shell and no new session/group: parent jobctl owns tree cleanup.
                    # Passing the lock also protects the launch-to-identity crash window.
                    process=subprocess.Popen(item['argv'],cwd=item['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                             close_fds=True,pass_fds=(lock.fileno(),))
                    state['active']['process']=process_identity(process.pid);atomic_json(state_path,state)
                    timeout=max(0.,min(deadline,worker_deadline)-time.monotonic())
                    try:code=process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        _UNSETTLED_PROCESSES.append(process)
                        elapsed=time.monotonic()-attempt_started;run_attempt_seconds+=elapsed;spent[plan_id]=spent.get(plan_id,0.)+elapsed
                        _write_new(attempt/'failure.json',dict(status='HARD_TIMEOUT',elapsed_seconds_lower_bound=elapsed,partial_outputs_preserved=True,process=state['active']['process']))
                        state['halted']='HARD_TIMEOUT_REQUIRES_REVIEW';atomic_json(state_path,state)
                        return finish('FAILED_HARD_TIMEOUT',active_process=state['active']['process'],unsettled_plan_seconds=True,
                                      reason='Worker remains owned by parent jobctl; no valid resumable result was returned')
                remaining_children=[p for p in group_members(os.getpgrp(),os.getsid(0)) if (p['pid'],p['token']) not in baseline_group]
                if remaining_children:raise RuntimeError('Worker left live group descendants; no next item: '+str(remaining_children))
                if code!=0:raise RuntimeError('Worker exited nonzero: '+str(code))
                checked_request(request_file);_history(state,request_ref,root,carryovers)
                result_ref=file_reference(attempt/'worker_result.json');result=_result(result_ref,invocation_ref)
                worker_request=read_json(item['request_reference']['path'])
                if (worker_request.get('canonical_export_policy') is not None and result['status']=='COMPLETE'
                        and result.get('numerical_audit_reference') is None):
                    raise ValueError('Complete canonical attempt requires original numerical audit')
                observed_elapsed=time.monotonic()-attempt_started
                closing_reserve=(request['shutdown_reserve_seconds'] if verified_wall and result['status']=='TIME_LIMIT' else 0.)
                elapsed=observed_elapsed+closing_reserve
                if spent.get(plan_id,0.)+elapsed>item['plan_budget_seconds']:
                    raise RuntimeError('Cumulative plan wall budget exceeded during final evidence verification')
                receipt=dict(item_id=item_id,plan_id=plan_id,request_reference=request_ref,invocation_reference=invocation_ref,
                    result_reference=result_ref,elapsed_seconds=observed_elapsed,status=result['status'],returncode=code,finished_at_utc=utcnow())
                if closing_reserve:
                    receipt.update(charged_elapsed_seconds=elapsed,closing_reserve_seconds=closing_reserve,
                        closing_accounting_policy='bounded_queue_seal_v1')
                receipt_ref=_write_new(attempt/'receipt.json',receipt)
                charged_elapsed=elapsed
                publication_ref=None
                if result['status']=='COMPLETE' and result.get('numerical_audit_reference') is not None:
                    finalized=_finalize_attempt(receipt_ref,result,spent.get(plan_id,0.),item,attempt,attempt_started,deadline)
                    if worker_request.get('canonical_export_policy') is None:
                        charged_elapsed,final_ref=finalized
                    else:
                        charged_elapsed,final_ref,publication_ref=finalized
                    state.setdefault('finalization_receipts',[]).append(final_ref)
                    if publication_ref is not None:state.setdefault('canonical_publication_receipts',[]).append(publication_ref)
                accounting_cursor=min(time.monotonic(),attempt_started+charged_elapsed)
                state['attempt_receipts'].append(receipt_ref);state['active']=None
                atomic_json(state_path,state);run_attempt_seconds+=observed_elapsed
                if publication_ref is not None and (time.monotonic()-attempt_started>charged_elapsed or time.monotonic()>deadline):
                    raise RuntimeError('Canonical publication state commit exceeded original seal allocation')
                spent[plan_id]=spent.get(plan_id,0.)+charged_elapsed;latest[item_id]=dict(receipt=receipt,result=result)
                if result['status']=='TIME_LIMIT':
                    terminal=finish('TIME_LIMIT',next_item_id=item_id,resumable=True,checkpoint=result['checkpoint'])
                    if closing_reserve and time.monotonic()>attempt_started+elapsed:
                        state['halted']='QUEUE_SEAL_EXCEEDED_CHARGED_RESERVE';atomic_json(state_path,state)
                        return finish('FAILED_QUEUE_SEAL',unsettled_plan_seconds=True,
                            closing_overrun_seconds=time.monotonic()-(attempt_started+elapsed),
                            reason='Queue seal exceeded charged closing reserve; no automatic resume')
                    return terminal
            except Exception as exc:
                state['halted']='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW';atomic_json(state_path,state)
                elapsed=time.monotonic()-attempt_started;run_attempt_seconds+=elapsed;spent[plan_id]=spent.get(plan_id,0.)+elapsed
                _write_new(attempt/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),elapsed_seconds=elapsed,
                    partial_outputs_preserved=True,process=None if process is None else
                    (state.get('active') or {}).get('process')))
                return finish('FAILED_VALIDITY_OR_WORKER',reason=str(exc),unsettled_plan_seconds=True)
        return finish('COMPLETE_BATCH')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--outputroot',type=Path,required=True)
    args=parser.parse_args()
    result=run_batch(args.request,args.outputroot);print(json.dumps(result,allow_nan=False))
    return 0 if result['status'] in ('COMPLETE_BATCH','TIME_LIMIT','BUDGET_EXHAUSTED') else 2

if __name__=='__main__':raise SystemExit(main())
