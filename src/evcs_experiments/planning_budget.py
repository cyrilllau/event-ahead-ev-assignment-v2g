"""One-plan cumulative C&CG budget with a separately reserved final evaluation.

This adapter schedules the unchanged C&CG engine one completed iteration at a
time. It never signs scientific acceptance. The caller must persist raw solver
evidence, validate physical inputs, and include all preparation time in the
supplied cumulative elapsed time. A parent job enforces the hard wall limit:
matrix assembly and evidence writes are not preemptible inside the engine.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

from .ccg import run_ccg, _engine_hash
from .evaluation_lock import file_reference
from .random_streams import canonical_hash
from .runtime.jobctl import atomic_json


def _read(path):
    return json.loads(Path(path).read_text())


def _verify(ref):
    if file_reference(ref['path']) != ref:
        raise ValueError('Planning budget evidence changed: '+ref['path'])


def _write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    return file_reference(path)


def validate_budget(budget):
    required={'cumulative_wall_seconds','single_master_seconds',
              'final_certificate_reserve_seconds','shutdown_reserve_seconds'}
    if not required <= set(budget):
        raise ValueError('Complete cumulative/master/certificate/shutdown budget required')
    for key in required:
        value=budget[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=7200:
            raise ValueError('Invalid planning budget: '+key)
    if (budget['single_master_seconds']+budget['final_certificate_reserve_seconds']+
            budget['shutdown_reserve_seconds'] >= budget['cumulative_wall_seconds']):
        raise ValueError('Planning budget leaves no ordinary work interval')
    return dict(budget)


def _progress(state):
    if state is None:
        return None
    pending=state['pending']
    return canonical_hash(dict(phase1=state['phase1'] is not None,
        active_indices=state['active_indices'], next_mip_gap=state['next_mip_gap'],
        pending=None if pending is None else dict(plan=pending['plan'],q=pending['q_usd']),
        best=None if state['best'] is None else dict(plan=state['best']['plan'],q=state['best']['q_usd']),
        lower=state['lower_bound_usd'],upper=state['upper_bound_usd']))


def run_budgeted_ccg(problem, *, outputroot, budget, input_binding,
                     engine_options, evidence_sink_factory,
                     spent_before_seconds=0., available_wall_seconds=None,
                     resume=False, previous_driver_snapshot=None, postprocessing_reserve_seconds=0.):
    """Execute bounded iterations and at most one final pending-only drain.

``spent_before_seconds`` is cumulative wall use up to entry, including previous
worker attempts, input loading and assembly. ``available_wall_seconds`` is the
parent invocation's remaining wall time AFTER subtracting that preparation;
it cannot grant a new cumulative budget. Every engine call, raw status and
immutable checkpoint snapshot is retained. Failed calls require diagnosis.
    The returned best plan still requires the independent full-support audit.
    Use only inside one campaign worker holding the queue lock. Input binding
    records identity; the caller verifies referenced input bytes on every use.
"""
    started=time.monotonic();budget=validate_budget(budget)
    if (type(postprocessing_reserve_seconds) not in (int,float) or not math.isfinite(postprocessing_reserve_seconds)
            or not 0<=postprocessing_reserve_seconds<budget["final_certificate_reserve_seconds"]):
        raise ValueError("Postprocessing reserve must fit inside existing final certificate reserve")
    if type(spent_before_seconds) not in (int,float) or not math.isfinite(spent_before_seconds) or spent_before_seconds<0:
        raise ValueError('Invalid cumulative elapsed time')
    left_at_entry=budget['cumulative_wall_seconds']-spent_before_seconds
    if available_wall_seconds is not None:
        if type(available_wall_seconds) not in (int,float) or not math.isfinite(available_wall_seconds) or available_wall_seconds<0:
            raise ValueError('Invalid parent remaining wall time')
        left_at_entry=min(left_at_entry,available_wall_seconds)
    forbidden={'checkpoint_path','resume','wall_seconds','rmp_time_limit_seconds',
               'max_iterations','evidence_sink','solver_log_directory'}
    if forbidden & set(engine_options):
        raise ValueError('Engine options cannot override cumulative execution boundaries')
    if not callable(evidence_sink_factory):
        raise ValueError('Raw evidence sink factory required')
    root=Path(outputroot).resolve();driver_path=root/'driver_state.json';checkpoint=root/'checkpoint.json'
    request=dict(schema_version=1,problem_identity=problem.identity,budget=budget,
        input_binding=input_binding,engine_options=engine_options,engine_sha256=_engine_hash(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if postprocessing_reserve_seconds:
        request["postprocessing_reserve_seconds"]=postprocessing_reserve_seconds
    request_hash=canonical_hash(request)
    if resume:
        if previous_driver_snapshot is None:
            raise ValueError('Explicit immutable driver snapshot required for resume')
        _verify(previous_driver_snapshot)
        if file_reference(driver_path)['sha256']!=previous_driver_snapshot['sha256']:
            raise ValueError('Mutable driver state differs from accepted resume snapshot')
        state=_read(driver_path)
        if state['request_sha256']!=request_hash or canonical_hash(_read(root/'request.json'))!=request_hash:
            raise ValueError('Planning source/problem/config/budget identity changed')
        if spent_before_seconds+1e-9<state['cumulative_seconds']:
            raise ValueError('Cumulative preparation/attempt time cannot reset')
        for ref in state['calls']:
            _verify(ref);call=_read(ref['path'])
            for artifact in call['artifacts']:_verify(artifact)
        expected={ref['path'] for ref in state['calls']}
        if {str(p.resolve()) for p in root.glob('call_*/record.json')}!=expected:
            raise ValueError('Planning call history was truncated')
        if state['calls']:
            saved=_read(state['calls'][-1]['path'])['checkpoint_snapshot']
            _verify(saved)
            if file_reference(checkpoint)['sha256']!=saved['sha256']:
                raise ValueError('CCG checkpoint differs from prior immutable snapshot')
        if state['status'] not in ('TIME_LIMIT','BUDGET_EXHAUSTED','CONVERGED'):
            raise ValueError('Failed or incomplete driver requires diagnosis before resume')
    else:
        if previous_driver_snapshot is not None:
            raise ValueError('Snapshot supplied without resume')
        root.mkdir(parents=True,exist_ok=False)
        _write_new(root/'request.json',request)
        state=dict(schema_version=1,request_sha256=request_hash,calls=[],cumulative_seconds=spent_before_seconds,
                   status='RUNNING',no_progress_calls=0,independent_scientific_audit_required=True)

    def left():return left_at_entry-(time.monotonic()-started)
    def save(status, **extra):
        elapsed=time.monotonic()-started
        if status in ('CONVERGED','TIME_LIMIT','BUDGET_EXHAUSTED') and (
                elapsed>left_at_entry or spent_before_seconds+elapsed>budget['cumulative_wall_seconds']):
            status='BUDGET_OVERRUN'
            extra['stop_reason']='actual_wall_exceeded_parent_or_cumulative_budget'
        state.update(status=status,cumulative_seconds=spent_before_seconds+elapsed,**extra)
        atomic_json(driver_path,state)
        return json.loads(json.dumps(state,allow_nan=False))

    if resume and state['status']=='CONVERGED':
        return save('CONVERGED',verified_reuse_without_solver=True)

    engine_state=None if not checkpoint.exists() else _read(checkpoint)['state']
    reserve=budget['final_certificate_reserve_seconds'];shutdown=budget['shutdown_reserve_seconds']
    save('RUNNING')
    while True:
        pending=engine_state is not None and engine_state['pending'] is not None
        if engine_state is not None and engine_state['status']=='CONVERGED':
            return save('CONVERGED',raw_engine_status='CONVERGED',stop_reason='engine_certificate_convergence')
        # A pending evaluation gets one explicit drain. In particular this is
        # never emulated by a zero master limit or an exception in the RMP sink.
        final_only=pending
        if final_only:
            grant=left()-shutdown-postprocessing_reserve_seconds
        else:
            if left()<=reserve+shutdown+budget['single_master_seconds']:
                return save('BUDGET_EXHAUSTED',stop_reason='preserve_final_certificate_reserve',
                            raw_engine_status=None if engine_state is None else engine_state['status'])
            grant=left()-reserve-shutdown
        if grant<=0:
            return save('TIME_LIMIT',stop_reason='parent_or_cumulative_wall_exhausted',
                        raw_engine_status=None if engine_state is None else engine_state['status'])
        if state['no_progress_calls']>=2:
            return save('NEEDS_DIAGNOSIS',stop_reason='two_calls_without_new_evidence')
        directory=root/f"call_{len(state['calls'])+1:04d}";directory.mkdir()
        before=_progress(engine_state);before_pending=None if not pending else engine_state['pending']
        call_start=time.monotonic()
        invocation=dict(mode='pending_only_final_certificate' if final_only else 'ordinary',
            granted_seconds=grant,remaining_before_seconds=left(),spent_before_seconds=spent_before_seconds+call_start-started,
            max_completed_iterations=1,master_seconds=budget['single_master_seconds'],
            prior_pending_iteration=None if before_pending is None else before_pending['iteration'],
            prior_completed_q=0 if before_pending is None else len(before_pending['q_usd']))
        _write_new(directory/'invocation.json',invocation)
        try:
            sink=evidence_sink_factory(directory)
            if not callable(sink):
                raise ValueError('Evidence sink factory must return a callable raw evidence sink')
            result=run_ccg(problem,checkpoint_path=checkpoint,resume=checkpoint.exists(),wall_seconds=grant,
                rmp_time_limit_seconds=budget['single_master_seconds'],max_iterations=1,
                evidence_sink=sink,solver_log_directory=directory/'solver_logs',**engine_options)
        except Exception as exc:
            _write_new(directory/'failure.json',dict(type=type(exc).__name__,message=str(exc),
                elapsed_seconds=time.monotonic()-call_start,raw_checkpoint_preserved=checkpoint.exists()))
            save('FAILED_EXCEPTION',stop_reason='inspect_raw_failure_before_any_retry')
            raise
        engine_state=result.state
        # An explicit snapshot is immutable while checkpoint.json remains the
        # engine's own byte-for-byte continuation file.
        snapshot=directory/'checkpoint_snapshot.json'
        with snapshot.open('xb') as f:f.write(checkpoint.read_bytes())
        snapshot_ref=file_reference(snapshot)
        _write_new(directory/'engine_result.json',result.as_dict())
        artifacts=[file_reference(p) for p in sorted(directory.rglob('*')) if p.is_file()]
        record=dict(**invocation,status=result.status,elapsed_seconds=time.monotonic()-call_start,
                    checkpoint_snapshot=snapshot_ref,artifacts=artifacts,
                    full_spec_pass=False,independent_scientific_audit_required=True)
        state['calls'].append(_write_new(directory/'record.json',record))
        state['no_progress_calls']=state['no_progress_calls']+1 if before==_progress(engine_state) else 0
        save('RUNNING',raw_engine_status=result.status)
        if left()<0 or state['cumulative_seconds']>budget['cumulative_wall_seconds']:
            return save('BUDGET_OVERRUN',stop_reason='nonpreemptible_work_exceeded_parent_or_cumulative_budget')
        if result.status=='CONVERGED':
            return save('CONVERGED',stop_reason='engine_certificate_convergence')
        if result.status not in ('TIME_LIMIT','ITERATION_LIMIT'):
            return save('NEEDS_DIAGNOSIS',stop_reason='engine_'+result.status)
        if final_only and postprocessing_reserve_seconds and engine_state['pending'] is None:
            # The integrated worker continues after a completed drain when
            # another ordinary unit fits; otherwise the normal admission gate
            # returns the preserved budget boundary for immediate final audit.
            continue
        if final_only:
            # Ending this boundary does not relabel a native TIME_LIMIT as
            # convergence, even if it returned a valid full-support best plan.
            return save('TIME_LIMIT',stop_reason='one_pending_evaluation_boundary_completed_or_exhausted')
        if left()<=shutdown:
            return save('TIME_LIMIT',stop_reason='actual_wall_exhausted_including_evidence')
