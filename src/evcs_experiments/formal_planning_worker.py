"""Campaign queue worker for EA/ARO and accepted onset OR; numerical acceptance is separate.

The CLI accepts only a hash-bound formal campaign_batch invocation. There is no
CLI development/data-bypass mode. A private development dependency fixture is
used by local unit tests with genuine small physical problems.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .campaign_batch import file_reference, verify_reference, read_json, checked_request
from .ccg import _engine_hash
from .config import get_paths
from .evaluation_metrics import plan_identity
from .first_stage import FirstStagePlan
from .failure_carryover import verify_failure_carryover, verify_prepared_problem
from .formal_planning import load_formal_problem
from .planning_budget import run_budgeted_ccg, validate_budget
from .random_streams import canonical_hash


MODEL='IEEE33-v54-Q1'
SEAL_RESERVE_SECONDS=10.


def _need(value,message):
    if not bool(value):raise ValueError(message)


def _read(ref):verify_reference(ref);return read_json(ref['path'])


def _new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,allow_nan=False,indent=2);f.write('\n')
    return file_reference(path)


def _copy(source,target):
    with Path(target).open('xb') as f:f.write(Path(source).read_bytes())
    return file_reference(target)


def _closure(ref,seen=None):
    """Verify checkpoint artifacts using the explicit inventory-aware validator."""
    from .checkpoint_closure import verify_checkpoint_closure
    return verify_checkpoint_closure(ref,seen)


@dataclass(frozen=True)
class _DevelopmentFixture:
    prepare: object
    budget: dict
    engine_options: dict
    seal_reserve_seconds: float=.05


def _formal_prerequisites(request,invocation):
    _need(request['domain']==invocation['domain']=='formal','Formal namespace required')
    _need(invocation['spec_id']==request['spec_id']=='S05' and invocation['model_id']==request['model_id']==MODEL,
        'Formal model/spec differs')
    batch,batch_ref=checked_request(invocation['request_reference']['path'])
    _need(batch_ref==invocation['request_reference'] and batch['domain']=='formal' and
        batch['spec_id']==invocation['spec_id'] and batch['model_id']==invocation['model_id'] and
        batch['batch_id']==invocation['batch_id'],'Worker is not bound to its validated queue request')
    items=[item for item in batch['items'] if item['item_id']==invocation['item_id']]
    _need(len(items)==1 and items[0]['plan_id']==invocation['plan_id'] and
        items[0]['request_reference']==invocation['item_request_reference'] and
        items[0]['plan_budget_seconds']==invocation['plan_budget_seconds'] and
        items[0]['argv']==invocation['argv'] and items[0]['cwd']==invocation['cwd'],'Worker item differs from fixed queue')
    _need(items[0].get('failure_carryover_reference')==request.get('failure_carryover_reference')==invocation.get('failure_carryover_reference'), 'Queue/worker/invocation carryover differs')
    parent=invocation['parent_job'];_need(isinstance(parent,dict),'Formal worker needs jobctl parent')
    _need(os.environ.get('EVCS_JOB_ID')==parent['job_id'] and
        str(Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve())==parent['directory'],'Worker parent environment differs')
    manifest=_read(parent['manifest_reference']);status=read_json(Path(parent['directory'])/'status.json')
    _need(manifest['job_id']==parent['job_id'] and manifest['spec_id']=='S05' and os.getsid(0)==status['supervisor_sid'],
        'Worker is outside supervising job session')
    _need(invocation['remaining_seconds']<=min(invocation['plan_remaining_seconds'],manifest['timeout_seconds']),
        'Worker remaining budget exceeds queue or parent timeout')
    config=_read(request['config_reference']);protocol=_read(request['protocol_reference'])['payload']
    _need(protocol['domain']=='formal' and protocol['configuration']==config,'Frozen formal config/protocol differs')
    _need(request['budget']==config['planning_budget'],'Common planning budget cannot change')
    b=validate_budget(request['budget'])
    expected=dict(cumulative_wall_seconds=7200,single_master_seconds=600,final_certificate_reserve_seconds=1790,
        shutdown_reserve_seconds=10,master_threads=4,recourse_workers=1,blas_threads=1,normal_gap_relative=.005,rmp_mip_gap=.001)
    _need(b==expected,'Unsupported frozen planning resources/gaps')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        _need(os.environ.get(key)=='1','Formal BLAS/OpenMP environment must be single-threaded: '+key)
    refs=request['planning_source_references'];_need(isinstance(refs,list) and bool(refs),'Planning source closure required')
    for ref in refs:verify_reference(ref)
    files=sorted(Path(__file__).parent.rglob('*.py'))
    _need(refs==[file_reference(p) for p in files],'Current planning source closure differs')
    if request['method_id']=='OR':
        from .formal_onset import validate_onset_acceptance
        _need(request.get('onset_acceptance_reference')==request['bank_acceptance_reference'],
            'OR worker must bind its own onset input acceptance')
        validate_onset_acceptance(onset_acceptance_reference=request['onset_acceptance_reference'],
            summary_reference=request['summary_reference'],audit_reference=request['audit_reference'],
            protocol_reference=request['protocol_reference'])
    else:
        _need('onset_acceptance_reference' not in request,'Baseline worker cannot use onset acceptance')
        card=_read(request['bank_acceptance_reference'])
        _need(card['status']=='PASS_COMPONENT' and card['component_id']=='FORMAL_BASELINE_INPUTS'
            and card['spec_id']=='S05' and card['increment_id']=='S05-I01','Formal input component not accepted')
        for key in ('summary_reference','audit_reference','source_manifest_reference','protocol_reference',
                    'independent_review_reference','job_result_reference','artifact_manifest_reference'):verify_reference(card[key])
        _need(card['summary_reference']==request['summary_reference'] and card['audit_reference']==request['audit_reference']
            and card['protocol_reference']==request['protocol_reference'],'Input acceptance does not bind requested bank')
        summary=_read(request['summary_reference']);audit=_read(request['audit_reference'])
        _need(card['source_manifest_reference']==summary['source_manifest'] and
            summary['formal_protocol_lock']==request['protocol_reference'] and audit['source_summary_sha256']==request['summary_reference']['sha256']
            and audit['source_manifest_sha256']==card['source_manifest_reference']['sha256']
            and audit['formal_protocol_lock']==request['protocol_reference'],'Bank audit/source/protocol binding differs')
        _need(audit['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT' and card['scenario_count']==audit['scenario_count']==1920
            and card['target_count']==8 and card['formal_OOS'] is False and card['formal_plans_solved'] is False
            and card['full_S05_pass'] is False,'Input acceptance scope differs')
    _need(request['target_id'] in protocol['target_ids'],'Unknown formal target')
    return b,dict(master_backend='gurobi',optimizer_threads=4,oracle_time_limit_seconds=60,
        mip_relative_gap=.001,optimality_relative_gap=.005,zero_presolve=True,native_start=True)


def _candidate(engine,prepared,request,driver,snapshot_refs,elapsed):
    best=engine.get('best') if engine is not None else None
    record=dict(schema_version=1,status='PENDING_INDEPENDENT_NUMERICAL_AUDIT',full_support_verified=False,
        formal_OOS=False,scientific_acceptance=False,method_id=request['method_id'],target_id=request['target_id'],
        plan_id=request['plan_id'],driver_status=driver['status'],raw_engine_status=None if engine is None else engine['status'],
        cumulative_accounted_seconds=elapsed,cumulative_seconds_provisional=True,
        accounting_authority='campaign_batch immutable attempt receipts including queue verification; reconcile before scientific lock',
        snapshots=snapshot_refs,candidate=None)
    if best is None:return record
    plan=FirstStagePlan(**best['plan']);digest=plan_identity(plan)
    _need(digest==best['plan_sha256'],'Best-plan identity changed')
    matches=[row for row in engine['trace'] if row['iteration']==best['iteration'] and row['plan_sha256']==digest]
    _need(len(matches)==1,'Best plan lacks its own completed iteration')
    row=matches[0];ids=prepared.problem.probability_set.support_ids
    _need(len(best['q_usd'])==len(ids) and best['q_usd']==row['q_usd'] and best['probability']==row['probability_audit']
        and best['original_rmp_dual']==row['raw_rmp_dual'],'Best evidence was replaced by another iteration')
    lb,ub=engine['lower_bound_usd'],engine['upper_bound_usd']
    _need(all(type(v) in (int,float) and math.isfinite(v) for v in (lb,ub)),'Candidate needs finite LB/UB')
    evidence=[]
    for ref in prepared.evidence.events:
        event=_read(ref)
        if event['iteration']==best['iteration'] and event['event'] in ('rmp','recourse','probability','iteration'):
            evidence.append(ref)
    recourse=[];probability_matches=[];rmp_matches=[]
    for ref in evidence:
        event=_read(ref);artifacts={Path(a['path']).name:a for a in event['artifacts']}
        _need(event['status']=='RAW_EVIDENCE_RETAINED_PENDING_AUDIT','Candidate includes failed raw event')
        if event['event']=='recourse' and event.get('plan_sha256')==digest and event.get('optimal_certificate_valid') is True:
            i=event['scenario_index'];_need(event['q_usd']==best['q_usd'][i],'Best q differs from original raw LP')
            recourse.append(i)
        elif event['event']=='probability' and _read(artifacts['solver.json'])==best['probability']:
            q_input=_read(artifacts['q_input.json'])
            _need(q_input['q_usd']==best['q_usd'] and q_input['support_ids']==list(ids) and q_input['plan']==best['plan'],
                'Best probability input differs from original raw fixed-plan evidence')
            probability_matches.append(ref)
        elif event['event']=='rmp' and _read(artifacts['solver.json'])==row['rmp']:
            solution=_read(artifacts['solution.json'])
            _need(all(solution[k]==best['plan'][k] for k in ('x','r','task_sha256','config_hash')),
                'Best plan differs from own original RMP incumbent')
            rmp_matches.append(ref)
    _need(set(recourse)==set(range(len(ids))) and len(probability_matches)==1 and len(rmp_matches)==1,
        'Candidate lacks its original full-support raw evidence')
    record['candidate']=dict(plan=best['plan'],plan_sha256=digest,best_iteration=best['iteration'],
        support_ids=list(ids),q_usd=best['q_usd'],probability=best['probability'],original_rmp_dual=best['original_rmp_dual'],
        original_rmp_residual_usd=best['original_rmp_residual_usd'],original_rmp=row['rmp'],
        native_master_status=row['rmp']['status'],native_master_solver_status=row['rmp']['solver_status'],
        lower_bound_usd=lb,upper_bound_usd=ub,gap_relative=max(0.,ub-lb)/max(1.,abs(ub)),
        first_stage_cost_usd=best['Fpre_usd'],own_raw_evidence=evidence,
        input_binding=prepared.input_binding,engine_identity=prepared.problem.identity)
    return record


def run_attempt(invocation_reference,*,_development_fixture=None):
    started=time.monotonic();invocation=_read(invocation_reference)
    created=datetime.fromisoformat(invocation['created_at_utc'])
    _need(created.tzinfo is not None,'Invocation timestamp must have timezone')
    dispatch=datetime.now(timezone.utc).timestamp()-created.timestamp()
    _need(dispatch>=0,'Clock reversal makes dispatch budget accounting invalid')
    attempt_started=invocation['attempt_started_monotonic'];deadline=invocation['worker_deadline_monotonic']
    _need(all(type(v) in (int,float) and math.isfinite(v) for v in (attempt_started,deadline))
        and attempt_started<=started<deadline,'Invalid or expired shared monotonic attempt deadline')
    request_ref=invocation['item_request_reference'];request=_read(request_ref)
    _need(invocation['schema_version']==request['schema_version']==1,'Unknown worker request schema')
    _need(request['method_id'] in ('EA','ARO','OR') and request['plan_id']==invocation['plan_id'],'Worker method/plan differs')
    attempt=Path(invocation['attempt_directory']).resolve();result_path=Path(invocation['result_path']).resolve()
    _need(result_path==attempt/'worker_result.json' and not result_path.exists(),'Worker result path/overwrite differs')
    if _development_fixture is None:
        budget,options=_formal_prerequisites(request,invocation)
        if request['method_id']=='OR':
            from .formal_onset import load_formal_onset_problem
            prepare=load_formal_onset_problem
        else:
            prepare=load_formal_problem
        seal=SEAL_RESERVE_SECONDS
    else:
        _need(isinstance(_development_fixture,_DevelopmentFixture) and request['domain']==invocation['domain']=='development',
            'Development dependency injection cannot bypass formal gates')
        budget=validate_budget(_development_fixture.budget);options=_development_fixture.engine_options
        prepare=_development_fixture.prepare;seal=_development_fixture.seal_reserve_seconds
        _need(request['budget']==budget,'Fixture still requires frozen common budget')
    _need(invocation['plan_budget_seconds']==budget['cumulative_wall_seconds'],'Queue/common plan budget differs')
    prior=invocation['cumulative_plan_seconds'];remaining=invocation['remaining_seconds']
    _need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (prior,remaining,dispatch)),'Invalid cumulative wall input')
    _need(abs(invocation['plan_remaining_seconds']-(budget['cumulative_wall_seconds']-prior))<1e-8,
        'Queue cumulative budget reset or mismatch')
    _need(type(request['output_root']) is str and Path(request['output_root']).is_absolute(),'Absolute frozen planning output root required')
    carryover_ref=request.get('failure_carryover_reference')
    _need(carryover_ref==invocation.get('failure_carryover_reference'),'Worker carryover invocation differs')
    carryover=None
    if carryover_ref is not None:
        queue=_read(invocation['request_reference'])
        carryover=verify_failure_carryover(carryover_ref,request,queue['queue_root'])
        _need(prior>=carryover['accounted_seconds'] and invocation['opening_carryover_seconds']==carryover['accounted_seconds'], 'Worker carryover cost reset')
    root=Path(request['output_root']).resolve();previous_ref=invocation['previous_checkpoint'];previous=None
    if previous_ref is not None:
        _closure(previous_ref);previous=_read(previous_ref)
        _need(previous['request_reference']==request_ref and previous['output_root']==str(root),'Worker checkpoint input/output identity differs')
        previous_result=_read(invocation['previous_result_reference'])
        _need(previous_result['checkpoint']==previous_ref and previous_result['status']=='TIME_LIMIT','Previous queue result is not resumable')
        _need(prior+1e-8>=previous['cumulative_accounted_seconds'],'Prior queue attempt time cannot reset')
    else:
        opening=0. if carryover is None else carryover['accounted_seconds']
        _need(invocation['previous_result_reference'] is None and prior==opening and not root.exists(),'Unbound existing work/history cannot restart')
        root.mkdir(parents=True,exist_ok=False)
    def used():return time.monotonic()-attempt_started
    try:
        audit_configuration=request.get('numerical_audit')
        audit_reserve=0.
        if audit_configuration is not None:
            from .planning_audit_pipeline import validate_audit_configuration
            audit_reserve=validate_audit_configuration(audit_configuration,formal=_development_fixture is None,
                planning_sources=request.get('planning_source_references',[]))
            _need(audit_reserve<budget['final_certificate_reserve_seconds'],'Audit reserve exceeds existing final evidence reserve')
        template_options={key:request[key] for key in
            ('template_index_reference','template_association_review_reference') if key in request}
        if request['method_id']=='OR':
            template_options['onset_acceptance_reference']=request.get('onset_acceptance_reference')
        prepared=prepare(summary_reference=request.get('summary_reference'),audit_reference=request.get('audit_reference'),
            target_id=request['target_id'],method_id=request['method_id'],evidence_root=root/'raw',
            resume_evidence=previous is not None,
            resume_snapshot_reference=None if previous is None else previous['raw_evidence_snapshot'],
            **template_options)
        if carryover is not None:verify_prepared_problem(carryover,prepared.problem.identity)
        available=max(0.,deadline-time.monotonic()-seal)
        driver=run_budgeted_ccg(prepared.problem,outputroot=root/'driver',budget=budget,
            input_binding=dict(worker_request=request_ref,physical=prepared.input_binding),engine_options=options,
            evidence_sink_factory=prepared.sink_factory,spent_before_seconds=prior+used(),
            available_wall_seconds=available,resume=previous is not None,
            previous_driver_snapshot=None if previous is None else previous['driver_snapshot'],
            **({'postprocessing_reserve_seconds':audit_reserve} if audit_configuration is not None else {}))
        _need(driver['status'] in ('CONVERGED','TIME_LIMIT','BUDGET_EXHAUSTED'),'Driver requires diagnosis: '+driver['status'])
        raw=prepared.evidence.snapshot(attempt/'raw_evidence_snapshot.json')
        ds=_copy(root/'driver'/'driver_state.json',attempt/'driver_state_snapshot.json')
        ep=root/'driver'/'checkpoint.json';engine_ref=None if not ep.exists() else _copy(ep,attempt/'engine_checkpoint_snapshot.json')
        envelope=None if engine_ref is None else _read(engine_ref)
        engine=None if envelope is None else envelope['state']
        if engine is not None:
            _need(envelope['sha256']==canonical_hash(engine) and engine['engine_sha256']==_engine_hash() and
                engine['problem_sha256']==canonical_hash(prepared.problem.identity),'Raw engine checkpoint identity differs')
        refs=dict(raw_evidence_snapshot=raw,driver_snapshot=ds,engine_checkpoint_snapshot=engine_ref)
        record=_candidate(engine,prepared,request,driver,refs,prior+used())
        candidate_ref=_new(attempt/'candidate_solver_record.json',record)
        cumulative_left=budget['cumulative_wall_seconds']-prior-used()
        cannot_start_more=cumulative_left<=budget['single_master_seconds']+budget['final_certificate_reserve_seconds']+budget['shutdown_reserve_seconds']+seal
        complete=driver['status']=='CONVERGED' or (cannot_start_more and record['candidate'] is not None)
        if (driver['status']=='CONVERGED' or cannot_start_more) and record['candidate'] is None:
            raise RuntimeError('Planning budget exhausted without a usable candidate; diagnosis required')
        numerical_audit_ref=None
        if complete and audit_configuration is not None:
            from .planning_audit_pipeline import audit_candidate_record
            numerical_audit_ref=audit_candidate_record(candidate_record_reference=candidate_ref,prepared=prepared,
                configuration=audit_configuration,request_reference=request_ref,
                output_directory=attempt/'independent_audit',deadline=deadline-seal)
        checkpoint=dict(schema_version=1,request_reference=request_ref,invocation_reference=invocation_reference,
            output_root=str(root),previous_checkpoint=previous_ref,previous_result_reference=invocation['previous_result_reference'],
            **refs,candidate_record_reference=candidate_ref,cumulative_accounted_seconds=prior+used(),
            driver_status=driver['status'],engine_status=None if engine is None else engine['status'],
            resume_without_engine_checkpoint=engine_ref is None,full_support_verified=False,formal_OOS=False)
        checkpoint_ref=_new(attempt/'worker_checkpoint.json',checkpoint)
        elapsed=used()
        _need(time.monotonic()<deadline and prior+elapsed<budget['cumulative_wall_seconds'],'Preparation/export exceeded remaining or cumulative wall budget')
        result=dict(schema_version=1,status='COMPLETE' if complete else 'TIME_LIMIT',invocation_sha256=invocation_reference['sha256'],
            artifacts=[raw,ds,*([] if engine_ref is None else [engine_ref]),candidate_ref,checkpoint_ref,*([] if numerical_audit_ref is None else [numerical_audit_ref])],checkpoint=checkpoint_ref,
            resumable=not complete,declared_work_complete=complete,scientific_acceptance=False,formal_OOS=False,
            full_support_verified=numerical_audit_ref is not None,numerical_audit_reference=numerical_audit_ref,
            numerical_status='PASS_INDEPENDENT_PLANNING_NUMERICAL_AUDIT' if numerical_audit_ref is not None else 'PENDING_INDEPENDENT_NUMERICAL_AUDIT',driver_status=driver['status'],
            raw_engine_status=None if engine is None else engine['status'],has_candidate=record['candidate'] is not None,
            cumulative_accounted_seconds=prior+elapsed,cumulative_seconds_provisional=True,
            accounting_authority='campaign_batch immutable attempt receipts including queue verification',
            attempt_accounted_seconds=elapsed,dispatch_seconds=dispatch,
            queue_prior_attempt_seconds=prior,worker_seal_reserve_seconds=seal,
            failure_carryover_reference=carryover_ref,opening_carryover_seconds=0. if carryover is None else carryover['accounted_seconds'],
            attempt_started_monotonic=attempt_started,worker_deadline_monotonic=deadline,
            stop_reason='bounded_planning_work_complete_pending_audit' if complete else 'parent_slice_or_pending_boundary')
        _new(result_path,result);return result
    except Exception as exc:
        failure=attempt/'worker_failure.json'
        if not failure.exists():_new(failure,dict(status='FAILED_WORKER_REQUIRES_DIAGNOSIS',error_type=type(exc).__name__,error=str(exc),
            cumulative_accounted_seconds=prior+used(),partial_outputs_preserved=True,formal_OOS=False,scientific_acceptance=False))
        raise


def worker_main():
    path=Path(os.environ['EVCS_BATCH_ATTEMPT_REQUEST']);ref=file_reference(path)
    _need(ref['sha256']==os.environ['EVCS_BATCH_INVOCATION_SHA256'],'Worker invocation SHA differs')
    _need(_read(ref)['domain']=='formal','Production CLI only accepts formal queue invocations')
    result=run_attempt(ref);print(json.dumps(result,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(worker_main())
