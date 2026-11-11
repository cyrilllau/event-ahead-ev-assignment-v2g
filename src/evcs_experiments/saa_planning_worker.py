"""Dedicated queue adapter for source-bound SAA phases; no C&CG resume.

This entry explicitly chooses bounded original assembly or a source-bound
preassembled manifest. Assembly is never inferred from an output directory. Original queue receipts
remain the accounting authority, including preparation and failed attempts.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from .campaign_batch import file_reference, verify_reference, read_json, _result
from .formal_planning_worker import _formal_prerequisites, _closure
from .formal_saa import prepare_formal_saa, assemble_prepared_saa
from .planning_budget import validate_budget
from .random_streams import canonical_hash
from .saa_planning_pipeline import run_saa_numerical_phases


def _need(ok,message):
    if not bool(ok):raise ValueError(message)


def _read(ref):
    verify_reference(ref);return read_json(ref['path'])


def _new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return file_reference(path)


def _preparation_receipt(receipt_ref, *, plan_id, queue_reference, model_reference):
    """Verify actual queue-produced model evidence, not an accounting-card claim."""
    _closure(receipt_ref);receipt=_read(receipt_ref)
    _need(receipt['plan_id']==plan_id and receipt['request_reference']==queue_reference
          and receipt['returncode']==0,'Preparation receipt plan/queue/exit differs')
    invocation=_read(receipt['invocation_reference'])
    _need(invocation['plan_id']==plan_id and invocation['request_reference']==queue_reference
          and invocation['item_id']==receipt['item_id']
          and str(Path(invocation['result_path']).resolve())==receipt['result_reference']['path'],
          'Preparation invocation producer differs')
    result=_result(receipt['result_reference'],receipt['invocation_reference'])
    _need(receipt['status']==result['status'],'Preparation receipt/result status differs')
    seconds=receipt['elapsed_seconds']
    _need(type(seconds) in (float,int) and math.isfinite(seconds) and seconds>=0,
          'Invalid preparation receipt elapsed time')
    produced=result['status']=='COMPLETE' and model_reference in result['artifacts']
    if produced:_closure(model_reference)
    return seconds,produced


@dataclass(frozen=True)
class _DevelopmentFixture:
    """Explicit tiny physical injection, impossible in a formal invocation."""
    prepare: object
    budget: dict
    master_seconds: float=5.
    audit_reserve_seconds: float=5.
    seal_reserve_seconds: float=1.


def run_attempt(invocation_reference, *, _development_fixture=None):
    entry=time.monotonic();inv=_read(invocation_reference)
    started=inv['attempt_started_monotonic'];deadline=inv['worker_deadline_monotonic']
    _need(all(type(x) in (float,int) and math.isfinite(x) for x in (started,deadline))
          and started<=entry<deadline,'Invalid original invocation deadline')
    reqref=inv['item_request_reference'];req=_read(reqref)
    _need(req['schema_version']==inv['schema_version']==1 and req['method_id']=='SAA'
          and req['plan_id']==inv['plan_id'],'SAA method/plan identity differs')
    attempt=Path(inv['attempt_directory']).resolve();result_path=Path(inv['result_path']).resolve()
    _need(result_path==attempt/'worker_result.json' and not result_path.exists(),'Wrong worker result destination')
    formal=_development_fixture is None
    if formal:
        budget,_=_formal_prerequisites(req,inv)
        prepare=prepare_formal_saa;master=600.;audit=600.;seal=10.
    else:
        _need(isinstance(_development_fixture,_DevelopmentFixture)
              and req['domain']==inv['domain']=='development','Fixture cannot bypass formal source admission')
        budget=validate_budget(_development_fixture.budget)
        _need(req['budget']==budget,'Development budget still must be frozen')
        prepare=_development_fixture.prepare;master=_development_fixture.master_seconds
        audit=_development_fixture.audit_reserve_seconds;seal=_development_fixture.seal_reserve_seconds
    checkpoint_reserve=req.get('model_checkpoint_reserve_seconds',60. if formal else seal)
    reload_reserve=req.get('model_reload_reserve_seconds',60. if formal else seal)
    _need(all(type(v) in (float,int) and math.isfinite(v) and seal<=v<=300.
              for v in (checkpoint_reserve,reload_reserve)), 'Invalid model checkpoint/reload reserve')
    prior=inv['cumulative_plan_seconds']
    _need(type(prior) in (float,int) and math.isfinite(prior) and prior>=0
          and inv['plan_budget_seconds']==budget['cumulative_wall_seconds']
          and abs(inv['plan_remaining_seconds']-(budget['cumulative_wall_seconds']-prior))<1e-8,
          'SAA queue budget reset or mismatch')
    # The queue starts accounting before source/history verification, whereas
    # remaining_seconds is the smaller grant available when it dispatches us.
    # Preserve that prelude in elapsed(), without pretending it is worker time.
    _need(type(inv['remaining_seconds']) in (float,int) and math.isfinite(inv['remaining_seconds'])
          and inv['remaining_seconds']>0
          and deadline-entry<=inv['remaining_seconds']+1e-8
          and deadline-started<=inv['plan_remaining_seconds']+1e-8
          and inv['remaining_seconds']<=inv['plan_remaining_seconds'], 'SAA deadline exceeds queue grant')
    _need(isinstance(req['output_root'],str) and Path(req['output_root']).is_absolute(),'Absolute SAA output root required')
    root=Path(req['output_root']).resolve();previous_ref=inv['previous_checkpoint'];previous=None
    carry=req.get('failure_carryover_reference')
    _need(carry==inv.get('failure_carryover_reference'),'SAA carryover request differs')
    if carry is not None:
        from . import saa_failure_carryover
        _need(_read(carry).get('kind')==saa_failure_carryover.KIND,
              'CCG failure carryover is not a SAA preparation receipt')
        _need(file_reference(saa_failure_carryover.__file__) in req['planning_source_references'],
              'SAA carryover helper source is not bound')
        queue=_read(inv['request_reference'])
        verified=saa_failure_carryover.verify_saa_failure_carryover(carry,req,queue['queue_root'])
        opening=verified['accounted_seconds']
        _need(verified['reference']==carry and verified['plan_id']==req['plan_id']
              and inv.get('opening_carryover_seconds')==opening and prior>=opening,
              'SAA failure opening balance differs')
        _need(req['model_input_mode']=='assemble','SAA failed preparation cannot import a free preassembled model')
        if previous_ref is None:
            _need(prior==opening and inv['previous_result_reference'] is None,
                  'SAA first successor attempt must inherit exact failed-parent cost')
    else:
        _need(inv.get('opening_carryover_seconds',0.)==0.,'SAA unbound opening carryover')
    if previous_ref is not None:
        _closure(previous_ref);previous=_read(previous_ref)
        oldresult=_read(inv['previous_result_reference'])
        _need(previous['request_reference']==reqref and previous['output_root']==str(root)
              and oldresult['status']=='TIME_LIMIT' and oldresult['checkpoint']==previous_ref
              and prior>=previous['cumulative_accounted_seconds'],'Changed or reset SAA resume')
    else:
        _need(not root.exists(),'Uncommitted SAA output cannot silently become resume')
        root.mkdir(parents=True,exist_ok=False)
    def elapsed():return time.monotonic()-started
    try:
        # These are request references, not opportunistic side files.
        mode=req['model_input_mode']
        _need(mode in ('assemble','preassembled'),'Explicit SAA model input mode required')
        verify_reference(req['helper_reference'])
        if mode=='preassembled':verify_reference(req['model_manifest_reference'])
        else:_need('model_manifest_reference' not in req,'Assembly cannot silently reuse side model')
        _need(isinstance(req['template_references'],list) and req['template_references'], 'Explicit original template inventory required')
        for ref in req['template_references']:verify_reference(ref)
        prepared=prepare(summary_reference=req['summary_reference'],audit_reference=req['audit_reference'],
            target_id=req['target_id'],template_index_reference=req.get('template_index_reference'),
            template_association_review_reference=req.get('template_association_review_reference'))
        physical=prepared.physical
        if formal and mode=='preassembled':
            # Provenance of upstream assembly is required; its cost must already
            # be covered by an explicit queue carryover, not erased by this root.
            preparation=_read(req['model_preparation_reference'])
            _need(preparation['kind']=='SAA_MODEL_PREPARATION_ACCOUNTING_V1'
                  and preparation['model_manifest_reference']==req['model_manifest_reference']
                  and preparation['plan_id']==req['plan_id'] and preparation['accounting_PASS'] is True
                  and type(preparation['accounted_seconds']) in (float,int)
                  and math.isfinite(preparation['accounted_seconds']) and 0<=preparation['accounted_seconds']<=prior, 
                  'Preassembled SAA model preparation cost/source not admitted')
            receipts=preparation['queue_receipt_references']
            _need(isinstance(receipts,list) and receipts,'SAA preparation requires original queue receipts')
            total=0.;model_produced=False
            for receipt_ref in receipts:
                seconds,produced=_preparation_receipt(receipt_ref,plan_id=req['plan_id'],
                    queue_reference=inv['request_reference'],model_reference=req['model_manifest_reference'])
                model_produced=model_produced or produced
                total+=seconds
            _need(len({r['path'] for r in receipts})==len(receipts)
                  and total==preparation['accounted_seconds'] and model_produced, 'SAA preparation receipt costs/model differ')
        if previous is not None:
            model_ref=previous['model_manifest_reference'];verify_reference(model_ref)
            if mode=='preassembled':_need(model_ref==req['model_manifest_reference'],'Changed preassembled model')
        elif mode=='preassembled':model_ref=req['model_manifest_reference']
        else:
            ceiling=req['max_coefficient_bytes']
            _need(type(ceiling) is int and ceiling>0,'Explicit assembly resource ceiling required')
            if formal:
                assembled=assemble_prepared_saa(prepared,output_directory=root/'model',
                    max_coefficient_bytes=ceiling,deadline=deadline-checkpoint_reserve)
            else:
                from .saa_streaming import assemble_streaming_saa
                assembled=assemble_streaming_saa(physical.first_stage,prepared.recourses,
                    scenario_ids=prepared.recourses.scenario_ids,scenario_sha256s=prepared.recourses.scenario_sha256s,
                    empirical_probabilities=prepared.empirical_probabilities,input_binding=prepared.input_binding,
                    scenario_cost_constants_usd=getattr(prepared,'scenario_cost_constants_usd',None),
                    output_directory=root/'model',max_coefficient_bytes=ceiling,deadline=deadline-checkpoint_reserve,domain='development')
            model_ref=file_reference(assembled.root/'manifest.json')
        # A completed assembly is independently resumable even before any numerical phase.
        if previous is not None:
            completion_ref=previous.get('model_completion_reference')
            _need(completion_ref is not None,'SAA resume lacks committed model completion')
            completion=_read(completion_ref)
            _need(completion['kind']=='SAA_COMPLETE_MODEL_CHECKPOINT_V1'
                  and completion['model_manifest_reference']==model_ref
                  and completion['request_reference']==reqref
                  and completion['planning_source_references']==req['planning_source_references']
                  and completion['model_input_mode']==mode
                  and completion['input_binding']==prepared.input_binding
                  and completion['model_checkpoint_reserve_seconds']==checkpoint_reserve
                  and completion['model_reload_reserve_seconds']==reload_reserve,
                  'SAA committed preparation/model/source differs')
        else:
            manifest=_read(model_ref)
            _need(manifest['status']=='ASSEMBLED_PENDING_RESOURCE_AND_INDEPENDENT_NUMERICAL_AUDIT'
                  and canonical_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})==manifest['manifest_sha256']
                  and manifest['domain']==req['domain'] and manifest['input_binding']==prepared.input_binding
                  and manifest['scenario_ids']==list(prepared.recourses.scenario_ids)
                  and manifest['scenario_sha256s']==list(prepared.recourses.scenario_sha256s),
                  'SAA assembly is incomplete or has changed identity')
            _need(time.monotonic()<deadline-seal,'No reserve to seal completed SAA model')
            completion_ref=_new(root/'model_completion.json',dict(kind='SAA_COMPLETE_MODEL_CHECKPOINT_V1',
                request_reference=reqref,model_manifest_reference=model_ref,model_input_mode=mode,
                planning_source_references=req['planning_source_references'],input_binding=prepared.input_binding,
                model_checkpoint_reserve_seconds=checkpoint_reserve,model_reload_reserve_seconds=reload_reserve,
                preparation_elapsed_seconds=elapsed(),prior_cumulative_seconds=prior,
                accounting_authority='queue receipt required; preparation time included in current attempt',
                numerical_audit_complete=False,scientific_acceptance=False,formal_OOS=False))
        admission_ref=None
        if formal:
            if previous is not None:admission_ref=previous['formal_admission_reference']
            else:admission_ref=_new(root/'formal_admission.json',dict(kind='SAA_FORMAL_PHASE_ADMISSION_V1',domain='formal',
                request_reference=reqref,model_manifest_reference=model_ref,
                planning_source_references=req['planning_source_references'],budget=budget,
                model_input_mode=mode,model_preparation_reference=req.get('model_preparation_reference')))
        phase_previous=None if previous is None else previous['phase_checkpoint_reference']
        final_reserve=1790. if formal else audit
        if phase_previous is None and deadline-time.monotonic()<=master+final_reserve+2*seal+reload_reserve:
            phase=dict(status='TIME_LIMIT_ASSEMBLY_COMPLETE_BEFORE_MODEL_RELOAD',checkpoint_reference=None,
                candidate_audit_reference=None,fixed_q_checkpoint_reference=None)
        else:
            phase=run_saa_numerical_phases(model_manifest_reference=model_ref,
                first_stage=physical.first_stage,recourses=prepared.recourses,case=physical.case,tasks=physical.tasks,
                scene_loader=physical.bank.load,input_binding=prepared.input_binding,
                scenario_ids=prepared.recourses.scenario_ids,scenario_sha256s=prepared.recourses.scenario_sha256s,
                template_references=req['template_references'],helper_reference=req['helper_reference'],
                source_references=req['planning_source_references'],output_directory=root/'phases',
                deadline=deadline-seal,cumulative_budget_seconds=budget['cumulative_wall_seconds'],
                prior_accounted_seconds=prior+elapsed(),
                resume_checkpoint_reference=phase_previous,
                master_seconds=master,audit_reserve_seconds=audit,seal_reserve_seconds=seal,
                domain='formal' if formal else 'development',formal_admission_reference=admission_ref)
        complete=phase['status'] in ('COMPLETE_FORMAL_SAA_NUMERICAL_PHASES','COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES')
        _need(complete or phase['status'].startswith('TIME_LIMIT_'),'SAA numerical failure requires diagnosis')
        joined_ref=phase['candidate_audit_reference'];joined=None;candidate=None;rawstatus=None
        if complete:
            _need(joined_ref is not None,'Complete SAA phase lacks joined independent audit')
            joined=_read(joined_ref)
            _need(joined['status']=='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT' and joined['passed'] is True,
                  'SAA joined audit did not pass')
            qcp=_read(phase['fixed_q_checkpoint_reference']);qrequest=_read(qcp['request_reference']);plan=qrequest['plan']
            _need(canonical_hash(plan)==joined['plan_sha256'],'SAA native/fixed-Q plan differs')
            rawstatus=joined['native_status']
            candidate=dict(plan=plan,plan_sha256=joined['plan_sha256'],lower_bound_usd=joined['lower_bound_usd'],
                upper_bound_usd=joined['upper_bound_usd'],gap_relative=joined['gap_relative'],
                native_master_status=rawstatus,native_master_solver_status=joined['native_solver_status'],
                risk_certificate_kind='fixed_empirical')
        driver_status=('CONVERGED' if joined['normal_gap_met'] else 'TIME_LIMIT') if complete else 'TIME_LIMIT'
        candidate_ref=_new(attempt/'candidate_solver_record.json',dict(plan_id=req['plan_id'],method_id='SAA',
            driver_status=driver_status,raw_engine_status=rawstatus,candidate=candidate,
            request_reference=reqref,candidate_audit_reference=joined_ref,formal_OOS=False))
        numerical_ref=None
        if complete:
            from .saa_record_bridge import build_saa_record_bridge
            numerical_ref=build_saa_record_bridge(request_reference=reqref,candidate_record_reference=candidate_ref,
                candidate_audit_reference=joined_ref,output_path=attempt/'numerical_audit.json')
        cp=_new(attempt/'worker_checkpoint.json',dict(schema_version=1,request_reference=reqref,output_root=str(root),
            phase_checkpoint_reference=phase['checkpoint_reference'],phase_status=phase['status'],
            model_completion_reference=completion_ref,model_manifest_reference=model_ref,formal_admission_reference=admission_ref,
            candidate_record_reference=candidate_ref,previous_checkpoint=previous_ref,
            cumulative_accounted_seconds=prior+elapsed(),formal_OOS=False))
        artifacts=[candidate_ref,cp,model_ref,completion_ref]
        if phase['checkpoint_reference'] is not None:artifacts.append(phase['checkpoint_reference'])
        for ref in (joined_ref,numerical_ref,admission_ref):
            if ref is not None and ref not in artifacts:artifacts.append(ref)
        used=elapsed()
        _need(time.monotonic()<deadline and prior+used<budget['cumulative_wall_seconds'],'SAA worker export exceeded original budget')
        result=dict(schema_version=1,status='COMPLETE' if complete else 'TIME_LIMIT',
            invocation_sha256=invocation_reference['sha256'],artifacts=artifacts,checkpoint=cp,
            resumable=not complete,declared_work_complete=complete,scientific_acceptance=False,formal_OOS=False,
            full_support_verified=complete,numerical_audit_reference=numerical_ref,
            driver_status=driver_status,raw_engine_status=rawstatus,has_candidate=candidate is not None,
            cumulative_accounted_seconds=prior+used,cumulative_seconds_provisional=True,
            attempt_accounted_seconds=used,queue_prior_attempt_seconds=prior,worker_seal_reserve_seconds=seal,
            attempt_started_monotonic=started,worker_deadline_monotonic=deadline,
            failure_carryover_reference=carry,opening_carryover_seconds=inv.get('opening_carryover_seconds',0.))
        _new(result_path,result);return result
    except Exception as exc:
        failure=attempt/'worker_failure.json'
        if not failure.exists():_new(failure,dict(status='FAILED_SAA_WORKER_REQUIRES_DIAGNOSIS',
            error_type=type(exc).__name__,error=str(exc),cumulative_accounted_seconds=prior+elapsed(),
            partial_outputs_preserved=True,formal_OOS=False,scientific_acceptance=False))
        raise


def worker_main():
    path=Path(os.environ['EVCS_BATCH_ATTEMPT_REQUEST']);ref=file_reference(path)
    _need(ref['sha256']==os.environ['EVCS_BATCH_INVOCATION_SHA256'],'SAA invocation hash differs')
    _need(_read(ref)['domain']=='formal','SAA CLI accepts formal invocations only')
    print(json.dumps(run_attempt(ref),allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(worker_main())
