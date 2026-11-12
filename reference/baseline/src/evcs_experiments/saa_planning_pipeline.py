"""SAA numerical phases from an explicitly admitted original assembled model.

Assembly/source admission is upstream. This connects native master, resumable
fixed-Q, and independent joined audit without CCG or formal queue acceptance.
"""
import json
import math
from pathlib import Path
import time

import numpy as np

from .campaign_batch import file_reference, verify_reference
from .runtime.jobctl import atomic_json
from . import saa_fixed_plan, saa_fixed_plan_audit, saa_candidate_audit, saa_streaming_store, saa_streaming_solve


def _need(ok,msg):
    if not ok:raise ValueError(msg)


def _read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())


def _write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return file_reference(path)


def _nested(value, deadline=None):
    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Phase evidence verification deadline')
    if isinstance(value,dict):
        if set(value)=={'path','sha256','size_bytes'}:verify_reference(value)
        else:
            for child in value.values():_nested(child,deadline)
    elif isinstance(value,list):
        for child in value:_nested(child,deadline)


def run_saa_numerical_phases(*, model_manifest_reference,first_stage,recourses,case,tasks,scene_loader,
        input_binding,scenario_ids,scenario_sha256s,template_references,helper_reference,source_references,
        output_directory,deadline,cumulative_budget_seconds,prior_accounted_seconds=0.,
        resume_checkpoint_reference=None,master_seconds=5.,audit_reserve_seconds=5.,seal_reserve_seconds=1.,
        domain='development',formal_admission_reference=None):
    """No formal admission. Resume reuses a committed master and each saved Q.

    Provisional elapsed time includes reload/source/solve/export/audit. Queue
    receipts must cover post-return work before any production integration.
    """
    began=time.monotonic()
    _need(domain in ('development','formal'),'Explicit SAA phase domain required')
    formal=domain=='formal'
    if formal:
        _need(formal_admission_reference is not None,'Formal SAA worker admission required')
        admission=_read(formal_admission_reference)
        _need(admission['kind']=='SAA_FORMAL_PHASE_ADMISSION_V1' and admission['domain']=='formal'
              and admission['model_manifest_reference']==model_manifest_reference
              and admission['planning_source_references']==source_references,
              'Formal SAA phase admission source/model differs')
        _need(cumulative_budget_seconds==7200 and master_seconds==600 and audit_reserve_seconds==600
              and seal_reserve_seconds==10,'Frozen formal SAA resource policy differs')
    else:
        _need(formal_admission_reference is None,'Development cannot claim formal admission')
    final_reserve=1790 if formal else audit_reserve_seconds
    complete_status='COMPLETE_FORMAL_SAA_NUMERICAL_PHASES' if formal else 'COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES'
    for v in (deadline,cumulative_budget_seconds,prior_accounted_seconds,master_seconds,audit_reserve_seconds,seal_reserve_seconds):
        _need(type(v) in (int,float) and math.isfinite(v) and v>=0,'Invalid finite time controls')
    _need(0<cumulative_budget_seconds<=7200 and 0<master_seconds<=600 and audit_reserve_seconds>0 and seal_reserve_seconds>0,
        'Invalid bounded phase controls')
    end=min(deadline,began+cumulative_budget_seconds-prior_accounted_seconds)
    _need(end>began+seal_reserve_seconds,'No remaining phase budget')
    for ref in source_references:verify_reference(ref)
    for path in (__file__,saa_fixed_plan.__file__,saa_fixed_plan_audit.__file__,saa_candidate_audit.__file__,
                 saa_streaming_store.__file__,saa_streaming_solve.__file__):
        _need(file_reference(path) in source_references,'Numerical phase source binding missing')
    header=_read(model_manifest_reference)
    _need(header['domain']==domain,'SAA model/phase domain differs')
    _need(header['input_binding']==input_binding and header['scenario_ids']==list(scenario_ids)
          and header['scenario_sha256s']==list(scenario_sha256s), 'SAA model input binding differs')
    identity=dict(kind='DEVELOPMENT_SAA_NUMERICAL_PHASES_V1',model_manifest_reference=model_manifest_reference,
        sources=source_references,helper_reference=helper_reference,input_binding=input_binding,
        scenario_ids=list(scenario_ids),scenario_sha256s=list(scenario_sha256s),template_references=template_references,
        case_sha256=case.case_sha256,task_sha256=tasks.task_sha256,cumulative_budget_seconds=cumulative_budget_seconds,
        master_seconds=master_seconds,audit_reserve_seconds=audit_reserve_seconds,seal_reserve_seconds=seal_reserve_seconds)
    if formal:
        identity.update(kind='FORMAL_SAA_NUMERICAL_PHASES_V1',formal_admission_reference=formal_admission_reference,
                        final_certificate_reserve_seconds=1790,master_threads=4,recourse_workers=1,master_mip_gap=.001)
    root=Path(output_directory).resolve();active=root/'checkpoint.json'
    if resume_checkpoint_reference is None:
        root.mkdir(parents=True,exist_ok=False);request_ref=_write(root/'request.json',identity)
        master_ref=q_ref=envelope_ref=audit_ref=None
    else:
        old=_read(resume_checkpoint_reference)
        _need(file_reference(active)['sha256']==resume_checkpoint_reference['sha256'],'Stale phase checkpoint')
        request_ref=old['request_reference'];_need(_read(request_ref)==identity,'Phase model/source/support binding changed')
        _need(prior_accounted_seconds>=old['cumulative_provisional_seconds'],'Cumulative budget reset')
        _need(old['status']!='FAILED','Failed phase needs diagnosis')
        master_ref=old['master_result_reference'];q_ref=old['fixed_q_checkpoint_reference']
        envelope_ref=old['template_envelope_reference'];audit_ref=old['candidate_audit_reference']
        for ref in (master_ref,q_ref,envelope_ref,audit_ref):
            if ref is not None:_nested(_read(ref),end)
        if q_ref is not None:
            q_saved=_read(q_ref);_nested(_read(q_saved['request_reference']),end)
            for row_ref in q_saved['attempts']:_nested(_read(row_ref),end)
        if envelope_ref is not None:
            for template_ref in _read(envelope_ref)['template_references']:
                _nested(_read(template_ref),end)
        _need(time.monotonic()<end,'Phase restore exceeded shared deadline')
    def used():return time.monotonic()-began
    def save(status):
        value=dict(status=status,request_reference=request_ref,master_result_reference=master_ref,
            fixed_q_checkpoint_reference=q_ref,template_envelope_reference=envelope_ref,candidate_audit_reference=audit_ref,
            prior_accounted_seconds=prior_accounted_seconds,attempt_provisional_seconds=used(),
            cumulative_provisional_seconds=prior_accounted_seconds+used(),queue_receipt_still_required=True,
            scientific_spec_pass=False,formal_OOS=False,formal_admission=formal)
        atomic_json(active,value);snapshots=root/'checkpoints';snapshots.mkdir(exist_ok=True)
        snapshot=snapshots/f'checkpoint_{len(list(snapshots.iterdir())):06d}.json'
        with snapshot.open('xb') as f:f.write(active.read_bytes())
        snapshot_ref=file_reference(snapshot)
        if status!='FAILED' and time.monotonic()>=end:raise TimeoutError('Phase checkpoint seal exceeded shared deadline')
        return dict(value,checkpoint_reference=snapshot_ref)
    try:
        # A complete source-bound model must survive a short grant without
        # paying its potentially large reload cost before master admission.
        if master_ref is None and end-time.monotonic()<=master_seconds+final_reserve+seal_reserve_seconds:
            return save('TIME_LIMIT_BEFORE_MASTER')
        model=saa_streaming_store.load_streaming_saa(model_manifest_reference,first_stage=first_stage,
            expected_input_binding=input_binding,scenario_ids=scenario_ids,scenario_sha256s=scenario_sha256s,deadline=end)
        _need(model.manifest['domain']==domain,'SAA model/phase domain differs')
        if time.monotonic()>=end:raise TimeoutError('Phase reload exceeded shared deadline')
        if audit_ref is not None:
            _need(end-time.monotonic()>seal_reserve_seconds,'No reserve for completed phase seal')
            return save(complete_status)
        if master_ref is None:
            if end-time.monotonic()<=master_seconds+final_reserve+seal_reserve_seconds:
                return save('TIME_LIMIT_BEFORE_MASTER')
            result=saa_streaming_solve.solve_streaming_saa(model,recourses,case=case,tasks=tasks,scene_loader=scene_loader,
                output_directory=root/'master',backend='gurobi' if formal else 'highs',time_limit_seconds=master_seconds,
                deadline=end-seal_reserve_seconds,evidence_reserve_seconds=final_reserve,
                optimizer_threads=4 if formal else 1,mip_relative_gap=.001)
            master_ref=file_reference(root/'master/result.json');save('MASTER_SAVED')
        else:result=_read(master_ref)
        _need(result['primal'] is not None and result['lower_bound_usd'] is not None,'SAA master lacks usable incumbent/bound')
        _nested(result);plan=model.plan(np.load(result['primal']['path'],allow_pickle=False))
        if q_ref is None or _read(q_ref)['status']!='COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT':
            if end-time.monotonic()<=audit_reserve_seconds+seal_reserve_seconds+2:return save('TIME_LIMIT_BEFORE_FIXED_Q')
            q=saa_fixed_plan.complete_saa_fixed_plan_q(plan=plan,case=case,tasks=tasks,scenario_ids=scenario_ids,
                scenario_sha256s=scenario_sha256s,template_sha256s=[r['template_sha256'] for r in model.manifest['scenarios']],
                scenario_cost_constants_usd=model.manifest['scenario_cost_constants_usd'],template_loader=recourses.__getitem__,
                scene_loader=scene_loader,source_references=source_references,input_binding={'pipeline_request':request_ref},
                output_directory=root/'fixed_q',deadline=end-audit_reserve_seconds-seal_reserve_seconds,
                cumulative_budget_seconds=cumulative_budget_seconds,prior_accounted_seconds=prior_accounted_seconds+used(),
                resume_checkpoint_reference=q_ref,domain=domain)
            q_ref=q['checkpoint_reference'];save('FIXED_Q_SAVED')
            if q['status']!='COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT':return save('TIME_LIMIT_FIXED_Q')
        q=_read(q_ref)
        if envelope_ref is None:
            envelope_ref=saa_fixed_plan_audit.bind_saa_template_envelope(request_reference=q['request_reference'],
                template_references=template_references,first_stage_sha256=first_stage.matrix_sha256,
                source_references=source_references,output_path=root/'template_envelope.json')
            save('READY_FOR_AUDIT')
        if end-time.monotonic()<=seal_reserve_seconds:return save('TIME_LIMIT_BEFORE_AUDIT')
        audit_ref=saa_candidate_audit.audit_saa_candidate(model=model,recourses=recourses,case=case,tasks=tasks,
            scene_loader=scene_loader,master_result_reference=master_ref,checkpoint_reference=q_ref,
            fixed_q_request_reference=q['request_reference'],template_envelope_reference=envelope_ref,
            helper_reference=helper_reference,source_references=source_references,deadline=end-seal_reserve_seconds,
            output_path=root/'candidate_audit.json')
        _need(time.monotonic()<end,'Numerical phases exceeded shared budget')
        return save(complete_status)
    except Exception:
        save('FAILED');raise
