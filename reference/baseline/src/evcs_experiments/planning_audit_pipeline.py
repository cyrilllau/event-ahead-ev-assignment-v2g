"""Independent new-plan audit orchestration inside the frozen planning budget."""
from pathlib import Path
import json
import math
import time
import numpy as np

from .campaign_batch import file_reference, verify_reference
from .raw_template_audit import document, record, need
from .raw_plan_audit import audit_fixed_plan_recourses, audit_selected_probability
from .random_streams import canonical_hash


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return file_reference(path)


def validate_audit_configuration(config, *, formal, planning_sources):
    need(config['schema_version']==1,'Unknown numerical audit configuration')
    reserve=config['postprocessing_reserve_seconds']
    need(type(reserve) in (float,int) and math.isfinite(reserve) and reserve>0,'Invalid numerical audit reserve')
    sources=config['sources'];need(set(sources)=={'helper','numerical_kernel'},'Exact numerical audit sources required')
    for r in sources.values():verify_reference(r)
    if formal:
        review=document(config['implementation_review_reference'])
        need(review['status']=='PASS_INDEPENDENT_GENERIC_PLANNING_PIPELINE_REVIEW' and review['passed'] is True,
             'Generic production pipeline source is not independently reviewed')
        need(review['planning_sources']==planning_sources and review['numerical_sources']==sources,
             'Reviewed pipeline source differs')
        # This split does not add a budget: ordinary admission still retains
        # all1790 seconds; final pending drain leaves600 for independent audit.
        need(reserve==600.,'Formal audit must use predeclared600s inside1790s final evidence reserve')
    return reserve


def audit_candidate_record(*, candidate_record_reference, prepared, configuration,
                           request_reference, output_directory, deadline):
    from .raw_master_audit import audit_selected_master
    started=time.monotonic()
    need(started<deadline and deadline-started<=7190,'Invalid shared audit deadline')
    record_value=document(candidate_record_reference);candidate=record_value['candidate']
    need(candidate is not None,'No candidate to audit')
    request=document(request_reference)
    need(record_value['plan_id']==request['plan_id'] and record_value['method_id']==request['method_id'] and
         record_value['target_id']==request['target_id'],'Candidate worker identity differs')
    need(candidate['engine_identity']==prepared.problem.identity and
         candidate['input_binding']==prepared.input_binding,'Candidate physical source differs')
    plan=candidate['plan'];iteration=candidate['best_iteration']
    need(canonical_hash(plan)==candidate['plan_sha256'],'Candidate plan semantic identity differs')
    snapshot_reference=record_value['snapshots']['raw_evidence_snapshot'];snapshot=document(snapshot_reference)
    expected_request=file_reference(prepared.evidence.root/'request.json')
    need(snapshot['request']==expected_request,'Candidate raw request differs')
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=False)
    sources=configuration['sources']
    recourse=audit_fixed_plan_recourses(snapshot_reference=snapshot_reference,
        expected_request_reference=expected_request,plan_dict=plan,iteration=iteration,
        case=prepared.evidence.case,tasks=prepared.evidence.tasks,scene_loader=prepared.evidence._scene,
        helper_source_reference=sources['helper'],numerical_kernel_source_reference=sources['numerical_kernel'],deadline=deadline)
    need(recourse['q_usd']==candidate['q_usd'],'Candidate Q differs from actual numerical proof')
    recourse_ref=_write(output/'recourse.json',recourse)
    probability=audit_selected_probability(recourse_audit=recourse,snapshot_reference=snapshot_reference,
        expected_request_reference=expected_request,plan_dict=plan)
    need(time.monotonic()<deadline,'Probability audit exceeded shared deadline')
    probability_ref=_write(output/'probability.json',probability)
    master_results=[]
    lower_witness_selected=False
    for reference in snapshot['events']:
        event=document(reference)
        if event['event']!='rmp' or event['status']!='RAW_EVIDENCE_RETAINED_PENDING_AUDIT':continue
        folder=Path(reference['path']).parent
        if not (folder/'solution.json').exists():
            # A no-incumbent search bound needs an explicit additional audit
            # contract; do not quietly treat it as a verified candidate bound.
            continue
        saved_solver=document(next(r for r in event['artifacts'] if Path(r['path']).name=='solver.json'))
        contributes_lb=(not lower_witness_selected and saved_solver.get('lower_bound_usd')==candidate['lower_bound_usd'])
        if event['iteration']!=iteration and not contributes_lb:
            continue
        lower_witness_selected=lower_witness_selected or contributes_lb
        solution=json.loads((folder/'solution.json').read_text())
        master_plan={key:solution[key] for key in ('x','r','task_sha256','config_hash')}
        value=audit_selected_master(snapshot_reference=snapshot_reference,
            expected_request_reference=expected_request,plan_dict=master_plan,iteration=event['iteration'],
            first_stage=prepared.problem.first_stage,recourse_loader=prepared.problem.recourses.__getitem__,
            ambiguity=prepared.problem.ambiguity,case=prepared.evidence.case,tasks=prepared.evidence.tasks,
            scene_loader=prepared.evidence._scene,helper_source_reference=sources['helper'],
            q_usd=candidate['q_usd'] if event['iteration']==iteration else None,deadline=deadline)
        master_results.append(value)
    need(master_results,'No audited original master bound')
    usable=[r['lower_bound_usd'] for r in master_results if r['solver_bound_usable'] and r['lower_bound_usd'] is not None]
    need(usable,'No usable audited native lower bound')
    lower=max(usable)
    need(abs(lower-candidate['lower_bound_usd'])<=1e-4+1e-7*abs(lower),'Candidate global lower bound is not supported by audited native masters')
    best=[r for r in master_results if r['iteration']==iteration]
    need(len(best)==1 and best[0]['plan_sha256']==candidate['plan_sha256'],'Candidate own master proof differs')
    need(candidate['support_ids']==recourse['support_ids'] and
         candidate['native_master_status']==best[0]['native_status'] and
         all(candidate['original_rmp_dual'].get(k)==v for k,v in best[0]['uncertainty_dual'].items() if v is not None),
         'Candidate original master summary differs from independent audit')
    need(np.allclose(candidate['original_rmp_residual_usd'],best[0]['dual_residuals']['residuals_usd'],atol=1e-8,rtol=0)
         and abs(candidate['first_stage_cost_usd']-recourse['first_stage_cost_usd'])<=1e-4,
         'Candidate first cost or original dual residual differs')
    master_ref=_write(output/'masters.json',dict(status='PASS_ORIGINAL_MASTER_EVIDENCE_SET',plan_sha256=candidate['plan_sha256'],
        lower_bound_usd=lower,selected_iteration=iteration,masters=master_results,scientific_acceptance=False,formal_OOS=False))
    # The independently recomputed probability dual bound, not the reported
    # incumbent objective, supplies a valid full-support upper bound.
    upper=recourse['first_stage_cost_usd']+probability['check']['bounds']['certified_upper_bound_usd']
    need(math.isfinite(upper) and lower<=upper+1e-4+1e-7*abs(upper),'Invalid independent bounds')
    need(abs(upper-candidate['upper_bound_usd'])<=1e-4+1e-7*abs(upper),'Candidate UB differs from full-support proof')
    need(time.monotonic()<deadline,'Independent audit exceeded shared budget')
    value=dict(status='PASS_INDEPENDENT_PLANNING_NUMERICAL_AUDIT',passed=True,plan_id=request['plan_id'],
        plan_sha256=candidate['plan_sha256'],request_reference=request_reference,
        candidate_record_reference=candidate_record_reference,full_support_verified=True,
        recourse_audit_reference=recourse_ref,probability_audit_reference=probability_ref,
        master_audit_reference=master_ref,lower_bound_usd=lower,upper_bound_usd=upper,
        gap_relative=max(0.,upper-lower)/max(1.,abs(upper)),native_master_status=best[0]['native_status'],
        raw_engine_status=record_value['raw_engine_status'],audit_elapsed_seconds=time.monotonic()-started,
        scientific_acceptance=False,formal_OOS=False)
    return _write(output/'audit.json',value)
