"""Append-only plan records for independently completed auxiliary candidates.

No solver, C&CG checkpoint mutation, paired-group lock or OOS access is called.
TIME_LIMIT describes the original RMP; the auxiliary bound has its own origin.
"""
from decimal import Decimal
from pathlib import Path
import json,math
from .evaluation_lock import file_reference,_verify_reference,verify_formal_numerical_records
from .planning_budget import validate_budget
from .random_streams import canonical_hash
from .config import get_paths

REASON='RMP_TIME_LIMIT_WITH_COMPLETED_EVIDENCE_AND_AUXILIARY_BOUND'
REQUIRED=('protocol','completion_result','completion_review','settlement','recourse_audit','probability_component',
          'auxiliary_result','auxiliary_review','previous_zero_probability_acceptance','original_checkpoint',
          'original_raw_request','candidate_plan','adapter_decision')

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def read(ref):_verify_reference(ref);return json.loads(Path(ref['path']).read_text())

def normal_reference(ref):
    """Preserve legacy bytes bindings while emitting canonical size_bytes refs."""
    size=ref.get('size_bytes',ref.get('bytes'))
    answer=dict(path=ref['path'],sha256=ref['sha256'],size_bytes=size)
    _verify_reference(answer);return answer

def new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return file_reference(path)

def finite(value,label):
    need(type(value) in (int,float) and math.isfinite(value),'Invalid '+label);return float(value)

def build_records(request,documents):
    """Validate already byte-verified documents and build the numeric payloads.

The exported prepare_formal_result additionally verifies every strong reference.
This pure function enables meaningful numerical/status/provenance hand tests.
"""
    refs=request['references'];identity=request['identity'];D=documents
    envelope=D['protocol'];protocol=envelope['payload']
    need(envelope['kind']=='protocol_lock' and envelope['payload_sha256']==canonical_hash(protocol) and protocol['domain']=='formal','Invalid frozen formal protocol')
    matches=[p for b in protocol['batches'] if b['batch_id']==request['batch_id'] for p in b['expected_plans'] if p==identity]
    need(len(matches)==1 and set(identity)=={'plan_id','target_id','method_id','setting_id'},'Plan identity is not predeclared')
    need(identity['target_id']=='michael_2018' and identity['method_id']=='EA_DRO','Unsupported bridge method/target')
    pb=validate_budget(protocol['configuration']['planning_budget']);cap=Decimal(str(pb['cumulative_wall_seconds']))
    cp=D['original_checkpoint'];state=cp['state'];problem=canonical_hash(state['identity']);raw=D['original_raw_request']
    need(cp['sha256']==canonical_hash(state) and state['problem_sha256']==problem and raw['problem_identity']==state['identity'] and raw['domain']=='formal','Original scientific problem changed')
    rmp=state['pending']['rmp'];need(rmp['status']=='TIME_LIMIT' and rmp['optimal'] is False and rmp['lower_bound_usd'] is None,'Wrong original native RMP condition')
    plan=D['candidate_plan'];need(set(plan)=={'x','r','task_sha256','config_hash'},'Candidate plan metadata changed')
    ph=canonical_hash(plan);review=D['completion_review'];completion=D['completion_result'];audit=D['recourse_audit'];prob=D['probability_component'];aux=D['auxiliary_result'];ar=D['auxiliary_review'];zero=D['previous_zero_probability_acceptance'];settled=D['settlement']
    need(review['status']=='PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW' and review['passed'] is True and review['scientific_candidate_complete'] is True,'Candidate lacks independent completion review')
    for name in ('completion_result','settlement','recourse_audit','probability_component','candidate_plan','original_checkpoint','auxiliary_result','auxiliary_review','previous_zero_probability_acceptance'):
        need(review['references'][name]==refs[name],'Independent completion review refers to different '+name)
    need(all(x['original_problem_sha256']==problem for x in (review,completion,audit,prob,ar)) and aux['problem_sha256']==problem and zero['problem_sha256']==problem,'Bounds or evidence belong to another problem')
    need(all(x['plan_sha256']==ph for x in (review,completion,audit,prob)) and aux['candidate_plan_sha256']==ar['candidate_plan_sha256']==ph,'Candidate evidence belongs to another plan')
    need(completion['status']=='FULL_SUPPORT_NUMERICAL_COMPLETION_PENDING_FORMAL_PLAN_LOCK' and completion['full_support_verified'] is True
         and completion['recourse_independent_check_passed'] is True and completion['probability_independent_check_passed'] is True and completion['complete_lp_count']==160,'Incomplete original-support evidence')
    need(completion['recourse_audit']==refs['recourse_audit'] and completion['probability_result']==refs['probability_component'],'Completion evidence references differ')
    need(audit['status']=='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT' and audit['passed'] is True and audit['complete_lp_count']==160
         and audit['support_ids']==state['identity']['scenario_ids'] and len(audit['q_usd'])==160,'Full160 audited support required')
    need(prob['probability_independent_check_passed'] is True and prob['complete_lp_count']==160,'Probability certificate missing')
    need(ar['status']=='PASS_INDEPENDENT_EPIGRAPH_AUXILIARY_RESULT_REVIEW' and ar['passed'] is True and ar['references']['result']==refs['auxiliary_result'],'Global auxiliary bound unreviewed')
    need(zero['status']=='PASS_PROBABILITY_RECOVERY_COMPONENT_REVIEW' and zero['probability_certificate_PASS'] is True,'Previous zero plan lacks a certified comparison')
    lb=finite(aux['lower_bound_usd'],'auxiliary LB');ub=finite(prob['upper_bound_usd'],'candidate UB');oldub=finite(zero['upper_bound_usd'],'previous zero UB')
    need(lb==ar['numerical_lower_bound_usd']==review['lower_bound_usd'] and ub==review['upper_bound_usd']==completion['upper_bound_usd'],'Accepted bounds disagree')
    need(oldub==review['previous_zero_upper_bound_usd'] and ub<=oldub+1e-4 and lb<=ub+1e-4,'Candidate is worse than accepted zero plan or bounds reversed')
    Fpre=finite(audit['Fpre_usd'],'Fpre');need(Fpre==prob['first_stage_cost_usd'] and abs(Fpre+prob['worst_recourse_upper_bound_usd']-ub)<=1e-8,'First-stage/scenario constants counted incorrectly')
    need(settled['kind']=='S05_CANDIDATE_COMPLETION_PARENT_ACCOUNTING_V1' and settled['status']=='SETTLED_NUMERICAL_CANDIDATE_PENDING_FORMAL_PLAN_LOCK'
         and settled['accounting_PASS'] is True and settled['parent_charged_once'] is True and settled['cumulative_seconds_provisional'] is False and settled['nested_seconds_added'] is False,'Cumulative accounting not settled')
    elapsed=Decimal(settled['cumulative_accounted_seconds_decimal']);remaining=cap-elapsed
    need(Decimal(settled['opening_cumulative_seconds_decimal'])+Decimal(settled['terminal_parent_seconds_decimal'])==elapsed
         and settled['references']['completion_result']==refs['completion_result']
         and settled['cumulative_cap_seconds']==settled['original_cumulative_cap_seconds']==pb['cumulative_wall_seconds'],'Parent accounting or cap differs')
    need(elapsed.is_finite() and 0<=elapsed<=cap and str(remaining)==settled['remaining_seconds_decimal'] and float(elapsed)==settled['cumulative_accounted_seconds'] and float(remaining)==settled['remaining_seconds'],'Exact budget totals differ')
    need(settled['plan_id']==identity['plan_id'],'Settlement plan differs')
    iteration_reserve=Decimal(str(pb['single_master_seconds']))+Decimal(str(pb['final_certificate_reserve_seconds']))+Decimal(str(pb['shutdown_reserve_seconds']))+Decimal(10)
    need(0<remaining<=iteration_reserve,'Another complete iteration fits or cap was exhausted; this stop reason does not apply')
    decision=D['adapter_decision'];need(decision['status']=='APPROVED_FORMAL_RESULT_BRIDGE' and decision['protocol_reference']==refs['protocol']
         and decision['original_problem_sha256']==problem and decision['candidate_plan_sha256']==ph and decision['auxiliary_bound_allowed'] is True
         and decision['ccg_convergence_claimed'] is False and decision['formal_OOS'] is False,'Unreviewed or incompatible implementation decision')
    for name in ('completion_review','settlement','auxiliary_review'):need(decision['references'][name]==refs[name],'Decision evidence differs')
    need(decision['planning_source']==request['planning_source'],'Decision planning source differs')
    gap=max(0.,ub-lb)/max(1.,abs(ub));threshold=float(pb['normal_gap_relative'])
    numeric=dict(schema_version=1,kind='FORMAL_AUXILIARY_CANDIDATE_NUMERICAL_RECORD_V1',domain='formal',status='TIME_LIMIT',stop_reason=REASON,
        **identity,plan_sha256=ph,original_problem_sha256=problem,full_support_verified=True,support_count=160,lower_bound_usd=lb,upper_bound_usd=ub,gap_relative=gap,
        lower_bound_origin='independently reviewed native global bound of full-support dual-cut/floor auxiliary master',upper_bound_origin='same-plan original160 recourse audit and certified worst-case probability',
        original_native_master_status='TIME_LIMIT',original_native_master_lower_bound_usd=None,original_engine_status=state['status'],original_native_checkpoint_unchanged=True,
        auxiliary_master_status=ar['native_status'],full_CCG_converged=False,normal_gap_met=gap<=threshold,normal_gap_relative=threshold,normal_ccg_checkpoint_compatible=False,
        cumulative_wall_seconds=float(elapsed),cumulative_wall_seconds_decimal=str(elapsed),remaining_wall_seconds_decimal=str(remaining),budget_exhausted=False,cannot_start_another_complete_iteration=True,
        complete_iteration_reserve_seconds_decimal=str(iteration_reserve),worker_seal_reserve_seconds=10,first_stage_cost_usd=Fpre,previous_zero_upper_bound_usd=oldub,
        candidate_improvement_over_zero_usd=oldub-ub,best_among_compared_certified_candidates=True,references=refs,scientific_acceptance=False,formal_OOS=False,paired_group_locked=False)
    budget=dict(schema_version=1,planning_budget=pb,cumulative_accounted_seconds_decimal=str(elapsed),remaining_seconds_decimal=str(remaining),accounting_reference=refs['settlement'],
        actual_wall_charged_once=True,nested_timers_added=False,budget_exhausted=False,cannot_start_another_complete_iteration=True,stop_reason=REASON)
    return numeric,budget

def prepare_formal_result(request,*,output):
    """Write plan, numeric, budget and plan-record files exclusively; never lock."""
    need(all(k in request['references'] for k in REQUIRED),'Missing bridge reference')
    docs={k:read(v) for k,v in request['references'].items()}
    for r in request['planning_source']:_verify_reference(r)
    need(file_reference(__file__) in request['planning_source'],'Bridge source must be explicitly bound')
    numeric,budget=build_records(request,docs)
    inputs=request['plan_input_references'];need(set(inputs)=={'tasks','routes','initial_state','planning_support'},'Incomplete plan input provenance')
    binding=docs['original_raw_request']['input_binding'];taskref=normal_reference(binding['task_source']);supportref=normal_reference(binding['target']['planning_index'])
    for k,r in inputs.items():_verify_reference(r);need(r==(supportref if k=='planning_support' else taskref),'Plan input differs from original frozen '+k)
    tasks=read(taskref);need(tasks['task_sha256']==docs['candidate_plan']['task_sha256'] and all(k in tasks for k in ('tasks','queries','vehicle_summaries')),'Task source does not retain route/state provenance')
    out=Path(output).resolve()
    protected=[Path(get_paths().data_root).resolve(),Path(get_paths().frozen_source_root).resolve()]
    protected += [Path(request['references'][key]['path']).resolve().parent for key in ('original_raw_request','original_checkpoint','candidate_plan')]
    protected += [Path(ref['path']).resolve().parent for ref in request['planning_source']]
    need(not any(out.is_relative_to(p) for p in protected),'Output overlaps frozen evidence or planning source')
    need(not any(Path(r['path']).resolve().is_relative_to(out) for r in request['references'].values()),'Output contains immutable input files')
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(docs['candidate_plan'],execution_mode='EA',departure_h=1)
    pr=new(out/'plan.json',plan);nr=new(out/'solver_record.json',numeric);br=new(out/'planning_budget.json',budget)
    record=dict(**request['identity'],plan_sha256=numeric['plan_sha256'],references=dict(plan=pr,solver_record=nr,budget=br,**inputs,planning_source=request['planning_source']))
    # The existing numerical gate is exercised directly; no batch/access API is called.
    verify_formal_numerical_records(docs['protocol']['payload'],{'plans':[record]})
    rr=new(out/'plan_record.json',record)
    manifest=new(out/'artifact_manifest.json',dict(files=[file_reference(p) for p in sorted(out.iterdir()) if p.is_file()]))
    return dict(status='FORMAL_PLAN_RECORD_PREPARED_PENDING_PAIRED_GROUP',plan_record_reference=rr,solver_record_reference=nr,budget_reference=br,plan_reference=pr,artifact_manifest_reference=manifest,
        plan_sha256=numeric['plan_sha256'],lower_bound_usd=numeric['lower_bound_usd'],upper_bound_usd=numeric['upper_bound_usd'],full_support_verified=True,
        formal_plan_locked=False,paired_group_locked=False,formal_OOS=False,scientific_acceptance=False,solver_calls=0)
