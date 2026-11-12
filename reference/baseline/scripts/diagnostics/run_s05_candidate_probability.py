"""Prepared, source-bound probability completion; no standalone formal execution.

A reviewed supervised caller must authorize/settle the parent budget. This
component checks saved recourse records, not an independent re-audit of the
original coefficient matrices. Its UB remains pending that scientific review.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from evcs_experiments import probability,probability_check
from evcs_experiments.random_streams import canonical_hash

CORRECTED_PROBABILITY_SHA='39a33c1dab694a65113eeb9424c2eaf7e056d4e3478ec4d0f3c84b454754d68c'


def need(ok,message):
    if not bool(ok):raise ValueError(message)


def reference(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4194304):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def verify(r):
    need(set(r)=={'path','sha256','size_bytes'} and Path(r['path']).is_absolute() and reference(r['path'])==r,'Changed strong reference')
    return Path(r['path'])


def read(r):return json.loads(verify(r).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return reference(path)


def artifact(row,name):
    matches=[r for r in row['artifacts'] if Path(r['path']).name==name]
    need(len(matches)==1,'Missing unique artifact: '+name);return matches[0]


def verify_inputs(refs):
    required={'full_q','checkpoint','first_stage_check','evaluation_result','execution','new_raw_request',
        'original_checkpoint','original_history','original_probability_input','saved160_acceptance',
        'original_binding','candidate_result','candidate_review','probability_source','checker_source','driver_source'}
    need(set(refs)==required,'Unexpected probability completion reference scope')
    for r in refs.values():verify(r)
    need(refs['probability_source']==reference(probability.__file__) and refs['probability_source']['sha256']==CORRECTED_PROBABILITY_SHA
         and refs['checker_source']==reference(probability_check.__file__) and refs['driver_source']==reference(__file__),
         'Wrong corrected probability/checker/driver source; old R4 is forbidden')
    full=read(refs['full_q']);need(full==read(refs['checkpoint']),'Full Q/checkpoint differ')
    need(full['status']=='FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY' and full['completed_count']==full['next_index']==160
         and full['error'] is None and full['formal_OOS'] is False and full['probability_evaluated'] is False,'Incomplete fresh full Q')
    plan=read(full['plan_reference']);ph=canonical_hash(plan);need(full['plan_sha256']==ph,'Plan content hash differs')
    adapted=read(full['candidate_binding_reference']);binding=read(refs['original_binding'])
    need(adapted['original_binding_reference']==refs['original_binding'] and all(adapted[k]==v for k,v in binding.items()),'Candidate adapter changed original binding')
    need(adapted['result_reference']==binding['source_native_result'] and adapted['candidate_result_reference']==refs['candidate_result']
         and adapted['candidate_review_reference']==refs['candidate_review'],'Candidate adapter/native source differs')
    native=read(binding['source_native_result']);verify(binding['source_request'])
    need(native['valid_incumbent'] is True and native['certificate_failures']==[],'Invalid native incumbent')
    candidate=read(refs['candidate_result']);review=read(refs['candidate_review'])
    need(review['status']=='PASS_INDEPENDENT_EPIGRAPH_AUXILIARY_RESULT_REVIEW' and review['passed'] is True
         and review['references']['result']==refs['candidate_result'] and review['references']['candidate_binding']==refs['original_binding']
         and review['references']['native_solver']==binding['source_native_result'],'Unreviewed candidate provenance')
    need(candidate['candidate_available'] is True and candidate['candidate_vector']==binding['vector_reference']
         and candidate['request_reference']==binding['source_request'] and binding['plan_sha256']==candidate['candidate_plan_sha256']==review['candidate_plan_sha256']==ph,
         'Native candidate plan differs')
    x=np.asarray(plan['x'],float);r=np.asarray(plan['r'],float);need(x.ndim==1 and r.ndim==2 and r.shape[0]==len(x),'Invalid plan shape')
    z=np.r_[x,r.ravel()];need(np.isfinite(z).all() and binding['first_stage_slice']==[0,len(z)],'Candidate slice differs')
    with np.load(verify(binding['vector_reference']),allow_pickle=False) as a:
        v=np.asarray(a[binding['vector_key']]);need(v.ndim==1 and len(v)>=len(z) and np.array_equal(v[:len(z)],z),'Native vector/plan differs; no rounding')
    need(Path(refs['first_stage_check']['path'])==Path(refs['full_q']['path']).parent/'first_stage_check.json','First-stage check is outside the fresh evaluation')
    first=read(refs['first_stage_check']);need(first['passed'] is True and first['task_sha256']==plan['task_sha256'] and first['config_hash']==plan['config_hash'],'First-stage physical check differs')
    fpre=float(candidate['candidate_first_cost_usd']);physical_cost=float(first['costs']['Fpre_usd'])
    # Matrix dot product and physical component sums may differ in last bits.
    # Keep the native original-matrix cost, without changing either record.
    need(math.isfinite(fpre) and math.isfinite(physical_cost) and abs(fpre-physical_cost)<=1e-7,'First-stage original cost differs')
    cp=read(refs['original_checkpoint']);state=cp['state'];identity=state['identity'];ps=canonical_hash(identity)
    need(cp['sha256']==canonical_hash(state) and state['problem_sha256']==ps==binding['problem_sha256']==candidate['problem_sha256']==review['original_problem_sha256']
         and binding['first_stage_sha256']==candidate['first_stage_sha256']==identity['first_stage_sha256'],'Original problem/source identity differs')
    need(Path(refs['new_raw_request']['path'])==Path(full['raw_history_reference']['path']).parent/'request.json','New raw request/history root differs')
    raw=read(refs['new_raw_request']);need(raw['problem_identity']==identity and raw['domain']=='formal' and raw['formal_OOS'] is False,'New evaluation used different original problem')
    card=read(refs['saved160_acceptance']);need(card['status']=='PASS_SAVED160_COMPONENT_REVIEW' and card['complete_lp_count']==160 and card['problem_sha256']==ps
         and refs['original_checkpoint'] in card['references'].values() and refs['original_history'] in card['references'].values(),'Original evidence acceptance differs')
    reuse=full['template_reuse_inventory'];need(reuse['references']['old_checkpoint']==refs['original_checkpoint'] and reuse['references']['old_history']==refs['original_history']
         and reuse['references']['saved160_acceptance']==refs['saved160_acceptance'] and reuse['old_q_reused'] is False and len(reuse['templates'])==160,'Template reuse provenance differs')
    history=read(refs['original_history']);need(len(history)==163,'Original history scope differs')
    oldprob=read(history[-1]);need(oldprob['event']=='probability' and artifact(oldprob,'probability_input.json')==refs['original_probability_input'],'Original P provenance differs')
    pinput=read(refs['original_probability_input']);a=pinput['effective_ambiguity'];original=pinput['input_ambiguity']
    P=probability.ProbabilitySet(tuple(a['support_ids']),tuple(a['feature_ids']),np.asarray(a['features']),np.asarray(a['lower']),np.asarray(a['upper']))
    need(P.ambiguity_sha256==a['ambiguity_sha256']==identity['effective_ambiguity_sha256'] and original['ambiguity_sha256']==identity['input_ambiguity_sha256'] and identity['risk_model']=='DRO','Original ambiguity differs')
    ids=list(P.support_ids);q=np.asarray(full['q_usd'],float);constants=np.asarray(identity['constants_usd'],float)
    need(len(ids)==160 and ids==identity['scenario_ids']==full['support_ids'] and q.shape==constants.shape==(160,) and np.isfinite(q).all() and np.isfinite(constants).all(),'Full Q/support/constants differ')
    events=full['recourse_event_references'];need(len(events)==160 and len({v['path'] for v in events})==160 and read(full['raw_history_reference'])==events,'New recourse history differs')
    for i,er in enumerate(events):
        event=read(er);need(event['event']=='recourse' and event['domain']=='formal' and event['scenario_index']==i and event['scenario_id']==ids[i]
            and event['plan_sha256']==ph and event['q_usd']==q[i] and event['optimal_certificate_valid'] is True,'New recourse event binding differs')
        for ar in event['artifacts']:verify(ar)
        need(read(artifact(event,'plan.json'))==plan,'New LP used different plan')
        solver=read(artifact(event,'solver.json'));cert=solver['certificate'];physical=read(artifact(event,'physical_check.json'))
        need(solver['optimal'] is True and solver['has_solution'] is True and cert['optimal_certificate_valid'] is True and cert['primal_feasible_within_tolerance'] is True
             and physical['passed'] is True,'New physical/KKT certificate failed')
        need(cert['certificate_status']=='PASS_NUMERIC_LP_CERTIFICATE' and cert['optimal_cost_bound_valid'] is True,'Missing original numeric certificate gate')
        for key,tol in [('max_inequality_violation',1e-7),('max_equality_violation',1e-7),('max_bound_violation',1e-7),('max_dual_stationarity_violation',1e-8),('max_dual_sign_violation',1e-8),('max_invalid_infinite_bound_marginal',1e-8)]:
            need(math.isfinite(float(cert[key])) and 0<=float(cert[key])<=tol,'Raw original-unit certificate residual failed')
        gap=abs(float(solver['objective_usd'])-float(cert['dual_objective_usd']));tol=1e-4+1e-7*max(abs(float(solver['objective_usd'])),abs(float(cert['dual_objective_usd'])),1.)
        need(math.isfinite(gap) and gap<=tol,'Original raw duality gap failed')
        need(float(solver['objective_usd'])+constants[i]==q[i],'Q must include original scenario constant exactly once')
        tr=read(artifact(event,'template_reference.json'));row=read(tr);receipt=reuse['templates'][i]
        need(receipt['index']==i and receipt['scenario_id']==ids[i] and receipt['all_original_artifacts_hashed'] is True and receipt['template_reference']==tr
             and row['scenario_id']==ids[i] and row['scenario_index']==i and row['template_sha256']==cert['template_sha256']==state['pending']['recourse_certificates'][i]['template_sha256'],'New certificate/original template differs')
        for name in ('fixed_rhs.npz','primal_vector.npz','solution.npz','dual_kkt.npz'):artifact(event,name)
    execution=read(refs['execution']);result=read(refs['evaluation_result'])
    need(result['checkpoint_reference']==refs['checkpoint'] and result['completed_count']==160 and result['original_candidate_binding']==refs['original_binding']
         and result['plan_sha256']==ph and result['problem_sha256']==ps and execution['plan_sha256']==ph and execution['original_problem_sha256']==ps
         and execution['candidate_binding_reference']==full['candidate_binding_reference'],'Actual evaluation execution/result differs')
    need(execution['source_references']['old_checkpoint']==refs['original_checkpoint'] and execution['source_references']['candidate_binding']==refs['original_binding']
         and execution['source_references']['saved160_acceptance']==refs['saved160_acceptance'] and execution['request_reference']==result['request_reference']
         and execution['probability_calls']==execution['master_calls']==0 and execution['formal_OOS'] is False and result['formal_OOS'] is False,'Evaluation source/request/scope differs')
    verify(result['request_reference'])
    return P,q,fpre,ph,ps


def complete_candidate_probability(refs,*,output,time_limit_seconds=30):
    """Parent-owned bounded call; never starts formal jobs or mutates prior proof."""
    need(type(time_limit_seconds) in (int,float) and math.isfinite(time_limit_seconds) and 0<time_limit_seconds<=30,'Invalid probability time cap')
    P,q,fpre,ph,ps=verify_inputs(refs)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'input_references.json',refs)
    solved=probability.solve_worst_distribution(P,q,q_support_ids=P.support_ids,time_limit_seconds=time_limit_seconds)
    solverref=write(out/'probability_result.json',solved.as_dict())
    checked=probability_check.check_probability_solution(P,q,solved,q_support_ids=P.support_ids)
    checkref=write(out/'probability_check.json',checked.as_dict())
    passed=checked.passed and solved.certified_optimal and solved.upper_bound_usd is not None
    upper=float(fpre+solved.upper_bound_usd) if passed else None
    need(upper is None or math.isfinite(upper),'Nonfinite original-unit upper bound')
    result=dict(kind='S05_NEW_CANDIDATE_PROBABILITY_COMPONENT_V1',status='NUMERICAL_PROBABILITY_PASS_PENDING_RECOURSE_AUDIT' if passed else 'PROBABILITY_FAILED',
        references=refs,probability_result=solverref,probability_check=checkref,probability_independent_check_passed=bool(checked.passed),
        original_problem_sha256=ps,plan_sha256=ph,q_sha256=solved.q_sha256,ambiguity_sha256=P.ambiguity_sha256,complete_lp_count=160,
        first_stage_cost_usd=fpre,worst_recourse_upper_bound_usd=solved.upper_bound_usd if passed else None,upper_bound_usd=upper,lower_bound_usd=None,
        scenario_constants_scope='Each original scenario constant is already included once in q; only Fpre is added after probability maximization.',
        original_matrix_independent_reaudit_performed=False,full_support_verified=False,scientific_acceptance=False,lockable=False,formal_OOS=False,
        normal_ccg_checkpoint_compatible=False,parent_budget_settlement_required=True)
    write(out/'result.json',result);return result


if __name__=='__main__':
    raise SystemExit('Prepared component only: a separately reviewed supervised parent and cumulative-budget settlement are required.')
