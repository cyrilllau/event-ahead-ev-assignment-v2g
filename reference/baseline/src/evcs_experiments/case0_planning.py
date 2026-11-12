"""Isolated Case0 full-formulation planning and independently evaluated bounds.

No formal campaign admission or C&CG claims. Caller locks one master backend
before a comparison group; LP recourse/probability use the existing HiGHS path.
The external job supervisor must enforce the absolute wall-clock deadline.
"""
from collections.abc import Mapping
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import save_npz

from .extensive import assemble_extensive, solve_extensive
from .first_stage import assemble_first_stage
from .first_stage_check import check_first_stage
from .recourse import assemble_recourse, solve_fixed_plan
from .recourse_check import check_recourse
from .probability import solve_worst_distribution
from .probability_check import check_probability_solution
from .probability_phase1 import solve_phase1
from .random_streams import canonical_hash


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    if isinstance(value, Mapping): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_json(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def _write(path, value):
    path.write_text(json.dumps(_json(value), indent=2, allow_nan=False)+'\n')


def _same_coefficients(left, right, sparse_keys, array_keys):
    for key in sparse_keys:
        a,b=getattr(left,key),getattr(right,key)
        _need(a.shape==b.shape and (a!=b).nnz==0, 'Original physical coefficient mismatch: '+key)
    for key in array_keys:
        _need(np.array_equal(getattr(left,key),getattr(right,key)), 'Original physical coefficient mismatch: '+key)


def audit_case0_recourse(template, plan, result):
    """Recompute original primal/dual feasibility and strong duality from arrays."""
    _need(result.optimal and result.status=='OPTIMAL' and result.solution is not None,
          'Original fixed-plan recourse is not optimal')
    cert = result.certificate
    vectors = [np.asarray(cert[key], dtype=float) for key in
               ('dual_inequality_marginals', 'dual_equality_marginals',
                'dual_lower_marginals', 'dual_upper_marginals')]
    y, eq, lo, hi = vectors
    primal = template.pack(result.solution)
    bub, beq = template.fixed_rhs(plan)
    _need([v.shape for v in vectors] == [(len(bub),), (len(beq),), primal.shape, primal.shape]
          and all(np.isfinite(v).all() for v in vectors), 'Invalid original dual arrays')
    fl, fu = np.isfinite(template.lower_bounds), np.isfinite(template.upper_bounds)
    maximum = lambda a: float(np.max(a, initial=0.))
    primal_error = max(maximum(template.A_ub@primal-bub), maximum(abs(template.A_eq@primal-beq)),
                       maximum(template.lower_bounds-primal), maximum(primal-template.upper_bounds))
    stationarity = maximum(abs(template.c-template.A_ub.T@y-template.A_eq.T@eq-lo-hi))
    sign_error = max(maximum(y), maximum(-lo), maximum(hi))
    infinite_error = max(maximum(abs(lo[~fl])), maximum(abs(hi[~fu])))
    q = float(template.c@primal)
    dual = float(bub@y+beq@eq+template.lower_bounds[fl]@lo[fl]+template.upper_bounds[fu]@hi[fu])
    tol = 1e-4+1e-7*max(1., abs(q), abs(dual))
    _need(np.isfinite(primal).all() and np.isfinite(q) and np.isfinite(dual)
          and primal_error<=1e-7 and max(stationarity, sign_error, infinite_error)<=1e-8
          and abs(q-dual)<=tol and abs(q-result.objective_usd)<=tol,
          'Independent original recourse primal/dual KKT check failed')
    return dict(passed=True, qop_usd=q, dual_usd=dual, primal_error=primal_error,
                stationarity_error=stationarity, sign_error=sign_error,
                infinite_bound_error=infinite_error, gap_usd=abs(q-dual), tolerance_usd=tol)


def audit_case0_master(model, solved):
    """Check native reported bound/constant and original incumbent coefficients."""
    _need(solved.solution is not None, 'No master incumbent')
    _need(solved.solution.model_sha256==model.model_sha256 and solved.solver_status==solved.raw_solver['status'], 'Master result identity/status mismatch')
    vector = np.asarray(solved.solution.vector)
    _need(vector.shape==model.c.shape and np.isfinite(vector).all(), 'Invalid master vector')
    maximum = lambda a: float(np.max(a, initial=0.))
    activity = model.A@vector
    row = np.maximum(np.maximum(model.row_lower-activity, activity-model.row_upper), 0.)
    col = np.maximum(np.maximum(model.lower_bounds-vector, vector-model.upper_bounds), 0.)
    violations = [maximum(row), maximum(row/np.maximum(1., abs(model.A)@abs(vector))),
                  maximum(col), maximum(col/np.maximum(1., abs(vector))),
                  maximum(abs(vector[model.integrality==1]-np.rint(vector[model.integrality==1])))]
    _need(max(violations)<=1e-8, 'Original extensive primal feasibility failed')
    raw = solved.raw_solver
    zero_check=None
    if 'zero_reduction' in raw:
        from .case0_planning_audit import audit_case0_zero_reduction
        arrays={key:getattr(model,key) for key in ('row_lower','row_upper','lower_bounds',
            'upper_bounds','c','objective_constant_usd')}
        zero_check=audit_case0_zero_reduction(model.A,arrays,_json(raw['zero_reduction']),vector,
            _json(raw['native_interface']) if raw.get('solver')=='gurobipy/Gurobi' else None)
    _need(raw['status'] in (0,1), 'Native status cannot supply a usable bound')
    lb = raw['raw_lower_bound_without_constant_usd']
    rawfun = raw['raw_objective_without_constant_usd']
    _need(lb is not None and rawfun is not None and np.isfinite(lb) and np.isfinite(rawfun),
          'Missing finite native bound/objective')
    lb = float(lb+model.objective_constant_usd)
    witness = float(model.c@vector+model.objective_constant_usd)
    tol = 1e-4+1e-7*max(1., abs(lb), abs(witness))
    _need(solved.lower_bound_usd is not None and abs(lb-solved.lower_bound_usd)<=tol
          and abs(witness-(rawfun+model.objective_constant_usd))<=tol
          and solved.objective_usd is not None and abs(witness-solved.objective_usd)<=tol
          and lb<=witness+tol, 'Native bound/objective constant mismatch')
    result=dict(passed=True, lower_bound_usd=lb, witness_objective_usd=witness,
                native_status=solved.status, native_status_code=raw['status'],
                native_branch_and_bound_proof_independently_replayed=False,
                max_original_primal_violation=max(violations))
    if zero_check is not None:result['zero_reduction_audit']=zero_check
    return result


def run_case0_planning(first, recourses, scenario_ids, probability_set, *, case, tasks, scenes,
                       method_id, output_dir, backend, deadline, configuration_started_at,
                       empirical_probabilities=None, scenario_cost_constants_usd=None,
                       master_seconds=1200., solver_feasibility_tolerance=None,
                       zero_presolve=False, compact_descriptive_metadata=False):
    """Return (unchanged FirstStagePlan, JSON-compatible numerical record).

    Absolute monotonic deadline and configuration_started_at include preparation.
    Total budget <=1800s; reserve 600s from now for independent evaluation. Every
    output directory must be fresh. Only EA_DRO/OR_DRO/SAA/ARO are optimized here.
    The OR caller supplies its independently prepared lead-zero physical model.
    """
    now = time.monotonic()
    _need(backend in ('highs','gurobi'), 'Unknown preselected group backend')
    _need(type(zero_presolve) is bool,'Case0 zero_presolve must be an explicit boolean')
    _need(type(compact_descriptive_metadata) is bool,'Case0 compact metadata must be an explicit boolean')
    _need(solver_feasibility_tolerance is None or (backend=='gurobi'
          and type(solver_feasibility_tolerance) in (int,float)
          and solver_feasibility_tolerance in (1e-9,1e-8)), 'Invalid Case0 solver refinement')
    _need(method_id in ('EA_DRO','OR_DRO','SAA','ARO'), 'Unsupported Case0 planning method')
    _need(all(np.isfinite(v) for v in (deadline, configuration_started_at, master_seconds))
          and configuration_started_at<=now<deadline
          and 0<deadline-configuration_started_at<=1800. and 0<master_seconds<=1200.,
          'Invalid total configuration budget')
    ids, recourses, scenes = tuple(scenario_ids), tuple(recourses), tuple(scenes)
    _need(len(ids)==len(recourses)==len(scenes)>0 and len(set(ids))==len(ids), 'Complete original support required')
    if compact_descriptive_metadata:
        from .compact_recourses import CompactRecourseTemplate
        _need(all(isinstance(t,CompactRecourseTemplate) for t in recourses),
              'Compact each uncached recourse before retaining the planning support')
    risk = 'DRO' if method_id in ('EA_DRO','OR_DRO') else method_id
    if risk=='SAA':
        pi = np.asarray(empirical_probabilities, dtype=float)
        _need(pi.shape==(len(ids),) and np.array_equal(pi,np.full(len(ids),1./len(ids))),
              'Case0 SAA requires original uniform empirical weights')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    def remaining(cap=60.):
        available = deadline-time.monotonic()
        if available<=0: raise TimeoutError('Case0 configuration deadline exhausted')
        return min(cap, available)
    rebuilt_first=assemble_first_stage(case,tasks)
    _need(rebuilt_first.matrix_sha256==first.matrix_sha256, 'First-stage physical source mismatch')
    _same_coefficients(first,rebuilt_first,('A',),('row_lower','row_upper','c','lower_bounds','upper_bounds','integrality'))
    for sid, template, scene in zip(ids, recourses, scenes):
        remaining()
        _need(scene.development is True and scene.split=='planning' and scene.regime=='R1',
              'Case0 planning requires development planning R1 inputs')
        rebuilt=assemble_recourse(case,tasks,scene)
        _need(scene.scenario_id==sid and rebuilt.template_sha256==template.template_sha256,
              'Recourse physical source/support mismatch')
        _same_coefficients(template,rebuilt,('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'),
                           ('b_ub','b_eq','c','lower_bounds','upper_bounds'))
        if compact_descriptive_metadata:del rebuilt
    if risk=='DRO':
        _need(probability_set is not None and tuple(probability_set.support_ids)==ids, 'Original ambiguity axis differs')
        phase = solve_phase1(probability_set,time_limit_seconds=remaining())
        _write(out/'phase1.json',phase.as_dict())
        _need(phase.feasible and phase.basis_verified, 'Ambiguity phase-I failed')
    model = assemble_extensive(first,recourses,risk_model=risk,scenario_ids=ids,
        probability_set=probability_set if risk=='DRO' else None,
        empirical_probabilities=empirical_probabilities if risk=='SAA' else None,
        scenario_cost_constants_usd=scenario_cost_constants_usd)
    _write(out/'model.json',model.as_dict()); save_npz(out/'matrix.npz',model.A)
    np.savez_compressed(out/'model_arrays.npz',c=model.c,row_lower=model.row_lower,row_upper=model.row_upper,
        lower_bounds=model.lower_bounds,upper_bounds=model.upper_bounds,integrality=model.integrality,
        objective_constant_usd=model.objective_constant_usd)
    allowance = min(master_seconds, deadline-time.monotonic()-600.)
    _need(allowance>0, 'Insufficient budget for master plus 600-second evaluation reserve')
    refinement = {} if solver_feasibility_tolerance is None else dict(solver_feasibility_tolerance=solver_feasibility_tolerance)
    solved = solve_extensive(model,time_limit_seconds=allowance,mip_relative_gap=.005,
                             master_backend=backend,optimizer_threads=1,solver_log_path=out/'master.log',
                             **({'zero_presolve':True} if zero_presolve else {}),**refinement)
    _write(out/'master.json',solved.as_dict()); remaining()
    _need(('zero_reduction' in solved.raw_solver)==zero_presolve,
          'Case0 requested exact reduction lacks its native proof or was not requested')
    master = audit_case0_master(model,solved)
    _write(out/'master_audit.json',master)
    if zero_presolve:_write(out/'zero_reduction_audit.json',master['zero_reduction_audit'])
    plan = solved.solution.plan
    np.savez_compressed(out/'plan_and_master_vector.npz',x=plan.x,r=plan.r,vector=solved.solution.vector,task_sha256=plan.task_sha256,config_hash=plan.config_hash)
    firstcheck = check_first_stage(case,tasks,plan)
    _write(out/'first_stage_check.json',firstcheck)
    _need(firstcheck['passed'], 'Independent first-stage physical check failed')
    q, evidence, cost_rows = [], [], []
    for i,(scene,template,witness,constant) in enumerate(zip(scenes,recourses,solved.solution.recourse_solutions,model.scenario_cost_constants_usd)):
        folder=out/f'scenario_{i:03d}';folder.mkdir()
        witnesscheck=check_recourse(case,tasks,scene,plan,witness,check_objective_bound=False)
        _need(witnesscheck['passed'], 'Extensive physical witness failed')
        fixed=solve_fixed_plan(template,plan,time_limit_seconds=remaining())
        _write(folder/'fixed_solver.json',fixed.as_dict());remaining()
        kkt=audit_case0_recourse(template,plan,fixed)
        physical=check_recourse(case,tasks,scene,plan,fixed.solution,check_objective_bound=True)
        _need(physical['passed'], 'Independent fixed-plan physical check failed')
        value=float(physical['costs']['Qop']+constant)
        tol=1e-4+1e-7*max(1.,abs(value),abs(kkt['qop_usd']+constant))
        _need(abs(value-(kkt['qop_usd']+constant))<=tol
              and value<=witnesscheck['costs']['Qop']+constant+tol, 'Independent cost/witness mismatch')
        np.savez_compressed(folder/'primal.npz',fixed=template.pack(fixed.solution),witness=template.pack(witness))
        _write(folder/'checks.json',dict(kkt=kkt,physical=physical,witness=witnesscheck))
        q.append(value);evidence.append(kkt);cost_rows.append(dict(scenario_id=ids[i],**physical['costs'],scenario_constant_usd=float(constant)))
    q=np.asarray(q)
    probability=None
    if risk=='DRO':
        probability=solve_worst_distribution(probability_set,q,q_support_ids=ids,time_limit_seconds=remaining())
        checked=check_probability_solution(probability_set,q,probability,q_support_ids=ids)
        _write(out/'probability.json',probability.as_dict());_write(out/'probability_check.json',checked.as_dict())
        _need(probability.certified_optimal and checked.passed, 'Independent probability optimality failed')
        riskvalue=float(probability.dual_objective_usd)
    elif risk=='ARO': riskvalue=float(q.max())
    else: riskvalue=float(np.asarray(empirical_probabilities)@q)
    ub=float(firstcheck['costs']['Fpre_usd']+riskvalue);lb=master['lower_bound_usd']
    tol=1e-4+1e-7*max(1.,abs(lb),abs(ub),abs(master['witness_objective_usd']))
    _need(lb<=ub+tol and ub<=master['witness_objective_usd']+tol, 'Native LB / independently evaluated UB / witness order failed')
    remaining(); gap=max(0.,ub-lb)/max(1.,abs(ub))
    record=dict(kind='CASE0_FULL_FORMULATION_NUMERICAL_RESULT_V1',method_id=method_id,backend=backend,
        risk_model=risk,model_sha256=model.model_sha256,scenario_ids=list(ids),
        plan=dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash),
        plan_sha256=canonical_hash(dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash)),
        q_usd=q.tolist(),cost_components=cost_rows,Fpre_usd=firstcheck['costs']['Fpre_usd'],
        risk_value_usd=riskvalue,lower_bound_usd=lb,upper_bound_usd=ub,gap=gap,
        numerical_valid=True,gap_passed=bool(gap<=.005),native_status=solved.status,
        native_status_code=solved.solver_status,master_wall_seconds=solved.wall_seconds,
        configuration_wall_seconds=time.monotonic()-configuration_started_at,
        extensive_witness_costs_are_optimal_q=False,full_support_independently_evaluated=True,
        formal_acceptance=False,Case1_acceptance=False,C_and_CG_claim=False,formal_OOS=False)
    if compact_descriptive_metadata:
        record['descriptive_metadata_representation']='lossless_categorical_rows_v1'
        record['descriptive_metadata_code_bytes']=sum(getattr(t,n).code_bytes for t in recourses
            for n in ('variable_map','constraint_map_ub','constraint_map_eq'))
    if solver_feasibility_tolerance is not None:
        record['solver_refinement']=dict(solver_feasibility_tolerance=solver_feasibility_tolerance,
            original_acceptance_tolerance=1e-8,acceptance_threshold_changed=False)
    if zero_presolve:
        record['zero_presolve']=True
        record['zero_reduction_proof_sha256']=master['zero_reduction_audit']['proof_sha256']
        record['zero_reduction_checker_reference']=master['zero_reduction_audit']['checker_source_reference']
    _write(out/'result.json',record)
    return plan,record
