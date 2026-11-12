"""One bounded SAA master solve with sequential original-physics witness checks.

An extensive incumbent is not an independently minimized full-support Q record.
No queue acceptance, budget reset, OOS permission or fallback backend is granted.
"""
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .evaluation_lock import file_reference
from .first_stage_check import check_first_stage
from .random_streams import canonical_hash
from .recourse_check import check_recourse
from .saa_streaming import StreamingSAA, _deadline, _need, _write


def _finite(raw, name):
    value=getattr(raw,name,None)
    return None if value is None or not np.isfinite(value) else float(value)


def _json(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_json(v) for v in value]
    return value


def solve_streaming_saa(model, recourses, *, case, tasks, scene_loader, output_directory,
                        backend, time_limit_seconds, deadline, evidence_reserve_seconds,
                        optimizer_threads=1, mip_relative_gap=.001):
    """Use explicit Gurobi for formal and explicit SciPy HiGHS for development.

    The shared absolute monotonic deadline covers import, solve, witness export
    and checks; a positive caller-selected evidence reserve is withheld from the
    solver grant. The native adapter also charges its own model import time.
    Full formal source review/resource admission remains the caller's gate.
    """
    began=time.monotonic(); _deadline(deadline)
    _need(isinstance(model,StreamingSAA),'Expected disk-backed SAA model')
    _need(backend in ('gurobi','highs'),'Explicit supported SAA backend required')
    _need(all(not isinstance(v,(bool,np.bool_)) and np.isfinite(v) and v>0
              for v in (deadline,time_limit_seconds,evidence_reserve_seconds)), 'Finite positive time controls required')
    _need(time_limit_seconds<=600,'Single SAA master cannot exceed frozen 600 seconds')
    _need(type(optimizer_threads) is int and 1<=optimizer_threads<=4,'Threads must be 1..4')
    _need(not isinstance(mip_relative_gap,(bool,np.bool_)) and np.isfinite(mip_relative_gap)
          and 0<=mip_relative_gap<=.005,'SAA gap cannot exceed frozen 0.5 percent')
    formal=model.manifest['domain']=='formal'
    _need(not formal or backend=='gurobi' and optimizer_threads==4,'Formal SAA requires explicit Gurobi with four master threads')
    _need(not formal or mip_relative_gap==.001,'Formal SAA master MIPGap must remain frozen 0.001')
    _need(backend!='highs' or optimizer_threads==1,'Development HiGHS uses one thread')
    _need(len(recourses)==len(model.manifest['scenario_ids']),'Full original SAA support required')
    _need(all(row['empirical_probability']==model.manifest['empirical_probabilities'][i]
              and row['scenario_id']==model.manifest['scenario_ids'][i]
              and row['scenario_sha256']==model.manifest['scenario_sha256s'][i]
              for i,row in enumerate(model.manifest['scenarios'])),'SAA repeated support/weight metadata differs')
    body=dict(model.manifest); digest=body.pop('manifest_sha256')
    _need(canonical_hash(body)==digest and json.loads((model.root/'manifest.json').read_text())==model.manifest,
          'SAA manifest changed')
    for ref in model.manifest['numeric_artifacts']:
        _deadline(deadline); _need(file_reference(ref['path'])==ref,'SAA numeric source changed')
    root=Path(output_directory).resolve(); root.mkdir(parents=True,exist_ok=False)
    grant=min(float(time_limit_seconds),deadline-time.monotonic()-evidence_reserve_seconds)
    _need(grant>0,'No shared deadline remains after SAA evidence reserve')
    options=dict(time_limit=grant,mip_rel_gap=float(mip_relative_gap),threads=optimizer_threads,
                 primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8)
    solve_began=time.monotonic()
    if backend=='gurobi':
        from .native_gurobi import solve_milp
        raw=solve_milp(model.c,model.integrality,model.lower_bounds,model.upper_bounds,model.A,
                       model.row_lower,model.row_upper,options=options,start=None,log_path=root/'native_solver.log')
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=RuntimeWarning,message='Unrecognized options detected.*')
            raw=milp(model.c,integrality=model.integrality,bounds=Bounds(model.lower_bounds,model.upper_bounds),
                constraints=LinearConstraint(model.A,model.row_lower,model.row_upper),options=options)
    solve_seconds=time.monotonic()-solve_began
    native=dict(backend=backend,status=int(raw.status),message=str(raw.message),
        raw_objective_without_constant_usd=_finite(raw,'fun'),raw_lower_bound_without_constant_usd=_finite(raw,'mip_dual_bound'),
        raw_mip_gap=_finite(raw,'mip_gap'),options=options,
        native_interface=_json(getattr(raw,'native_record',None)))
    _write(root/'native_result.json',native)
    vector=getattr(raw,'x',None)
    if vector is not None: np.save(root/'primal.npy',np.asarray(vector),allow_pickle=False)
    _deadline(deadline)
    raw_bound=native['raw_lower_bound_without_constant_usd']
    lower=None if raw_bound is None or raw.status not in (0,1) else raw_bound+model.objective_constant_usd
    objective=None; checks=[]; feasible=False; plan=None
    if vector is not None:
        vector=np.asarray(vector,dtype=float); plan=model.plan(vector)
        activity=model.A@vector
        maxima=dict(rows=float(np.max(np.maximum(model.row_lower-activity,activity-model.row_upper),initial=0)),
            bounds=float(np.max(np.maximum(model.lower_bounds-vector,vector-model.upper_bounds),initial=0)),
            integer=float(np.max(abs(vector[model.integrality==1]-np.rint(vector[model.integrality==1])),initial=0)))
        _need(all(value<=1e-8 for value in maxima.values()),'SAA reduced incumbent violates original tolerance')
        check=check_first_stage(case,tasks,plan); _need(check['passed'],'SAA original first-stage physics failed')
        _write(root/'first_stage_check.json',check)
        full_objective=float(model.first_stage.c@vector[:len(model.first_stage.c)])
        for i,row in enumerate(model.manifest['scenarios']):
            _deadline(deadline)
            template=recourses[i]; scene=scene_loader(i)
            _need(template.template_sha256==row['template_sha256'] and template.scenario_sha256==row['scenario_sha256']
                  and scene.scenario_id==row['scenario_id'] and scene.scenario_sha256==row['scenario_sha256']
                  and scene.development is (not formal),'SAA witness original template/scene identity differs')
            y=model.original_recourse_vector(i,vector); solution=template.unpack(y)
            bub,beq=template.fixed_rhs(plan)
            errors=[np.max(template.A_ub@y-bub,initial=0),np.max(abs(template.A_eq@y-beq),initial=0),
                    np.max(template.lower_bounds-y,initial=0),np.max(y-template.upper_bounds,initial=0)]
            _need(all(value<=1e-8 for value in errors),'SAA expanded original coefficient witness failed')
            check=check_recourse(case,tasks,scene,plan,solution,check_objective_bound=False)
            _need(check['passed'],'SAA original recourse physics failed')
            cost=float(template.c@y+model.manifest['scenario_cost_constants_usd'][i])
            full_objective+=row['empirical_probability']*cost
            _write(root/f'witness_{i:03d}.json',dict(scenario_id=row['scenario_id'],template_sha256=row['template_sha256'],
                physical_check=check,cost_usd=cost,independently_minimized=False,
                original_vector_reconstruction='primal.npy plus source-bound recourse_columns.npy'))
            checks.append(file_reference(root/f'witness_{i:03d}.json'))
            del template,scene,y,solution,bub,beq
        objective=float(model.c@vector+model.objective_constant_usd)
        tolerance=1e-4+1e-7*max(1,abs(objective))
        _need(abs(full_objective-objective)<=tolerance,'SAA original weighted objective differs')
        _need(native['raw_objective_without_constant_usd'] is not None and
              abs(objective-(native['raw_objective_without_constant_usd']+model.objective_constant_usd))<=tolerance,
              'SAA native objective differs')
        _need(lower is None or lower<=objective+tolerance,'SAA native lower bound exceeds incumbent')
        feasible=True
    _deadline(deadline)
    gap=None if objective is None or lower is None else max(0.,objective-lower)/max(1.,abs(objective))
    native_status=((getattr(raw,'limit_status',None) or 'TIME_LIMIT') if raw.status==1 else
                   {0:'OPTIMAL',2:'INFEASIBLE',3:'UNBOUNDED'}.get(raw.status,'SOLVER_ERROR'))
    result=dict(status=native_status,solver_status=int(raw.status),algorithm_id=model.manifest['algorithm_id'],
        objective_usd=objective,lower_bound_usd=lower,gap=gap,raw_mip_gap=native['raw_mip_gap'],
        incumbent_physical_checks_passed=feasible,normal_convergence=bool(raw.status==0 and feasible and gap is not None and gap<=.005),
        native_bound_source_only=True,branch_and_bound_proof_replayed=False,
        objective_constant_usd=model.objective_constant_usd,model_manifest=file_reference(model.root/'manifest.json'),
        raw_native_result=file_reference(root/'native_result.json'),
        primal=None if vector is None else file_reference(root/'primal.npy'),original_witness_checks=checks,
        solve_seconds=solve_seconds,total_seconds=time.monotonic()-began,
        full_support_fixed_plan_reoptimization_required=True,independent_numerical_audit_required=True,
        full_support_verified=False,scientific_spec_pass=False,formal_OOS=False)
    _write(root/'result.json',result)
    _deadline(deadline)
    return result
