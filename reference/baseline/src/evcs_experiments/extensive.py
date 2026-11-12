"""Full finite-support SAA/DRO/ARO reference MILPs with unfixed first stage.

Every original task and scenario recourse copy is retained. A scenario witness
in a risk epigraph need not minimize its own recourse cost; independently
resolve the returned fixed plan before calling its costs a full-support q.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from types import MappingProxyType
from typing import Mapping
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack

from .first_stage import FirstStagePlan, SparseFirstStage
from .probability import ProbabilitySet
from .recourse import RecourseTemplate


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _readonly(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _vector(value, n, name):
    a = np.asarray(value, dtype=float)
    _require(a.shape == (n,) and np.isfinite(a).all(), f"{name} must be finite [{n}]")
    return _readonly(a)


@dataclass(frozen=True)
class ExtensiveSolution:
    vector: np.ndarray
    plan: FirstStagePlan
    recourse_solutions: tuple
    scenario_witness_costs_usd: np.ndarray
    theta0: float | None
    alpha: np.ndarray | None
    beta: np.ndarray | None
    risk_threshold: float | None
    model_sha256: str

    def as_dict(self):
        return {"model_sha256": self.model_sha256, "x": self.plan.x.tolist(), "r": self.plan.r.tolist(),
                "task_sha256": self.plan.task_sha256, "config_hash": self.plan.config_hash,
                "scenario_witness_costs_usd": self.scenario_witness_costs_usd.tolist(),
                "scenario_costs_are_independently_minimized": False,
                "theta0": self.theta0, "alpha": _json(self.alpha), "beta": _json(self.beta),
                "risk_threshold": self.risk_threshold}


@dataclass(frozen=True)
class SparseExtensive:
    A: csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    c: np.ndarray
    integrality: np.ndarray
    objective_constant_usd: float
    first_stage: SparseFirstStage
    recourses: tuple[RecourseTemplate, ...]
    scenario_ids: tuple[str, ...]
    scenario_sha256s: tuple[str, ...]
    risk_model: str
    probability_set: ProbabilitySet | None
    empirical_probabilities: np.ndarray | None
    scenario_cost_constants_usd: np.ndarray
    variable_slices: Mapping
    block_map: tuple
    model_sha256: str

    def unpack(self, vector):
        v = _vector(vector, len(self.c), "extensive solution")
        h, t = len(self.first_stage.task_ids), len(self.first_stage.periods)
        plan = FirstStagePlan(v[:h], v[h:h+h*t].reshape(h, t), self.first_stage.task_sha256,
                              self.first_stage.config_hash)
        recourse = tuple(template.unpack(v[slice(*self.variable_slices["recourses"][sid])])
                         for sid, template in zip(self.scenario_ids, self.recourses))
        costs = _readonly([template.c @ template.pack(solution)+constant
                           for template, solution, constant in
                           zip(self.recourses, recourse, self.scenario_cost_constants_usd)])
        theta = alpha = beta = threshold = None
        if self.risk_model == "DRO":
            theta = float(v[self.variable_slices["theta0"][0]])
            alpha = _readonly(v[slice(*self.variable_slices["alpha"])])
            beta = _readonly(v[slice(*self.variable_slices["beta"])])
        elif self.risk_model == "ARO":
            threshold = float(v[self.variable_slices["risk_threshold"][0]])
        return ExtensiveSolution(v, plan, recourse, costs, theta, alpha, beta, threshold, self.model_sha256)

    def as_dict(self):
        return {"risk_model": self.risk_model, "model_sha256": self.model_sha256,
                "shape": list(self.A.shape), "nnz": int(self.A.nnz),
                "task_ids": list(self.first_stage.task_ids), "periods": list(self.first_stage.periods),
                "task_sha256": self.first_stage.task_sha256, "config_hash": self.first_stage.config_hash,
                "first_stage_sha256": self.first_stage.matrix_sha256,
                "scenario_ids": list(self.scenario_ids), "scenario_sha256s": list(self.scenario_sha256s),
                "recourse_template_sha256s": [v.template_sha256 for v in self.recourses],
                "ambiguity_sha256": None if self.probability_set is None else self.probability_set.ambiguity_sha256,
                "empirical_probabilities": _json(self.empirical_probabilities),
                "scenario_cost_constants_usd": self.scenario_cost_constants_usd.tolist(),
                "objective_constant_usd": self.objective_constant_usd,
                "variable_slices": _json(self.variable_slices), "block_map": _json(self.block_map),
                "units": {"objective": "USD", "theta0": "USD", "risk_threshold": "USD",
                          "alpha": "USD/feature (lower moment)", "beta": "USD/feature (upper moment)",
                          "physical_columns": "inherit each indexed first-stage/recourse variable map"},
                "first_stage_fixed": False, "all_original_tasks_retained": True,
                "ambiguity_nonemptiness_prechecked_by_assembler": False,
                "risk_equation": "eq:executable_restricted_master (all support)",
                "scenario_witness_costs_need_fixed_plan_reoptimization": True}


@dataclass(frozen=True)
class ExtensiveSolveResult:
    status: str
    optimal: bool
    solution: ExtensiveSolution | None
    objective_usd: float | None
    lower_bound_usd: float | None
    mip_gap: float | None
    solver_status: int
    message: str
    wall_seconds: float
    certificate: Mapping
    raw_solver: Mapping

    def as_dict(self):
        return {"status": self.status, "optimal": self.optimal,
                "solution": None if self.solution is None else self.solution.as_dict(),
                "objective_usd": self.objective_usd, "lower_bound_usd": self.lower_bound_usd,
                "mip_gap": self.mip_gap, "solver_status": self.solver_status, "message": self.message,
                "wall_seconds": self.wall_seconds, "certificate": _json(self.certificate),
                "raw_solver": _json(self.raw_solver)}


def assemble_extensive(first, recourses, *, risk_model, scenario_ids, probability_set=None,
                       empirical_probabilities=None, scenario_cost_constants_usd=None):
    """Assemble one unfixed first stage and every complete recourse copy.

    scenario_ids explicitly bind the recourse tuple to the probability axis.
    For SAA an explicit probability vector is required; it is never normalized.
    Nonemptiness of a DRO ambiguity set is a separate phase-I prerequisite.
    """
    _require(isinstance(first, SparseFirstStage), "Expected SparseFirstStage")
    _require(risk_model in ("SAA", "DRO", "ARO"), "Unknown risk model")
    templates = tuple(recourses); ids = tuple(scenario_ids)
    _require(len(templates) == len(ids) > 0 and len(set(ids)) == len(ids) and
             all(isinstance(v, str) and v for v in ids), "Invalid explicit scenario IDs")
    h, t, nfirst, count = len(first.task_ids), len(first.periods), len(first.c), len(ids)
    _require(nfirst == h+h*t and np.array_equal(first.integrality, np.r_[np.ones(h), np.zeros(h*t)]),
             "First-stage columns/integrality differ from x then r")
    for item in templates:
        _require(isinstance(item, RecourseTemplate), "Expected RecourseTemplate")
        _require(item.task_ids == first.task_ids and item.periods == first.periods and
                 item.task_sha256 == first.task_sha256 and item.config_hash == first.config_hash and
                 item.first_stage.matrix_sha256 == first.matrix_sha256, "First-stage/recourse identity mismatch")
        _require(item.Bx_ub.shape == (item.A_ub.shape[0], h) and
                 item.Br_ub.shape == (item.A_ub.shape[0], h*t) and
                 item.Bx_eq.shape == (item.A_eq.shape[0], h) and
                 item.Br_eq.shape == (item.A_eq.shape[0], h*t), "Recourse parameter column mismatch")
    if probability_set is not None:
        _require(isinstance(probability_set, ProbabilitySet) and probability_set.support_ids == ids,
                 "Probability support IDs/order mismatch")
    _require(risk_model != "DRO" or probability_set is not None, "DRO requires ProbabilitySet")
    pi = None if empirical_probabilities is None else _vector(empirical_probabilities, count, "empirical probabilities")
    if pi is not None:
        _require(np.all(pi >= 0) and abs(float(pi.sum())-1.) <= 1e-8, "Invalid empirical probability law")
    _require(risk_model != "SAA" or pi is not None, "SAA requires explicit empirical probabilities")
    constants = _vector(np.zeros(count) if scenario_cost_constants_usd is None else scenario_cost_constants_usd,
                        count, "scenario cost constants")
    slices = {"x": (0,h), "r": (h,nfirst), "first_stage": (0,nfirst), "recourses": {}}
    cursor = nfirst
    for sid, item in zip(ids, templates):
        slices["recourses"][sid] = (cursor,cursor+len(item.c)); cursor += len(item.c)
    nrisk_start = cursor
    if risk_model == "DRO":
        f = len(probability_set.feature_ids)
        for name, length in (("theta0",1),("alpha",f),("beta",f)):
            slices[name] = (cursor,cursor+length); cursor += length
    elif risk_model == "ARO":
        slices["risk_threshold"] = (cursor,cursor+1); cursor += 1
    n = cursor
    row_matrices, lowers, uppers, blocks = [], [], [], []
    row_cursor = 0

    def append(matrix, lower, upper, scope, kind, source_hash):
        nonlocal row_cursor
        row_matrices.append(matrix); lowers.append(lower); uppers.append(upper)
        blocks.append({"scope": scope, "kind": kind, "row_start": row_cursor,
                       "row_stop": row_cursor+matrix.shape[0], "source_sha256": source_hash})
        row_cursor += matrix.shape[0]

    append(hstack((first.A, csr_matrix((first.A.shape[0], n-nfirst))), format="csr"),
           first.row_lower, first.row_upper, "first_stage", "first_stage", first.matrix_sha256)
    for sid, item in zip(ids, templates):
        start, stop = slices["recourses"][sid]
        for sense in ("ub", "eq"):
            a, bx, br, b = (getattr(item, name+"_"+sense) for name in ("A", "Bx", "Br", "b"))
            matrix = hstack((-bx, -br, csr_matrix((a.shape[0],start-nfirst)), a,
                             csr_matrix((a.shape[0],n-stop))), format="csr")
            append(matrix, np.full(len(b),-np.inf) if sense == "ub" else b, b,
                   sid, "recourse_"+sense, item.template_sha256)
    c = np.zeros(n); c[:nfirst] = first.c
    lower = np.r_[first.lower_bounds, *[v.lower_bounds for v in templates], np.zeros(n-nrisk_start)]
    upper = np.r_[first.upper_bounds, *[v.upper_bounds for v in templates], np.full(n-nrisk_start,np.inf)]
    integrality = np.r_[first.integrality, np.zeros(n-nfirst,dtype=int)]
    offset = 0.
    if risk_model == "SAA":
        for weight, sid, item in zip(pi, ids, templates):
            c[slice(*slices["recourses"][sid])] = weight*item.c
        offset = float(pi @ constants)
    else:
        risk_index = nrisk_start; c[risk_index] = 1.; lower[risk_index] = -np.inf
        if risk_model == "DRO":
            c[slice(*slices["alpha"])] = -probability_set.lower
            c[slice(*slices["beta"])] = probability_set.upper
        for index, (sid, item) in enumerate(zip(ids, templates)):
            start, stop = slices["recourses"][sid]
            risk = np.r_[-1., probability_set.features[index], -probability_set.features[index]] if risk_model == "DRO" else [-1.]
            matrix = hstack((csr_matrix((1,start)),csr_matrix(item.c.reshape(1,-1)),
                             csr_matrix((1,nrisk_start-stop)),csr_matrix(np.asarray(risk).reshape(1,-1))),format="csr")
            append(matrix, np.array([-np.inf]), np.array([-constants[index]]), sid,
                   "risk_epigraph", item.template_sha256)
    matrix = vstack(row_matrices,format="csr"); matrix.sum_duplicates(); matrix.sort_indices()
    rl, ru = np.concatenate(lowers), np.concatenate(uppers)
    _require(np.isfinite(matrix.data).all() and np.isfinite(c).all() and np.isfinite(offset) and
             not any(np.isnan(v).any() for v in (rl,ru,lower,upper)) and
             np.all(rl<=ru) and np.all(lower<=upper), "Invalid assembled matrix or bounds")
    metadata = {"risk_model":risk_model,"scenario_ids":ids,"scenario_sha256s":[v.scenario_sha256 for v in templates],
                "template_sha256s":[v.template_sha256 for v in templates],"first_stage_sha256":first.matrix_sha256,
                "ambiguity_sha256":None if probability_set is None else probability_set.ambiguity_sha256,
                "empirical_probabilities":None if pi is None else pi.tolist(),
                "slices":slices,"objective_constant_usd":offset}
    digest = hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(",", ":")).encode())
    for a in (matrix.data,matrix.indices,matrix.indptr,rl,ru,lower,upper,c,integrality,constants):
        v = np.asarray(a,dtype="<i8" if np.asarray(a).dtype.kind in "iu" else "<f8")
        digest.update(str(v.shape).encode()); digest.update(v.tobytes())
    for name in ("data","indices","indptr"):
        a = getattr(matrix,name); setattr(matrix,name,_readonly(a,a.dtype))
    return SparseExtensive(matrix,_readonly(rl),_readonly(ru),_readonly(lower),_readonly(upper),_readonly(c),
        _readonly(integrality,int),offset,first,templates,ids,tuple(v.scenario_sha256 for v in templates),risk_model,
        probability_set,pi,constants,_freeze(slices),tuple(_freeze(v) for v in blocks),digest.hexdigest())


def full_shedding_start(model):
    """A checked feasible hint; never change the MILP decision domain."""
    from .recourse import full_shedding_fallback
    first = model.first_stage
    plan = FirstStagePlan(np.zeros(len(first.task_ids)), np.zeros((len(first.task_ids),len(first.periods))),
                          first.task_sha256, first.config_hash)
    _require(first.evaluate(plan).feasible, 'Zero first-stage plan is not feasible')
    v = np.zeros(len(model.c)); costs = []
    for sid,template,constant in zip(model.scenario_ids,model.recourses,model.scenario_cost_constants_usd):
        y = template.pack(full_shedding_fallback(template,plan))
        v[slice(*model.variable_slices['recourses'][sid])] = y
        costs.append(float(template.c@y+constant))
    if model.risk_model != 'SAA':
        key = 'theta0' if model.risk_model == 'DRO' else 'risk_threshold'
        v[slice(*model.variable_slices[key])] = max(costs)
    activity = model.A@v
    violation = max(float(np.max(np.maximum(model.row_lower-activity,activity-model.row_upper),initial=0)),
                    float(np.max(np.maximum(model.lower_bounds-v,v-model.upper_bounds),initial=0)))
    _require(violation<=1e-8, 'Full-shedding start failed original matrix check')
    return v


def solve_extensive(model, *, time_limit_seconds=30, mip_relative_gap=1e-9, zero_presolve=False,
                    native_start=False, solver_log_path=None, master_backend='highs', optimizer_threads=1,
                    solver_feasibility_tolerance=None):
    """Solve the reference MILP; retain a timeout incumbent and any solver LB.

    LP duals of physical recourse copies are not inferred from MIP output.
    Missing solver bounds remain None. The SAA objective constant shifts both
    incumbent and solver lower bound, while the raw solver gap is retained.
    """
    _require(isinstance(model,SparseExtensive), "Expected SparseExtensive")
    _require(not isinstance(time_limit_seconds,(bool,np.bool_)) and np.isfinite(time_limit_seconds) and time_limit_seconds>0,
             "Positive finite time limit required")
    _require(not isinstance(mip_relative_gap,(bool,np.bool_)) and np.isfinite(mip_relative_gap) and mip_relative_gap>=0,
             "Nonnegative finite MIP gap required")
    _require(master_backend in ('highs', 'gurobi'), 'Unknown master backend')
    _require(isinstance(optimizer_threads, int) and not isinstance(optimizer_threads, bool)
             and 1 <= optimizer_threads <= 4, 'Optimizer threads must be 1..4')
    _require(master_backend != 'highs' or optimizer_threads == 1, 'HiGHS reference uses one thread')
    _require(master_backend != 'gurobi' or solver_log_path is not None, 'Gurobi requires a retained solver log')
    options={"time_limit":float(time_limit_seconds),"mip_rel_gap":float(mip_relative_gap),"threads":optimizer_threads,
             "primal_feasibility_tolerance":1e-8,"dual_feasibility_tolerance":1e-8}
    if solver_feasibility_tolerance is not None:
        _require(master_backend=='gurobi' and type(solver_feasibility_tolerance) in (int,float)
                 and solver_feasibility_tolerance in (1e-9,1e-8),
                 'Explicit solver refinement requires Gurobi tolerance 1e-9 or 1e-8')
        options['solver_feasibility_tolerance']=solver_feasibility_tolerance
    began=time.monotonic()
    reduction = None
    if zero_presolve:
        from .zero_presolve import reduce_zero_columns
        reduction = reduce_zero_columns(model.A, model.row_lower, model.row_upper,
                                        model.lower_bounds, model.upper_bounds)
    columns = np.arange(len(model.c)) if reduction is None else reduction.column_indices
    rows = np.arange(model.A.shape[0]) if reduction is None else reduction.row_indices
    matrix = model.A if reduction is None else reduction.matrix
    # An all-zero model is left to the original backend instead of passing an
    # unsupported zero-column MILP. No synthetic solver bound is invented.
    if not len(columns):
        reduction = None; columns = np.arange(len(model.c)); rows = np.arange(model.A.shape[0]); matrix = model.A
    if zero_presolve:
        options['time_limit'] = max(1e-6, float(time_limit_seconds)-(time.monotonic()-began))
    if master_backend == 'gurobi':
        from .native_gurobi import solve_milp
        seed = full_shedding_start(model)[columns] if native_start else None
        options['time_limit'] = max(1e-6,float(time_limit_seconds)-(time.monotonic()-began))
        raw = solve_milp(model.c[columns],model.integrality[columns],model.lower_bounds[columns],
            model.upper_bounds[columns],matrix,model.row_lower[rows],model.row_upper[rows],
            options=options,start=seed,log_path=solver_log_path)
    elif native_start:
        from .native_highs import solve_with_start
        seed = full_shedding_start(model)
        options['time_limit'] = max(1e-6,float(time_limit_seconds)-(time.monotonic()-began))
        raw = solve_with_start(model.c[columns],model.integrality[columns],model.lower_bounds[columns],
            model.upper_bounds[columns],matrix,model.row_lower[rows],model.row_upper[rows],seed[columns],
            options,solver_log_path)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning,message="Unrecognized options detected.*")
            raw=milp(model.c[columns],integrality=model.integrality[columns],
                 bounds=Bounds(model.lower_bounds[columns],model.upper_bounds[columns]),
                 constraints=LinearConstraint(matrix,model.row_lower[rows],model.row_upper[rows]),options=options)
    if reduction is not None and raw.x is not None:
        raw.x = reduction.expand(raw.x)
    elapsed=time.monotonic()-began
    def finite(name):
        value=getattr(raw,name,None)
        return None if value is None or not np.isfinite(value) else float(value)
    raw_bound=finite("mip_dual_bound"); raw_gap=finite("mip_gap")
    lower_bound=None if raw_bound is None or raw.status not in (0,1) else raw_bound+model.objective_constant_usd
    solution=None; objective=None; feasible=False
    certificate={"model_sha256":model.model_sha256,"primal_feasible":False,"original_unit_tolerance":1e-8,
        "normalized_tolerance":1e-8,"integrality_tolerance":1e-8,"objective_constant_usd":model.objective_constant_usd,
        "raw_mip_gap":raw_gap,"requested_mip_relative_gap":float(mip_relative_gap),
        "solver_bound_available":lower_bound is not None,"full_support_fixed_plan_reoptimization_required":True,
        "infeasibility_certificate_available":False}
    if raw.x is not None and np.asarray(raw.x).shape==model.c.shape and np.isfinite(raw.x).all():
        solution=model.unpack(raw.x); v=solution.vector; activity=model.A@v
        violation=np.maximum(np.maximum(model.row_lower-activity,activity-model.row_upper),0.)
        bound_violation=np.maximum(np.maximum(model.lower_bounds-v,v-model.upper_bounds),0.)
        row_scale=np.maximum(1.,np.abs(model.A)@np.abs(v))
        integer=abs(v[model.integrality==1]-np.rint(v[model.integrality==1]))
        maxima={"max_row_violation":float(np.max(violation,initial=0.)),
                "max_normalized_row_violation":float(np.max(violation/row_scale,initial=0.)),
                "max_bound_violation":float(np.max(bound_violation,initial=0.)),
                "max_normalized_bound_violation":float(np.max(bound_violation/np.maximum(1.,abs(v)),initial=0.)),
                "max_integrality_violation":float(np.max(integer,initial=0.))}
        feasible=all(x<=1e-8 for x in maxima.values())
        objective=float(model.c@v+model.objective_constant_usd)
        _require(np.isfinite(objective),"Nonfinite extensive objective")
        firstcost=float(model.first_stage.c@v[slice(*model.variable_slices["first_stage"])])
        certificate.update(maxima,primal_feasible=feasible,first_stage_cost_usd=firstcost,
                           risk_epigraph_cost_usd=objective-firstcost)
    objective_tol=None if objective is None else 1e-4+1e-7*max(1.,abs(objective),abs(lower_bound or 0.))
    bound_gap=None if objective is None or lower_bound is None else objective-lower_bound
    bound_valid=bound_gap is not None and bound_gap>=-objective_tol
    gap_passed=bound_valid and abs(bound_gap)<=objective_tol
    raw_gap_passed=raw_gap is not None and 0<=raw_gap<=mip_relative_gap
    raw_fun=finite("fun")
    objective_matches=bool(objective is not None and raw_fun is not None and
                          abs(objective-(raw_fun+model.objective_constant_usd))<=objective_tol)
    optimal=bool(raw.status==0 and feasible and gap_passed and raw_gap_passed and objective_matches)
    certificate.update(objective_tolerance_usd=objective_tol,incumbent_minus_solver_bound_usd=bound_gap,
                       solver_bound_consistent=bound_valid,gap_passed=bool(gap_passed),
                       solver_bound_usable=bool(lower_bound is not None and (objective is None or bound_valid)),
                       raw_gap_passed=raw_gap_passed,raw_gap_roundoff_allowance=0.,
                       objective_matches_solver=objective_matches,certified_optimal_within_tolerance=optimal)
    if optimal: status="OPTIMAL"
    elif raw.status==1: status=getattr(raw,'limit_status','TIME_LIMIT')
    elif raw.status==2: status="INFEASIBLE"
    elif raw.status==3: status="UNBOUNDED"
    elif raw.status==0: status="NUMERICAL_CERTIFICATE_FAILED"
    else: status="SOLVER_ERROR"
    raw_record={"solver":"scipy.optimize.milp/HiGHS","status":int(raw.status),"message":str(raw.message),
        "success":bool(getattr(raw,"success",False)),"raw_objective_without_constant_usd":raw_fun,
        "raw_lower_bound_without_constant_usd":raw_bound,"raw_mip_gap":raw_gap,
        "mip_node_count":finite("mip_node_count"),"options":options}
    if master_backend == 'gurobi':
        raw_record['mip_start_supplied'] = bool(native_start)
    if reduction is not None:
        raw_record['zero_reduction'] = reduction.as_dict()
    if master_backend == 'gurobi':
        raw_record['native_interface'] = raw.native_record
        raw_record['solver'] = 'gurobipy/Gurobi'
    elif native_start:
        raw_record['native_interface'] = raw.native_record
        raw_record['solver'] = 'SciPy-bundled HiGHS native/setSolution'
    if lower_bound is not None and objective is not None and not bound_valid:
        lower_bound=None  # Retain the contradictory report only in raw_solver.
    return ExtensiveSolveResult(status,optimal,solution,objective,lower_bound,raw_gap,int(raw.status),str(raw.message),
                                elapsed,_freeze(certificate),_freeze(raw_record))
