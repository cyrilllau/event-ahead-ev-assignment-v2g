"""Explicit, optional Gurobi 12.0.3 adapter for the unchanged sparse MILP.

Importing this module never imports Gurobi or starts a license. The caller owns
objective constants and any expansion from a reduced model to original units.
No other solver is used when Gurobi is unavailable or rejects the model.
"""
import hashlib
import importlib
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix


GUROBI_VERSION = (12, 0, 3)
TOLERANCE = 1e-8
_STATUSES = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD',
             5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT',
             9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC',
             13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT',
             16: 'WORK_LIMIT', 17: 'MEM_LIMIT'}
_LIMIT_STATUSES = frozenset((7, 8, 9, 10, 11, 15, 16, 17))


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite number') from exc
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(name + ' is outside the allowed range')
    return value


def _parameters(options):
    if not isinstance(options, dict) or 'time_limit' not in options:
        raise ValueError('Gurobi requires options with a positive time_limit')
    allowed = {'time_limit', 'mip_rel_gap', 'threads',
               'primal_feasibility_tolerance', 'dual_feasibility_tolerance',
               'solver_feasibility_tolerance'}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError('Unsupported Gurobi options: ' + ', '.join(sorted(unknown)))
    seconds = _number(options['time_limit'], 'time_limit', positive=True)
    gap = _number(options.get('mip_rel_gap', 1e-9), 'mip_rel_gap')
    threads = _number(options.get('threads', 1), 'threads', positive=True)
    if threads != int(threads) or threads > 4:
        raise ValueError('Gurobi threads must be an integer from 1 to 4')
    for key in ('primal_feasibility_tolerance', 'dual_feasibility_tolerance'):
        if key in options and _number(options[key], key, positive=True) != TOLERANCE:
            raise ValueError(key + ' must remain 1e-8')
    solver_tol = options.get('solver_feasibility_tolerance', TOLERANCE)
    if type(solver_tol) not in (int, float) or solver_tol not in (1e-9, TOLERANCE):
        raise ValueError('solver_feasibility_tolerance must be 1e-9 or 1e-8')
    # Solver refinement is separate from the unchanged original-array 1e-8 gate.
    return seconds, dict(OutputFlag=1, LogToConsole=0, Threads=int(threads), Seed=0,
                        FeasibilityTol=solver_tol, OptimalityTol=TOLERANCE,
                        IntFeasTol=TOLERANCE, MIPGap=gap, SoftMemLimit=24., Method=-1)


def _model_arrays(c, integrality, lower, upper, matrix, row_lower, row_upper):
    if any(np.iscomplexobj(value) for value in
           (c, integrality, lower, upper, matrix, row_lower, row_upper)):
        raise ValueError('Complex model coefficients and bounds are unsupported')
    arrays = [np.asarray(value, dtype=float) for value in
              (c, integrality, lower, upper, row_lower, row_upper)]
    c, integer, lower, upper, row_lower, row_upper = arrays
    if c.ndim != 1 or not c.size or any(v.ndim != 1 for v in arrays):
        raise ValueError('Model vectors must be one-dimensional, with at least one column')
    if any(v.shape != c.shape for v in (integer, lower, upper)):
        raise ValueError('Column vector shape mismatch')
    if row_lower.shape != row_upper.shape:
        raise ValueError('Row bound shape mismatch')
    if not np.isfinite(c).all() or not np.isin(integer, (0, 1)).all():
        raise ValueError('Finite objective and integrality types 0/1 required')
    a = csr_matrix(matrix, dtype=float, copy=True)
    a.sum_duplicates(); a.sort_indices()
    if a.shape != (row_lower.size, c.size) or not np.isfinite(a.data).all():
        raise ValueError('Invalid sparse matrix shape or coefficients')
    for lo, hi in ((lower, upper), (row_lower, row_upper)):
        if (np.isnan(lo).any() or np.isnan(hi).any() or np.isposinf(lo).any()
                or np.isneginf(hi).any() or np.any(lo > hi)):
            raise ValueError('Invalid or reversed model bounds')
    # Gurobi treats sufficiently large finite bounds/RHS as infinity. Refuse
    # inputs that could silently change the feasible set on translation.
    for value in (lower, upper, row_lower, row_upper):
        if np.any(np.abs(value[np.isfinite(value)]) >= 1e20):
            raise ValueError('Finite model bounds must have magnitude below 1e20')
    if np.any((np.abs(a.data) > 0) & (np.abs(a.data) < 1e-13)):
        raise ValueError('Gurobi would discard matrix coefficients smaller than 1e-13')
    return c, integer.astype(np.int8), lower, upper, a, row_lower, row_upper


def _check_vector(c, integer, lower, upper, a, row_lower, row_upper, vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != c.shape or not np.isfinite(v).all():
        return {'passed': False, 'reason': 'Invalid solution vector shape or values'}
    activity = a @ v
    objective = float(c @ v)
    if not np.isfinite(activity).all() or not np.isfinite(objective):
        return {'passed': False, 'reason': 'Nonfinite original activity or objective'}
    errors = dict(max_row_violation=max(0., float(np.max(
                      np.maximum(row_lower-activity, activity-row_upper), initial=0.))),
                  max_bound_violation=max(0., float(np.max(
                      np.maximum(lower-v, v-upper), initial=0.))),
                  max_integrality_violation=float(np.max(
                      abs(v[integer == 1]-np.rint(v[integer == 1])), initial=0.)))
    return dict(errors, passed=all(v <= TOLERANCE for v in errors.values()),
                objective_without_constant_usd=objective)


def _row_map(row_lower, row_upper):
    eq = np.isfinite(row_lower) & np.isfinite(row_upper) & (row_lower == row_upper)
    return dict(eq=np.flatnonzero(eq), le=np.flatnonzero(np.isfinite(row_upper) & ~eq),
                ge=np.flatnonzero(np.isfinite(row_lower) & ~eq))


def _attribute(gp, model, name):
    try:
        return getattr(model, name)
    except (AttributeError, gp.GurobiError):
        return None


def _finite(value, *, infinity=math.inf):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and abs(number) < infinity else None


def _extract_result(gp, model, variables, arrays, record):
    """Extract only checked incumbents and solver-certified finite bounds."""
    native_status = int(model.Status)
    status_name = _STATUSES.get(native_status, 'STATUS_' + str(native_status))
    status = (0 if native_status == 2 else 1 if native_status in _LIMIT_STATUSES
              else 2 if native_status == 3 else 3 if native_status == 5 else 4)
    sol_count = int(model.SolCount)
    is_mip = bool(model.IsMIP)
    objective = None if not sol_count else _finite(_attribute(gp, model, 'ObjVal'))
    vector = None if not sol_count else np.asarray(variables.X, dtype=float)
    check = None if vector is None else _check_vector(*arrays, vector)
    objective_tol = None if objective is None else 1e-4 + 1e-7*max(1., abs(objective))
    matches = bool(check is not None and check.get('objective_without_constant_usd') is not None
                   and objective is not None and
                   abs(check['objective_without_constant_usd']-objective) <= objective_tol)
    valid_incumbent = bool(check is not None and check['passed'] and matches)
    raw_bound = None
    source = None
    if status in (0, 1):
        if is_mip:
            raw_bound = _finite(_attribute(gp, model, 'ObjBound'), infinity=gp.GRB.INFINITY)
            source = 'MIP_ObjBound'
        elif status == 0 and valid_incumbent:
            # A stopped LP's primal objective is not a certified lower bound.
            raw_bound = objective
            source = 'optimal_LP_objective'
    consistent = raw_bound is not None and (objective is None or raw_bound <= objective+objective_tol)
    bound = raw_bound if consistent else None
    certificate_failures = []
    if sol_count and not valid_incumbent:
        certificate_failures.append('Solver incumbent fails original feasibility/objective certificate')
    if raw_bound is not None and not consistent:
        certificate_failures.append('Solver bound exceeds the reported incumbent')
    if certificate_failures:
        status = 4
        bound = None
    raw_gap = _finite(_attribute(gp, model, 'MIPGap')) if is_mip else (0. if status == 0 else None)
    gap = raw_gap if valid_incumbent and bound is not None and raw_gap is not None and raw_gap >= 0 else None
    nodes = _finite(_attribute(gp, model, 'NodeCount'))
    record.update(raw_model_status=native_status, status=status_name, solution_count=sol_count,
        is_mip=is_mip, original_certificate=check, valid_incumbent=valid_incumbent,
        certificate_failures=certificate_failures,
        objective_matches=matches, incumbent_without_constant_usd=objective,
        raw_solver_bound_without_constant_usd=raw_bound,
        solver_bound_without_constant_usd=bound, solver_bound_consistent=bool(consistent),
        bound_source=source if bound is not None else None, raw_mip_gap=raw_gap,
        solver_run_seconds=_finite(_attribute(gp, model, 'Runtime')),
        max_solver_memory_gb=_finite(_attribute(gp, model, 'MaxMemUsed')),
        solver_memory_gb=_finite(_attribute(gp, model, 'MemUsed')),
        simplex_iterations=_finite(_attribute(gp, model, 'IterCount')),
        barrier_iterations=_finite(_attribute(gp, model, 'BarIterCount')),
        model_rows=int(model.NumConstrs), model_columns=int(model.NumVars))
    return OptimizeResult(x=vector if valid_incumbent else None,
        fun=objective if valid_incumbent else None, status=status,
        success=bool(status == 0 and valid_incumbent and bound is not None),
        message='Gurobi native: ' + status_name + (
            '; certificate failed: ' + '; '.join(certificate_failures) if certificate_failures else ''),
        limit_status=status_name if native_status in _LIMIT_STATUSES else None, mip_dual_bound=bound,
        mip_gap=gap, mip_node_count=nodes, native_record=record)


def solve_milp(c, integrality, lower, upper, matrix, row_lower, row_upper, *,
               options, start=None, log_path=None):
    """Solve with an explicit bounded budget and a new, preserved native log.

    ``options`` uses the SciPy spellings time_limit, mip_rel_gap, and threads.
    A complete feasible start is a hint only; it never fixes any variable.
    Model loading is charged against time_limit. Licensing and native failures
    propagate to the caller, without an implicit backend change.
    """
    began = time.monotonic()
    seconds, parameters = _parameters(options)
    arrays = _model_arrays(c, integrality, lower, upper, matrix, row_lower, row_upper)
    c, integer, lower, upper, a, row_lower, row_upper = arrays
    hint = None if start is None else _check_vector(*arrays, start)
    if hint is not None and not hint['passed']:
        raise ValueError('MIP start violates the supplied original/reduced model')
    if log_path is None:
        raise ValueError('Gurobi requires an explicit unique per-run log_path')
    log_path = Path(log_path).resolve()
    if log_path.exists():
        raise ValueError('Refusing to overwrite native solver log')
    gp = importlib.import_module('gurobipy')
    version = tuple(gp.gurobi.version())
    if version != GUROBI_VERSION:
        raise RuntimeError('Gurobi 12.0.3 is required; found ' + '.'.join(map(str, version)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation also closes the check/open race between runs.
    try:
        with log_path.open('x'):
            pass
    except FileExistsError as exc:
        raise ValueError('Refusing to overwrite native solver log') from exc
    parameters['LogFile'] = str(log_path)
    mapping = _row_map(row_lower, row_upper)
    summary = {name: {'count': int(len(rows)), 'source_rows_sha256': hashlib.sha256(
                       rows.astype('<i8').tobytes()).hexdigest()} for name, rows in mapping.items()}
    summary.update(original_rows=int(a.shape[0]),
                   translated_rows=sum(len(rows) for rows in mapping.values()),
                   free_rows=int(np.count_nonzero(np.isneginf(row_lower) & np.isposinf(row_upper))),
                   ranged_rows=int(np.count_nonzero(np.isfinite(row_lower) & np.isfinite(row_upper)
                                                    & (row_lower < row_upper))))
    record = dict(backend='gurobipy', gurobi_version=list(version), effective_options=parameters,
                  requested_time_limit_seconds=seconds, log_path=str(log_path), row_map_summary=summary,
                  original_matrix_shape=list(a.shape), original_matrix_nnz=int(a.nnz),
                  start_certificate=hint, start_is_not_fixed_decision=start is not None)
    if start is not None:
        record.update(start_sha256=hashlib.sha256(np.asarray(start, dtype='<f8').tobytes()).hexdigest(),
                      start_objective_without_constant_usd=hint['objective_without_constant_usd'])
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            for name, value in parameters.items():
                model.setParam(name, value)
            model.ModelSense = gp.GRB.MINIMIZE
            variables = model.addMVar(a.shape[1], lb=lower, ub=upper, obj=c,
                vtype=np.where(integer == 1, gp.GRB.INTEGER, gp.GRB.CONTINUOUS))
            for name, sense, rhs in (('eq', gp.GRB.EQUAL, row_lower),
                                     ('le', gp.GRB.LESS_EQUAL, row_upper),
                                     ('ge', gp.GRB.GREATER_EQUAL, row_lower)):
                rows = mapping[name]
                if rows.size:
                    model.addMConstr(a[rows], variables, sense, rhs[rows], name=name)
            if start is not None:
                variables.Start = np.asarray(start, dtype=float)
            model.update()
            load_seconds = time.monotonic() - began
            remaining = seconds - load_seconds
            record.update(load_seconds=load_seconds, effective_run_time_limit_seconds=max(0., remaining))
            if remaining <= 0:
                record.update(status='MODEL_LOADING_TIME_LIMIT', raw_model_status=int(model.Status),
                              solver_run_seconds=0., wall_seconds=load_seconds, valid_incumbent=False)
                return OptimizeResult(x=None, fun=None, status=1, success=False,
                    message='Gurobi native: model loading exhausted time budget', limit_status='TIME_LIMIT',
                    mip_dual_bound=None, mip_gap=None, mip_node_count=None, native_record=record)
            parameters['TimeLimit'] = remaining
            model.setParam('TimeLimit', remaining)
            model.optimize()
            result = _extract_result(gp, model, variables, arrays, record)
            record['wall_seconds'] = time.monotonic() - began
            return result


def solve_with_start(c, integrality, lower, upper, matrix, row_lower, row_upper,
                     start, options, log_path=None):
    """Compatibility spelling shared with the native HiGHS adapter."""
    return solve_milp(c, integrality, lower, upper, matrix, row_lower, row_upper,
                      options=options, start=start, log_path=log_path)
