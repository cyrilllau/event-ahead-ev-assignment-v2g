"""Separate, pinned-backend diagnostic of an unchanged saved integer RMP.

Requires gurobipy in a separately provisioned environment and an existing
license. Does not modify campaign C&CG state or claim a full-support UB.
"""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import time

import numpy as np
from scipy.sparse import csr_matrix, load_npz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_check(matrix, arrays, vector, *, check_integer=True):
    x = np.asarray(vector, dtype=float)
    if x.shape != arrays['c'].shape or not np.isfinite(x).all():
        raise ValueError('Invalid original solution vector')
    activity = matrix @ x
    rows = np.maximum(arrays['row_lower']-activity, activity-arrays['row_upper'])
    bounds = np.maximum(arrays['lower_bounds']-x, x-arrays['upper_bounds'])
    ints = x[arrays['integrality'] == 1]
    errors = dict(row_violation=max(0., float(np.max(rows, initial=0.))),
                  bound_violation=max(0., float(np.max(bounds, initial=0.))),
                  integrality_violation=float(np.max(abs(ints-np.rint(ints)), initial=0.)) if check_integer else None)
    errors['passed'] = all(v <= 1e-8 for v in errors.values() if v is not None)
    errors['objective_without_constant_usd'] = float(arrays['c'] @ x)
    return errors


def solve_saved(matrix, arrays, start, log_path, *, seconds=480., threads=4, wall_seconds=580.):
    import gurobipy as gp
    began = time.monotonic(); a = csr_matrix(matrix)
    if np.any(~np.isin(arrays['integrality'], (0, 1))):
        raise ValueError('Unsupported original integrality type')
    if Path(log_path).exists():
        raise ValueError('Refusing to overwrite solver log')
    hint = None if start is None else original_check(a, arrays, start)
    if hint is not None and not hint['passed']:
        raise ValueError('MIP hint violates the original model')
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0); env.start()
        with gp.Model(env=env) as model:
            params = dict(OutputFlag=1, LogToConsole=0, LogFile=str(log_path), Threads=int(threads), Seed=0,
                          FeasibilityTol=1e-8, OptimalityTol=1e-8, IntFeasTol=1e-8,
                          MIPGap=.001, SoftMemLimit=24., Method=-1)
            for name, value in params.items():
                model.setParam(name, value)
            variables = model.addMVar(a.shape[1], lb=arrays['lower_bounds'], ub=arrays['upper_bounds'],
                obj=arrays['c'], vtype=np.where(arrays['integrality'] == 1, gp.GRB.INTEGER, gp.GRB.CONTINUOUS))
            rl, ru = arrays['row_lower'], arrays['row_upper']
            eq = np.isfinite(rl) & np.isfinite(ru) & (rl == ru)
            row_map = dict(eq=np.flatnonzero(eq), le=np.flatnonzero(np.isfinite(ru) & ~eq),
                           ge=np.flatnonzero(np.isfinite(rl) & ~eq))
            for name, sense, rhs in (('eq', gp.GRB.EQUAL, rl), ('le', gp.GRB.LESS_EQUAL, ru), ('ge', gp.GRB.GREATER_EQUAL, rl)):
                rows = row_map[name]
                if len(rows):
                    model.addMConstr(a[rows], variables, sense, rhs[rows], name=name)
            if start is not None:
                variables.Start = np.asarray(start)
            model.update()
            remaining = min(float(seconds), float(wall_seconds)-(time.monotonic()-began)-10.)
            if remaining <= 0:
                raise TimeoutError('Model loading exhausted diagnostic budget')
            params['TimeLimit'] = remaining; model.setParam('TimeLimit', remaining)
            load_seconds = time.monotonic()-began
            model.optimize()
            status = int(model.Status)
            names = {gp.GRB.OPTIMAL:'OPTIMAL', gp.GRB.TIME_LIMIT:'TIME_LIMIT', gp.GRB.INFEASIBLE:'INFEASIBLE',
                     gp.GRB.INF_OR_UNBD:'INF_OR_UNBD', gp.GRB.UNBOUNDED:'UNBOUNDED', gp.GRB.INTERRUPTED:'INTERRUPTED',
                     gp.GRB.NUMERIC:'NUMERIC', gp.GRB.MEM_LIMIT:'MEM_LIMIT'}
            solution = None if model.SolCount == 0 else np.asarray(variables.X)
            check = None if solution is None else original_check(a, arrays, solution)
            is_mip = bool(model.IsMIP)
            bound = None
            bound_source = None
            if status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.MEM_LIMIT, gp.GRB.INTERRUPTED):
                if is_mip:
                    bound = float(model.ObjBound); bound_source = 'MIP_ObjBound'
                elif status == gp.GRB.OPTIMAL:
                    bound = float(model.ObjVal); bound_source = 'optimal_LP_objective'
            bound = bound if bound is not None and np.isfinite(bound) and abs(bound) < gp.GRB.INFINITY else None
            objective = None if solution is None else float(model.ObjVal)
            tolerance = None if objective is None else 1e-4+1e-7*max(1., abs(objective))
            objective_matches = bool(check is not None and abs(check['objective_without_constant_usd']-objective) <= tolerance)
            bound_consistent = bound is not None and (objective is None or bound <= objective+tolerance)
            record = dict(status=names.get(status, 'STATUS_'+str(status)), native_status=status,
                gurobi_version=list(gp.gurobi.version()), effective_parameters=params,
                solution_count=int(model.SolCount), incumbent_without_constant_usd=objective,
                solver_bound_without_constant_usd=bound, solver_bound_consistent=bound_consistent,
                mip_gap=None if not is_mip or solution is None or bound is None else float(model.MIPGap),
                is_mip=is_mip, bound_source=bound_source,
                original_certificate=check, objective_matches=objective_matches,
                root_or_tree_lp_iterations=float(model.IterCount), barrier_iterations=int(model.BarIterCount),
                node_count=float(model.NodeCount), solver_seconds=float(model.Runtime),
                load_seconds=load_seconds, max_solver_memory_gb=float(model.MaxMemUsed),
                hint=hint, model_columns=int(model.NumVars), model_rows=int(model.NumConstrs),
                original_matrix_shape=list(a.shape), integers=int(np.count_nonzero(arrays['integrality'])),
                full_support_upper_bound=False, accepted_campaign_lower_bound=False, full_S03_pass=False)
            record['valid_incumbent'] = bool(check is not None and check['passed'] and objective_matches)
            values = {name+'_source_rows': rows for name, rows in row_map.items()}
            if solution is not None:
                values['vector'] = solution
            return record, values


def main():
    began = time.monotonic(); ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True); args = ap.parse_args()
    request = json.loads(args.request.read_text()); out = Path(request['output']); out.mkdir(parents=True, exist_ok=False)
    for v in request['references'].values():
        if sha(v['path']) != v['sha256']:
            raise ValueError('Changed diagnostic reference: '+v['path'])
    project = Path(request['project'])
    for v in json.loads((project/'RELEASE_MANIFEST.json').read_text())['files']:
        if sha(project/v['path']) != v['sha256']:
            raise ValueError('Original release changed')
    refs = request['references']; matrix = load_npz(refs['matrix']['path']).tocsr()
    with np.load(refs['model_arrays']['path'], allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    with np.load(refs['mip_start']['path'], allow_pickle=False) as z:
        start = z['vector']
    (out/'execution.json').write_text(json.dumps(dict(request=request, request_sha256=sha(args.request),
        script_sha256=sha(__file__), original_matrix_sha256=sha(refs['matrix']['path']), OOS=False,
        full_S03_pass=False, original_integer_model=True), indent=2)+'\n')
    record, values = solve_saved(matrix, arrays, start, out/'gurobi.log', seconds=request['solver_seconds'],
                                threads=request['threads'], wall_seconds=request['wall_seconds']-(time.monotonic()-began))
    np.savez_compressed(out/'solution.npz', **values)
    record.update(wall_seconds=time.monotonic()-began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  objective_constant_usd=float(arrays['objective_constant_usd']), independent_scientific_audit_required=True)
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    (out/'artifact_sha256.json').write_text(json.dumps({f.name:sha(f) for f in out.iterdir() if f.is_file()}, indent=2)+'\n')
    print(json.dumps(record))
    return 0 if record['status'] in ('OPTIMAL','TIME_LIMIT','MEM_LIMIT') else 2


if __name__ == '__main__':
    raise SystemExit(main())
