"""Bounded IPM diagnostic of a saved RMP's continuous relaxation.

This produces separate diagnostic evidence. It never edits C&CG checkpoints or
claims an LP solution is an integer plan or an accepted campaign lower bound.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import resource
import time

import numpy as np
from scipy.optimize._highspy import _core as h
from scipy.sparse import csc_matrix, load_npz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certificate(matrix, arrays, x, row_dual, col_dual):
    """Check bounded-row LP primal/dual signs, stationarity and objective gap."""
    a = csc_matrix(matrix)
    x, y, z = (np.asarray(v, dtype=float) for v in (x, row_dual, col_dual))
    c, lo, hi, rl, ru = (np.asarray(arrays[k]) for k in
                         ('c', 'lower_bounds', 'upper_bounds', 'row_lower', 'row_upper'))
    if x.shape != c.shape or z.shape != c.shape or y.shape != rl.shape:
        raise ValueError('Certificate axis mismatch')
    if not all(np.isfinite(v).all() for v in (x, y, z)):
        raise ValueError('Nonfinite certificate')
    activity = a @ x
    def maximum(v):
        return max(0., float(np.max(v, initial=0.)))
    def scaled_violation(v, lower, upper):
        finite_scale = np.maximum(np.where(np.isfinite(lower), abs(lower), 0.),
                                  np.where(np.isfinite(upper), abs(upper), 0.))
        return maximum(np.maximum(lower-v, v-upper)/np.maximum(1., finite_scale))
    def support(dual, lower, upper):
        wrong = max(maximum(dual[~np.isfinite(lower)]), maximum(-dual[~np.isfinite(upper)]))
        pos = (dual > 0) & np.isfinite(lower)
        neg = (dual < 0) & np.isfinite(upper)
        return float(dual[pos] @ lower[pos] + dual[neg] @ upper[neg]), wrong
    rd, re = support(y, rl, ru)
    cd, ce = support(z, lo, hi)
    primal = float(c @ x)
    dual = rd + cd
    gap = abs(primal-dual)
    stationarity = maximum(abs(c-a.T@y-z)/np.maximum(1., abs(c)))
    feasibility = max(scaled_violation(activity, rl, ru), scaled_violation(x, lo, hi))
    tolerance = 1e-4 + 1e-7*max(1., abs(primal), abs(dual))
    return dict(passed=feasibility <= 1e-8 and stationarity <= 1e-8 and
                max(re, ce) <= 1e-8 and gap <= tolerance,
                scaled_primal_residual=feasibility, scaled_stationarity=stationarity,
                dual_sign_violation=max(re, ce), primal_objective=primal,
                dual_objective=dual, gap_usd=gap, objective_tolerance_usd=tolerance)


def solve_relaxation(matrix, arrays, time_limit, log_path):
    a = csc_matrix(matrix); a.sort_indices()
    solver = h._Highs()
    options = dict(solver='ipm', run_crossover='on', presolve='on', threads=1,
                   random_seed=0, time_limit=float(time_limit),
                   primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8,
                   ipm_optimality_tolerance=1e-9, output_flag=True,
                   log_to_console=False, log_file=str(log_path))
    if Path(log_path).exists():
        raise ValueError('Refusing to overwrite diagnostic log')
    for name, value in options.items():
        if solver.setOptionValue(name, value) == h.HighsStatus.kError:
            raise ValueError('Rejected option '+name)
    lp = h.HighsLp(); lp.num_col_ = a.shape[1]; lp.num_row_ = a.shape[0]
    lp.col_cost_ = arrays['c']; lp.col_lower_ = arrays['lower_bounds']; lp.col_upper_ = arrays['upper_bounds']
    lp.row_lower_ = arrays['row_lower']; lp.row_upper_ = arrays['row_upper']
    # Explicit continuous relaxation; original integrality is retained in inputs.
    lp.integrality_ = []
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = a.shape[1]; lp.a_matrix_.num_row_ = a.shape[0]
    lp.a_matrix_.start_ = a.indptr; lp.a_matrix_.index_ = a.indices; lp.a_matrix_.value_ = a.data
    if solver.passModel(lp) == h.HighsStatus.kError:
        raise ValueError('Rejected saved relaxation')
    run = solver.run(); status = solver.getModelStatus(); info = solver.getInfo(); sol = solver.getSolution()
    values = dict(x=np.asarray(sol.col_value), row_dual=np.asarray(sol.row_dual), col_dual=np.asarray(sol.col_dual))
    valid = bool(info.valid and sol.value_valid and sol.dual_valid and
                 info.primal_solution_status == 2 and info.dual_solution_status == 2)
    checked = certificate(a, arrays, **values) if valid else None
    basis = solver.getBasis()
    values.update(basis_col_status=np.array([int(v) for v in basis.col_status]),
                  basis_row_status=np.array([int(v) for v in basis.row_status]))
    record = dict(status=solver.modelStatusToString(status), raw_model_status=int(status),
                  run_status=int(run), highs_version=solver.version(), effective_options=options,
                  info_valid=bool(info.valid), primal_solution_status=int(info.primal_solution_status),
                  dual_solution_status=int(info.dual_solution_status), value_valid=bool(sol.value_valid),
                  dual_valid=bool(sol.dual_valid), basis_valid=bool(basis.valid),
                  solve_seconds=solver.getRunTime(), ipm_iterations=int(info.ipm_iteration_count),
                  simplex_iterations=int(info.simplex_iteration_count), crossover_iterations=int(info.crossover_iteration_count),
                  certificate=checked, accepted_campaign_lower_bound=False, integer_plan=False)
    record['certified_optimal_relaxation'] = bool(status == h.HighsModelStatus.kOptimal and
        run != h.HighsStatus.kError and checked is not None and checked['passed'])
    return record, values


def main():
    began = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True)
    args = ap.parse_args(); request = json.loads(args.request.read_text())
    out = Path(request['output']); out.mkdir(parents=True, exist_ok=False)
    for item in request['references'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Diagnostic input changed: '+item['path'])
    project = Path(request['project'])
    for item in json.loads((project/'RELEASE_MANIFEST.json').read_text())['files']:
        if sha(project/item['path']) != item['sha256']:
            raise ValueError('Release source changed')
    refs = request['references']
    matrix = load_npz(refs['matrix']['path']).tocsr()
    with np.load(refs['model_arrays']['path'], allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    raw = json.loads(Path(refs['rmp_solver']['path']).read_text())
    proof = raw['raw_solver']['zero_reduction']
    spec = importlib.util.spec_from_file_location('zero_audit', project/'scripts/audits/audit_zero_reduction.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    proof_audit = audit.audit_zero_reduction(matrix, arrays['row_lower'], arrays['row_upper'],
                                           arrays['lower_bounds'], arrays['upper_bounds'], proof)
    rows = np.asarray(proof['retained_rows']); cols = np.asarray(proof['retained_columns'])
    reduced = matrix[rows][:, cols].tocsc()
    reduced_arrays = {k: arrays[k][cols] for k in ('c', 'lower_bounds', 'upper_bounds', 'integrality')}
    reduced_arrays.update({k: arrays[k][rows] for k in ('row_lower', 'row_upper')})
    load_seconds = time.monotonic()-began
    remaining = min(float(request['solver_seconds']), float(request['wall_seconds'])-load_seconds-15.)
    if remaining <= 0:
        raise TimeoutError('Input checks exhausted diagnostic budget')
    (out/'execution.json').write_text(json.dumps(dict(request=request, request_sha256=sha(args.request),
        script_sha256=sha(__file__), load_and_proof_seconds=load_seconds, proof_audit=proof_audit,
        original_shape=list(matrix.shape), reduced_shape=list(reduced.shape),
        integrality_relaxed_for_diagnosis_only=True, OOS=False), indent=2)+'\n')
    record, values = solve_relaxation(reduced, reduced_arrays, remaining, out/'ipm.log')
    np.savez_compressed(out/'lp_solution.npz', **values)
    if record['value_valid']:
        original = np.zeros(matrix.shape[1]); original[cols] = values['x']
        activity = matrix @ original
        record['original_relaxed_max_violation'] = max(0., float(np.max(np.maximum(
            arrays['row_lower']-activity, activity-arrays['row_upper']))), float(np.max(np.maximum(
            arrays['lower_bounds']-original, original-arrays['upper_bounds']))))
        ints = original[arrays['integrality'] == 1]
        record['max_fractionality'] = float(np.max(abs(ints-np.rint(ints)), initial=0))
        record['fractional_binary_count'] = int(np.sum(abs(ints-np.rint(ints)) > 1e-8))
        np.savez_compressed(out/'original_relaxed_vector.npz', vector=original)
    record.update(wall_seconds=time.monotonic()-began,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  original_objective_constant_usd=float(arrays['objective_constant_usd']),
                  full_S03_pass=False, no_ccg_checkpoint_modified=True)
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    manifest = {f.name: sha(f) for f in out.iterdir() if f.is_file()}
    (out/'artifact_sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(record))
    return 0 if record['status'] in ('Optimal', 'Time limit reached', 'Iteration limit reached') else 2


if __name__ == '__main__':
    raise SystemExit(main())
