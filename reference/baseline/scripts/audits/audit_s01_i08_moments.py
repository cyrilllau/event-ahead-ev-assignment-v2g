#!/usr/bin/env python3
"""Audit saved I08 bootstrap and LP evidence without importing campaign kernels.

Reads raw feature matrices, explicitly recomputes complete-row bootstrap means,
checks the RNG address and both historical array-hash encodings, reconstructs
all saved primal/dual certificates, and solves a separately ordered repair LP.
The audit writes only to a new directory outside the source evidence directory.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform
import warnings

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_file, historical_path, legacy_transfer_manifest, LEGACY_CASE_NAME, source_provenance, verify_scientific_kernel_identity)
from scipy.optimize import OptimizeWarning, linprog


TOL = 1e-8
ARRAY_TOL = 3e-14
LOWER = np.array([0., 0., 0., 0., .91])
UPPER = np.array([1., 1., 1., 1., 1.09])


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, label, tolerance=ARRAY_TOL):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, err_msg=label)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def joint_array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def bootstrap_array_sha(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({'shape': a.shape, 'dtype': str(a.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def explicit_mean(rows):
    # math.fsum is independent of NumPy's axis reduction and preserves exact
    # constants explicitly instead of allowing repeated-addition drift.
    return np.array([rows[0, i] if np.all(rows[:, i] == rows[0, i])
                     else math.fsum(float(v) for v in rows[:, i]) / len(rows)
                     for i in range(5)])


def lp_matrices(features, lower, upper):
    n = len(features)
    return np.zeros(n), np.concatenate((features.T, -features.T)), np.r_[upper, -lower], np.ones((1, n)), np.ones(1)


def audit_certificate(record, c, a, b, eq, rhs, label):
    check(record['success'] and record['solver_status'] == 0 and record['valid_witness'], label + ': not declared solved')
    x = np.asarray(record['witness'], dtype=float)
    check(x.shape == c.shape and np.isfinite(x).all(), label + ': malformed witness')
    slack, equality = b - a @ x, eq @ x - rhs
    violation = max(0., float(-slack.min()), float(np.abs(equality).max()), float(-x.min()))
    check(violation <= TOL, label + ': invalid primal witness')
    residuals = record['primal_residuals']
    close(residuals['inequality_slacks'], slack, label + ': reported inequality slacks')
    close(residuals['equality_residuals'], equality, label + ': reported equality residuals')
    close(residuals['maximum_violation'], violation, label + ': reported primal violation')
    close(residuals['minimum_inequality_slack'], slack.min(), label + ': reported minimum slack')
    close(residuals['minimum_variable'], x.min(), label + ': reported minimum variable')
    check(record['objective'] is not None, label + ': missing objective')
    close(record['objective'], c @ x, label + ': primal objective')
    dual = record['dual_certificate']
    check(dual is not None and dual['certified_optimal_within_tolerance'], label + ': no dual certificate')
    yi = np.asarray(dual['inequality_marginals'], dtype=float)
    ye = np.asarray(dual['equality_marginals'], dtype=float)
    zl = np.asarray(dual['lower_bound_marginals'], dtype=float)
    check(yi.shape == b.shape and ye.shape == rhs.shape and zl.shape == x.shape,
          label + ': wrong dual dimensions')
    check(all(np.isfinite(v).all() for v in (yi, ye, zl)), label + ': nonfinite dual')
    stationarity = c - a.T @ yi - eq.T @ ye - zl
    sign_stationarity = max(0., float(yi.max()), float(-zl.min()), float(np.abs(stationarity).max()))
    primal_objective, dual_objective = float(c @ x), float(b @ yi + rhs @ ye)
    gap = abs(primal_objective - dual_objective)
    complementarity = max(float(np.abs(yi * slack).max()), float(np.abs(zl * x).max()))
    check(max(sign_stationarity, gap, complementarity) <= TOL, label + ': invalid optimality certificate')
    for key, value in [('dual_objective', dual_objective), ('primal_objective', primal_objective),
                       ('absolute_objective_gap', gap), ('stationarity_max_abs', np.abs(stationarity).max()),
                       ('dual_sign_or_stationarity_violation', sign_stationarity),
                       ('complementarity_max_abs', complementarity)]:
        close(dual[key], value, label + ': reported ' + key)
    return {'primal_violation': violation, 'dual_sign_or_stationarity_violation': sign_stationarity,
            'objective_gap': gap, 'complementarity': complementarity}


def independent_repair(features, center, width, scale, fix_zero=False):
    """Independently order delta FIRST and alternate coordinate constraints."""
    n = len(features)
    objective = np.r_[1., np.zeros(n)]
    rows, rhs = [], []
    for i in range(5):
        for sign in (1., -1.):
            rows.append(np.r_[-.5 * scale[i], sign * features[:, i]])
            rhs.append(sign * center[i] + .5 * width[i])
    a, b = np.asarray(rows), np.asarray(rhs)
    eq = np.array([np.r_[0., np.ones(n)]])
    bounds = [(0., 0.) if fix_zero else (0., None)] + [(0., None)] * n
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
        result = linprog(objective, A_ub=a, b_ub=b, A_eq=eq, b_eq=[1.], bounds=bounds,
                         method='highs-ds', options={'threads': 1, 'primal_feasibility_tolerance': TOL,
                                                   'dual_feasibility_tolerance': TOL})
    if result.success:
        violation = max(0., float(np.max(a @ result.x - b)),
                        abs(float(result.x[1:].sum()) - 1.), float(-result.x.min()))
        check(violation <= TOL, 'Independent reference LP returned an invalid witness')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--audit-output', required=True, type=Path)
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.audit_output.resolve()
    check(destination != source and source not in destination.parents,
          'Audit output must be outside the immutable source evidence directory')
    destination.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / 'increment_validation.json').read_text())
    check(summary['status'] in ('PASS_PREFLIGHT', 'PASS_INCREMENT_AUTOMATED'), 'Input increment did not pass automated checks')
    check(not summary['formal_OOS_outcomes_generated'], 'Unexpected formal OOS evidence')
    manifest = json.loads((source / 'artifact_sha256.json').read_text())
    development = summary['upstream_validation']['development']
    check(type(development) is bool, 'Missing explicit development domain')
    is_preflight = summary['status'] == 'PASS_PREFLIGHT'
    records, checked_sources, shared_indices = [], {}, {}

    def verified(name):
        path = source / name
        check(name in manifest, 'Source artifact missing from manifest: ' + name)
        digest = file_sha(path)
        check(digest == manifest[name], 'Source artifact checksum mismatch: ' + name)
        checked_sources[name] = digest
        return path

    verified('increment_validation.json')
    expected_events = [r['event'] for r in summary['targets']]
    check(len(expected_events) == len(set(expected_events)), 'Duplicate summary target')
    check(len(expected_events) == (1 if is_preflight else 8), 'Unexpected target count for preflight/full bank')
    check({p.name[:-len('.moments.json')] for p in source.glob('*.moments.json')} == set(expected_events),
          'Moment artifacts do not match declared targets')
    for target in summary['targets']:
        event = target['event']
        features = {}
        for split, count in [('planning', 8 if is_preflight else 160), ('calibration', 8 if is_preflight else 80)]:
            with np.load(verified(f'{event}.{split}.npz'), allow_pickle=False) as archive:
                features[split] = archive['features']
                scenario_ids = archive['scenario_ids']
            ledger = json.loads(verified(f'{event}.{split}.ledger.json').read_text())
            check(features[split].shape == (count, 5), 'Unexpected feature matrix dimensions')
            check(np.isfinite(features[split]).all(), 'Nonfinite features')
            check(len(set(scenario_ids.tolist())) == count, 'Duplicate scenario IDs')
            check(scenario_ids.tolist() == [r['scenario_id'] for r in ledger['scenarios']], 'Feature/ledger row order differs')
            feature_hash = joint_array_sha(features[split])
            check(feature_hash == ledger['array_payload_hashes']['features'], 'Ledger feature hash mismatch')
            check(feature_hash == summary['array_payload_hashes'][f'{event}/{split}']['features'], 'Summary feature hash mismatch')
            for row, values in zip(ledger['scenarios'], features[split]):
                check(row['target_event'] == event and row['split'] == split and row['development'] is development,
                      'Wrong feature row identity/domain')
                check(row['array_hashes']['features'] == joint_array_sha(values), 'Per-scenario feature hash mismatch')
        planning, calibration = features['planning'], features['calibration']
        task_index = json.loads(verified(f'{event}.task_index.json').read_text())
        lower, upper = LOWER.copy(), UPPER.copy()
        if not task_index['task_ids']:
            upper[3] = 0.
        scale = upper - lower
        check(all(np.all((value >= lower) & (value <= upper)) for value in (planning, calibration)),
              'Features outside true physical domains')
        boot_meta = json.loads(verified(f'{event}.bootstrap.json').read_text())
        with np.load(verified(f'{event}.bootstrap.npz'), allow_pickle=False) as archive:
            indices, means, saved_center, saved_width = (archive[key] for key in ('indices', 'means', 'center', 'halfwidth'))
        check(indices.shape == (2000, len(calibration)) and indices.dtype == np.dtype('int64'), 'Wrong bootstrap shape/dtype')
        root = 20260911 if development else 20260902
        key = (1, 2 if development else 1, 2, 0, 0, 0, 10, 0, 0)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(root, spawn_key=key)))
        expected_indices = rng.integers(0, len(calibration), size=indices.shape, dtype=np.int64)
        check(np.array_equal(indices, expected_indices), 'Bootstrap indices differ from independently addressed RNG')
        identity = boot_meta['stream_identity']
        check(identity['root_seed'] == root and tuple(identity['spawn_key']) == key
              and identity['mechanism_id'] == 10 and identity['bit_generator'] == 'PCG64'
              and identity['split'] == 'calibration' and identity['development'] is development,
              'Bootstrap metadata stream identity mismatch')
        check(identity['stream_id'] == canonical_sha({'root_seed': root, 'spawn_key': key}), 'Bootstrap canonical stream hash mismatch')
        indices_hash = bootstrap_array_sha(indices)
        check(indices_hash == boot_meta['index_sha256'] == target['bootstrap_index_sha256'], 'Bootstrap index hash mismatch')
        sharing_key = (root, indices.shape)
        if sharing_key in shared_indices:
            check(shared_indices[sharing_key] == indices_hash, 'Indices were not shared across targets')
        shared_indices[sharing_key] = indices_hash
        center = explicit_mean(calibration)
        reference_means = np.array([explicit_mean(calibration[row]) for row in indices])
        deviations = np.sort(np.abs(reference_means - center), axis=0)
        width = deviations[math.ceil(.99 * (len(indices) - 1))]
        close(saved_center, center, event + ': calibration center')
        close(means, reference_means, event + ': explicit complete-row bootstrap means')
        close(saved_width, width, event + ': higher .99 quantile')
        constant = np.all(calibration == calibration[0], axis=0)
        check(np.array_equal(saved_center[constant], calibration[0, constant]), 'Constant centers lost exact source values')
        check(np.all(saved_width[constant] == 0), 'Constant observations acquired nonzero bootstrap width')
        moment = json.loads(verified(f'{event}.moments.json').read_text())
        check(moment['status'] == 'PASS_MOMENT_CALIBRATION', 'Saved moment status is not PASS')
        # Use saved centers/widths after independent fsum comparison so that
        # reconstruction of reported floating-point residuals is bit-stable.
        center, width = saved_center, saved_width
        close(moment['center'], center, event + ': unchanged moment center')
        close(moment['original_halfwidth'], width, event + ': original bootstrap width')
        delta, optimum, margin = (float(moment[k]) for k in ('delta', 'optimal_delta', 'numeric_margin_delta'))
        check(all(math.isfinite(v) and v >= 0 for v in (delta, optimum, margin)), 'Invalid repair delta')
        close(delta, optimum + margin, event + ': delta decomposition')
        close(target['delta'], delta, event + ': summary delta')
        repaired = width + delta * scale
        close(moment['repaired_halfwidth'], repaired, event + ': single common base halfwidth repair')
        repair = moment['repair_certificate']
        close(repair['physical_scale'], scale, event + ': physical normalized scale')
        check(repair['center_unchanged'] and repair['common_repaired_base_width_for_all_rhos']
              and repair['minimum_rho'] == .5, 'Wrong repair protocol claims')
        raw_lower, raw_upper = np.maximum(lower, center - .5 * width), np.minimum(upper, center + .5 * width)
        raw = moment['raw_feasibility']
        close(raw['lower'], raw_lower, event + ': raw lower')
        close(raw['upper'], raw_upper, event + ': raw upper')
        certificates = []
        independent_raw = independent_repair(planning, center, width, scale, fix_zero=True)
        if raw['success']:
            check(independent_raw.success and not repair['required'], 'Raw feasibility/repair decision mismatch')
            check(optimum == margin == delta == 0., 'Feasible raw set received unsolicited repair')
            certificates.append(audit_certificate(raw, *lp_matrices(planning, raw_lower, raw_upper), event + ': raw'))
        else:
            check(raw['solver_status'] == independent_raw.status == 2 and repair['required'], 'Infeasibility not independently reproduced')
            n = len(planning)
            c = np.r_[np.zeros(n), 1.]
            a = np.concatenate((np.c_[planning.T, -.5 * scale], np.c_[-planning.T, -.5 * scale]))
            b = np.r_[center + .5 * width, -center + .5 * width]
            eq = np.array([np.r_[np.ones(n), 0.]])
            certificates.append(audit_certificate(repair, c, a, b, eq, np.ones(1), event + ': repair'))
            close(repair['witness'][-1], optimum, event + ': repair optimum witness')
            close(repair['probability_witness'], repair['witness'][:-1], event + ': repair p indexing')
        reference = independent_repair(planning, center, width, scale)
        check(reference.success, 'Independent minimum-delta LP failed')
        close(optimum, reference.fun, event + ': independently minimal delta', TOL)
        if margin:
            p = np.asarray(repair['probability_witness'])
            positive = scale > 0
            needed = max(0., float(np.max((2 * np.abs(planning.T @ p - center)[positive] - width[positive]) / scale[positive], initial=0.)))
            check(margin <= 2 * TOL / scale[positive].min(), 'Numerical margin exceeds frozen tolerance limit')
            expected_addition = float(np.nextafter(max(needed, optimum), np.inf)) - optimum
            close(margin, expected_addition, event + ': measured numerical margin')
            close(repair['numeric_margin_record']['measured_required_delta'], needed, event + ': margin evidence')
        check(set(moment['bounds_by_rho']) == {'0.5', '1.0', '1.5', '2.0'}, 'Wrong rho sweep')
        previous = None
        for rho in (.5, 1., 1.5, 2.):
            bounds = moment['bounds_by_rho'][str(rho)]
            lo, hi = np.maximum(lower, center - rho * repaired), np.minimum(upper, center + rho * repaired)
            close(bounds['lower'], lo, event + ': final lower')
            close(bounds['upper'], hi, event + ': final upper')
            check(bounds['rho'] == rho, 'Rho label mismatch')
            if previous is not None:
                check(np.all(lo <= previous[0]) and np.all(hi >= previous[1]), 'Moment bounds are not nested')
            previous = lo, hi
            certificates.append(audit_certificate(bounds['feasibility'], *lp_matrices(planning, lo, hi), event + f': rho {rho}'))
            p = np.asarray(bounds['feasibility']['witness'])
            close(bounds['feasibility']['implied_feature_means'], planning.T @ p, event + ': implied mean')
        record = {'event': event, 'planning_rows': len(planning), 'calibration_rows': len(calibration),
                  'bootstrap_replications': len(indices), 'index_sha256': indices_hash,
                  'max_bootstrap_mean_difference': float(np.max(np.abs(means - reference_means))),
                  'higher_quantile_order_index': math.ceil(.99 * (len(indices) - 1)),
                  'constant_observation_columns': np.flatnonzero(constant).tolist(),
                  'empty_H': not bool(task_index['task_ids']), 'delta': delta,
                  'independent_optimal_delta': float(reference.fun), 'checked_certificates': len(certificates),
                  'maximum_primal_violation': max(r['primal_violation'] for r in certificates),
                  'maximum_dual_violation': max(r['dual_sign_or_stationarity_violation'] for r in certificates),
                  'maximum_objective_gap': max(r['objective_gap'] for r in certificates)}
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    result = {'status': 'PASS_INDEPENDENT_MOMENT_AUDIT', 'scope': 'preflight' if is_preflight else 'full_development_bank',
              'source_output': str(source), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'targets': records, 'source_artifact_hashes': checked_sources,
              'source_manifest_sha256': file_sha(source / 'artifact_sha256.json'),
              'auditor_sha256': file_sha(Path(__file__).resolve()),
              'independence': 'Only configuration/provenance infrastructure imported; no scientific model imports; explicit fsum row-bootstrap, direct SeedSequence/PCG64, independently ordered delta-first HiGHS-DS reference LP and raw certificate reconstruction',
              'tolerances': {'array_absolute': ARRAY_TOL, 'LP_primal_dual': TOL},
              'formal_OOS_outcomes_generated': False, 'runtime': {'host': platform.node(), 'python': platform.python_version(), 'numpy': np.__version__}}
    result.update(source_provenance())
    with (destination / 'moment_bootstrap_audit.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'status': result['status'], 'targets': len(records), 'audit_output': str(destination)}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
