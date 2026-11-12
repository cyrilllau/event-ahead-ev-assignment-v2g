#!/usr/bin/env python3
"""Replay S03-I01 saved probability witnesses and two LP certificates; never solve.

The formulas below are independent of probability.py/probability_check.py.
Diagnostic q vectors are not physical recourse costs or a C&CG experiment.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import historical_path, sha256, verify_accepted_sources

FEATURES = ('Phi_W', 'Phi_L', 'Phi_R', 'Phi_D', 'Phi_load')
RHOS = (.5, 1., 1.5, 2.)
VECTORS = ('zero_q', 'constant_q', 'affine_q', 'nonlinear_q')
OBJECTIVE_TRANSFORM = {
    'name': 'subtract_minimum_then_scale',
    'scaled_cost': '(q - q_offset_usd) / q_scale_usd',
    'primal_solver_objective': '-scaled_cost @ p',
    'dual_solver_rhs': '-scaled_cost',
    'primal_objective_recovery': 'q_offset_usd * sum(p) - q_scale_usd * raw_objective_scaled',
    'dual_theta0_recovery': 'q_offset_usd + q_scale_usd * raw_theta0',
    'dual_multiplier_recovery': 'q_scale_usd * raw_multiplier',
    'raw_primal_witness': 'primal_solver.raw_solution_scaled',
    'raw_dual_witness': 'dual_solver.raw_solution_scaled',
}
TRANSFORM_SCOPE = 'q_centered_min_v1; original q and scientific gates unchanged'
S01_SHA = 'dc3328d2fdf3fede8813a36493850b574cc21506ef95200926728bf0d06651be'
S02_SHA = '49b9b9cea05e9fd391862629ee1746a90f20024fd8d0cb21053f9b4b95d0cd68'


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def read(path, expected=None):
    p = historical_path(path)
    if expected is not None:
        need(sha256(p) == expected, f'SHA256 mismatch: {p}')
    return json.loads(p.read_text())


def child(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts, f'Unsafe artifact path: {name}')
    q = (root / p).resolve()
    need(root.resolve() in q.parents, f'Artifact escaped evidence root: {name}')
    return q


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    need(a.shape == shape and np.isfinite(a).all(), f'Invalid numeric witness: {label}')
    return a


def close(a, b, label, tolerance=1e-8):
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    need(aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
         and np.allclose(aa, bb, atol=tolerance, rtol=0), label)


def maximum(a):
    return float(np.max(np.asarray(a), initial=0.))


def artifact_manifest(root, expected=None):
    m = read(root / 'artifact_sha256.json', expected)
    for name, digest in m.items():
        need(sha256(child(root, name)) == digest, f'Artifact changed: {name}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    need(actual == set(m) | {'artifact_sha256.json'}, 'Artifact manifest is not exhaustive')
    return len(m)


def release_manifest(root, expected=None):
    rows = read(root / 'RELEASE_MANIFEST.json', expected)['files']
    need(len(rows) == len({r['path'] for r in rows}), 'Duplicate release source identity')
    for row in rows:
        p = child(root, row['path'])
        need(p.stat().st_size == row['size_bytes'] and sha256(p) == row['sha256'], f'Release file changed: {p}')
    return len(rows)


def source_manifest(root, code):
    rows = code['current_source_hashes']
    need(bool(rows), 'Empty source manifest')
    for name, digest in rows.items():
        need(sha256(child(root, name)) == digest, f'Model source changed: {name}')
    return len(rows)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def anchor(source, digest):
    values = list(strings(source))
    need(digest in values, 'Missing externally pinned acceptance digest')
    paths = {v for v in values if v.endswith('/acceptance.json')}
    matches = [p for p in paths if sha256(historical_path(p)) == digest]
    need(len(matches) == 1, 'Missing or ambiguous externally pinned acceptance path')
    return matches[0]


def verify_refs(value):
    """Check every explicit path/SHA pair without rewriting historical paths."""
    count = 0
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            p = historical_path(value['path'])
            need(sha256(p) == value['sha256'], f'Referenced source changed: {p}')
            count += 1
        for key, item in value.items():
            if key.endswith('_path') and key[:-5] + '_sha256' in value:
                p = historical_path(item)
                need(sha256(p) == value[key[:-5] + '_sha256'], f'Referenced source changed: {p}')
                count += 1
            count += verify_refs(item)
    elif isinstance(value, list):
        count += sum(verify_refs(v) for v in value)
    return count


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def prerequisite_inputs(source):
    s01_path = anchor(source, S01_SHA)
    s02_path = anchor(source, S02_SHA)
    s01, s02 = read(s01_path, S01_SHA), read(s02_path, S02_SHA)
    need(s01['status'] == 'PASS' and s01['spec_id'] == 'S01', 'S01 not accepted')
    need(s02['status'] == 'PASS' and s02['spec_id'] == 'S02' and s02['full_S02_pass'], 'S02 not fully accepted')
    pack = historical_path(s01['deliverables'])
    m = read(s01['deliverables_manifest'], s01['deliverables_manifest_sha256'])
    verify_refs(m)
    verify_accepted_sources(s01['source_implementation_snapshot'])
    need(Path(s01['deliverables_manifest']).parent == Path(s01['deliverables']), 'S01 pack/manifest mismatch')
    for name in ('features.csv', 'moment_bounds.csv', 'generator_contract.json'):
        row = m[name]
        need(Path(row['path']) == Path(s01['deliverables']) / name, 'S01 input path not in accepted pack')
        p = historical_path(row['path'])
        need(p.stat().st_size == row['bytes'] and sha256(p) == row['sha256'], f'S01 input changed: {name}')
    gen = read(pack / 'generator_contract.json')
    targets = tuple(gen['target_order'])
    need(len(targets) == len(set(targets)) == 8 and tuple(gen['feature_order']) == FEATURES, 'S01 axes changed')
    need(gen['development_only'] and gen['formal_OOS_locked'] and not gen['formal_outcomes_generated'], 'S01 scope changed')
    features = csv_rows(pack / 'features.csv')
    moments = csv_rows(pack / 'moment_bounds.csv')
    need(len(features) == 1920 and len(moments) == 160, 'S01 source row counts changed')
    inputs = {}
    checked_moments = {}
    for target in targets:
        planning = [r for r in features if r['target_event'] == target and r['split'] == 'planning']
        need(len(planning) == 160 and len({r['scenario_id'] for r in planning}) == 160, 'Planning atoms missing or duplicated')
        ids = [r['scenario_id'] for r in planning]
        need(all('/development/planning/' in v for v in ids), 'Formal/calibration atom in planning support')
        phi = np.array([[float(r[f]) for f in FEATURES] for r in planning])
        need(np.isfinite(phi).all(), 'Nonfinite accepted features')
        previous = None
        for rho in RHOS:
            rows = [r for r in moments if r['target_event'] == target and float(r['rho']) == rho]
            rows.sort(key=lambda r: int(r['feature_index']))
            need([r['feature'] for r in rows] == list(FEATURES), 'Moment feature axes changed')
            for r in rows:
                key = (r['source_moments_path'], r['source_moments_sha256'])
                if key not in checked_moments:
                    checked_moments[key] = read(*key)
                original = checked_moments[key]
                i = int(r['feature_index']); original_bounds = original['bounds_by_rho'][r['rho']]
                need(original['status'] == 'PASS_MOMENT_CALIBRATION' and
                     float(r['lower']) == original_bounds['lower'][i] and float(r['upper']) == original_bounds['upper'][i],
                     'Accepted moment table differs from original calibration JSON')
            lo = np.array([float(r['lower']) for r in rows]); hi = np.array([float(r['upper']) for r in rows])
            need(np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo <= hi), 'Invalid moment intervals')
            if previous is not None:
                need(np.all(lo <= previous[0]) and np.all(hi >= previous[1]), 'Rho sets are not nested')
            previous = lo, hi
            inputs[target, rho] = (phi, lo, hi, ids)
    need(len(checked_moments) == 8, 'Moment source count changed')
    # Historical S02 is validated against its own release, never this new S03 code.
    historical = []
    queue = [(s02_path, S02_SHA)]
    seen = set()
    while queue:
        p, d = queue.pop()
        if p in seen:
            continue
        seen.add(p); card = read(p, d)
        old = Path(card['source_root']); evidence = historical_path(card['evidence'])
        artifacts = artifact_manifest(evidence, card['artifact_manifest_sha256'])
        code = read(card['code_manifest']['path'], card['code_manifest']['sha256'])
        n = source_manifest(old, code)
        ref = card['current_source_manifest']
        need(Path(ref['path']) == old / 'RELEASE_MANIFEST.json', 'Historical release binding differs')
        release_files = release_manifest(old, ref['sha256'])
        if 'validation_sha256' in card:
            read(evidence / 'increment_validation.json', card['validation_sha256'])
        for ap, ad in card['independent_audits'].items():
            need(read(ap, ad)['status'].startswith('PASS_INDEPENDENT'), 'Accepted historical audit did not pass')
        ex = read(evidence / 'execution.json')
        need(sha256(ex['config_path']) == ex['config_sha256'], 'Historical config changed')
        need(sha256(old / 'docs/experiments/runtime_policy.json') == ex['runtime_policy_sha256'], 'Historical policy changed')
        for item in card.get('prerequisite_acceptances', []):
            queue.append((item['path'], item['sha256']))
        historical.append({'acceptance': p, 'artifact_files': artifacts, 'source_files': n, 'release_files': release_files})
    return targets, inputs, historical


def diagnostic_q(phi, name):
    if name == 'zero_q': return np.zeros(len(phi))
    if name == 'constant_q': return np.full(len(phi), 260000.)
    if name == 'affine_q': return 100000. + phi @ np.array([10000., 20000., 30000., 40000., 50000.])
    if name == 'nonlinear_q': return 100000. + 20000.*phi[:, 0]*phi[:, 2] + 70000.*phi[:, 3]**2 + 10000.*phi[:, 4]**2
    raise ValueError(f'Unexpected diagnostic q: {name}')


def raw_lp(saved, c, a, b, ae, be, lower, upper, expected_z):
    """Recompute minimization KKT/dual value from raw SciPy marginals."""
    need(saved['status'] == 0 and saved['success'], 'Raw LP was not optimal')
    n = len(c)
    z = array(saved['raw_solution_scaled'], (n,), 'scaled LP primal')
    close(z, expected_z, 'Scaled LP witness differs from published witness', 1e-10)
    y = array(saved['ineqlin_marginals_scaled'], (len(b),), 'inequality marginals')
    lam = array(saved['eqlin_marginals_scaled'], (len(be),), 'equality marginals')
    lm = array(saved['lower_marginals_scaled'], (n,), 'lower marginals')
    um = array(saved['upper_marginals_scaled'], (n,), 'upper marginals')
    fl, fu = np.isfinite(lower), np.isfinite(upper)
    primal = max(maximum(a @ z - b), maximum(np.abs(ae @ z - be)), maximum(lower-z), maximum(z-upper))
    stationarity = maximum(np.abs(c - a.T @ y - ae.T @ lam - lm - um))
    signs = max(maximum(y), maximum(-lm), maximum(um))
    infinite = max(maximum(np.abs(lm[~fl])), maximum(np.abs(um[~fu])))
    primal_objective = float(c @ z)
    dual_objective = float(b @ y + be @ lam + lower[fl] @ lm[fl] + upper[fu] @ um[fu])
    gap = abs(primal_objective-dual_objective)
    complementarity = max(maximum(np.abs(y*(b-a@z))), maximum(np.abs(lm[fl]*(z[fl]-lower[fl]))), maximum(np.abs(um[fu]*(upper[fu]-z[fu]))))
    need(max(primal, stationarity, signs, infinite) <= 1e-8 and gap <= 1e-7 and complementarity <= 1e-7, 'Raw scaled LP numeric certificate failed')
    close(saved['raw_objective_scaled'], primal_objective, 'Raw LP objective changed', 1e-9)
    need(saved['options']['threads'] == 1, 'LP threads policy changed')
    return {'primal_residual': primal, 'stationarity': stationarity, 'sign_residual': signs,
            'infinite_bound_marginal': infinite, 'primal_objective_scaled': primal_objective,
            'dual_objective_scaled': dual_objective, 'gap_scaled': gap, 'complementarity_scaled': complementarity}


def audit_raw_probability_lp(phi, lo, hi, q, saved):
    """Check the explicit affine objective transform and both untouched LPs.

    Objective translation uses sum(p)=1 and the free dual theta. Certification
    and separation continue to use original USD, not the centered q range.
    This helper supports small handwritten witnesses as well as the real bank.
    """
    phi = np.asarray(phi, dtype=float); q = np.asarray(q, dtype=float)
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    n, f = phi.shape
    need(n > 0 and q.shape == (n,) and lo.shape == hi.shape == (f,)
         and all(np.isfinite(v).all() for v in (phi, lo, hi, q)), 'Invalid raw probability input axes')
    ids = saved['support_ids']; feature_ids = saved['feature_ids']
    need(len(ids) == len(set(ids)) == n and len(feature_ids) == len(set(feature_ids)) == f, 'Invalid raw probability identities')
    need(saved['q_sha256'] == canonical({'support_ids': ids, 'q': q.tolist()}), 'Original q identity changed')
    need(saved['ambiguity_sha256'] == canonical({'support_ids': ids, 'feature_ids': feature_ids,
         'features': phi.tolist(), 'lower': lo.tolist(), 'upper': hi.tolist(),
         'feature_orientation': 'scenario_by_feature', 'dual_convention': 'alpha_lower_beta_upper'}), 'Original ambiguity identity changed')
    offset = float(np.min(q)); centered = q-offset
    need(np.isfinite(centered).all(), 'Nonfinite centered objective')
    scale = max(1., float(np.max(np.abs(centered))))
    original_scale = max(1., float(np.max(np.abs(q))))
    cert = saved['certificate']
    need(cert['objective_transform'] == OBJECTIVE_TRANSFORM, 'Objective transform schema changed')
    for key, expected in (('q_offset_usd', offset), ('q_scale_usd', scale),
                          ('certification_scale_usd', original_scale), ('separation_scale_usd', original_scale)):
        need(cert[key] == expected, f'Recorded {key} changed')
    close(cert['separation_tolerance_usd'], 1e-4+1e-8*original_scale, 'Original-q separation tolerance changed', 1e-12)
    p = array(saved['probabilities'], (n,), 'raw published probability')
    theta = float(saved['theta0']); need(np.isfinite(theta), 'Nonfinite restored theta')
    alpha = array(saved['alpha'], (f,), 'restored alpha')
    beta = array(saved['beta'], (f,), 'restored beta')
    raw_p = array(saved['primal_solver']['raw_solution_scaled'], (n,), 'untouched primal vector')
    raw_z = array(saved['dual_solver']['raw_solution_scaled'], (1+2*f,), 'untouched dual vector')
    need(np.array_equal(raw_p, p), 'Published probabilities differ from raw LP witness')
    close(theta, offset+scale*raw_z[0], 'Free theta restoration changed', 1e-8)
    close(alpha, scale*raw_z[1:1+f], 'Alpha restoration changed', 1e-8)
    close(beta, scale*raw_z[1+f:], 'Beta restoration changed', 1e-8)
    objective = centered/scale
    primal_report = raw_lp(saved['primal_solver'], -objective, np.vstack((phi.T, -phi.T)), np.r_[hi, -lo],
                          np.ones((1,n)), np.ones(1), np.zeros(n), np.full(n,np.inf), raw_p)
    dual_report = raw_lp(saved['dual_solver'], np.r_[1., -lo, hi], np.column_stack((-np.ones(n), phi, -phi)), -objective,
                        np.empty((0,1+2*f)), np.empty(0), np.r_[-np.inf,np.zeros(2*f)], np.full(1+2*f,np.inf), raw_z)
    # Keep the actual normalization residual; replacing sum(p) by 1 here would
    # conceal its dollar contribution under a large constant objective offset.
    primal_recovered = offset*float(p.sum())-scale*float(saved['primal_solver']['raw_objective_scaled'])
    dual_recovered = offset+scale*float(saved['dual_solver']['raw_objective_scaled'])
    close(primal_recovered, float(q@p), 'Primal objective recovery changed', 1e-8)
    close(dual_recovered, float(theta-lo@alpha+hi@beta), 'Dual objective recovery changed', 1e-8)
    return {'q_offset_usd': offset, 'q_scale_usd': scale, 'certification_scale_usd': original_scale,
            'raw_primal_LP': primal_report, 'raw_dual_LP': dual_report,
            'primal_objective_recovered_usd': primal_recovered, 'dual_objective_recovered_usd': dual_recovered}


def audit_case(folder, record, expected):
    phi, lo, hi, ids = expected
    with np.load(folder / 'inputs.npz', allow_pickle=False) as archive:
        need(set(archive.files) == {'features', 'lower', 'upper', 'q', 'support_ids', 'feature_ids'}, 'Unexpected probability NPZ schema')
        for name, value in (('features', phi), ('lower', lo), ('upper', hi)):
            need(np.array_equal(archive[name], value), f'Accepted input changed: {name}')
        need(archive['support_ids'].tolist() == ids and archive['feature_ids'].tolist() == list(FEATURES), 'Input IDs/order changed')
        q = np.array(archive['q'], dtype=float)
    close(q, diagnostic_q(phi, record['vector']), 'Diagnostic cost definition changed', 1e-9)
    need(q.shape == (160,) and np.isfinite(q).all(), 'Invalid diagnostic q axis')
    ambiguity = canonical({'support_ids': ids, 'feature_ids': list(FEATURES), 'features': phi.tolist(),
                           'lower': lo.tolist(), 'upper': hi.tolist(), 'feature_orientation': 'scenario_by_feature',
                           'dual_convention': 'alpha_lower_beta_upper'})
    qhash = canonical({'support_ids': ids, 'q': q.tolist()})
    saved = read(folder / 'solution.json')
    for obj in (record, saved):
        need(obj['ambiguity_sha256'] == ambiguity and obj['q_sha256'] == qhash, 'Probability semantic hash changed')
    need(saved['support_ids'] == ids and saved['feature_ids'] == list(FEATURES), 'Saved witness identity changed')
    need(saved['status'] == 'OPTIMAL' and saved['certified_optimal'], 'Missing certified optimum')
    for key in ('probabilities_clipped_or_renormalized', 'moment_bounds_repaired', 'raw_RMP_dual_replaced', 'phase_I_initialization_in_scope'):
        need(saved['certificate'][key] is False, f'Primitive scope or raw-witness semantics changed: {key}')
    need(saved['certificate']['feature_orientation'] == 'scenario_by_feature', 'Feature orientation changed')
    p = array(saved['probabilities'], (160,), 'probabilities')
    alpha, beta = array(saved['alpha'], (5,), 'alpha'), array(saved['beta'], (5,), 'beta')
    theta = float(saved['theta0']); need(np.isfinite(theta), 'Nonfinite free theta')
    means = phi.T @ p
    lhs = theta + phi @ (beta-alpha)
    raw = {'probability_nonnegative': maximum(-p), 'probability_normalization': abs(float(p.sum())-1.),
           'moment_lower': maximum(lo-means), 'moment_upper': maximum(means-hi),
           'alpha_nonnegative': maximum(-alpha), 'beta_nonnegative': maximum(-beta), 'dual_full_support_usd': maximum(q-lhs)}
    scales = np.maximum(1., np.maximum(np.maximum(abs(lo), abs(hi)), abs(phi).T @ abs(p)))
    dualscale = np.maximum(1., np.maximum(abs(q), abs(theta) + abs(phi) @ (abs(alpha)+abs(beta))))
    normalized = max(raw['probability_nonnegative'], raw['probability_normalization']/max(1., float(abs(p).sum())),
                     maximum((lo-means)/scales), maximum((means-hi)/scales),
                     maximum(-alpha/np.maximum(1.,abs(alpha))), maximum(-beta/np.maximum(1.,abs(beta))), maximum((q-lhs)/dualscale))
    need(max(raw.values()) <= 1e-8 and normalized <= 1e-8, 'Original-unit or normalized primal/dual probability gate failed')
    primal, dual = float(q@p), float(theta-lo@alpha+hi@beta)
    tolerance = 1e-4 + 1e-7*max(1., abs(primal), abs(dual))
    gap = abs(dual-primal)
    comp = max(maximum(abs(p*(lhs-q))), maximum(abs(alpha*(means-lo))), maximum(abs(beta*(hi-means))))
    need(gap <= tolerance and comp <= tolerance, 'Dollar strong duality or complementary slackness failed')
    for name, value in (('primal_objective_usd', primal), ('dual_objective_usd', dual),
                        ('lower_bound_usd', primal), ('upper_bound_usd', dual)):
        close(saved[name], value, f'Reported {name} changed', tolerance)
    need(primal >= float(q.min())-tolerance and dual <= float(q.max())+tolerance, 'Probability optimum outside scenario cost hull')
    uniform_means = phi.mean(axis=0)
    uniform_feasible = bool(np.all(uniform_means >= lo-1e-8) and np.all(uniform_means <= hi+1e-8))
    if uniform_feasible:
        need(primal >= float(q.mean())-tolerance, 'SAA <= DRO fails despite a feasible uniform witness')
    if record['vector'] in ('zero_q', 'constant_q'):
        expected_value = 0. if record['vector'] == 'zero_q' else 260000.
        close(primal, expected_value, 'Analytic constant-q primal optimum failed', tolerance)
        close(dual, expected_value, 'Analytic constant-q dual optimum failed', tolerance)
    raw_certificates = audit_raw_probability_lp(phi, lo, hi, q, saved)
    checked = read(folder / 'physical_probability_check.json')
    need(checked['passed'] and checked['primal_feasible'] and checked['dual_feasible'] and checked['gap_passed'], 'Saved independent checker did not pass')
    for name, value in (('primal_usd', primal), ('dual_usd', dual), ('absolute_gap_usd', gap)):
        close(checked['objective_components'][name], value, f'Saved checker {name} differs', tolerance)
    sep = 1e-4 + 1e-8*max(1., float(np.max(np.abs(q))))
    need(checked['separation']['support_ids'] == ids and checked['separation']['passed'], 'Separation support incomplete')
    close(checked['separation']['tolerance_usd'], sep, 'Separation tolerance changed', 1e-12)
    close(checked['separation']['max_positive_residual_usd'], maximum(q-lhs), 'Separation maximum changed', 1e-8)
    close(checked['separation']['residuals_usd'], q-lhs, 'Separation vector changed', 1e-8)
    return {'target': record['target'], 'rho': float(record['rho']), 'vector': record['vector'],
            'relative_path': record['relative_path'], 'ambiguity_sha256': ambiguity, 'q_sha256': qhash,
            'support_count': 160, 'raw_positive_support_count': int(np.count_nonzero(p>0)),
            'thresholded_positive_support_count': int(np.count_nonzero(p>1e-8)),
            'primal_usd': primal, 'dual_usd': dual, 'absolute_gap_usd': gap, 'complementarity_usd': comp,
            'raw_residuals': raw, 'max_normalized_violation': normalized, 'separation_tolerance_usd': sep,
            'uniform_feasible': uniform_feasible, 'uniform_cost_usd': float(q.mean()),
            'aro_cost_usd': float(q.max()), **raw_certificates,
            'probabilities': p.tolist(), 'support_ids': ids}


def audit_tables(root, reports):
    summaries = csv_rows(root / 'probability_duality_audit.csv')
    probabilities = csv_rows(root / 'worst_distribution.csv')
    need(len(summaries) == 128 and len(probabilities) == 20480, 'Diagnostic CSV row coverage changed')
    index = {r['relative_path']: r for r in reports}
    need(len(index) == 128 and {r['relative_path'] for r in summaries} == set(index), 'Summary CSV case coverage changed')
    def identity(row, result):
        need(row['target'] == result['target'] and float(row['rho']) == result['rho'] and row['vector'] == result['vector']
             and row['ambiguity_sha256'] == result['ambiguity_sha256'] and row['q_sha256'] == result['q_sha256'], 'CSV identity changed')
    for row in summaries:
        r = index[row['relative_path']]; identity(row, r)
        need(row['status'] == 'OPTIMAL' and row['certified_optimal'] == row['independent_passed'] == 'True', 'CSV certification changed')
        need(row['empirical_law_feasible'] == str(r['uniform_feasible']), 'CSV empirical feasibility changed')
        for key, other in (('primal_objective_usd','primal_usd'), ('dual_objective_usd','dual_usd'),
                           ('gap_usd','absolute_gap_usd'), ('uniform_expectation_usd','uniform_cost_usd'), ('support_maximum_usd','aro_cost_usd'),
                           ('q_offset_usd','q_offset_usd'), ('q_scale_usd','q_scale_usd'), ('certification_scale_usd','certification_scale_usd')):
            close(float(row[key]), r[other], f'CSV {key} changed', 1e-8)
        need(row['scope'] == 'algebraic diagnostic q; not recourse', 'CSV scope changed')
    grouped = {key: [] for key in index}
    for row in probabilities:
        need(row['relative_path'] in index, 'Unknown distribution case')
        grouped[row['relative_path']].append(row)
    for key, rows in grouped.items():
        r = index[key]
        need([row['scenario_id'] for row in rows] == r['support_ids'], 'Distribution CSV reordered or omitted whole atoms')
        for row in rows: identity(row, r)
        close([float(row['probability']) for row in rows], r['probabilities'], 'Distribution CSV clipped, normalized or changed p', 0.)
    return {'summary_rows': len(summaries), 'complete_distribution_rows': len(probabilities)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); root = args.evidence.resolve(); output = args.output.resolve()
    need(not output.exists() and output != root and root not in output.parents, 'Audit output must be a new file outside run evidence')
    artifact_count = artifact_manifest(root)
    paths = get_paths(); execution = read(root / 'execution.json')
    need(execution['paths'] == paths.as_dict(), 'Audit configuration differs from execution')
    need(Path(execution['paths']['project_root']).resolve() == paths.project_root.resolve(), 'Audit release differs from execution')
    need(execution['config_path'] and sha256(execution['config_path']) == execution['config_sha256'], 'Execution config changed')
    need(execution['probability_objective_transform'] == TRANSFORM_SCOPE, 'Execution objective transform changed')
    need(sha256(paths.spec_root / 'runtime_policy.json') == execution['runtime_policy_sha256'], 'Runtime policy changed')
    source_count = source_manifest(paths.project_root, read(root / 'model_code_manifest.json'))
    release_count = release_manifest(paths.project_root)
    log = (root / 'tests.stderr.log').read_text()
    match = re.search(r'Ran (\d+) tests? in', log)
    need(match and int(match[1]) >= 230 and log.rstrip().endswith('OK')
         and not re.search(r'skipp?ed|expected failure|unexpected success', log, re.I), '230+ tests must pass without skips')
    for name in ('test_probability.', 'test_probability_check.', 'test_probability_audit.', 'test_recourse.', 'test_s02_closure.'):
        need(name in log, f'Missing test evidence: {name}')
    validation = read(root / 'increment_validation.json')
    required = {'increment': 'S03-I01', 'status': 'PASS_INCREMENT_AUTOMATED', 'full_S03_pass': False,
                'scientific_acceptance_pending': True, 'targets': 8, 'rho_values': list(RHOS),
                'diagnostic_vectors': 4, 'probability_problems': 128, 'primal_dual_LP_pairs': 128,
                'support_per_problem': 160, 'feature_count': 5, 'probability_rows': 20480,
                'new_scenarios_generated': 0, 'moments_changed': False,
                'physical_recourse_evaluated': False, 'formal_OOS_outcomes_generated': False,
                'probability_objective_transform': TRANSFORM_SCOPE}
    for name, value in required.items(): need(validation[name] == value, f'Incorrect increment boundary: {name}')
    need(validation['tests'] == int(match[1]), 'Test count differs from actual log')
    source = read(root / 'sources.json'); reference_count = verify_refs(source)
    targets, inputs, history = prerequisite_inputs(source)
    records = read(root / 'records.json')
    expected = {(t, r, v) for t in targets for r in RHOS for v in VECTORS}
    actual = [(r['target'], float(r['rho']), r['vector']) for r in records]
    need(len(records) == 128 and len(set(actual)) == 128 and set(actual) == expected, '128 probability problems incomplete or duplicated')
    need(len({r['relative_path'] for r in records}) == 128, 'Duplicate probability evidence path')
    reports = []
    for record in records:
        folder = child(root, record['relative_path'])
        expected_path = f"cases/{record['target']}/rho_{float(record['rho'])}/{record['vector']}"
        need(record['relative_path'] == expected_path, 'Noncanonical evidence identity')
        reports.append(audit_case(folder, record, inputs[record['target'], float(record['rho'])]))
    for target in targets:
        for name in VECTORS:
            rows = sorted((r for r in reports if r['target'] == target and r['vector'] == name), key=lambda r: r['rho'])
            for earlier, later in zip(rows, rows[1:]):
                tolerance = 1e-4 + 1e-7*max(1., abs(earlier['primal_usd']), abs(later['primal_usd']))
                need(later['primal_usd'] >= earlier['primal_usd']-tolerance, 'Worst expectation decreased under nested rho')
    sweep = read(root / 'rho_sweep_checks.json')
    need(len(sweep) == 8 and [r['target'] for r in sweep] == list(targets)
         and all(r['rho_order'] == list(RHOS) and r['nested_expectations_checked'] for r in sweep), 'Rho sweep summary differs')
    table_counts = audit_tables(root, reports)
    # Full probabilities already have an audited lossless CSV; avoid duplicating them in the audit report.
    for row in reports:
        row.pop('probabilities'); row.pop('support_ids')
    result = {'status': 'PASS_INDEPENDENT_S03_I01_AUDIT', 'increment': 'S03-I01', 'full_S03_pass': False,
              'scope': 'Independent direct primal/dual formulas, raw minimization LP marginal certificates, accepted CSV bindings; no solve or draw',
              'diagnostic_q_is_physical_recourse': False, 'new_scenarios_generated': 0,
              'tests': int(match[1]), 'artifact_files': artifact_count, 'current_source_files': source_count,
              'current_release_files': release_count, 'verified_source_references': reference_count,
              'historical_S02_chain': history, 'target_count': 8, 'probability_problems': 128,
              'analytic_zero_cases': 32, 'analytic_constant_cases': 32, 'raw_LP_certificates': 256,
              'tables': table_counts, 'reports': reports,
              'max_original_unit_violation': max(max(r['raw_residuals'].values()) for r in reports),
              'max_normalized_violation': max(r['max_normalized_violation'] for r in reports),
              'max_absolute_gap_usd': max(r['absolute_gap_usd'] for r in reports),
              'unit_test_evidence_scope': 'Negative costs/free theta, lower/upper signs, infeasible sets, identity and missing-witness behavior are actual unit-test evidence; no saved golden-toy solver pack is claimed',
              'artifact_manifest_sha256': sha256(root / 'artifact_sha256.json'), 'audit_source_sha256': sha256(__file__),
              'remaining_gate': 'Independent scientific acceptance of I01; phase-I, extensive model, C&CG and complete S03 remain unaccepted'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(output), 'probability_problems': 128,
                      'raw_LP_certificates': 256, 'max_absolute_gap_usd': result['max_absolute_gap_usd']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
