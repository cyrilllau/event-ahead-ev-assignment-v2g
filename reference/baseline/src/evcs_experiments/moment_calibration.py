"""Joint complete-scenario bootstrap and certified moment-set repair.

SciPy is imported only by LP calls. Mechanism 10 is a new aggregate calibration
bootstrap address; previously frozen mechanisms 1..9 are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping
import warnings

import numpy as np

from .case_inputs import InputValidationError
from .random_streams import Stream, SCHEMA, DEVELOPMENT_ROOT, ROOTS

BOOTSTRAP_MECHANISM_ID = 10
PHYSICAL_LOWER = (0., 0., 0., 0., .91)
PHYSICAL_UPPER = (1., 1., 1., 1., 1.09)


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _features(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 5 or result.shape[0] == 0 or not np.isfinite(result).all():
        raise InputValidationError(f'{name} must be a nonempty finite (N,5) feature matrix')
    return result


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise InputValidationError(f'{name} must be a finite five-vector')
    return result


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps({'shape': array.shape, 'dtype': str(array.dtype)}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BootstrapMoments:
    center: np.ndarray
    halfwidth: np.ndarray
    bootstrap_means: np.ndarray
    indices: np.ndarray
    index_sha256: str
    stream_identity: Mapping
    replications: int
    alpha: float
    quantile_probability: float

    def as_dict(self):
        return {'center': self.center.tolist(), 'halfwidth': self.halfwidth.tolist(),
                'replications': self.replications, 'alpha': self.alpha,
                'quantile_probability': self.quantile_probability, 'quantile_method': 'higher',
                'index_sha256': self.index_sha256, 'stream_identity': dict(self.stream_identity),
                'bootstrap_means_sha256': _array_hash(self.bootstrap_means),
                'calibration_rows': self.indices.shape[1]}


def bootstrap_moments(calibration_features, *, development=True, replications=2000, alpha=.05):
    values = _features(calibration_features, 'Calibration features')
    if np.any(values < np.asarray(PHYSICAL_LOWER)) or np.any(values > np.asarray(PHYSICAL_UPPER)):
        raise InputValidationError('Calibration scenario features lie outside physical feature bounds')
    if type(development) is not bool or type(replications) is not int or replications <= 0:
        raise InputValidationError('Expected explicit development flag and positive integer replications')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise InputValidationError('Confidence alpha must lie in (0,1)')
    identity = Stream(DEVELOPMENT_ROOT if development else ROOTS['calibration'],
                      (SCHEMA, 2 if development else 1, 2, 0, 0, 0, BOOTSTRAP_MECHANISM_ID, 0, 0))
    indices = identity.generator().integers(0, len(values), size=(replications, len(values)), dtype=np.int64)
    center = values.mean(axis=0)
    means = values[indices].mean(axis=1)
    # Exact constant observations have an exact constant empirical law. Avoid
    # repeated-summation drift (e.g. 80 copies of 1.09 averaging above 1.09),
    # which could falsely violate a physical boundary or create a halfwidth.
    # This recognizes equality in already validated raw data; it never clips
    # varying observations or an out-of-domain input.
    constants = np.all(values == values[0], axis=0)
    center[constants] = values[0, constants]
    means[:, constants] = values[0, constants]
    probability = 1 - alpha / 5
    halfwidth = np.quantile(np.abs(means - center), probability, axis=0, method='higher')
    stream_identity = {**identity.as_dict(), 'mechanism_id': BOOTSTRAP_MECHANISM_ID,
                       'mechanism': 'calibration_bootstrap', 'split': 'calibration',
                       'development': development,
                       'policy': 'Resample whole calibration rows; indices shared across targets/methods/settings'}
    return BootstrapMoments(_readonly(center), _readonly(halfwidth), _readonly(means),
                            _readonly(indices, np.int64), _array_hash(indices), MappingProxyType(stream_identity),
                            replications, float(alpha), float(probability))


def _lp(c, a_ub, b_ub, a_eq, b_eq, tolerance):
    from scipy.optimize import OptimizeWarning, linprog

    options = {'primal_feasibility_tolerance': tolerance, 'dual_feasibility_tolerance': tolerance,
               'ipm_optimality_tolerance': tolerance, 'threads': 1}
    # SciPy forwards the recognized-by-HiGHS threads setting while emitting an
    # OptimizeWarning that it is not one of SciPy's own option names.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                         bounds=[(0., None)] * len(c), method='highs', options=options)
    record = {'solver': 'scipy.optimize.linprog/HiGHS', 'solver_status': int(result.status),
              'message': str(result.message), 'success': bool(result.success),
              'iterations': int(result.nit), 'objective': None if result.fun is None else float(result.fun),
              'solver_options': options, 'witness': None, 'valid_witness': False,
              'primal_residuals': None, 'dual_certificate': None,
              'infeasibility_certificate_available': False}
    if result.x is None:
        return record
    x = np.asarray(result.x, dtype=float)
    slack, equality = b_ub - a_ub @ x, a_eq @ x - b_eq
    primal_violation = max(float(np.max(-slack, initial=0)),
                          float(np.max(np.abs(equality), initial=0)), float(np.max(-x, initial=0)))
    record.update({'witness': x.tolist(),
                   'valid_witness': bool(np.isfinite(x).all() and primal_violation <= tolerance),
                   'primal_residuals': {
                       'maximum_violation': primal_violation,
                       'minimum_inequality_slack': float(np.min(slack)),
                       'maximum_equality_residual': float(np.max(np.abs(equality), initial=0)),
                       'minimum_variable': float(np.min(x)),
                       'inequality_slacks': slack.tolist(), 'equality_residuals': equality.tolist()}})
    if result.success:
        yi = np.asarray(result.ineqlin.marginals, dtype=float)
        ye = np.asarray(result.eqlin.marginals, dtype=float)
        zl = np.asarray(result.lower.marginals, dtype=float)
        stationarity = c - a_ub.T @ yi - a_eq.T @ ye - zl
        primal_objective, dual_objective = float(c @ x), float(b_ub @ yi + b_eq @ ye)
        dual_violation = max(float(np.max(yi, initial=0)), float(np.max(-zl, initial=0)),
                             float(np.max(np.abs(stationarity), initial=0)))
        complementarity = max(float(np.max(np.abs(yi * slack), initial=0)),
                              float(np.max(np.abs(zl * x), initial=0)))
        record['dual_certificate'] = {
            'inequality_marginals': yi.tolist(), 'equality_marginals': ye.tolist(),
            'lower_bound_marginals': zl.tolist(), 'dual_objective': dual_objective,
            'primal_objective': primal_objective, 'absolute_objective_gap': abs(primal_objective - dual_objective),
            'stationarity_max_abs': float(np.max(np.abs(stationarity), initial=0)),
            'dual_sign_or_stationarity_violation': dual_violation,
            'complementarity_max_abs': complementarity,
            'certified_optimal_within_tolerance': bool(record['valid_witness'] and dual_violation <= tolerance
                                                      and abs(primal_objective - dual_objective) <= tolerance
                                                      and complementarity <= tolerance)}
    return record


def _bounds(center, width, rho, lower, upper):
    return np.maximum(lower, center - rho * width), np.minimum(upper, center + rho * width)


def _feasibility(features, lower, upper, tolerance):
    matrix = features.T
    record = _lp(np.zeros(len(features)), np.vstack([matrix, -matrix]),
                 np.concatenate([upper, -lower]), np.ones((1, len(features))), np.ones(1), tolerance)
    if record['witness'] is not None:
        p = np.asarray(record['witness'])
        record['primal_residuals']['sum_residual'] = abs(float(p.sum()) - 1.)
        record['primal_residuals']['min_probability'] = float(p.min())
        record['implied_feature_means'] = (matrix @ p).tolist()
    return record


@dataclass(frozen=True)
class MomentCalibration:
    status: str
    center: np.ndarray
    original_halfwidth: np.ndarray
    repaired_halfwidth: np.ndarray
    delta: float
    optimal_delta: float
    numeric_margin_delta: float
    raw_feasibility: Mapping
    repair_certificate: Mapping
    bounds_by_rho: Mapping

    def as_dict(self):
        return {'status': self.status, 'center': self.center.tolist(),
                'original_halfwidth': self.original_halfwidth.tolist(),
                'repaired_halfwidth': self.repaired_halfwidth.tolist(),
                'delta': self.delta, 'optimal_delta': self.optimal_delta,
                'numeric_margin_delta': self.numeric_margin_delta,
                'raw_feasibility': dict(self.raw_feasibility),
                'repair_certificate': dict(self.repair_certificate),
                'bounds_by_rho': dict(self.bounds_by_rho)}


def calibrate_moments(planning_features, bootstrap, *, physical_lower=PHYSICAL_LOWER,
                      physical_upper=PHYSICAL_UPPER, rhos=(.5, 1., 1.5, 2.), feasibility_tolerance=1e-8):
    features = _features(planning_features, 'Planning features')
    if not isinstance(bootstrap, BootstrapMoments):
        raise InputValidationError('Expected BootstrapMoments from the independent calibration bank')
    center, width = _vector(bootstrap.center, 'Center'), _vector(bootstrap.halfwidth, 'Halfwidth')
    lower, upper = _vector(physical_lower, 'Physical lower'), _vector(physical_upper, 'Physical upper')
    if (np.any(width < 0) or np.any(upper < lower) or not np.isfinite(feasibility_tolerance)
            or feasibility_tolerance < 1e-10 or feasibility_tolerance > 1e-4):
        raise InputValidationError('Invalid physical range, halfwidth or LP tolerance')
    rho_values = tuple(float(rho) for rho in rhos)
    if not rho_values or min(rho_values) != .5 or len(set(rho_values)) != len(rho_values):
        raise InputValidationError('Moment sweep must contain unique rho values with smallest rho=0.5')
    if any(not np.isfinite(rho) or rho < .5 for rho in rho_values):
        raise InputValidationError('Rho values must be finite and at least 0.5')
    # Physical bounds concern exact attainable values; widening uncertainty
    # cannot legitimize a generator or feature outside its physical domain.
    if (np.any(features < lower) or np.any(features > upper)
            or np.any(center < lower) or np.any(center > upper)):
        raise InputValidationError('Planning support or calibration center lies outside physical feature bounds')
    scale = upper - lower
    constants = scale == 0
    if constants.any() and (np.any(features[:, constants] != lower[constants])
                            or np.any(center[constants] != lower[constants])):
        raise InputValidationError('Constant physical feature is inconsistent and cannot be repaired')
    raw_lower, raw_upper = _bounds(center, width, .5, lower, upper)
    raw = _feasibility(features, raw_lower, raw_upper, feasibility_tolerance)
    raw['rho'] = .5
    raw['lower'], raw['upper'] = raw_lower.tolist(), raw_upper.tolist()
    optimal_delta = margin = 0.
    repair = {'required': False, 'reason': 'Raw rho=0.5 bounds have a verified probability witness'}
    if not (raw['success'] and raw['valid_witness']):
        if raw['solver_status'] != 2:
            return MomentCalibration('FAIL_RAW_LP', _readonly(center), _readonly(width), _readonly(width),
                                     0., 0., 0., raw, {'required': False, 'reason': 'Unresolved raw LP status'}, {})
        n = len(features)
        matrix = features.T
        a_ub = np.vstack([np.column_stack([matrix, -.5 * scale]),
                          np.column_stack([-matrix, -.5 * scale])])
        b_ub = np.concatenate([center + .5 * width, -center + .5 * width])
        a_eq = np.zeros((1, n + 1)); a_eq[0, :n] = 1
        objective = np.zeros(n + 1); objective[-1] = 1
        repair = _lp(objective, a_ub, b_ub, a_eq, np.ones(1), feasibility_tolerance)
        repair['required'] = True
        repair['reason'] = 'Raw rho=0.5 moment set is infeasible on the fixed planning support'
        repair['physical_scale'] = scale.tolist()
        if not (repair['success'] and repair['valid_witness']
                and repair['dual_certificate']['certified_optimal_within_tolerance']):
            return MomentCalibration('FAIL_REPAIR_LP', _readonly(center), _readonly(width), _readonly(width),
                                     0., 0., 0., raw, repair, {})
        optimal_delta = max(0., float(repair['witness'][-1]))
        p = np.asarray(repair['witness'][:-1])
        repair['probability_witness'] = p.tolist()
        repair['minimum_probability'] = float(p.min())
        repair['sum_probability_residual'] = abs(float(p.sum()) - 1.)
    repaired = width + optimal_delta * scale
    final = {}
    for rho in rho_values:
        lo, hi = _bounds(center, repaired, rho, lower, upper)
        certificate = _feasibility(features, lo, hi, feasibility_tolerance)
        final[str(rho)] = {'rho': rho, 'lower': lo.tolist(), 'upper': hi.tolist(), 'feasibility': certificate}
    # An optimal repair witness may suffer final floating-point boundary loss.
    # Only a directly measured, tolerance-sized shortage permits extra width.
    first = final['0.5']['feasibility']
    if not (first['success'] and first['valid_witness']) and repair.get('required'):
        p = np.asarray(repair['probability_witness'])
        positive = scale > 0
        required = max(0., float(np.max((2 * np.abs(features.T @ p - center)[positive]
                                         - width[positive]) / scale[positive], initial=0.)))
        addition = max(0., float(np.nextafter(max(required, optimal_delta), np.inf)) - optimal_delta)
        permitted = 2 * feasibility_tolerance / float(scale[positive].min()) if positive.any() else 0.
        if addition > 0 and addition <= permitted:
            margin = addition
            repaired = width + (optimal_delta + margin) * scale
            repair['numeric_margin_record'] = {'measured_required_delta': required,
                                               'addition': margin, 'maximum_permitted': permitted,
                                               'reason': 'Final rho=0.5 LP lacked a verified witness; repair witness gives the measured boundary shortage'}
            for rho in rho_values:
                lo, hi = _bounds(center, repaired, rho, lower, upper)
                final[str(rho)] = {'rho': rho, 'lower': lo.tolist(), 'upper': hi.tolist(),
                                   'feasibility': _feasibility(features, lo, hi, feasibility_tolerance)}
    passed = all(value['feasibility']['success'] and value['feasibility']['valid_witness'] for value in final.values())
    repair.update({'center_unchanged': True, 'minimum_rho': .5,
                   'common_repaired_base_width_for_all_rhos': True,
                   'numeric_margin_delta': margin, 'physical_scale': scale.tolist()})
    return MomentCalibration('PASS_MOMENT_CALIBRATION' if passed else 'FAIL_FINAL_FEASIBILITY',
                             _readonly(center), _readonly(width), _readonly(repaired),
                             optimal_delta + margin, optimal_delta, margin, raw, repair, final)


def generator_contract():
    return {'version': 'S01-I08-moments-v1', 'dimension': 5,
            'bootstrap': '2000 resamples of whole calibration scenario rows with replacement; equal sample size',
            'center': 'Independent calibration feature mean; never move center to planning support',
            'halfwidth': 'Coordinatewise quantile(|bootstrap_mean-center|), q=1-.05/5=.99, method=higher',
            'stream': {'mechanism_id': BOOTSTRAP_MECHANISM_ID, 'mechanism': 'calibration_bootstrap',
                       'key': '(schema,domain,calibration=2,0,0,0,10,0,0)',
                       'formal_root': ROOTS['calibration'], 'development_root': DEVELOPMENT_ROOT,
                       'shared_indices': 'Same N and B share indices across targets, methods and settings; no OOS stream'},
            'physical_bounds': {'lower': PHYSICAL_LOWER, 'upper': PHYSICAL_UPPER},
            'repair': 'Min delta>=0 over p>=0,sum p=1: abs(Phi.T p-center)<=.5*(width+delta*physical_range)',
            'sweep': 'Keep support, center and repaired base width fixed for rho=.5,1,1.5,2; intersect bounds with physical range',
            'solver': 'SciPy HiGHS, one thread, primal/dual/IPM tolerance 1e-8; retain raw status, primal and dual certificates',
            'numerical_margin': 'Only measured tolerance-sized optimal-witness boundary shortage; record exact addition',
            'constant_features': 'Zero physical range requires exact support/center consistency; no width repair can hide a mismatch'}
