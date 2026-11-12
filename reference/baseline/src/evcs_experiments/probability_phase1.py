"""Feasible initialization of the original, finite-support ambiguity set.

A deterministic zero-objective LP obtains a full raw probability witness.
Every nonzero component is retained, including arbitrarily small positive
components and any solver roundoff below zero. No probabilities or moments are
repaired. Basicness is a separate numerical active-constraint rank check; a
solver success flag alone never establishes it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from types import MappingProxyType
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

from .probability import FEASIBILITY_TOLERANCE, ProbabilitySet
from .random_streams import canonical_hash


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _freeze(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class Phase1Result:
    status: str
    feasible: bool
    basis_verified: bool
    probabilities: np.ndarray | None
    support_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    ambiguity_sha256: str
    witness_sha256: str | None
    initial_support_indices: tuple[int, ...]
    initial_support_ids: tuple[str, ...]
    initial_probabilities: np.ndarray | None
    certificate: Mapping
    solver: Mapping
    wall_seconds: float

    def __post_init__(self):
        for key in ('probabilities', 'initial_probabilities', 'certificate', 'solver'):
            object.__setattr__(self, key, _freeze(getattr(self, key)))

    def as_dict(self):
        return {key: _json(value) for key, value in vars(self).items()}


def _solve_lp(a, time_limit_seconds):
    s = len(a.support_ids)
    options = {'time_limit': float(time_limit_seconds), 'threads': 1,
               'primal_feasibility_tolerance': FEASIBILITY_TOLERANCE,
               'dual_feasibility_tolerance': FEASIBILITY_TOLERANCE}
    began = time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning,
                                message='Unrecognized options detected.*')
        raw = linprog(np.zeros(s), A_ub=np.vstack((a.features.T, -a.features.T)),
                      b_ub=np.r_[a.upper, -a.lower], A_eq=np.ones((1, s)),
                      b_eq=np.ones(1), bounds=[(0., None)] * s,
                      method='highs-ds', options=options)
    record = {'solver': 'scipy.optimize.linprog/HiGHS dual simplex',
              'status': int(raw.status), 'success': bool(raw.success),
              'message': str(raw.message), 'iterations': int(getattr(raw, 'nit', 0) or 0),
              'objective': 'deterministic zero vector; feasibility only',
              'raw_objective': None if raw.fun is None else float(raw.fun),
              'raw_solution': None if raw.x is None else np.asarray(raw.x, dtype=float).copy(),
              'options': options, 'wall_seconds': time.monotonic() - began,
              'infeasibility_certificate_available': False}
    for name in ('ineqlin', 'eqlin', 'lower', 'upper'):
        group = getattr(raw, name, None)
        if group is not None:
            for item in ('residual', 'marginals'):
                value = getattr(group, item, None)
                # Upper-bound residuals can legitimately be infinite because
                # the original LP has no finite probability upper bounds.
                if value is not None and np.isfinite(value).all():
                    record[name + '_' + item] = np.asarray(value, dtype=float).copy()
    return raw, record


def _basis_certificate(a, p, means, feasible):
    """Rank of the full active system, compressed along exact-zero columns.

    Exact zero bounds contribute independent identity rows. Row scaling is
    used only to assess rank, never to alter the probability LP. Moment
    activity uses a floating-point roundoff scale, not the 1e-8 feasibility
    tolerance, so a merely nearby interior moment is not treated as binding.
    """
    retained = np.flatnonzero(p != 0.)
    zero = np.flatnonzero(p == 0.)
    magnitude = np.maximum(1., np.maximum(np.abs(a.lower), np.abs(a.upper)))
    magnitude = np.maximum(magnitude, np.abs(a.features).T @ np.abs(p))
    activity_tol = 64. * np.finfo(float).eps * magnitude
    lower_active = np.abs(means - a.lower) <= activity_tol
    upper_active = np.abs(means - a.upper) <= activity_tol
    rows = [np.ones(len(retained))]
    row_ids = ['normalization']
    for side, mask in (('lower', lower_active), ('upper', upper_active)):
        for j in np.flatnonzero(mask):
            rows.append(a.features[retained, j])
            row_ids.append(f'{side}:{a.feature_ids[j]}')
    matrix = np.asarray(rows, dtype=float)
    row_scale = np.maximum(1., np.max(np.abs(matrix), axis=1, initial=0.))
    scaled = matrix / row_scale[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank_tol = max(scaled.shape) * np.finfo(float).eps * float(np.max(singular, initial=0.))
    rank = int(np.count_nonzero(singular > rank_tol))
    full_rank = len(zero) + rank
    verified = bool(feasible and np.all(p >= 0.) and full_rank == len(p))
    return {'verified': verified, 'kind': 'numerical active-constraint rank; not exact arithmetic',
            'strict_nonnegative': bool(np.all(p >= 0.)),
            'zero_constraint_indices': zero.tolist(),
            'compressed_column_indices': retained.tolist(), 'active_moment_row_ids': row_ids,
            'activity_tolerance_by_feature': activity_tol,
            'activity_roundoff_multiplier': 64.,
            'compressed_matrix': matrix, 'rank_row_scale': row_scale,
            'singular_values': singular, 'rank_tolerance': rank_tol,
            'compressed_rank': rank, 'full_active_rank': full_rank,
            'required_full_rank': len(p), 'support_cardinality': len(retained),
            'basic_support_upper_bound': len(a.feature_ids) + 1,
            'support_bound_satisfied': len(retained) <= len(a.feature_ids) + 1,
            'solver_status_used_as_basis_proof': False}


def solve_phase1(probability_set, *, time_limit_seconds=30):
    """Return a full witness and its unchanged, safe initial support.

    ``feasible`` means the original-unit and normalized 1e-8 checks pass.
    ``basis_verified`` additionally requires a full-rank numerical active
    system and no negative raw component. A feasible nonbasic witness is still
    a valid numerical initialization; it does not justify a sparse-BFS claim.
    TIME_LIMIT remains TIME_LIMIT even when a feasible incumbent is available.
    INFEASIBLE without the explicit constant-column proof is only the solver's
    report, not a Farkas certificate. No new scenario or random draw is used.
    """
    began = time.monotonic()
    _require(isinstance(probability_set, ProbabilitySet), 'Expected ProbabilitySet')
    a = probability_set
    _require(not isinstance(time_limit_seconds, (bool, np.bool_)) and
             np.isfinite(time_limit_seconds) and time_limit_seconds > 0,
             'Positive finite time limit required')
    s, f = a.features.shape
    tol = FEASIBILITY_TOLERANCE
    certificate = {'problem': 'find p>=0, sum(p)=1, lower<=features.T@p<=upper',
                   'feature_orientation': 'scenario_by_feature',
                   'initial_objective': 'fixed zero vector',
                   'feasibility_tolerance': tol, 'normalized_feasibility_tolerance': tol,
                   'probabilities_clipped_or_renormalized': False,
                   'moment_bounds_repaired': False, 'new_scenarios_or_random_draws': False,
                   'support_selection': 'all raw nonzero entries, including negative roundoff',
                   'support_threshold': 0., 'restricted_lift_exactly_matches_full_witness': False,
                   'restricted_witness_feasible': False, 'analytic_infeasibility': None,
                   'infeasibility_certified': False, 'basis': None}
    constants = np.all(a.features == a.features[:1], axis=0)
    bad = constants & ((a.features[0] < a.lower) | (a.features[0] > a.upper))
    if np.any(bad):
        certificate['analytic_infeasibility'] = {
            'kind': 'constant_feature_outside_moment_interval',
            'feature_ids': [a.feature_ids[j] for j in np.flatnonzero(bad)],
            'constant_values': a.features[0, bad], 'lower': a.lower[bad], 'upper': a.upper[bad]}
        certificate['infeasibility_certified'] = True
        record = {'status': None, 'success': False, 'message': 'Not run: exact constant-column contradiction',
                  'raw_solution': None, 'infeasibility_certificate_available': False}
        return Phase1Result('INFEASIBLE', False, False, None, a.support_ids, a.feature_ids,
                            a.ambiguity_sha256, None, (), (), None, certificate, record,
                            time.monotonic() - began)
    raw, record = _solve_lp(a, float(time_limit_seconds))
    p = None if raw.x is None else np.asarray(raw.x, dtype=float).copy()
    feasible = basic = False
    indices = ()
    subset_ids = ()
    subset_p = None
    witness_hash = None
    certificate['raw_witness_shape_valid'] = p is not None and p.shape == (s,)
    certificate['raw_witness_finite'] = p is not None and bool(np.isfinite(p).all())
    if p is not None and p.shape == (s,) and np.isfinite(p).all():
        means = a.features.T @ p
        feature_scale = np.maximum(1., np.maximum(np.max(np.abs(a.features), axis=0),
                                                  np.maximum(np.abs(a.lower), np.abs(a.upper))))
        violation = np.maximum(np.maximum(a.lower - means, means - a.upper), 0.)
        norm_residual = abs(float(p.sum()) - 1.)
        negative = float(np.max(-p, initial=0.))
        norm_scale = max(1., float(np.abs(p).sum()))
        original_max = max(norm_residual, negative, float(np.max(violation, initial=0.)))
        normalized_max = max(norm_residual / norm_scale, negative,
                             float(np.max(violation / feature_scale, initial=0.)))
        feasible = bool(np.isfinite(means).all() and original_max <= tol and normalized_max <= tol)
        certificate['primal'] = {'sum_probability': float(p.sum()),
            'normalization_residual': norm_residual, 'minimum_probability': float(p.min()),
            'nonnegative_violation': negative, 'feature_means': means,
            'lower_slack': means - a.lower, 'upper_slack': a.upper - means,
            'moment_violation': violation, 'feature_scale': feature_scale,
            'maximum_original_violation': original_max, 'maximum_normalized_violation': normalized_max}
        witness_hash = canonical_hash({'ambiguity_sha256': a.ambiguity_sha256,
                                       'support_ids': list(a.support_ids), 'probabilities': p.tolist()})
        certificate['basis'] = _basis_certificate(a, p, means, feasible)
        basic = certificate['basis']['verified']
        if feasible:
            indices = tuple(int(i) for i in np.flatnonzero(p != 0.))
            subset_ids = tuple(a.support_ids[i] for i in indices)
            subset_p = p[list(indices)].copy()
            lifted = np.zeros(s)
            lifted[list(indices)] = subset_p
            same = bool(np.array_equal(lifted, p))
            restricted_means = a.features[list(indices)].T @ subset_p
            subset_violation = np.maximum(np.maximum(a.lower - restricted_means, restricted_means - a.upper), 0.)
            subset_normalization = abs(float(subset_p.sum()) - 1.)
            subset_negative = float(np.max(-subset_p, initial=0.))
            restricted_ok = bool(same and max(subset_normalization, subset_negative,
                float(np.max(subset_violation, initial=0.))) <= tol and
                float(np.max(subset_violation / feature_scale, initial=0.)) <= tol)
            certificate.update(restricted_lift_exactly_matches_full_witness=same,
                restricted_witness_feasible=restricted_ok, restricted_feature_means=restricted_means,
                restricted_normalization_residual=subset_normalization,
                restricted_moment_violation=subset_violation,
                retained_negative_probability_ids=[a.support_ids[i] for i in indices if p[i] < 0.],
                retained_tiny_positive_probability_ids=[a.support_ids[i] for i in indices if 0. < p[i] <= tol])
            feasible = bool(feasible and restricted_ok)
            if not feasible:
                indices = (); subset_ids = (); subset_p = None; basic = False
    if raw.status == 1:
        status = 'TIME_LIMIT'
    elif feasible and raw.status == 0:
        status = 'FEASIBLE_BASIC' if basic else 'FEASIBLE_NONBASIC'
    elif feasible:
        status = 'SOLVER_STATUS_CONFLICT'
    elif raw.status == 2:
        status = 'INFEASIBLE'
    elif raw.status in (3, 4):
        status = 'NUMERICAL_ERROR'
    else:
        status = 'NUMERICAL_CERTIFICATE_FAILED'
    certificate.update(feasible=feasible, basis_verified=basic,
                       solver_optimal=bool(raw.status == 0),
                       numerical_certificate_only=True)
    return Phase1Result(status, feasible, basic, p, a.support_ids, a.feature_ids,
                        a.ambiguity_sha256, witness_hash, indices, subset_ids, subset_p,
                        certificate, record, time.monotonic() - began)
