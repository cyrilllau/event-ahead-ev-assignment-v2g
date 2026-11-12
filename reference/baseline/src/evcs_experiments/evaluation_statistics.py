"""Fixed-template paired statistics for IEEE33 S04; no scenario generation.

Arrays use [target, base-weather, variant, method]. The same bootstrap draw of
variant positions is used for every target and method, within each fixed base.
Cross-target summaries average the per-target statistic with equal weight;
an average of target CVaRs is explicitly not a pooled-distribution CVaR.
Undefined observations propagate, with NA rates, rather than becoming zero or
silently changing the comparison population.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from .random_streams import Stream, canonical_hash


def _ids(values, name):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError(f'{name} must contain unique nonempty string IDs')
    return values


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PairedSamples:
    values: np.ndarray
    target_ids: tuple[str, ...]
    base_ids: tuple[str, ...]
    variant_ids: np.ndarray
    method_ids: tuple[str, ...]
    sample_ids: np.ndarray
    na_reasons: np.ndarray | None = None

    def __post_init__(self):
        targets = _ids(self.target_ids, 'target_ids')
        bases = _ids(self.base_ids, 'base_ids')
        methods = _ids(self.method_ids, 'method_ids')
        variants = np.asarray(self.variant_ids, dtype=object)
        if variants.ndim != 2 or variants.shape[0] != len(bases) or not variants.shape[1]:
            raise ValueError('variant_ids must have nonempty [base, variant] shape')
        for row in variants:
            _ids(row, 'variant_ids within each base')
        shape = (len(targets), len(bases), variants.shape[1], len(methods))
        values = np.asarray(self.values, dtype=float)
        if values.shape != shape or np.isinf(values).any():
            raise ValueError('Values must have [target, base, variant, method] shape and no infinity')
        sample_ids = np.asarray(self.sample_ids, dtype=object)
        if sample_ids.shape != shape or any(not isinstance(v, str) or not v for v in sample_ids.flat):
            raise ValueError('sample_ids must explicitly identify every observation and method')
        if not np.all(sample_ids == sample_ids[..., :1]):
            raise ValueError('Methods must have exactly matched sample IDs in the same order')
        _ids(sample_ids[..., 0].flat, 'sample IDs across target/base/variant positions')
        reasons = (np.full(shape, '', dtype=object) if self.na_reasons is None
                   else np.asarray(self.na_reasons, dtype=object))
        if reasons.shape != shape or any(not isinstance(v, str) for v in reasons.flat):
            raise ValueError('na_reasons must be a string array matching values')
        if not np.array_equal(np.isnan(values), reasons != ''):
            raise ValueError('Every NaN needs an explicit NA reason; finite values must have no NA reason')
        for key, value in (('target_ids', targets), ('base_ids', bases), ('method_ids', methods)):
            object.__setattr__(self, key, value)
        for key, value in (('values', values), ('variant_ids', variants), ('sample_ids', sample_ids), ('na_reasons', reasons)):
            object.__setattr__(self, key, _readonly(value))

    @property
    def layout_identity(self):
        return dict(target_ids=list(self.target_ids), base_ids=list(self.base_ids),
            variant_ids=self.variant_ids.tolist(), method_ids=list(self.method_ids),
            sample_ids=self.sample_ids.tolist(), shape=list(self.values.shape))


def _tail_weights(n, tail_fraction):
    if isinstance(tail_fraction, bool):
        raise ValueError('Tail fraction must be in (0, 1]')
    try:
        fraction = Fraction(str(tail_fraction))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError('Tail fraction must be finite and in (0, 1]') from error
    if not 0 < fraction <= 1:
        raise ValueError('Tail fraction must be in (0, 1]')
    mass = fraction * n
    whole = mass.numerator // mass.denominator
    boundary = mass - whole
    weights = np.zeros(n)
    weights[:whole] = 1.
    if boundary:
        weights[whole] = float(boundary)
    return weights, float(mass), whole, float(boundary)


def fractional_tail(values, *, tail_fraction, side):
    """Equal-probability empirical tail mean, retaining exact fractional mass.

Ties affect ordering only; they never add observations to the specified tail.
The returned count is the number of observations with positive tail weight.
"""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or np.isinf(values).any():
        raise ValueError('Tail values must be a nonempty vector without infinity')
    if side not in ('upper', 'lower'):
        raise ValueError('Tail side must be upper or lower')
    weights, mass, whole, boundary = _tail_weights(len(values), tail_fraction)
    missing = int(np.isnan(values).sum())
    ordered = np.sort(values)
    if side == 'upper':
        ordered = ordered[::-1]
    value = None if missing else float(np.dot(ordered, weights / mass))
    return dict(value=value, status='NA' if missing else 'DEFINED',
        reason='undefined_observations_propagated' if missing else None,
        side=side, tail_probability=float(Fraction(str(tail_fraction))),
        sample_count=len(values), na_count=missing, na_rate=missing / len(values),
        tail_mass_in_samples=mass, full_tail_observations=whole,
        boundary_observation_weight=boundary, tail_observation_count=whole + int(boundary > 0))


def empirical_cvar95(values):
    """Mean of the highest-cost 5% probability mass; larger means worse cost."""
    return fractional_tail(values, tail_fraction=.05, side='upper')


def worst_decile_clsr(values):
    """Mean of the lowest CLSR 10% probability mass; larger means better service."""
    array = np.asarray(values, dtype=float)
    if np.any((array < 0) | (array > 1)):
        raise ValueError('CLSR must be in [0, 1] when defined')
    return fractional_tail(array, tail_fraction=.1, side='lower')


def _statistic(values, name):
    # Reduce [target, sample, method] without silently dropping undefined rows.
    valid = np.all(np.isfinite(values), axis=1)
    if name == 'mean':
        result = np.mean(values, axis=1)
    else:
        weights, mass, _, _ = _tail_weights(values.shape[1], .05 if name == 'cvar95' else .1)
        ordered = np.sort(values, axis=1)
        if name == 'cvar95':
            ordered = ordered[:, ::-1, :]
        result = np.sum(ordered * (weights / mass)[None, :, None], axis=1)
    return np.where(valid, result, np.nan)


def _summary(point, draws):
    if not np.isfinite(point):
        return dict(estimate=None, ci95=[None, None], status='NA',
            reason='undefined_observations_propagated')
    if not np.isfinite(draws).all():
        return dict(estimate=float(point), ci95=[None, None], status='NA_INTERVAL',
            reason='undefined_bootstrap_replicates_propagated')
    return dict(estimate=float(point), ci95=np.quantile(draws, [.025, .975], method='linear').tolist(),
        status='DEFINED', reason=None)


def _json_numbers(array):
    # JSON consumers receive None with the summary reason, never NaN literals.
    return np.where(np.isfinite(array), array, None).tolist()


def paired_bootstrap(samples, *, seed, reference_method, statistic='mean',
                     replications=2000, development=True):
    """Paired, fixed-base percentile 95% intervals, including per-target results.

Differences are statistic(method) minus statistic(reference), using matched
bootstrap samples. For tails this is a difference of tail means, not the tail
mean of per-sample differences. Missing targets never receive zero weight.
"""
    if not isinstance(samples, PairedSamples):
        raise ValueError('Expected validated PairedSamples')
    if statistic not in ('mean', 'cvar95', 'worst_decile_clsr'):
        raise ValueError('Unknown statistic')
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(replications) is not int or replications != 2000:
        raise ValueError('The locked S04 protocol requires exactly 2000 replications')
    if type(development) is not bool or reference_method not in samples.method_ids:
        raise ValueError('Explicit development flag and an existing reference method are required')
    if statistic == 'worst_decile_clsr' and np.any((samples.values < 0) | (samples.values > 1)):
        raise ValueError('CLSR must be in [0, 1] when defined')
    target_count, base_count, variant_count, method_count = samples.values.shape
    stream = Stream(int(seed), (1, 2 if development else 1, 3, 0, 0, 0, 11, 0, 0))
    indices = stream.generator().integers(0, variant_count,
        size=(replications, base_count, variant_count))
    point = _statistic(samples.values.reshape(target_count, -1, method_count), statistic)
    draws = np.empty((replications, target_count, method_count))
    for i, index in enumerate(indices):
        resampled = np.take_along_axis(samples.values, index[None, :, :, None], axis=2)
        draws[i] = _statistic(resampled.reshape(target_count, -1, method_count), statistic)
    ref = samples.method_ids.index(reference_method)
    differences = point - point[:, ref:ref + 1]
    difference_draws = draws - draws[:, :, ref:ref + 1]
    mean_point, mean_draws = np.mean(point, axis=0), np.mean(draws, axis=1)
    mean_difference, mean_difference_draws = np.mean(differences, axis=0), np.mean(difference_draws, axis=1)
    targets = []
    for t, target in enumerate(samples.target_ids):
        methods = {}
        for m, method in enumerate(samples.method_ids):
            missing = np.isnan(samples.values[t, :, :, m])
            reasons, counts = np.unique(samples.na_reasons[t, :, :, m][missing], return_counts=True)
            methods[method] = dict(**_summary(point[t, m], draws[:, t, m]),
                difference_from_reference=_summary(differences[t, m], difference_draws[:, t, m]),
                sample_count=base_count * variant_count, na_count=int(missing.sum()),
                na_rate=float(missing.mean()), na_reasons=dict(zip(reasons.tolist(), counts.tolist())))
        targets.append(dict(target_id=target, methods=methods))
    aggregate = {method: dict(**_summary(mean_point[m], mean_draws[:, m]),
        difference_from_reference=_summary(mean_difference[m], mean_difference_draws[:, m]),
        na_rate=float(np.isnan(samples.values[:, :, :, m]).mean()),
        undefined_target_count=int(np.isnan(point[:, m]).sum()))
        for m, method in enumerate(samples.method_ids)}
    tail_definition = None
    if statistic != 'mean':
        fraction = .05 if statistic == 'cvar95' else .1
        _, mass, whole, boundary = _tail_weights(base_count * variant_count, fraction)
        tail_definition = dict(side='upper' if statistic == 'cvar95' else 'lower',
            probability_mass=fraction, per_target_sample_count=base_count * variant_count,
            tail_mass_in_samples=mass, full_tail_observations=whole,
            boundary_observation_weight=boundary, tail_observation_count=whole + int(boundary > 0),
            ties_policy='retain the fixed fractional probability mass; never expand at ties')
    return dict(statistic=statistic, reference_method=reference_method,
        tail_definition=tail_definition,
        replications=replications, confidence_level=.95, interval_method='percentile_linear',
        na_policy='propagate; never fill zero or drop observations/targets',
        layout=samples.layout_identity, layout_sha256=canonical_hash(samples.layout_identity),
        target_weights=[1. / target_count] * target_count,
        cross_target_estimand='equal-weight mean of per-target statistics',
        inference_scope='conditional on the fixed base-weather templates; not independent new-storm inference',
        random_stream=stream.as_dict(), resampling_axis='matched (base-weather, variant), shared across targets and methods',
        bootstrap_indices=indices.tolist(), bootstrap_indices_sha256=canonical_hash(indices.tolist()),
        per_target=targets, aggregate=aggregate,
        bootstrap_statistics=_json_numbers(draws), bootstrap_differences=_json_numbers(difference_draws))


def paired_bootstrap_mean(samples, *, seed, reference_method, replications=2000, development=True):
    return paired_bootstrap(samples, seed=seed, reference_method=reference_method,
        statistic='mean', replications=replications, development=development)
