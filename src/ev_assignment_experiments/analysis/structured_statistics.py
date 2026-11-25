"""Preregistered paired OOS bootstrap; no IO, generation, or optimization.

The caller binds these ordered records to frozen training policies and the OOS
manifest. Intervals describe OOS sampling uncertainty conditional on those
policies, not training uncertainty or ambiguity-set coverage. Costs use audited
primal upper values; numerical recourse gaps are reported separately.
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from ev_assignment.domain.planning import Control


METHODS = ("DET", "SAA", "ARO", "P3", "P2", "P3-noadvance")
COMPARATORS = tuple(method for method in METHODS if method != "P3")
METRICS = ("total_cost_mean", "total_cost_cvar95", "critical_ENS_mean")
REPLICATES = 500


def summarize_paired(records, *, control: Control, seed, repetitions=500, expected_n=1000):
    """Worker API: ``seed`` is already 910000 + validation seed (no extra offset).

    The preregistered repetition count is fixed at 500. See ``paired_bootstrap``
    for schema, eligibility, estimation and deadline semantics. ``expected_n``
    may only be overridden for explicitly declared artificial fixtures.
    """
    control.check()
    if (not isinstance(repetitions, Integral) or isinstance(repetitions, bool)
            or repetitions != REPLICATES):
        raise ValueError("The preregistered bootstrap requires exactly 500 repetitions")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 910000:
        raise ValueError("seed must be 910000 + the nonnegative validation seed")
    return paired_bootstrap(records, int(seed) - 910000, control, expected_n=expected_n)


def _metrics(cost, critical_ens, weights):
    """Weighted means and upper 5% tail mean, splitting boundary atoms."""
    order = np.argsort(cost, axis=0, kind="stable")[::-1]
    descending = np.take_along_axis(cost, order, axis=0)
    ordered_weights = weights[order]
    mass_before = np.cumsum(ordered_weights, axis=0) - ordered_weights
    tail_mass = 1.0 - .95
    take = np.minimum(ordered_weights, np.maximum(0., tail_mass - mass_before))
    cvar = np.sum(take * descending, axis=0) / tail_mass
    return np.column_stack((weights @ cost, cvar, weights @ critical_ens))


def paired_bootstrap(records, validation_seed, control: Control, *, expected_n=1000):
    """Return P3-minus-comparator point differences and percentile 95% CIs.

    ``records`` is the frozen ordered sequence of OOS rows, each containing
    ``stratum`` (integer 0..3), ``weight``, and ``methods[method]`` with finite
    ``qU``, ``qL``, ``first_cost`` and ``critical_ENS_kwh`` for all six METHODS.
    Optional ``sid`` values, when supplied, must be present on every row and
    unique. The caller remains responsible for manifest/policy identity audits.

    Exactly 500 replicates use local PCG64(seed=910000+validation_seed). Within
    each of the four strata, uniformly resample the original number of rows
    with replacement; use the SAME indices for every method and metric. The
    generator's equal weights within each stratum are required. Original
    stratum counts and masses stay fixed, with only roundoff normalization of
    the total probability mass permitted (absolute tolerance 1e-12).

    CVaR95 is the weighted upper-tail mean of EACH method's total cost, followed
    by subtraction; it is not CVaR of paired scenario differences. Percentiles
    use NumPy's explicit linear interpolation, without multiplicity correction.
    ``qL`` checks numerical evidence and does not replace ``qU`` in estimates.

    A partial sample or missing method returns ``status='NOT_ELIGIBLE'``,
    ``formal_ci=False`` and no estimates/intervals. Inconsistent IDs or weights
    raise ValueError even on a partial sample. Invalid metrics also raise.
    Expired or
    stopped Control propagates TimeoutError/InterruptedError, never a partial
    replicate CI. ``expected_n`` is an explicit toy-fixture override; production
    callers must leave it at 1000. No input or global RNG state is mutated.
    """
    control.check()
    if (not isinstance(expected_n, Integral) or isinstance(expected_n, bool)
            or expected_n < 4):
        raise ValueError("expected_n must be an integer >= 4; production requires 1000")
    if (not isinstance(validation_seed, Integral) or isinstance(validation_seed, bool)
            or validation_seed < 0):
        raise ValueError("validation_seed must be a nonnegative integer")
    n = len(records)
    if n > expected_n:
        raise ValueError("More OOS rows than the declared complete sample")
    missing = {method: 0 for method in METHODS}
    for row in records:
        control.check()
        if not isinstance(row, Mapping):
            raise ValueError("Each OOS row must be a mapping")
        methods = row.get("methods", {})
        if not isinstance(methods, Mapping):
            raise ValueError("OOS methods must be a mapping")
        for method in METHODS:
            missing[method] += methods.get(method) is None
    missing = {method: count for method, count in missing.items() if count}
    result = dict(
        protocol="structured-paired-stratified-bootstrap-v1",
        status="NOT_ELIGIBLE", formal_ci=False,
        expected_n=int(expected_n), observed_n=n, fixture_override=expected_n != 1000,
        missing_method_rows=missing, reference="P3", comparators=list(COMPARATORS),
        difference="P3 minus comparator", seed=910000 + int(validation_seed),
        validation_seed=int(validation_seed), rng="PCG64", replicates=REPLICATES,
        completed_replicates=0, percentiles=[2.5, 97.5], percentile_method="linear",
        simultaneous_intervals=False,
        uncertainty="OOS sampling conditional on frozen training policies; not probability-set coverage",
        cost_basis="first_cost + qU; percentile CIs exclude numerical recourse uncertainty",
        point_metrics={}, comparisons={},
        ineligible_reasons=(["INCOMPLETE_SAMPLE"] if n != expected_n else [])
        + (["MISSING_METHODS"] if missing else []),
    )
    if any("sid" in row for row in records):
        ids = [row.get("sid") for row in records]
        if (any(not isinstance(sid, str) or not sid for sid in ids)
                or len(set(ids)) != n):
            raise ValueError("OOS scenario IDs must be complete and unique")
    strata = np.empty(n, dtype=int)
    weights = np.empty(n)
    for i, row in enumerate(records):
        control.check()
        stratum = row.get("stratum")
        if (not isinstance(stratum, Integral) or isinstance(stratum, bool)
                or stratum not in range(4)):
            raise ValueError("OOS stratum must be an integer in 0..3")
        strata[i] = stratum
        try:
            weights[i] = row["weight"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Incomplete/non-numeric OOS weights") from exc
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("OOS weights must be finite and strictly positive")
    original_mass = float(weights.sum())
    if original_mass > 1. + 1e-12 or (n == expected_n and abs(original_mass - 1.) > 1e-12):
        raise ValueError("Complete OOS weights must sum to one")
    groups = [np.flatnonzero(strata == s) for s in range(4)]
    if n == expected_n and any(len(indices) == 0 for indices in groups):
        raise ValueError("All four OOS strata must be represented")
    if any(len(indices) and not np.all(weights[indices] == weights[indices[0]])
           for indices in groups):
        raise ValueError("Preregistered stratified bootstrap requires equal within-stratum weights")
    if n != expected_n or missing:
        control.check()
        return result

    values = np.empty((n, len(METHODS), 4))
    for i, row in enumerate(records):
        control.check()
        try:
            for j, method in enumerate(METHODS):
                answer = row["methods"][method]
                values[i, j] = [answer[key] for key in
                                ("qU", "qL", "first_cost", "critical_ENS_kwh")]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Incomplete/non-numeric OOS metrics") from exc
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite OOS metrics cannot enter a formal CI")
    gaps = values[:, :, 0] - values[:, :, 1]
    if np.any(gaps < -1e-5):
        raise ValueError("Recourse qL exceeds qU beyond the project objective tolerance")
    result["recourse_gap_ranges"] = {
        method: [float(gaps[:, j].min()), float(gaps[:, j].max())]
        for j, method in enumerate(METHODS)
    }
    result["original_weight_sum"] = original_mass
    result["strata"] = [dict(stratum=s, count=len(indices), mass=float(weights[indices].sum()))
                         for s, indices in enumerate(groups)]
    weights = weights / original_mass
    cost = values[:, :, 0] + values[:, :, 2]
    if not np.isfinite(cost).all():
        raise ValueError("Nonfinite total costs cannot enter a formal CI")
    ens = values[:, :, 3]
    point = _metrics(cost, ens, weights)
    reference = METHODS.index("P3")
    comparisons = [METHODS.index(method) for method in COMPARATORS]
    difference = point[reference] - point[comparisons]
    draws = np.empty((REPLICATES, len(COMPARATORS), len(METRICS)))
    rng = np.random.Generator(np.random.PCG64(result["seed"]))
    sample = np.empty(n, dtype=int)
    for replicate in range(REPLICATES):
        control.check()
        for indices in groups:
            sample[indices] = indices[rng.integers(len(indices), size=len(indices))]
        # Fixed weight slots preserve each stratum's mass/count in every draw.
        estimate = _metrics(cost[sample], ens[sample], weights)
        draws[replicate] = estimate[reference] - estimate[comparisons]
    control.check()
    limits = np.percentile(draws, [2.5, 97.5], axis=0, method="linear")
    result["point_metrics"] = {
        method: {metric: float(point[j, k]) for k, metric in enumerate(METRICS)}
        for j, method in enumerate(METHODS)
    }
    result["comparisons"] = {
        method: {metric: dict(difference=float(difference[j, k]),
                              ci95=[float(limits[0, j, k]), float(limits[1, j, k])])
                 for k, metric in enumerate(METRICS)}
        for j, method in enumerate(COMPARATORS)
    }
    control.check()
    result.update(status="PAIRED_STRATIFIED_BOOTSTRAP_COMPLETE", formal_ci=True,
                  completed_replicates=REPLICATES)
    return result
