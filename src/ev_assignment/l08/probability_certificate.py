"""Pure numerical repair for a valid probability cut; no optimizer, files, or global paths.

A zero all-rows margin is allowed.  The reference distribution need only be
certifiably feasible.  `check` must verify the ORIGINAL complete ambiguity set
(and, when required by the caller, the stored scaled solver rows), not only
numerically active rows.  The bound vector must dominate the master theta box.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, Mapping, Any, Sequence
import math
import numpy as np

DENOMINATOR = 1 << 52


class ProbabilityCertificateError(ValueError):
    pass


def dyadic_simplex(values: Sequence[float]) -> np.ndarray:
    """Construct a nonnegative, exactly mass-one dyadic CANDIDATE.

    Rounding/normalizing is NOT a feasibility certificate; `check` is mandatory.
    Exact largest remainders avoid integer overflow and negative residual cells.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ProbabilityCertificateError("INVALID_PROBABILITY_VECTOR")
    weights = [Fraction(float(max(v, 0.0))) for v in a]
    total = sum(weights, Fraction(0))
    if total <= 0:
        raise ProbabilityCertificateError("NONPOSITIVE_CANDIDATE_MASS")
    scaled = [w * DENOMINATOR / total for w in weights]
    nums = [v.numerator // v.denominator for v in scaled]
    missing = DENOMINATOR - sum(nums)
    # Largest-remainder allocation: 0 <= missing < n.
    if not 0 <= missing < len(nums):
        raise ProbabilityCertificateError("DYADIC_ALLOCATION_FAILURE")
    order = sorted(range(len(nums)), key=lambda i: (scaled[i] - nums[i], -i), reverse=True)
    for i in order[:missing]:
        nums[i] += 1
    result = np.array(nums, dtype=np.float64) / DENOMINATOR
    if sum(nums) != DENOMINATOR or min(nums) < 0:
        raise ProbabilityCertificateError("DYADIC_SIMPLEX_FAILURE")
    return result


def compensation(p: np.ndarray, pstar: np.ndarray, qbars: Sequence[float], ids: Sequence[int]):
    """Exact binary64-coefficient box error, rounded upward for row storage."""
    b = np.asarray(qbars, dtype=np.float64)
    idx = np.asarray(ids)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ProbabilityCertificateError("INVALID_SCENARIO_IDS")
    if len(idx) != len(b) or len(set(map(int, idx))) != len(idx):
        raise ProbabilityCertificateError("DUPLICATE_OR_MISMATCHED_SCENARIO_IDS")
    if np.any(idx < 0) or np.any(idx >= len(p)) or not np.isfinite(b).all() or np.any(b < 0):
        raise ProbabilityCertificateError("INVALID_THETA_BOUND")
    exact = sum(
        (
            Fraction(float(q)) * max(Fraction(float(p[i])) - Fraction(float(pstar[i])), Fraction(0))
            for i, q in zip(idx, b)
        ),
        Fraction(0),
    )
    try:
        guard = float(exact)
    except OverflowError as exc:
        raise ProbabilityCertificateError("COMPENSATION_OVERFLOW") from exc
    if not math.isfinite(guard):
        raise ProbabilityCertificateError("COMPENSATION_OVERFLOW")
    if Fraction(guard) < exact:
        guard = float(np.nextafter(guard, np.inf))
    if not math.isfinite(guard):
        raise ProbabilityCertificateError("COMPENSATION_OVERFLOW")
    return guard, exact


def certify_with_feasible_anchor(
    p: Sequence[float],
    p0: Sequence[float],
    qbars: Sequence[float],
    ids: Sequence[int],
    check: Callable[[np.ndarray], Mapping[str, Any]],
    *,
    nontrivial_margin_hint: float | None = None,
    max_mixed_trials: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a verified feasible pstar and a globally safe cut compensation.

    Positive slack is only an optional step-size heuristic.  After bounded
    mixing trials the original, already-verified anchor is used EXACTLY.
    Therefore true equality rows / boundary anchors do not cause assertion
    failure.  An invalid anchor still causes a real correctness failure.
    """
    raw = np.asarray(p, dtype=np.float64).copy()
    anchor = np.asarray(p0, dtype=np.float64).copy()
    if (
        raw.ndim != 1
        or raw.shape != anchor.shape
        or not raw.size
        or not np.isfinite(raw).all()
        or not np.isfinite(anchor).all()
    ):
        raise ProbabilityCertificateError("INVALID_PROBABILITY_INPUT")
    if not isinstance(max_mixed_trials, int) or not 0 <= max_mixed_trials <= 60:
        raise ProbabilityCertificateError("INVALID_TRIAL_LIMIT")
    c0 = dict(check(anchor))
    if not c0.get("passed", False):
        raise ProbabilityCertificateError(f"REFERENCE_DISTRIBUTION_NOT_CERTIFIED: {c0}")
    # Fail bound/ID errors before doing expensive matrix checks.
    compensation(raw, anchor, qbars, ids)
    candidate = dyadic_simplex(raw)
    ck = dict(check(candidate))
    history = [{"kind": "candidate", "mix": 0.0, "check": ck}]
    chosen, mix, mode = candidate, 0.0, "DYADIC_CANDIDATE"
    if not ck.get("passed", False):
        v = max(
            [0.0]
            + [
                -float(ck[k])
                for k in ("raw_margin", "scaled_margin")
                if k in ck and math.isfinite(float(ck[k]))
            ]
        )
        hint = nontrivial_margin_hint
        if hint is not None and math.isfinite(float(hint)) and hint > 0:
            initial = min(1.0, max(2.0**-52, 4.0 * v / float(hint)))
        else:
            initial = 2.0**-40
        t = initial
        for _ in range(max_mixed_trials):
            if t >= 1.0:
                break
            mixed = dyadic_simplex((1.0 - t) * candidate + t * anchor)
            trial = dict(check(mixed))
            history.append({"kind": "mixed", "mix": t, "check": trial})
            if trial.get("passed", False):
                chosen, ck, mix, mode = mixed, trial, t, "VERIFIED_MIX"
                break
            t = min(1.0, 4.0 * t)
        else:
            pass
        if not ck.get("passed", False):
            # No fresh normalization at the endpoint: use the verified bytes.
            chosen, ck, mix, mode = anchor.copy(), c0, 1.0, "VERIFIED_ANCHOR_FALLBACK"
            history.append({"kind": "anchor_fallback", "mix": 1.0, "check": ck})
    guard, exact = compensation(raw, chosen, qbars, ids)
    return chosen, {
        "passed": True,
        "epsilon": guard,
        "mix": mix,
        "mode": mode,
        "anchor_check": c0,
        "pstar_check": ck,
        "trial_count": len(history),
        "nontrivial_margin_hint": nontrivial_margin_hint,
        "trials": history,
        "compensation_exact": [exact.numerator, exact.denominator],
        "raw_coefficients_unchanged": True,
        "proof": (
            "For verified feasible pstar and 0<=theta_s<=Qbar_s, "
            "p^T theta-pstar^T theta<=sum Qbar_s max(p_s-pstar_s,0). "
            "The full ambiguity set is unchanged. epsilon is rounded upward "
            "from exact arithmetic on the stored binary64 coefficients."
        ),
    }
