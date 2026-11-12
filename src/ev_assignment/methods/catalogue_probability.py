"""Explicit, small synthetic moment domain, separate from old L08 calibration."""

from fractions import Fraction
import hashlib
import numpy as np
from scipy import sparse

from ev_assignment.domain.planning import AmbiguitySet
from ev_assignment.l08.certificates import enclosure
from ev_assignment.l08.probability_certificate import dyadic_simplex


def calibrated(identities, features, calibration_features, control, *, marginal=False,
               lower, upper, calibration_rule):
    """Binary G,B,G*B moments with explicit preregistered CAL-only intervals.

    Three coordinates are prespecified, never added after seeing policies. Full
    retains all three; marginal drops only product. This is a NEW probability
    model, not the original K4L4 probability set. The common original support is
    preserved. A separate max-interior feasibility LP constructs a checked anchor.
    """
    from scipy.optimize import linprog
    features, cal = np.asarray(features, float), np.asarray(calibration_features, float)
    if features.shape != (3, len(identities)) or cal.ndim != 2 or cal.shape[0] != 3:
        raise ValueError("Expected explicit three-feature-by-scenario arrays")
    for x in [features, cal]:
        if np.any((x != 0) & (x != 1)) or not np.array_equal(x[2], x[0] * x[1]):
            raise ValueError("Fixed binary G/B/product feature identities do not hold")
    mean = cal.mean(axis=1)
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    if low.shape != (3,) or high.shape != (3,) or np.any(low > high):
        raise ValueError("Explicit ordered calibration intervals required")
    k = 2 if marginal else 3
    phi = sparse.csr_matrix(features[:k])
    lo, hi = low[:k], high[:k]
    def check(p):
        p = np.asarray(p, float)
        if p.shape != (len(identities),) or not np.isfinite(p).all() or np.any(p < 0):
            return dict(passed=False, reason="nonnegative_full_support")
        exact = sum(map(lambda x: Fraction(float(x)), p), Fraction(0)) == 1
        pl, pu = enclosure(phi, p)
        # Margins are evaluated outward; no feasibility tolerance weakens P.
        violation = float(max(0., np.max(lo - pl), np.max(pu - hi)))
        return dict(passed=bool(exact and violation == 0), exact_simplex=exact,
                    original_moment_violation=violation, feature_count=k)
    A = sparse.vstack([sparse.hstack([phi, np.ones((k, 1))]),
                       sparse.hstack([-phi, np.ones((k, 1))])], format="csr")
    control.check()
    lp = linprog(np.r_[np.zeros(len(identities)), -1], A_ub=A, b_ub=np.r_[hi, -lo],
                 A_eq=np.r_[np.ones(len(identities)), 0][None, :], b_eq=[1],
                 bounds=[(0, None)] * len(identities) + [(0, None)], method="highs",
                 options=dict(threads=1, time_limit=max(.01, control.remaining),
                              primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if not lp.success:
        raise ValueError(f"Synthetic moment/support set infeasible or unproved: {lp.message}")
    anchor = dyadic_simplex(lp.x[:-1])
    if not check(anchor)["passed"]:
        raise ValueError("No certified interior anchor on this complete support")
    identity = hashlib.sha256(features.tobytes() + cal.tobytes() + str(marginal).encode()).hexdigest()
    ambiguity = AmbiguitySet(tuple(identities), phi, lo, hi, identity, check, anchor)
    return ambiguity, dict(rule=calibration_rule,
                           original_K4L4_set=False, calibration_n=cal.shape[1],
                           calibration_mean=mean.tolist(), lower=lo.tolist(), upper=hi.tolist(),
                           full_support=len(identities), feature_rows=k, identity=identity,
                           anchor_check=check(anchor), native_feasibility_status=int(lp.status))
