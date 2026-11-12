"""Explicit, small synthetic moment domain, separate from old L08 calibration."""

from fractions import Fraction
import hashlib
import json
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
    identity = hashlib.sha256(features.tobytes() + cal.tobytes() + lo.tobytes() + hi.tobytes()
                + json.dumps(dict(marginal=marginal, support=list(identities), rule=calibration_rule),
                             sort_keys=True).encode()).hexdigest()
    ambiguity = AmbiguitySet(tuple(identities), phi, lo, hi, identity, check, anchor)
    return ambiguity, dict(rule=calibration_rule,
                           original_K4L4_set=False, calibration_n=cal.shape[1],
                           calibration_mean=mean.tolist(), lower=lo.tolist(), upper=hi.tolist(),
                           full_support=len(identities), feature_rows=k, identity=identity,
                           anchor_check=check(anchor), native_feasibility_status=int(lp.status))


def finite_domain(identities, *, weights=None):
    """Simplex for ARO, or a recorded exact-dyadic singleton for SAA/DET.

    Singleton weights retain their input and machine-rounding discrepancy in the
    returned metadata; the numeric probability domain has an exact unit sum.
    No calibration or optimization occurs in this constructor.
    """
    n = len(identities)
    if n < 1:
        raise ValueError("Empty support")
    anchor = np.zeros(n)
    anchor[0] = 1
    original = None
    if weights is not None:
        original = np.asarray(weights, float)
        if original.shape != (n,) or not np.isfinite(original).all() or np.any(original < 0):
            raise ValueError("Invalid singleton weights")
        if abs(original.sum() - 1) > 1e-12:
            raise ValueError("Sampling weights do not sum to one")
        anchor = dyadic_simplex(original)
    def check(p):
        p = np.asarray(p, float)
        okay = p.shape == (n,) and np.isfinite(p).all() and np.all(p >= 0)
        exact = okay and sum((Fraction(float(v)) for v in p), Fraction(0)) == 1
        member = original is None or np.array_equal(p, anchor)
        return dict(passed=bool(exact and member), exact_simplex=bool(exact), singleton_match=bool(member))
    phi = sparse.csr_matrix((0, n)) if original is None else sparse.eye(n, format="csr")
    low = np.zeros(0) if original is None else anchor.copy()
    kind = "full_simplex_ARO" if original is None else "fixed_sampling_distribution"
    info = dict(kind=kind, support=list(identities), weights=anchor.tolist() if original is not None else None,
                input_weights=original.tolist() if original is not None else None,
                rounding_l1=float(abs(anchor - original).sum()) if original is not None else 0.,
                moment_rows=phi.shape[0], original_322311=False)
    identity = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return AmbiguitySet(tuple(identities), phi, low, low.copy(), identity, check, anchor), info
