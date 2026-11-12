"""Full-ambiguity probability separation and guarded dual upper certificate."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .certificates import enclosure
from .probability_certificate import certify_with_feasible_anchor, dyadic_simplex


def solve(ambiguity, q_lower, q_upper, qbars, control):
    control.check()
    q_lower = np.asarray(q_lower, float)
    q_upper = np.asarray(q_upper, float)
    n = len(ambiguity.support_ids)
    if (
        q_lower.shape != (n,)
        or q_upper.shape != (n,)
        or not np.isfinite(q_upper).all()
        or np.any(q_lower > q_upper)
    ):
        raise ValueError("Incomplete or invalid full-support risk vectors")
    normal = max(1.0, float(abs(q_lower).max()))
    A = sparse.vstack([ambiguity.phi, -ambiguity.phi], format="csc")
    b = np.r_[ambiguity.upper, -ambiguity.lower]
    lp = linprog(
        -q_lower / normal,
        A_ub=A,
        b_ub=b,
        A_eq=np.ones((1, n)),
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
        options=dict(
            time_limit=max(0.01, control.remaining),
            primal_feasibility_tolerance=1e-9,
            dual_feasibility_tolerance=1e-9,
            threads=1,
        ),
    )
    if not lp.success:
        raise RuntimeError("Full-ambiguity probability LP not optimal")
    m = len(ambiguity.lower)
    plus = np.maximum(-lp.ineqlin.marginals[:m] * normal, 0.0)
    minus = np.maximum(-lp.ineqlin.marginals[m:] * normal, 0.0)
    beta0 = float(-lp.eqlin.marginals[0] * normal)
    plane = beta0 + np.asarray(ambiguity.phi.T @ (plus - minus)).ravel()
    magnitude = abs(beta0) + float(
        abs(ambiguity.upper * plus).sum() + abs(ambiguity.lower * minus).sum()
    )
    guard = max(1e-7, 256 * np.finfo(float).eps * max(1.0, magnitude))
    shift = max(0.0, float(np.max(q_upper - plane))) + guard
    beta0 += shift
    plane += shift
    if np.min(plane - q_upper) < 0:
        raise ValueError("Risk dual does not dominate every scenario")
    upper = float(
        np.nextafter(beta0 + ambiguity.upper @ plus - ambiguity.lower @ minus + guard, np.inf)
    )
    anchor = ambiguity.feasible_anchor
    if anchor is None:
        anchor = dyadic_simplex(lp.x)
        if not ambiguity.check_original(anchor).get("passed"):
            raise ValueError("A verified original feasible anchor is required")
    pstar, repair = certify_with_feasible_anchor(
        lp.x, anchor, qbars, list(range(n)), ambiguity.check_original
    )
    lower = enclosure(sparse.csr_matrix(q_lower[None, :]), pstar)[0][0]
    return dict(
        pstar=pstar,
        risk_L=float(np.nextafter(float(lower), -np.inf)),
        risk_U=upper,
        beta0=beta0,
        plus=plus,
        minus=minus,
        plane=plane,
        guard=guard,
        original_probability_check=ambiguity.check_original(pstar),
        repair=repair,
        all_support_ids=list(ambiguity.support_ids),
    )
