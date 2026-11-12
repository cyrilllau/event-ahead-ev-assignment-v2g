"""Original L08 fixed-plan LP certificate, with no filesystem writes."""

import time
import numpy as np
from scipy.optimize import linprog


def certify(mat, z, scene, seconds=None):
    hi = mat.hi.copy()
    ix = mat.index
    hi[ix["short"]] = z[-len(ix["short"]) :]
    hi[ix["grid_p"]] = scene["load_p_kw"].sum(1)
    hi[ix["grid_q"]] = scene["load_q_kvar"].sum(1)
    assert np.isfinite(mat.lo).all() and np.isfinite(hi).all()
    b = mat.b - mat.B @ z
    d = mat.d - mat.G @ z
    opts = dict(threads=1, primal_feasibility_tolerance=1e-09, dual_feasibility_tolerance=1e-09)
    if seconds is not None:
        opts["time_limit"] = max(0.01, seconds)
    started = time.monotonic()
    sol = linprog(
        mat.cost,
        A_ub=mat.A,
        b_ub=b,
        A_eq=mat.H,
        b_eq=d,
        bounds=np.c_[mat.lo, hi],
        method="highs",
        options=opts,
    )
    if not sol.success:
        raise RuntimeError("LP_NOT_CERTIFIED " + str(sol.status) + " " + sol.message)
    y = sol.x
    res = mat.residual(y, z)
    assert res <= 1e-07 and max(y - hi, default=0) <= 1e-07
    lam = np.minimum(sol.ineqlin.marginals, 0.0)
    mu = sol.eqlin.marginals
    red = mat.cost - mat.A.T @ lam - mat.H.T @ mu
    bound_terms = np.minimum(red * mat.lo, red * hi)
    dual = float(lam @ b + mu @ d + bound_terms.sum())
    primal = float(mat.cost @ y)
    magnitude = float(
        abs(lam * b).sum() + abs(mu * d).sum() + abs(bound_terms).sum() + abs(mat.cost * y).sum()
    )
    guard = max(1e-07, 128 * np.finfo(float).eps * max(1, magnitude))
    pguard = max(guard, 10 * res * float(abs(mat.cost).sum()))
    qL = float(np.nextafter(dual - guard, -np.inf))
    qU = float(np.nextafter(primal + pguard, np.inf))
    assert qL <= qU and qU - qL <= 0.001, (qL, qU, res)
    return dict(
        y=y,
        lambda_ub=lam,
        mu_eq=mu,
        reduced_cost=red,
        lower=mat.lo,
        upper=hi,
        solver_lower_marginal=sol.lower.marginals,
        solver_upper_marginal=sol.upper.marginals,
        certificate=dict(
            qL=qL,
            qU=qU,
            primal=primal,
            lagrangian=dual,
            gap=qU - qL,
            primal_residual=res,
            dual_guard=guard,
            primal_guard=pguard,
            numerical_not_exact_rational=True,
            primal_tolerance=1e-07,
            elapsed_s=time.monotonic() - started,
            all_variable_bounds_accounted=True,
            implied_bounds_proof="short<=contract; total P balance with nonnegative V2G/shed gives gridP<=total demand, total Q similarly. Finite-bound Lagrangian minimizes each residual coefficient on its full interval.",
            threads=1,
        ),
    )
