"""Finite-policy, robust inverse costs; never asserts an unseen global ranking."""

import numpy as np


def preference_rows(features, intervals, targets):
    """Robust sufficient comparisons use competitor L minus target U."""
    features = np.asarray(features, float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Finite policy features required")
    rows, rhs, labels = [], [], []
    for method, target in targets.items():
        q = np.asarray(intervals[method], float)
        if q.shape != (len(features), 2) or not np.isfinite(q).all() or np.any(q[:, 0] > q[:, 1]):
            raise ValueError("Complete ordered risk intervals required")
        for j in range(len(features)):
            if j != target:
                rows.append(np.r_[features[target] - features[j], 1.0])
                rhs.append(q[j, 0] - q[target, 1])
                labels.append(dict(method=method, target=target, competitor=j))
    return np.asarray(rows), np.asarray(rhs), labels


def solve_inverse_cost_lp(features, intervals, control, *, bounds=(.8, 1.2),
                          meaningful_margin=1.0):
    """Maximize common margin then minimum dimensionless L1 move to achieve $1.

    Four features are each original COST component, not unweighted mixed units.
    An identically-zero baseline component is fixed at multiplier one. Operational
    costs never change. Nonpositive margin is a mechanism failure, not a license
    to enlarge the box. This function performs real LP calls; remote execution.
    """
    from scipy.optimize import linprog
    features = np.asarray(features, float)
    A, b, labels = preference_rows(features, intervals,
                                   {"DET": 0, "Full-DRO": 1, "Marginal-DRO": 2, "ARO": 2})
    d = features.shape[1]
    box = [(1, 1) if not np.any(features[:, j]) else bounds for j in range(d)]
    def options():
        control.check()
        return dict(threads=1, time_limit=max(.01, control.remaining),
                    primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
    first = linprog(np.r_[np.zeros(d), -1.], A_ub=A, b_ub=b,
                    bounds=box + [(None, None)], method="highs", options=options())
    if not first.success:
        return dict(status="INVERSE_LP_NOT_OPTIMAL", native_status=int(first.status),
                    message=first.message, passed=False)
    delta = float(first.x[-1])
    result = dict(max_margin=delta, max_margin_weights=first.x[:d].tolist(),
                  targets=labels, cost_box=[list(v) for v in box],
                  first_native_status=int(first.status), passed=False)
    if delta < meaningful_margin:
        return dict(result, status="NO_NONEMPTY_MECHANISM_REGION" if delta <= 0
                    else "BELOW_PREREGISTERED_ONE_DOLLAR_MARGIN")
    # Keep meaningful margin, then minimize relative L1 change. Multipliers have
    # baseline one, hence equal L1 weights implement normalization by base rate.
    AA = np.vstack([np.c_[A[:, :d], np.zeros((len(b), d))],
                    np.c_[np.eye(d), -np.eye(d)], np.c_[-np.eye(d), -np.eye(d)]])
    bb = np.r_[b - meaningful_margin, np.ones(d), -np.ones(d)]
    second = linprog(np.r_[np.zeros(d), np.ones(d)], A_ub=AA, b_ub=bb,
                     bounds=box + [(0, None)] * d, method="highs", options=options())
    if not second.success:
        return dict(result, status="L1_STAGE_NOT_OPTIMAL", second_native_status=int(second.status))
    w = second.x[:d]
    margins = b - A[:, :d] @ w
    if np.min(margins) < meaningful_margin - 1e-7:
        raise ValueError("Inverse LP primal failed original margin check")
    return dict(result, passed=True, status="FINITE_REFERENCE_POLICY_MECHANISM_PASS",
                weights=w.tolist(), margin=float(np.min(margins)), relative_l1=float(abs(w - 1).sum()),
                second_native_status=int(second.status), global_ranking_claim=False)
