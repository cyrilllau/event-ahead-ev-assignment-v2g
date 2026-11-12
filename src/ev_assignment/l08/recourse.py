"""Lazy global-domain recourse, finite bounds, and original cut certificates."""

from dataclasses import dataclass
import numpy as np
from scipy import sparse

from .builders import first_stage, add_scene, arrays, parameters
from .certificates import finite_bounds, row_audit, replay_cut, enclosure


@dataclass(frozen=True)
class RecourseTemplate:
    W: sparse.csr_matrix
    T: sparse.csr_matrix
    data: dict
    names: tuple[str, ...]
    bound_proof: tuple[dict, ...]
    qbar: float


def build_template(problem, sid, *, env):
    model, first = first_stage(problem, env=env)
    try:
        base = arrays(model)
        nx = model.NumVars
        nr = model.NumConstrs
        rc = add_scene(model, first, problem, sid)
        model.setObjective(rc["objective"])
        whole = arrays(model)
        if np.any(whole["cost"][:nx]) or np.min(whole["cost"][nx:]) < 0:
            raise ValueError("Recourse objective is not nonnegative and separate")
        if any(v != "C" for v in whole["vtype"][nx:]):
            raise ValueError("Recourse must be continuous")
        lo = whole["lo"].copy()
        hi = whole["hi"].copy()
        lo[lo <= -1e90] = -np.inf
        hi[hi >= 1e90] = np.inf
        fl, fh, proof = finite_bounds(
            whole["A"], whole["rhs"], whole["sense"], lo, hi, whole["names"]
        )
        if not np.isfinite(fl).all() or not np.isfinite(fh).all():
            raise ValueError("Unproven finite global-domain bounds")
        W = whole["A"][nr:, nx:].tocsr()
        T = whole["A"][nr:, :nx].tocsr()
        a = dict(
            h=whole["rhs"][nr:],
            sense=whole["sense"][nr:],
            c=whole["cost"][nx:],
            lo=lo[nx:],
            hi=hi[nx:],
            finite_lo=fl[nx:],
            finite_hi=fh[nx:],
            xlo=fl[:nx],
            xhi=fh[:nx],
        )
        qbar = float(
            np.nextafter(
                float(np.asarray(a["c"], np.longdouble) @ np.asarray(fh[nx:], np.longdouble))
                + 1e-6,
                np.inf,
            )
        )
        if base["names"] != whole["names"][:nx] or (base["A"] != whole["A"][:nr, :nx]).nnz:
            raise ValueError("Scenario changed the shared first stage")
        return RecourseTemplate(W, T, a, tuple(whole["names"][nx:]), tuple(proof), qbar)
    finally:
        model.dispose()


def audit(template, x, y, pi):
    a = template.data
    check = row_audit(template.W, a["h"] - template.T @ x, a["sense"], a["lo"], a["hi"], y)
    if max(check.values()) > 1e-9:
        raise ValueError(("Original recourse residual", check))
    alpha, beta, corrected = replay_cut(template.W, template.T, a, pi)
    ql = float(
        np.longdouble(alpha) + np.asarray(beta, np.longdouble) @ np.asarray(x, np.longdouble)
    )
    _, up = enclosure(sparse.csr_matrix(a["c"][None, :]), y)
    qu = float(np.nextafter(float(up[0]), np.inf))
    if qu - ql < -1e-5:
        raise ValueError("Negative signed recourse certificate gap")
    return dict(
        residual=check,
        alpha=alpha,
        beta=beta,
        Pi=corrected,
        qL=ql,
        qU_audited=qu,
        qU_independently_certified=None,
        UB_status="NUMERICALLY_AUDITED_ONLY",
        global_cut_valid=True,
    )


def solve(template, x, options, control, *, env):
    import gurobipy as gp

    control.check()
    a = template.data
    with gp.Model("L08_RECOURSE_ORACLE", env=env) as m:
        y = m.addMVar(len(a["c"]), lb=a["lo"], ub=a["hi"])
        m.addMConstr(template.W, y, a["sense"], a["h"] - template.T @ x)
        m.setObjective(a["c"] @ y)
        parameters(m, options, control.remaining)
        m.Params.Threads = 1
        control.event(dict(kind="native_begin", stage="recourse"))
        m.optimize(lambda model, where: model.terminate() if control.stop() else None)
        control.event(
            dict(kind="native_end", stage="recourse", native_status=m.Status, runtime=m.Runtime)
        )
        if m.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("Recourse is not optimal: " + str(m.Status))
        yy = np.asarray(y.X)
        pi = np.asarray(m.getAttr("Pi"))
        result = audit(template, x, yy, pi)
        result.update(y=yy, raw_pi=pi, native_status=m.Status, runtime=m.Runtime)
        return result
