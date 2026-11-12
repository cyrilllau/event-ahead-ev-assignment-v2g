"""L08 DET and finite-support DRO/ARO controllers with explicit state and hooks.

DRO retains per-scenario theta supports and full-ambiguity probability separation;
ARO retains a shared integer first stage and one max epigraph. Numerical and
strict certification levels remain distinct. No campaign paths or persistence.
"""

from __future__ import annotations
import hashlib
import time
import numpy as np
from scipy import sparse

from ev_assignment.domain.planning import (
    Bound,
    BoundScope,
    Control,
    DecisionPlan,
    SolveResult,
    AuditReport,
    combine_bounds,
)
from ev_assignment.l08 import builders, native, recourse, risk, theta
from ev_assignment.l08.certificates import enclosure
from ev_assignment.l08.point import canonical
from ev_assignment.l08.reduction import lift
from ev_assignment.validation.l08 import audit_plan


def point_id(x):
    return hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest()


def finite_bound(value):
    return (
        float(value) - 1e-6
        if value is not None and np.isfinite(value) and abs(value) < 1e90
        else None
    )


def _target(lower, upper, options):
    if lower is None or upper is None:
        return False
    gap = upper - lower
    if gap < 0:
        raise ValueError("Negative external gap requires scientific review")
    return gap <= options.absolute_gap or gap / max(1.0, abs(upper)) <= options.mip_gap


def solve_det(problem, options, *, control=None, checkpoint=None, strengthening=None):
    import gurobipy as gp

    control = control or Control(options.seconds)
    begun = time.monotonic()
    scope = BoundScope(problem.identity, "DET", problem.scenario_ids)
    if checkpoint is not None and (
        checkpoint.get("problem") != problem.identity or checkpoint.get("method") != "DET"
    ):
        raise ValueError("DET checkpoint science mismatch")
    with gp.Env(params={"OutputFlag": 0}) as env:
        m, fs, copies, _ = builders.extensive(problem, "DET", options, env=env)
        try:
            if strengthening is not None:
                strengthening(m, fs, copies)
            m.update()
            nx = sum(1 for name in m.getAttr("VarName") if not name.startswith("s0_"))
            if checkpoint and checkpoint.get("native_start") is not None:
                start = np.asarray(checkpoint["native_start"], float)
                if start.shape != (m.NumVars,):
                    raise ValueError("DET checkpoint model dimension changed")
                m.setAttr("Start", m.getVars(), start.tolist())
            control.check()
            builders.parameters(m, options, control.remaining)
            control.event(dict(kind="native_begin", stage="det"))
            m.optimize(lambda model, where: model.terminate() if control.stop() else None)
            control.event(
                dict(kind="native_end", stage="det", native_status=m.Status, runtime=m.Runtime)
            )
            lower = None
            if m.Status not in (gp.GRB.NUMERIC, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED):
                try:
                    lower = finite_bound(m.ObjBoundC)
                except gp.GurobiError:
                    pass
            if not m.SolCount:
                return SolveResult(
                    "DET",
                    m.Status,
                    "NO_INCUMBENT",
                    None,
                    Bound(lower, scope, "native-master", "native_solver")
                    if lower is not None
                    else None,
                    None,
                    None,
                    time.monotonic() - begun,
                    False,
                    dict(problem=problem.identity, method="DET"),
                )
            x = np.asarray(m.getAttr("X"))
            control.event(
                dict(
                    kind="checkpoint",
                    state=dict(
                        problem=problem.identity,
                        method="DET",
                        native_start=x.tolist(),
                        lower=lower,
                        upper=None,
                        branch_tree_restored=False,
                    ),
                )
            )
            entries = tuple(native.decode(fs, x))
            raw = {sid: lift(rc["y"].X, rc["pfmap"]) for sid, rc in copies.items()}
            audit = audit_plan(
                problem,
                entries,
                control,
                raw_recourse=raw,
                record_scenario=lambda row: control.event(
                    dict(kind="physical_evidence", evidence=row)
                ),
            )
            upper = audit.checks["first_cost"] + audit.checks["scenarios"][0]["qU"]
            if checkpoint and checkpoint.get("lower") is not None:
                lower = checkpoint["lower"] if lower is None else max(lower, checkpoint["lower"])
            state = dict(
                problem=problem.identity,
                method="DET",
                native_start=x.tolist(),
                lower=lower,
                upper=upper,
                branch_tree_restored=False,
            )
            control.event(dict(kind="checkpoint", state=state))
            return SolveResult(
                "DET",
                m.Status,
                "RETURNED_WITH_PHYSICAL_AUDIT",
                DecisionPlan(problem.identity, entries, x[:nx]),
                Bound(lower, scope, "native-master", "native_solver")
                if lower is not None
                else None,
                Bound(upper, scope, "original-physical-audit", "numerical_audit"),
                audit,
                time.monotonic() - begun,
                _target(lower, upper, options),
                state,
            )
        finally:
            m.dispose()


def solve_decomposition(
    problem, method, options, *, control=None, checkpoint=None, template_provider=None
):
    import gurobipy as gp

    if method not in {"DRO", "ARO"} or (method == "DRO" and problem.ambiguity is None):
        raise ValueError("Invalid decomposition method or ambiguity set")
    control = control or Control(options.seconds)
    begun = time.monotonic()
    scope = BoundScope(problem.identity, method, problem.scenario_ids)
    state = dict(
        schema_version=1,
        algorithm="L08_THETA_AND_MAX_V1",
        problem=problem.identity,
        method=method,
        scenario_ids=list(problem.scenario_ids),
        phase="EVALUATE",
        point=None,
        partial=[],
        cuts=[],
        probabilities=[],
        lower=None,
        upper=None,
        best=None,
        history=[],
        completed={},
        outer=0,
        native_status=None,
    )
    if checkpoint is not None:
        for key in ("schema_version", "algorithm", "problem", "method", "scenario_ids"):
            if checkpoint.get(key) != state[key]:
                raise ValueError("Checkpoint science/algorithm mismatch: " + key)
        state = dict(checkpoint)
    termination = "BUDGET_LIMIT"
    audit_report = None
    best_plan = None
    with gp.Env(params={"OutputFlag": 0}) as env:
        m, fs = builders.first_stage(problem, env=env)
        fa = builders.arrays(m)
        n = m.NumVars
        try:
            columns = m.getVars()
            xvar = gp.MVar.fromlist(columns)
            rho = m.addVar(lb=0, name="DRO_RHO" if method == "DRO" else "ARO_ETA")
            theta_vars = (
                {
                    sid: m.addVar(lb=0, name="theta_" + str(i))
                    for i, sid in enumerate(problem.scenario_ids)
                }
                if method == "DRO"
                else {}
            )
            m.setObjective(fs["objective"] + rho)
            m.update()

            def put_cut(c):
                beta = sparse.csr_matrix(
                    (c["values"], ([0] * len(c["indices"]), c["indices"])), shape=(1, n)
                )
                if method == "DRO":
                    _, _, proof = theta.add_supports(
                        m,
                        m.getVars(),
                        n,
                        theta_vars[c["sid"]].index,
                        beta,
                        np.array([c["alpha"]]),
                        [c["id"]],
                        fa["lo"],
                        fa["hi"],
                        "support/" + c["sid"],
                    )
                else:
                    m.addConstr(
                        rho
                        >= c["alpha"]
                        + gp.LinExpr(c["values"], [columns[int(i)] for i in c["indices"]]),
                        name="support/" + c["sid"] + "/" + c["id"],
                    )

            def put_probability(p):
                m.update()
                theta.add_probability(
                    m,
                    m.getVars(),
                    rho.index,
                    {i: theta_vars[s].index for i, s in enumerate(problem.scenario_ids)},
                    np.asarray(p),
                    "probability/" + point_id(p),
                )

            for c in state["cuts"]:
                put_cut(c)
            for p in state["probabilities"]:
                put_probability(p)
            m.update()
            if state["point"] is None:
                state["point"] = canonical(np.zeros(n), fa, control=control, env=env).tolist()

            def save():
                control.event(dict(kind="checkpoint", state=state))

            def get_template(sid):
                return (
                    template_provider(sid, env)
                    if template_provider
                    else recourse.build_template(problem, sid, env=env)
                )

            while True:
                try:
                    control.check()
                except (TimeoutError, InterruptedError):
                    break
                if state["phase"] == "EVALUATE":
                    x = canonical(state["point"], fa, control=control, env=env)
                    pid = point_id(x)
                    records = list(state["partial"])
                    known = {r["sid"] for r in records}
                    try:
                        for sid in problem.scenario_ids:
                            if sid in known:
                                continue
                            control.check()
                            template = get_template(sid)
                            r = recourse.solve(template, x, options, control, env=env)
                            nz = np.flatnonzero(r["beta"])
                            cut = dict(
                                sid=sid,
                                alpha=r["alpha"],
                                indices=nz.tolist(),
                                values=r["beta"][nz].tolist(),
                                id=point_id(np.r_[r["alpha"], r["beta"]]),
                            )
                            witness = dict(
                                sid=sid,
                                point=x,
                                y=r["y"],
                                Pi=r["raw_pi"],
                                cut=cut,
                                residual=r["residual"],
                                qL=r["qL"],
                                qU=r["qU_audited"],
                            )
                            ref = control.event(dict(kind="recourse_evidence", evidence=witness))
                            records.append(
                                dict(
                                    sid=sid,
                                    qL=r["qL"],
                                    qU=r["qU_audited"],
                                    qbar=template.qbar,
                                    cut=cut,
                                    evidence=ref,
                                )
                            )
                            state["partial"] = records
                            save()
                    except (TimeoutError, InterruptedError):
                        break
                    if len(records) != len(problem.scenario_ids):
                        raise ValueError("Partial evaluation is not a bound")
                    by = {r["sid"]: r for r in records}
                    records = [by[s] for s in problem.scenario_ids]
                    F = float(enclosure(sparse.csr_matrix(fa["cost"][None, :]), x)[1][0])
                    ql = np.array([r["qL"] for r in records])
                    qu = np.array([r["qU"] for r in records])
                    if method == "DRO":
                        rr = risk.solve(
                            problem.ambiguity, ql, qu, [r["qbar"] for r in records], control
                        )
                        upper = float(np.nextafter(F + rr["risk_U"], np.inf))
                        p = rr["pstar"].tolist()
                        if p not in state["probabilities"]:
                            put_probability(p)
                            state["probabilities"].append(p)
                        control.event(dict(kind="risk_evidence", evidence=rr))
                    else:
                        upper = float(
                            np.nextafter(np.longdouble(F) + max(map(np.longdouble, qu)), np.inf)
                        )
                    if state["upper"] is None or upper < state["upper"]:
                        state["upper"] = upper
                        state["best"] = x.tolist()
                    cut_ids = {(c["sid"], c["id"]) for c in state["cuts"]}
                    added = 0
                    for r in records:
                        c = r["cut"]
                        if (c["sid"], c["id"]) not in cut_ids:
                            put_cut(c)
                            state["cuts"].append(c)
                            cut_ids.add((c["sid"], c["id"]))
                            added += 1
                    state["completed"][pid] = dict(upper=upper, cut_count=len(state["cuts"]))
                    state["history"].append(
                        dict(
                            outer=state["outer"],
                            lower=state["lower"],
                            upper=state["upper"],
                            new_cuts=added,
                        )
                    )
                    state.update(phase="SOLVE", partial=[])
                    save()
                    if _target(state["lower"], state["upper"], options):
                        termination = "NUMERICAL_GAP_TARGET"
                        break
                control.check()
                builders.parameters(m, options, min(control.remaining, options.master_seconds))
                m.update()
                control.event(dict(kind="native_begin", stage="master", outer=state["outer"]))
                m.optimize(lambda model, where: model.terminate() if control.stop() else None)
                state["native_status"] = m.Status
                control.event(
                    dict(
                        kind="native_end", stage="master", native_status=m.Status, runtime=m.Runtime
                    )
                )
                if m.Status in (
                    gp.GRB.INFEASIBLE,
                    gp.GRB.INF_OR_UNBD,
                    gp.GRB.UNBOUNDED,
                    gp.GRB.NUMERIC,
                ):
                    raise ValueError("Master native correctness failure: " + str(m.Status))
                try:
                    bound = finite_bound(m.ObjBoundC)
                except gp.GurobiError:
                    bound = None
                if bound is not None:
                    state["lower"] = bound if state["lower"] is None else max(bound, state["lower"])
                save()
                if _target(state["lower"], state["upper"], options):
                    termination = "NUMERICAL_GAP_TARGET"
                    break
                if not m.SolCount:
                    continue
                x = canonical(xvar.X, fa, control=control, env=env)
                pid = point_id(x)
                if pid in state["completed"] and m.Status == gp.GRB.OPTIMAL:
                    termination = "NO_PROGRESS_DIAGNOSTIC"
                    break
                state.update(
                    point=x.tolist(), phase="EVALUATE", partial=[], outer=state["outer"] + 1
                )
                save()
            if state["best"] is not None:
                x = np.asarray(state["best"])
                entries = tuple(native.decode(fs, x))
                best_plan = DecisionPlan(problem.identity, entries, x)
                # Physical audit is independent of the compact/global-cut equations.
                if control.remaining > 0 and not control.stop():
                    audit_report = audit_plan(
                        problem,
                        entries,
                        control,
                        record_scenario=lambda row: control.event(
                            dict(kind="physical_evidence", evidence=row)
                        ),
                    )
                    rows = audit_report.checks["scenarios"]
                    qu = np.array([r["qU"] for r in rows])
                    if method == "DRO":
                        rr = risk.solve(
                            problem.ambiguity, np.array([r["qL"] for r in rows]), qu, qu, control
                        )
                        audit_report = AuditReport(True, dict(audit_report.checks, risk=rr))
                        risk_upper = rr["risk_U"]
                    else:
                        risk_upper = float(qu.max())
                    # Final upper is bound to the independently replayable physical witness.
                    state["upper"] = float(
                        np.nextafter(audit_report.checks["first_cost"] + risk_upper, np.inf)
                    )
                    save()
            lower = (
                Bound(state["lower"], scope, "native-master", "native_solver")
                if state["lower"] is not None
                else None
            )
            upper = (
                Bound(
                    state["upper"], scope, "complete-support-recourse-and-risk", "numerical_audit"
                )
                if state["upper"] is not None
                else None
            )
            combine_bounds(lower, upper)
            return SolveResult(
                method,
                state["native_status"],
                termination,
                best_plan,
                lower,
                upper,
                audit_report,
                time.monotonic() - begun,
                _target(state["lower"], state["upper"], options),
                state,
            )
        finally:
            m.dispose()


def solve(problem, method, options, **kwargs):
    return (
        solve_det(problem, options, **kwargs)
        if method == "DET"
        else solve_decomposition(problem, method, options, **kwargs)
    )
