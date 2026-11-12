"""Independent explicit task/physics audit of a compact L08 solution."""

import numpy as np
from ev_assignment.domain.planning import AuditReport
from ev_assignment.l08.physics import task, components
from ev_assignment.l08.energy import battery_trace
from ev_assignment.l08.matrix import assemble, first_stage
from ev_assignment.l08.fixed_plan import certify


def audit_plan(
    problem,
    entries,
    control,
    *,
    raw_recourse=None,
    record_scenario=None,
    retain_witness=False,
    witness_provider=None,
):
    routes = {r["route_id"]: r for r in problem.routes}
    seen = set()
    tasks = []
    contracts = np.zeros(problem.fleet_size)
    for entry in entries:
        q = routes[entry["route_id"]]
        e = q["e"]
        if e in seen or entry["e"] != e:
            raise ValueError("Duplicate or mismatched vehicle in plan")
        seen.add(e)
        for key in ("c", "d", "arrival", "distance", "departure_dc"):
            if entry[key] != q[key]:
                raise ValueError("Plan changed a frozen route")
        if entry["k"] not in problem.starts or entry["r"] + 1 not in problem.ends:
            raise ValueError("Plan outside frozen time domain")
        t = task(q, entry["k"], entry["r"])
        if not 15 - 1e-6 <= entry["contract"] <= t["Ebar"] + 1e-6:
            raise ValueError("Plan violates procurement/energy bound")
        tasks.append(t)
        contracts[e] = entry["contract"]
    z = np.r_[np.ones(len(tasks)), contracts]
    A, b, parts, _, _ = first_stage(
        tasks, problem.fleet_size, len(problem.network["station_buses"]), problem.energy_bounds
    )
    first_residual = float(np.max(A @ z - b, initial=0))
    if first_residual > 1e-6:
        raise ValueError("First-stage physics failed")
    rows = []
    route_index = {q["route_id"]: i for i, q in enumerate(problem.routes)}
    for sid in problem.scenario_ids:
        control.check()
        sc = problem.load_scenario(sid)
        states = [sc.route_states[route_index[t["route_id"]]] for t in tasks]
        mat = assemble(
            problem.network, tasks, sc.physical, states, problem.fleet_size, problem.energy_bounds
        )
        if raw_recourse is not None:
            if mat.residual(raw_recourse[sid], z) > 1e-6:
                raise ValueError("Compact witness violates original physical equations")
        sol = (
            check_witness(mat, z, sc.physical, witness_provider(sid))
            if witness_provider
            else certify(mat, z, sc.physical, seconds=control.remaining)
        )
        y = sol["y"]
        pd = y[mat.index["p"]]
        traces = []
        for t, st in zip(tasks, states):
            if st["success"] and st["arrival"] < (t["r"] + 1) / 4:
                trace = battery_trace(
                    t["B"] + 100,
                    st["arrival"],
                    (t["r"] + 1) / 4,
                    pd[t["e"]].sum(0),
                    extra=[t["k"] / 4],
                )
                if trace["min_dc"] < 100 - 1e-6:
                    raise ValueError("Battery departure floor violated")
                traces.append(dict(e=t["e"], **trace))
        connected = 0.0
        for k in range(96):
            root = next(
                g
                for g in components(problem.network, sc.physical["line_operating_outages"][k])
                if 1 in g
            )
            for c, bus in enumerate(problem.network["station_buses"]):
                if bus in root:
                    connected = max(connected, float(pd[:, c, k].sum()))
        if connected > 1e-6:
            raise ValueError("Positive-cost V2G in grid-connected component")
        shed = y[mat.index["shed"]]
        critical = np.asarray(problem.network["shed_cost"]) == 50
        row = dict(
            sid=sid,
            **sol["certificate"],
            cost_parts=(mat.parts @ y).tolist(),
            critical_ENS=float(0.25 * shed[:, critical].sum()),
            other_ENS=float(0.25 * shed[:, ~critical].sum()),
            actual_V2G_ac=float(0.25 * pd.sum()),
            shortfall_ac=float(y[mat.index["short"]].sum()),
            max_grid_connected_V2G_kw=connected,
            traces=traces,
            y=y.tolist(),
            z=z.tolist(),
            lambda_ub=sol["lambda_ub"].tolist(),
            mu_eq=sol["mu_eq"].tolist(),
            lower=sol["lower"].tolist(),
            upper=sol["upper"].tolist(),
        )
        reference = record_scenario(row) if record_scenario else None
        if not retain_witness:
            row = {
                k: v
                for k, v in row.items()
                if k not in {"y", "z", "lambda_ub", "mu_eq", "lower", "upper"}
            }
        if reference is not None:
            row["witness"] = reference
        rows.append(row)
    return AuditReport(
        True,
        dict(
            problem=problem.identity,
            first_residual=first_residual,
            first_cost=float(parts.sum(0) @ z),
            first_cost_parts=(parts @ z).tolist(),
            scenarios=rows,
            scenario_order=list(problem.scenario_ids),
            original_physics=True,
        ),
    )


def check_witness(mat, z, scene, witness):
    """Recompute a finite-bound LP certificate without executing any solver."""
    keys = ("y", "z", "lambda_ub", "mu_eq", "lower", "upper")
    w = {k: np.asarray(witness[k], float) for k in keys}
    shapes = dict(
        y=mat.cost.shape,
        z=z.shape,
        lambda_ub=mat.b.shape,
        mu_eq=mat.d.shape,
        lower=mat.lo.shape,
        upper=mat.hi.shape,
    )
    if any(w[k].shape != shapes[k] or not np.isfinite(w[k]).all() for k in keys):
        raise ValueError("Invalid witness dimensions or nonfinite values")
    hi = mat.hi.copy()
    hi[mat.index["short"]] = z[-len(mat.index["short"]) :]
    hi[mat.index["grid_p"]] = scene["load_p_kw"].sum(1)
    hi[mat.index["grid_q"]] = scene["load_q_kvar"].sum(1)
    if not all(
        np.array_equal(a, b) for a, b in ((w["z"], z), (w["lower"], mat.lo), (w["upper"], hi))
    ):
        raise ValueError("Witness changed plan or implied physical bounds")
    y, lam, mu = w["y"], w["lambda_ub"], w["mu_eq"]
    residual = mat.residual(y, z)
    if residual > 1e-7 or np.max(y - hi, initial=0) > 1e-7 or np.max(lam, initial=0) > 0:
        raise ValueError("Invalid primal or dual witness")
    b, d = mat.b - mat.B @ z, mat.d - mat.G @ z
    red = mat.cost - mat.A.T @ lam - mat.H.T @ mu
    terms = np.minimum(red * mat.lo, red * hi)
    dual, primal = float(lam @ b + mu @ d + terms.sum()), float(mat.cost @ y)
    magnitude = float(
        abs(lam * b).sum() + abs(mu * d).sum() + abs(terms).sum() + abs(mat.cost * y).sum()
    )
    guard = max(1e-7, 128 * np.finfo(float).eps * max(1, magnitude))
    pguard = max(guard, 10 * residual * float(abs(mat.cost).sum()))
    lo, up = (
        float(np.nextafter(dual - guard, -np.inf)),
        float(np.nextafter(primal + pguard, np.inf)),
    )
    if not 0 <= up - lo <= 0.001:
        raise ValueError("Original LP certificate tolerance failed")
    for key, val in dict(qL=lo, qU=up, primal=primal, lagrangian=dual).items():
        if not np.isclose(witness[key], val, rtol=0, atol=1e-8):
            raise ValueError("Stored certificate differs from recomputation: " + key)
    cert = dict(
        qL=lo,
        qU=up,
        primal=primal,
        lagrangian=dual,
        gap=up - lo,
        primal_residual=residual,
        numerical_not_exact_rational=True,
    )
    return dict(w, certificate=cert)


def check_risk(ambiguity, upper_costs, evidence):
    """Independently verify every original probability row and upper dual plane."""
    if evidence["all_support_ids"] != list(ambiguity.support_ids):
        raise ValueError("Risk support order changed")
    p = np.asarray(evidence["pstar"], float)
    plus, minus = np.asarray(evidence["plus"], float), np.asarray(evidence["minus"], float)
    if plus.shape != ambiguity.upper.shape or minus.shape != ambiguity.lower.shape:
        raise ValueError("Risk dual shape changed")
    if (
        not np.isfinite(np.r_[plus, minus, evidence["beta0"]]).all()
        or min(plus.min(initial=0), minus.min(initial=0)) < 0
    ):
        raise ValueError("Invalid risk dual")
    if not ambiguity.check_original(p).get("passed"):
        raise ValueError("Original ambiguity check failed")
    plane = evidence["beta0"] + np.asarray(ambiguity.phi.T @ (plus - minus)).ravel()
    if np.min(plane - np.asarray(upper_costs)) < 0:
        raise ValueError("Risk plane misses a scenario")
    upper = float(evidence["beta0"] + ambiguity.upper @ plus - ambiguity.lower @ minus)
    if not np.isfinite(evidence["risk_U"]) or evidence["risk_U"] < upper:
        raise ValueError("Risk upper bound is below its certificate")
    return evidence["risk_U"]


def check_first_vector(problem, entries, vector):
    """Reconstruct the compact variable order from explicit route/window choices."""
    E, C = problem.fleet_size, len(problem.network["station_buses"])
    routes = {q["route_id"]: i for i, q in enumerate(problem.routes)}
    v = np.zeros(len(routes))
    s = np.zeros((E, len(problem.starts)))
    f = np.zeros((E, len(problem.ends)))
    u = np.zeros(E)
    K = np.zeros(E)
    J = np.zeros(E)
    en = np.zeros(E)
    w = np.zeros((E, 96))
    b = np.zeros((E, C))
    wc = np.zeros((E, C, 96))
    arrival_prefix = np.zeros((E, 97))
    start_prefix = np.zeros((E, 97))
    end_prefix = np.zeros((E, 97))
    for entry in entries:
        e, c = entry["e"], entry["c"]
        k, j = entry["k"], entry["r"] + 1
        if u[e]:
            raise ValueError("Duplicate vehicle in compact vector")
        v[routes[entry["route_id"]]] = 1
        s[e, problem.starts.index(k)] = 1
        f[e, problem.ends.index(j)] = 1
        u[e] = 1
        K[e] = k / 4
        J[e] = j / 4
        en[e] = entry["contract"]
        w[e, k:j] = 1
        b[e, c] = 1
        wc[e, c, k:j] = 1
        arrival_slot = int(np.searchsorted(np.arange(97) / 4, entry["arrival"]))
        arrival_prefix[e, arrival_slot:] = 1
        start_prefix[e, k:] = 1
        end_prefix[e, j:] = 1
    expected = np.concatenate(
        [
            a.ravel()
            for a in (v, s, f, u, K, J, en, w, b, wc, arrival_prefix, start_prefix, end_prefix)
        ]
    )
    actual = np.asarray(vector, float)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or np.max(abs(actual - expected), initial=0) > 1e-9
    ):
        raise ValueError("Compact first-stage vector does not match the audited plan")
    return True
