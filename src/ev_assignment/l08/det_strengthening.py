"""Exact forecast-only screening and the original 84 integer capacity cuts."""

from fractions import Fraction as F
import numpy as np
from .physics import components
from .matrix import assemble


def verify(problem, proof, pfmap):
    if problem.scenario_ids != ("forecast44",) or problem.fleet_size != 30:
        raise ValueError("R38 proof applies only to the locked DET44 integer domain")
    net = problem.network
    scene = problem.load_scenario("forecast44")
    if net["shed_cost"] != [
        50.0 if b in {2, 4, 8, 10, 14, 21, 24, 27, 30, 32} else 10.0 for b in net["buses"]
    ]:
        raise ValueError("Capacity proof price policy changed")
    cert = proof["screening"]
    required = [
        "single_forecast",
        "root_zero_feasible",
        "zero_root_price",
        "nonnegative_costs",
        "no_mandatory_purchase",
        "no_mandatory_discharge",
        "components_separable",
        "capability_shortfall",
        "no_grid_temporal_coupling",
        "forecast_equals_recourse",
    ]
    if not cert["passed"] or not all(cert["prerequisites"].get(k) is True for k in required):
        raise ValueError("Screening prerequisites missing")
    if net["big_m"] + 1e-12 < net["u_max"] - net["u_min"]:
        raise ValueError("Outage voltage rows couple components")
    for q, st in zip(problem.routes, scene.route_states):
        if (
            not st["success"]
            or st["arrival"] != q["arrival"]
            or not q["d"] / 4 <= q["arrival"] <= 23
        ):
            raise ValueError("Forecast/recourse or nonnegative participation changed")
    mat = assemble(net, [], scene.physical, [], problem.fleet_size, problem.energy_bounds)
    zero = np.asarray(proof["zero_y"])
    if (
        mat.residual(zero, np.zeros(problem.fleet_size)) > 1e-6
        or np.max(abs(zero[mat.index["p"]])) != 0
    ):
        raise ValueError("Zero replacement witness failed original physics")
    if (
        np.min(mat.cost) < 0
        or np.max(abs(mat.cost[mat.index["grid_p"]])) != 0
        or np.max(abs(mat.cost[mat.index["grid_q"]])) != 0
    ):
        raise ValueError("Screening cost prerequisites changed")
    bi = {b: i for i, b in enumerate(net["buses"])}
    root = np.zeros((96, len(bi)), bool)
    blocks = []
    for t in range(96):
        for comp in sorted(
            components(net, scene.physical["line_operating_outages"][t]), key=lambda c: min(c)
        ):
            if 1 in comp:
                root[t, [bi[b] for b in comp]] = True
                if np.max(zero[mat.index["shed"]][t, [bi[b] for b in comp]], initial=0) > 1e-6:
                    raise ValueError("Root replacement sheds load")
            else:
                cs = [c for c, b in enumerate(net["station_buses"]) if b in comp]
                if cs:
                    blocks.append(dict(t=t, buses=sorted(comp), station_indices=cs))
    removed = [
        i for i, q in enumerate(problem.routes) if root[:, bi[net["station_buses"][q["c"]]]].all()
    ]
    if removed != cert["removed_routes"] or len(removed) != 13104:
        raise ValueError("Screening route identity mismatch")
    defs = proof["definitions"]
    rows = proof["cuts"]["rows"]
    pos = {int(k): i for i, k in enumerate(pfmap["keep"])}
    if (
        len(blocks) != len(defs)
        or len(blocks) != 43
        or len(rows) != 84
        or len({r["name"] for r in rows}) != 84
    ):
        raise ValueError("Incomplete frozen capacity proof")
    ri = 0
    minimum = float("inf")
    for block, ref in zip(blocks, defs):
        if any(block[k] != ref[k] for k in block):
            raise ValueError("Island proof order mismatch")
        t = block["t"]
        bs = block["buses"]
        d = sum((F(float(scene.physical["load_p_kw"][t, bi[b]])) for b in bs), F(0))
        critical = sum(
            (
                F(float(scene.physical["load_p_kw"][t, bi[b]]))
                for b in bs
                if net["shed_cost"][bi[b]] == 50
            ),
            F(0),
        )

        def f(n):
            return (10 * max(d - 90 * n, 0) + 40 * max(critical - 90 * n, 0)) / 4

        for intercept, slope, j in ref["rows"]:
            if slope == intercept == 0:
                continue
            row = rows[ri]
            ri += 1
            if (
                abs(float(f(j + 1) - f(j)) - slope) >= 1e-9
                or abs(float(f(j)) - slope * j - intercept) >= 1e-9
            ):
                raise ValueError("Integer secant changed")
            expected = {
                f"reserved[{e},{c},{t}]": slope for e in range(30) for c in block["station_indices"]
            }
            for b in bs:
                expected[f"s0_y[{pos[int(mat.index['shed'][t, bi[b]])]}]"] = (
                    -0.25 * net["shed_cost"][bi[b]]
                )
            if dict(row["terms"]) != expected or abs(row["rhs"] - (-intercept + 1e-8)) >= 1e-12:
                raise ValueError("Capacity row coefficient mismatch")
            for n in range(31):
                slack = f(n) - (F(slope) * n - F(row["rhs"]))
                if slack < 0:
                    raise ValueError("Capacity row is not a valid integer lower support")
                minimum = min(minimum, float(slack))
    if ri != 84:
        raise ValueError("Incomplete capacity rows")
    return dict(
        passed=True,
        removed=removed,
        rows=rows,
        minimum_exact_slack=minimum,
        scope="integer optimum; not LP optimum or every labelled feasible plan",
    )


def install(model, problem, copies):
    import gurobipy as gp

    proof = verify(problem, problem.det_proof, copies["forecast44"]["pfmap"])
    for i in proof["removed"]:
        model.getVarByName(f"route[{i}]").UB = 0
    variables = {v.VarName: v for v in model.getVars()}
    for row in proof["rows"]:
        if model.getConstrByName(row["name"]) is not None:
            raise ValueError("Duplicate capacity cut")
        model.addConstr(
            gp.LinExpr([v for _, v in row["terms"]], [variables[n] for n, _ in row["terms"]])
            <= row["rhs"],
            name=row["name"],
        )
    model.update()
    return proof
