"""L08 compact route/window MIP and exact recourse lifting.

Mathematical bodies extracted from the frozen R36 implementation. Imports and
constant access are isolated; no data root, logging or model optimize on import.
"""

import numpy as np
from .physics import task
from .matrix import assemble
from .reduction import pf_reduce


def add_first(m, qs, E, C, M, starts=None, ends=None):
    CFG = {"aux_dc_kw": 1.0, "ports": 20}
    import gurobipy as gp
    from gurobipy import GRB

    starts = list(range(93)) if starts is None else list(starts)
    ends = list(range(1, 97)) if ends is None else list(ends)
    assert np.asarray(M).shape == (E,) and all((q["B"] >= 50 for q in qs))
    v = m.addVars(len(qs), vtype=GRB.BINARY, name="route")
    s = m.addVars(E, starts, vtype=GRB.BINARY, name="start")
    f = m.addVars(E, ends, vtype=GRB.BINARY, name="end")
    u = m.addVars(E, ub=1, name="selected")
    K = m.addVars(E, ub=23, name="K")
    J = m.addVars(E, ub=24, name="J")
    en = m.addVars(E, ub=M.tolist(), name="E")
    w = m.addVars(E, 96, ub=1, name="window")
    b = m.addVars(E, C, ub=1, name="station")
    wc = m.addVars(E, C, 96, ub=1, name="reserved")
    by_e = [[i for i, q in enumerate(qs) if q["e"] == e] for e in range(E)]
    for e, ids in enumerate(by_e):
        m.addConstr(u[e] == gp.quicksum((v[i] for i in ids)), name=f"one_route/{e}")
        m.addConstr(s.sum(e, "*") == u[e])
        m.addConstr(f.sum(e, "*") == u[e])
        m.addConstr(K[e] == gp.quicksum((0.25 * k * s[e, k] for k in starts)))
        m.addConstr(J[e] == gp.quicksum((0.25 * j * f[e, j] for j in ends)))
        m.addConstr(J[e] - K[e] >= u[e])
        ah = gp.LinExpr([qs[i]["arrival"] for i in ids], [v[i] for i in ids])
        bb = gp.LinExpr([qs[i]["B"] for i in ids], [v[i] for i in ids])
        m.addConstr(ah <= K[e])
        m.addConstr(en[e] >= 15 * u[e])
        m.addConstr(en[e] <= M[e] * u[e])
        m.addConstr(en[e] <= 0.95 * (bb - CFG["aux_dc_kw"] * (J[e] - ah)))
        m.addConstr(en[e] <= 90 * (J[e] - K[e]))
        for t in range(96):
            m.addConstr(
                w[e, t] - (w[e, t - 1] if t else 0)
                == (s[e, t] if t in starts else 0) - (f[e, t] if t in ends else 0),
                name=f"window_recurrence/{e}/{t}",
            )
        for c in range(C):
            m.addConstr(b[e, c] == gp.quicksum((v[i] for i in ids if qs[i]["c"] == c)))
            for t in range(96):
                m.addConstr(wc[e, c, t] <= b[e, c])
                m.addConstr(wc[e, c, t] <= w[e, t])
                m.addConstr(wc[e, c, t] >= b[e, c] + w[e, t] - 1)
    for c in range(C):
        for t in range(96):
            m.addConstr(
                gp.quicksum((wc[e, c, t] for e in range(E))) <= CFG["ports"], name=f"ports/{c}/{t}"
            )
    parts = [
        25 * gp.quicksum(u.values()),
        30 * gp.quicksum(J.values()) - gp.LinExpr([30 * q["d"] / 4 for q in qs], list(v.values())),
        gp.quicksum((J[e] - K[e] for e in range(E))),
        0.02 * gp.quicksum(en.values()),
    ]
    return dict(
        qs=qs,
        Ecount=E,
        Ccount=C,
        M=np.asarray(M),
        v=v,
        s=s,
        f=f,
        u=u,
        K=K,
        J=J,
        E=en,
        w=w,
        b=b,
        wc=wc,
        by_e=by_e,
        parts=parts,
        objective=sum(parts),
        starts=starts,
        ends=ends,
    )


def decode(fs, values=None):
    get = (lambda x: x.X) if values is None else lambda x: values[x.index]
    plans = []
    for e, ids in enumerate(fs["by_e"]):
        sel = [i for i in ids if get(fs["v"][i]) > 0.5]
        if not sel:
            assert abs(get(fs["E"][e])) <= 1e-06
            continue
        assert len(sel) == 1
        ks = [k for k in fs["starts"] if get(fs["s"][e, k]) > 0.5]
        js = [j for j in fs["ends"] if get(fs["f"][e, j]) > 0.5]
        assert len(ks) == len(js) == 1
        h = task(fs["qs"][sel[0]], ks[0], js[0] - 1)
        h["contract"] = float(get(fs["E"][e]))
        plans.append(h)
    return plans


def fix_first(fs, plans):
    sel = {h["e"]: h for h in plans}
    for i, q in enumerate(fs["qs"]):
        h = sel.get(q["e"])
        val = int(h is not None and (q["c"], q["d"]) == (h["c"], h["d"]))
        fs["v"][i].LB = fs["v"][i].UB = val
    for e in range(fs["Ecount"]):
        h = sel.get(e)
        for k in fs["starts"]:
            fs["s"][e, k].LB = fs["s"][e, k].UB = int(h is not None and h["k"] == k)
        for j in fs["ends"]:
            fs["f"][e, j].LB = fs["f"][e, j].UB = int(h is not None and h["r"] + 1 == j)
        fs["E"][e].LB = fs["E"][e].UB = h["contract"] if h else 0


def add_strengthening(m, fs, enabled=(1, 2, 3)):
    import gurobipy as gp

    groups = {}
    for i, q in enumerate(fs["qs"]):
        t = int(np.searchsorted(np.arange(97) / 4, q["arrival"]))
        groups.setdefault((q["e"], t), []).append(i)
    if sum(map(len, groups.values())) != len(fs["qs"]):
        raise ValueError("ROUTE_GROUPING")
    A = m.addVars(fs["Ecount"], 97, lb=0, ub=1, name="r7_arrival_prefix")
    S = m.addVars(fs["Ecount"], 97, lb=0, ub=1, name="r7_start_prefix")
    F = m.addVars(fs["Ecount"], 97, lb=0, ub=1, name="r7_end_prefix")
    for e in range(fs["Ecount"]):
        for t in range(97):
            m.addConstr(
                A[e, t] - (A[e, t - 1] if t else 0)
                == gp.quicksum((fs["v"][i] for i in groups.get((e, t), []))),
                name=f"R7/Arec/{e}/{t}",
            )
            m.addConstr(
                S[e, t] - (S[e, t - 1] if t else 0) == (fs["s"][e, t] if t in fs["starts"] else 0),
                name=f"R7/Srec/{e}/{t}",
            )
            m.addConstr(
                F[e, t] - (F[e, t - 1] if t else 0) == (fs["f"][e, t] if t in fs["ends"] else 0),
                name=f"R7/Frec/{e}/{t}",
            )
            if 1 in enabled:
                m.addConstr(S[e, t] <= A[e, t], name=f"R7/VI1/{e}/{t}")
            if 2 in enabled:
                m.addConstr(F[e, t] <= (S[e, t - 4] if t >= 4 else 0), name=f"R7/VI2/{e}/{t}")
        for t in range(96):
            if 3 in enabled:
                m.addConstr(
                    gp.quicksum((fs["wc"][e, c, t] for c in range(fs["Ccount"]))) == fs["w"][e, t],
                    name=f"R7/VI3/{e}/{t}",
                )
    return dict(
        A=A,
        S=S,
        F=F,
        route_occurrences=sum(map(len, groups.values())),
        group_count=len(groups),
        proof="integer_route_start_end_prefix_and_exclusive_station",
    )


def compact_base(net, scene, E, M):
    original = assemble(net, [], scene, [], E, global_M=M)
    reduced, mapping = pf_reduce(original, net, scene)
    return (original, reduced, mapping)


def original_expressions(y, mapping):
    keep = mapping["keep"]
    pos = {int(k): i for i, k in enumerate(keep)}
    yy = [None] * mapping["original_columns"]
    vv = y.tolist()
    for k, i in pos.items():
        yy[k] = vv[i]
    if mapping["applicable"]:
        for q, p in zip(mapping["q_columns"], mapping["p_columns"]):
            yy[int(q)] = mapping["rho"] * vv[pos[int(p)]]
    assert all((v is not None for v in yy))
    return yy


def add_recourse(m, fs, net, scene, states, sid):
    CFG = {"aux_dc_kw": 1.0, "ports": 20}
    import gurobipy as gp

    E = fs["Ecount"]
    C = fs["Ccount"]
    qs = fs["qs"]
    v = fs["v"]
    M = fs["M"]
    pre = f"s{sid}"
    original, base, pfmap = compact_base(net, scene, E, M)
    y = m.addMVar(len(base.cost), lb=base.lo, ub=base.hi, name=pre + "_y")
    yy = original_expressions(y, pfmap)
    ix = original.index
    retain = [
        i
        for i, n in enumerate(base.row_names["ub"])
        if not n.startswith(("discharge_", "short_le_", "p_availability/", "incident_capability/"))
    ]
    m.addMConstr(base.A[retain], y, "<", base.b[retain], name=pre + "_network_ub")
    m.addMConstr(base.H, y, "=", base.d, name=pre + "_network_eq")
    a = m.addVars(E, 96, ub=1, name=pre + "_arrived")
    over = m.addVars(E, 96, ub=1, name=pre + "_overlap")
    W = m.addVars(E, ub=24, name=pre + "_AUX_HOURS")
    cap = m.addVars(E, name=pre + "_capacity_aux")
    D = m.addVars(E, ub=1, name=pre + "_incident")
    H = m.addVars(E, name=pre + "_activated_contract")
    groups = {}
    good = []
    for i, (q, st) in enumerate(zip(qs, states)):
        g = bool(
            st["success"]
            and np.isfinite(st["arrival"])
            and (q["departure_dc"] - 1.2 * q["distance"] >= 150)
            and (st["arrival"] <= 23)
        )
        good.append(g)
        if g:
            assert st["arrival"] >= 0
            t = int(np.searchsorted(np.arange(96) / 4, st["arrival"]))
            groups.setdefault((q["e"], t), []).append(i)
    zeta = np.asarray(base.metadata["zeta"], bool)
    for e, ids in enumerate(fs["by_e"]):
        valid = [i for i in ids if good[i]]
        g = gp.quicksum((v[i] for i in valid))
        ar = gp.LinExpr([states[i]["arrival"] for i in valid], [v[i] for i in valid])
        B = gp.LinExpr([qs[i]["B"] for i in valid], [v[i] for i in valid])
        for t in range(96):
            m.addConstr(
                a[e, t] - (a[e, t - 1] if t else 0)
                == gp.quicksum((v[i] for i in groups.get((e, t), []))),
                name=f"{pre}/arrival/{e}/{t}",
            )
            m.addConstr(over[e, t] <= a[e, t])
            m.addConstr(over[e, t] <= fs["w"][e, t])
            m.addConstr(over[e, t] >= a[e, t] + fs["w"][e, t] - 1)
            m.addConstr(gp.quicksum((yy[ix["p"][e, c, t]] for c in range(C))) <= 90 * over[e, t])
            for c in range(C):
                m.addConstr(yy[ix["p"][e, c, t]] <= 90 * fs["b"][e, c])
        m.addConstr(W[e] <= 24 * g)
        m.addConstr(W[e] >= fs["J"][e] - ar - 24 * (fs["u"][e] - g))
        m.addConstr(cap[e] <= 0.95 * (B - CFG["aux_dc_kw"] * W[e]))
        m.addConstr(cap[e] <= 22.5 * gp.quicksum((over[e, t] for t in range(96))))
        energy = 0.25 * gp.quicksum((yy[i] for i in ix["p"][e].flat))
        m.addConstr(energy <= cap[e])
        m.addConstr(energy <= fs["E"][e])
        incident_vars = [fs["wc"][e, c, t] for c, t in zip(*np.where(zeta))]
        m.addConstr(D[e] <= fs["u"][e])
        m.addConstr(D[e] <= gp.quicksum(incident_vars))
        for wc in incident_vars:
            m.addConstr(D[e] >= wc)
        m.addConstr(H[e] <= fs["E"][e])
        m.addConstr(H[e] <= M[e] * D[e])
        m.addConstr(H[e] >= fs["E"][e] - M[e] * (1 - D[e]))
        m.addConstr(yy[ix["short"][e]] >= H[e] - cap[e])
        m.addConstr(yy[ix["short"][e]] <= fs["E"][e])
    parts = [gp.LinExpr(row.tolist(), y.tolist()) for row in base.parts]
    return dict(
        y=y,
        base=base,
        original=original,
        pfmap=pfmap,
        parts=parts,
        objective=sum(parts),
        arrived=a,
        overlap=over,
        W=W,
        capacity=cap,
        D=D,
        H=H,
        arrival_group_route_count=sum(map(len, groups.values())),
        qualified_route_count=sum(good),
        network_retained_rows=retain,
    )
