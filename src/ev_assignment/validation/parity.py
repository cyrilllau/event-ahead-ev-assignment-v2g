"""Fixed historical plan parity; never grants global MIP convergence/adoption."""

import numpy as np

from ev_assignment.l08.physics import task, components
from ev_assignment.l08.matrix import assemble, first_stage
from ev_assignment.l08.energy import battery_trace
from ev_assignment.l08.reduction import pf_reduce, lift
from .l08 import check_first_vector


def decode_first(problem, vector):
    """Decode the frozen compact order and check every auxiliary first column."""
    x = np.asarray(vector, float)
    n, e = len(problem.routes), problem.fleet_size
    ns, nf = e * len(problem.starts), e * len(problem.ends)
    if x.ndim != 1 or len(x) < n + ns + nf + 4 * e or not np.isfinite(x).all():
        raise ValueError("Invalid historical first vector")
    start = x[n : n + ns].reshape(e, -1)
    end = x[n + ns : n + ns + nf].reshape(e, -1)
    energy = x[n + ns + nf + 3 * e : n + ns + nf + 4 * e]
    entries = []
    for i, route in enumerate(problem.routes):
        if x[i] > 0.5:
            ev = route["e"]
            ks, js = np.flatnonzero(start[ev] > 0.5), np.flatnonzero(end[ev] > 0.5)
            if len(ks) != 1 or len(js) != 1:
                raise ValueError("Historical plan has no unique window")
            h = task(route, problem.starts[ks[0]], problem.ends[js[0]] - 1)
            entries.append(dict(h, contract=float(energy[ev])))
    entries.sort(key=lambda h: h["e"])
    check_first_vector(problem, entries, x)
    return entries


def interval_parity(old_lower, old_upper, new_lower, new_upper):
    """Require overlapping numerical certificates without changing their guards."""
    values = np.asarray([old_lower, old_upper, new_lower, new_upper], float)
    if not np.isfinite(values).all() or old_lower > old_upper or new_lower > new_upper:
        raise ValueError("Invalid comparison interval")
    separation = max(0.0, float(old_lower - new_upper), float(new_lower - old_upper))
    return dict(
        passed=separation == 0,
        separation=separation,
        old=[float(old_lower), float(old_upper)],
        new=[float(new_lower), float(new_upper)],
    )


def saved_dispatch(problem, entries, sid, *, compact_y=None, named_values=None):
    """Lift a historical dispatch and audit the original P/Q/energy equations.

    This is a primal feasibility/cost check. It does not verify a historical dual
    or certify optimality; that requires the independent fixed-plan LP interval.
    """
    if (compact_y is None) == (named_values is None):
        raise ValueError("Exactly one historical dispatch representation is required")
    scene = problem.load_scenario(sid)
    route_index = {q["route_id"]: i for i, q in enumerate(problem.routes)}
    tasks = [task(problem.routes[route_index[h["route_id"]]], h["k"], h["r"]) for h in entries]
    energy = np.zeros(problem.fleet_size)
    for h in entries:
        energy[h["e"]] = h["contract"]
    z = np.r_[np.ones(len(tasks)), energy]
    states = [scene.route_states[route_index[h["route_id"]]] for h in entries]
    mat = assemble(
        problem.network, tasks, scene.physical, states, problem.fleet_size, problem.energy_bounds
    )
    _, mapping = pf_reduce(mat, problem.network, scene.physical)
    if not mapping["applicable"]:
        raise ValueError("Historical compact PF mapping is not applicable")
    size = len(mapping["keep"])
    if named_values is not None:
        reduced = np.asarray([named_values[f"s0_y[{i}]"] for i in range(size)])
    else:
        reduced = np.asarray(compact_y)[:size]
    if reduced.shape != (size,) or not np.isfinite(reduced).all():
        raise ValueError("Invalid historical dispatch")
    y = lift(reduced, mapping)
    A, b, parts, _, _ = first_stage(
        tasks, problem.fleet_size, len(problem.network["station_buses"]), problem.energy_bounds
    )
    first_residual = float(np.max(A @ z - b, initial=0))
    residual = mat.residual(y, z)
    if max(first_residual, residual) > 1e-6:
        raise ValueError(f"Historical original physics failed: {first_residual}, {residual}")
    pd, shed = y[mat.index["p"]], y[mat.index["shed"]]
    min_dc = None
    for h, st in zip(tasks, states):
        if st["success"] and st["arrival"] < (h["r"] + 1) / 4:
            tr = battery_trace(
                h["B"] + 100, st["arrival"], (h["r"] + 1) / 4, pd[h["e"]].sum(0), extra=[h["k"] / 4]
            )
            min_dc = tr["min_dc"] if min_dc is None else min(min_dc, tr["min_dc"])
    connected = 0.0
    for k in range(96):
        root = next(
            g
            for g in components(problem.network, scene.physical["line_operating_outages"][k])
            if 1 in g
        )
        for c, bus in enumerate(problem.network["station_buses"]):
            if bus in root:
                connected = max(connected, float(pd[:, c, k].sum()))
    if (min_dc is not None and min_dc < 100 - 1e-6) or connected > 1e-6:
        raise ValueError("Historical SOC/grid-connected V2G failed")
    critical = np.asarray(problem.network["shed_cost"]) == 50
    return dict(
        primal=float(mat.cost @ y),
        cost_parts=(mat.parts @ y).tolist(),
        first_cost=float(parts.sum(0) @ z),
        first_cost_parts=(parts @ z).tolist(),
        original_residual=residual,
        first_residual=first_residual,
        critical_ENS=float(0.25 * shed[:, critical].sum()),
        other_ENS=float(0.25 * shed[:, ~critical].sum()),
        actual_V2G_ac=float(0.25 * pd.sum()),
        shortfall_ac=float(y[mat.index["short"]].sum()),
        min_dc=min_dc,
        max_grid_connected_V2G_kw=connected,
    )
