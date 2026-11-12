"""Solver-free, bounded type/package sidecar for the *new* catalogue domain.

Times d/k/r are quarter-hour indices; forecast ``arrival`` and scenario arrivals
are hours, departure_dc is kWh DC, distance is km, E/Ebar/S are kWh AC and P is
kW. Ebar is an explicit input, never recomputed or enlarged here. Generators must
freeze identical coefficients and route responses for all vehicles of a type.

Count variables are [n_h, E_h]; individual variables are [x_eh, E_eh]. The latter
is a perspective formulation with a contract per package. It is integer-equivalent
to the original per-vehicle-contract matrix on this same catalogue, NOT asserted
LP-equivalent. Neither representation recovers omitted tasks from the old domain.
Aggregating unequal individual contracts can change fixed-policy recourse value;
only canonical equal-share expansion identifies the same count policy. No Q/cut
cache identity, probability model, solver, data loading or orchestration lives here.
"""

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from numbers import Integral

import numpy as np
from scipy import sparse as sp

from . import matrix
from .certificates import enclosure, finite_bounds, row_audit
from .physics import incident, qualify, tou
from .recourse import RecourseTemplate


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _signature(packages):
    return tuple(tuple(p[k] for k in (
        "g", "c", "d", "k", "r", "departure_dc", "distance", "arrival", "Ebar"
    )) for p in packages)


def _identity(first):
    return (_signature(first["catalogue"]), first["type_counts"], first["station_count"],
            tuple(first["weights"]), first["ports"], first["individual"])


def _upper_product(a, b):
    """A redundant global bound must not round an exact product inward."""
    values = []
    for aa, bb in zip(a, b):
        exact = Fraction(float(aa)) * Fraction(float(bb))
        value = float(exact)
        values.append(np.nextafter(value, np.inf) if Fraction(value) < exact else value)
    return np.asarray(values, float)


def build_first(packages, type_counts, station_count, weights=(1, 1, 1, 1),
                ports=20, individual=False):
    """Return explicit CSR first-stage arrays (all row senses are ``<``).

    ``package_map[j]`` is the original package's zero-based list index;
    ``packages[j]`` is a copy of that dictionary. ``owner_map[j]`` is a type index
    in count mode or global vehicle index in individual mode. Vehicles are ordered
    by type, then within type; individual decisions by vehicle, then source package.
    The first D variables select tasks, the final D are contracts. ``names`` names
    variables; ``row_names`` names rows. ``parts`` retains the four ORIGINAL cost
    components; only ``cost = weights @ parts`` applies the nonnegative weights.
    These weights affect first-stage costs only, never operating costs.
    """
    counts = tuple(_integer(n, "type count") for n in type_counts)
    C = _integer(station_count, "station_count")
    ports = _integer(ports, "ports")
    weights = np.asarray(weights, float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be four finite nonnegative numbers")
    packages = deepcopy(tuple(packages))
    for p in packages:
        for key in ("g", "c", "d", "k", "r"):
            _integer(p[key], key)
        if p["g"] >= len(counts) or p["c"] >= C:
            raise ValueError("Package type/station index outside supplied domain")
        if not 0 <= p["d"] <= p["k"] <= p["r"] < 96:
            raise ValueError("Expected quarter-hour indices 0 <= d <= k <= r < 96")
        for key in ("departure_dc", "distance", "arrival", "Ebar"):
            if not np.isfinite(p[key]) or p[key] < 0:
                raise ValueError(f"Package {key} must be finite and nonnegative")
    vehicle_types = np.repeat(np.arange(len(counts), dtype=int), counts)
    if individual:
        pairs = [(e, h) for e, g in enumerate(vehicle_types)
                 for h, p in enumerate(packages) if p["g"] == g]
    else:
        pairs = [(p["g"], h) for h, p in enumerate(packages)]
    owners = np.asarray([e for e, _ in pairs], dtype=int)
    pm = np.asarray([h for _, h in pairs], dtype=int)
    decisions = tuple(deepcopy(packages[h]) for h in pm)
    D = len(pm)
    cap = np.asarray([1 if individual else counts[p["g"]] for p in decisions], float)
    ebar = np.asarray([p["Ebar"] for p in decisions], float)
    parts = np.zeros((4, 2 * D))
    for j, p in enumerate(decisions):
        parts[:, j] = [25, 30 * (p["r"] + 1 - p["d"]) / 4,
                       (p["r"] - p["k"] + 1) / 4, 0]
    parts[3, D:] = .02
    rows = matrix.Rows(0, 2 * D)
    capacities = [1] * len(vehicle_types) if individual else counts
    for owner, capacity in enumerate(capacities):
        label = "one_task" if individual else "type_capacity"
        rows.add(f"{label}/{owner}", z=[(j, 1) for j in np.flatnonzero(owners == owner)],
                 rhs=capacity)
    for j in range(D):
        rows.add(f"contract_min/{j}", z=[(j, 15), (D + j, -1)])
        rows.add(f"contract_max/{j}", z=[(j, -ebar[j]), (D + j, 1)])
    for c in range(C):
        for t in range(96):
            rows.add(f"ports/{c}/{t}", z=[(j, 1) for j, p in enumerate(decisions)
                                         if p["c"] == c and p["k"] <= t <= p["r"]],
                     rhs=ports)
    _, A, rhs = rows.matrices()
    hi = np.r_[cap, _upper_product(cap, ebar)]
    if not np.isfinite(hi).all():
        raise ValueError("Nonfinite catalogue bounds")
    return dict(A=A, rhs=rhs, sense=np.full(len(rhs), "<"), cost=weights @ parts,
                parts=parts, lo=np.zeros(2 * D), hi=hi,
                vtype=np.asarray(["B" if individual else "I"] * D + ["C"] * D),
                names=tuple([f"{'x' if individual else 'n'}/{j}" for j in range(D)] +
                            [f"E/{j}" for j in range(D)]), row_names=tuple(rows.names),
                package_map=pm, owner_map=owners, packages=decisions, catalogue=packages,
                type_counts=counts, station_count=C, ports=ports, weights=weights.copy(),
                individual=bool(individual), vehicle_types=vehicle_types,
                index=dict(n=np.arange(D), E=np.arange(D, 2 * D)))


def _inputs(net, packages, route_states, first):
    if _signature(packages) != _signature(first["catalogue"]):
        raise ValueError("Packages differ from the first-stage catalogue")
    if len(net["station_buses"]) != first["station_count"] or len(route_states) != len(packages):
        raise ValueError("Station or route-state cardinality mismatch")
    buses = list(net["buses"])
    if len(set(buses)) != len(buses) or buses.count(1) != 1:
        raise ValueError("Grid requires unique buses and root bus 1")
    if any(b not in buses for b in [*net["station_buses"], *net["from"], *net["to"]]):
        raise ValueError("Unknown grid endpoint or station bus")


def _fingerprint(*arrays):
    digest = sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _scenario_key(net, scene, route_states):
    # Deliberately strict: these keys identify mapping inputs, not an equivalence
    # class of scenarios and not a scientific Run or cross-policy Q-cache key.
    return _fingerprint(
        *(net[k] for k in ("buses", "station_buses", "from", "to", "p_limit", "q_limit",
                          "u_min", "u_max", "u_ref", "mp", "mq", "big_m", "shed_cost")),
        *(scene[k] for k in ("line_operating_outages", "load_p_kw", "load_q_kvar")),
        [s["arrival"] for s in route_states], [s["success"] for s in route_states])


def build_recourse(net, packages, scene, route_states, first):
    """Build a finite ``RecourseTemplate`` with ``W y + T x {sense} h``.

    Supply one route state {arrival: hours, success: bool} per ORIGINAL package,
    shared by all clones. x may be fractional: no integer decoding occurs here.
    y contains the original empty-fleet grid variables, then P[D,96], then S[D].
    ``data['index']`` locates them; ``data['parts']`` gives original operating cost
    components. The original P/Q balances, .33 EV power factor, voltage, outage,
    reference, shedding and line equations are retained, without PF reduction.

    Global P <= 90*N_g (90 for individuals), S <= N_g*Ebar follow from capacity
    and contract rows. Remaining infinite grid bounds are proved by the ORIGINAL
    rational finite_bounds propagation on [first rows; recourse rows], columns
    [x;y]. Its row numbers in bound_proof refer to precisely that joined matrix.
    Finite bounds are imposed and also passed to the existing cut certificates.
    """
    _inputs(net, packages, route_states, first)
    grid = matrix.assemble(net, [], scene, [], E=0, global_M=[], ports=first["ports"])
    pm = first["package_map"]
    D, nx, ng = len(pm), len(first["cost"]), len(grid.cost)
    ny = ng + 97 * D
    ix = {key: value.copy() for key, value in grid.index.items() if key not in {"p", "short"}}
    ix["p"] = np.arange(ng, ng + 96 * D).reshape(D, 96)
    ix["short"] = np.arange(ng + 96 * D, ny)
    source_mu = np.zeros((len(packages), 96), bool)
    source_use = np.zeros(len(packages))
    source_av = np.zeros(len(packages))
    chi = np.zeros(len(packages))
    zeta, _ = incident(net, scene)
    for h, (p, state) in enumerate(zip(packages, route_states)):
        _, source_mu[h], source_use[h], source_av[h] = qualify(
            p["departure_dc"], p["distance"], state["arrival"], p["k"], p["r"],
            state["success"])
        chi[h] = zeta[p["c"], p["k"]:p["r"] + 1].any()
    mu, use, available = source_mu[pm], source_use[pm], source_av[pm]
    cap = first["hi"][:D]
    lo = np.r_[grid.lo, np.zeros(97 * D)]
    hi = np.r_[grid.hi, np.repeat(_upper_product(cap, np.full(D, 90.)), 96), first["hi"][D:]]
    parts = np.pad(grid.parts, ((0, 0), (0, 97 * D)))
    parts[1, ix["short"]] = 1
    parts[2, ix["p"]] = .25 * tou()
    parts[3, ix["p"]] = .25 * .05 / .95
    names = [""] * ny
    for key, indices in ix.items():
        for at in np.ndindex(indices.shape):
            names[int(indices[at])] = key + "/" + "/".join(map(str, at))

    grid_rows = tuple(grid.row_names["ub"] + grid.row_names["eq"])
    base = sp.vstack([grid.A, grid.H], format="csr")
    W = sp.hstack([base, sp.csr_matrix((base.shape[0], 97 * D))], format="csr")
    lookup = {name: i for i, name in enumerate(grid_rows)}
    rr, cc, vv = [], [], []
    for j, p in enumerate(first["packages"]):
        bus = net["station_buses"][p["c"]]
        for t in range(96):
            for name, value in ((f"station_power/{p['c']}/{t}", 1),
                                (f"P_balance/{bus}/{t}", 1),
                                (f"Q_balance/{bus}/{t}", .33)):
                rr.append(lookup[name])
                cc.append(ix["p"][j, t])
                vv.append(value)
    W += sp.coo_matrix((vv, (rr, cc)), shape=W.shape).tocsr()
    rows = matrix.Rows(ny, nx)
    for j, h in enumerate(pm):
        power = [(i, .25) for i in ix["p"][j]]
        rows.add(f"discharge_contract/{j}", power, [(D + j, -1)])
        rows.add(f"discharge_physical/{j}", power, [(j, -.95 * use[j])])
        rows.add(f"short_le_contract/{j}", [(ix["short"][j], 1)], [(D + j, -1)])
        rows.add(f"incident_capability/{j}", [(ix["short"][j], -1)],
                 [(D + j, chi[h]), (j, -available[j])])
        for t in range(96):
            rows.add(f"p_availability/{j}/{t}", [(ix["p"][j, t], 1)],
                     [(j, -90 * mu[j, t])])
    extra_W, extra_T, extra_h = rows.matrices()
    W = sp.vstack([W, extra_W], format="csr")
    T = sp.vstack([sp.csr_matrix((len(grid_rows), nx)), extra_T], format="csr")
    h = np.r_[grid.b, grid.d, extra_h]
    sense = np.asarray(["<"] * len(grid.b) + ["="] * len(grid.d) + ["<"] * len(extra_h))
    joint = sp.vstack([sp.hstack([first["A"], sp.csr_matrix((len(first["rhs"]), ny))]),
                       sp.hstack([T, W])], format="csr")
    fl, fh, proof = finite_bounds(joint, np.r_[first["rhs"], h],
                                  np.r_[first["sense"], sense], np.r_[first["lo"], lo],
                                  np.r_[first["hi"], hi], first["names"] + tuple(names))
    if not np.isfinite(fl).all() or not np.isfinite(fh).all() or np.any(fl > fh):
        raise ValueError("Unproven or inconsistent finite catalogue/grid bounds")
    cost = parts.sum(0)
    if not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("Recourse requires finite nonnegative costs")
    _, upper = enclosure(sp.csr_matrix(cost[None, :]), fh[nx:])
    qbar = float(np.nextafter(float(upper[0] + 1e-6), np.inf))
    if not np.isfinite(qbar):
        raise ValueError("Nonfinite recourse objective bound")
    data = dict(h=h, sense=sense, c=cost, lo=fl[nx:], hi=fh[nx:],
                finite_lo=fl[nx:].copy(), finite_hi=fh[nx:].copy(),
                xlo=fl[:nx], xhi=fh[:nx], index=ix, parts=parts,
                row_names=grid_rows + tuple(rows.names), grid_size=ng,
                first_identity=_identity(first), scenario_key=_scenario_key(net, scene, route_states),
                mu=mu, Euse=use, Eav=available, chi=chi[pm])
    return RecourseTemplate(W, T, data, tuple(names), tuple(proof), qbar)


def _first_point(first, x, integer=True):
    x = np.asarray(x, float)
    if x.shape != first["lo"].shape or not np.isfinite(x).all():
        raise ValueError("Invalid first-stage vector")
    D = len(first["package_map"])
    if integer and np.max(abs(x[:D] - np.rint(x[:D])), initial=0) > 1e-9:
        raise ValueError("Integer mapping cannot decode fractional decisions")
    check = row_audit(first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], x)
    if max(check.values()) > 1e-9:
        raise ValueError(("First-stage residual", check))
    return x


def _recourse_point(first, template, x, y):
    if template.data["first_identity"] != _identity(first):
        raise ValueError("Recourse/first-stage identity mismatch")
    a = template.data
    y = np.asarray(y, float)
    if y.shape != a["lo"].shape or not np.isfinite(y).all():
        raise ValueError("Invalid recourse vector")
    check = row_audit(template.W, a["h"] - template.T @ x, a["sense"], a["lo"], a["hi"], y)
    if max(check.values()) > 1e-9:
        raise ValueError(("Recourse residual", check))
    return y


def _pair(count_first, individual_first):
    if count_first["individual"] or not individual_first["individual"]:
        raise ValueError("Expected count and individual first stages")
    if _identity(count_first)[:-1] != _identity(individual_first)[:-1]:
        raise ValueError("Catalogue identity mismatch")


def _mapped_recourse(source, target, source_first, target_first, x, y):
    if source is None or target is None:
        raise ValueError("Both recourse templates are required with y")
    if (source.data["scenario_key"] != target.data["scenario_key"] or
            target.data["first_identity"] != _identity(target_first)):
        raise ValueError("Recourse scenario/first-stage identity mismatch")
    y = _recourse_point(source_first, source, x, y)
    out = np.zeros(len(target.names))
    ng = source.data["grid_size"]
    out[:ng] = y[:ng]
    return y, out


def expand_count(count_first, individual_first, x, y=None, *,
                 count_recourse=None, individual_recourse=None):
    """Return (individual_x, individual_y_or_None), assigning lowest free owners.

    Iterate source packages in order and allocate distinct vehicles of their type.
    Contracts, P and S are divided equally by integer n. Repeating for any scenario
    uses exactly the same owners and contracts. Both templates (same net/scenario)
    are required when y is supplied. All mappings audit at the original 1e-9
    tolerance and never solve/project a point. Fractional counts are rejected.
    """
    _pair(count_first, individual_first)
    x = _first_point(count_first, x)
    H, J = len(count_first["package_map"]), len(individual_first["package_map"])
    out = np.zeros(2 * J)
    lookup = {(e, h): j for j, (e, h) in enumerate(zip(
        individual_first["owner_map"], individual_first["package_map"]))}
    free = [list(np.flatnonzero(individual_first["vehicle_types"] == g))
            for g in range(len(count_first["type_counts"]))]
    assignments = []
    for h, p in enumerate(count_first["packages"]):
        n = int(round(x[h]))
        owners, free[p["g"]] = free[p["g"]][:n], free[p["g"]][n:]
        for e in owners:
            j = lookup[e, h]
            out[j], out[J + j] = 1, x[H + h] / n
            assignments.append((h, j, n))
    _first_point(individual_first, out)
    yy = None
    if y is not None:
        y, yy = _mapped_recourse(count_recourse, individual_recourse, count_first,
                                 individual_first, x, y)
        a, b = count_recourse.data["index"], individual_recourse.data["index"]
        for h, j, n in assignments:
            yy[b["p"][j]] = y[a["p"][h]] / n
            yy[b["short"][j]] = y[a["short"][h]] / n
        _recourse_point(individual_first, individual_recourse, out, yy)
    return out, yy


def sum_individual(individual_first, count_first, x, y=None, *,
                   individual_recourse=None, count_recourse=None):
    """Return (count_x, count_y_or_None) by summing each original package.

    Feasible integer individual primals (including unequal contracts) map with
    unchanged linear cost. Canonical re-expansion need not recover the contracts
    or fixed-policy value of an unequal allocation. Grid variables are copied.
    """
    _pair(count_first, individual_first)
    x = _first_point(individual_first, x)
    H, J = len(count_first["package_map"]), len(individual_first["package_map"])
    pm = individual_first["package_map"]
    out = np.zeros(2 * H)
    np.add.at(out, pm, x[:J])
    np.add.at(out, H + pm, x[J:])
    _first_point(count_first, out)
    yy = None
    if y is not None:
        y, yy = _mapped_recourse(individual_recourse, count_recourse, individual_first,
                                 count_first, x, y)
        a, b = individual_recourse.data["index"], count_recourse.data["index"]
        for j, h in enumerate(pm):
            yy[b["p"][h]] += y[a["p"][j]]
            yy[b["short"][h]] += y[a["short"][j]]
        _recourse_point(count_first, count_recourse, out, yy)
    return out, yy


def build_original_adapter(net, packages, scene, route_states, individual_first):
    """Build the ORIGINAL matrix.assemble with every actual cloned task present.

    Returns matrix, first-stage arrays, tasks/states and global_M. Original z has
    J task binaries then one contract per vehicle (including idle vehicles).
    M[e] = max Ebar over ALL that vehicle's packages (0 for no packages), so even
    an unchosen package's chi*E-S <= available+M is redundant at an integer plan.
    No Big-M rows are dropped. This is the independent A1 audit target; constructing
    it or mapping primals does not prove solver optimality or complete the A1 gate.
    """
    if not individual_first["individual"]:
        raise ValueError("Original adapter requires individual first stage")
    _inputs(net, packages, route_states, individual_first)
    fs = individual_first
    tasks = [dict(deepcopy(packages[h]), e=int(e))
             for e, h in zip(fs["owner_map"], fs["package_map"])]
    states = [deepcopy(route_states[h]) for h in fs["package_map"]]
    N, J = sum(fs["type_counts"]), len(tasks)
    M = np.zeros(N)
    for p in tasks:
        M[p["e"]] = max(M[p["e"]], p["Ebar"])
    A, rhs, parts, _, row_names = matrix.first_stage(
        tasks, N, fs["station_count"], M, ports=fs["ports"])
    original = matrix.assemble(net, tasks, scene, states, N, M, ports=fs["ports"])
    first = dict(A=A, rhs=rhs, sense=np.full(len(rhs), "<"), parts=parts,
                 cost=fs["weights"] @ parts, lo=np.zeros(J + N), hi=np.r_[np.ones(J), M],
                 vtype=np.asarray(["B"] * J + ["C"] * N), row_names=tuple(row_names))
    return dict(matrix=original, first=first, tasks=tasks, route_states=states, global_M=M,
                first_identity=_identity(fs), scenario_key=_scenario_key(net, scene, route_states))


def _original_point(adapter, z, y):
    f, m = adapter["first"], adapter["matrix"]
    z, y = np.asarray(z, float), np.asarray(y, float)
    if (z.shape != f["lo"].shape or y.shape != m.lo.shape or
            not np.isfinite(z).all() or not np.isfinite(y).all()):
        raise ValueError("Invalid original vector")
    J = len(adapter["tasks"])
    if np.max(abs(z[:J] - np.rint(z[:J])), initial=0) > 1e-9:
        raise ValueError("Original mapping requires integer tasks")
    check = row_audit(f["A"], f["rhs"], f["sense"], f["lo"], f["hi"], z)
    if max(check.values()) > 1e-9 or m.residual(y, z) > 1e-9:
        raise ValueError(("Original matrix residual", check, m.residual(y, z)))
    return z, y


def to_original(adapter, individual_first, individual_recourse, x, y):
    """Return audited (original_z, original_y); retain ALL unchosen task rows."""
    if (adapter["first_identity"] != _identity(individual_first) or
            adapter["scenario_key"] != individual_recourse.data["scenario_key"]):
        raise ValueError("Original adapter identity mismatch")
    x = _first_point(individual_first, x)
    y = _recourse_point(individual_first, individual_recourse, x, y)
    fs, old = individual_first, adapter["matrix"]
    J = len(fs["package_map"])
    z = np.r_[x[:J], np.zeros(len(adapter["global_M"]))]
    np.add.at(z, J + fs["owner_map"], x[J:])
    out = np.zeros(len(old.cost))
    ix = individual_recourse.data["index"]
    for key in ix.keys() - {"p", "short"}:
        out[old.index[key]] = y[ix[key]]
    for j, p in enumerate(fs["packages"]):
        e = fs["owner_map"][j]
        out[old.index["p"][e, p["c"]]] += y[ix["p"][j]]
        out[old.index["short"][e]] += y[ix["short"][j]]
    return _original_point(adapter, z, out)


def from_original(adapter, individual_first, individual_recourse, z, y):
    """Return audited (individual_x, individual_y) from original integer primals.

    Assign each vehicle's contract and recourse to its selected package. This
    includes unequal contracts; sum_individual then gives the count primal.
    """
    if (adapter["first_identity"] != _identity(individual_first) or
            adapter["scenario_key"] != individual_recourse.data["scenario_key"]):
        raise ValueError("Original adapter identity mismatch")
    z, y = _original_point(adapter, z, y)
    fs, old = individual_first, adapter["matrix"]
    J = len(fs["package_map"])
    x = np.r_[z[:J], np.zeros(J)]
    out = np.zeros(len(individual_recourse.names))
    ix = individual_recourse.data["index"]
    for key in ix.keys() - {"p", "short"}:
        out[ix[key]] = y[old.index[key]]
    for j, p in enumerate(fs["packages"]):
        if round(z[j]) == 1:
            e = fs["owner_map"][j]
            x[J + j] = z[J + e]
            out[ix["p"][j]] = y[old.index["p"][e, p["c"]]]
            out[ix["short"][j]] = y[old.index["short"][e]]
    _first_point(fs, x)
    _recourse_point(fs, individual_recourse, x, out)
    return x, out
