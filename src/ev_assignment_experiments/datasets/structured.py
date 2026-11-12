"""Declared synthetic fleet family on read-only IEEE33/L08 source assets.

Public interfaces: ``read_source``, ``preregistered_candidates``, ``make_case``,
``audit_physical_case``, ``make_reference_policies``. No optimizer is imported
or called. This is NOT frozen forecast44/TRAIN400 or its 322311-row DRO model.

Source root must contain the five named files and source-manifest.json, with
schema_version=1, kind=structured-physical-source, and files=[{path,bytes,
sha256,...}]. Source ingestion is fail-closed and performs no writes. Package
indices g/c/d/k/r are zero-based (time indices are quarters); arrival is hours,
departure_dc kWh DC, Ebar/contract kWh AC. r is inclusive. Array axes are
physical time, then source bus/line/arc order. Reference x = concatenate(n,E).

Rules below are frozen before any policy evaluation. Only the three candidate
keys may change. Same-type vehicles copy the entire normal trajectory, not a
single SOC: departure energy may vary with d. All methods share one catalogue.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ev_assignment.l08 import energy as energy_kernel, physics as physics_kernel
from ev_assignment.physics import route_replay as replay_kernel
from ev_assignment.l08.physics import components, ebar, first_cost, qualify
from ev_assignment.physics.route_replay import replay_route


FAMILY = "structured_fleet_event_ahead_v1"
SOURCE_FILES = (
    "NETWORK_NEW_CLOCK.json", "ROAD_ARCS.csv", "C__evcs_bus_road_mapping.csv",
    "NORMAL_BUSINESS_DAY_LEDGER.csv", "DRO_FEATURE_REGISTRY.json",
)
CRITICAL = (2, 4, 8, 10, 14, 21, 24, 27, 30, 32)
FEATURE_NAMES = ("L/control_line20/t66", "R/corridor1/t66", "LR/line20_corridor1/t66_t66")
JOINT_ROW_INDICES = (2,)
FEATURE_ROWS = (("L", 19, -1, 66, -1), ("R", 0, -1, 66, -1),
                ("LR", 19, 0, 66, 66))

# Read-only source inspection, 2026-11-12: assets() iterates unchanged arcs
# [::2], whose corridor IDs are exactly 1..44; physical_roads=40. The supplied
# L08 split-arc parent IDs preserve that ordered physical-road interpretation.
REGISTRY_SEMANTICS_EVIDENCE = {
    "road_order": list(range(1, 45)),
    "ROAD_ARCS_UNCHANGED.csv_sha256":
        "08197c58e1ccbf27d4f5dc3655c3928755cdf27b08661c8d9cc311dec6a8cb5a",
    "sectionii/physics.py_sha256":
        "ca063dc4013585d116e61f6177ffaae2596e5cddd2f1fec02332b866fa11529e",
    "moment_equations.py_sha256":
        "260302006148ff8a15dd7fdff47abdd99fd3eb125cde1814bc09d8f5285fd9fd",
    "interpretation": "L uses recovery.outage_current; R uses persistent physical road; "
                      "LR multiplies the named coordinates; times are zero-based quarters",
    "evidence_level": "read-only source inspection; ancillary files not in this input bundle",
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def preregistered_candidates():
    """Twelve candidates, in fixed product order; all first evaluated on seed101."""
    return [dict(gap_h=g, early_duration_h=d, joint_nominal=p)
            for g, d, p in itertools.product((.5, 1., 2.), (.25, .5), (.08, .02))]


def _config(config):
    value = dict(config)
    if value not in preregistered_candidates():
        raise ValueError("Expected exactly gap_h, early_duration_h, joint_nominal from frozen12")
    return {k: float(value[k]) for k in ("gap_h", "early_duration_h", "joint_nominal")}


def read_source(source_root):
    """Verify the explicit manifest and return a read-only source snapshot in memory."""
    root = Path(source_root).expanduser().resolve(strict=True)
    mp = root / "source-manifest.json"
    if mp.is_symlink():
        raise ValueError("Source manifest must not be a symlink")
    manifest_raw = mp.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "structured-physical-source":
        raise ValueError("Unsupported structured source manifest")
    entries = manifest["files"]
    if len(entries) != len(SOURCE_FILES) or {e["path"] for e in entries} != set(SOURCE_FILES):
        raise ValueError("Source manifest must declare exactly the five physical inputs")
    data = {}
    for entry in entries:
        p = root / entry["path"]
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root):
            raise ValueError("Source file escapes explicit root")
        raw = p.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Source digest mismatch: " + entry["path"])
        text = raw.decode("utf-8-sig")
        data[entry["path"]] = json.loads(text) if p.suffix == ".json" else list(
            csv.DictReader(io.StringIO(text)))
    net = data[SOURCE_FILES[0]]
    stations = data[SOURCE_FILES[2]]
    if (net["buses"] != list(range(1, 34)) or net["line_ids"] != list(range(1, 33))
            or net["fixed_open_ties"] != list(range(33, 38))
            or net["dt_h"] != .25 or net["fixed_q_over_p"] != .33
            or len(stations) != 8 or net["station_ports"] != [20] * 8
            or net["station_kw"] != [1800] * 8
            or np.asarray(net["baseline_p_15min"]).shape != (96, 33)):
        raise ValueError("Not the declared IEEE33/L08 physical source")
    if (net["station_buses"] != [int(s["n_c"]) for s in stations]
            or net["station_road_nodes"] != [int(s["road_node_c"]) for s in stations]
            or net["station_buses"][0] != 21 or net["station_buses"][2] != 24):
        raise ValueError("Station mapping disagrees with the fixed CS1/CS3 areas")
    net["from"], net["to"] = net["line_from"][:], net["line_to"][:]
    net["shed_cost"] = [50. if b in CRITICAL else 10. for b in net["buses"]]
    arcs = [dict(id=int(a["a"]), source=int(a["m_road"]), target=int(a["n_road"]),
                 distance_km=float(a["D_a_km"]), freeflow_h=float(a["T0_a_min"]) / 60,
                 corridor_id=int(a["corridor_id"]), parent_arc_id=int(a["parent_arc_id"]))
            for a in data[SOURCE_FILES[1]]]
    if (len(arcs) != 104 or len({a["id"] for a in arcs}) != 104
            or {a["corridor_id"] for a in arcs} != set(range(1, 45))
            or any((a["parent_arc_id"] - 1) // 2 + 1 != a["corridor_id"] for a in arcs)
            or any(a["distance_km"] <= 0 or a["freeflow_h"] <= 0 for a in arcs)
            or not all(math.isfinite(a[k]) for a in arcs for k in ("distance_km", "freeflow_h"))):
        raise ValueError("Road source is not the declared parent-corridor subdivision")
    if {a[k] for a in arcs for k in ("source", "target")} != set(range(1, 37)):
        raise ValueError("Expected 28 backbone/depot nodes plus eight station access nodes")
    by_ev = {}
    for row in data[SOURCE_FILES[3]]:
        by_ev.setdefault(row["ev_id"], []).append(row)
    if len(by_ev) != 30:
        raise ValueError("Expected the original 30 normal business trajectories")
    for rows in by_ev.values():
        rows.sort(key=lambda r: float(r["start_relative_dispatch_h"]))
        previous_t, previous_dc = 0., float(rows[0]["initial_dc_kwh"])
        for r in rows:
            start, end = (float(r[k]) for k in ("start_relative_dispatch_h", "end_relative_dispatch_h"))
            a, b = float(r["battery_start_dc_kwh"]), float(r["battery_end_dc_kwh"])
            expected = a - 1.2 * float(r["distance_km"]) + float(r["normal_charging_dc_kwh"]) - float(r["normal_idle_dc_kwh"])
            if (abs(start - previous_t) > 1e-7 or end <= start or abs(a - previous_dc) > 1e-7
                    or abs(b - expected) > 1e-7 or not 0 <= min(a, b) <= max(a, b) <= 500):
                raise ValueError("Normal trajectory discontinuity or energy reset")
            previous_t, previous_dc = end, b
        if abs(previous_t - 24.) > 1e-7:
            raise ValueError("Normal trajectory must cover the entire operating day")
    registry = data[SOURCE_FILES[4]]
    if registry.get("raw_N_phi") != len(registry["rows"]):
        raise ValueError("Registry row count mismatch")
    if (registry.get("line_source") != "recovery.outage_current"
            or registry.get("road_source") != "physical_road_blocked persistent"
            or len({tuple(row) for row in registry["rows"]}) != len(registry["rows"])):
        raise ValueError("Unsupported or duplicate original registry coordinates")
    return dict(network=net, arcs=arcs, business=by_ev, registry=registry,
                source_refs=dict(manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                                 files=entries, snapshot_sha256=_digest([
                                     (e["path"], e["bytes"], e["sha256"]) for e in sorted(entries, key=lambda e: e["path"])])))


def _normal_state(rows, departure, offset):
    row = next(r for r in rows if float(r["start_relative_dispatch_h"]) <= departure < float(r["end_relative_dispatch_h"]))
    a, b = float(row["start_relative_dispatch_h"]), float(row["end_relative_dispatch_h"])
    fraction = (departure - a) / (b - a)
    dc = float(row["battery_start_dc_kwh"]) + fraction * (float(row["battery_end_dc_kwh"]) - float(row["battery_start_dc_kwh"])) - offset
    return dict(ev=row["ev_id"], event_id=row["source_event_id"], kind=row["kind"],
                time_h=departure, event_start_h=a, event_end_h=b, event_fraction=fraction,
                node=int(row["start_node"]), arc_id=int(row["road_arc_id"]) if row["kind"] == "TRAVEL" else None,
                departure_dc=dc, initial_dc=float(row["initial_dc_kwh"]) - offset,
                initial_offset_dc=offset, source_start_dc=float(row["battery_start_dc_kwh"]),
                source_end_dc=float(row["battery_end_dc_kwh"]),
                rule="linear interpolation of the entire source ledger with one initial offset; no resets")


def _shortest(arcs, origin, destination):
    """Deterministic free-flow shortest path; ties broken by ordered arc IDs."""
    adj = {}
    for a in arcs:
        adj.setdefault(a["source"], []).append(a)
    todo = [(0., (), origin)]
    seen = set()
    while todo:
        cost, ids, node = heapq.heappop(todo)
        if node in seen:
            continue
        if node == destination:
            return list(ids)
        seen.add(node)
        for a in adj.get(node, ()):
            if a["target"] not in seen:
                heapq.heappush(todo, (cost + a["freeflow_h"], ids + (a["id"],), a["target"]))
    raise ValueError("No forecast route between normal location and declared station")


def _packages(source, config, groups, fleet_size):
    arcs, net = source["arcs"], source["network"]
    by_arc = {a["id"]: a for a in arcs}
    # 30 types are available for the predeclared heterogeneity axis; each source
    # template is used once before any cloning. Offsets change INITIAL charge.
    roster = sorted(source["business"])
    packages, types = [], []
    departures = [64 - int(4 * config["gap_h"]), 64 - int(2 * config["gap_h"]), 64]
    travel = np.tile([a["freeflow_h"] * 1.05 for a in arcs], (96, 1))
    blocked = np.zeros(travel.shape, bool)
    counts = [fleet_size // groups + int(g < fleet_size % groups) for g in range(groups)]
    for g in range(groups):
        rows = source["business"][roster[g]]
        offset = 60. + 50. * g / max(1, groups - 1)
        if any(float(r["battery_end_dc_kwh"]) - offset < 100 for r in rows):
            raise ValueError("Declared initial charge offset violates normal battery floor")
        types.append(dict(g=g, count=counts[g], source_ev=roster[g], capacity_kwh=500.,
                          initial_dc=float(rows[0]["initial_dc_kwh"]) - offset,
                          initial_offset_dc=offset, power_kw=90., traction_kwh_per_km=1.2,
                          discharge_efficiency=.95, auxiliary_kw=1., qualification_dc=150.,
                          terminal_floor_dc=100., normal_trajectory_sha256=_digest(rows)))
        for c in (0, 2):
            destination = net["station_road_nodes"][c]
            for choice, d in enumerate(departures):
                state = _normal_state(rows, d / 4, offset)
                origin = state["node"]
                prefix, fractions = [], []
                if state["arc_id"] is not None and state["event_fraction"] > 0:
                    arc = by_arc[state["arc_id"]]
                    origin = arc["target"]
                    prefix, fractions = [arc["id"]], [1. - state["event_fraction"]]
                tail = _shortest(arcs, origin, destination)
                ids, fractions = prefix + tail, fractions + [1.] * len(tail)
                route = replay_route(ids, d / 4, arcs, travel, blocked, fractions=fractions,
                                     origin=state["node"], destination=destination)
                # The common start is fixed before results. It preserves a
                # meaningful deployment cost difference at a shared service time.
                k = max(66, math.ceil(4 * route["arrival"]))
                for duration in (4, 8):
                    r = k + duration - 1
                    if not route["success"] or r >= 96:
                        raise ValueError("Fixed catalogue contains a forecast-infeasible route")
                    cap = ebar(state["departure_dc"], route["distance"], route["arrival"], k, r)
                    if cap < 15:
                        raise ValueError("Fixed catalogue contains an infeasible minimum contract")
                    h = len(packages)
                    packages.append(dict(id=f"g{g}/c{c}/d{d}/r{r}", package_id=h,
                                         route_id=h, g=g, c=c, d=d, k=k, r=r,
                                         departure_dc=state["departure_dc"], distance=route["distance"],
                                         arrival=route["arrival"], Ebar=cap,
                                         B=state["departure_dc"] - 1.2 * route["distance"] - 100.,
                                         arc_ids=ids, arc_fractions=fractions,
                                         route=dict(arc_ids=ids, fractions=fractions, origin=state["node"],
                                                    destination=destination, forecast_steps=route["steps"]),
                                         normal_trajectory=state, departure_choice=choice,
                                         reservation_hours=duration / 4))
    return packages, types, counts


def _rng(seed, split, bank, index, config, count=None):
    address = dict(family=FAMILY, seed=seed, split=split, bank=bank, index=index,
                   config=config, count=count)
    key = _digest(address)
    return np.random.Generator(np.random.PCG64(int(key, 16))), dict(address=address, sha256=key, bit_generator="PCG64")


def _probabilities(config):
    p = config["joint_nominal"]
    return np.array([.5 + p, .25 - p, .25 - p, p])  # 00,10,01,11


def _physical(source, config, rng, stream, *, stratum=None, forecast=False):
    probabilities = _probabilities(config)
    edges = np.r_[0., np.cumsum(probabilities)]
    if forecast:
        weather = .5 * probabilities[0]
    else:
        weather = float(rng.uniform()) if stratum is None else float(rng.uniform(edges[stratum], edges[stratum + 1]))
    regime = min(3, int(np.searchsorted(edges[1:], weather, side="right")))
    grid, road = bool(regime & 1), bool(regime & 2)
    severity = 0.5 if forecast else float(rng.uniform())
    clock = np.arange(96) / 4
    block_h = 16. - config["gap_h"] / 2
    duration = config["early_duration_h"]
    off = np.zeros((96, 32), bool)
    failure = np.full(32, np.inf)
    recovery = np.full(32, np.inf)
    if forecast:
        failure[[19, 22]], recovery[[19, 22]] = 17., [17.5, 17.25]
    elif grid:
        failure[19], recovery[19] = 16.5, 16.5 + duration
        # A smaller second area suffers only under the shared severe event.
        if road:
            failure[22], recovery[22] = 16.5, 16.5 + duration
    off[:] = (clock[:, None] >= failure) & (clock[:, None] < recovery)
    arcs = source["arcs"]
    corridor = np.array([a["corridor_id"] for a in arcs])
    blocked = (clock[:, None] >= block_h) & (corridor[None, :] == 1) & road
    freeflow = np.array([a["freeflow_h"] for a in arcs])
    jitter = np.zeros((96, len(arcs))) if forecast else rng.uniform(-.01, .01, (96, len(arcs)))
    multiplier = 1.04 + .02 * severity + jitter
    travel = np.where(blocked, np.inf, freeflow[None, :] * multiplier)
    baseline = np.asarray(source["network"]["baseline_p_15min"], float)
    load = baseline * .35
    # Preserve source diurnal profiles, declare new area magnitudes (kW at
    # t=16.5): 90+10 core versus 27+3 marginal, ratio 10:3. All other loads
    # remain positive wherever the source baseline is positive.
    for bus, kw in ((21, 90.), (22, 10.), (24, 27.), (25, 3.)):
        if baseline[66, bus - 1] <= 0:
            raise ValueError("Missing source demand in declared service area")
        load[:, bus - 1] = kw * baseline[:, bus - 1] / baseline[66, bus - 1]
    load_jitter = np.zeros((96, 33)) if forecast else rng.uniform(-.03, .03, (96, 33))
    load *= 1. + load_jitter
    physical = dict(line_operating_outages=off, load_p_kw=load, load_q_kvar=.33 * load,
                    road_blocked=blocked, road_travel_hours=travel, times_h=clock,
                    line_first_failure_h=failure, line_recovery_h=recovery,
                    road_block_h=block_h if road else math.inf)
    # The event is read back from generated physical arrays, not a post-policy label.
    control_arcs = np.flatnonzero(corridor == 1)
    G = float(off[66, 19])
    B = float(blocked[66, control_arcs].all())
    features = np.array([G, B, G * B])
    primitive = dict(stream=stream, weather_uniform=weather, weather_regime=regime,
                     severity=severity, stratum_probabilities=probabilities.tolist(),
                     road_block_h=physical["road_block_h"],
                     innovations_sha256=_array_digest(jitter, load_jitter),
                     mechanism="shared weather regime -> line20 failure / corridor1 block; "
                               "joint severe regime -> smaller line23 island; independent road/load jitter",
                     forecast=forecast)
    primitive["sha256"] = _digest(primitive)
    return physical, features, primitive


def _scene(source, config, packages, rng, stream, **kwargs):
    physical, features, primitive = _physical(source, config, rng, stream, **kwargs)
    states = []
    cache = {}
    for p in packages:
        key = (p["g"], p["c"], p["d"])
        if key not in cache:
            cache[key] = replay_route(p["arc_ids"], p["d"] / 4, source["arcs"],
                                     physical["road_travel_hours"], physical["road_blocked"],
                                     fractions=p["arc_fractions"], origin=p["route"]["origin"],
                                     destination=p["route"]["destination"])
        states.append(dict(cache[key]))
    fingerprint = _array_digest(physical["line_operating_outages"], physical["load_p_kw"],
                                physical["road_blocked"], physical["road_travel_hours"])
    return dict(physical=physical, route_states=states, features=features,
                primitive_identity=primitive, physical_sha256=fingerprint,
                identity=_digest(dict(primitive=primitive["sha256"], physical=fingerprint,
                                      packages=_digest(packages))))


def _calibration(source, config, seed, split, count):
    rows, ids, hashes = [], [], []
    for i in range(count):
        rng, address = _rng(seed, split, "CAL", i, config)
        physical, features, primitive = _physical(source, config, rng, address)
        rows.append(features)
        ids.append(primitive["sha256"])
        hashes.append(_array_digest(physical["load_p_kw"], physical["road_travel_hours"]))
    values = np.array(rows)
    mean = values.mean(axis=0)
    z = 1.959963984540054
    center = (mean + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(mean * (1 - mean) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return dict(features=values, mean=mean, lower=np.maximum(0., center - half),
                upper=np.minimum(1., center + half), weights=np.full(count, 1 / count),
                primitive_ids=ids, physical_hashes=hashes,
                identity=_digest(dict(streams=ids, physical=hashes, features=values)),
                rule=dict(name="two-sided pointwise Wilson score", confidence=.95, z=z,
                          sample_count=count, iid=True, weighted=False,
                          formula="center=(mean+z²/(2N))/(1+z²/N); half=z*sqrt(mean*(1-mean)/N+z²/(4N²))/(1+z²/N); clip(center±half,0,1)",
                          simultaneous_coverage_claim=False,
                          support_based_widening=False, force_nominal_into_bounds=False,
                          note="CAL-only binary intervals; no optimizer/policy observations; no promise of nonempty finite-support ambiguity set"))


def _feature_scope(source):
    registry = source["registry"]
    lookup = {tuple(row): i for i, row in enumerate(registry["rows"])}
    refs = [lookup.get(row) for row in FEATURE_ROWS]
    original_counts = dict(Counter(row[0] for row in registry["rows"]))
    retained = Counter(row[0] for row, ref in zip(FEATURE_ROWS, refs) if ref is not None)
    return dict(feature_names=list(FEATURE_NAMES), joint_row_indices=list(JOINT_ROW_INDICES),
                original_rows=[list(r) for r in FEATURE_ROWS], original_row_indices=refs,
                registry_match=all(i is not None for i in refs),
                event_moment_status="REGISTERED_COORDINATE_NEW_CALIBRATION" if all(i is not None for i in refs) else "NEW_MOMENT_ASSUMPTION",
                registry_semantics_evidence=REGISTRY_SEMANTICS_EVIDENCE,
                original_row_count=registry["raw_N_phi"], declared_row_count=3,
                retained_by_family=dict(retained), omitted_by_family={k: n - retained[k] for k, n in original_counts.items()},
                marginal_rule="remove only joint_row_indices=[2]; preserve support and all other rows/bounds",
                identical_to_frozen322311=False,
                differences=[
                    "Only declared L/R/LR coordinates retained; all W, LL, RR, load and load_lag and other L/R/LR coordinates omitted.",
                    "Fresh synthetic physical support, fresh independent CAL2000, pointwise Wilson intervals; no old bounds/certificates reused.",
                    "Training is weighted four-regime stratification, not the old pool2000 medoid/represented-source weighting.",
                    "Shared categorical engineering weather response replaces the old spatial wind/tree/hazard/recovery law.",
                    "Only lines20/23 fail, once each, with short fixed repair windows; corridor1 alone has a persistent block.",
                    "New demand profiles, initial charges, cloned business types, fixed catalogue and forecast; original network/energy/qualification kernels retained.",
                    "Actual shortest feeder trips are retained even below the suggested 10-40km development range; no artificial detours or SOC resets.",
                    "No 45139-row compression, support projection, equivalence-class reduction or old frozen-model identity is claimed."])


def make_case(config, source_root, split, seed, scenario_count=400, fleet_size=30, groups=6):
    """Build a fresh physical case; a failed strict-margin audit is returned honestly.

    TRAIN uses deterministic proportional four-regime strata, at least one per
    regime, then a seeded permutation. Every member has independent continuous
    road/load innovations. ``weights[s]=nominal_mass[stratum]/stratum_count``;
    these MUST be used for SAA. CAL is independent IID (2000 rows, equal weights).
    Validation/OOS generation is not opened implicitly by this interface.
    """
    config = _config(config)
    seed = _integer(seed, "seed", 0)
    if split not in {"development", "validation", "stress", "oos"}:
        raise ValueError("Explicit development/validation/stress/oos split required")
    if split == "development" and seed not in (101, 202):
        raise ValueError("Development seeds are predeclared as 101 and optional neighbor 202")
    count = _integer(scenario_count, "scenario_count", 4)
    fleet_size, groups = _integer(fleet_size, "fleet_size"), _integer(groups, "groups")
    if groups > min(30, fleet_size):
        raise ValueError("Each of at most 30 exact source types requires a vehicle")
    source = read_source(source_root)
    packages, types, type_counts = _packages(source, config, groups, fleet_size)
    scope = _feature_scope(source)
    probabilities = _probabilities(config)
    # Hamilton-style sequential largest deficit with a one-member minimum.
    allocations = np.ones(4, dtype=int)
    for _ in range(count - 4):
        allocations[int(np.argmax(probabilities * count - allocations))] += 1
    strata = np.repeat(np.arange(4), allocations)
    order_rng, _ = _rng(seed, split, "TRAIN_ORDER", 0, config, count)
    strata = order_rng.permutation(strata)
    scenarios = []
    for i, s in enumerate(strata):
        rng, address = _rng(seed, split, "TRAIN", i, config, count)
        scene = _scene(source, config, packages, rng, address, stratum=int(s))
        scene.update(id=f"{split}/train{count}/{i:04d}", stratum=int(s),
                     weight=float(probabilities[s] / allocations[s]))
        scenarios.append(scene)
    rng, address = _rng(0, "forecast", "DETERMINISTIC", 0, config)
    forecast = _scene(source, config, packages, rng, address, forecast=True)
    forecast["id"] = "structured_forecast_later_internal_outages"
    calibration = _calibration(source, config, seed, split, 2000)
    weights = np.array([s["weight"] for s in scenarios])
    case = dict(network=source["network"], packages=packages, type_counts=type_counts,
                types=types, roads=source["arcs"], scenarios=scenarios,
                calibration_features=calibration["features"], calibration=calibration,
                calibration_mean=calibration["mean"], calibration_lower=calibration["lower"],
                calibration_upper=calibration["upper"], calibration_rule=calibration["rule"],
                forecast=forecast, feature_names=list(FEATURE_NAMES),
                joint_row_indices=list(JOINT_ROW_INDICES), feature_scope=scope,
                features=np.array([s["features"] for s in scenarios]), weights=weights,
                sampling=dict(mode="proportional_stratified", stratum_order=["00", "10", "01", "11"],
                              probabilities=probabilities, counts=allocations,
                              natural_iid=False, saa_weights_required=True,
                              unweighted_empirical_weights=np.full(count, 1 / count),
                              unweighted_weights_are_nominal=False),
                config=config, source_refs=source["source_refs"])
    scientific = dict(family=FAMILY, config=config, split=split, seed=seed,
                      scenario_count=count, calibration_count=2000, fleet_size=fleet_size,
                      groups=groups, source=source["source_refs"]["snapshot_sha256"],
                      catalogue=_digest(packages), calibration=calibration["identity"],
                      scenarios=[s["identity"] for s in scenarios], weights=weights,
                      forecast=forecast["identity"], features=scope,
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      kernel_sha256={m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                     for m in (energy_kernel, physics_kernel, replay_kernel)},
                      semantics="500kWh/90kW/1.2kWh-km/0.95/1kW dwell/150DC qualification/100DC floor/15AC minimum; costs25/30/1/.02; shedding50/10")
    case["identity"] = dict(scientific, sha256=_digest(scientific),
                            prior_bound_reuse_allowed=False, old_frozen_case=False)
    case["physical_audit"] = audit_physical_case(case)
    return case


def _zero_ev(net, physical):
    """Construct/check the lossless radial zero-EV witness, without optimization."""
    off, load = physical["line_operating_outages"], physical["load_p_kw"]
    fr, to = np.array(net["from"]) - 1, np.array(net["to"]) - 1
    connected = np.zeros((96, 33), bool)
    connected[:, 0] = True
    if any(u >= v for u, v in zip(fr, to)):
        raise ValueError("Expected the source topological IEEE33 line order")
    for j, (u, v) in enumerate(zip(fr, to)):
        connected[:, v] = connected[:, u] & ~off[:, j]
    shed = np.where(connected, 0., load)
    downstream = load - shed
    flow = np.zeros((96, 32))
    for j in reversed(range(32)):
        flow[:, j] = np.where(off[:, j], 0., downstream[:, to[j]])
        downstream[:, fr[j]] += flow[:, j]
    voltage = np.full((96, 33), net["u_ref"], dtype=float)
    for j, (u, v) in enumerate(zip(fr, to)):
        voltage[:, v] = np.where(connected[:, v], voltage[:, u] - flow[:, j] * (net["mp"][j] + .33 * net["mq"][j]), net["u_ref"])
    pbal = shed - load
    pbal[:, 0] += downstream[:, 0]
    for j, (u, v) in enumerate(zip(fr, to)):
        pbal[:, u] -= flow[:, j]
        pbal[:, v] += flow[:, j]
    drop = voltage[:, to] - voltage[:, fr] + flow * (np.array(net["mp"]) + .33 * np.array(net["mq"]))
    residual = max(float(np.max(abs(pbal))),
                   float(np.max(np.maximum(abs(flow) - np.array(net["p_limit"]), 0))),
                   float(np.max(np.maximum(.33 * abs(flow) - np.array(net["q_limit"]), 0))),
                   float(np.max(np.maximum(abs(drop) - net["big_m"] * off, 0))),
                   float(max(0, net["u_min"] - voltage.min(), voltage.max() - net["u_max"])))
    return dict(passed=residual <= 1e-7, max_residual=residual,
                forced_island_ens_kwh=float(.25 * shed.sum()),
                status="constructed radial no-EV witness; no optimization", solver_calls=0)


def audit_physical_case(case):
    """Generator-level checks only; independent A1 primal/dual audit is separate."""
    errors, margins = [], []
    net, packages, arcs = case["network"], case["packages"], case["roads"]
    corridor_ids = {a["id"] for a in arcs if a["corridor_id"] == 1}
    zero_results = []
    for scene in [case["forecast"], *case["scenarios"]]:
        ph = scene["physical"]
        if (ph["line_operating_outages"].shape != (96, 32)
                or ph["load_p_kw"].shape != (96, 33)
                or not np.array_equal(ph["load_q_kvar"], .33 * ph["load_p_kw"])
                or not np.isfinite(ph["load_p_kw"]).all() or (ph["load_p_kw"] < 0).any()):
            errors.append("PHYSICAL_AXES_OR_POWER_FACTOR")
        if len(scene["route_states"]) != len(packages):
            errors.append("PACKAGE_ROUTE_ALIGNMENT")
            continue
        zero_results.append(_zero_ev(net, ph))
        for p, state in zip(packages, scene["route_states"]):
            replay = replay_route(p["arc_ids"], p["d"] / 4, arcs, ph["road_travel_hours"], ph["road_blocked"],
                                  fractions=p["arc_fractions"], origin=p["route"]["origin"], destination=p["route"]["destination"])
            if replay != state:
                errors.append("ROUTE_REPLAY_DIFFERS")
            if abs(p["Ebar"] - ebar(p["departure_dc"], p["distance"], p["arrival"], p["k"], p["r"])) > 1e-10:
                errors.append("FORECAST_ENERGY_CAP_DIFFERS")
            qualified, _, _, available = qualify(p["departure_dc"], p["distance"], state["arrival"], p["k"], p["r"], state["success"])
            if not 0 <= available <= 90 * (p["r"] - p["k"] + 1) / 4 or (not state["success"] and qualified):
                errors.append("ORIGINAL_QUALIFICATION_FAILED")
            if p["c"] == 0 and p["departure_choice"] in (0, 2) and math.isfinite(ph["road_block_h"]):
                entries = [s["entry_h"] for s in replay["steps"] if s["arc_id"] in corridor_ids]
                if not entries:
                    errors.append("CONTROL_CORRIDOR_NOT_ON_REFERENCE_ROUTE")
                else:
                    margin = ph["road_block_h"] - max(entries) if p["departure_choice"] == 0 else min(entries) - ph["road_block_h"]
                    margins.append(margin)
                    if p["departure_choice"] == 0 and (not replay["success"] or replay["arrival"] > 16.5):
                        errors.append("EARLY_ROUTE_NOT_AVAILABLE_AT_CRITICAL_TIME")
        # h must be the actual registered L/R product, not any outage/any delay.
        ids = [j for j, a in enumerate(arcs) if a["corridor_id"] == 1]
        g, b = float(ph["line_operating_outages"][66, 19]), float(ph["road_blocked"][66, ids].all())
        if not np.array_equal(scene["features"], [g, b, g * b]):
            errors.append("FEATURE_PHYSICAL_MISMATCH")
        if np.any(np.diff(ph["road_blocked"].astype(int), axis=0) < 0):
            errors.append("ROAD_NOT_PERSISTENT")
    if not all(z["passed"] for z in zero_results):
        errors.append("ZERO_EV_NETWORK_WITNESS_FAILED")
    if zero_results[0]["forced_island_ens_kwh"] <= 0 or case["forecast"]["physical"]["road_blocked"].any():
        errors.append("FORECAST_POSITIVE_DEMAND_OR_PASSABLE_ROADS_FAILED")
    if not margins or min(margins) <= .25:
        errors.append("STRICT_QUARTER_HOUR_MARGIN_FAILED")
    # Only line20 separates core buses21/22: prove this on the actual topology.
    mask = np.zeros(32, bool)
    mask[19] = True
    core = next(g for g in components(net, mask) if 21 in g)
    if core != [21, 22]:
        errors.append("CONTROL_LINE_CORE_COMPONENT_MISMATCH")
    unique = len({s["physical_sha256"] for s in case["scenarios"]})
    if unique != len(case["scenarios"]):
        errors.append("DUPLICATE_PHYSICAL_TRAJECTORIES")
    if len(packages) != len(case["type_counts"]) * 12:
        errors.append("CATALOGUE_SIZE")
    return dict(passed=not errors, errors=sorted(set(errors)), solver_calls=0,
                scope="generator physical checks; not independent A1 equivalence or convergence",
                min_block_margin_h=min(margins) if margins else None,
                required_margin="strictly greater than .25h on both sides at actual arc entry",
                independent_physical_trajectories=unique, operational_equivalence_classes=None,
                nominal_scenarios=len(case["scenarios"]), packages=len(packages),
                type_counts=case["type_counts"], within_type_exact_by_construction=True,
                departure_dc_constant_across_times=False,
                route_distance_km_range=[min(p["distance"] for p in packages), max(p["distance"] for p in packages)],
                departure_soc_range=[min(p["departure_dc"] for p in packages) / 500,
                                     max(p["departure_dc"] for p in packages) / 500],
                core_component=core, registry_match=case["feature_scope"]["registry_match"],
                forecast_no_ev=zero_results[0], max_zero_ev_residual=max(z["max_residual"] for z in zero_results),
                old_physical_kernel_preserved=True, original_distribution_preserved=False)


def make_reference_policies(case):
    """Three canonical, equal-share count policies; never selected from Q/results.

    z0: one late core vehicle, z1: one early core, z2: that early core plus
    one early marginal vehicle. Each uses the one-hour reservation and 45kWh
    AC per vehicle (or its smaller physical cap, always >=15). This is a fixed
    reference comparison, not a claim these are global optima or ordered risks.
    """
    packages = case["packages"]
    count = len(packages)
    policies = []
    for name, specs in (("z0", [(0, 2)]), ("z1", [(0, 0)]), ("z2", [(0, 0), (2, 0)])):
        n, en = np.zeros(count), np.zeros(count)
        used = np.zeros(len(case["type_counts"]), int)
        for c, choice in specs:
            eligible = [(h, p) for h, p in enumerate(packages)
                        if p["c"] == c and p["departure_choice"] == choice
                        and p["reservation_hours"] == 1 and used[p["g"]] < case["type_counts"][p["g"]]]
            if not eligible:
                raise ValueError("Reference policies need at least two available vehicles")
            h, p = eligible[0]
            n[h] += 1
            en[h] += min(45., p["Ebar"])
            used[p["g"]] += 1
        parts = np.array([25 * n.sum(),
                          sum(30 * (p["r"] + 1 - p["d"]) / 4 * v for p, v in zip(packages, n)),
                          sum((p["r"] - p["k"] + 1) / 4 * v for p, v in zip(packages, n)),
                          .02 * en.sum()])
        if any(en[h] < 15 * v or en[h] > packages[h]["Ebar"] * v for h, v in enumerate(n)):
            raise ValueError("Illegal reference contract")
        for c in range(len(case["network"]["station_buses"])):
            for t in range(96):
                if sum(v for p, v in zip(packages, n) if p["c"] == c and p["k"] <= t <= p["r"]) > case["network"]["station_ports"][c]:
                    raise ValueError("Reference port limit exceeded")
        cost = sum(first_cost(p["d"], p["k"], p["r"]) * v for p, v in zip(packages, n)) + .02 * en.sum()
        if abs(cost - parts.sum()) > 1e-8 or cost <= 0:
            raise ValueError("Reference baseline cost mismatch")
        policies.append(dict(name=name, n=n, E=en, x=np.r_[n, en],
                             package_ids=np.flatnonzero(n).tolist(), cost_parts=parts,
                             first_stage_cost=cost, base_cost_rates=[25., 30., 1., .02],
                             canonical_equal_share=True, case_identity=case["identity"]["sha256"]))
    return policies
