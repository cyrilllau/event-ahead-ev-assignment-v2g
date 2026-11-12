"""Portable physical-generator tests. No private fixture and no solver calls.

The fixture has the real input schemas/cardinalities, with explicitly synthetic
roads and loads. Private-source verification is separately run against the
user-supplied external root; these fixture tests never claim private data parity.
"""

import csv
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
import pytest

from ev_assignment.l08.energy import battery_trace
from ev_assignment.l08.physics import ebar, qualify
from ev_assignment.physics.route_replay import replay_route
from ev_assignment_experiments.datasets import structured as s


def _write_manifest(root):
    files = []
    for name in s.SOURCE_FILES:
        raw = (root / name).read_bytes()
        files.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    (root / "source-manifest.json").write_text(json.dumps(dict(
        schema_version=1, kind="structured-physical-source", files=files)))


def _csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def source_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("explicit-structured-source")
    fr = list(range(1, 18)) + [2, 19, 20, 21, 3, 23, 24, 6, 26, 27, 28, 29, 30, 31, 32]
    to = list(range(2, 19)) + [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    station_buses = [21, 15, 24, 3, 10, 14, 32, 27]
    net = dict(buses=list(range(1, 34)), line_ids=list(range(1, 33)),
               fixed_open_ties=list(range(33, 38)), line_from=fr, line_to=to,
               station_buses=station_buses, station_road_nodes=list(range(29, 37)),
               station_kw=[1800] * 8, station_ports=[20] * 8,
               p_limit=[10000.] * 32, q_limit=[4000.] * 32,
               mp=[1e-6] * 32, mq=[1e-6] * 32, u_ref=1., u_min=.9025,
               u_max=1.1025, big_m=.2, dt_h=.25, fixed_q_over_p=.33,
               baseline_p_15min=[[0.] + [10.] * 32 for _ in range(96)])
    (root / s.SOURCE_FILES[0]).write_text(json.dumps(net))
    # 40 backbone corridors on 24 road nodes, four depot connectors. Eight
    # corridor subdivisions add the L08 station access nodes, for 104 arcs.
    edges = [(i, i + 1) for i in range(1, 24)] + [(24, 1)]
    edges += [(i, i + 3) for i in range(1, 17)]
    edges += [(25 + i, i + 1) for i in range(4)]
    split_nodes = dict(zip((1, 7, 6, 2, 10, 8, 12, 14), range(29, 37)))
    arcs = []
    for corridor, (u, v) in enumerate(edges, 1):
        chain = [u, split_nodes[corridor], v] if corridor in split_nodes else [u, v]
        for direction, nodes in enumerate((chain, chain[::-1])):
            for a, b in zip(nodes, nodes[1:]):
                arcs.append(dict(a=len(arcs) + 1, m_road=a, n_road=b, D_a_km=2.,
                                 T0_a_min=1.2, corridor_id=corridor,
                                 parent_arc_id=2 * corridor - 1 + direction))
    _csv(root / s.SOURCE_FILES[1], arcs)
    _csv(root / s.SOURCE_FILES[2], [dict(c=f"CS{i+1}", n_c=bus, road_node_c=29+i)
                                 for i, bus in enumerate(station_buses)])
    business = []
    # All vehicles have an actual normal trip crossing a departure time, so a
    # constant-per-type departure SOC would be observably wrong.
    travel_arc = next(a for a in arcs if a["m_road"] == 3 and a["n_road"] == 4)
    for g in range(30):
        for i, (a, b, kind, u, v, distance, start, end, idle) in enumerate((
            (0., 14.75, "OFFSITE_STANDBY", 3, 3, 0., 500., 400., 100.),
            (14.75, 15.25, "TRAVEL", 3, 4, 2., 400., 397.6, 0.),
            (15.25, 24., "OFFSITE_STANDBY", 4, 4, 0., 397.6, 397.6, 0.),
        )):
            business.append(dict(ev_id=f"V{g:02d}", source_event_id=f"V{g:02d}/E{i}",
                                 start_relative_dispatch_h=a, end_relative_dispatch_h=b,
                                 kind=kind, start_node=u, end_node=v, distance_km=distance,
                                 road_arc_id=travel_arc["a"] if kind == "TRAVEL" else "",
                                 initial_dc_kwh=500., battery_start_dc_kwh=start,
                                 battery_end_dc_kwh=end, normal_charging_dc_kwh=0.,
                                 normal_idle_dc_kwh=idle))
    _csv(root / s.SOURCE_FILES[3], business)
    registry = dict(rows=[list(r) for r in s.FEATURE_ROWS], raw_N_phi=3,
                    line_source="recovery.outage_current", road_source="physical_road_blocked persistent")
    (root / s.SOURCE_FILES[4]).write_text(json.dumps(registry))
    _write_manifest(root)
    return root


@pytest.fixture(scope="module")
def case(source_root):
    return s.make_case(dict(gap_h=1., early_duration_h=.25, joint_nominal=.08),
                       source_root, "development", 101, scenario_count=16)


def test_preregistered_candidates_and_strict_configuration(source_root):
    candidates = s.preregistered_candidates()
    assert len(candidates) == 12
    assert len({tuple(c.items()) for c in candidates}) == 12
    for bad in ({}, dict(candidates[0], fee=1.), dict(candidates[0], gap_h=.75)):
        with pytest.raises(ValueError, match="frozen12"):
            s.make_case(bad, source_root, "development", 101)
    with pytest.raises(ValueError, match="Development seeds"):
        s.make_case(candidates[0], source_root, "development", 1101)


def test_original_energy_business_path_and_catalogue(case):
    assert case["physical_audit"]["passed"], case["physical_audit"]
    assert len(case["packages"]) == 72 and case["type_counts"] == [5] * 6
    for g in range(6):
        packages = [p for p in case["packages"] if p["g"] == g]
        assert len(packages) == 12
        assert len({(p["c"], p["d"], p["r"] - p["k"] + 1) for p in packages}) == 12
        assert len({p["departure_dc"] for p in packages}) > 1
        for p in packages:
            n = p["normal_trajectory"]
            expected = n["source_start_dc"] + n["event_fraction"] * (n["source_end_dc"] - n["source_start_dc"]) - n["initial_offset_dc"]
            assert p["departure_dc"] == expected
            assert p["Ebar"] == ebar(p["departure_dc"], p["distance"], p["arrival"], p["k"], p["r"])
            assert p["normal_trajectory"] == next(q["normal_trajectory"] for q in packages if q["d"] == p["d"])
            qualified, mu, _, available = qualify(p["departure_dc"], p["distance"], p["arrival"], p["k"], p["r"])
            assert qualified and available == p["Ebar"]
            dispatch = mu.astype(float) * min(available, 45.) / (.25 * mu.sum())
            trace = battery_trace(p["departure_dc"] - 1.2 * p["distance"], p["arrival"], (p["r"] + 1) / 4, dispatch)
            assert trace["min_dc"] >= 100.
    partial = next(p for p in case["packages"] if p["normal_trajectory"]["kind"] == "TRAVEL")
    assert partial["arc_fractions"][0] == .5


def test_replay_uses_actual_entry_not_departure_or_precomputed_arrival():
    arcs = [dict(id=7, source=1, target=2, distance_km=2.),
            dict(id=9, source=2, target=3, distance_km=3.)]
    times = np.full((96, 2), .5)
    blocked = np.zeros((96, 2), bool)
    blocked[2:, 1] = True
    early = replay_route([7, 9], 0., arcs, times, blocked)
    assert not early["success"] and math.isinf(early["arrival"])
    assert early["steps"][1]["entry_h"] == .5
    assert early["steps"][1]["slot"] == 2
    residual = replay_route([7, 9], 0., arcs, times, blocked, fractions=[.25, 1.])
    assert residual["success"] and residual["arrival"] == .625
    assert residual["distance"] == 3.5
    with pytest.raises(ValueError, match="connected path"):
        replay_route([9, 7], 0., arcs, times, blocked)


def test_joint_rows_calibration_and_stratified_weight_semantics(case):
    assert case["feature_names"] == list(s.FEATURE_NAMES)
    assert case["joint_row_indices"] == [2]
    assert case["features"].shape == (16, 3)
    assert case["calibration_features"].shape == (2000, 3)
    assert case["calibration_rule"]["name"] == "two-sided pointwise Wilson score"
    assert not case["calibration_rule"]["support_based_widening"]
    features = case["features"]
    np.testing.assert_array_equal(features[:, 2], features[:, 0] * features[:, 1])
    np.testing.assert_allclose(case["weights"] @ features, [.25, .25, .08], atol=1e-15)
    assert not np.array_equal(case["weights"], case["sampling"]["unweighted_empirical_weights"])
    cal = case["calibration"]
    np.testing.assert_array_equal(cal["mean"], cal["features"].mean(axis=0))
    # Independent score-test inversion, not a repeat of the interval formula.
    z = cal["rule"]["z"]
    for bound in (cal["lower"], cal["upper"]):
        statistic = (cal["mean"] - bound) ** 2 / (bound * (1 - bound) / 2000)
        np.testing.assert_allclose(statistic, z ** 2, rtol=2e-13)
    assert len(set(cal["physical_hashes"])) == 2000
    assert not set(cal["primitive_ids"]) & {r["primitive_identity"]["sha256"] for r in case["scenarios"]}
    assert not case["feature_scope"]["identical_to_frozen322311"]
    assert case["feature_scope"]["omitted_by_family"] == {"L": 0, "R": 0, "LR": 0}


def test_forecast_is_a_real_internal_outage(case):
    scene = case["forecast"]
    assert scene["physical"]["line_operating_outages"].any()
    assert not scene["physical"]["road_blocked"].any()
    assert all(r["success"] for r in scene["route_states"])
    assert scene["physical"]["line_first_failure_h"][19] == 17.
    assert case["physical_audit"]["forecast_no_ev"]["forced_island_ens_kwh"] > 0
    assert not scene["features"].any()  # no EARLY event, not no disaster


def test_canonical_reference_counts_and_original_costs(case):
    policies = s.make_reference_policies(case)
    assert [p["name"] for p in policies] == ["z0", "z1", "z2"]
    np.testing.assert_allclose([p["first_stage_cost"] for p in policies], [71.9, 101.9, 203.8])
    for policy in policies:
        np.testing.assert_array_equal(policy["x"], np.r_[policy["n"], policy["E"]])
        assert policy["canonical_equal_share"]
        for g, capacity in enumerate(case["type_counts"]):
            assert sum(n for p, n in zip(case["packages"], policy["n"]) if p["g"] == g) <= capacity
        for p, n, en in zip(case["packages"], policy["n"], policy["E"]):
            assert 15 * n <= en <= n * p["Ebar"]
    # Same fixed reference policies, simply a smaller fleet and distinct types.
    small = dict(case, type_counts=[1] * 6)
    broad = s.make_reference_policies(small)[2]
    assert len({case["packages"][h]["g"] for h in broad["package_ids"]}) == 2


def test_boundary_is_a_failed_case_not_a_deleted_candidate(source_root):
    c = s.make_case(dict(gap_h=.5, early_duration_h=.25, joint_nominal=.08),
                    source_root, "development", 101, scenario_count=8, fleet_size=6)
    assert not c["physical_audit"]["passed"]
    assert "STRICT_QUARTER_HOUR_MARGIN_FAILED" in c["physical_audit"]["errors"]
    assert len(c["packages"]) == 72 and len(c["scenarios"]) == 8
    assert c["physical_audit"]["min_block_margin_h"] <= .25


def test_reproducibility_distinct_stage_identity_and_no_solver(source_root, case, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("gurobipy", "scipy.optimize")):
            raise AssertionError("Generator attempted a solver import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    same = s.make_case(case["config"], source_root, "development", 101, scenario_count=16)
    assert same["identity"]["sha256"] == case["identity"]["sha256"]
    np.testing.assert_array_equal(same["features"], case["features"])
    large = s.make_case(case["config"], source_root, "development", 101, scenario_count=64)
    assert large["identity"]["sha256"] != case["identity"]["sha256"]
    assert large["calibration"]["identity"] == case["calibration"]["identity"]
    assert not {x["physical_sha256"] for x in large["scenarios"]} & {x["physical_sha256"] for x in case["scenarios"]}
    assert large["identity"]["prior_bound_reuse_allowed"] is False
    neighbor = s.make_case(case["config"], source_root, "development", 202, scenario_count=16)
    assert neighbor["calibration"]["identity"] != case["calibration"]["identity"]


def test_source_tamper_and_manifest_paths_fail_closed(source_root, tmp_path):
    import shutil

    dest = tmp_path / "source"
    shutil.copytree(source_root, dest)
    p = dest / s.SOURCE_FILES[0]
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="digest mismatch"):
        s.read_source(dest)
    manifest = json.loads((dest / "source-manifest.json").read_text())
    manifest["files"][0]["path"] = "../outside.json"
    (dest / "source-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly the five"):
        s.read_source(dest)


def test_registry_missing_pair_is_explicit_new_assumption(source_root, tmp_path):
    import shutil

    dest = tmp_path / "source"
    shutil.copytree(source_root, dest)
    path = dest / s.SOURCE_FILES[4]
    reg = json.loads(path.read_text())
    reg["rows"] = reg["rows"][:2]
    reg["raw_N_phi"] = 2
    path.write_text(json.dumps(reg))
    _write_manifest(dest)
    scope = s._feature_scope(s.read_source(dest))
    assert scope["original_row_indices"] == [0, 1, None]
    assert scope["event_moment_status"] == "NEW_MOMENT_ASSUMPTION"
    assert not scope["registry_match"]


def test_audit_detects_manually_changed_route_state(case):
    corrupted = deepcopy(case)
    corrupted["scenarios"][0]["route_states"][0]["arrival"] += .1
    audit = s.audit_physical_case(corrupted)
    assert not audit["passed"] and "ROUTE_REPLAY_DIFFERS" in audit["errors"]
