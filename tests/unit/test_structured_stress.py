"""G stress on temporary synthetic sources only: no private inputs or solves."""

from copy import deepcopy
import csv
import math
import runpy

import numpy as np
import pytest

from ev_assignment.l08 import catalogue, matrix, recourse, risk
from ev_assignment.l08.physics import ebar
from ev_assignment_experiments.datasets import structured as source
from ev_assignment_experiments.datasets import structured_round2 as fixed
from ev_assignment_experiments.datasets import structured_stress as stress


@pytest.fixture(scope="module", autouse=True)
def no_solvers():
    import scipy.optimize

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic stress tests must never call a solver")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in ((matrix.Matrix, "solve"), (matrix, "linprog"),
                            (scipy.optimize, "linprog"), (scipy.optimize, "milp"),
                            (recourse, "solve"), (risk, "solve")):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def synthetic_inputs(pytestconfig, tmp_path_factory):
    definitions = runpy.run_path(str(pytestconfig.rootpath / "tests/unit/test_structured_generator.py"))
    root = definitions["source_root"].__wrapped__(tmp_path_factory)
    ledger = root / source.SOURCE_FILES[3]
    with ledger.open() as stream:
        rows = list(csv.DictReader(stream))
    # Different source initial energies make accidental total-SOC equality
    # observable. Change the entire business trace, preserving continuity.
    for row in rows:
        shift = .5 * int(row["ev_id"][1:])
        for key in ("initial_dc_kwh", "battery_start_dc_kwh", "battery_end_dc_kwh"):
            row[key] = float(row[key]) - shift
    definitions["_csv"](ledger, rows)
    definitions["_write_manifest"](root)
    return root


@pytest.fixture(scope="module")
def base(synthetic_inputs, no_solvers):
    return fixed.make_fixed_load_case(synthetic_inputs, "stress", 2101, 4)


@pytest.fixture
def paired_base(base, synthetic_inputs, monkeypatch):
    raw = source.read_source(synthetic_inputs)
    invocations = []

    def build(inputs, split, seed, scenario_count, **kwargs):
        invocations.append((inputs, split, seed, scenario_count, kwargs))
        return deepcopy(base)

    def forbidden(*args, **kwargs):
        pytest.fail("G conversion must not redraw any physical or CAL randomness")

    monkeypatch.setattr(fixed, "make_fixed_load_case", build)
    monkeypatch.setattr(source, "read_source", lambda inputs: deepcopy(raw))
    for name in ("_rng", "_physical", "_calibration", "make_case"):
        monkeypatch.setattr(source, name, forbidden)
    return raw, invocations


@pytest.mark.parametrize("groups", stress.GROUPS)
def test_real_type_rebuild_pairs_every_physical_array(groups, base, paired_base):
    raw, calls = paired_base
    before = source._digest(base)
    case = stress.make_stress_case("synthetic-memory-source", groups, scenario_count=4)
    assert calls == [("synthetic-memory-source", "stress", 2101, 4,
                      dict(fleet_size=30, groups=6))]
    assert source._digest(base) == before
    assert case["type_counts"] == [30 // groups] * groups
    assert sum(case["type_counts"]) == 30
    assert len(case["packages"]) == 12 * groups
    assert case["stress_protocol"]["first_stage_variable_count"] == 24 * groups
    assert [t["source_ev"] for t in case["types"]] == sorted(raw["business"])[:groups]
    for g, typ in enumerate(case["types"]):
        assert typ["initial_offset_dc"] == 60 + 50 * g / (groups - 1)
        assert typ["initial_dc"] == 500 - .5 * g - typ["initial_offset_dc"]
        assert typ["capacity_kwh"] == 500 and typ["power_kw"] == 90
        assert typ["normal_trajectory_sha256"] == source._digest(raw["business"][typ["source_ev"]])
    assert case["network"] == base["network"] and case["roads"] == base["roads"]
    assert case["network"]["station_ports"] == [20] * 8
    assert case["fleet_scale"] == 1. and case["config"] == fixed.BASE_CONFIG
    for old, new in zip([base["forecast"], *base["scenarios"]],
                        [case["forecast"], *case["scenarios"]]):
        assert old["physical_sha256"] == new["physical_sha256"]
        assert old["primitive_identity"] == new["primitive_identity"]
        assert new["parent_scene_identity"] == old["identity"] != new["identity"]
        assert len(new["route_states"]) == 12 * groups
        for key in old["physical"]:
            a, b = np.asarray(old["physical"][key]), np.asarray(new["physical"][key])
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        for p, state in zip(case["packages"], new["route_states"]):
            ph = old["physical"]
            expected = source.replay_route(p["arc_ids"], p["d"] / 4, raw["arcs"],
                ph["road_travel_hours"], ph["road_blocked"], fractions=p["arc_fractions"],
                origin=p["route"]["origin"], destination=p["route"]["destination"])
            assert state == expected
    for key in ("calibration", "calibration_features", "calibration_mean", "calibration_lower",
                "calibration_upper", "calibration_rule", "weights", "features", "sampling",
                "feature_names", "joint_row_indices", "source_refs"):
        assert source._digest(case[key]) == source._digest(base[key])
    assert case["physical_audit"]["passed"], case["physical_audit"]
    assert case["physical_audit"]["strict_audits_completed"]
    assert case["healthy_grid_audit"]["scenario_count"] == 5
    assert all(s["original_matrix_checked"] for s in case["healthy_grid_audit"]["scenarios"])
    assert "reference_policies" not in case and "pool_expansion" not in case
    assert case["identity"]["groups"] == groups and case["identity"]["fleet_size"] == 30
    assert case["identity"]["sha256"] != base["identity"]["sha256"]
    assert not case["identity"]["prior_bound_reuse_allowed"]
    assert not case["stress_protocol"]["r3_physical_revision"]


@pytest.mark.parametrize("groups", stress.GROUPS)
def test_soc_weighting_counts_vehicles_once_and_energy_semantics_survive(groups, paired_base):
    case = stress.make_stress_case("synthetic-memory-source", groups, scenario_count=4)
    report = case["soc_report"]["stress"]
    counts = case["type_counts"]
    expected_initial = math.fsum((30 // groups) * (500 - .5 * g - 60 - 50 * g / (groups - 1))
                                 for g in range(groups))
    assert report["weighted_initial_dc_kwh"] == pytest.approx(expected_initial, abs=1e-10)
    assert case["soc_report"]["initial_dc_delta_kwh"] != 0
    assert len(report["departures"]) == 3
    for row in report["departures"]:
        assert row["weighted_total_dc_kwh"] == math.fsum(
            c * e for c, e in zip(counts, row["per_type_dc_kwh"]))
        assert row["weighted_mean_dc_kwh"] == row["weighted_total_dc_kwh"] / 30
    for p in case["packages"]:
        trace = p["normal_trajectory"]
        assert p["departure_dc"] == (trace["source_start_dc"] + trace["event_fraction"] *
            (trace["source_end_dc"] - trace["source_start_dc"]) - trace["initial_offset_dc"])
        assert p["Ebar"] == ebar(p["departure_dc"], p["distance"], p["arrival"], p["k"], p["r"])
    first = catalogue.build_first(case["packages"], counts, 8)
    assert len(first["cost"]) == 24 * groups
    assert np.all(first["parts"][0, :12 * groups] == 25)
    assert np.all(first["parts"][3, 12 * groups:] == .02)


def test_default_contract_is_frozen_before_io(monkeypatch):
    class StopBeforeGeneration(Exception):
        pass

    calls = []

    def stop(*args, **kwargs):
        calls.append((args, kwargs))
        raise StopBeforeGeneration

    monkeypatch.setattr(fixed, "make_fixed_load_case", stop)
    plan = stress.preregistered_protocol()
    with pytest.raises(StopBeforeGeneration):
        stress.make_stress_case("unopened-source", 30)
    assert calls == [(("unopened-source", "stress", 2101, 400), dict(fleet_size=30, groups=6))]
    assert plan["groups"] == [3, 15, 30] and not plan["equal_total_soc_claim"]
    plan["groups"].append(6)
    assert stress.preregistered_protocol()["groups"] == [3, 15, 30]


@pytest.mark.parametrize("groups,seed,count", [(6, 2101, 400), (True, 2101, 400),
    (15., 2101, 400), (31, 2101, 400), (3, -1, 400), (3, 2101, 3)])
def test_invalid_request_fails_before_io(groups, seed, count, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid stress request reached source IO")

    monkeypatch.setattr(fixed, "make_fixed_load_case", forbidden)
    monkeypatch.setattr(source, "read_source", forbidden)
    with pytest.raises(ValueError):
        stress.make_stress_case("unopened-source", groups, seed, count)


def test_strict_margin_failure_is_rerun_and_retained(paired_base, monkeypatch):
    original = source._packages

    def late_early_route(*args):
        packages, types, counts = original(*args)
        # A deliberately bad synthetic early label must not inherit G6's PASS.
        for p in packages:
            if p["g"] == 2 and p["c"] == 0 and p["departure_choice"] == 2:
                p["departure_choice"] = 0
        return packages, types, counts

    monkeypatch.setattr(source, "_packages", late_early_route)
    case = stress.make_stress_case("synthetic-memory-source", 3, scenario_count=4)
    audit = case["physical_audit"]
    assert case["base_physical_audit"]["passed"]
    assert not audit["passed"] and audit["status"] == "PHYSICAL_GATE_FAILED"
    assert "STRICT_QUARTER_HOUR_MARGIN_FAILED" in audit["errors"]
    assert "EARLY_ROUTE_NOT_AVAILABLE_AT_CRITICAL_TIME" in audit["errors"]
    assert audit["min_block_margin_h"] <= .25


def test_construction_failure_is_explicit_not_repaired(paired_base, monkeypatch):
    def fail(*args):
        raise ValueError("Declared initial charge offset violates normal battery floor")

    monkeypatch.setattr(source, "_packages", fail)
    case = stress.make_stress_case("synthetic-memory-source", 30, scenario_count=4)
    assert not case["catalogue_complete"] and not case["physical_audit"]["passed"]
    assert case["physical_audit"]["errors"] == ["PACKAGE_CONSTRUCTION_FAILED"]
    assert "normal battery floor" in case["stress_protocol"]["construction_error"]
    assert not case["physical_audit"]["strict_audits_completed"]
    assert case["physical_audit"]["model_infeasible"] is None
    assert case["soc_report"]["stress"] is None
    assert not case["packages"] and not case["types"]
    assert all(not s["route_states"] for s in case["scenarios"])


def test_source_change_between_base_and_type_read_fails(paired_base, monkeypatch):
    raw, _ = paired_base
    altered = deepcopy(raw)
    altered["source_refs"]["snapshot_sha256"] = "changed-input-bundle"
    monkeypatch.setattr(source, "read_source", lambda inputs: altered)
    with pytest.raises(ValueError, match="Source bundle changed"):
        stress.make_stress_case("synthetic-memory-source", 3, scenario_count=4)


def test_group_identity_and_probability_support_are_new(paired_base):
    cases = [stress.make_stress_case("synthetic-memory-source", g, scenario_count=4) for g in stress.GROUPS]
    assert len({c["identity"]["sha256"] for c in cases}) == 3
    assert len({s["identity"] for c in cases for s in c["scenarios"]}) == 12
    assert len({c["identity"]["calibration"] for c in cases}) == 1
    assert len({source._digest(c["identity"]["physical_scenarios"]) for c in cases}) == 1
    again = stress.make_stress_case("synthetic-memory-source", 3, scenario_count=4)
    assert source._digest(again["identity"]) == source._digest(cases[0]["identity"])
