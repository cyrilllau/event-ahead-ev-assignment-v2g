#!/usr/bin/env python3
"""Independent I08 assembly audit using only previously accepted I01-I07 kernels.

Does not import or call joint_scenarios, validate_s01_joint, or scenario_features.
Every scenario redraws its addressed primitive streams independently, without
the joint generator's cache. Grid, line and road wind are sampled separately.
The five aggregate features and bootstrap/moment LP are audited elsewhere.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import platform
import sys
import time

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_file, historical_path, legacy_transfer_manifest, LEGACY_CASE_NAME, source_provenance, verify_scientific_kernel_identity)


def sha(path):
    return hashlib.sha256(historical_path(path).read_bytes()).hexdigest()


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def read_json(path):
    return json.loads(historical_path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case-root", "joint-output", "route-evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    case, source, routes, out = (p.resolve() for p in
        (args.case_root, args.joint_output, args.route_evidence, args.output))
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runner = frozen_runner_root()

    from evcs_experiments.case_inputs import load_case_inputs
    from evcs_experiments.wind_scenarios import load_wind_bank, draw_parameters
    from evcs_experiments.wind_field import latlon_to_xy
    from evcs_experiments.line_outages import load_line_model
    from evcs_experiments.road_disruptions import load_road_model
    from evcs_experiments.road_connectors import derive_road_connectors
    from evcs_experiments.load_scenarios import load_load_model, draw_load_errors, realize_load
    from evcs_experiments.forecast_routes import Arc, ForecastNetwork
    from evcs_experiments.task_replay import prepare_task_bank, replay_prepared
    from evcs_experiments.random_streams import canonical_hash, event_order

    summary = read_json(source / "increment_validation.json")
    assert summary["status"] in ("PASS_PREFLIGHT", "PASS_INCREMENT_AUTOMATED")
    assert summary["formal_OOS_outcomes_generated"] is False
    assert summary["production_experiments_executed"] is False
    artifact_hashes = read_json(source / "artifact_sha256.json")
    for name, digest in artifact_hashes.items():
        assert sha(source / name) == digest, ("source artifact hash", name)
    # This guard also records the assembly implementation bytes, without
    # importing or executing that implementation.
    for name, digest in summary["source_hashes"].items():
        assert sha(runner / name) == digest, ("source code hash", name)
    upstream = read_json(source / "upstream_validation.json")
    for name, digest in upstream["input_hashes"].items():
        assert sha(case / name) == digest, ("physical source hash", name)
    assert sha(upstream["last_acceptance"]) == upstream["last_acceptance_sha256"]
    assert sha(upstream["route_acceptance"]) == upstream["route_acceptance_sha256"]
    route_acceptance = read_json(upstream["route_acceptance"])
    assert route_acceptance["status"] == "PASS_INCREMENT"
    assert route_acceptance["increment"] == "S01-I06"

    # Load independent accepted components, never a JointContext.
    verify_scientific_kernel_identity()
    case_inputs = load_case_inputs(case)
    wind = load_wind_bank(case)
    lines = load_line_model(case)
    roads = load_road_model(case, include_connectors=True)
    load = load_load_model(case)
    events = event_order(case_inputs.event_ids)
    expected_targets = events[:1] if upstream["preflight_only"] else events
    assert tuple(row["event"] for row in summary["targets"]) == expected_targets

    # Construct all native grid coordinates directly from source field axes.
    latitude = wind.fields[events[0]].latitude
    longitude = wind.fields[events[0]].longitude
    assert latitude.shape == longitude.shape == (21,)
    for event in events:
        assert np.array_equal(wind.fields[event].latitude, latitude)
        assert np.array_equal(wind.fields[event].longitude, longitude)
    native_lat, native_lon = np.meshgrid(latitude, longitude, indexing="ij")
    latlon = np.column_stack((native_lat.reshape(-1), native_lon.reshape(-1)))
    grid_points = latlon_to_xy(latlon)
    grid_hash = canonical_hash({"latitude": latitude.tolist(), "longitude": longitude.tolist(),
        "points_latlon_sha256": array_hash(latlon), "points_xy_sha256": array_hash(grid_points),
        "point_order": "latitude-major ij meshgrid, longitude varying fastest"})
    assert grid_points.shape == (441, 2) and grid_hash == upstream["grid_sha256"]

    connectors = derive_road_connectors(case)
    with (case / "04_transportation/A__directed_road_arcs.csv").open(newline="", encoding="utf-8-sig") as handle:
        arc_rows = list(csv.DictReader(handle)) + list(connectors.arc_rows)
    arc_rows.sort(key=lambda row: int(row["a"]))
    arcs = tuple(Arc(int(row["a"]), int(row["m_road"]), int(row["n_road"]),
                    float(row["D_a_km"]), float(row["T0_a_min"]) / 60.) for row in arc_rows)
    assert tuple(arc.a for arc in arcs) == roads.arc_ids == tuple(range(1, 89))
    assert lines.line_ids == tuple(range(1, 33))

    comparisons = {}
    def compare(name, actual, expected, context):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            detail = {"field": name, "context": context, "actual_shape": actual.shape,
                      "expected_shape": expected.shape, "actual_dtype": str(actual.dtype),
                      "expected_dtype": str(expected.dtype)}
            if actual.shape == expected.shape:
                finite = np.isfinite(actual) & np.isfinite(expected)
                detail["nonfinite_masks_equal"] = bool(np.array_equal(np.isfinite(actual), np.isfinite(expected)))
                if np.any(finite):
                    detail["max_finite_difference"] = float(np.max(np.abs(actual[finite].astype(float) - expected[finite].astype(float))))
            write_json(out / "mismatch.json", detail)
            raise AssertionError(detail)
        record = comparisons.setdefault(name, {"comparisons": 0, "scalar_elements": 0, "maximum_difference": 0.})
        record["comparisons"] += 1
        record["scalar_elements"] += int(actual.size)

    static_expect = {"points_xy_km": grid_points, "point_ids": np.arange(441),
        "line_points_xy_km": lines.points_xy_km, "road_points_xy_km": roads.points_xy_km,
        "line_ids": np.array(lines.line_ids), "road_ids": np.array(roads.arc_ids),
        "bus_ids": np.array(load.bus_ids), "baseline_p_kw": load.baseline_p_kw,
        "baseline_q_kvar": load.baseline_q_kvar, "baseline_archived_q_kvar": load.archived_baseline_q_kvar}
    with np.load(source / "static_axes.npz", allow_pickle=False) as static:
        assert set(static.files) == set(static_expect)
        for name, value in static_expect.items():
            compare("static/" + name, static[name], value, "static")

    completed, per_target = [], []
    for target_event in expected_targets:
        bank_path = routes / f"{target_event}.tasks.json"
        assert sha(bank_path) == route_acceptance["raw_artifacts"][bank_path.name]["sha256"]
        bank = read_json(bank_path)
        target = case_inputs.target(target_event, lead_h=1.)
        prepared = prepare_task_bank(bank, case_inputs.vehicle_parameters, target)
        times = target.replay_times_h
        assert np.array_equal(times, np.arange(-4, 96) / 4)
        task_index = read_json(source / f"{target_event}.task_index.json")
        assert task_index["target_event"] == target_event
        assert task_index["assignment_state_sha256"] == target.state_sha256
        assert task_index["task_sha256"] == prepared.task_sha256
        assert tuple(task_index["task_ids"]) == prepared.task_ids
        assert np.array_equal(task_index["times_h"], times)
        target_count = 0
        for split, full_variants in (("planning", 20), ("calibration", 10)):
            ledger_file = read_json(source / f"{target_event}.{split}.ledger.json")
            ledger, saved_diagnostics = ledger_file["scenarios"], ledger_file["diagnostics"]
            variants = 1 if upstream["preflight_only"] else full_variants
            expected_addresses = [(base, variant) for variant in range(variants) for base in events]
            assert [(row["base_weather_event"], row["variant_index"]) for row in ledger] == expected_addresses
            with np.load(source / f"{target_event}.{split}.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            assert tuple(arrays["task_ids"].tolist()) == prepared.task_ids
            compare("times_h", arrays["times_h"], times, f"{target_event}/{split}")
            assert arrays["scenario_ids"].tolist() == [row["scenario_id"] for row in ledger]
            for name, digest in ledger_file["array_payload_hashes"].items():
                assert array_hash(arrays[name]) == digest, ("saved array hash", target_event, split, name)
            for position, row in enumerate(ledger):
                base, variant = row["base_weather_event"], row["variant_index"]
                expected_id = f"{target_event}/development/{split}/{base}/v{variant+1:03d}/R1"
                assert row["scenario_id"] == expected_id
                # Fresh addressed primitives on every iteration: no cache.
                parameters = draw_parameters(events, target_event, base, variant, split=split, development=True)
                line_threshold = lines.draw_thresholds(events, base, variant, split=split, development=True)
                road_threshold = roads.draw_thresholds(events, base, variant, split=split, development=True)
                load_draw = draw_load_errors(load, events.index(base), variant, split=split, development=True)
                realized_load = realize_load(load, load_draw)
                # JSON transport converts tuple-valued dataclass fields to
                # lists. Compare their canonical JSON meaning, not Python's
                # container-type equality.
                assert canonical_hash(row["wind_parameters"]) == canonical_hash(parameters.as_dict())
                assert row["line_stream_id"] == line_threshold.stream_id
                assert row["road_stream_ids"] == list(road_threshold.stream_ids)
                assert row["load_draw"] == load_draw.as_dict()
                assert row["assignment_state_sha256"] == target.state_sha256
                assert row["task_sha256"] == prepared.task_sha256
                assert row["grid_sha256"] == grid_hash
                assert row["residual_donor_event"] == parameters.donor_event != target_event
                # Sample each domain separately, so a concatenation-offset
                # mistake in assembly cannot reproduce itself in this audit.
                grid_weather = wind.sample(parameters, times, grid_points, regime="R1")
                line_weather = wind.sample(parameters, times, lines.points_xy_km, regime="R1")
                road_weather = wind.sample(parameters, times, roads.points_xy_km, regime="R1")
                assert row["wind_parameters_sha256"] == grid_weather.parameters_sha256 == line_weather.parameters_sha256 == road_weather.parameters_sha256
                line = lines.replay(times, line_weather.vectors, line_threshold, lead_h=1.)
                road = roads.replay(times, road_weather.vectors, road_threshold, line, lead_h=1., regime="R1")
                identity = {"scenario_id": expected_id, "target_event": target_event,
                    "base_weather_event": base, "variant": variant + 1, "variant_index": variant,
                    "split": split, "development": True, "regime": "R1", "lead_h": 1.,
                    "residual_donor_event": parameters.donor_event}
                for key, value in identity.items():
                    assert row[key] == value, ("scenario identity", key, expected_id)
                network = ForecastNetwork(arcs, times, road.travel_time_min / 60.,
                                          network_hash=canonical_hash(identity))
                replay = replay_prepared(network, prepared, split=split, development=True, scenario_identity=identity)
                assert replay.replay_sha256 == row["task_replay_sha256"]
                expected_arrays = {"grid_hazard": grid_weather.hazard,
                    "line_wind_vectors_m_s": line_weather.vectors, "road_wind_vectors_m_s": road_weather.vectors,
                    "line_outages": line.outages, "line_operating_outages": line.operating_outages,
                    "road_blocked": road.blocked, "road_movable": road.movable,
                    "road_travel_time_min": road.travel_time_min, "load_p_kw": realized_load.p_kw,
                    "load_q_kvar": realized_load.q_kvar, "load_archived_q_kvar": realized_load.archived_q_kvar,
                    "load_epsilon": load_draw.epsilon, "load_standard_innovation": load_draw.accepted_standard_innovation,
                    "line_standard_normals": line_threshold.standard_normals,
                    "line_threshold_uniforms": line_threshold.uniforms,
                    "road_corridor_uniforms": road_threshold.corridor_uniforms}
                for field in ("arrival_h", "gamma_route", "route_time_h", "travel_energy_kwh", "arrival_soc",
                              "varsigma", "mu_full", "usable_energy_dc_kwh", "failed_arc_ids"):
                    expected_arrays["task_" + field] = getattr(replay, field)
                for name, value in expected_arrays.items():
                    compare(name, arrays[name][position], value, expected_id)
                    assert array_hash(value) == row["array_hashes"][name], ("scenario field hash", expected_id, name)
                diagnostics = saved_diagnostics[position]
                assert diagnostics["temporal_projection_slots"] == grid_weather.temporal_projection_count
                assert diagnostics["spatial_projection_points_all_domains"] == sum(w.spatial_projection_count for w in
                    (grid_weather, line_weather, road_weather))
                assert diagnostics["sampled_points"] == 561 and diagnostics["weather_grid_points"] == 441
                assert diagnostics["load_rejection_count"] == load_draw.rejection_count
                assert diagnostics["unique_task_routes"] == replay.unique_routes_replayed
                completed.append({"scenario_id": expected_id, "scenario_sha256": row["scenario_sha256"],
                    "nonfeature_fields_compared": len(expected_arrays), "fresh_primitives": True,
                    "domains_sampled_separately": [441, 32, 88]})
                target_count += 1
            del arrays
        per_target.append({"target_event": target_event, "scenarios": target_count,
            "task_sha256": prepared.task_sha256, "assignment_state_sha256": target.state_sha256})
        print(json.dumps(per_target[-1]), flush=True)
        write_json(out / "progress.json", {"completed_targets": per_target, "scenarios": len(completed)})
    assert len(completed) == summary["total_scenarios"]
    assert not any(name.endswith("joint_scenarios") or name == "validate_s01_joint" for name in sys.modules)
    assert "evcs_experiments.scenario_features" not in sys.modules
    imported_paths = {name: str(module.__file__) for name, module in sys.modules.items()
                      if name.startswith("evcs_experiments.") and getattr(module, "__file__", None)}
    result = {"status": "PASS_INDEPENDENT_CHAIN_AUDIT", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_output": str(source), "source_manifest_sha256": sha(source / "artifact_sha256.json"),
        "source_validation_sha256": sha(source / "increment_validation.json"), "preflight_only": upstream["preflight_only"],
        "scenario_count": len(completed), "targets": per_target, "comparisons": comparisons,
        "grid_sha256": grid_hash, "all_nonfeature_arrays_exact": True,
        "fresh_addressed_primitives_per_scenario": True, "joint_cache_used": False,
        "wind_domains_sampled_separately": [441, 32, 88], "feature_and_moment_audit_in_scope": False,
        "audit_script_sha256": sha(__file__), "accepted_kernel_hashes": {name: sha(path) for name, path in imported_paths.items()},
        "runtime_seconds": time.monotonic() - started, "host": platform.node(),
        "python": platform.python_version(), "numpy": np.__version__, "full_S01_pass": False}
    result.update(source_provenance())
    write_json(out / "scenario_checks.json", completed)
    write_json(out / "chain_audit.json", result)
    print(json.dumps({"status": result["status"], "scenarios": len(completed),
                      "output": str(out), "runtime_seconds": result["runtime_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
