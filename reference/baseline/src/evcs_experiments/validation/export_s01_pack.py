#!/usr/bin/env python3
"""Export the accepted S01 development evidence without drawing new samples.

This deterministic organization step does not sign S01 scientific acceptance.
The indexed per-target/split NPZ archives are the lossless equivalent of one
large merged task_scenario_replay.npz and remain immutable on the server.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import shutil
import time
import zipfile

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_source_file, frozen_control_root, historical_path, legacy_campaign_root, source_provenance)

FEATURES = ("Phi_W", "Phi_L", "Phi_R", "Phi_D", "Phi_load")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Unexpected empty export table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})


def npz_schema(path):
    """Read only NPY headers; large array values are not needed for schema."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                raise ValueError("Unexpected non-NPY member in scientific NPZ")
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported NPY header version: {version}")
            if dtype.hasobject:
                raise ValueError("Object arrays are not accepted scientific payloads")
            result[name[:-4]] = {"shape": list(shape), "dtype": dtype.str, "fortran_order": bool(fortran)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case-root", "campaign-root", "i08-acceptance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    case, campaign, i08_path, out = (p.resolve() for p in
        (args.case_root, args.campaign_root, args.i08_acceptance, args.output))
    if out == campaign or out in (case, i08_path.parent):
        raise ValueError("Output must be a new dedicated export directory")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes = {}
    def sha(path):
        path = Path(path).resolve()
        if path not in hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path] = digest.hexdigest()
        return hashes[path]
    def reference(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    def check(path, digest):
        if sha(path) != digest:
            raise ValueError(f"Accepted source/artifact hash mismatch: {path}")

    state_path = campaign / "state.json"
    state = read_json(state_path)
    # State advances after final scientific acceptance; retain the exact
    # export-time bytes so the pack's snapshot reference remains verifiable.
    shutil.copyfile(state_path, out / "state_snapshot.json")
    if state.get("campaign") != "ieee33-v54-q1-full-experiments":
        raise ValueError("State belongs to another scientific campaign")
    paths = {row["id"]: historical_path(row["acceptance"]) for row in state["specs"]["S01"]["accepted_increments"]}
    paths["S01-I08"] = i08_path
    wanted = tuple(f"S01-I{i:02d}" for i in range(1, 9))
    if not all(key in paths for key in wanted):
        raise ValueError("Unified export requires every accepted I01-I08 increment")
    cards, ancestry, source_hashes = {}, [], {}
    for key in wanted:
        path = paths[key].resolve()
        card = read_json(path)
        if card["increment"] != key or card["status"] != "PASS_INCREMENT":
            raise ValueError(f"Increment is not scientifically accepted: {key}")
        evidence = historical_path(card["evidence"]).resolve()
        allowed_roots = (campaign, legacy_campaign_root(), get_paths().project_root / "provenance", get_paths().project_root / "artifacts/legacy_transfer")
        if not any(root in evidence.parents for root in allowed_roots) or not any(root in path.parents for root in allowed_roots):
            raise ValueError("Increment evidence is outside the current or explicitly retained legacy campaign")
        for name, digest in card.get("source_hashes", {}).items():
            source = frozen_source_file(name)
            check(source, digest)
            if name in source_hashes and source_hashes[name] != digest:
                raise ValueError("Accepted increments disagree on immutable source ancestry")
            source_hashes[name] = digest
        manifest_paths = [evidence / "artifact_sha256.json"]
        if (evidence / "post_acceptance_artifact_sha256.json").exists():
            manifest_paths.append(evidence / "post_acceptance_artifact_sha256.json")
        for manifest_path in manifest_paths:
            for name, entry in read_json(manifest_path).items():
                check(evidence / name, entry["sha256"] if isinstance(entry, dict) else entry)
        for name, entry in card.get("raw_artifacts", {}).items():
            check(evidence / name, entry["sha256"] if isinstance(entry, dict) else entry)
        for audit_path, digest in card.get("independent_audits", {}).items():
            check(historical_path(audit_path), digest["sha256"] if isinstance(digest, dict) else digest)
        cards[key] = card
        ancestry.append({"increment": key, "acceptance": reference(path), "status": card["status"],
            "evidence_directory": str(evidence), "evidence_manifests": [reference(p) for p in manifest_paths],
            "test_count": card.get("test_count"), "accepted_at_utc": card.get("accepted_at_utc")})

    source = historical_path(cards["S01-I08"]["evidence"])
    routes = historical_path(cards["S01-I06"]["evidence"])
    validation = read_json(source / "increment_validation.json")
    upstream = read_json(source / "upstream_validation.json")
    if (validation["status"] != "PASS_INCREMENT_AUTOMATED" or validation["total_scenarios"] != 1920
            or upstream["preflight_only"] or not upstream["development"]
            or upstream["formal_OOS_outcomes_generated"] or upstream["regime"] != "R1"):
        raise ValueError("Expected the full 1920-scenario development R1 bank")
    for name, digest in upstream["input_hashes"].items():
        check(case / name, digest)
    events = tuple(row["event"] for row in validation["targets"])
    if len(events) != 8 or len(set(events)) != 8:
        raise ValueError("Full development delivery needs eight distinct targets")
    for name in ("features.csv", "scenario_ledger.csv"):
        shutil.copyfile(source / name, out / name)
        check(out / name, sha(source / name))

    tasks_rows, route_rows, moment_rows, archive_index, diagnostics_rows = [], [], [], [], []
    task_count = task_scenario_rows = 0
    moments_refs, task_refs, bootstrap_identities = [], [], []
    units = {"grid_hazard": "dimensionless [0,1]", "line_wind_vectors_m_s": "m/s east/north",
        "road_wind_vectors_m_s": "m/s east/north", "road_travel_time_min": "minutes",
        "load_p_kw": "kW", "load_q_kvar": "kvar; exact Q1 rated power factor",
        "load_archived_q_kvar": "kvar; archived rounded Q retained for trace",
        "task_arrival_h": "hours from service horizon start; inf means failed route",
        "task_route_time_h": "hours; failed-task accounting zero",
        "task_travel_energy_kwh": "battery kWh; failed-task accounting zero",
        "task_arrival_soc": "fraction; failed-task accounting zero",
        "task_usable_energy_dc_kwh": "battery-side DC kWh; efficiency applied later",
        "task_mu_full": "boolean, complete reserve-period availability", "features": "ordered five dimensionless features"}
    for event in events:
        bank_path = routes / f"{event}.tasks.json"
        bank = read_json(bank_path)
        index = read_json(source / f"{event}.task_index.json")
        if bank["task_sha256"] != index["task_sha256"] or not bank["certified"] or bank["status"] != "COMPLETE":
            raise ValueError("Frozen H identity/certification mismatch")
        tasks = bank["tasks"]
        ids = [task["task_id"] for task in tasks]
        if ids != index["task_ids"] or len(ids) != len(set(ids)):
            raise ValueError("Task order changed in joint bank")
        task_refs.append({"target_event": event, "frozen_tasks": reference(bank_path),
                          "task_index": reference(source / f"{event}.task_index.json"), "task_sha256": bank["task_sha256"]})
        for position, task in enumerate(tasks):
            tasks_rows.append({"target_event": event, "task_index": position, "task_sha256": bank["task_sha256"], **task})
            arcs = task["arc_ids"]
            for arc_position, arc_id in enumerate(arcs or [None], 1):
                route_rows.append({"target_event": event, "task_index": position, "task_id": task["task_id"],
                    "vehicle_id": task["vehicle_id"], "station_id": task["station_id"],
                    "route_is_empty": not bool(arcs), "route_arc_count": len(arcs),
                    "arc_position_one_based": arc_position if arcs else "", "arc_id": arc_id if arcs else "",
                    "route_distance_km": task["distance_km"], "route_freeflow_h": task["freeflow_h"]})
        moment_path = source / f"{event}.moments.json"
        moment = read_json(moment_path)
        if moment["status"] != "PASS_MOMENT_CALIBRATION" or set(moment["bounds_by_rho"]) != {"0.5", "1.0", "1.5", "2.0"}:
            raise ValueError("Missing accepted moment bounds across all four rho values")
        moments_refs.append(reference(moment_path))
        bootstrap_identities.append({"target_event": event, **read_json(source / f"{event}.bootstrap.json")})
        for rho_key in ("0.5", "1.0", "1.5", "2.0"):
            bounds = moment["bounds_by_rho"][rho_key]
            witness = bounds["feasibility"]
            if not witness["success"] or not witness["valid_witness"]:
                raise ValueError("Moment set lacks accepted feasibility witness")
            for feature_position, feature in enumerate(FEATURES):
                moment_rows.append({"target_event": event, "rho": float(rho_key), "feature": feature,
                    "feature_index": feature_position, "center": moment["center"][feature_position],
                    "original_base_halfwidth": moment["original_halfwidth"][feature_position],
                    "repaired_base_halfwidth": moment["repaired_halfwidth"][feature_position],
                    "lower": bounds["lower"][feature_position], "upper": bounds["upper"][feature_position],
                    "common_normalized_delta": moment["delta"], "optimal_delta": moment["optimal_delta"],
                    "numeric_margin_delta": moment["numeric_margin_delta"],
                    "witness_maximum_violation": witness["primal_residuals"]["maximum_violation"],
                    "source_moments_path": str(moment_path), "source_moments_sha256": sha(moment_path)})
        scenario_count = 0
        sums = {"canonical_road_blocked": 0., "canonical_road_movable": 0., "task_qualified": 0., "Phi_D": 0.}
        qualified_vehicles = set()
        for split, count in (("planning", 160), ("calibration", 80)):
            archive_path = source / f"{event}.{split}.npz"
            schema = npz_schema(archive_path)
            if schema["task_varsigma"]["shape"] != [count, len(tasks)] or schema["task_mu_full"]["shape"] != [count, len(tasks), 24]:
                raise ValueError("Task replay archive has unexpected scenario/task/hour shape")
            with np.load(archive_path, allow_pickle=False) as arrays:
                if arrays["task_ids"].tolist() != ids or len(arrays["scenario_ids"]) != count:
                    raise ValueError("NPZ task/scenario index mismatch")
                scenario_ids = arrays["scenario_ids"].tolist()
                blocked, movable, success, features = (arrays[name] for name in
                    ("road_blocked", "road_movable", "task_varsigma", "features"))
                if blocked.shape != (count, 100, 88) or features.shape != (count, 5):
                    raise ValueError("Unexpected full development domain dimensions")
                sums["canonical_road_blocked"] += float(blocked[:, :, :80].mean()) * count
                sums["canonical_road_movable"] += float(movable[:, :, :80].mean()) * count
                sums["task_qualified"] += float(success.sum())
                sums["Phi_D"] += float(features[:, 3].sum())
                for task_position in np.flatnonzero(np.any(success, axis=0)):
                    qualified_vehicles.add(tasks[int(task_position)]["vehicle_id"])
            scenario_count += count
            task_scenario_rows += count * len(tasks)
            archive_index.append({"target_event": event, "split": split, "development": True, "regime": "R1",
                "file": reference(archive_path), "scenario_count": count, "task_count": len(tasks),
                "task_sha256": bank["task_sha256"], "task_index": reference(source / f"{event}.task_index.json"),
                "scenario_ledger": reference(source / f"{event}.{split}.ledger.json"),
                "scenario_ids": scenario_ids, "array_schema": schema,
                "index_axes": {"scenario": "NPZ scenario_ids", "task": "NPZ task_ids in task_index.json order",
                               "replay_time": "NPZ times_h, [-1,24) at 0.25h", "operation_hour": "hours 0..23"}})
        vehicle_ids = {task["vehicle_id"] for task in tasks}
        summaries = bank["vehicle_summaries"]
        diagnostics_rows.append({"target_event": event, "task_count": len(tasks),
            "moving_tasks": sum(task["distance_km"] > 0 for task in tasks),
            "zero_distance_tasks": sum(task["distance_km"] == 0 for task in tasks),
            "forecast_qualified_vehicles_with_H": len(vehicle_ids),
            "vehicles_total": len(summaries), "vehicles_with_empty_H": sum(bool(row["empty_H"]) for row in summaries),
            "realized_qualified_vehicles_at_least_once": len(qualified_vehicles),
            "development_scenarios": scenario_count,
            "canonical80_blocked_fraction": sums["canonical_road_blocked"] / scenario_count,
            "canonical80_movable_fraction": sums["canonical_road_movable"] / scenario_count,
            "qualified_task_scenario_fraction": sums["task_qualified"] / (scenario_count * len(tasks)) if tasks else "NA",
            "mean_Phi_D": sums["Phi_D"] / scenario_count,
            "task_sha256": bank["task_sha256"], "assignment_state_sha256": index["assignment_state_sha256"]})
        task_count += len(tasks)
    if len(moment_rows) != 160 or len(archive_index) != 16 or task_count != 18720 or task_scenario_rows != 4492800:
        raise ValueError("Unified development pack counts differ from accepted full I08 evidence")
    write_csv(out / "tasks.csv", tasks_rows)
    write_csv(out / "task_route_arcs.csv", route_rows)
    write_csv(out / "moment_bounds.csv", moment_rows)
    write_csv(out / "target_diagnostics.csv", diagnostics_rows)
    write_json(out / "task_scenario_replay_index.json", {"format": "lossless partitioned NPZ index",
        "equivalent_logical_artifact": "task_scenario_replay.npz", "merging_required": False,
        "meaning": "The sixteen immutable partitions plus their explicit task/scenario IDs contain every field without rerunning or duplicating physical samples.",
        "static_axes": reference(source / "static_axes.npz"), "units": units, "archives": archive_index})
    contracts = sorted(frozen_control_root().glob("*CONTRACT*.md"))
    contracts += [frozen_control_root() / "PROTOCOL.md", frozen_control_root() / "specs/S01.md",
                  frozen_control_root() / "runtime_policy.json"]
    generator_contract = {"schema_version": 1, "campaign": state["campaign"], "scope": "accepted S01 development generator",
        "paper": reference(case / "00_documentation/paper_v54_IEEE33_CCG_corrected.tex"),
        "contracts": [reference(path) for path in contracts], "accepted_increment_ancestry": ancestry,
        "input_hashes": upstream["input_hashes"], "accepted_source_hashes": source_hashes, **source_provenance(),
        "exporter": reference(Path(__file__)), "target_order": list(events),
        "random_mechanisms": {"wind_amplitude": 1, "wind_location": 2, "wind_lag": 3,
            "residual_donor_order": 4, "residual_block": 5, "residual_scale": 6,
            "line_gaussian": 7, "road_uniform": 8, "load_innovation": 9, "calibration_bootstrap": 10},
        "formal_seed_roots_metadata_only": {"planning": 20260901, "calibration": 20260902, "oos": 20260903},
        "development_root": 20260911, "static_road_root": 20260904,
        "seed_key_fields": ["schema", "domain", "split", "shared_target_zero", "base_one_based_or_aggregate_zero",
                            "variant_one_based_or_aggregate_zero", "mechanism", "asset", "component"],
        "bootstrap_identities": bootstrap_identities, "feature_order": FEATURES,
        "axes": {"targets": 8, "weather_grid_points": 441, "closed_lines": 32, "road_feature_arcs": 80,
            "routing_arcs": 88, "replay_slots": 100, "replay_hours": "[-1,24) step 0.25",
            "operation_hours": 24, "buses": 33, "planning_per_target": 160, "calibration_per_target": 80},
        "units": units, "frozen_task_sources": task_refs, "moment_sources": moments_refs,
        "empty_route_encoding": "task_route_arcs.csv keeps one explicit row with route_is_empty=true and blank arc_id/position; no fictitious arc is traversed",
        "development_only": True, "formal_OOS_locked": True, "formal_outcomes_generated": False,
        "formal_planning_calibration_production_entry": "Deferred orchestration gate before S04; this pack contains accepted development seeds only",
        "additional_random_samples_drawn_by_export": 0}
    write_json(out / "generator_contract.json", generator_contract)
    evidence_for_gate = {
        1: ("Reproducible numeric payloads on identical manifests", ("S01-I01", "S01-I08")),
        2: ("Route boundary/toy behavior, reverse-arc identity and persistent failures", ("S01-I03", "S01-I04", "S01-I06", "S01-I07")),
        3: ("Frozen H, excluded target residual donor and separated random streams", ("S01-I01", "S01-I02", "S01-I06", "S01-I08")),
        4: ("Complete physical dependency chain and independent bounded hourly load", ("S01-I05", "S01-I07", "S01-I08")),
        5: ("Five decision-independent features, vehicle weights and nonempty moment sets", ("S01-I08",)),
        6: ("Eight target H/mobility/road diagnostics without a positive-benefit requirement", ("S01-I06", "S01-I08"))}
    gates = [{"gate": number, "requirement": text, "status": "ACCEPTED_INCREMENT_EVIDENCE_LINKED",
        "acceptance_references": [reference(paths[key]) for key in increments],
        "evidence_directories": [cards[key]["evidence"] for key in increments]}
        for number, (text, increments) in evidence_for_gate.items()]
    write_json(out / "generation_audit.json", {"status": "EXPORTED_PENDING_INDEPENDENT_S01_REVIEW",
        "full_S01_pass": False, "state_modified": False, "source_artifacts_modified": False,
        "random_samples_generated": 0, "formal_OOS_outcomes_generated": False,
        **source_provenance(), "created_at_utc": datetime.now(timezone.utc).isoformat(), "state_snapshot": reference(out / "state_snapshot.json"),
        "original_mutable_state_path": str(state_path),
        "gates": gates, "independent_I08_audits": cards["S01-I08"].get("independent_audits", {}),
        "accepted_increment_ancestry": ancestry, "verified_unique_file_count": len(hashes),
        "counts": {"targets": 8, "scenarios": 1920, "tasks": task_count,
            "task_scenario_rows": task_scenario_rows, "moment_bound_rows": len(moment_rows),
            "route_arc_rows_including_empty_markers": len(route_rows), "npz_partitions": len(archive_index)},
        "logical_delivery_substitution": "An explicit path/hash/schema index over 16 immutable archives replaces a physically merged task_scenario_replay.npz without loss of fields or indices",
        "runtime_seconds": time.monotonic() - started, "exporter_sha256": sha(__file__)})
    write_json(out / "artifact_manifest.json", {path.name: reference(path) for path in sorted(out.iterdir()) if path.is_file()})
    print(json.dumps({"status": "EXPORTED_PENDING_INDEPENDENT_S01_REVIEW", "output": str(out),
        "targets": 8, "scenarios": 1920, "tasks": task_count, "moment_rows": 160,
        "runtime_seconds": time.monotonic() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
