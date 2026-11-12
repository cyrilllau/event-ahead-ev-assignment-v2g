#!/usr/bin/env python3
"""Run and record S01-I01 input/seed checks, never full-spec/science acceptance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_paths, frozen_control_root, source_provenance, acceptance_for, historical_path, verify_scientific_kernel_identity)
from evcs_experiments.case_inputs import FORECAST_KEYS, load_case_inputs
from evcs_experiments.random_streams import canonical_hash, contract, ledger, stream


def write_json(path, value):
    if Path(path).name == "increment_validation.json":
        value = {**value, **source_provenance()}
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    verify_scientific_kernel_identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_root = args.case_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner = frozen_runner_root()
    command = [sys.executable, "-m", "unittest", "discover",
               "-s", str(get_paths().project_root / 'tests'), "-p", "test_*.py", "-v"]
    tests = subprocess.run(command, cwd=case_root, capture_output=True, text=True, timeout=60)
    (output / "tests.stdout.log").write_text(tests.stdout)
    (output / "tests.stderr.log").write_text(tests.stderr)
    if tests.returncode:
        write_json(output / "increment_validation.json",
                   {"increment": "S01-I01", "status": "FAIL", "test_command": command,
                    "test_returncode": tests.returncode, "full_S01_pass": False})
        return tests.returncode
    case = load_case_inputs(case_root)
    targets, states = [], []
    for event in sorted(case.event_ids):
        for lead in (0.0, 0.5, 1.0, 2.0):
            target = case.target(event, lead)
            targets.append({
                "target_id": target.target_id, "event_id": event, "lead_h": lead,
                "state_sha256": target.state_sha256,
                "assignment_time_local": target.assignment_time_local.isoformat(),
                "service_start_time_local": target.service_start_time_local.isoformat(),
                "forecast_run_time_utc": target.forecast_run_time_utc.isoformat(),
                "forecast_issued_after_assignment": target.forecast_issued_after_assignment,
                "vehicle_count": len(target.vehicle_states),
                "station_count": len(target.station_ids),
                "operating_periods": target.operating_intervals_h,
                "replay_times_h": target.replay_times_h.tolist(),
            })
            for vehicle in target.vehicle_states:
                states.append({
                    "target_id": target.target_id, "event_id": event, "lead_h": lead,
                    "vehicle_id": vehicle.vehicle_id, "depot_id": vehicle.depot_id,
                    "road_node": vehicle.road_node, "soc": vehicle.soc,
                    "x_km": vehicle.x_km, "y_km": vehicle.y_km,
                    "longitude": vehicle.longitude, "latitude": vehicle.latitude,
                })
    rows = ledger(case.event_ids, "planning", 2, development=True)
    primitive_probes = []
    order = sorted(case.event_ids)
    for row in rows[::8]:
        base_index = order.index(row["base_weather_event"])
        primitive_probes.append({
            "pair_id": row["pair_id"],
            "diagnostic_uniforms": {
                name: stream("planning", base_index, row["variant"] - 1, name,
                             development=True).generator().random(4).tolist()
                for name in row["streams"]
            },
        })
    write_json(output / "target_manifest.json", targets)
    write_csv(output / "target_vehicle_states.csv", states)
    write_json(output / "development_seed_ledger.json", rows)
    write_csv(output / "development_sample_index.csv",
              [{k: r[k] for k in ("target_event", "base_weather_event", "variant",
                                  "split", "domain", "sample_id", "pair_id", "error_donor_event")}
               for r in rows])
    write_json(output / "development_rng_probes.json", primitive_probes)
    seed_contract = contract(case.event_ids)
    input_contract = {
        "increment": "S01-I01", "scope": "forecast-only inputs, clocks, indices and seeds",
        "forecast_npz_keys_read": list(FORECAST_KEYS),
        "analysis_arrays_read": False,
        "time_interpolation_boundary": "Endpoint hold; original archive timestamps preserved.",
        "future_physical_wind_rule": "Construct native-grid gust direction vectors before component interpolation; at_times scalar interpolation is an input convenience, not the complete physical wind generator.",
        "input_hashes": dict(case.input_hashes), "random_stream_contract": seed_contract,
        "full_scenario_generator_implemented": False,
        "tasks_and_routes_generated": False, "OOS_outcomes_generated": False,
    }
    write_json(output / "generator_contract_increment.json", input_contract)
    scientific_payload_hash = canonical_hash({
        "target_states": states, "seed_ledger": rows, "development_rng_probes": primitive_probes,
        "input_contract": input_contract,
    })
    source_files = frozen_source_paths()
    source = {str(p.relative_to(runner)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_files}
    summary = {
        "increment": "S01-I01", "status": "PASS_INCREMENT_AUTOMATED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_S01_pass": False, "production_experiments_executed": False,
        "target_lead_instances": len(targets), "vehicle_state_rows": len(states),
        "development_ledger_rows": len(rows),
        "development_shared_pair_count": len(primitive_probes),
        "OOS_outcomes_generated": False, "test_command": command,
        "test_returncode": tests.returncode, "test_output_file": "tests.stderr.log",
        "scientific_payload_sha256": scientific_payload_hash, "source_hashes": source,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "host": platform.node()},
        "remaining_S01": ["physical wind/residual generator", "line/road/load chain",
                          "forecast route search and H", "fixed-route replay",
                          "five features and moment feasibility"],
    }
    write_json(output / "increment_validation.json", summary)
    manifest = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "artifact_sha256.json", manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
