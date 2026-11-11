"""S02-I01 development checks against immutable accepted S01 evidence.

This writes automated evidence only; it cannot sign scientific acceptance.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy import sparse

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import historical_path, sha256, source_provenance
from evcs_experiments.first_stage import FirstStagePlan, assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def checked_json(path, expected):
    path = historical_path(path)
    if sha256(path) != expected:
        raise ValueError(f"Accepted evidence SHA256 mismatch: {path}")
    return json.loads(path.read_text())


def write_mapping(path, records):
    records = [asdict(row) if is_dataclass(row) else dict(row) for row in records]
    keys = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def minimum_plan(case, tasks):
    """Deterministic feasible fixture; never an optimization policy or H filter."""
    H, T = len(tasks.task_ids), len(case.periods)
    x, r = np.zeros(H), np.zeros((H, T))
    occupied = np.zeros((len(case.station_ids), T))
    reserve = np.zeros_like(occupied)
    used = set()
    station_index = {station: c for c, station in enumerate(case.station_ids)}
    for h, vehicle in enumerate(tasks.vehicle_ids):
        if vehicle in used:
            continue
        c = station_index[tasks.station_ids[h]]
        amount = tasks.block_mask[h] * tasks.p_block_kw[h]
        if (np.any(amount > tasks.p_dis_kw[h] * tasks.chi_res[h])
                or amount.sum() * case.dt_h / tasks.eta_dis[h] > tasks.e_forecast_dc_kwh[h] + 1e-10
                or np.any(occupied[c] + tasks.chi_occ[h] > case.station_plugs[c])
                or np.any(reserve[c] + amount > case.station_interface_kw[c])):
            continue
        x[h], r[h] = 1., amount
        occupied[c] += tasks.chi_occ[h]
        reserve[c] += amount
        used.add(vehicle)
    return FirstStagePlan(x, r, tasks.task_sha256, tasks.config_hash)


def inspect_plan(case, tasks, matrix, plan):
    direct = check_first_stage(case, tasks, plan)
    assembled = matrix.evaluate(plan)
    if direct["passed"] != bool(assembled.feasible):
        raise ValueError("Independent checker disagrees with sparse matrix")
    error = abs(direct["costs"]["Fpre_usd"] - assembled.objective_usd)
    if error > 1e-4 + 1e-7 * abs(assembled.objective_usd):
        raise ValueError("First-stage objective reconstruction differs")
    return {"independent": direct, "matrix_feasible": bool(assembled.feasible),
            "matrix_objective_usd": float(assembled.objective_usd),
            "matrix_max_row_violation": float(assembled.max_row_violation),
            "matrix_max_bound_violation": float(assembled.max_bound_violation),
            "matrix_max_integrality_violation": float(assembled.max_integrality_violation)}


def negative_plans(case, tasks):
    """Keep the full H and inject explicit failures without changing inputs."""
    H, T = len(tasks.task_ids), len(case.periods)
    if not H:
        return
    def plan(x, r):
        return FirstStagePlan(x, r, tasks.task_sha256, tasks.config_hash)
    x, r = np.zeros(H), np.zeros((H, T))
    r[0, 0] = 1.
    yield "unselected_reserve", plan(x, r)
    x, r = np.zeros(H), np.zeros((H, T))
    x[0] = .5
    yield "fractional_selection", plan(x, r)
    x[0] = 1.
    yield "missing_minimum_block", plan(x, r)
    x, r = np.zeros(H), np.zeros((H, T))
    r[0, 0] = -1.
    yield "negative_reserve", plan(x, r)
    for a in range(H):
        b = next((j for j in range(a + 1, H)
                  if tasks.vehicle_ids[j] == tasks.vehicle_ids[a]), None)
        if b is not None:
            x, r = np.zeros(H), np.zeros((H, T))
            x[[a, b]] = 1.
            r[a] = tasks.block_mask[a] * tasks.p_block_kw[a]
            r[b] = tasks.block_mask[b] * tasks.p_block_kw[b]
            yield "same_vehicle_two_tasks", plan(x, r)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=get_paths().data_root)
    parser.add_argument("--s01-acceptance", type=Path, required=True)
    parser.add_argument("--s01-acceptance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = get_paths()
    config_path = Path(os.environ["EVCS_CONFIG"]).resolve() if os.environ.get("EVCS_CONFIG") else None
    write_json(out / "execution.json", {"argv": sys.argv, "python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
        "paths": paths.as_dict(), "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256(config_path) if config_path else None,
        "runtime_policy_sha256": sha256(paths.spec_root / "runtime_policy.json"),
        "thread_environment": {k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}})
    write_json(out / "progress.json", {"phase": "tests", "status": "RUNNING"})
    with (out / "tests.stdout.log").open("w") as stdout, (out / "tests.stderr.log").open("w") as stderr:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                                 str(paths.project_root / "tests"), "-p", "test_*.py", "-v"],
                                cwd=paths.project_root, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise ValueError("Test suite failed; inspect the preserved test logs")
    accepted = checked_json(args.s01_acceptance, args.s01_acceptance_sha256)
    if accepted.get("status") != "PASS" or accepted.get("spec_id") != "S01":
        raise ValueError("S01 must be scientifically accepted")
    manifest = checked_json(accepted["deliverables_manifest"], accepted["deliverables_manifest_sha256"])
    contract = checked_json(manifest["generator_contract.json"]["path"],
                            manifest["generator_contract.json"]["sha256"])
    replay_index = checked_json(manifest["task_scenario_replay_index.json"]["path"],
                                manifest["task_scenario_replay_index.json"]["sha256"])
    if not contract["development_only"] or contract["formal_outcomes_generated"]:
        raise ValueError("I01 requires the accepted development bank")
    case = load_physical_case(args.case_root)
    axes_ref = replay_index["static_axes"]
    axes_path = historical_path(axes_ref["path"])
    if sha256(axes_path) != axes_ref["sha256"]:
        raise ValueError("Accepted static scenario axes changed")
    with np.load(axes_path, allow_pickle=False) as axes:
        if (tuple(axes["bus_ids"].tolist()) != case.bus_ids
                or tuple(axes["line_ids"].tolist()) != case.line_ids
                or not np.array_equal(axes["baseline_p_kw"], case.baseline_p_kw)
                or not np.array_equal(axes["baseline_q_kvar"], case.baseline_q_kvar)):
            raise ValueError("Physical adapter axes or baseline differ from accepted S01 arrays")
    evidence, objectives, summaries, matrices = {}, [], [], []
    refs = contract["frozen_task_sources"]
    if len(refs) != 8 or len({row["target_event"] for row in refs}) != 8:
        raise ValueError("Eight distinct accepted target banks are required")
    for reference in refs:
        target = reference["target_event"]
        write_json(out / "progress.json", {"phase": "first_stage", "target": target, "status": "RUNNING"})
        task_ref = reference["frozen_tasks"]
        tasks = load_frozen_tasks(historical_path(task_ref["path"]), case, task_ref["sha256"])
        if tasks.task_sha256 != reference["task_sha256"]:
            raise ValueError("H content identity differs from accepted S01")
        matrix = assemble_first_stage(case, tasks)
        target_out = out / target
        target_out.mkdir()
        sparse.save_npz(target_out / "first_stage_matrix.npz", matrix.A)
        np.savez_compressed(target_out / "bounds_objective.npz", row_lower=matrix.row_lower,
                            row_upper=matrix.row_upper, lower_bounds=matrix.lower_bounds,
                            upper_bounds=matrix.upper_bounds, integrality=matrix.integrality, c=matrix.c)
        write_mapping(target_out / "variable_map.csv", matrix.variable_map)
        write_mapping(target_out / "constraint_map.csv", matrix.constraint_map)
        matrices.append({"target": target, "variables": f"{target}/variable_map.csv",
                         "constraints": f"{target}/constraint_map.csv", "matrix_sha256": matrix.matrix_sha256})
        plans = {"zero": FirstStagePlan(np.zeros(len(tasks.task_ids)),
                  np.zeros((len(tasks.task_ids), len(case.periods))), tasks.task_sha256, tasks.config_hash),
                 "minimum_fixture": minimum_plan(case, tasks)}
        checks = {}
        for label, plan in plans.items():
            checks[label] = inspect_plan(case, tasks, matrix, plan)
            if not checks[label]["independent"]["passed"]:
                raise ValueError(f"Feasible fixture rejected: {target}/{label}")
            np.savez_compressed(target_out / (label + ".plan.npz"), x=plan.x, r=plan.r,
                                task_ids=np.asarray(tasks.task_ids), task_sha256=tasks.task_sha256,
                                config_hash=tasks.config_hash)
            objectives.append({"target": target, "plan": label,
                               **checks[label]["independent"]["costs"]})
        for label, plan in negative_plans(case, tasks):
            checks[label] = inspect_plan(case, tasks, matrix, plan)
            if checks[label]["independent"]["passed"]:
                raise ValueError(f"Injected physical failure passed: {target}/{label}")
            np.savez_compressed(target_out / (label + ".plan.npz"), x=plan.x, r=plan.r,
                                task_ids=np.asarray(tasks.task_ids), task_sha256=tasks.task_sha256,
                                config_hash=tasks.config_hash)
        write_json(target_out / "physical_check_report.json", checks)
        plan = plans["minimum_fixture"]
        expected = float(np.sum(tasks.c_act_usd * plan.x) +
                         np.sum(tasks.c_cap_usd_per_kwh * plan.r) * case.dt_h)
        if abs(expected - checks["minimum_fixture"]["matrix_objective_usd"]) > 1e-6:
            raise ValueError("Minimum procurement hand sum differs")
        summaries.append({"target": target, "H": len(tasks.task_ids), "matrix_rows": matrix.A.shape[0],
                          "matrix_columns": matrix.A.shape[1], "nnz": matrix.A.nnz,
                          "selected_fixture_tasks": int(plan.x.sum()), "checks": len(checks),
                          "task_sha256": tasks.task_sha256, "config_hash": tasks.config_hash,
                          "task_source": task_ref})
        # The explicit first development scene only validates the adapter.
        from evcs_experiments.physical_inputs import load_scenario_physical_inputs
        entry = next(row for row in replay_index["archives"]
                     if row["target_event"] == target and row["split"] == "planning")
        scenario = load_scenario_physical_inputs(historical_path(entry["file"]["path"]),
                    historical_path(entry["scenario_ledger"]["path"]), 0, case, tasks,
                    expected_npz_sha256=entry["file"]["sha256"],
                    expected_ledger_sha256=entry["scenario_ledger"]["sha256"])
        if (scenario.target_event != target or scenario.split != "planning"
                or scenario.scenario_id != entry["scenario_ids"][0]):
            raise ValueError("Physical scenario identity differs from accepted index selection")
        summaries[-1]["scenario_sha256"] = scenario.scenario_sha256
        summaries[-1]["scenario_id"] = scenario.scenario_id
        summaries[-1]["scenario_source"] = entry["file"]
        summaries[-1]["scenario_ledger"] = entry["scenario_ledger"]
        evidence[target] = checks
    write_mapping(out / "variable_constraint_map.csv", matrices)
    write_mapping(out / "objective_components.csv", objectives)
    write_json(out / "physical_check_report.json", evidence)
    write_json(out / "model_code_manifest.json", source_provenance())
    write_json(out / "increment_validation.json", {
        "increment": "S02-I01", "status": "PASS_INCREMENT_AUTOMATED",
        "scientific_acceptance_pending": True, "full_S02_pass": False,
        "scope": "Immutable adapters and first-stage sparse matrix/checker; recourse deferred to I02",
        "development_only": True, "production_experiments_executed": False,
        "new_scenarios_generated": 0, "targets": summaries,
        "case_sha256": case.case_sha256, "static_input_hashes": dict(case.input_hashes),
        "static_axes": axes_ref,
        "s01_acceptance": {"path": str(args.s01_acceptance), "sha256": args.s01_acceptance_sha256},
        "s01_manifest_sha256": accepted["deliverables_manifest_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started})
    write_json(out / "progress.json", {"phase": "complete", "status": "AWAITING_INDEPENDENT_REVIEW"})
    write_json(out / "artifact_sha256.json", {str(path.relative_to(out)): sha256(path)
               for path in sorted(out.rglob("*")) if path.is_file()})
    print(json.dumps({"status": "PASS_INCREMENT_AUTOMATED", "targets": len(summaries),
                      "output": str(out), "scientific_acceptance_pending": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
