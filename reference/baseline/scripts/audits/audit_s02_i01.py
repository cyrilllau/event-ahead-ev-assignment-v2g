#!/usr/bin/env python3
"""Independently replay saved first-stage plans and original-unit inequalities.

No experiment generation, acceptance signing, matrix assembly or solver calls.
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
from scipy import sparse

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import historical_path, sha256
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    if output.exists() or evidence == output or evidence in output.parents:
        raise ValueError("Audit must use a new file outside immutable run evidence")
    manifest = json.loads((evidence / "artifact_sha256.json").read_text())
    for name, expected in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha256(evidence / relative) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    paths = get_paths()
    provenance = json.loads((evidence / "model_code_manifest.json").read_text())
    for name, expected in provenance["current_source_hashes"].items():
        if sha256(paths.project_root / name) != expected:
            raise ValueError(f"Audit must run with the exact submitted source: {name}")
    execution = json.loads((evidence / "execution.json").read_text())
    if (execution["config_path"] and sha256(execution["config_path"]) != execution["config_sha256"]
            or sha256(paths.spec_root / "runtime_policy.json") != execution["runtime_policy_sha256"]):
        raise ValueError("Execution configuration or numeric policy changed")
    tests = (evidence / "tests.stderr.log").read_text()
    count = re.search(r"Ran (\d+) tests? in", tests)
    if (count is None or int(count[1]) < 150 or not tests.rstrip().endswith("OK") or "skipped=" in tests
            or any(name not in tests for name in
                   ("test_physical_inputs.", "test_first_stage.", "test_first_stage_check."))):
        raise ValueError("Full existing and new test suite must pass without skips")
    summary = json.loads((evidence / "increment_validation.json").read_text())
    if (summary["status"] != "PASS_INCREMENT_AUTOMATED" or summary["full_S02_pass"]
            or not summary["development_only"] or summary["new_scenarios_generated"] != 0
            or summary["production_experiments_executed"] or len(summary["targets"]) != 8
            or len({row["target"] for row in summary["targets"]}) != 8):
        raise ValueError("Wrong increment scope or incomplete targets")
    case = load_physical_case()
    if case.case_sha256 != summary["case_sha256"]:
        raise ValueError("Physical baseline identity mismatch")
    static = summary["static_axes"]
    if sha256(historical_path(static["path"])) != static["sha256"]:
        raise ValueError("Accepted static scenario axes changed")
    totals, outputs, negative_count = 0, [], 0
    for target in summary["targets"]:
        source = target["task_source"]
        tasks = load_frozen_tasks(historical_path(source["path"]), case, source["sha256"])
        if tasks.task_sha256 != target["task_sha256"] or tasks.config_hash != target["config_hash"]:
            raise ValueError("Target task/config identity mismatch")
        for key in ("scenario_source", "scenario_ledger"):
            reference = target[key]
            if sha256(historical_path(reference["path"])) != reference["sha256"]:
                raise ValueError("Accepted scenario evidence changed")
        root = evidence / target["target"]
        matrix = sparse.load_npz(root / "first_stage_matrix.npz")
        with np.load(root / "bounds_objective.npz", allow_pickle=False) as data:
            bounds = {key: data[key] for key in data.files}
        saved = json.loads((root / "physical_check_report.json").read_text())
        for name, result in saved.items():
            expected_feasible = name in ("zero", "minimum_fixture")
            if result["independent"]["passed"] != expected_feasible or result["matrix_feasible"] != expected_feasible:
                raise ValueError("A feasible plan failed or a fault injection passed")
        for name in saved:
            expected_feasible = name in ("zero", "minimum_fixture")
            with np.load(root / (name + ".plan.npz"), allow_pickle=False) as data:
                if tuple(data["task_ids"].tolist()) != tasks.task_ids:
                    raise ValueError("Saved plan task order changed")
                plan = FirstStagePlan(data["x"], data["r"], str(data["task_sha256"]), str(data["config_hash"]))
            direct = check_first_stage(case, tasks, plan)
            z = np.concatenate((plan.x, plan.r.ravel()))
            activity = matrix @ z
            violation = float(max(np.max(bounds["row_lower"] - activity, initial=0.),
                                  np.max(activity - bounds["row_upper"], initial=0.),
                                  np.max(bounds["lower_bounds"] - z, initial=0.),
                                  np.max(z - bounds["upper_bounds"], initial=0.)))
            integers = bounds["integrality"] == 1
            integer_violation = float(np.max(np.abs(z[integers] - np.rint(z[integers])), initial=0.))
            matrix_feasible = violation <= 1e-8 and integer_violation <= 1e-8
            cost = float(bounds["c"] @ z)
            if (direct["passed"] != expected_feasible or matrix_feasible != expected_feasible
                    or abs(cost - direct["costs"]["Fpre_usd"]) > 1e-4):
                raise ValueError("Saved plan failed independent physical/matrix/objective replay")
            negative_count += int(not expected_feasible)
            outputs.append({"target": target["target"], "plan": name, "audit_passed": True,
                            "expected_feasible": expected_feasible,
                            "costs": direct["costs"], "max_matrix_violation": violation,
                            "max_normalized_violation": direct["max_normalized_violation"]})
        totals += len(tasks.task_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS_INDEPENDENT_FIRST_STAGE_AUDIT", "increment": "S02-I01",
              "full_S02_pass": False, "test_count": int(count[1]), "targets": 8,
              "total_tasks": totals, "saved_plans_replayed": len(outputs), "results": outputs,
              "negative_plans_replayed": negative_count, "scenario_physics_recomputed": False,
              "evidence": str(evidence), "artifact_manifest_sha256": sha256(evidence / "artifact_sha256.json"),
              "audit_source_sha256": sha256(__file__),
              "acceptance_note": "Requires review of scenario adapter/unit-test evidence before I01 acceptance; Q1 recourse remains I02."}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output), "test_count": report["test_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
