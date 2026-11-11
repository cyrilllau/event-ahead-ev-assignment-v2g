#!/usr/bin/env python3
"""Replay saved Q1 solutions and numeric LP dual evidence, without re-solving."""
from pathlib import Path
import argparse
import json
import re

import numpy as np
from scipy import sparse

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256
from evcs_experiments.recourse import RecourseSolution
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.validate_s02_recourse import load_case_from_i01, SOLUTION_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    if output.exists() or output == evidence or evidence in output.parents:
        raise ValueError("Independent audit requires a new file outside run evidence")
    manifest_path = evidence / "artifact_sha256.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(evidence / name) != expected:
            raise ValueError(f"Submitted evidence changed: {name}")
    paths = get_paths()
    code = json.loads((evidence / "model_code_manifest.json").read_text())
    for name, expected in code["current_source_hashes"].items():
        if sha256(paths.project_root / name) != expected:
            raise ValueError(f"Current audit source differs from submitted version: {name}")
    execution = json.loads((evidence / "execution.json").read_text())
    if (execution["config_path"] and sha256(execution["config_path"]) != execution["config_sha256"]
            or sha256(paths.spec_root / "runtime_policy.json") != execution["runtime_policy_sha256"]):
        raise ValueError("Execution configuration or numeric policy changed")
    log = (evidence / "tests.stderr.log").read_text()
    count = re.search(r"Ran (\d+) tests? in", log)
    if (count is None or int(count[1]) < 179 or not log.rstrip().endswith("OK") or "skipped=" in log
            or any(name not in log for name in ("test_recourse.", "test_recourse_check.", "test_recourse_physics."))):
        raise ValueError("Full existing and new physics tests must pass without skips")
    validation = json.loads((evidence / "increment_validation.json").read_text())
    if (validation["status"] != "PASS_INCREMENT_AUTOMATED" or validation["increment"] != "S02-I02"
            or validation["full_S02_pass"] or validation["production_experiments_executed"]
            or validation["formal_OOS_outcomes_generated"] or validation["new_scenarios_generated"] != 0):
        raise ValueError("Wrong increment scope or unaccepted automated evidence")
    source = json.loads((evidence / "input_reference.json").read_text())
    case, tasks, scene, plans, _ = load_case_from_i01(source["i01_acceptance"], source["i01_acceptance_sha256"])
    if (source["task_subset_used"] or source["H"] != len(tasks.task_ids) or source["H"] != 1008
            or source["scenario_sha256"] != scene.scenario_sha256):
        raise ValueError("The submitted full-H case identity changed")
    meta = json.loads((evidence / "template_metadata.json").read_text())
    matrices = {name: sparse.load_npz(evidence / (name + ".npz")) for name in
                ("A_ub", "A_eq", "Bx_ub", "Br_ub", "Bx_eq", "Br_eq")}
    with np.load(evidence / "template_arrays.npz", allow_pickle=False) as bank:
        coefficients = {key: bank[key] for key in bank.files}
    reports = []
    for name, plan in plans.items():
        root = evidence / name
        with np.load(root / "plan.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["x"], plan.x) or not np.array_equal(saved["r"], plan.r):
                raise ValueError("Saved first-stage plan changed from I01")
        rhs_ub = coefficients["b_ub"] + matrices["Bx_ub"] @ plan.x + matrices["Br_ub"] @ plan.r.ravel()
        rhs_eq = coefficients["b_eq"] + matrices["Bx_eq"] @ plan.x + matrices["Br_eq"] @ plan.r.ravel()
        for label in ("fallback", "solution"):
            with np.load(root / (label + ".npz"), allow_pickle=False) as data:
                solution = RecourseSolution(**{key: data[key] for key in SOLUTION_FIELDS},
                    task_sha256=str(data["task_sha256"]), config_hash=str(data["config_hash"]),
                    scenario_sha256=str(data["scenario_sha256"]))
            check = check_recourse(case, tasks, scene, plan, solution, check_objective_bound=True)
            if not check["passed"]:
                raise ValueError(f"Independent original-unit check failed: {name}/{label}")
            z = np.empty_like(coefficients["c"])
            for key, (start, end) in meta["variable_slices"].items():
                z[start:end] = getattr(solution, key).ravel()
            lower, upper = coefficients["lower_bounds"], coefficients["upper_bounds"]
            matrix_violation = float(max(np.max(matrices["A_ub"] @ z - rhs_ub, initial=0.),
                np.max(np.abs(matrices["A_eq"] @ z - rhs_eq), initial=0.),
                np.max(lower-z, initial=0.), np.max(z-upper, initial=0.)))
            if matrix_violation > 1e-7:
                raise ValueError("Saved matrix primal residual exceeds certificate threshold")
            objective = float(coefficients["c"] @ z)
            if abs(objective - check["costs"]["Qop"]) > 1e-4:
                raise ValueError("Independent recourse cost differs from saved objective")
            row = {"plan": name, "solution": label, "physical_passed": True, "costs": check["costs"],
                   "matrix_max_violation": matrix_violation,
                   "max_normalized_violation": check["max_normalized_violation"]}
            if label == "solution":
                with np.load(root / "dual_marginals.npz", allow_pickle=False) as duals:
                    y, lam, lo, hi = (duals[key] for key in
                        ("dual_inequality_marginals", "dual_equality_marginals", "dual_lower_marginals", "dual_upper_marginals"))
                if not all(np.all(np.isfinite(a)) for a in (y, lam, lo, hi)):
                    raise ValueError("Nonfinite dual witness")
                stationarity = coefficients["c"] - matrices["A_ub"].T @ y - matrices["A_eq"].T @ lam - lo - hi
                sign_violation = float(max(np.max(y, initial=0.), np.max(-lo, initial=0.), np.max(hi, initial=0.)))
                finite_lo, finite_hi = np.isfinite(lower), np.isfinite(upper)
                infinite_violation = float(max(np.max(np.abs(lo[~finite_lo]), initial=0.), np.max(np.abs(hi[~finite_hi]), initial=0.)))
                dual = float(rhs_ub @ y + rhs_eq @ lam + lower[finite_lo] @ lo[finite_lo] + upper[finite_hi] @ hi[finite_hi])
                gap = abs(objective - dual)
                tolerance = 1e-4 + 1e-7 * max(abs(objective), abs(dual), 1.)
                error = float(np.max(np.abs(stationarity), initial=0.))
                if max(error, sign_violation, infinite_violation) > 1e-8 or gap > tolerance:
                    raise ValueError("Independent numeric dual certificate failed")
                row.update(primal_objective_usd=objective, dual_objective_usd=dual, gap_usd=gap,
                    dual_stationarity_violation=error, dual_sign_violation=sign_violation,
                    invalid_infinite_bound_marginal=infinite_violation, certificate_status="PASS_NUMERIC_LP_CERTIFICATE")
            reports.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS_INDEPENDENT_RECOURSE_AUDIT", "increment": "S02-I02", "full_S02_pass": False,
        "scope": "Four saved complete-H solutions/fallbacks, two independent numeric dual replays; no new solve or scenario draw",
        "tests": int(count[1]), "H": len(tasks.task_ids), "scenario_id": scene.scenario_id, "reports": reports,
        "artifact_manifest_sha256": sha256(manifest_path), "audit_source_sha256": sha256(__file__),
        "remaining_gate": "Independent stage coverage review including original 13 Q1 gates before complete S02 acceptance"}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
