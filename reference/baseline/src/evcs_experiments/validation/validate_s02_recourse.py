"""S02-I02: Q1 recourse on the unchanged complete-H Michael development case.

Writes automated evidence, never scientific acceptance or a formal policy.
"""
from __future__ import annotations

import argparse
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
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks, load_scenario_physical_inputs
from evcs_experiments.recourse import assemble_recourse, full_shedding_fallback, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.validate_s02_first_stage import checked_json, write_json, write_mapping

SOLUTION_FIELDS = ("rav", "p", "q", "short", "shed", "line_p", "line_q", "u", "soc_dep")
DUAL_FIELDS = ("dual_inequality_marginals", "dual_equality_marginals", "dual_lower_marginals", "dual_upper_marginals")


def load_accepted_i01(path, expected_sha256):
    card = checked_json(path, expected_sha256)
    if card["status"] != "PASS_INCREMENT" or card["increment"] != "S02-I01" or card["full_S02_pass"]:
        raise ValueError("This increment requires accepted S02-I01")
    evidence = historical_path(card["evidence"])
    manifest = checked_json(evidence / "artifact_sha256.json", card["artifact_manifest_sha256"])
    for name, expected in manifest.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(evidence / name) != expected:
            raise ValueError("Accepted I01 artifact changed")
    sources = checked_json(card["code_manifest"]["path"], card["code_manifest"]["sha256"])
    # Historical I01 source identity is checked at its retained release. New
    # recourse code has a distinct manifest and does not relabel old evidence.
    source_root = Path(card["source_root"])
    for name, expected in sources["current_source_hashes"].items():
        if sha256(source_root / name) != expected:
            raise ValueError(f"Historical I01 source changed: {name}")
    return card, evidence, checked_json(evidence / "increment_validation.json", card["validation_sha256"])


def load_case_from_i01(card_path, expected_sha256, case_root=None):
    card, evidence, accepted = load_accepted_i01(card_path, expected_sha256)
    record = next(row for row in accepted["targets"] if row["target"] == "michael_2018")
    case = load_physical_case(case_root)
    ref = record["task_source"]
    tasks = load_frozen_tasks(historical_path(ref["path"]), case, ref["sha256"])
    if tasks.task_sha256 != record["task_sha256"] or case.case_sha256 != accepted["case_sha256"]:
        raise ValueError("Current physical inputs do not match accepted I01")
    scene = load_scenario_physical_inputs(historical_path(record["scenario_source"]["path"]),
        historical_path(record["scenario_ledger"]["path"]), 0, case, tasks,
        expected_npz_sha256=record["scenario_source"]["sha256"],
        expected_ledger_sha256=record["scenario_ledger"]["sha256"],
        expected_scenario_sha256=record["scenario_sha256"])
    if scene.scenario_id != record["scenario_id"] or scene.target_event != "michael_2018":
        raise ValueError("Accepted scenario selection changed")
    plans = {}
    for name in ("zero", "minimum_fixture"):
        path = evidence / "michael_2018" / (name + ".plan.npz")
        with np.load(path, allow_pickle=False) as arrays:
            if tuple(arrays["task_ids"].tolist()) != tasks.task_ids:
                raise ValueError("Fixed plan task ordering changed")
            plans[name] = FirstStagePlan(arrays["x"], arrays["r"], str(arrays["task_sha256"]), str(arrays["config_hash"]))
    return case, tasks, scene, plans, record


def save_solution(path, solution):
    np.savez_compressed(path, **{name: getattr(solution, name) for name in SOLUTION_FIELDS},
        task_sha256=solution.task_sha256, config_hash=solution.config_hash, scenario_sha256=solution.scenario_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=get_paths().data_root)
    parser.add_argument("--i01-acceptance", type=Path, required=True)
    parser.add_argument("--i01-acceptance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lp-time-limit", type=float, default=180.)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = get_paths()
    started = time.monotonic()
    config = Path(os.environ["EVCS_CONFIG"]).resolve() if "EVCS_CONFIG" in os.environ else None
    write_json(out / "execution.json", {"argv": sys.argv, "python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(), "paths": paths.as_dict(),
        "config_path": str(config) if config else None, "config_sha256": sha256(config) if config else None,
        "runtime_policy_sha256": sha256(paths.spec_root / "runtime_policy.json"),
        "scope": "development; no scenario draws; full accepted H; two inherited fixed first-stage fixtures"})
    write_json(out / "progress.json", {"phase": "tests", "status": "RUNNING"})
    with (out / "tests.stdout.log").open("w") as stdout, (out / "tests.stderr.log").open("w") as stderr:
        run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(paths.project_root / "tests"),
                              "-p", "test_*.py", "-v"], cwd=paths.project_root, stdout=stdout, stderr=stderr)
    if run.returncode:
        raise ValueError("Tests failed; original test logs retained")
    case, tasks, scene, plans, reference = load_case_from_i01(args.i01_acceptance, args.i01_acceptance_sha256, args.case_root)
    write_json(out / "input_reference.json", {"i01_acceptance": str(args.i01_acceptance),
        "i01_acceptance_sha256": args.i01_acceptance_sha256, "case_sha256": case.case_sha256,
        "target": scene.target_event, "scenario_id": scene.scenario_id, "scenario_sha256": scene.scenario_sha256,
        "task_sha256": tasks.task_sha256, "config_hash": tasks.config_hash, "H": len(tasks.task_ids),
        "bus_count": len(case.bus_ids), "periods": len(case.periods), "accepted_record": reference,
        "task_subset_used": False, "formal_optimization": False, "formal_OOS": False})
    write_json(out / "progress.json", {"phase": "assemble_full_H", "status": "RUNNING", "tasks": len(tasks.task_ids)})
    build_started = time.monotonic()
    template = assemble_recourse(case, tasks, scene)
    build_seconds = time.monotonic() - build_started
    matrices = ("A_ub", "Bx_ub", "Br_ub", "A_eq", "Bx_eq", "Br_eq")
    for name in matrices:
        sparse.save_npz(out / (name + ".npz"), getattr(template, name))
    np.savez_compressed(out / "template_arrays.npz", **{name: getattr(template, name) for name in
        ("b_ub", "b_eq", "c", "lower_bounds", "upper_bounds", "component_objectives")})
    write_mapping(out / "variable_map.csv", template.variable_map)
    write_mapping(out / "constraint_map_ub.csv", template.constraint_map_ub)
    write_mapping(out / "constraint_map_eq.csv", template.constraint_map_eq)
    metadata = {"template_sha256": template.template_sha256, "task_ids": list(tasks.task_ids),
        "scenario_sha256": scene.scenario_sha256, "task_sha256": tasks.task_sha256, "config_hash": tasks.config_hash,
        "variable_shapes": {name: list(shape) for name, shape in template.variable_shapes.items()},
        "variable_slices": {name: list(value) for name, value in template.variable_slices.items()},
        "matrix_shapes": {name: list(getattr(template, name).shape) for name in matrices},
        "reference_bus_ids_by_period": [list(row) for row in template.reference_bus_ids_by_period],
        "cmax_usd": template.cmax_usd, "build_seconds": build_seconds,
        "rhs_form": "b + Bx @ x + Br @ r.ravel(order='C')", "units": dict(template.units)}
    write_json(out / "template_metadata.json", metadata)
    checks, objectives, solves = {}, [], {}
    for name, plan in plans.items():
        write_json(out / "progress.json", {"phase": "fixed_plan_LP", "plan": name, "status": "RUNNING"})
        folder = out / name
        folder.mkdir()
        np.savez_compressed(folder / "plan.npz", x=plan.x, r=plan.r, task_sha256=plan.task_sha256, config_hash=plan.config_hash)
        fallback = full_shedding_fallback(template, plan)
        save_solution(folder / "fallback.npz", fallback)
        fallback_check = check_recourse(case, tasks, scene, plan, fallback, check_objective_bound=True)
        write_json(folder / "fallback_check.json", fallback_check)
        if not fallback_check["passed"]:
            raise ValueError("Full-shedding fallback violates an independent physical check")
        solved = solve_fixed_plan(template, plan, time_limit_seconds=args.lp_time_limit)
        certificate = dict(solved.certificate)
        duals = {key: certificate.pop(key) for key in DUAL_FIELDS if key in certificate}
        if duals:
            np.savez_compressed(folder / "dual_marginals.npz", **duals)
            certificate["dual_marginals_file"] = "dual_marginals.npz"
            certificate["dual_marginals_sha256"] = sha256(folder / "dual_marginals.npz")
        result = {"status": solved.status, "optimal": solved.optimal, "solver_status": solved.solver_status,
            "message": solved.message, "objective_usd": solved.objective_usd,
            "objective_components": dict(solved.objective_components), "certificate": certificate,
            "wall_seconds": solved.wall_seconds}
        write_json(folder / "solver_result.json", result)
        solves[name] = result
        if solved.solution is not None:
            save_solution(folder / "solution.npz", solved.solution)
            checked = check_recourse(case, tasks, scene, plan, solved.solution, check_objective_bound=solved.optimal)
            write_json(folder / "physical_check.json", checked)
            checks[name] = checked
        else:
            raise ValueError("Recourse LP returned no solution; solver result preserved")
        if not solved.optimal or not certificate.get("optimal_certificate_valid") or not checked["passed"]:
            raise ValueError("Fixed-plan LP requires numeric certificate and independent physical feasibility")
        for component in ("Fshed", "Fshort", "FV2G", "Fdeg", "Qop"):
            if abs(checked["costs"][component] - solved.objective_components[component]) > 1e-4:
                raise ValueError("Independent recourse objective component differs")
        objectives.append({"plan": name, "case": "optimal", **checked["costs"],
                           "cmax_usd": checked["cmax_usd"], "fallback_cost_usd": checked["fallback_cost_usd"]})
        objectives.append({"plan": name, "case": "full_shedding_fallback", **fallback_check["costs"],
                           "cmax_usd": fallback_check["cmax_usd"], "fallback_cost_usd": fallback_check["fallback_cost_usd"]})
    write_mapping(out / "objective_components.csv", objectives)
    write_mapping(out / "variable_constraint_map.csv", [{"variables": "variable_map.csv",
        "inequalities": "constraint_map_ub.csv", "equalities": "constraint_map_eq.csv",
        "template_sha256": template.template_sha256}])
    write_json(out / "physical_check_report.json", checks)
    write_json(out / "model_code_manifest.json", source_provenance())
    write_json(out / "increment_validation.json", {"increment": "S02-I02", "status": "PASS_INCREMENT_AUTOMATED",
        "scientific_acceptance_pending": True, "full_S02_pass": False,
        "scope": "Q1 parameterized recourse LP, complete-H physical/certificate development checks",
        "H": len(tasks.task_ids), "bus_count": len(case.bus_ids), "period_count": len(case.periods),
        "fixed_plans": list(plans), "scenario_id": scene.scenario_id, "scenario_sha256": scene.scenario_sha256,
        "template_sha256": template.template_sha256, "build_seconds": build_seconds,
        "total_wall_seconds": time.monotonic() - started, "new_scenarios_generated": 0,
        "production_experiments_executed": False, "formal_OOS_outcomes_generated": False,
        "deferred": "Independent saved-solution/dual audit, historical 13-gate coverage review and full S02 acceptance"})
    write_json(out / "progress.json", {"phase": "complete", "status": "AWAITING_INDEPENDENT_REVIEW"})
    write_json(out / "artifact_sha256.json", {str(path.relative_to(out)): sha256(path)
        for path in sorted(out.rglob("*")) if path.is_file()})
    print(json.dumps({"status": "PASS_INCREMENT_AUTOMATED", "increment": "S02-I02", "H": len(tasks.task_ids),
                      "output": str(out), "scientific_acceptance_pending": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
