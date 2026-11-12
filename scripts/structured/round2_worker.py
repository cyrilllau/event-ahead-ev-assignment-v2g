"""Bounded Round 2 calls into the existing catalogue solver; external IO only."""
from pathlib import Path
import json
import pickle
import sys
import time
import traceback
import hashlib

import numpy as np
from ev_assignment.domain.planning import Control
from ev_assignment.methods.catalogue import solve_structured
from ev_assignment.methods.catalogue_probability import calibrated
from ev_assignment_experiments.datasets.structured import _jsonable
from ev_assignment_experiments.analysis.diagnostic import read, reference, save


def dump(path, value, replace=False):
    if replace:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, indent=2))
        tmp.replace(path)
    else:
        save(path, _jsonable(value))


def healthy_fallback(case, control):
    """Resolve failed sufficient witnesses using the original zero-shed LP."""
    audit = case.get("healthy_grid_audit")
    if audit is None or audit["passed"]:
        return
    import gurobipy as gp
    from ev_assignment_experiments.datasets.structured_round2 import healthy_zero_ev_system
    from ev_assignment.l08.certificates import row_audit
    checks = audit["scenarios"]
    with gp.Env(params={"OutputFlag": 0}) as env:
        for scene, check in zip([case["forecast"], *case["scenarios"]], checks):
            if check["passed"]:
                continue
            control.check()
            mat = healthy_zero_ev_system(case["network"], scene["physical"])["matrix"]
            with gp.Model(env=env) as model:
                model.Params.Threads = 1
                model.Params.TimeLimit = min(60., control.remaining)
                model.Params.FeasibilityTol = 1e-9
                model.Params.OptimalityTol = 1e-9
                v = model.addMVar(len(mat.cost), lb=mat.lo, ub=mat.hi)
                model.addMConstr(mat.A, v, "<", mat.b)
                model.addMConstr(mat.H, v, "=", mat.d)
                model.optimize()
                result = dict(native_status=int(model.Status), runtime=float(model.Runtime),
                              passed=False, original_zero_shed_lp=True)
                if model.SolCount:
                    y = np.asarray(v.X)
                    a = row_audit(mat.A, mat.b, np.full(len(mat.b), "<"), mat.lo, mat.hi, y)
                    e = row_audit(mat.H, mat.d, np.full(len(mat.d), "="), mat.lo, mat.hi, y)
                    result.update(residual=max(*a.values(), *e.values()), witness=y.tolist())
                    result["passed"] = result["residual"] <= 1e-9
                check["feasibility_lp"] = result
                check["passed"] = result["passed"]
                check["status"] = ("PASS_BY_ORIGINAL_FEASIBILITY_LP" if result["passed"] else
                                   "HEALTHY_ZERO_SHED_INFEASIBLE" if model.Status == gp.GRB.INFEASIBLE else
                                   "HEALTHY_CHECK_UNRESOLVED")
    audit["passed"] = all(c["passed"] for c in checks)
    audit["status"] = "PASS" if audit["passed"] else "HEALTHY_CHECK_FAILED_OR_UNRESOLVED"
    case["physical_audit"]["passed"] = audit["passed"] and not case["physical_audit"]["errors"]
    case["physical_audit"]["status"] = "PASS" if case["physical_audit"]["passed"] else audit["status"]


def main():
    out, seconds, mode, data = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    control, begun = Control(seconds), time.monotonic()
    result = dict(status="RUNNING", mode=mode, moment_scope="P3-DRO", original_322311=False,
                  source=reference(__file__), passed=False)
    try:
        if mode in ["decompose", "direct"]:
            with data.open("rb") as f:
                case = pickle.load(f)
            result["original_frozen_case"] = reference(data)
            solver_seconds = min(240., control.remaining - 3)
        else:
            from ev_assignment_experiments.datasets.structured_round2 import make_fixed_load_case
            support = int(sys.argv[5])
            case = make_fixed_load_case(data, "development", 101, support, fleet_size=30, groups=6)
            solver_seconds = control.remaining - 5
        healthy_fallback(case, control)
        with (out / "CASE.pkl").open("xb") as f:
            pickle.dump(case, f, protocol=5)
        result.update(case_identity=case["identity"], case=reference(out / "CASE.pkl"),
                      N=sum(case["type_counts"]), G=len(case["type_counts"]),
                      packages=len(case["packages"]), support=len(case["scenarios"]),
                      physical_screen=case["physical_audit"],
                      probability_features=case["feature_scope"])
        if case.get("fixed_load_round2") and not case["physical_audit"]["passed"]:
            result.update(status="PHYSICAL_DESIGN_GATE_FAILED_OR_UNRESOLVED",
                          total_seconds=time.monotonic() - begun)
            dump(out / "RESULT.json", result)
            return
        ambiguity, cal = calibrated([s["identity"] for s in case["scenarios"]],
            case["features"].T, case["calibration_features"].T, control,
            lower=case["calibration_lower"], upper=case["calibration_upper"],
            calibration_rule=case["calibration_rule"])
        cal.update(name="P3-DRO", marginal_name="P2-DRO", not_original_322311=True)
        dump(out / "PROBABILITY_DOMAIN.json", cal)
        result["probability_identity"] = ambiguity.identity
        result["build_and_probability_seconds"] = time.monotonic() - begun
        def witness(kind, phase, iteration, sid, x, answer):
            dest = out / "witnesses"
            dest.mkdir(exist_ok=True)
            stem = f"{kind}_{phase}_{iteration:03d}_{sid:04d}"
            if kind == "risk":
                dump(dest / (stem + ".json"), dict(answer, x=x))
                return reference(dest / (stem + ".json"))
            if kind == "master_bound":
                A, a = answer["A"], answer["arrays"]
                np.savez_compressed(dest / (stem + ".npz"), A_data=A.data, A_indices=A.indices,
                    A_indptr=A.indptr, A_shape=A.shape,
                    raw_pi=answer["raw_pi"] if answer["raw_pi"] is not None else np.zeros(0), **a)
                dump(dest / (stem + ".json"), dict(names=answer["names"], certificate=answer["certificate"],
                                                  arrays=reference(dest / (stem + ".npz"))))
                return reference(dest / (stem + ".json"))
            arrays = {k: answer[k] for k in ["y", "raw_pi", "alpha", "beta"] if k in answer}
            np.savez_compressed(dest / (stem + ".npz"), x=x, **arrays)
            if "audit" in answer:
                dump(dest / (stem + ".json"), answer["audit"])
        cc = Control(max(.01, min(solver_seconds, control.remaining - 2)))
        if mode == "direct":
            from ev_assignment.methods.catalogue_direct import solve_direct
            solved = solve_direct(case, ambiguity, cc,
                event=lambda row: dump(out / "PROGRESS.json", row, True), witness=witness)
        else:
            solved = solve_structured(case, ambiguity, cc,
                event=lambda row: dump(out / "PROGRESS.json", row, True), witness=witness)
        result.update(optimization=solved, passed=solved.get("passed", False),
                      status=solved["status"])
        if solved.get("best") is not None:
            x = np.asarray(solved["best"])
            H = len(case["packages"])
            result["deployment"] = dict(selected_vehicles=float(x[:H].sum()),
                selected_fraction=float(x[:H].sum() / result["N"]),
                contract_kwh=float(x[H:].sum()),
                packages=[dict(case["packages"][h], count=float(v), total_contract=float(x[H+h]))
                          for h, v in enumerate(x[:H]) if v > 0],
                signature=hashlib.sha256(x.tobytes()).hexdigest())
        if mode in ["decompose", "direct"]:
            previous = read(data.parent / "RESULT.json")
            result["reference_policy_P3_upper_objectives"] = [
                p["policy"]["first_stage_cost"] + q[1] for p, q in
                zip(previous["policies"], previous["risk_intervals"]["Full-DRO"])]
            result["reference_comparison_scope"] = "same frozen N6/S8 case and weights; not inherited optimization bounds"
    except Exception as exc:
        result.update(status="IMPLEMENTATION_ERROR", error=f"{type(exc).__name__}: {exc}")
        (out / "ERROR.txt").write_text(traceback.format_exc())
    result["total_seconds"] = time.monotonic() - begun
    dump(out / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ["status", "passed", "total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
