"""Balanced, paired frozen-policy OOS; original matrices audited per common scene."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time
import traceback

import numpy as np
from ev_assignment.domain.planning import Control, SolverOptions
from ev_assignment.l08 import catalogue as cat, recourse
from ev_assignment_experiments.analysis.diagnostic import reference
from ev_assignment_experiments.datasets.structured import _digest
from round2_worker import dump


def tail_mean(values, weights, alpha=.95):
    order = np.argsort(values)[::-1]
    left, value = 1 - alpha, 0.
    for i in order:
        take = min(left, weights[i])
        value += take * values[i]
        left -= take
        if left <= 1e-15:
            break
    return float(value / (1 - alpha))


def main():
    import gurobipy as gp
    out, seconds, casepath, mapping = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    cc, started = Control(max(.01, seconds - 5)), time.monotonic()
    result = dict(status="OOS_INCOMPLETE", passed=False, sample=reference(casepath),
                  missing_methods=[], common_completed=0, expected=1000, records=[])
    try:
        specification = json.loads(mapping.read_text())
        manifest = json.loads(Path(specification["oos_manifest"]).read_text())
        train = json.loads(Path(specification["train_manifest"]).read_text())
        actual = reference(casepath)
        if actual != manifest["case"]:
            raise ValueError("OOS pickle differs from the pre-optimization frozen manifest")
        with casepath.open("rb") as f:
            case = pickle.load(f)
        identity, seed = case["identity"], specification["seed"]
        ids = [s["identity"] for s in case["scenarios"]]
        weights = np.asarray(case["weights"], float)
        if (identity["split"] != "oos" or len(ids) != 1000 or len(set(ids)) != 1000
                or identity["seed"] != seed or manifest["seed"] != seed or train["seed"] != seed
                or identity["sha256"] != manifest["identity"]["sha256"]
                or identity["sha256"] != _digest({k: v for k, v in identity.items() if k != "sha256"})
                or ids != manifest["ordered_ids"] or ids != identity["scenarios"]
                or weights.shape != (1000,) or not np.isfinite(weights).all() or np.any(weights < 0)
                or abs(weights.sum() - 1) > 1e-12 or not np.array_equal(weights, manifest["weights"])
                or not np.array_equal(weights, identity["weights"])
                or identity["catalogue"] != _digest(case["packages"])
                or not case["physical_audit"]["passed"] or not manifest["physical_audit"]["passed"]
                or train["split"] != "validation"):
            raise ValueError("OOS frozen seed/support/weights/physical identity mismatch")
        result.update(frozen_manifest=reference(specification["oos_manifest"]), seed=seed,
                      train_manifest=reference(specification["train_manifest"]))
        methods, unique = {}, {}
        sources = specification["methods"]
        expected_names = dict(DET="DET", SAA="SAA", ARO="ARO", P3="P3-DRO", P2="P2-DRO", **{"P3-noadvance": "P3-DRO/no-advance"})
        if set(sources) != set(expected_names):
            raise ValueError("OOS requires the frozen six-method mapping")
        for method, path in sources.items():
            if not Path(path).exists():
                result["missing_methods"].append(method)
                continue
            run = json.loads(Path(path).read_text())
            if (run.get("seed") != seed or run.get("method") != method
                    or run.get("base_case_identity", {}).get("sha256") != train["identity"]["sha256"]):
                raise ValueError("OOS policy comes from a different training seed, method or case")
            solved = run.get("optimization", {})
            if not solved.get("original_expansion_audited") or solved.get("best") is None:
                result["missing_methods"].append(method)
                continue
            x = np.asarray(solved["best"], float)
            if solved.get("representation") != "count":
                raise ValueError("OOS accepts frozen core count policies only")
            identity = solved["science_identity"]
            if (identity.get("catalogue") != case["identity"]["catalogue"]
                    or identity.get("source") != case["identity"]["source"]
                    or identity.get("type_counts") != case["type_counts"]
                    or solved.get("first_stage_weights") != [1, 1, 1, 1]
                    or identity.get("seed") != seed or solved.get("method") != expected_names[method]
                    or solved.get("first_stage_restriction") != ("latest_departure_only" if method == "P3-noadvance" else "all_packages")):
                raise ValueError("Frozen policy catalogue, available pool, physical source or fees mismatch")
            key = hashlib.sha256(x.tobytes()).hexdigest()
            methods[method] = dict(key=key, source=reference(path), certified_near_optimal=solved.get("passed", False))
            unique[key] = x
        if not methods:
            raise ValueError("No complete-support audited frozen policies available")
        result["methods"] = methods
        result["unique_policies"] = len(unique)
        result["case_identity"] = case["identity"]
        first = cat.build_first(case["packages"], case["type_counts"], len(case["network"]["station_buses"]))
        fi = cat.build_first(case["packages"], case["type_counts"], len(case["network"]["station_buses"]), individual=True)
        options = SolverOptions(seconds=60., threads=1, seed=101, soft_memory_gb=20)
        contracts = {}
        with gp.Env(params={"OutputFlag": 0}) as env:
            for sid, scene in enumerate(case["scenarios"]):
                cc.check()
                count = cat.build_recourse(case["network"], case["packages"], scene["physical"], scene["route_states"], first)
                indiv = cat.build_recourse(case["network"], case["packages"], scene["physical"], scene["route_states"], fi)
                adapter = cat.build_original_adapter(case["network"], case["packages"], scene["physical"], scene["route_states"], fi)
                original, answers = adapter["matrix"], {}
                for key, x in unique.items():
                    cc.check()
                    answer = recourse.solve(count, x, options, cc, env=env)
                    xi, yi = cat.expand_count(first, fi, x, answer["y"], count_recourse=count, individual_recourse=indiv)
                    if key in contracts and not np.array_equal(contracts[key], xi):
                        raise ValueError("OOS contract changed across scenes")
                    contracts[key] = xi.copy()
                    zo, yo = cat.to_original(adapter, fi, indiv, xi, yi)
                    component_delta = original.parts @ yo - count.data["parts"] @ answer["y"]
                    first_delta = adapter["first"]["cost"] @ zo - first["cost"] @ x
                    if max(np.max(abs(component_delta)), abs(first_delta)) > 1e-5:
                        raise ValueError("OOS original objective mismatch")
                    power, shed = yo[original.index["p"]], yo[original.index["shed"]]
                    record = dict(qL=answer["qL"], qU=answer["qU_audited"],
                        first_cost=float(first["cost"] @ x), residual=original.residual(yo, zo),
                        component_delta=component_delta.tolist(), first_delta=float(first_delta),
                        discharging_vehicles=int(np.count_nonzero(power.sum(axis=(1, 2)) > 1e-9)),
                        discharge_AC_kwh=float(.25 * power.sum()), total_ENS_kwh=float(.25 * shed.sum()),
                        critical_ENS_kwh=float(.25 * shed[:, np.asarray(case["network"]["shed_cost"]) == 50].sum()),
                        shortfall_kwh=float(yo[original.index["short"]].sum()))
                    np.savez_compressed(out / f"scene{sid:04d}_{key[:16]}.npz", x=x, y=answer["y"],
                                        raw_pi=answer["raw_pi"], alpha=answer["alpha"], beta=answer["beta"])
                    answers[key] = record
                # Commit a paired scene only after every available method passed.
                row = dict(index=sid, sid=scene["identity"], stratum=int(scene["stratum"]), weight=float(case["weights"][sid]),
                           policies=answers, methods={m: answers[v["key"]] for m, v in methods.items()})
                result["records"].append(row)
                with (out / "PAIRED_SCENES.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                result["common_completed"] = sid + 1
                dump(out / "PROGRESS.json", {k: v for k, v in result.items() if k != "records"}, True)
        result["passed"] = not result["missing_methods"]
        result["status"] = "OOS_COMPLETE_ALL_METHODS" if result["passed"] else "OOS_COMPLETE_AVAILABLE_METHODS_ONLY"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, (TimeoutError, RuntimeError, InterruptedError)):
            result["status"] = "IMPLEMENTATION_OR_IDENTITY_ERROR"
        (out / "ERROR.txt").write_text(traceback.format_exc())
    if result["records"]:
        weights = np.array([r["weight"] for r in result["records"]])
        result["completed_probability_mass"] = float(weights.sum())
        weights /= weights.sum()
        summaries = {}
        for method in result["methods"]:
            records = [r["methods"][method] for r in result["records"]]
            cost = np.array([r["first_cost"] + r["qU"] for r in records])
            lower_cost = np.array([r["first_cost"] + r["qL"] for r in records])
            summaries[method] = dict(mean_cost=float(weights @ cost), cvar95_cost=tail_mean(cost, weights),
                mean_cost_interval=[float(weights @ lower_cost), float(weights @ cost)],
                cvar95_cost_interval=[tail_mean(lower_cost, weights), tail_mean(cost, weights)],
                cost_evidence="lower cut values and numerically audited primal upper values, not exact-rational evidence",
                **{k: float(weights @ np.array([r[k] for r in records])) for k in
                   ["critical_ENS_kwh", "total_ENS_kwh", "shortfall_kwh", "discharge_AC_kwh", "discharging_vehicles"]})
        result["summaries"] = summaries
        result["inference_scope"] = ("complete frozen stratified distribution" if result["common_completed"] == 1000 else
                                      "DESCRIPTIVE ONLY: shared predetermined prefix, missing outcomes not zero-filled")
    result["elapsed_seconds"] = time.monotonic() - started
    dump(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
