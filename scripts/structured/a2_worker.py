"""Remote, bounded finite candidate screening; no validation/OOS consumption."""

from pathlib import Path
import json
import pickle
import sys
import time
import traceback

import numpy as np
from scipy import sparse

from ev_assignment.domain.planning import Control, SolverOptions
from ev_assignment.l08 import catalogue, recourse, risk
from ev_assignment.l08.certificates import enclosure
from ev_assignment.methods.catalogue_probability import calibrated
from ev_assignment.methods.inverse_design import solve_inverse_cost_lp
from ev_assignment_experiments.datasets.structured import make_case, make_reference_policies, _jsonable
from ev_assignment_experiments.analysis.diagnostic import read, reference, save


def jsave(path, value):
    save(path, _jsonable(value))


def evaluate(case, out, control, env):
    policies = make_reference_policies(case)
    first = catalogue.build_first(case["packages"], case["type_counts"],
                                   len(case["network"]["station_buses"]))
    ids = [s["identity"] for s in case["scenarios"]]
    domains, calibration = {}, {}
    for method in ["Full-DRO", "Marginal-DRO"]:
        domains[method], calibration[method] = calibrated(ids, case["features"].T,
                case["calibration_features"].T, control, marginal=method == "Marginal-DRO",
                lower=case["calibration_lower"], upper=case["calibration_upper"],
                calibration_rule=case["calibration_rule"])
    jsave(out / "CALIBRATION.json", calibration)
    qmax = {}
    for method, ambiguity in domains.items():
        h = case["features"][:, 2]
        rr = risk.solve(ambiguity, h, h, np.ones(len(h)), control)
        qmax[method] = {k: v for k, v in rr.items() if k not in {"plus", "minus", "plane"}}
    jsave(out / "JOINT_EVENT_BOUNDS.json", qmax)
    options = SolverOptions(seconds=max(.01, control.remaining), threads=1, seed=101,
                            absolute_gap=1e-7, soft_memory_gb=20)
    templates, results = {}, []
    # Scene order shared by all three policies; every saved point binds its
    # explicit complete catalogue and scenario identity. No old cut/Q cache.
    scenes = [case["forecast"], *case["scenarios"]]
    qrows = [[] for _ in policies]
    for sid, scene in enumerate(scenes):
        control.check()
        template = catalogue.build_recourse(case["network"], case["packages"],
                                             scene["physical"], scene["route_states"], first)
        templates[sid] = template
        for j, policy in enumerate(policies):
            control.check()
            answer = recourse.solve(template, policy["x"], options, control, env=env)
            q = dict(scenario_id=scene["identity"], qL=answer["qL"], qU=answer["qU_audited"],
                     qbar=template.qbar, residual=answer["residual"], native_status=answer["native_status"],
                     original_physics="unchanged grid+energy equations, A1-verified count representation",
                     scope="canonical_fixed_policy_only_not_global_optimization")
            qrows[j].append(q)
            np.savez_compressed(out / f"Q_{sid:02d}_{j}.npz", x=policy["x"], y=answer["y"],
                                 raw_pi=answer["raw_pi"], beta=answer["beta"], alpha=answer["alpha"])
            jsave(out / f"Q_{sid:02d}_{j}.json", q)
        print(json.dumps(dict(event="scenario_complete", candidate=out.name,
                              completed=sid + 1, total=len(scenes), seconds_left=control.remaining)), flush=True)
    intervals = {m: [] for m in ["DET", "SAA", "Full-DRO", "Marginal-DRO", "ARO"]}
    for j, policy in enumerate(policies):
        ql = np.array([r["qL"] for r in qrows[j][1:]])
        qu = np.array([r["qU"] for r in qrows[j][1:]])
        qbars = np.array([r["qbar"] for r in qrows[j][1:]])
        intervals["DET"].append([qrows[j][0]["qL"], qrows[j][0]["qU"]])
        intervals["ARO"].append([float(ql.max()), float(qu.max())])
        # SAA uses the generator's declared STRATIFIED weights, never 1/N.
        lo = enclosure(sparse.csr_matrix(case["weights"][None, :]), ql)[0][0]
        hi = enclosure(sparse.csr_matrix(case["weights"][None, :]), qu)[1][0]
        intervals["SAA"].append([float(np.nextafter(float(lo), -np.inf)), float(np.nextafter(float(hi), np.inf))])
        for method, ambiguity in domains.items():
            rr = risk.solve(ambiguity, ql, qu, qbars, control)
            intervals[method].append([rr["risk_L"], rr["risk_U"]])
            jsave(out / f"RISK_{method}_{j}.json", rr)
        results.append(dict(policy=policy, Q=qrows[j]))
    features = np.asarray([p["cost_parts"] for p in policies])
    region = solve_inverse_cost_lp(features, intervals, control)
    return dict(policies=results, risk_intervals=intervals, inverse=region,
                joint_probability_intervals={m: [r["risk_L"], r["risk_U"]] for m, r in qmax.items()},
                first_stage_size=dict(count_integer=len(case["packages"]),
                    count_continuous=len(case["packages"]), individual_binary=sum(case["type_counts"]) * 12,
                    rows=first["A"].shape[0], nnz=first["A"].nnz),
                recourse_size=dict(vars=len(templates[1].names), rows=templates[1].W.shape[0],
                                   W_nnz=templates[1].W.nnz, T_nnz=templates[1].T.nnz),
                complete_support=len(ids), full400=False, solver_calls=3 * len(scenes))


def main():
    import gurobipy as gp
    out, seconds, inputs = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    root = out.parents[1]
    control = Control(seconds)
    candidates = read(root / "PREREGISTERED_CANDIDATES.json")["candidates"]
    result = dict(stage="A2", status="RUNNING", passed=False, candidates=[], finalists=[],
                  development_only=True, scenario_count=8, fleet_size=6, types=6,
                  validation_opened=False, revisions_used=0, inverse_margin_threshold_dollars=1.0)
    with gp.Env(params={"OutputFlag": 0}) as env:
        for index, candidate in enumerate(candidates):
            if control.remaining < 8:
                result["candidates"].extend(dict(id=c["id"], status="NOT_TESTED_BUDGET") for c in candidates[index:])
                break
            path = out / candidate["id"]
            path.mkdir()
            started = time.monotonic()
            allowance = max(1., (control.remaining - 3) / (len(candidates) - index))
            cc = Control(allowance)
            row = dict(id=candidate["id"], parameters=candidate, passed=False, status="RUNNING",
                       budget_allocated_seconds=allowance, candidate_participates_in_selection=True,
                       seed=101, support_count=8, full400=False)
            try:
                physical_config = {k: candidate[k] for k in ("gap_h", "early_duration_h", "joint_nominal")}
                case = make_case(physical_config, inputs, "development", 101, scenario_count=8, fleet_size=6, groups=6)
                # Full raw case is an external replay artifact. Only locally
                # generated trusted pickles are read by subsequent stage workers.
                with (path / "CASE.pkl").open("xb") as f:
                    pickle.dump(case, f, protocol=5)
                row.update(science_identity=case["identity"], physical_audit=case["physical_audit"],
                           case_file=reference(path / "CASE.pkl"), feature_scope=case["feature_scope"],
                           nominal=dict(vehicles=6, types=6, periods=96, scenarios=8),
                           effective=dict(packages=len(case["packages"]),
                                          unique_physical=case["physical_audit"]["independent_physical_trajectories"],
                                          proved_operational_classes=None))
                if not case["physical_audit"]["passed"]:
                    row["status"] = "PHYSICAL_GATE_FAILED"
                else:
                    evidence = evaluate(case, path, cc, env)
                    row.update(evidence)
                    row.update(passed=evidence["inverse"]["passed"], status=evidence["inverse"]["status"])
            except Exception as exc:
                row.update(status="BUDGET_OR_IMPLEMENTATION_INCOMPLETE" if isinstance(exc, (TimeoutError, RuntimeError))
                           else "CANDIDATE_ERROR", error=f"{type(exc).__name__}: {exc}")
                (path / "ERROR.txt").write_text(traceback.format_exc())
            row["elapsed_seconds"] = time.monotonic() - started
            jsave(path / "RESULT.json", row)
            result["candidates"].append(row)
            event = dict(stage="A2", candidate=candidate["id"], status=row["status"],
                         elapsed_seconds=row["elapsed_seconds"], evidence=reference(path / "RESULT.json"))
            with (root / "CASE_SEARCH_LEDGER.jsonl").open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)
    passed = [c for c in result["candidates"] if c.get("passed")]
    passed.sort(key=lambda c: (-c["inverse"]["max_margin"], c["inverse"]["relative_l1"], c["id"]))
    result["finalists"] = [c["id"] for c in passed[:2]]
    result.update(passed=bool(passed), status="REFERENCE_MECHANISM_CANDIDATES_FOUND" if passed
                  else "NO_CANDIDATE_PASSED_A2", elapsed_seconds=seconds - control.remaining)
    jsave(out / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
