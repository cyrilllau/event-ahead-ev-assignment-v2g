"""Charged complete-support fixed-policy evidence, without candidate selection."""
from pathlib import Path
import pickle
import json
import sys
import time
import traceback

import numpy as np
from scipy import sparse
from ev_assignment.domain.planning import Control, SolverOptions
from ev_assignment.l08 import catalogue as cat, recourse, risk
from ev_assignment.l08.certificates import enclosure
from ev_assignment.methods.catalogue_probability import calibrated
from ev_assignment_experiments.datasets.structured_round2 import make_reference_policies
from round2_worker import dump


def main():
    import gurobipy as gp
    out, seconds, source = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    cc, began = Control(seconds), time.monotonic()
    result = dict(status="INCOMPLETE", passed=False, purpose="fixed_policy_evidence_not_parameter_search")
    try:
        with source.open("rb") as f:
            case = pickle.load(f)
        result["case_identity"] = case["identity"]
        optimized = json.loads((source.parent / "RESULT.json").read_text())
        audits = optimized["optimization"].get("original_audit", [])
        if [a["sid"] for a in audits] != [s["identity"] for s in case["scenarios"]]:
            raise ValueError("Full400 acceptance does not bind the complete original audit support")
        result["optimized_service"] = {k: dict(
            nominal_weighted=float(case["weights"] @ np.asarray([a[k] for a in audits])),
            minimum=min(a[k] for a in audits), maximum=max(a[k] for a in audits))
            for k in ["discharging_vehicles", "discharge_AC_kwh", "critical_ENS_kwh", "total_ENS_kwh", "shortfall_kwh"]}
        result["optimized_original_max_residual"] = max(float(a["residual"]) for a in audits)
        policies = make_reference_policies(case)
        first = cat.build_first(case["packages"], case["type_counts"], len(case["network"]["station_buses"]))
        domains, metadata = {}, {}
        for m in ["P3-DRO", "P2-DRO"]:
            domains[m], metadata[m] = calibrated([s["identity"] for s in case["scenarios"]],
                case["features"].T, case["calibration_features"].T, cc, marginal=m == "P2-DRO",
                lower=case["calibration_lower"], upper=case["calibration_upper"],
                calibration_rule=case["calibration_rule"])
        result["domains"] = metadata
        rows = [[] for _ in policies]
        options = SolverOptions(seconds=60., threads=1, seed=101, soft_memory_gb=20)
        with gp.Env(params={"OutputFlag": 0}) as env:
            for sid, scene in enumerate([case["forecast"], *case["scenarios"]]):
                cc.check()
                template = cat.build_recourse(case["network"], case["packages"], scene["physical"], scene["route_states"], first)
                result["recourse_size"] = dict(variables=template.W.shape[1], rows=template.W.shape[0],
                                               W_nnz=template.W.nnz, T_nnz=template.T.nnz)
                for j, policy in enumerate(policies):
                    answer = recourse.solve(template, policy["x"], options, cc, env=env)
                    rows[j].append(dict(sid=scene["identity"], qL=answer["qL"], qU=answer["qU_audited"],
                                        qbar=template.qbar, residual=answer["residual"],
                                        global_cut_valid=answer["global_cut_valid"]))
                    np.savez_compressed(out / f"policy{j}_scene{sid:04d}.npz", x=policy["x"],
                        y=answer["y"], raw_pi=answer["raw_pi"], alpha=answer["alpha"], beta=answer["beta"])
                if sid % 8 == 0:
                    dump(out / "PROGRESS.json", dict(completed=sid + 1, total=1 + len(case["scenarios"]),
                                                     elapsed_seconds=time.monotonic() - began), True)
        table = []
        for j, policy in enumerate(policies):
            ql = np.array([r["qL"] for r in rows[j][1:]])
            qu = np.array([r["qU"] for r in rows[j][1:]])
            qb = np.array([r["qbar"] for r in rows[j][1:]])
            lo = enclosure(sparse.csr_matrix(case["weights"][None, :]), ql)[0][0]
            hi = enclosure(sparse.csr_matrix(case["weights"][None, :]), qu)[1][0]
            intervals = dict(DET=[rows[j][0]["qL"], rows[j][0]["qU"]], SAA=[float(lo), float(hi)],
                             ARO=[float(ql.max()), float(qu.max())])
            for m, domain in domains.items():
                rr = risk.solve(domain, ql, qu, qb, cc)
                intervals[m] = [rr["risk_L"], rr["risk_U"]]
                dump(out / f"risk_{m}_policy{j}.json", rr)
            table.append(dict(policy=policy, recourse_risk=intervals,
                total={m: [policy["first_stage_cost"] + v for v in r] for m, r in intervals.items()},
                Q=rows[j]))
        result.update(status="COMPLETE_FIXED_POLICY_TABLE", passed=True, policies=table,
                      first_size=dict(variables=len(first["cost"]), rows=first["A"].shape[0], nnz=first["A"].nnz),
                      full_support=len(case["scenarios"]),
                      original_expansion_scope="optimized policy separately audited on every scene; reference policies audited in exact count formulation")
    except Exception as exc:
        result.update(status="BUDGET_OR_ORACLE_INCOMPLETE" if isinstance(exc, (TimeoutError, RuntimeError, InterruptedError)) else
                      "IMPLEMENTATION_ERROR", error=f"{type(exc).__name__}: {exc}")
        (out / "ERROR.txt").write_text(traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - began
    dump(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
