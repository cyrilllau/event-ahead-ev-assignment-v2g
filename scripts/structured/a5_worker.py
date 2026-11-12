"""Failure closeout: diagnose frozen load-scaling gate, without new candidates."""

from pathlib import Path
import json
import pickle
import sys
import time

import numpy as np
import gurobipy as gp

from ev_assignment.domain.planning import Control, SolverOptions
from ev_assignment.l08 import matrix
from ev_assignment.l08.builders import parameters
from ev_assignment_experiments.datasets.structured import _zero_ev, _jsonable
from ev_assignment_experiments.analysis.diagnostic import save, reference


def constructive(net, scene):
    off, load = scene["line_operating_outages"], scene["load_p_kw"]
    fr, to = np.asarray(net["from"]) - 1, np.asarray(net["to"]) - 1
    connected = np.zeros(load.shape, bool)
    connected[:, 0] = True
    for j, (u, v) in enumerate(zip(fr, to)):
        connected[:, v] = connected[:, u] & ~off[:, j]
    work = np.where(connected, load, 0.)
    flow = np.zeros(off.shape)
    for j in reversed(range(len(fr))):
        flow[:, j] = np.where(off[:, j], 0., work[:, to[j]])
        work[:, fr[j]] += flow[:, j]
    reports = []
    for key, ratio, cap in [("active_power_kw", 1., net["p_limit"]),
                            ("reactive_power_kvar", .33, net["q_limit"])]:
        violations = abs(ratio * flow) - np.asarray(cap)
        for flat in np.argsort(violations.ravel())[-3:][::-1]:
            t, j = np.unravel_index(flat, violations.shape)
            if violations[t, j] > 1e-9:
                reports.append(dict(kind=key, line=int(j + 1), from_bus=int(fr[j] + 1),
                    to_bus=int(to[j] + 1), period=int(t), time_h=float(t / 4),
                    required=float(abs(ratio * flow[t, j])), limit=float(cap[j]),
                    violation=float(violations[t, j])))
    return connected, reports


def main():
    out, seconds, prior = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    control, begun = Control(seconds), time.monotonic()
    result = dict(stage="A5", status="STOP_REPORT_EVIDENCE_COMPLETE", passed=True,
                  search_status="NO_COMPUTABLE_FAMILY_FOUND_WITHIN_BUDGET",
                  stop_gate="A3_frozen_constructive_zero_EV_supply_gate",
                  new_physical_revisions=0, new_candidate_ids=0, validation_opened=False,
                  full400_solved=False, full400_LB=None, full400_UB=None, full400_gap=None,
                  zero_EV_LPs=[], all_shedding_audits=[], solver_calls=0)
    seen = set()
    options = SolverOptions(seconds=seconds, threads=1, seed=101, absolute_gap=1e-7)
    with gp.Env(params={"OutputFlag": 0}) as env:
        for cid in ["C11", "C12"]:
            case_path = prior / cid / "CASE.pkl"
            with case_path.open("rb") as f:
                case = pickle.load(f)
            scenes = [case["forecast"], *case["scenarios"]]
            residuals = [_zero_ev(case["network"], s["physical"])["max_residual"] for s in scenes]
            selected = {0, int(np.argmax(residuals))}
            for si, scene in enumerate(scenes):
                control.check()
                net, physical = case["network"], scene["physical"]
                m = matrix.assemble(net, [], physical, [], E=0, global_M=[])
                # Universal conservative load-shedding witness. This checks
                # physical FEASIBILITY, not healthy-grid full-supply suitability.
                y = np.zeros(len(m.cost))
                y[m.index["u"]] = net["u_ref"]
                y[m.index["shed"]] = physical["load_p_kw"]
                residual = m.residual(y, np.zeros(0))
                if residual > 1e-9:
                    raise ValueError("Original all-shedding witness infeasible")
                result["all_shedding_audits"].append(dict(candidate=cid, scenario=si,
                    scene_identity=scene["identity"], residual=residual, case=reference(case_path)))
                if si not in selected or scene["physical_sha256"] in seen:
                    continue
                seen.add(scene["physical_sha256"])
                connected, violations = constructive(net, physical)
                with gp.Model("original_zero_EV_closeout", env=env) as model:
                    yy = model.addMVar(len(m.cost), lb=m.lo, ub=m.hi)
                    model.addMConstr(m.A, yy, "<", m.b)
                    model.addMConstr(m.H, yy, "=", m.d)
                    model.setObjective(m.cost @ yy)
                    parameters(model, options, min(60., control.remaining))
                    result["solver_calls"] += 1
                    model.optimize()
                    row = dict(candidate=cid, scenario=si, scene_identity=scene["identity"],
                        physical_sha256=scene["physical_sha256"], native_status=int(model.Status),
                        scope="fixed_zero_EV_original_LP_NOT_global_EV_optimization",
                        constructive_full_supply_residual=residuals[si], violations=violations)
                    if model.Status == gp.GRB.OPTIMAL:
                        primal = np.asarray(yy.X)
                        audit = m.residual(primal, np.zeros(0))
                        if audit > 1e-9:
                            raise ValueError("Original zero-EV native primal failed audit")
                        shed = primal[m.index["shed"]]
                        healthy = ~physical["line_operating_outages"].any(axis=1)
                        row.update(objective=float(model.ObjVal), original_row_residual=audit,
                            total_ENS_kwh=float(.25 * shed.sum()),
                            disconnected_ENS_kwh=float(.25 * shed[~connected].sum()),
                            connected_ENS_kwh=float(.25 * shed[connected].sum()),
                            all_lines_healthy_ENS_kwh=float(.25 * shed[healthy].sum()),
                            all_lines_healthy_periods=int(healthy.sum()),
                            original_LP_feasible=True, native_runtime=float(model.Runtime))
                        np.savez_compressed(out / f"{cid}_scene{si:03d}_zero_EV.npz", y=primal,
                                             raw_pi=np.asarray(model.getAttr("Pi")))
                    result["zero_EV_LPs"].append(row)
    result["elapsed_seconds"] = time.monotonic() - begun
    result["interpretation"] = ("Frozen connected-full-supply witness failed; 130 original all-shedding "
        "witnesses establish feasibility for both 64-support cases plus forecasts. "
        "This is not original-model infeasibility or solver nonconvergence. No gate, physical "
        "data or acceptance tolerance was changed after failure; A4 and validation remain closed.")
    save(out / "RESULT.json", _jsonable(result))
    print(json.dumps({k: result[k] for k in ["stage", "status", "solver_calls", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
