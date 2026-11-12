"""Bounded full-coordinate recourse probe, using the original saved W/T LPs.

ROOT CONFIG OUTPUT STORE. CONFIG locks one point and at most eight scenario IDs.
Unsolved scenarios have no invented upper bound. No integer decoder is used.
"""

from pathlib import Path
import hashlib
import signal
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.certificates import row_audit, replay_cut
from ev_assignment_experiments.analysis.diagnostic import (
    affine_bounds,
    digest,
    npz,
    read,
    reference,
    save,
)
from ev_assignment_experiments.analysis.diagnostic_replay import AROOT, FIRST
from ev_assignment_experiments.runtime.ledger import BudgetLedger


def verified(ref):
    if digest(ref["path"]) != ref["sha256"]:
        raise ValueError("Fractional input identity changed")
    return Path(ref["path"])


def run(root, config, out, deadline):
    import gurobipy as gp

    start = time.monotonic()
    ids = config["scenario_ids"]
    if len(ids) > 8 or len(set(ids)) != len(ids) or any(s < 0 or s >= 400 for s in ids):
        raise ValueError("Initial fractional probe is limited to eight distinct scenarios")
    z = npz(verified(config["point_and_bounds"]))
    x = z["x"]
    point_sha = hashlib.sha256(x.tobytes()).hexdigest()
    if len(x) != 60780 or point_sha != config["point_sha256"]:
        raise ValueError("Full fractional point mapping mismatch")
    first = npz(root / FIRST / "FIRST_ARRAYS.npz")
    A = sp.load_npz(root / FIRST / "FIRST_A.npz")
    first_check = row_audit(A, first["rhs"], first["sense"], first["lo"], first["hi"], x)
    if max(first_check.values()) > 1e-9:
        raise ValueError(("Raw fractional point fails original first-stage audit", first_check))
    ql = np.maximum(0, z["archive_lower"].copy())
    qu = [None] * 400
    scope = []
    for sid in range(400):
        base = root / AROOT / "oracles" / f"{sid:04d}"
        a = npz(base / "ARRAYS.npz")
        if np.any(x < a["xlo"]) or np.any(x > a["xhi"]):
            raise ValueError(("Point outside full-domain cut finite box", sid))
        if np.min(a["c"]) < 0 or np.any(a["lo"][a["c"] > 0] < 0):
            raise ValueError("Nonnegative baseline lower bound not established")
        scope.append(reference(base / "ARRAYS.npz"))
    np.savez_compressed(out / "POINT.npz", x=x)
    save(
        out / "DOMAIN.json",
        dict(
            point_sha256=point_sha,
            first_stage_residual=first_check,
            all400_finite_boxes_checked=True,
            all400_nonnegative_cost_checked=True,
            arrays=scope,
            integer_rounding=False,
            integer_task_pruning=False,
            zero_scenarios_assumed=0,
            evidence_grade="NUMERICALLY_AUDITED_FIRST_STAGE",
        ),
    )
    records = []
    env = gp.Env(empty=True)
    env.setParam("LogFile", str(out / "ENVIRONMENT.log"))
    try:
        env.start()
    except gp.GurobiError as exc:
        save(
            out / "RESULT.json", dict(status="ENV_BLOCKED", solver_calls=0, missing_reason=str(exc))
        )
        return
    with env:
        for sid in ids:
            tick = time.monotonic()
            if time.time() >= deadline - 5:
                break
            folder = out / f"scenario_{sid:04d}"
            folder.mkdir()
            base = root / AROOT / "oracles" / f"{sid:04d}"
            W, T, a = (
                sp.load_npz(base / "W.npz"),
                sp.load_npz(base / "T.npz"),
                npz(base / "ARRAYS.npz"),
            )
            row = dict(
                sid=sid,
                point_sha256=point_sha,
                rows=W.shape[0],
                cols=W.shape[1],
                nnz=W.nnz + T.nnz,
                native_status=None,
                q_lower=float(ql[sid]),
                q_upper=None,
                bound_is_infinite=True,
                primal_residual=None,
                missing_reason=None,
                input_refs=[reference(base / n) for n in ["W.npz", "T.npz", "ARRAYS.npz"]],
            )
            with gp.Model(f"E4_FRACTIONAL_{sid}", env=env) as m:
                y = m.addMVar(len(a["c"]), lb=a["lo"], ub=a["hi"])
                m.addMConstr(W, y, a["sense"], a["h"] - T @ x)
                m.setObjective(a["c"] @ y)
                m.update()
                row["build_seconds"] = time.monotonic() - tick
                for key, value in dict(
                    Threads=1,
                    Method=1,
                    Seed=0,
                    FeasibilityTol=1e-9,
                    OptimalityTol=1e-9,
                    SoftMemLimit=6,
                    LogFile=str(folder / "NATIVE.log"),
                ).items():
                    m.setParam(key, value)
                remaining = min(30 - (time.monotonic() - tick) - 2, deadline - time.time() - 3)
                if remaining <= 0:
                    row["missing_reason"] = "scenario build exhausted 30 second cap"
                else:
                    m.Params.TimeLimit = remaining
                    m.optimize()
                    row.update(native_status=int(m.Status), native_runtime=m.Runtime)
                    raw = {}
                    try:
                        pi = np.asarray(m.getAttr("Pi"))
                        alpha, beta, corrected = replay_cut(W, T, a, pi)
                        lower, _ = affine_bounds(sp.csr_matrix(beta[None, :]), [alpha], x)
                        ql[sid] = max(ql[sid], float(lower[0]))
                        raw.update(pi=pi, corrected_pi=corrected, alpha=alpha, beta=beta)
                    except (gp.GurobiError, AttributeError):
                        row["missing_reason"] = "no LP dual witness"
                    if m.SolCount:
                        yy = np.asarray(y.X)
                        raw["y"] = yy
                        check = row_audit(W, a["h"] - T @ x, a["sense"], a["lo"], a["hi"], yy)
                        row["primal_residual"] = check
                        if max(check.values()) <= 1e-9:
                            _, upper = affine_bounds(sp.csr_matrix(a["c"][None, :]), [0], yy)
                            qu[sid] = float(upper[0])
                        else:
                            row["missing_reason"] = "primal failed unchanged 1e-9 numerical audit"
                    np.savez_compressed(folder / "RAW.npz", **raw)
                row.update(
                    q_lower=float(ql[sid]),
                    q_upper=qu[sid],
                    bound_is_infinite=qu[sid] is None,
                    wrapper_seconds=time.monotonic() - tick,
                )
            save(folder / "RESULT.json", row)
            records.append(row)
            print(
                "fractional", sid, row["native_status"], row["q_lower"], row["q_upper"], flush=True
            )
            if row["native_status"] == gp.GRB.INFEASIBLE:
                break
    first_cost = float(np.asarray(first["cost"], np.longdouble) @ x)
    risk_lowers = []
    for ref in config["feasible_probability_certificates"]:
        p = npz(verified(ref))["pstar"]
        lower, _ = affine_bounds(sp.csr_matrix(p[None, :]), [0], ql)
        risk_lowers.append(float(lower[0]))
    np.savez_compressed(
        out / "SCENARIO_BOUNDS.npz",
        q_lower=ql,
        q_upper=np.asarray([v if v is not None else np.inf for v in qu]),
    )
    save(
        out / "RESULT.json",
        dict(
            status="PARTIAL",
            point_sha256=point_sha,
            records=records,
            first_cost=first_cost,
            point_DRO_lower=first_cost + max(risk_lowers),
            point_ARO_lower=first_cost + float(ql.max()),
            point_upper=None,
            bound_is_infinite=True,
            explicitly_solved_count=len(records),
            bounded_only_count=400 - len(records),
            certified_zero_count=0,
            missing_upper_bound_count=sum(v is None for v in qu),
            coverage_complete=False,
            accuracy_met=False,
            full400_expansion_started=False,
            global_continuous_lower=None,
            point_lower_is_not_global_lower=True,
            solver_wrapper_seconds=time.monotonic() - start,
            solver_calls=len(records),
            missing_reason="Unsolved scenarios have no point-feasible recourse upper witness; no integer zero shortcut used",
        ),
    )


if __name__ == "__main__":
    root, config_path, out, store = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    config = read(config_path)
    ledger = BudgetLedger(store, "diagnostic-round1-20261111")
    claim = ledger.claim(
        dict(stage="E4_PROBE", config=reference(config_path), source=reference(__file__)), 300.0
    )
    save(out / "CLAIM.json", claim)
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("owned diagnostic deadline"))
    )
    rc = 1
    try:
        run(root, config, out, time.time() + claim["reserved_seconds"] - 2)
        rc = 0
    finally:
        save(out / "WAIT.json", dict(returncode=rc))
        ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
        save(out / "BUDGET_AFTER.json", ledger.read())
