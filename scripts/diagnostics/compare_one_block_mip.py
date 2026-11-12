"""One matched short MIP pair: frozen master versus one exact recourse block.

ROOT ROUND CONFIG VARIANT OUTPUT. Variant is base or one_block. No Q oracle
is run here, and native incumbents are never promoted to full400 audited UBs.
"""

from pathlib import Path
import math
import re
import signal
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.certificates import row_audit
from ev_assignment_experiments.analysis.diagnostic import npz, read, reference, save, digest
from ev_assignment_experiments.analysis.diagnostic_replay import AROOT
from ev_assignment_experiments.runtime.ledger import BudgetLedger


def run(root, round_root, config, variant, out, deadline):
    import gurobipy as gp

    started = time.monotonic()
    if variant not in {"base", "one_block"}:
        raise ValueError("Only the predeclared two-arm experiment is admitted")
    for ref in config["sources"]:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Paired experiment source changed")
    base = round_root / "E2_SCAN/DRO"
    A, a, names = sp.load_npz(base / "A.npz"), npz(base / "ARRAYS.npz"), read(base / "NAMES.json")
    best = read(round_root / "replay-r1/E1_ENVELOPES.json")["points"][0]
    z = npz(round_root / "replay-r1/points/DRO" / best["point"] / "POINT.npz")
    start = np.zeros(A.shape[1])
    start[:60780] = z["x"]
    start[60780] = best["true_risk"][1] + 1e-6
    theta = {
        int(n.rsplit("_", 1)[1]): j
        for j, n in enumerate(names["cols"])
        if n.startswith("THETA_SCENARIO_")
    }
    for sid, col in theta.items():
        start[col] = z["qU"][sid]
    result = dict(
        variant=variant,
        scenario=config["scenario"],
        source_config=config,
        scientific_identity=config["scientific_identity"],
        native_status=None,
        native_best_bound=None,
        native_incumbent=None,
        root_completed=False,
        evaluated_integer_UB=34938.625629835806,
        new_full400_UB=False,
        historical_adopted_LB=12536.584767617334,
        build_seconds=None,
        presolve_seconds=None,
        root_seconds=None,
        mip_tree_seconds=None,
        seed=0,
        threads=4,
        fullQ_calls=0,
    )
    if variant == "one_block":
        sid = config["scenario"]
        oracle = root / AROOT / "oracles" / f"{sid:04d}"
        W, T, rec = (
            sp.load_npz(oracle / "W.npz"),
            sp.load_npz(oracle / "T.npz"),
            npz(oracle / "ARRAYS.npz"),
        )
        original_cols = A.shape[1]
        nx, ny = 60780, W.shape[1]
        cost_row = sp.lil_matrix((1, original_cols + ny))
        cost_row[0, theta[sid]] = 1.0
        cost_row[0, original_cols:] = -rec["c"]
        block = sp.hstack([T, sp.csr_matrix((W.shape[0], original_cols - nx)), W], format="csr")
        fullA = sp.vstack(
            [sp.hstack([A, sp.csr_matrix((A.shape[0], ny))]), block, cost_row], format="csr"
        )
        if (
            (fullA[: A.shape[0], :original_cols] != A).nnz
            or (fullA[A.shape[0] : -1, :nx] != T).nnz
            or (fullA[A.shape[0] : -1, original_cols:] != W).nnz
        ):
            raise ValueError("Exact original recourse block embedding failed")
        fulla = dict(
            cost=np.r_[a["cost"], np.zeros(ny)],
            lo=np.r_[a["lo"], rec["lo"]],
            hi=np.r_[a["hi"], rec["hi"]],
            vtype=np.r_[a["vtype"], ["C"] * ny],
            rhs=np.r_[a["rhs"], rec["h"], 0.0],
            sense=np.r_[a["sense"], rec["sense"], ">"],
            objcon=a["objcon"],
        )
        witness = npz(config["warm_start_recourse"]["path"])
        if not np.array_equal(witness["x"], z["x"]):
            raise ValueError("Warm start uses a different first-stage point")
        start = np.r_[start, witness["y"]]
        result["exact_embedding"] = dict(
            original_cols=original_cols,
            new_columns=ny,
            new_rows=W.shape[0] + 1,
            W=reference(oracle / "W.npz"),
            T=reference(oracle / "T.npz"),
            arrays=reference(oracle / "ARRAYS.npz"),
            matrix_coefficients_exact=True,
            theta_lower_enforces_original_recourse_cost=True,
            domain="full continuous first-stage domain",
        )
        A, a = fullA, fulla
    check = row_audit(A, a["rhs"], a["sense"], a["lo"], a["hi"], start)
    if max(check.values()) > 1e-9:
        raise ValueError(("Paired start failed original matrix audit", check))
    result.update(
        rows=A.shape[0],
        cols=A.shape[1],
        nnz=A.nnz,
        binary_count=int(np.sum(a["vtype"] == "B")),
        warm_start_audit=check,
        common_warm_start_objective=float(a["cost"] @ start),
    )
    save(out / "ADMISSION.json", result)
    env = gp.Env(empty=True)
    env.setParam("LogFile", str(out / "ENVIRONMENT.log"))
    try:
        env.start()
    except gp.GurobiError as exc:
        save(
            out / "RESULT.json",
            dict(result, diagnostic_status="ENV_BLOCKED", missing_reason=str(exc)),
        )
        return
    with env, gp.Model("E5_" + variant, env=env) as m:
        v = m.addMVar(A.shape[1], lb=a["lo"], ub=a["hi"], vtype=a["vtype"])
        m.addMConstr(A, v, a["sense"], a["rhs"])
        m.setObjective(a["cost"] @ v + float(a["objcon"]))
        m.update()
        delta = m.getA() - A
        delta.eliminate_zeros()
        if delta.nnz:
            raise ValueError("Native storage altered the admitted coefficient matrix")
        v.Start = start
        (out / "nodefiles").mkdir()
        params = dict(
            Threads=4,
            Seed=0,
            Method=1,
            NodeMethod=1,
            PreSparsify=1,
            MIPFocus=0,
            MIPGap=0.001,
            MIPGapAbs=1.0,
            FeasibilityTol=1e-9,
            IntFeasTol=1e-9,
            OptimalityTol=1e-9,
            SoftMemLimit=20,
            NodefileStart=0.5,
            NodefileDir=str(out / "nodefiles"),
            LogFile=str(out / "NATIVE.log"),
        )
        for key, value in params.items():
            m.setParam(key, value)
        m.Params.TimeLimit = max(0.01, deadline - time.time() - 5)
        result["build_seconds"] = time.monotonic() - started
        marks = dict(first_mip=None, first_root_relaxation=None, branch_start=None)
        begin = time.monotonic()

        def callback(model, where):
            elapsed = time.monotonic() - begin
            if time.time() >= deadline - 5:
                model.terminate()
            if where == gp.GRB.Callback.MIP:
                if marks["first_mip"] is None:
                    marks["first_mip"] = elapsed
                if model.cbGet(gp.GRB.Callback.MIP_NODCNT) >= 1 and marks["branch_start"] is None:
                    marks["branch_start"] = elapsed
            if (
                where == gp.GRB.Callback.MIPNODE
                and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL
                and marks["first_root_relaxation"] is None
            ):
                marks["first_root_relaxation"] = elapsed

        save(
            out / "NATIVE_BEGIN.json",
            dict(parameters=params, time_limit=m.Params.TimeLimit, model=result),
        )
        m.optimize(callback)

        def finite(value):
            return float(value) if math.isfinite(value) and abs(value) < 1e90 else None

        result.update(
            native_status=int(m.Status),
            native_runtime=m.Runtime,
            native_best_bound=finite(m.ObjBoundC),
            native_incumbent=finite(m.ObjVal) if m.SolCount else None,
            solution_count=m.SolCount,
            nodes=m.NodeCount,
            work=m.Work,
            max_memory_GB=m.MaxMemUsed,
            phase_marks=marks,
            root_completed=marks["branch_start"] is not None or m.Status == gp.GRB.OPTIMAL,
            diagnostic_status="OPTIMAL" if m.Status == gp.GRB.OPTIMAL else "BOUNDED_PARTIAL",
            new_native_bound_scope="finite-cut INTEGER master; valid lower for original DRO, not LP optimum",
            evidence_grade="NATIVE_NUMERICAL_MIP_BOUND",
        )
        times = re.findall(r"Presolve time: ([0-9.]+)s", (out / "NATIVE.log").read_text())
        if times:
            result["presolve_seconds"] = float(times[-1])
        if marks["branch_start"] is not None:
            result["mip_tree_seconds"] = max(0, m.Runtime - marks["branch_start"])
            if result["presolve_seconds"] is not None:
                result["root_seconds"] = max(0, marks["branch_start"] - result["presolve_seconds"])
        if m.SolCount:
            np.savez_compressed(out / "NATIVE_INCUMBENT.npz", x=np.asarray(v.X))
        result["adopted_global_LB"] = max(
            result["historical_adopted_LB"],
            result["native_best_bound"] if result["native_best_bound"] is not None else -np.inf,
        )
    result["wrapper_seconds"] = time.monotonic() - started
    save(out / "RESULT.json", result)


if __name__ == "__main__":
    root, round_root, config_path = map(Path, sys.argv[1:4])
    variant, out = sys.argv[4], Path(sys.argv[5])
    out.mkdir(parents=True, exist_ok=False)
    config = read(config_path)
    ledger = BudgetLedger(round_root / "store", "diagnostic-round1-20261111")
    claim = ledger.claim(
        dict(stage="E5_" + variant, config=reference(config_path), source=reference(__file__)),
        600.0,
        charge_process_start=True,
    )
    save(out / "CLAIM.json", claim)
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("owned diagnostic deadline"))
    )
    rc = 1
    try:
        run(root, round_root, config, variant, out, time.time() + claim["reserved_seconds"] - 2)
        rc = 0
    finally:
        save(out / "WAIT.json", dict(returncode=rc))
        ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
        save(out / "BUDGET_AFTER.json", ledger.read())
