"""One bounded diagnostic LP arm on PowerAI, under the shared round ledger.

Parent owns and waits for its child. Failures consume the same cumulative
budget; no production process or campaign state is opened for writing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.certificates import row_audit, lagrangian
from ev_assignment.l08.theta import lagrangian as theta_certificate
from ev_assignment_experiments.runtime.identity import wait_identity
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from .diagnostic import npz, read, reference, save
from .diagnostic_replay import R45


def parent(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(args.store, "diagnostic-round1-20261111")
    ledger.create("convergence-diagnostic-round1", 7200.0)
    claim = ledger.claim(
        dict(stage=args.stage, code_commit=args.code_commit, output=str(out)), args.seconds
    )
    save(out / "CLAIM.json", claim)
    command = [
        sys.executable,
        "-u",
        "-m",
        "ev_assignment_experiments.analysis.diagnostic_native",
        "--role",
        "worker",
        "--root",
        args.root,
        "--replay",
        args.replay,
        "--out",
        str(out),
        "--stage",
        args.stage,
        "--store",
        args.store,
        "--code-commit",
        args.code_commit,
        "--deadline",
        str(time.time() + claim["reserved_seconds"] - 2),
        "--seconds",
        str(args.seconds),
    ]
    rc, stopped = None, False
    try:
        with (out / "worker.log").open("x") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
            child = wait_identity(process.pid)
            ledger.child(claim["attempt"], child)
            try:
                rc = process.wait(
                    timeout=max(
                        0.1,
                        claim["reserved_seconds"] - (time.monotonic() - claim["started_monotonic"]),
                    )
                )
            except subprocess.TimeoutExpired:
                stopped = True
                process.terminate()
                try:
                    rc = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    rc = process.wait()
    finally:
        receipt = dict(returncode=rc, own_child_wait=True, wrapper_deadline=stopped)
        save(out / "WAIT.json", receipt)
        ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
        save(out / "BUDGET_AFTER.json", ledger.read())


def worker(args):
    if args.stage == "ENTRYPOINT_CHECK":
        save(Path(args.out) / "ENTRYPOINT.json", dict(solver_calls=0, runtime_probe_only=True))
        raise RuntimeError("Intentional runtime probe failure before solver import")
    import gurobipy as gp

    started = time.monotonic()
    out, root = Path(args.out), Path(args.root)
    result = dict(
        stage=args.stage,
        code_commit=args.code_commit,
        seed=0,
        threads=4,
        solver_version=list(gp.gurobi.version()),
        process_id=os.getpid(),
        native_status=None,
        optimize_calls=0,
        diagnostic_status="PREPARING",
        floor_policy="no added inherited scalar floor",
        shared_host=True,
        build_seconds=None,
        presolve_seconds=None,
        lp_seconds=None,
        audit_seconds=None,
        root_completed=False,
        cgroup=Path("/proc/self/cgroup").read_text(),
    )
    try:
        # One environment admission per arm, and the caller stops the entire
        # native sequence on ENV_BLOCKED rather than retrying the license.
        env = gp.Env(empty=True)
        env.setParam("LogFile", str(out / "environment.log"))
        try:
            env.start()
        except gp.GurobiError as exc:
            result.update(
                diagnostic_status="ENV_BLOCKED",
                missing_reason=str(exc),
                error_code=exc.errno,
                stop_reason="single_license_admission_failed",
            )
            return
        with env:
            if args.stage == "E2_DRO_ACTIVE":
                base = Path(args.replay) / "DRO"
                A, a, names = (
                    sp.load_npz(base / "A.npz"),
                    npz(base / "ARRAYS.npz"),
                    read(base / "NAMES.json"),
                )
            elif args.stage in {"E3_B0", "E3_R45"}:
                base = root / R45
                prefix = "INPUT" if args.stage == "E3_B0" else "PROJECTED_LP"
                A, a, names = (
                    sp.load_npz(base / f"{prefix}_A.npz"),
                    npz(base / f"{prefix}_ARRAYS.npz"),
                    read(base / "INPUT_NAMES.json"),
                )
                a["rhs"] = a.pop("b")
                a["vtype"] = a.pop("types")
                names = {"cols": names["columns"]}
            else:
                raise ValueError("Unadmitted diagnostic arm")
            cols = names["cols"] if isinstance(names, dict) else names
            if len(cols) != A.shape[1]:
                raise ValueError("Column mapping mismatch")
            result.update(
                rows=A.shape[0],
                cols=A.shape[1],
                nnz=A.nnz,
                source_binary_count=int(np.sum(a["vtype"] == "B")),
                binary_count=0,
                ordinary_lp=True,
                source_refs=[
                    reference(
                        base / ("A.npz" if args.stage.startswith("E2") else f"{prefix}_A.npz")
                    ),
                    reference(
                        base
                        / ("ARRAYS.npz" if args.stage.startswith("E2") else f"{prefix}_ARRAYS.npz")
                    ),
                ],
            )
            with gp.Model("DIAGNOSTIC_" + args.stage, env=env) as model:
                v = model.addMVar(A.shape[1], lb=a["lo"], ub=a["hi"], vtype="C")
                model.addMConstr(A, v, a["sense"], a["rhs"])
                model.setObjective(a["cost"] @ v + float(a.get("objcon", 0)))
                model.update()
                for key, value in dict(
                    Threads=4,
                    Seed=0,
                    Method=1,
                    FeasibilityTol=1e-9,
                    OptimalityTol=1e-9,
                    SoftMemLimit=20,
                    LogFile=str(out / "NATIVE.log"),
                ).items():
                    model.setParam(key, value)
                result["build_seconds"] = time.monotonic() - started
                remaining = args.deadline - time.time() - 20
                if remaining <= 0:
                    raise TimeoutError("Build exhausted bounded arm")
                model.Params.TimeLimit = remaining
                save(out / "NATIVE_BEGIN.json", dict(**result, time_limit=remaining))
                result["optimize_calls"] = 1
                model.optimize(
                    lambda m, where: m.terminate() if time.time() >= args.deadline - 20 else None
                )
                result.update(
                    native_status=int(model.Status),
                    native_runtime=model.Runtime,
                    root_completed=model.Status == gp.GRB.OPTIMAL,
                    iterations=model.IterCount,
                    solution_count=model.SolCount,
                    max_memory_GB=model.MaxMemUsed,
                )
                native_text = (out / "NATIVE.log").read_text()
                times = re.findall(r"Presolve time: ([0-9.]+)s", native_text)
                if times:
                    result["presolve_seconds"] = float(times[-1])
                    result["lp_seconds"] = max(0.0, model.Runtime - float(times[-1]))
                audit_start = time.monotonic()
                raw = {}
                if model.SolCount:
                    raw["x"] = np.asarray(v.X)
                    result["primal_value"] = model.ObjVal
                    result["primal_residual"] = row_audit(
                        A, a["rhs"], a["sense"], a["lo"], a["hi"], raw["x"]
                    )
                    groups = {}
                    for j, name in enumerate(cols):
                        family = name.split("[")[0].split("_")[0]
                        groups[family] = groups.get(family, 0.0) + float(a["cost"][j] * raw["x"][j])
                    result["cost_by_native_family"] = groups
                    selected = np.array([n.startswith("selected[") for n in cols])
                    binary = a["vtype"] == "B"
                    fraction = np.minimum(
                        abs(raw["x"] - np.floor(raw["x"])), abs(np.ceil(raw["x"]) - raw["x"])
                    )
                    result["activation_mass"] = float(raw["x"][selected].sum())
                    result["fractional_activations"] = int(np.sum(fraction[selected] > 1e-6))
                    result["fractional_original_binary"] = int(np.sum(fraction[binary] > 1e-6))
                if model.Status == gp.GRB.OPTIMAL:
                    raw["pi"] = np.asarray(model.getAttr("Pi"))
                    if args.stage.startswith("E2"):
                        cert, corrected = theta_certificate(A, a, raw["pi"], 60780, 60780)
                        result["dual_bound"] = cert["LB"]
                        save(out / "LOWER_CERTIFICATE.json", cert)
                        raw["corrected_pi"] = corrected
                    else:
                        bounds = npz(root / R45 / "PROVED_FINITE_BOUNDS.npz")
                        lower, corrected = lagrangian(
                            A,
                            a["rhs"],
                            a["sense"],
                            a["cost"],
                            bounds["lo"],
                            bounds["hi"],
                            raw["pi"],
                            float(a.get("objcon", 0)),
                        )
                        result["dual_bound"] = lower
                        raw["corrected_pi"] = corrected
                np.savez_compressed(out / "RAW.npz", **raw)
                result["audit_seconds"] = time.monotonic() - audit_start
                result["diagnostic_status"] = (
                    "OPTIMAL" if model.Status == gp.GRB.OPTIMAL else "PARTIAL_NATIVE"
                )
                result["evidence_grade"] = "OUTWARD_DUAL_BOUND_AND_NUMERICALLY_AUDITED_PRIMAL"
    except Exception as exc:
        result.update(diagnostic_status="ERROR", missing_reason=repr(exc))
        raise
    finally:
        result["worker_wrapper_seconds"] = time.monotonic() - started
        save(out / "RESULT.json", result)


def main():
    p = argparse.ArgumentParser()
    for key in ["root", "replay", "out", "store", "stage", "code-commit"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--role", choices=["parent", "worker"], default="parent")
    p.add_argument("--seconds", type=float, default=900.0)
    p.add_argument("--deadline", type=float)
    args = p.parse_args()
    if args.stage.startswith("E3") and args.seconds > 300:
        p.error("DET root LP cap is 300 seconds")
    if args.seconds > 900 or args.seconds <= 0:
        p.error("LP arm cap is 900 seconds")
    (parent if args.role == "parent" else worker)(args)


if __name__ == "__main__":
    main()
