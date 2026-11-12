from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from fractions import Fraction

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True

import numpy as np
from scipy import sparse as sp
import gurobipy as gp


BASE = Path("/home/Shixin/ev-assignment")
REPAIR = BASE / "L08_DRO_R43_REPAIR_01_FULL_HISTORY_SCOPE"
R42 = BASE / "L08_DRO_R42_JOINT_SEPARATION_PILOT"
R41 = BASE / "L08_DRO_R41_NESTED_BENDERS_ROOT/REPAIR_01"
R36 = BASE / "L08_K4L4_DRO_READY_PARALLEL_R36"
R29 = BASE / "L08_K4L4_CALIBRATION_AND_FORECAST_R29"
DET = BASE / "L08_DET_R38_CAPACITY_CONTINUATION_V1/RECOVERY_01"
ROOT = BASE / "L08_DRO_R44_POOLED_ROOT_CONTINUATION"
OUT = ROOT / "CONTINUE_01"
PREFLIGHT_MODE = False
PYTHON = Path("/home/Shixin/ev-assignment/releases/case0-expanded-r5-20260910T045000Z/.venv/bin/python")
WALL_LIMIT = 900.0
MAX_BATCHES = 10
FIRST_N = 60780


def cv(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, obj):
    path = Path(path)
    assert OUT == path or OUT in path.parents, path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=cv, allow_nan=False) + "\n")
    os.replace(tmp, path)


def ledger_event(call_id, phase, call_type, context, extra=None):
    """Append one cross-process-safe optimizer call event.

    The JSONL ledger is authoritative.  Per-call immutable JSON files provide
    an independent directory-scan cross-check, but are not used as the sole
    call counter.
    """
    import fcntl
    event = {
        "scientific_optimizer_call_id": call_id,
        "phase": phase,
        "call_type": call_type,
        "context": context,
        "pid": os.getpid(),
        "epoch": time.time(),
        "monotonic": time.monotonic(),
    }
    if extra:
        event.update(extra)
    calls = OUT / "optimizer_calls"
    calls.mkdir(parents=True, exist_ok=True)
    immutable = calls / f"{call_id}_{phase}.json"
    with immutable.open("x") as f:
        f.write(json.dumps(event, indent=2, sort_keys=True, default=cv, allow_nan=False) + "\n")
    ledger = OUT / "OPTIMIZER_CALL_LEDGER.jsonl"
    lock = OUT / "OPTIMIZER_CALL_LEDGER.lock"
    with lock.open("a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        with ledger.open("a") as f:
            f.write(json.dumps(event, sort_keys=True, default=cv, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    return event


def tracked_optimize(model, call_type, context, extra=None):
    if PREFLIGHT_MODE:
        raise RuntimeError("PREFLIGHT_OPTIMIZER_FORBIDDEN")
    import uuid
    call_id = uuid.uuid4().hex
    ledger_event(call_id, "BEGIN", call_type, context, extra)
    try:
        model.optimize()
    except BaseException as exc:
        ledger_event(call_id, "END", call_type, context, {"raised": type(exc).__name__, "message": str(exc)})
        raise
    ledger_event(call_id, "END", call_type, context, {
        "status": int(model.Status),
        "runtime": float(model.Runtime),
        "iterations": float(getattr(model, "IterCount", 0.0)),
        "work": float(getattr(model, "Work", 0.0)),
    })
    return call_id


def optimizer_ledger_summary():
    ledger = OUT / "OPTIMIZER_CALL_LEDGER.jsonl"
    rows = []
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    begins = {r["scientific_optimizer_call_id"]: r for r in rows if r["phase"] == "BEGIN"}
    ends = {r["scientific_optimizer_call_id"]: r for r in rows if r["phase"] == "END"}
    scan = {p.name[:-11] for p in (OUT / "optimizer_calls").glob("*_BEGIN.json")} if (OUT / "optimizer_calls").exists() else set()
    return {
        "explicit_begin_count": len(begins),
        "explicit_end_count": len(ends),
        "unfinished_call_ids": sorted(set(begins) - set(ends)),
        "directory_scan_begin_count": len(scan),
        "directory_scan_matches_ledger": scan == set(begins),
        "by_type": {k: sum(r["call_type"] == k for r in begins.values()) for k in sorted({r["call_type"] for r in begins.values()})},
    }


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def proc(pid):
    p = Path(f"/proc/{pid}")
    try:
        raw = (p / "stat").read_text()
        left = raw.find("(")
        right = raw.rfind(")")
        if left <= 0 or right <= left or right + 2 > len(raw):
            raise ValueError("malformed /proc/PID/stat")
        parsed_pid = int(raw[:left].strip())
        if parsed_pid != pid:
            raise ValueError("PID field mismatch")
        comm = raw[left + 1:right]
        tail = raw[right + 1:].strip().split()
        if len(tail) < 20:
            raise ValueError("short /proc/PID/stat tail")
        argv = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        rss = 0
        for line in (p / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
        return {"pid": pid, "comm": comm, "ppid": int(tail[1]), "state": tail[0], "start_ticks": tail[19], "argv": argv, "rss": rss, "stat_raw": raw.rstrip("\n")}
    # A process can disappear between opening /proc and reading all identity
    # fields.  Treat that exactly like a missing PID; never let a transient
    # or malformed record crash the parent supervisor.
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError, IndexError):
        return None


def descendants(root_pid):
    table = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        q = proc(int(p.name))
        if q:
            table[q["pid"]] = q
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, q in table.items():
            if q["ppid"] in found and pid not in found:
                found.add(pid)
                changed = True
    return [table[p] for p in found if p in table]


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def det_snapshot():
    owner = read(DET / "PROCESS.json")["owner"]
    actual = proc(int(owner["pid"]))
    return {"registered": owner, "actual": actual, "boot_id": boot_id(), "model": record(DET / "MODEL.mps.gz"), "native_begin": record(DET / "NATIVE_BEGIN.json"), "bounds": read(DET / "BOUNDS.json"), "progress": read(DET / "PROGRESS.json")}


def required_inputs():
    paths = [
        REPAIR / "short_test/C_POOL/ITER_002/SOLUTION.npz",
        REPAIR / "short_test/C_POOL/ITER_002/BASIS.npz",
        REPAIR / "short_test/C_POOL/FINAL_MODEL.mps.gz",
        REPAIR / "payload/COMPRESSION/COMPRESSED_A.npz",
        REPAIR / "payload/COMPRESSION/COMPRESSED_ARRAYS.npz",
        REPAIR / "payload/COMPRESSION/COMPRESSED_NAMES.json",
        REPAIR / "payload/COMPRESSION/DORMANT_A.npz",
        REPAIR / "payload/COMPRESSION/DORMANT_ARRAYS.npz",
        REPAIR / "payload/COMPRESSION/DORMANT_NAMES.json",
        REPAIR / "payload/ANCHOR/FINAL_A.npz",
        REPAIR / "payload/ANCHOR/FINAL_ARRAYS.npz",
        REPAIR / "payload/ANCHOR/FINAL_NAMES.json",
        REPAIR / "DELTA_PROBE.npz",
        REPAIR / "short_test/C_POOL/ITER_000/SCAN.npz",
        REPAIR / "short_test/C_POOL/ITER_001/SCAN.npz",
        REPAIR / "short_test/C_POOL/SUMMARY.json",
        R41 / "LOCK.json",
        R41 / "identity/FIRST_A.npz",
        R41 / "identity/FIRST_ARRAYS.npz",
        R41 / "identity/FIRST_NAMES.json",
        R41 / "identity/FIRST_ROWS.json",
        R42 / "PROBABILITY_INTERIOR_STRICT/P_STAR.npz",
        R42 / "PROBABILITY_INTERIOR_STRICT/CERTIFICATE.json",
        R42 / "C0_UPPER_REPLAY.json",
        R42 / "GLOBAL_BOUND_PROOFS.json",
        R42 / "RECOURSE_CUT_AUDIT.json",
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    return paths, missing


def protection_snapshot(input_records=None):
    files = input_records or []
    return {
        "epoch": time.time(),
        "DET": det_snapshot(),
        "input_hashes": {str(p): sha(p) for p in files},
        "production_signals_sent": [],
        "production_writes": [],
    }


def enclosure(A, x):
    ld = np.longdouble
    eps = np.finfo(ld).eps
    n = A.shape[1]
    AA = A.astype(ld)
    xx = np.asarray(x, ld)
    value = AA @ xx
    mass = abs(AA) @ abs(xx)
    gamma = ld(2 * n + 8) * eps / (1 - ld(2 * n + 8) * eps)
    err = gamma * mass + np.finfo(ld).tiny
    return value - err, value + err


def row_audit(A, rhs, sense, lo, hi, x):
    ld = np.longdouble
    value = A.astype(ld) @ np.asarray(x, ld) - np.asarray(rhs, ld)
    rv = np.where(sense == "=", abs(value), np.where(sense == "<", value, -value))
    return {
        "row": float(max(0, np.max(rv, initial=0))),
        "bound": float(max(0, np.max(np.asarray(lo, ld) - x, initial=0), np.max(np.asarray(x, ld) - hi, initial=0))),
    }


def lagrangian(A, rhs, sense, cost, lo, hi, pi, objcon=0.0):
    ld = np.longdouble
    eps = np.finfo(ld).eps
    cert_pi = np.where(sense == "<", np.minimum(pi, 0), np.where(sense == ">", np.maximum(pi, 0), pi))
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    lower_at, upper_at = enclosure(A.T, cert_pi)
    lower_rc = np.asarray(cost, ld) - upper_at
    upper_rc = np.asarray(cost, ld) - lower_at
    err = eps * (abs(cost) + np.maximum(abs(lower_at), abs(upper_at)))
    lower_rc -= err
    upper_rc += err
    terms = np.minimum.reduce([lower_rc * lo, lower_rc * hi, upper_rc * lo, upper_rc * hi])
    guard = ld(8 * len(terms) + 8) * eps * sum(abs(terms))
    guard += ld(8 * len(cert_pi) + 8) * eps * sum(abs(np.asarray(cert_pi, ld) * rhs))
    value = ld(objcon) + np.asarray(cert_pi, ld) @ np.asarray(rhs, ld) + sum(terms) - guard
    return float(np.nextafter(float(value), -np.inf)), np.asarray(cert_pi, float)


def scan_dormant(A, rhs, sense, x, threshold=1e-7):
    lower, upper = enclosure(A, x)
    rhs = np.asarray(rhs, np.longdouble)
    violation = np.where(sense == ">", rhs - lower, np.where(sense == "<", upper - rhs, np.maximum(abs(lower - rhs), abs(upper - rhs))))
    add = np.flatnonzero(~np.isfinite(violation) | (violation > threshold))
    return add, np.asarray(violation, float)


def matrix_sha(A):
    A = A.tocsr()
    return hashlib.sha256(A.data.tobytes() + A.indices.tobytes() + A.indptr.tobytes()).hexdigest()


def row_fingerprint(Arow, rhs, sense):
    row = Arow.tocsr()
    h = hashlib.sha256()
    h.update(row.indices.tobytes())
    h.update(row.data.tobytes())
    h.update(np.asarray([rhs], dtype=np.float64).tobytes())
    h.update(str(sense).encode())
    return h.hexdigest()


def scenario_name(name):
    return name.startswith(("reused_benders/", "benders/", "r42_benders/", "r43_probe/", "r44_benders/"))


def probability_name(name):
    return name.startswith(("probability/", "historical_upper/", "r42_probability/", "r44_probability/"))


def load_csr_npz(path):
    z = np.load(path)
    if {"data", "indices", "indptr", "shape"}.issubset(z.files):
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return sp.load_npz(path).tocsr()


def directed_row_violations(A, rhs, sense, x):
    ld = np.longdouble
    value = A.astype(ld) @ np.asarray(x, ld) - np.asarray(rhs, ld)
    return np.asarray(np.where(sense == "=", abs(value), np.where(sense == "<", value, -value)), ld)


def endpoint_residual_audit(A, rhs, sense, lo, hi, names, dA, drhs, dsense, x):
    rv = directed_row_violations(A, rhs, sense, x)
    active = np.asarray([scenario_name(n) for n in names], dtype=bool)
    probability = np.asarray([probability_name(n) for n in names], dtype=bool)
    physical = ~(active | probability)
    bound = max(
        np.longdouble(0),
        np.max(np.asarray(lo, np.longdouble) - np.asarray(x, np.longdouble), initial=np.longdouble(0)),
        np.max(np.asarray(x, np.longdouble) - np.asarray(hi, np.longdouble), initial=np.longdouble(0)),
    )
    active_ids = np.flatnonzero(active)
    active_local = int(np.argmax(rv[active]))
    active_row = int(active_ids[active_local])
    dormant_add, dormant_v = scan_dormant(dA, drhs, dsense, x, threshold=1e-7)
    over = np.flatnonzero(rv > np.longdouble(1e-9))
    audit = {
        "arithmetic": "float64 stored payload evaluated with numpy.longdouble; dormant uses directed interval enclosure",
        "physical_max_violation": float(max(np.longdouble(0), np.max(rv[physical], initial=np.longdouble(0)))),
        "probability_max_violation": float(max(np.longdouble(0), np.max(rv[probability], initial=np.longdouble(0)))),
        "bound_max_violation": float(bound),
        "active_scenario_max_violation": float(max(np.longdouble(0), rv[active_row])),
        "active_scenario_max_row_name": names[active_row],
        "active_scenario_max_row_index": active_row,
        "active_scenario_rows_above_1e_9": int(np.sum(rv[active] > np.longdouble(1e-9))),
        "active_scenario_rows_above_1e_8": int(np.sum(rv[active] > np.longdouble(1e-8))),
        "all_rows_above_1e_9_count": int(len(over)),
        "all_rows_above_1e_9_exclusively_active_scenario": bool(np.all(active[over])),
        "dormant_max_directed_violation": float(np.max(dormant_v)) if len(dormant_v) else None,
        "dormant_rows_above_1e_7": int(len(dormant_add)),
        "known_active_benders_primal_violation_is_not_strict_primal_feasibility": True,
        "known_active_benders_primal_violation_is_not_claimed_as_roundoff": True,
        "saved_primal_objective_not_used_for_LB_or_UB": True,
    }
    expected = {
        "physical_max_violation": 8.640310689145281e-14,
        "probability_max_violation": 1.0432559221888738e-13,
        "bound_max_violation": 0.0,
        "active_scenario_max_violation": 4.709423251370026e-08,
        "active_scenario_max_row_name": "r42_benders/360/4",
        "active_scenario_rows_above_1e_9": 35,
        "active_scenario_rows_above_1e_8": 16,
        "dormant_max_directed_violation": -180.33308919002837,
    }
    checks = {
        "physical_at_1e_9": audit["physical_max_violation"] <= 1e-9,
        "probability_at_1e_9": audit["probability_max_violation"] <= 1e-9,
        "bounds_at_1e_12": audit["bound_max_violation"] <= 1e-12,
        "all_gt_1e_9_only_active_benders": audit["all_rows_above_1e_9_exclusively_active_scenario"],
        "active_max_row_matches_archive": audit["active_scenario_max_row_name"] == expected["active_scenario_max_row_name"],
        "active_max_value_matches_archive": abs(audit["active_scenario_max_violation"] - expected["active_scenario_max_violation"]) <= 1e-12,
        "active_count_gt_1e_9_matches_archive": audit["active_scenario_rows_above_1e_9"] == expected["active_scenario_rows_above_1e_9"],
        "active_count_gt_1e_8_matches_archive": audit["active_scenario_rows_above_1e_8"] == expected["active_scenario_rows_above_1e_8"],
        "dormant_no_readd_at_1e_7": audit["dormant_rows_above_1e_7"] == 0,
        "dormant_max_matches_archive": abs(audit["dormant_max_directed_violation"] - expected["dormant_max_directed_violation"]) <= 1e-9,
    }
    audit["expected_archive_values"] = expected
    audit["checks"] = checks
    audit["passed"] = all(checks.values())
    return audit


def restore_endpoint(out):
    t0 = time.monotonic()
    cA = sp.load_npz(REPAIR / "payload/COMPRESSION/COMPRESSED_A.npz").tocsr()
    cz = dict(np.load(REPAIR / "payload/COMPRESSION/COMPRESSED_ARRAYS.npz"))
    cn = read(REPAIR / "payload/COMPRESSION/COMPRESSED_NAMES.json")
    pA = load_csr_npz(REPAIR / "DELTA_PROBE.npz")
    pz = np.load(REPAIR / "DELTA_PROBE.npz")
    anchorA = sp.load_npz(REPAIR / "payload/ANCHOR/FINAL_A.npz").tocsr()
    anchorz = dict(np.load(REPAIR / "payload/ANCHOR/FINAL_ARRAYS.npz"))
    anchorn = read(REPAIR / "payload/ANCHOR/FINAL_NAMES.json")
    readd0 = np.load(REPAIR / "short_test/C_POOL/ITER_000/SCAN.npz")["readd_indices"].astype(int)
    readd1 = np.load(REPAIR / "short_test/C_POOL/ITER_001/SCAN.npz")["readd_indices"].astype(int)
    readd = np.r_[readd0, readd1]
    assert len(readd) == 72 and len(np.unique(readd)) == 72
    A = sp.vstack([cA, pA, anchorA[readd]], format="csr")
    rhs = np.r_[cz["rhs"], pz["rhs"], anchorz["rhs"][readd]]
    sense = np.r_[cz["sense"], pz["sense"], anchorz["sense"][readd]]
    names = {
        "columns": cn["columns"],
        "rows": cn["rows"] + pz["names"].tolist() + [anchorn["rows"][i] for i in readd],
    }
    sol = dict(np.load(REPAIR / "short_test/C_POOL/ITER_002/SOLUTION.npz"))
    basis = dict(np.load(REPAIR / "short_test/C_POOL/ITER_002/BASIS.npz"))
    dormant_idx = np.load(REPAIR / "short_test/C_POOL/ITER_002/SCAN.npz")["dormant_indices"].astype(int)
    dA = anchorA[dormant_idx].tocsr()
    drhs = anchorz["rhs"][dormant_idx]
    dsense = anchorz["sense"][dormant_idx]
    dnames = [anchorn["rows"][i] for i in dormant_idx]
    assert A.shape == (91298, 60874) and A.nnz == 1651240
    assert len(sol["x"]) == A.shape[1] and len(sol["Pi"]) == A.shape[0]
    assert len(basis["VBasis"]) == A.shape[1] and len(basis["CBasis"]) == A.shape[0]
    physical = sum(not scenario_name(n) and not probability_name(n) for n in names["rows"])
    probability = sum(probability_name(n) for n in names["rows"])
    scenario = sum(scenario_name(n) for n in names["rows"])
    assert (physical, probability, scenario) == (90768, 231, 299), (physical, probability, scenario)
    assert len(dnames) == 2577 and all(scenario_name(n) for n in dnames)
    aset = {row_fingerprint(A.getrow(i), rhs[i], sense[i]) for i, n in enumerate(names["rows"]) if scenario_name(n)}
    dset = {row_fingerprint(dA.getrow(i), drhs[i], dsense[i]) for i in range(dA.shape[0])}
    assert len(aset) == 299 and len(dset) == 2577 and not (aset & dset)
    assert len(aset | dset) == 2876
    initial_lb, cert_pi = lagrangian(A, rhs, sense, cz["cost"], cz["lo"], cz["hi"], sol["Pi"])
    assert abs(initial_lb - 10116.746983790255) <= 1e-5, initial_lb
    residual = endpoint_residual_audit(A, rhs, sense, cz["lo"], cz["hi"], names["rows"], dA, drhs, dsense, sol["x"])
    atomic_json(out / "ENDPOINT_RESIDUAL_AUDIT.json", residual)
    if not residual["passed"]:
        raise RuntimeError(("ENDPOINT_IDENTITY_OR_RESIDUAL_MISMATCH", residual))
    final_mps = gp.read(str(REPAIR / "short_test/C_POOL/FINAL_MODEL.mps.gz"))
    mps_check = {"rows": final_mps.NumConstrs, "columns": final_mps.NumVars, "nnz": final_mps.NumNZs, "row_names_match": final_mps.getAttr("ConstrName") == names["rows"], "column_names_match": final_mps.getAttr("VarName") == names["columns"]}
    final_mps.dispose()
    assert mps_check == {"rows": 91298, "columns": 60874, "nnz": 1651240, "row_names_match": True, "column_names_match": True}
    identity = {
        "restored_from_true_R43_endpoint": True,
        "rows": A.shape[0], "columns": A.shape[1], "nnz": A.nnz,
        "physical_original": physical, "probability": probability, "active_scenario_cuts": scenario,
        "dormant_scenario_cuts": len(dnames), "historical_scenario_total": scenario + len(dnames),
        "probe_rows": pA.shape[0], "readded_rows": len(readd), "matrix_sha": matrix_sha(A),
        "initial_solution_residual_by_category": residual, "MPS_crosscheck": mps_check,
        "initial_certified_LB": initial_lb, "restore_wall": time.monotonic() - t0,
    }
    atomic_json(out / "MODEL_AND_POOL_IDENTITY.json", identity)
    np.savez_compressed(out / "INITIAL_STATE.npz", x=sol["x"], Pi=cert_pi, RC=sol["RC"], VBasis=basis["VBasis"], CBasis=basis["CBasis"])
    atomic_json(out / "ENDPOINT_LB_REPLAY.json", {
        "passed": True, "certified_LB": initial_lb, "expected": 10116.746983790255,
        "source": record(REPAIR / "short_test/C_POOL/ITER_002/SOLUTION.npz"),
        "matrix_sha": matrix_sha(A), "rows": A.shape[0], "columns": A.shape[1], "nnz": A.nnz,
        "proof": "complete dual, registered variable bounds, matrix identity and directed Lagrangian replay; saved primal objective is not used",
    })
    atomic_json(out / "INITIAL_BOUNDS.json", {"certified_LB": initial_lb, "source": record(REPAIR / "short_test/C_POOL/ITER_002/SOLUTION.npz"), "audited_UB": None, "certified_UB": None})
    return {
        "A": A, "rhs": rhs, "sense": sense, "cost": cz["cost"], "lo": cz["lo"], "hi": cz["hi"],
        "names": names, "x": sol["x"], "Pi": cert_pi, "RC": sol["RC"], "VBasis": basis["VBasis"], "CBasis": basis["CBasis"],
        "dA": dA, "drhs": drhs, "dsense": dsense, "dnames": dnames, "dsource": dormant_idx.tolist(), "LB": initial_lb,
    }


def build_master(state, log):
    m = gp.Model("R44_POOLED_CONTINUOUS_ROOT")
    m.Params.OutputFlag = 0
    v = m.addMVar(len(state["cost"]), lb=state["lo"], ub=state["hi"], name="v")
    m.addMConstr(state["A"], v, state["sense"], state["rhs"])
    m.setObjective(state["cost"] @ v)
    m.update()
    m.setAttr("VarName", m.getVars(), state["names"]["columns"])
    m.setAttr("ConstrName", m.getConstrs(), state["names"]["rows"])
    m.update()
    m.setAttr("VBasis", m.getVars(), np.asarray(state["VBasis"], int).tolist())
    m.setAttr("CBasis", m.getConstrs(), np.asarray(state["CBasis"], int).tolist())
    m.update()
    return m, v


def master_config(m, log, remaining):
    m.Params.Threads = 1
    m.Params.Method = 1
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    m.Params.SoftMemLimit = 3
    m.Params.LPWarmStart = 1
    m.Params.TimeLimit = max(0.1, remaining)
    m.Params.OutputFlag = 1
    m.Params.LogFile = str(log)


def basis_accepted(log):
    try:
        t = Path(log).read_text(errors="replace")
    except FileNotFoundError:
        return False
    return "LP warm-start: use basis" in t


def configure_oracle(m, log, remaining):
    m.Params.Threads = 1
    m.Params.Method = 1
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    m.Params.SoftMemLimit = 3
    m.Params.TimeLimit = max(0.1, remaining)
    m.Params.OutputFlag = 1
    m.Params.LogFile = str(log)


def replay_cut(W, T, a, pi):
    pi = np.asarray(pi, float)
    pi = np.where(a["sense"] == "<", np.minimum(pi, 0), np.where(a["sense"] == ">", np.maximum(pi, 0), pi))
    lower_at, upper_at = enclosure(W.T, pi)
    ld = np.longdouble
    eps = np.finfo(ld).eps
    rrl = np.asarray(a["c"], ld) - upper_at
    rru = np.asarray(a["c"], ld) - lower_at
    er = eps * (abs(a["c"]) + np.maximum(abs(lower_at), abs(upper_at)))
    rrl -= er
    rru += er
    terms = np.minimum.reduce([rrl * a["finite_lo"], rrl * a["finite_hi"], rru * a["finite_lo"], rru * a["finite_hi"]])
    bl, bu = enclosure(T.T, -pi)
    beta = np.asarray((bl + bu) / 2, float)
    err = np.maximum(abs(bl - beta), abs(bu - beta))
    correction = sum(err * np.maximum(abs(a["xlo"]), abs(a["xhi"])))
    hl, _ = enclosure(sp.csr_matrix(a["h"][None, :]), pi)
    alpha = hl[0] + sum(terms) - correction
    alpha -= eps * (8 * len(terms) + 8) * sum(abs(terms)) + eps * (8 * len(err) + 8) * correction
    return float(np.nextafter(float(alpha), -np.inf)), beta, pi


class Oracle:
    def __init__(self, sid, root):
        self.sid = int(sid)
        self.root = Path(root)
        self.out = self.root / "oracles" / f"{self.sid:04d}"
        self.out.mkdir(parents=True, exist_ok=True)
        src = R41 / "oracles" / f"{self.sid:04d}"
        self.a = dict(np.load(src / "ARRAYS.npz"))
        self.W = sp.load_npz(src / "W.npz").tocsr()
        self.T = sp.load_npz(src / "T.npz").tocsr()
        self.qbar = read(src / "QBAR_PROOF.json")["Qbar"]
        self.model = None
        self.y = None
        self.rows = None

    def cache_candidates(self):
        roots = [R42 / "B_JOINT_SEPARATION_FROM_C0/oracles" / f"{self.sid:04d}", R41 / "oracles" / f"{self.sid:04d}"]
        for root in roots:
            if not root.exists():
                continue
            for result in sorted(root.glob("**/RESULT.json"), reverse=True):
                try:
                    rr = read(result)
                    point = Path(rr.get("primal", result.parent / "POINT.npz"))
                    cut = Path(rr.get("cut", result.parent / "CUT.npz"))
                    if point.is_file() and cut.is_file():
                        yield result, point, cut
                except Exception:
                    continue

    def validate_point(self, x, y, pi):
        rhs = self.a["h"] - self.T @ x
        check = row_audit(self.W, rhs, self.a["sense"], self.a["lo"], self.a["hi"], y)
        assert max(check.values()) <= 1e-9, (self.sid, check)
        alpha, beta, pi = replay_cut(self.W, self.T, self.a, pi)
        ql = float(np.longdouble(alpha) + np.asarray(beta, np.longdouble) @ np.asarray(x, np.longdouble))
        _, up = enclosure(sp.csr_matrix(self.a["c"][None, :]), y)
        qu = float(np.nextafter(float(up[0]), np.inf))
        assert qu - ql >= -1e-5, (self.sid, ql, qu)
        return check, alpha, beta, pi, ql, qu

    def solve(self, x, label, deadline):
        d = self.out / label
        d.mkdir(parents=True, exist_ok=False)
        point_sha = hashlib.sha256(np.asarray(x, float).tobytes()).hexdigest()
        for result, point, cut in self.cache_candidates():
            z = np.load(point)
            if "x" not in z.files or not np.array_equal(z["x"], x):
                continue
            y, pi = z["y"], z["Pi"]
            check, alpha, beta, pi, ql, qu = self.validate_point(x, y, pi)
            source_cut = np.load(cut)
            if "alpha" in source_cut.files and abs(float(source_cut["alpha"]) - alpha) > 1e-5:
                continue
            np.savez_compressed(d / "CUT.npz", alpha=alpha, beta=beta, Pi=pi)
            rec = {"sid": self.sid, "point_sha": point_sha, "qL": ql, "qU_audited": qu, "qU_independently_certified": None, "UB_status": "NUMERICALLY_AUDITED_ONLY", "cut": str(d / "CUT.npz"), "primal": str(point), "residual": check, "global_cut_valid": True, "cache_hit": True, "new_optimize_calls": 0, "source_result": record(result), "source_point": record(point), "source_cut": record(cut)}
            atomic_json(d / "RESULT.json", rec)
            return rec
        if self.model is None:
            self.model = gp.Model(f"R44_ORACLE_{self.sid}")
            self.model.Params.OutputFlag = 0
            self.y = self.model.addMVar(len(self.a["c"]), lb=self.a["lo"], ub=self.a["hi"])
            self.rows = self.model.addMConstr(self.W, self.y, self.a["sense"], self.a["h"])
            self.model.setObjective(self.a["c"] @ self.y)
        remaining = deadline - time.monotonic() - 30
        if remaining <= 0:
            raise TimeoutError("R44_DEADLINE_BEFORE_RECOURSE")
        self.rows.RHS = self.a["h"] - self.T @ x
        native = d / "NATIVE"
        native.mkdir()
        configure_oracle(self.model, native / "NATIVE.log", remaining)
        atomic_json(native / "BEGIN.json", {"epoch": time.time(), "sid": self.sid, "remaining": remaining})
        call_id = tracked_optimize(self.model, "RECOURSE_ORACLE", {"sid": self.sid, "label": label}, {"remaining": remaining})
        atomic_json(native / "END.json", {"epoch": time.time(), "status": self.model.Status, "runtime": self.model.Runtime, "iterations": self.model.IterCount, "work": self.model.Work, "max_mem_GB": self.model.MaxMemUsed})
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(("RECOURSE_NOT_OPTIMAL", self.sid, self.model.Status))
        y = np.asarray(self.y.X)
        pi = np.asarray(self.model.getAttr("Pi"))
        check, alpha, beta, pi, ql, qu = self.validate_point(x, y, pi)
        np.savez_compressed(d / "POINT.npz", x=x, y=y, Pi=pi, RC=self.model.getAttr("RC"))
        np.savez_compressed(d / "CUT.npz", alpha=alpha, beta=beta, Pi=pi)
        rec = {"sid": self.sid, "point_sha": point_sha, "qL": ql, "qU_audited": qu, "qU_independently_certified": None, "UB_status": "NUMERICALLY_AUDITED_ONLY", "cut": str(d / "CUT.npz"), "primal": str(d / "POINT.npz"), "residual": check, "global_cut_valid": True, "cache_hit": False, "new_optimize_calls": 1, "scientific_optimizer_call_id": call_id}
        atomic_json(d / "RESULT.json", rec)
        return rec

    def close(self):
        if self.model is not None:
            self.model.dispose()


def oracle_worker(conn, ids, root):
    objects = {}
    try:
        while True:
            msg = conn.recv()
            if msg["op"] == "close":
                break
            ans = []
            for sid in ids:
                if sid not in objects:
                    objects[sid] = Oracle(sid, root)
                ans.append(objects[sid].solve(msg["x"], msg["label"], msg["deadline"]))
            conn.send({"passed": True, "results": ans})
    except Exception:
        try:
            conn.send({"passed": False, "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        for obj in objects.values():
            obj.close()
        conn.close()


class Workers:
    def __init__(self, ids, root):
        self.root = Path(root)
        self.ids = list(ids)
        self.processes = []
        ctx = mp.get_context("spawn")
        d = self.root / "workers"
        d.mkdir()
        for k in range(min(4, len(ids))):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=oracle_worker, args=(child, self.ids[k::4], str(root)))
            p.start()
            child.close()
            self.processes.append((p, parent))
            atomic_json(d / f"{k}_PROCESS.json", {"pid": p.pid, "assigned": self.ids[k::4]})

    def run(self, x, label, deadline):
        for p, conn in self.processes:
            conn.send({"op": "solve", "x": x, "label": label, "deadline": deadline})
        results = []
        errors = []
        for p, conn in self.processes:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not conn.poll(remaining):
                    raise TimeoutError("R44_WORKER_WAIT_DEADLINE")
                msg = conn.recv()
            except EOFError:
                msg = {"passed": False, "traceback": "WORKER_EOF"}
            if msg["passed"]:
                results.extend(msg["results"])
            else:
                errors.append(msg)
        if errors:
            raise RuntimeError(errors)
        return sorted(results, key=lambda z: z["sid"])

    def close(self):
        for p, conn in self.processes:
            if p.is_alive():
                try:
                    conn.send({"op": "close"})
                except (EOFError, BrokenPipeError):
                    pass
        for k, (p, conn) in enumerate(self.processes):
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join()
            atomic_json(self.root / "workers" / f"{k}_WAIT.json", {"actual_parent_wait": True, "returncode": p.exitcode, "pid": p.pid})
            conn.close()


@functools.lru_cache(None)
def moment_data():
    sys.path.insert(0, str(R36 / "tools"))
    import r36_probability
    a = r36_probability.moments()
    d = R29 / "05_DRO_MOMENTS"
    raw_unique = sp.csr_matrix(np.load(d / "SUPPORT_UNIQUE.npy"))
    bounds = np.load(d / "EFFECTIVE_INTERVALS.npz")
    return {"a": a, "raw_unique": raw_unique, "lower": bounds["lower"], "upper": bounds["upper"], "raw_min": np.asarray(raw_unique.min(axis=1).toarray()).ravel(), "raw_max": np.asarray(raw_unique.max(axis=1).toarray()).ravel(), "scaled_min": np.asarray(a["phi"].min(axis=1).toarray()).ravel(), "scaled_max": np.asarray(a["phi"].max(axis=1).toarray()).ravel()}


def probability_check(p):
    p = np.asarray(p, float)
    den = 2**52
    numerators = np.rint(p * den).astype(np.int64)
    exact = bool(np.array_equal(numerators.astype(float) / den, p) and sum(map(int, numerators)) == den and (numerators >= 0).all())
    d = moment_data()
    lo, hi = enclosure(d["raw_unique"], p)
    if exact:
        lo = np.maximum(lo, d["raw_min"])
        hi = np.minimum(hi, d["raw_max"])
    raw_margin = min(float(np.min(lo - d["lower"])), float(np.min(d["upper"] - hi)))
    a = d["a"]
    sl, sh = enclosure(a["phi"], p)
    if exact:
        sl = np.maximum(sl, d["scaled_min"])
        sh = np.minimum(sh, d["scaled_max"])
    scaled_margin = min(float(np.min(sl - a["lower"])), float(np.min(a["upper"] - sh)))
    return {"passed": exact and raw_margin >= 0 and scaled_margin >= 0, "exact_simplex": exact, "raw_margin": raw_margin, "scaled_margin": scaled_margin, "all_effective_rows": 45139, "all_original_rows": 322311}


def dyadic(p):
    den = 2**52
    n = np.rint(np.maximum(p, 0) * den).astype(np.int64)
    n[int(np.argmax(n))] += den - sum(map(int, n))
    assert (n >= 0).all()
    return n.astype(float) / den


def certify_probability_cut(p, qbars, ids, out):
    p = np.asarray(p, float)
    p0 = np.load(R42 / "PROBABILITY_INTERIOR_STRICT/P_STAR.npz")["p"]
    pp = dyadic(p)
    ck = probability_check(pp)
    mix = 0.0
    if not ck["passed"]:
        c0 = probability_check(p0)
        assert c0["passed"]
        base_margin = min(c0["raw_margin"], c0["scaled_margin"])
        assert base_margin > 0
        violation = max(0, -ck["raw_margin"], -ck["scaled_margin"])
        mix = min(1.0, max(np.finfo(float).eps, 4 * violation / base_margin))
        for _ in range(12):
            pp = dyadic((1 - mix) * dyadic(p) + mix * p0)
            ck = probability_check(pp)
            if ck["passed"]:
                break
            mix = min(1.0, 2 * mix)
        assert ck["passed"], ck
    eps = sum((Fraction(float(qb)) * max(Fraction(float(p[s])) - Fraction(float(pp[s])), Fraction(0)) for s, qb in zip(ids, qbars)), Fraction(0))
    guard = float(eps)
    if Fraction(guard) < eps:
        guard = np.nextafter(guard, np.inf)
    out.mkdir(parents=True)
    np.savez_compressed(out / "PROBABILITY.npz", p=p, pstar=pp, Qbar=qbars, ids=ids)
    rec = {"passed": True, "epsilon": guard, "mix": mix, "pstar_check": ck, "compensation_exact": [eps.numerator, eps.denominator], "proof": "For exact feasible pstar and 0<=theta<=Qbar, (p-pstar)theta <= sum Qbar max(p-pstar,0)."}
    atomic_json(out / "CERTIFICATE.json", rec)
    return rec


def risk_upper(q, b0, bp, bm):
    d = moment_data()
    a = d["a"]
    ld = np.longdouble
    eps = np.finfo(ld).eps
    scale = np.asarray(a["scale"], ld)
    bp = np.maximum(bp, 0)
    bm = np.maximum(bm, 0)
    wp = np.asarray(bp, ld) / scale
    wm = np.asarray(bm, ld) / scale
    w = wp - wm
    we = 4 * eps * (abs(wp) + abs(wm))
    low, high = enclosure(d["raw_unique"].T, w)
    correction = abs(d["raw_unique"].T).astype(ld) @ we
    low -= correction
    high += correction
    low += ld(b0)
    high += ld(b0)
    low -= eps * (abs(b0) + abs(low))
    shift = max(ld(0), np.max(np.asarray(q, ld) - low))
    newb = float(np.nextafter(float(ld(b0) + shift), np.inf))
    terms = np.asarray(d["upper"], ld) * wp - np.asarray(d["lower"], ld) * wm
    err = 8 * eps * (abs(d["upper"] * wp) + abs(d["lower"] * wm))
    upper = ld(newb) + sum(terms) + sum(err) + (8 * len(terms) + 8) * eps * sum(abs(terms))
    return {"risk_upper": float(np.nextafter(float(upper), np.inf)), "beta0": newb, "plus": np.asarray(bp), "minus": np.asarray(bm), "shift": float(shift), "all400_plane_certified": True}


class Probability:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir()
        self.a = moment_data()["a"]
        d = moment_data()
        self.trlo = self.a["lower"] <= d["scaled_min"]
        self.trhi = self.a["upper"] >= d["scaled_max"]
        self.model = gp.Model("R44_RISK_ROWGEN")
        self.model.Params.OutputFlag = 0
        self.p = self.model.addMVar(400, lb=0, name="p")
        self.simplex = self.model.addConstr(self.p.sum() == 1, name="simplex")
        self.active = {}
        best = Path(read(R41 / "LOCK.json")["best"]["evaluation"]) / "probability/CERTIFICATE.npz"
        z = np.load(best)
        self.add([("U", int(i)) for i in np.flatnonzero((z["plus"] > 0) & ~self.trhi)] + [("L", int(i)) for i in np.flatnonzero((z["minus"] > 0) & ~self.trlo)])

    def add(self, items):
        for kind, i in items:
            if (kind, i) in self.active:
                continue
            row = self.a["phi"].getrow(i)
            expr = gp.LinExpr(row.data.tolist(), [self.p[j].item() for j in row.indices])
            co = self.model.addConstr(expr <= self.a["upper"][i], name=f"U/{i}") if kind == "U" else self.model.addConstr(expr >= self.a["lower"][i], name=f"L/{i}")
            self.active[(kind, i)] = co
        self.model.update()

    def solve(self, q, label, deadline):
        d = self.out / label
        d.mkdir()
        q = np.asarray(q, float)
        normal = max(1.0, float(abs(q).max()))
        self.model.setObjective((q / normal) @ self.p, gp.GRB.MAXIMIZE)
        history = []
        for k in range(200):
            remaining = deadline - time.monotonic() - 20
            if remaining <= 0:
                raise TimeoutError("R44_DEADLINE_PROBABILITY")
            call = d / f"CALL_{k:03d}"
            call.mkdir()
            self.model.Params.Threads = 1
            self.model.Params.Method = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.OptimalityTol = 1e-9
            self.model.Params.SoftMemLimit = 6
            self.model.Params.TimeLimit = remaining
            self.model.Params.OutputFlag = 1
            self.model.Params.LogFile = str(call / "NATIVE.log")
            atomic_json(call / "BEGIN.json", {"epoch": time.time(), "remaining": remaining, "active_sides": len(self.active)})
            call_id = tracked_optimize(self.model, "PROBABILITY_ROWGEN", {"label": label, "rowgen_iteration": k}, {"remaining": remaining, "active_sides": len(self.active)})
            atomic_json(call / "END.json", {"epoch": time.time(), "status": self.model.Status, "runtime": self.model.Runtime, "iterations": self.model.IterCount, "work": self.model.Work})
            if self.model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(("PROBABILITY_NOT_OPTIMAL", self.model.Status))
            p = np.asarray(self.p.X)
            ex = self.a["phi"] @ p
            lv = self.a["lower"] - ex
            uv = ex - self.a["upper"]
            violations = sorted([(float(v), "L", int(i)) for i, v in enumerate(lv) if v > 1e-9] + [(float(v), "U", int(i)) for i, v in enumerate(uv) if v > 1e-9], reverse=True)
            history.append({"iteration": k, "active_sides": len(self.active), "violations": len(violations), "maximum": violations[0][0] if violations else 0, "runtime": self.model.Runtime, "scientific_optimizer_call_id": call_id})
            atomic_json(d / "HISTORY.json", history)
            if not violations:
                break
            new = [(kind, i) for _, kind, i in violations if (kind, i) not in self.active][:500]
            assert new
            self.add(new)
        else:
            raise RuntimeError("PROBABILITY_ROWGEN_NOT_CLOSED")
        bp = np.zeros(45139)
        bm = np.zeros(45139)
        for (kind, i), row in self.active.items():
            if kind == "U":
                bp[i] = max(0, row.Pi) * normal
            else:
                bm[i] = max(0, -row.Pi) * normal
        cert = risk_upper(q, float(self.simplex.Pi) * normal, bp, bm)
        strict = certify_probability_cut(p, np.asarray(GLOBAL_QBARS), GLOBAL_POOL, d / "STRICT_CUT")
        np.savez_compressed(d / "CERTIFICATE.npz", p=p, q=q, beta0=cert["beta0"], plus=bp, minus=bm, scale=self.a["scale"])
        raw = self.a["raw"]
        rb = self.a["raw_bounds"]
        raw_ex = p @ raw
        raw_residual = max(float(np.max(rb["lower"] - raw_ex, initial=0)), float(np.max(raw_ex - rb["upper"], initial=0)), 0.0)
        rec = {"risk_primal": float(p @ q), "risk_upper": cert["risk_upper"], "signed_gap": cert["risk_upper"] - float(p @ q), "raw_residual": raw_residual, "simplex_residual": abs(float(p.sum()) - 1), "min_p": float(p.min()), "active_sides": len(self.active), "strict_probability_cut": strict, "all400_support": True}
        atomic_json(d / "CERTIFICATE.json", rec)
        return p, cert, strict, rec

    def close(self):
        self.model.dispose()


GLOBAL_POOL = []
GLOBAL_QBARS = []


def point_repair(raw, out, deadline):
    out = Path(out)
    resolved = out.resolve()
    if resolved != OUT.resolve() and OUT.resolve() not in resolved.parents:
        raise ValueError("point_repair output outside isolated OUT")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "POINTS.npz").exists():
        raise FileExistsError("POINTS already exists")
    A = sp.load_npz(R41 / "identity/FIRST_A.npz").tocsr()
    z = dict(np.load(R41 / "identity/FIRST_ARRAYS.npz"))
    raw = np.asarray(raw, float)
    before = row_audit(A, z["rhs"], z["sense"], z["lo"], z["hi"], raw)
    assert max(before.values()) <= 1e-9, before
    y = np.clip(raw, z["lo"], z["hi"])
    anchors = (z["lo"] == 0) & (abs(raw) <= 1e-9)
    y[anchors] = 0
    after = row_audit(A, z["rhs"], z["sense"], z["lo"], z["hi"], y)
    projection = None
    if max(after.values()) > 1e-9:
        remaining = deadline - time.monotonic() - 20
        assert remaining > 0
        scale = 1e-9
        ld = np.longdouble
        rhs = np.asarray((np.asarray(z["rhs"], ld) - A.astype(ld) @ np.asarray(raw, ld)) / ld(scale), float)
        low = np.maximum(np.asarray((np.asarray(z["lo"], ld) - raw) / ld(scale), float), -100)
        high = np.minimum(np.asarray((np.asarray(z["hi"], ld) - raw) / ld(scale), float), 100)
        low[anchors] = -raw[anchors] / scale
        high[anchors] = low[anchors]
        m = gp.Model("R44_POINT_REPAIR")
        dv = m.addMVar(len(raw), lb=low, ub=high)
        radius = m.addVar(lb=0, ub=100)
        m.addMConstr(A, dv, z["sense"], rhs)
        m.addConstr(dv <= radius)
        m.addConstr(-dv <= radius)
        m.setObjective(radius)
        m.Params.Threads = 1; m.Params.Method = 1; m.Params.FeasibilityTol = 1e-9; m.Params.OptimalityTol = 1e-9; m.Params.SoftMemLimit = 3; m.Params.TimeLimit = remaining; m.Params.LogFile = str(out / "PROJECTION_NATIVE.log")
        call_id = tracked_optimize(m, "POINT_REPAIR", {"directory": str(out)}, {"remaining": remaining})
        assert m.Status == gp.GRB.OPTIMAL
        y = np.asarray(np.asarray(raw, ld) + ld(scale) * np.asarray(dv.X, ld), float)
        y = np.clip(y, z["lo"], z["hi"]); y[anchors] = 0
        projection = {"status": m.Status, "runtime": m.Runtime, "radius_scaled": radius.X, "scientific_optimizer_call_id": call_id}
        m.dispose()
        after = row_audit(A, z["rhs"], z["sense"], z["lo"], z["hi"], y)
    displacement = float(abs(y - raw).max())
    assert max(after.values()) <= 1e-9 and displacement <= 1e-7
    np.savez_compressed(out / "POINTS.npz", raw=raw, accepted=y)
    rec = {"before": before, "after": after, "displacement": displacement, "projection": projection, "raw_sha": hashlib.sha256(raw.tobytes()).hexdigest(), "accepted_sha": hashlib.sha256(y.tobytes()).hexdigest()}
    atomic_json(out / "AUDIT.json", rec)
    return y, rec


def append_rows(state, model, v, rows, rhs, sense, names, cbasis_value=0):
    if rows.shape[0] == 0:
        return
    vb = np.asarray(model.getAttr("VBasis"), int)
    cb = np.asarray(model.getAttr("CBasis"), int)
    model.addMConstr(rows, v, sense, rhs)
    model.update()
    model.setAttr("ConstrName", model.getConstrs()[-rows.shape[0]:], names)
    model.update()
    model.setAttr("VBasis", model.getVars(), vb.tolist())
    model.setAttr("CBasis", model.getConstrs(), np.r_[cb, np.full(rows.shape[0], cbasis_value, dtype=int)].tolist())
    model.update()
    state["A"] = sp.vstack([state["A"], rows], format="csr")
    state["rhs"] = np.r_[state["rhs"], rhs]
    state["sense"] = np.r_[state["sense"], sense]
    state["names"]["rows"].extend(list(names))


def add_scenario_cut(state, model, v, sid, alpha, beta, name, all_fingerprints):
    k = GLOBAL_POOL.index(sid)
    nz = np.flatnonzero(beta)
    cols = np.r_[nz, FIRST_N + k]
    data = np.r_[-beta[nz], 1.0]
    row = sp.csr_matrix((data, (np.zeros(len(cols), dtype=int), cols)), shape=(1, len(state["cost"])))
    fp = row_fingerprint(row, alpha, ">")
    if fp in all_fingerprints:
        return False, fp
    append_rows(state, model, v, row, np.array([alpha]), np.array([">"], dtype="<U1"), [name])
    all_fingerprints.add(fp)
    return True, fp


def add_probability_cut(state, model, v, p, guard, name, probability_fingerprints):
    cols = np.r_[np.arange(FIRST_N, FIRST_N + len(GLOBAL_POOL)), len(state["cost"]) - 1]
    data = np.r_[-p[GLOBAL_POOL], 1.0]
    row = sp.csr_matrix((data, (np.zeros(len(cols), dtype=int), cols)), shape=(1, len(state["cost"])))
    fp = row_fingerprint(row, -guard, ">")
    if fp in probability_fingerprints:
        return False, fp
    append_rows(state, model, v, row, np.array([-guard]), np.array([">"], dtype="<U1"), [name])
    probability_fingerprints.add(fp)
    return True, fp


def solve_master(state, model, v, directory, deadline, history, best_lb, pool_closed_before, require_basis=False):
    directory.mkdir(parents=True)
    remaining = deadline - time.monotonic() - 15
    if remaining <= 0:
        raise TimeoutError("R44_DEADLINE_MASTER")
    master_config(model, directory / "NATIVE.log", remaining)
    atomic_json(directory / "BEGIN.json", {"epoch": time.time(), "rows": model.NumConstrs, "columns": model.NumVars, "nnz": model.NumNZs, "remaining": remaining, "pool_closed_before": pool_closed_before})
    t = time.monotonic(); call_id = tracked_optimize(model, "MASTER", {"directory": str(directory)}, {"remaining": remaining, "rows": model.NumConstrs, "nnz": model.NumNZs}); native_wall = time.monotonic() - t
    end = {"epoch": time.time(), "status": model.Status, "runtime": model.Runtime, "iterations": model.IterCount, "work": model.Work, "max_mem_GB": model.MaxMemUsed, "native_wall": native_wall, "basis_accepted": basis_accepted(directory / "NATIVE.log"), "basis_acceptance_required_phrase": "LP warm-start: use basis", "scientific_optimizer_call_id": call_id}
    atomic_json(directory / "END.json", end)
    if require_basis and not end["basis_accepted"]:
        raise RuntimeError("BASIS_NOT_ACCEPTED")
    try:
        pi = np.asarray(model.getAttr("Pi"))
        lb, cert_pi = lagrangian(state["A"], state["rhs"], state["sense"], state["cost"], state["lo"], state["hi"], pi)
        best_lb = max(best_lb, lb)
        np.savez_compressed(directory / "DUAL_CERTIFICATE.npz", Pi=cert_pi, RC=np.asarray(model.getAttr("RC")))
        atomic_json(directory / "LOWER_CERTIFICATE.json", {"certified_LB": lb, "best_certified_LB": best_lb, "matrix_sha": matrix_sha(state["A"]), "rows": state["A"].shape[0], "columns": state["A"].shape[1], "nnz": state["A"].nnz, "pool_closed": False})
    except Exception:
        lb = None
        cert_pi = None
    if model.Status != gp.GRB.OPTIMAL:
        return {"completed": False, "status": model.Status, "LB": lb, "best_LB": best_lb, "end": end}, best_lb
    x = np.asarray(v.X)
    pi = np.asarray(model.getAttr("Pi"))
    rc = np.asarray(model.getAttr("RC"))
    vb = np.asarray(model.getAttr("VBasis"), int)
    cb = np.asarray(model.getAttr("CBasis"), int)
    audit = row_audit(state["A"], state["rhs"], state["sense"], state["lo"], state["hi"], x)
    rv = directed_row_violations(state["A"], state["rhs"], state["sense"], x)
    sm = np.asarray([scenario_name(n) for n in state["names"]["rows"]], dtype=bool)
    pm = np.asarray([probability_name(n) for n in state["names"]["rows"]], dtype=bool)
    fm = ~(sm | pm)
    by_category = {
        "physical_original_max_violation": float(max(np.longdouble(0), np.max(rv[fm], initial=np.longdouble(0)))),
        "probability_max_violation": float(max(np.longdouble(0), np.max(rv[pm], initial=np.longdouble(0)))),
        "active_scenario_max_violation": float(max(np.longdouble(0), np.max(rv[sm], initial=np.longdouble(0)))),
        "bound_max_violation": audit["bound"],
        "all_finite": bool(np.isfinite(rv).all() and np.isfinite(x).all()),
    }
    by_category["passed"] = bool(by_category["all_finite"] and by_category["physical_original_max_violation"] <= 1e-9 and by_category["probability_max_violation"] <= 1e-9 and by_category["bound_max_violation"] <= 1e-9 and by_category["active_scenario_max_violation"] <= 1e-7)
    np.savez_compressed(directory / "SOLUTION.npz", x=x, Pi=pi, RC=rc)
    np.savez_compressed(directory / "BASIS.npz", VBasis=vb, CBasis=cb)
    atomic_json(directory / "PRIMAL_AUDIT.json", audit)
    atomic_json(directory / "PRIMAL_AUDIT_BY_CATEGORY.json", by_category)
    if not by_category["passed"]:
        raise RuntimeError(("NUMERICAL_CORRECTNESS_FAILED", by_category))
    return {"completed": True, "status": model.Status, "x": x, "Pi": pi, "RC": rc, "VBasis": vb, "CBasis": cb, "LB": lb, "best_LB": best_lb, "audit": audit, "end": end}, best_lb


def save_checkpoint(state, solution, basis, outer, pool_closed, best_lb, audited_ub, counters):
    d = OUT / "checkpoints"
    d.mkdir(exist_ok=True)
    tmp = d / f"CHECKPOINT_{outer:03d}.json.tmp"
    if solution:
        z = np.load(solution)
        assert len(z["Pi"]) == state["A"].shape[0]
    if basis:
        bz = np.load(basis)
        assert len(bz["CBasis"]) == state["A"].shape[0]
        assert len(bz["VBasis"]) == state["A"].shape[1]
    payload = {"outer": outer, "epoch": time.time(), "rows": state["A"].shape[0], "columns": state["A"].shape[1], "nnz": state["A"].nnz, "active_scenario": sum(scenario_name(n) for n in state["names"]["rows"]), "dormant_scenario": state["dA"].shape[0], "pool_closed": pool_closed, "best_certified_LB": best_lb, "best_audited_UB": audited_ub, "counters": counters, "matrix_sha": matrix_sha(state["A"]), "solution": str(solution) if solution else None, "basis": str(basis) if basis else None, "row_order_sha": hashlib.sha256("\n".join(state["names"]["rows"]).encode()).hexdigest()}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, d / f"CHECKPOINT_{outer:03d}.json")


def maybe_recompress(state, model, v, solution, outer):
    pi = solution["Pi"]
    cb = solution["CBasis"]
    scenario_ids = np.array([i for i, n in enumerate(state["names"]["rows"]) if scenario_name(n)], dtype=int)
    removable = scenario_ids[(pi[scenario_ids] == 0.0) & (cb[scenario_ids] == 0)]
    candidate_nnz = int(state["A"][removable].nnz) if len(removable) else 0
    ratio = candidate_nnz / max(1, state["A"].nnz)
    rec = {"outer": outer, "active_scenario": len(scenario_ids), "candidate_rows": len(removable), "candidate_nnz": candidate_nnz, "active_nnz": state["A"].nnz, "ratio": ratio, "threshold": 0.30, "executed": False}
    if ratio < 0.30 or not len(removable):
        atomic_json(OUT / f"outer_{outer:03d}/RECOMPRESSION.json", rec)
        return model, v, solution, rec
    before_lb, _ = lagrangian(state["A"], state["rhs"], state["sense"], state["cost"], state["lo"], state["hi"], pi)
    keep = np.ones(state["A"].shape[0], dtype=bool); keep[removable] = False
    removed_A = state["A"][removable].tocsr(); removed_rhs = state["rhs"][removable]; removed_sense = state["sense"][removable]; removed_names = [state["names"]["rows"][i] for i in removable]
    vb = solution["VBasis"]
    constraints = model.getConstrs()
    model.remove([constraints[i] for i in removable])
    model.update()
    state["A"] = state["A"][keep].tocsr(); state["rhs"] = state["rhs"][keep]; state["sense"] = state["sense"][keep]; state["names"]["rows"] = [n for i, n in enumerate(state["names"]["rows"]) if keep[i]]
    state["dA"] = sp.vstack([state["dA"], removed_A], format="csr"); state["drhs"] = np.r_[state["drhs"], removed_rhs]; state["dsense"] = np.r_[state["dsense"], removed_sense]; state["dnames"].extend(removed_names); state["dsource"].extend([None] * len(removable))
    new_pi = pi[keep]; new_cb = cb[keep]
    model.setAttr("VBasis", model.getVars(), vb.tolist()); model.setAttr("CBasis", model.getConstrs(), new_cb.tolist()); model.update()
    after_lb, _ = lagrangian(state["A"], state["rhs"], state["sense"], state["cost"], state["lo"], state["hi"], new_pi)
    assert abs(before_lb - after_lb) <= 1e-5, (before_lb, after_lb)
    solution = dict(solution, Pi=new_pi, CBasis=new_cb)
    od = OUT / f"outer_{outer:03d}"
    post_solution = od / "POST_RECOMPRESSION_SOLUTION.npz"
    post_basis = od / "POST_RECOMPRESSION_BASIS.npz"
    np.savez_compressed(post_solution, x=solution["x"], Pi=new_pi, RC=solution["RC"])
    np.savez_compressed(post_basis, VBasis=vb, CBasis=new_cb)
    post_identity = {
        "rows": state["A"].shape[0], "columns": state["A"].shape[1], "nnz": state["A"].nnz,
        "matrix_sha": matrix_sha(state["A"]),
        "row_order_sha": hashlib.sha256("\n".join(state["names"]["rows"]).encode()).hexdigest(),
        "Pi_length": len(new_pi), "CBasis_length": len(new_cb), "VBasis_length": len(vb),
        "solution": record(post_solution), "basis": record(post_basis),
    }
    assert post_identity["Pi_length"] == post_identity["CBasis_length"] == post_identity["rows"]
    assert post_identity["VBasis_length"] == post_identity["columns"]
    atomic_json(od / "POST_RECOMPRESSION_MATRIX_IDENTITY.json", post_identity)
    atomic_json(od / "POST_RECOMPRESSION_LB_TRANSFER.json", {"passed": abs(before_lb-after_lb) <= 1e-5, "before_LB": before_lb, "after_LB": after_lb, "removed_rows": len(removable), "removed_rows_have_exact_zero_Pi": bool(np.all(pi[removable] == 0.0)), "removed_rows_have_CBasis_zero": bool(np.all(cb[removable] == 0))})
    rec.update({"executed": True, "before_LB": before_lb, "after_LB": after_lb, "after_rows": state["A"].shape[0], "after_nnz": state["A"].nnz, "dormant_after": state["dA"].shape[0], "checkpoint_solution": str(post_solution), "checkpoint_basis": str(post_basis)})
    atomic_json(OUT / f"outer_{outer:03d}/RECOMPRESSION.json", rec)
    return model, v, solution, rec


def state_identity(state):
    return {"matrix_sha": matrix_sha(state["A"]), "rows": state["A"].shape[0], "columns": state["A"].shape[1], "nnz": state["A"].nnz,
            "rows_sha": hashlib.sha256(json.dumps(state["names"]["rows"]).encode()).hexdigest(),
            "columns_sha": hashlib.sha256(json.dumps(state["names"]["columns"]).encode()).hexdigest(),
            "objective_bounds_sha": hashlib.sha256(b"".join(np.asarray(state[k]).tobytes() for k in ("cost", "lo", "hi"))).hexdigest()}


def freeze_solved(state, solution, lb):
    import copy
    assert len(solution["Pi"]) == len(solution["CBasis"]) == state["A"].shape[0]
    assert len(solution["x"]) == len(solution["RC"]) == len(solution["VBasis"]) == state["A"].shape[1]
    return {"state": copy.deepcopy(state), "solution": {k: np.array(solution[k], copy=True) for k in ("x", "Pi", "RC", "VBasis", "CBasis")}, "identity": state_identity(state), "LB": lb}


def export_state(directory, state, solved=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    identity = state_identity(state)
    if solved is not None:
        if solved["identity"] != identity:
            raise RuntimeError("EXPORT_SOLVED_MATRIX_IDENTITY_MISMATCH")
        z = solved["solution"]
        assert len(z["Pi"]) == len(z["CBasis"]) == identity["rows"]
        assert len(z["x"]) == len(z["RC"]) == len(z["VBasis"]) == identity["columns"]
        np.savez_compressed(directory / "SOLUTION.npz", x=z["x"], Pi=z["Pi"], RC=z["RC"])
        np.savez_compressed(directory / "BASIS.npz", VBasis=z["VBasis"], CBasis=z["CBasis"])
        lb, _ = lagrangian(state["A"], state["rhs"], state["sense"], state["cost"], state["lo"], state["hi"], z["Pi"])
        atomic_json(directory / "LB_CERTIFICATE.json", {"replayed_LB": lb, "retained_LB": solved["LB"], "identity": identity})
    sp.save_npz(directory / "A.npz", state["A"])
    np.savez_compressed(directory / "ARRAYS.npz", **{k: state[k] for k in ("rhs", "sense", "cost", "lo", "hi")})
    atomic_json(directory / "NAMES.json", state["names"])
    sp.save_npz(directory / "DORMANT_A.npz", state["dA"])
    np.savez_compressed(directory / "DORMANT_ARRAYS.npz", rhs=state["drhs"], sense=state["dsense"])
    atomic_json(directory / "DORMANT_NAMES.json", {"rows": state["dnames"], "source": state["dsource"]})
    atomic_json(directory / "IDENTITY.json", dict(identity, solved=solved is not None, pending_has_no_Pi_or_basis=solved is None))


def sealed_manifest(directory):
    directory = Path(directory)
    files = [p for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SEALED_MANIFEST.json" and "delivery" not in p.parts and not p.name.endswith((".tmp", ".lock"))]
    atomic_json(directory / "SEALED_MANIFEST.json", {"files": [record(p) for p in files], "excluded": ["SEALED_MANIFEST.json", "delivery", "*.tmp", "*.lock"]})
    return all(sha(Path(r["path"])) == r["sha256"] for r in read(directory / "SEALED_MANIFEST.json")["files"])


def audit_historical_ub(pool):
    source = R42 / "C0_UPPER_REPLAY.json"
    x = read(source)
    point = Path(x.get("point_source", {}).get("path", ""))
    checks = {}
    checks["point_source_exists"] = point.is_file()
    checks["point_source_sha_matches"] = checks["point_source_exists"] and sha(point) == x.get("point_source", {}).get("sha256")
    accepted_sha = None
    q = None
    if checks["point_source_exists"]:
        z = np.load(point)
        accepted_sha = hashlib.sha256(np.asarray(z["x"], float).tobytes()).hexdigest() if "x" in z.files else None
        q = np.asarray(z["qU"], float) if "qU" in z.files else None
    checks["accepted_x_sha_matches"] = accepted_sha == x.get("accepted_x_sha")
    recourse = x.get("recourse", [])
    checks["complete_93_recourse_entries"] = len(recourse) == 93 and {int(r["sid"]) for r in recourse} == set(pool)
    recourse_point_hashes = []
    recourse_sources_valid = True
    for r in recourse:
        p = Path(r.get("source", {}).get("path", ""))
        if not p.is_file() or sha(p) != r.get("source", {}).get("sha256"):
            recourse_sources_valid = False
            continue
        rz = np.load(p)
        if "x" not in rz.files:
            recourse_sources_valid = False
        else:
            recourse_point_hashes.append(hashlib.sha256(np.asarray(rz["x"], float).tobytes()).hexdigest())
    checks["recourse_sources_and_same_point"] = recourse_sources_valid and len(recourse_point_hashes) == 93 and set(recourse_point_hashes) == {accepted_sha}
    checks["same_frozen93_identity"] = checks["complete_93_recourse_entries"]
    checks["remaining_307_Q_zero_same_definition"] = q is not None and len(q) == 400 and int(np.count_nonzero(q)) <= 93 and all(q[s] >= 0 for s in pool)
    checks["risk_dual_marked_independently_certified"] = x.get("risk_dual_independently_certified") is True
    # C0_UPPER_REPLAY does not separately identify the first-stage F enclosure
    # and risk-upper certificate files needed by this new inheritance rule.
    checks["explicit_F_and_risk_upper_source_records"] = False
    inherited = all(checks.values())
    audit = {
        "passed_for_inheritance": inherited,
        "checks": checks,
        "source": record(source),
        "source_point": record(point) if point.is_file() else None,
        "source_numerically_audited_value": x.get("numerically_audited"),
        "historical_uninherited_reference": 14538.864206851838,
        "inherited_value": float(x["numerically_audited"]) if inherited else None,
        "inherited_point_sha": accepted_sha if inherited else None,
        "strict_certified_UB": None,
        "full400_integer_UB": False,
        "reason": "Inherited only if accepted_x, all 93 same-point recourse, F/risk sources, frozen identity and the 307-zero definition are all complete. Missing explicit F/risk source records keeps best_audited_UB null." if not inherited else "All registered historical numerically-audited sources passed; this is not a strict or full400 integer UB.",
    }
    atomic_json(OUT / "HISTORICAL_UB_INHERITANCE_AUDIT.json", audit)
    return audit


def run_worker():
    global GLOBAL_POOL, GLOBAL_QBARS
    start_epoch = time.time(); start_mono = time.monotonic(); deadline = start_mono + WALL_LIMIT
    atomic_json(OUT / "BEGIN.json", {"epoch": start_epoch, "monotonic": start_mono, "pid": os.getpid(), "wall_limit": WALL_LIMIT, "max_batches": MAX_BATCHES, "budget_includes": ["restore", "load", "oracle_initialization", "solve", "scan", "certificates", "runtime_writes"]})
    state = restore_endpoint(OUT)
    GLOBAL_POOL = list(map(int, read(R41 / "LOCK.json")["pool"]))
    GLOBAL_QBARS = [read(R41 / "oracles" / f"{sid:04d}" / "QBAR_PROOF.json")["Qbar"] for sid in GLOBAL_POOL]
    assert len(GLOBAL_POOL) == 93 and len(set(GLOBAL_POOL)) == 93
    atomic_json(OUT / "PARAMETERS.json", {"Gurobi": ".".join(map(str, gp.gurobi.version())), "master": {"Threads": 1, "Method": 1, "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9, "SoftMemLimit_GB": 3, "LPWarmStart": 1}, "recourse_workers": 4, "recourse_threads": 1, "probability_threads": 1, "wall_limit": WALL_LIMIT, "max_started_batches": MAX_BATCHES, "dormant_scan": 1e-7, "cut_violation": 1e-6, "recompression_gate": 0.30})
    assert gp.gurobi.version() == (12, 0, 3)
    tests = {"endpoint_counts": True, "endpoint_LB": True, "pool93": True, "probability_support400": moment_data()["a"]["raw"].shape[0] == 400, "strict_pstar": probability_check(np.load(R42 / "PROBABILITY_INTERIOR_STRICT/P_STAR.npz")["p"])["passed"], "no_integer_variables": True}
    atomic_json(OUT / "TESTS.json", {"passed": all(tests.values()), "tests": tests})
    model, v = build_master(state, OUT / "INITIAL_NATIVE.log")
    assert model.NumIntVars == 0
    all_scenario_fps = {row_fingerprint(state["A"].getrow(i), state["rhs"][i], state["sense"][i]) for i, n in enumerate(state["names"]["rows"]) if scenario_name(n)}
    all_scenario_fps |= {row_fingerprint(state["dA"].getrow(i), state["drhs"][i], state["dsense"][i]) for i in range(state["dA"].shape[0])}
    probability_fps = {row_fingerprint(state["A"].getrow(i), state["rhs"][i], state["sense"][i]) for i, n in enumerate(state["names"]["rows"]) if probability_name(n)}
    workers = Workers(GLOBAL_POOL, OUT)
    probability = Probability(OUT / "probability")
    best_lb = state["LB"]
    historical_ub = audit_historical_ub(GLOBAL_POOL)
    best_audited_ub = historical_ub["inherited_value"]
    raw = np.asarray(state["x"])
    current_solution = {"x": state["x"], "Pi": state["Pi"], "RC": state["RC"], "VBasis": state["VBasis"], "CBasis": state["CBasis"]}
    last_solved = freeze_solved(state, current_solution, best_lb)
    history = []
    counters = {"started_batches": 0, "complete_Q_batches": 0, "complete_outer_iterations": 0, "new_recourse_solves": 0, "cache_hits": 0, "master_calls": 0, "probability_calls": 0, "readded_rows": 0, "recompressed_rows": 0}
    timing = {"endpoint_restore": read(OUT / "MODEL_AND_POOL_IDENTITY.json")["restore_wall"], "point_repair": 0.0, "recourse_batch_wall": 0.0, "probability": 0.0, "master_native_wall": 0.0, "dormant_scan": 0.0, "readd": 0.0, "recompression": 0.0, "serialization": 0.0}
    stop_reason = "RUNNING"
    milestones = {"0": {"certified_LB": best_lb, "available_wall": time.monotonic() - start_mono}, "300": None, "600": None, "900": None}
    seen_points = set()
    try:
        for outer in range(MAX_BATCHES):
            remaining = deadline - time.monotonic()
            if remaining <= 90:
                stop_reason = "WALL_BUDGET_NO_NEW_BATCH"
                break
            od = OUT / f"outer_{outer:03d}"; od.mkdir()
            counters["attempted_outer_iterations"] = counters.get("attempted_outer_iterations", 0) + 1
            t = time.monotonic(); accepted, repair_rec = point_repair(raw[:FIRST_N], od / "POINT_REPAIR", deadline); timing["point_repair"] += time.monotonic() - t
            point_sha = repair_rec["accepted_sha"]
            if point_sha in seen_points and outer > 0:
                atomic_json(od / "STAGNATION.json", {"reason": "REPEATED_ACCEPTED_POINT", "point_sha": point_sha})
            seen_points.add(point_sha)
            counters["started_batches"] += 1
            t = time.monotonic(); results = workers.run(accepted, f"OUTER_{outer:03d}", deadline); timing["recourse_batch_wall"] += time.monotonic() - t
            if len(results) != 93 or {r["sid"] for r in results} != set(GLOBAL_POOL):
                stop_reason = "PARTIAL_Q_BATCH"
                atomic_json(od / "PARTIAL_Q.json", {"completed": len(results), "expected": 93})
                break
            counters["complete_Q_batches"] += 1
            counters["cache_hits"] += sum(bool(r["cache_hit"]) for r in results)
            counters["new_recourse_solves"] += sum(int(r["new_optimize_calls"]) for r in results)
            q = np.zeros(400)
            new_scenario = []
            for rr in results:
                q[rr["sid"]] = rr["qU_audited"]
                z = np.load(rr["cut"]); alpha = float(z["alpha"]); beta = z["beta"]
                theta_index = GLOBAL_POOL.index(rr["sid"])
                violation = float(np.longdouble(alpha) + np.asarray(beta, np.longdouble) @ np.asarray(raw[:FIRST_N], np.longdouble) - np.longdouble(raw[FIRST_N + theta_index]))
                if violation > 1e-6:
                    name = f"r44_benders/{rr['sid']}/{outer}"
                    added, cut_id = add_scenario_cut(state, model, v, rr["sid"], alpha, beta, name, all_scenario_fps)
                    if added:
                        new_scenario.append({"sid": rr["sid"], "name": name, "cut_id": cut_id, "violation": violation, "source": record(rr["cut"])})
            t = time.monotonic(); p, risk, strict, risk_rec = probability.solve(q, f"OUTER_{outer:03d}", deadline); timing["probability"] += time.monotonic() - t; counters["probability_calls"] += 1
            pname = f"r44_probability/{outer}"
            p_added, p_cut_id = add_probability_cut(state, model, v, p, strict["epsilon"], pname, probability_fps)
            firstA = sp.load_npz(R41 / "identity/FIRST_A.npz"); firstz = dict(np.load(R41 / "identity/FIRST_ARRAYS.npz"))
            _, fup = enclosure(sp.csr_matrix(firstz["cost"][None, :]), accepted)
            audited_ub = float(np.nextafter(float(fup[0]) + risk["risk_upper"], np.inf))
            if best_audited_ub is None or audited_ub < best_audited_ub:
                best_audited_ub = audited_ub
            np.savez_compressed(od / "EVALUATION.npz", accepted_x=accepted, qU_audited=q, p=p)
            atomic_json(od / "EVALUATION.json", {"point_sha": point_sha, "F_upper": float(fup[0]), "risk": risk_rec, "audited_UB": audited_ub, "best_audited_UB": best_audited_ub, "certified_UB": None, "scenarios": results, "new_scenario_cuts": new_scenario, "probability_cut": {"added": p_added, "cut_id": p_cut_id, "name": pname, "guard": strict["epsilon"]}, "same_point": True, "missing_307_Q_zero": True})
            pool_closed = False
            solve_index = 0
            outer_master = []
            while time.monotonic() < deadline - 15:
                md = od / f"MASTER_{solve_index:03d}"
                result, best_lb = solve_master(state, model, v, md, deadline, history, best_lb, pool_closed, require_basis=(counters["master_calls"] == 0))
                counters["master_calls"] += 1; timing["master_native_wall"] += result["end"]["native_wall"]
                outer_master.append({k: vv for k, vv in result.items() if k not in ("x", "Pi", "RC", "VBasis", "CBasis")})
                if not result["completed"]:
                    stop_reason = "MASTER_INCOMPLETE"
                    break
                current_solution = result
                last_solved = freeze_solved(state, current_solution, best_lb)
                scan_t = time.monotonic()
                local, violations = scan_dormant(state["dA"], state["drhs"], state["dsense"], result["x"])
                timing["dormant_scan"] += time.monotonic() - scan_t
                np.savez_compressed(md / "DORMANT_SCAN.npz", upper_violation=violations, readd_local=local)
                if len(local) == 0:
                    pool_closed = True
                    atomic_json(md / "POOL_CLOSED.json", {"passed": True, "threshold": 1e-7, "dormant": state["dA"].shape[0], "max_upper_violation": float(np.max(violations)) if len(violations) else None})
                    break
                tadd = time.monotonic()
                rows = state["dA"][local].tocsr(); rhs = state["drhs"][local]; sense = state["dsense"][local]; names = [state["dnames"][i] for i in local]
                append_rows(state, model, v, rows, rhs, sense, names)
                keep = np.ones(state["dA"].shape[0], dtype=bool); keep[local] = False
                state["dA"] = state["dA"][keep].tocsr(); state["drhs"] = state["drhs"][keep]; state["dsense"] = state["dsense"][keep]; state["dnames"] = [n for i, n in enumerate(state["dnames"]) if keep[i]]; state["dsource"] = [n for i, n in enumerate(state["dsource"]) if keep[i]]
                counters["readded_rows"] += len(local); timing["readd"] += time.monotonic() - tadd
                atomic_json(md / "READD.json", {"count": len(local), "names": names, "dormant_remaining": state["dA"].shape[0]})
                solve_index += 1
                if solve_index >= 20:
                    stop_reason = "READD_LOOP_LIMIT"
                    break
            atomic_json(od / "MASTER_HISTORY.json", outer_master)
            if not pool_closed:
                if stop_reason == "RUNNING": stop_reason = "POOL_NOT_CLOSED"
                break
            counters["complete_outer_iterations"] += 1
            t = time.monotonic(); model, v, current_solution, recomp = maybe_recompress(state, model, v, current_solution, outer); timing["recompression"] += time.monotonic() - t
            if recomp["executed"]: counters["recompressed_rows"] += recomp["candidate_rows"]
            last_solved = freeze_solved(state, current_solution, best_lb)
            raw = np.asarray(current_solution["x"])
            checkpoint_solution = recomp.get("checkpoint_solution", str(od / f"MASTER_{solve_index:03d}/SOLUTION.npz"))
            checkpoint_basis = recomp.get("checkpoint_basis", str(od / f"MASTER_{solve_index:03d}/BASIS.npz"))
            save_checkpoint(state, checkpoint_solution, checkpoint_basis, outer, True, best_lb, best_audited_ub, counters)
            row = {"outer": outer, "wall": time.monotonic() - start_mono, "point_sha": point_sha, "complete_Q": True, "pool_closed": True, "new_scenario_cuts": len(new_scenario), "new_probability_cut": int(p_added), "master_calls": len(outer_master), "active_rows": state["A"].shape[0], "active_nnz": state["A"].nnz, "active_scenario": sum(scenario_name(n) for n in state["names"]["rows"]), "dormant_scenario": state["dA"].shape[0], "best_certified_LB": best_lb, "best_audited_UB": best_audited_ub, "certified_UB": None, "recompressed": recomp["executed"]}
            history.append(row)
            with (OUT / "ROOT_HISTORY.jsonl").open("a") as f: f.write(json.dumps(row, sort_keys=True) + "\n")
            atomic_json(OUT / "LIVE.json", row)
            for mark in (300, 600, 900):
                if milestones[str(mark)] is None and row["wall"] >= mark:
                    milestones[str(mark)] = {"certified_LB": best_lb, "available_wall": row["wall"]}
            if not new_scenario and not p_added:
                stop_reason = "STAGNATION_DIAGNOSTIC"
                break
        else:
            stop_reason = "BATCH_LIMIT"
        if stop_reason == "RUNNING": stop_reason = "WALL_BUDGET"
    except TimeoutError as e:
        stop_reason = "WALL_BUDGET_PARTIAL"
        atomic_json(OUT / "PARTIAL.json", {"reason": str(e), "traceback": traceback.format_exc()})
    except Exception as exc:
        msg = str(exc)
        if "BASIS_NOT_ACCEPTED" in msg:
            stop_reason = "BASIS_NOT_ACCEPTED"
        elif "NUMERICAL_CORRECTNESS_FAILED" in msg:
            stop_reason = "NUMERICAL_CORRECTNESS_FAILED"
        elif isinstance(exc, (OSError, EOFError, BrokenPipeError)):
            stop_reason = "INFRASTRUCTURE_FAILED"
        else:
            stop_reason = "CORRECTNESS_FAILED"
        atomic_json(OUT / "FAILURE.json", {"traceback": traceback.format_exc()})
    finally:
        solve_end_epoch = time.time(); solve_end_mono = time.monotonic()
        workers.close(); probability.close()
        serialize_start = time.monotonic()
        final = OUT / "final_payload"; final.mkdir(exist_ok=True)
        export_state(OUT / "LAST_SOLVED", last_solved["state"], last_solved)
        export_state(OUT / "WORKING_PENDING", state)
        sp.save_npz(final / "FINAL_A.npz", state["A"])
        np.savez_compressed(final / "FINAL_ARRAYS.npz", rhs=state["rhs"], sense=state["sense"], cost=state["cost"], lo=state["lo"], hi=state["hi"])
        atomic_json(final / "FINAL_NAMES.json", state["names"])
        sp.save_npz(final / "DORMANT_A.npz", state["dA"])
        np.savez_compressed(final / "DORMANT_ARRAYS.npz", rhs=state["drhs"], sense=state["dsense"])
        atomic_json(final / "DORMANT_NAMES.json", {"rows": state["dnames"], "source_indices": state["dsource"]})
        if current_solution and state_identity(state) == last_solved["identity"]:
            np.savez_compressed(final / "FINAL_SOLUTION.npz", x=current_solution["x"], Pi=current_solution["Pi"], RC=current_solution["RC"])
            np.savez_compressed(final / "FINAL_BASIS.npz", VBasis=current_solution["VBasis"], CBasis=current_solution["CBasis"])
        model.write(str(final / "FINAL_MODEL.mps.gz"))
        model.dispose()
        timing["serialization"] += time.monotonic() - serialize_start
        with (OUT / "ROOT_HISTORY.csv").open("w", newline="") as f:
            cols = ["outer", "wall", "point_sha", "complete_Q", "pool_closed", "new_scenario_cuts", "new_probability_cut", "master_calls", "active_rows", "active_nnz", "active_scenario", "dormant_scenario", "best_certified_LB", "best_audited_UB", "certified_UB", "recompressed"]
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(history)
        atomic_json(OUT / "TIMING_BREAKDOWN.json", {"stages": timing, "solve_wall": solve_end_mono - start_mono, "cleanup_and_serialization_wall": time.monotonic() - solve_end_mono, "total_wall": time.monotonic() - start_mono, "begin_epoch": start_epoch, "solve_end_epoch": solve_end_epoch, "parallel_native_not_summed_as_wall": True})
        atomic_json(OUT / "FINAL_BOUNDS.json", {"initial_certified_LB": state.get("LB"), "best_certified_LB": best_lb, "certified_LB_gain": best_lb - state.get("LB"), "best_numerically_audited_UB": best_audited_ub, "certified_UB": None, "audited_gap": None if best_audited_ub is None else best_audited_ub - best_lb, "certified_gap": None, "milestones": milestones})
        call_summary = optimizer_ledger_summary()
        atomic_json(OUT / "OPTIMIZER_CALL_SUMMARY.json", call_summary)
        scientific_optimizer_calls = call_summary["explicit_begin_count"]
        complete = {"status": stop_reason, "started_batches": counters["started_batches"], "complete_Q_batches": counters["complete_Q_batches"], "complete_outer_iterations": counters["complete_outer_iterations"], "counters": counters, "scientific_optimizer_calls": scientific_optimizer_calls, "optimizer_call_ledger_crosscheck": call_summary, "best_certified_LB": best_lb, "best_audited_UB": best_audited_ub, "historical_ub_inherited": historical_ub["passed_for_inheritance"], "historical_ub_source_point_sha": historical_ub["inherited_point_sha"], "certified_UB": None, "wall_limit": WALL_LIMIT, "solve_wall": solve_end_mono - start_mono, "total_wall": time.monotonic() - start_mono, "integer_solution": False, "new_full400_recourse": False, "production_deployed": False}
        atomic_json(OUT / "COMPLETE.json", complete)
        atomic_json(OUT / "END.json", {"epoch": time.time(), "complete": complete})
        scientific_report = f"""# R44 RESUME_02 科学阶段报告\n\n物理、概率与变量界门槛不变；已知的active Benders primal数值残差被单独分类，不用于严格上界，而起点LB由独立对偶证书继承。\n\n- 从R43 C_POOL/ITER_002真实终点恢复：是。\n- 起点认证下界：{state.get('LB'):.12f}。\n- 启动/完成Q批次：{counters['started_batches']} / {counters['complete_Q_batches']}。\n- 完整外层：{counters['complete_outer_iterations']}。\n- 显式科学优化调用：{scientific_optimizer_calls}。\n- 最终认证下界：{best_lb:.12f}，增益：{best_lb-state.get('LB'):.12f}。\n- audited UB：{best_audited_ub}；历史继承：{historical_ub['passed_for_inheritance']}；strict certified UB：无。\n- 停止原因：{stop_reason}。\n- 新整数方案：无；full400新recourse：无；自动部署：无。\n"""
        (OUT / "SCIENTIFIC_REPORT_ZH.md").write_text(scientific_report)
        atomic_json(OUT / "ANALYSIS_MANIFEST.json", {"files": [record(p) for p in sorted(OUT.rglob("*")) if p.is_file() and "final_payload" not in p.parts], "external_dependencies": [str(R41), str(R42), str(REPAIR)]})
        atomic_json(OUT / "FULL_REPLAY_MANIFEST.json", {"payload": [record(p) for p in sorted(final.iterdir()) if p.is_file()], "external_dependencies": [record(R41 / "identity/FIRST_A.npz"), record(R41 / "identity/FIRST_ARRAYS.npz"), record(R42 / "GLOBAL_BOUND_PROOFS.json"), record(R42 / "RECOURSE_CUT_AUDIT.json")]})
        print(json.dumps(complete, indent=2), flush=True)
    if stop_reason in ("INFRASTRUCTURE_FAILED", "CORRECTNESS_FAILED", "NUMERICAL_CORRECTNESS_FAILED", "BASIS_NOT_ACCEPTED"):
        raise SystemExit(1)


def root_write_new(name, value):
    path = ROOT / name
    if path.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE_ROOT_EVIDENCE {path}")
    text = json.dumps(value, indent=2, sort_keys=True, default=cv, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(text)


def same_identity(a, b):
    return bool(a and b and a["pid"] == b["pid"] and a["start_ticks"] == b["start_ticks"])


def supervisor_tests():
    import ctypes
    tests = {}
    helper = "import ctypes,time; ctypes.CDLL(None).prctl(15,b'a b(c)d',0,0,0); time.sleep(5)"
    p = subprocess.Popen([sys.executable, "-c", helper])
    time.sleep(0.1)
    q = proc(p.pid)
    tests["comm_with_space_and_parentheses"] = bool(q and q["comm"] == "a b(c)d" and q["ppid"] == os.getpid())
    raw_evidence = q["stat_raw"] if q else None
    original = dict(q) if q else None
    fake = dict(q, start_ticks=str(int(q["start_ticks"]) + 1)) if q else None
    tests["pid_identity_starttime_mismatch"] = bool(q and not same_identity(original, fake))
    p.terminate()
    rc_named = p.wait()
    tests["exited_pid"] = proc(p.pid) is None

    normal = subprocess.Popen(["/bin/sleep", "0.05"])
    normal_identity = proc(normal.pid)
    normal_rc = normal.wait()
    tests["normal_child_real_wait"] = bool(normal_identity and normal_identity["ppid"] == os.getpid() and normal_rc == 0 and proc(normal.pid) is None)

    timeout_child = subprocess.Popen(["/bin/sleep", "5"])
    timeout_identity = proc(timeout_child.pid)
    timed_out = False
    try:
        timeout_child.wait(timeout=0.05)
    except subprocess.TimeoutExpired:
        timed_out = True
    same_before_signal = same_identity(timeout_identity, proc(timeout_child.pid))
    if timed_out and same_before_signal:
        timeout_child.terminate()
    timeout_rc = timeout_child.wait()
    tests["timeout_child_owned_signal_and_real_wait"] = bool(timed_out and same_before_signal and timeout_rc < 0 and proc(timeout_child.pid) is None)
    result = {
        "passed": all(tests.values()),
        "tests": tests,
        "raw_stat_with_special_comm": raw_evidence,
        "named_test_child_returncode": rc_named,
        "timeout_test_child_returncode": timeout_rc,
        "test_owned_signals": [{"pid": p.pid, "signal": "SIGTERM"}, {"pid": timeout_child.pid, "signal": "SIGTERM"}],
        "production_signals_sent": [],
        "scientific_optimizer_calls": 0,
    }
    atomic_json(OUT / "SUPERVISOR_TESTS.json", result)
    return result


def resume_safety_gate(paths):
    prior = ROOT / "RESUME_01"
    attempt_process = read(prior / "execution/main/PROCESS.json")
    old_child_now = proc(int(attempt_process["child"]["pid"]))
    old_parent_now = proc(int(attempt_process["parent"]["pid"]))
    old_child_absent = not same_identity(old_child_now, attempt_process["child"])
    old_parent_absent = not same_identity(old_parent_now, attempt_process["parent"])
    lease = read(prior / "RESOURCE_LEASE.json")
    stale = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        q = proc(int(p.name))
        if not q or q["pid"] == os.getpid():
            continue
        if any(token in q["argv"] for token in ("run_r44.py worker", "run_r44_resume01.py worker", "run_r44_resume02.py worker")):
            stale.append(q)
    failure_before = read(prior / "PROTECTION_AFTER.json")
    det_now = det_snapshot()
    native = read(DET / "NATIVE_BEGIN.json")
    det_ok = (
        same_identity(failure_before["DET"]["actual"], det_now["actual"])
        and failure_before["DET"]["boot_id"] == det_now["boot_id"]
        and native["model_sha"] == det_now["model"]["sha256"]
        and failure_before["DET"]["bounds"]["science"] == det_now["bounds"]["science"]
    )
    static_ok = all(failure_before["input_hashes"].get(str(p)) == sha(p) for p in paths)
    gate = {
        "passed": old_child_absent and old_parent_absent and lease.get("owner") is None and not stale and det_ok and static_ok,
        "old_child_absent": old_child_absent,
        "old_parent_absent": old_parent_absent,
        "old_lease_released": lease.get("owner") is None and int(lease.get("reserved_bytes", 0)) == 0,
        "stale_R44_optimizers": stale,
        "DET_identity_unchanged": det_ok,
        "DET_model_sha": det_now["model"]["sha256"],
        "DET_model_sha_matches_native_begin": native["model_sha"] == det_now["model"]["sha256"],
        "static_inputs_unchanged": static_ok,
        "production_signals_sent": [],
    }
    atomic_json(OUT / "RESUME_SAFETY_GATE.json", gate)
    return gate


def prior_evidence_manifest():
    roots = [ROOT / "RESUME_01"]
    singles = [ROOT / "ATTEMPT_00_SUPERVISOR_PARSE_FAILURE.json", ROOT / "RESUME_AUTHORIZATION.json", ROOT / "FAILURE.json", ROOT / "execution/main/WAIT_UNAVAILABLE.json"]
    files = [p for r in roots for p in r.rglob("*") if p.is_file()] + [p for p in singles if p.is_file()]
    return {str(p.relative_to(ROOT)): record(p) for p in sorted(files)}


def legacy_parent_not_used():
    import uuid
    if OUT.exists():
        raise SystemExit("RESUME_02_ALREADY_EXISTS_NO_RESUME_03")
    prior = ROOT / "RESUME_01"
    required_prior = [
        ROOT / "ATTEMPT_00_SUPERVISOR_PARSE_FAILURE.json",
        prior / "FAILURE.json", prior / "execution/main/WAIT.json",
        prior / "RESOURCE_LEASE.json", prior / "PROTECTION_AFTER.json",
    ]
    if any(not p.is_file() for p in required_prior):
        raise SystemExit("PRIOR_FAILURE_EVIDENCE_INCOMPLETE")
    prior_wait = read(prior / "execution/main/WAIT.json")
    if prior_wait.get("returncode") != 1 or not prior_wait.get("actual_parent_wait"):
        raise SystemExit("RESUME_01_WAIT_IDENTITY_MISMATCH")
    prior_manifest = prior_evidence_manifest()
    OUT.mkdir()
    (OUT / "tools").mkdir()
    shutil.copy2(Path(__file__), OUT / "tools/run_r44_resume02.py")
    authorization = {
        "authorized": True,
        "ATTEMPT_00_scientific_optimizer_calls": 0,
        "RESUME_01_scientific_optimizer_calls": 0,
        "RESUME_01_real_parent_WAIT_returncode": 1,
        "RESUME_01_failure": "restore_endpoint unified all-row residual assertion",
        "reason": "one-time repair of endpoint residual classification and execution records; no scientific model or algorithm change",
        "resume_directory": "RESUME_02",
        "RESUME_02_failure_forbids_automatic_RESUME_03": True,
        "production_signals_sent": [],
    }
    atomic_json(OUT / "RESUME_AUTHORIZATION_02.json", authorization)
    atomic_json(OUT / "PRIOR_EVIDENCE_MANIFEST_BEFORE.json", prior_manifest)
    paths, missing = required_inputs()
    before = protection_snapshot(paths if not missing else [])
    before["prior_evidence_manifest"] = prior_manifest
    atomic_json(OUT / "PROTECTION_BEFORE.json", before)
    if missing:
        atomic_json(OUT / "BLOCKER.json", {"status": "INPUT_INCOMPLETE", "paths": missing})
        return
    tests = supervisor_tests()
    if not tests["passed"]:
        atomic_json(OUT / "BLOCKER.json", {"status": "SUPERVISOR_UNIT_TEST_FAILED"})
        return
    gate = resume_safety_gate(paths)
    if not gate["passed"]:
        atomic_json(OUT / "BLOCKER.json", {"status": "RESUME_SAFETY_GATE_FAILED", "gate": gate})
        return
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, value, *_ = line.replace(":", "").split()
        mem[k] = int(value) * 1024
    request = 24 * 1024**3
    admitted = mem.get("MemAvailable", 0) >= request + 8 * 1024**3
    atomic_json(OUT / "RESOURCE_ADMISSION.json", {"admitted": admitted, "requested_bytes": request, "MemAvailable": mem.get("MemAvailable"), "workers": 4, "threads_each": 1, "production_modified": False})
    if not admitted:
        atomic_json(OUT / "BLOCKER.json", {"status": "RESOURCE_NOT_ADMITTED"})
        return
    lease_id = uuid.uuid4().hex
    atomic_json(OUT / "RESOURCE_LEASE.json", {"lease_id": lease_id, "owner": proc(os.getpid()), "reserved_bytes": request, "purpose": "R44 RESUME_02 finite pooled continuous root"})
    ex = OUT / "execution/main"; ex.mkdir(parents=True)
    cmd = [str(PYTHON), "-B", str(OUT / "tools/run_r44_resume02.py"), "worker"]
    started = time.monotonic()
    with (ex / "stdout.log").open("xb") as so, (ex / "stderr.log").open("xb") as se:
        child = subprocess.Popen(cmd, cwd=OUT, stdout=so, stderr=se)
        atomic_json(ex / "PROCESS.json", {"parent": proc(os.getpid()), "child": proc(child.pid), "command": cmd})
        peak = 0
        while child.poll() is None:
            tree = descendants(child.pid)
            rss = sum(p["rss"] for p in tree)
            peak = max(peak, rss)
            atomic_json(OUT / "STATUS.json", {"epoch": time.time(), "elapsed": time.monotonic() - started, "tree_rss": rss, "peak_rss": peak, "child": proc(child.pid)})
            time.sleep(3)
        rc = child.wait()
    atomic_json(ex / "WAIT.json", {"actual_parent_wait": True, "returncode": rc, "elapsed": time.monotonic() - started, "peak_rss": peak, "child_pid": child.pid, "epoch": time.time()})
    atomic_json(OUT / "RESOURCE_LEASE.json", {"lease_id": lease_id, "owner": None, "reserved_bytes": 0, "purpose": None, "released_epoch": time.time()})
    after = protection_snapshot(paths)
    same_det = (
        before["DET"]["actual"]
        and after["DET"]["actual"]
        and before["DET"]["actual"]["pid"] == after["DET"]["actual"]["pid"]
        and before["DET"]["actual"]["start_ticks"] == after["DET"]["actual"]["start_ticks"]
        and before["DET"]["boot_id"] == after["DET"]["boot_id"]
    )
    after["DET_identity_unchanged"] = bool(same_det)
    after["static_inputs_unchanged"] = before["input_hashes"] == after["input_hashes"]
    after_prior = prior_evidence_manifest()
    after["prior_evidence_unchanged"] = prior_manifest == after_prior
    after["production_signals_sent"] = []
    after["passed"] = after["DET_identity_unchanged"] and after["static_inputs_unchanged"] and after["prior_evidence_unchanged"]
    atomic_json(OUT / "PROTECTION_AFTER.json", after)
    attempt0 = read(ROOT / "ATTEMPT_00_SUPERVISOR_PARSE_FAILURE.json")
    resume01_wall = prior_wait["elapsed"]
    resume02_wall = read(ex / "WAIT.json")["elapsed"]
    walls = {"ATTEMPT_00_wall": attempt0["ATTEMPT_00_wall"], "RESUME_01_wall": resume01_wall, "RESUME_02_wall": resume02_wall, "combined_total_wall": attempt0["ATTEMPT_00_wall"] + resume01_wall + resume02_wall}
    atomic_json(OUT / "COMBINED_WALL.json", walls)
    call_summary = optimizer_ledger_summary()
    complete = read(OUT / "COMPLETE.json") if (OUT / "COMPLETE.json").exists() else None
    endpoint = read(OUT / "ENDPOINT_RESIDUAL_AUDIT.json") if (OUT / "ENDPOINT_RESIDUAL_AUDIT.json").exists() else None
    ub = read(OUT / "HISTORICAL_UB_INHERITANCE_AUDIT.json") if (OUT / "HISTORICAL_UB_INHERITANCE_AUDIT.json").exists() else None
    masters = sorted(OUT.glob("outer_*/MASTER_*/END.json"))
    first_basis = read(masters[0]).get("basis_accepted") if masters else None
    final = {
        "actual_parent_wait": read(ex / "WAIT.json"), "child_returncode": rc,
        "walls": walls, "optimizer_calls": call_summary, "complete": complete,
        "endpoint_residual_audit": endpoint, "historical_ub_inheritance": ub,
        "first_master_basis_explicitly_accepted": first_basis,
        "production_unchanged": after["passed"], "production_signals_sent": [],
        "automatic_RESUME_03_allowed": False,
    }
    atomic_json(OUT / "PARENT_COMPLETE.json", final)
    status = complete.get("status") if complete else "FAILED_BEFORE_COMPLETE"
    final_lb = complete.get("best_certified_LB") if complete else None
    audited_ub = complete.get("best_audited_UB") if complete else None
    report = f"""# R44 RESUME_02 最终报告\n\n- ATTEMPT_00 wall：{walls['ATTEMPT_00_wall']:.6f}s。\n- RESUME_01 wall：{walls['RESUME_01_wall']:.6f}s。\n- RESUME_02 wall：{walls['RESUME_02_wall']:.6f}s。\n- combined total wall：{walls['combined_total_wall']:.6f}s。\n- RESUME_02真正进入科学求解：{call_summary['explicit_begin_count'] > 0}。\n- 显式科学优化调用：{call_summary['explicit_begin_count']}（目录交叉核验一致：{call_summary['directory_scan_matches_ledger']}）。\n- 启动/完成Q批次：{complete.get('started_batches') if complete else 0} / {complete.get('complete_Q_batches') if complete else 0}。\n- 完整外层：{complete.get('complete_outer_iterations') if complete else 0}。\n- 最终认证LB：{final_lb}。\n- audited UB：{audited_ub}；历史继承：{ub.get('passed_for_inheritance') if ub else None}；strict UB：无。\n- 起点active Benders最大primal违反：{endpoint.get('active_scenario_max_violation') if endpoint else None}，该值不作为严格上界或起点LB。\n- 第一次master basis由原生日志明确接受：{first_basis}。\n- 停止原因：{status}；子进程真实WAIT={rc}。\n- 新整数方案：无；full400新recourse：无；生产进程未修改：{after['passed']}；production_signals_sent=[]。\n\n准确口径：物理、概率与变量界门槛不变；已知的active Benders primal数值残差被单独分类，不用于严格上界，而起点LB由独立对偶证书继承。\n"""
    (OUT / "FINAL_REPORT_ZH.md").write_text(report)
    if rc:
        raise SystemExit(rc)


def owned_run(command, directory, timeout=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    lease = {"lease_id": __import__("uuid").uuid4().hex, "owner": os.getpid()}
    atomic_json(directory / "LEASE.json", lease)
    child = None; started = time.monotonic(); rc = None; timed_out = False
    try:
        with (directory / "stdout.log").open("xb") as so, (directory / "stderr.log").open("xb") as se:
            child = subprocess.Popen(command, stdout=so, stderr=se)
            identity = proc(child.pid)
            atomic_json(directory / "PROCESS.json", {"child": identity, "command": command, "parent": proc(os.getpid())})
            try:
                rc = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if same_identity(identity, proc(child.pid)):
                    child.terminate()
                rc = child.wait()
            atomic_json(directory / "WAIT.json", {"actual_parent_wait": True, "returncode": rc, "pid": child.pid, "elapsed": time.monotonic()-started, "timed_out": timed_out})
    finally:
        if child is not None and child.poll() is None:
            child.wait()
        atomic_json(directory / "LEASE.json", dict(lease, owner=None, reserved_bytes=0, released=True))
    return rc


def preflight():
    global OUT, PREFLIGHT_MODE, Oracle
    import copy
    science_out = OUT
    test_number = len(list((science_out / "PREFLIGHT").glob("TEST_*")))+1
    OUT = science_out / "PREFLIGHT" / f"TEST_{test_number:02d}"
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); PREFLIGHT_MODE = True
    original_optimize = gp.Model.optimize
    def forbidden(*args, **kwargs):
        raise RuntimeError("PREFLIGHT_OPTIMIZER_FORBIDDEN")
    gp.Model.optimize = forbidden
    checks = {}
    try:
        checks["supervisor"] = supervisor_tests()["passed"]
        state = restore_endpoint(OUT)
        target = OUT / "ABSENT" / "POINT_REPAIR"
        accepted, audit = point_repair(state["x"][:FIRST_N], target, time.monotonic()+60)
        z = np.load(target / "POINTS.npz")
        checks["real_endpoint_point_repair"] = bool(audit["displacement"] == 0 and audit["projection"] is None and np.array_equal(z["raw"], z["accepted"]) and np.array_equal(accepted,z["accepted"]))
        try:
            point_repair(state["x"][:FIRST_N], target, time.monotonic()+60)
            checks["no_overwrite"] = False
        except FileExistsError:
            checks["no_overwrite"] = True
        # A regular file used as a directory is deterministically unwritable,
        # including under privileged accounts.
        blocker = OUT / "not_a_directory"; blocker.write_text("injected")
        try:
            point_repair(accepted, blocker / "POINT_REPAIR", time.monotonic()+60)
            checks["unwritable_directory_rejected"] = False
        except OSError:
            checks["unwritable_directory_rejected"] = True
        constructed = []
        class FakeOracle:
            def __init__(self,sid,root): constructed.append(sid)
            def solve(self,x,label,deadline): return {"mock": True}
            def close(self): pass
        class FakeConn:
            def __init__(self): self.messages=iter([{"op":"solve","x":[],"label":"a","deadline":0},{"op":"solve","x":[],"label":"b","deadline":0},{"op":"close"}]); self.results=[]
            def recv(self): return next(self.messages)
            def send(self,x): self.results.append(x)
            def close(self): pass
        real_oracle=Oracle
        try:
            Oracle=FakeOracle; conn=FakeConn(); oracle_worker(conn,[1,2],OUT)
        finally: Oracle=real_oracle
        checks["same_worker_two_sid_two_messages_only_two_constructions"] = constructed == [1,2] and len(conn.results)==2
        tiny = {"A":sp.csr_matrix([[1.],[1.],[1.]]),"rhs":np.array([0.,0.,0.]),"sense":np.array([">",">",">"]),"cost":np.array([1.]),"lo":np.array([0.]),"hi":np.array([1.]),"names":{"rows":["physical","benders/1","benders/2"],"columns":["x"]},"dA":sp.csr_matrix((0,1)),"drhs":np.array([]),"dsense":np.array([],dtype="U1"),"dnames":[],"dsource":[]}
        sol={"x":np.array([0.]),"Pi":np.array([1.,0.,0.]),"RC":np.array([0.]),"VBasis":np.array([0]),"CBasis":np.array([-1,0,0])}
        frozen=freeze_solved(tiny,sol,0.)
        for name in ("ADDED_NOT_SOLVED","READDED_NOT_SOLVED"):
            pending=copy.deepcopy(tiny); pending["A"]=sp.vstack([pending["A"],sp.csr_matrix([[1.]])],format="csr"); pending["rhs"]=np.r_[pending["rhs"],0.]; pending["sense"]=np.concatenate([pending["sense"],np.array([">"])]); pending["names"]["rows"].append("benders/new")
            export_state(OUT/name/"LAST_SOLVED",tiny,frozen)
            export_state(OUT/name/"WORKING_PENDING",pending)
            checks[name]=not (OUT/name/"WORKING_PENDING/SOLUTION.npz").exists()
            try: export_state(OUT/name/"MISMATCH_REJECTED",pending,frozen); checks[name]=False
            except RuntimeError: pass
        class FakeModel:
            def getConstrs(self): return list(range(3))
            def getVars(self): return [0]
            def remove(self,rows): pass
            def update(self): pass
            def setAttr(self,*args): pass
        (OUT/"outer_000").mkdir()
        compressed=copy.deepcopy(tiny)
        _,_,post,rec=maybe_recompress(compressed,FakeModel(),None,sol,0)
        export_state(OUT/"COMPRESSED_LAST_SOLVED",compressed,freeze_solved(compressed,post,0.))
        save_checkpoint(compressed,rec["checkpoint_solution"],rec["checkpoint_basis"],0,True,0.,None,{})
        checks["post_recompression_checkpoint_identity"] = rec["executed"] and read(OUT/"checkpoints/CHECKPOINT_000.json")["rows"]==1
        checks["early_worker_nonzero_WAIT"] = owned_run([sys.executable,"-c","raise SystemExit(7)"],OUT/"EARLY_EXIT")==7
        checks["timeout_worker_nonzero_WAIT"] = owned_run([sys.executable,"-c","import time; time.sleep(5)"],OUT/"TIMEOUT",0.1)<0
        checks["leases_released"] = all(read(OUT/n/"LEASE.json")["owner"] is None for n in ("EARLY_EXIT","TIMEOUT"))
        checks["no_scientific_optimizer_calls"] = optimizer_ledger_summary()["explicit_begin_count"]==0
        checks["optimizer_intercept"] = False
        try:
            original_test=gp.Model(); original_test.optimize()
        except RuntimeError as exc: checks["optimizer_intercept"] = str(exc)=="PREFLIGHT_OPTIMIZER_FORBIDDEN"
        finally: original_test.dispose()
        checks["sealed_manifest"] = sealed_manifest(OUT/"EARLY_EXIT")
        result={"passed":all(checks.values()),"checks":checks,"wall":time.monotonic()-start,"source_sha":sha(Path(__file__)),"mock_results_are_not_scientific_progress":True}
        atomic_json(OUT/"PREFLIGHT_RESULT.json",result)
        return result
    finally:
        gp.Model.optimize=original_optimize; PREFLIGHT_MODE=False; OUT=science_out


def parent():
    preflight_retry = len(sys.argv)==2 and sys.argv[1]=="preflight-resume"
    if OUT.exists() and any(OUT.iterdir()) and not (preflight_retry and not (OUT/"execution/main").exists() and not (OUT/"BEGIN.json").exists()):
        raise SystemExit("RESUME_03_NONEMPTY_NO_RESUME_04")
    OUT.mkdir(exist_ok=True)
    # Freeze the source in the isolated output before tests or execution.
    (OUT/"tools").mkdir(exist_ok=True)
    version=len(list((OUT/"tools").glob("*.py")))+1
    source=OUT/"tools"/f"run_r44_resume03_v{version:02d}.py"
    shutil.copy2(Path(__file__),source)
    if not (OUT/"RESUME_AUTHORIZATION_03.json").exists():
        atomic_json(OUT/"RESUME_AUTHORIZATION_03.json",{"authorized":True,"one_attempt":True,"automatic_RESUME_04":False,"prior_failed_wall":22.673524806275964,"production_signals_sent":[]})
    paths,missing=required_inputs()
    before=protection_snapshot(paths if not missing else [])
    old=ROOT/"RESUME_02"
    prior_files=[p for p in ROOT.rglob("*") if p.is_file() and OUT not in p.parents]
    prior_hashes={str(p):sha(p) for p in prior_files}
    atomic_json(OUT/f"PROTECTION_BEFORE_V{version:02d}.json",dict(before,prior_hashes=prior_hashes))
    tests=preflight()
    atomic_json(OUT/"PREFLIGHT_RESULT.json",tests)
    if not tests["passed"]:
        atomic_json(OUT/"BLOCKER.json",{"status":"PREFLIGHT_FAILED"}); return
    checks={"inputs_complete":not missing,"old_lease_released":read(old/"RESOURCE_LEASE.json")["owner"] is None,
            "source_frozen":sha(source)==tests["source_sha"],"DET_identity":same_identity(read(old/"PROTECTION_AFTER.json")["DET"]["actual"],before["DET"]["actual"]),
            "DET_model":before["DET"]["model"]["sha256"]==read(old/"PROTECTION_AFTER.json")["DET"]["model"]["sha256"]}
    for prior in (ROOT,ROOT/"RESUME_01",old):
        p=read(prior/"execution/main/PROCESS.json")
        checks[str(prior)+"_child_exited"]=not same_identity(proc(p["child"]["pid"]),p["child"])
        checks[str(prior)+"_parent_exited"]=not same_identity(proc(p["parent"]["pid"]),p["parent"])
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    checks["resource_admitted"]=mem["MemAvailable"]>=32*1024**3
    atomic_json(OUT/"SAFETY_GATE.json",{"passed":all(checks.values()),"checks":checks,"MemAvailable":mem["MemAvailable"],"reserved_bytes":24*1024**3,"production_signals_sent":[]})
    if not all(checks.values()):
        atomic_json(OUT/"BLOCKER.json",{"status":"RESUME_03_SAFETY_GATE_FAILED"}); return
    rc=owned_run([str(PYTHON),"-B",str(source),"worker"],OUT/"execution/main")
    after=protection_snapshot(paths)
    after["prior_evidence_unchanged"]=all(sha(Path(p))==h for p,h in prior_hashes.items())
    after["DET_identity_unchanged"]=same_identity(before["DET"]["actual"],after["DET"]["actual"])
    after["static_inputs_unchanged"]=before["input_hashes"]==after["input_hashes"]
    atomic_json(OUT/"PROTECTION_AFTER.json",after)
    wait=read(OUT/"execution/main/WAIT.json")
    complete=read(OUT/"COMPLETE.json") if (OUT/"COMPLETE.json").exists() else {"status":"FAILED_BEFORE_SCIENTIFIC_LOOP","scientific_optimizer_calls":optimizer_ledger_summary()["explicit_begin_count"]}
    atomic_json(OUT/"FINAL_EXECUTION.json",{"preflight_wall":tests["wall"],"RESUME_03_wall":wait["elapsed"],"previous_failed_wall":22.673524806275964,"combined_total_wall":22.673524806275964+tests["wall"]+wait["elapsed"],"actual_WAIT":wait,"scientific":complete,"automatic_RESUME_04":False})
    (OUT/"FINAL_REPORT_ZH.md").write_text("R44 RESUME_03\n\n"+json.dumps(read(OUT/"FINAL_EXECUTION.json"),indent=2)+"\n\n生产保护："+json.dumps({k:after[k] for k in ("prior_evidence_unchanged","DET_identity_unchanged","static_inputs_unchanged")})+"\n无新整数方案；无full400新recourse；production_signals_sent=[]。\n")
    sealed_manifest(OUT)
    if rc: raise SystemExit(rc)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "worker":
            run_worker()
        elif len(sys.argv) == 1 or (len(sys.argv)==2 and sys.argv[1]=="preflight-resume"):
            parent()
        else:
            raise SystemExit("usage: run_r44.py [worker]")
    except Exception:
        if OUT.exists():
            atomic_json(OUT / "FAILURE.json", {"epoch": time.time(), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
