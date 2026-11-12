#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, fcntl, hashlib, importlib.util, json, math, os, shutil
import subprocess, sys, threading, time, traceback, uuid
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
import numpy as np
BASE=Path('/home/Shixin/ev-assignment')
OLD_JOB=BASE/'L08_ARO_DRO_GAP1_ACCELERATION_V1'
JOB=OLD_JOB/'MEMORY_EXIT_REPAIR_V1/CONTINUE_01'
TOOLS=Path(__file__).resolve().parent
ARO_SRC=BASE/'L08_ARO400_SHARED_INTEGER_AUTONOMOUS_V1'
ARO_REV=ARO_SRC/'CODE_REVISIONS/REPAIR_02_PREFLIGHT_REUSE'
CONTROL=BASE/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
PYTHON=BASE/'releases/case0-expanded-r5-20260910T045000Z/.venv/bin/python'
sys.path.insert(0,str(TOOLS))
from gap_policy import evaluate_gap,native_number
ARO_SCOPE='FULL400_SHARED_INTEGER_MINMAX'
MODEL_REGISTRY=[]

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()

def atomic(path, value, immutable=False):
    path = Path(path)
    if JOB.resolve() not in path.resolve().parents:
        raise ValueError("WRITE_OUTSIDE_JOB")
    data = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError("IMMUTABLE_CONFLICT " + str(path))
        return
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if path.read_bytes() != data:
        raise ValueError("PUBLISHED_OBJECT_READBACK_FAILED")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

def verify_ref(reference):
    path = Path(reference["path"])
    if not path.is_file() or path.stat().st_size != reference.get("bytes", path.stat().st_size):
        raise ValueError("REFERENCE_BYTES " + str(path))
    if sha(path) != reference["sha256"]:
        raise ValueError("REFERENCE_SHA " + str(path))
    return path

def read(path):
    return json.loads(Path(path).read_text())

def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

def proc(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        right = raw.rfind(") ")
        if right < 0:
            return None
        fields = raw[right + 2:].split()
        argv = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                "start_ticks": fields[19], "boot_id": boot_id(),
                "argv": [x.decode(errors="replace") for x in argv if x]}
    except (OSError, ValueError, IndexError):
        return None

def same(a, b):
    return bool(a and b and a.get("pid") == b.get("pid") and
                str(a.get("start_ticks")) == str(b.get("start_ticks")) and
                a.get("boot_id") == b.get("boot_id") and b.get("state") != "Z")

def remaining(run):
    if run["boot_id"] == boot_id():
        return run["deadline_monotonic"] - time.monotonic()
    return run["deadline_utc"] - time.time()

def aro_imports():
    os.environ["ARO_ROOT"] = str(ARO_SRC)
    sys.path.insert(0, str(ARO_REV / "lifecycle"))
    sys.path.insert(0, str(ARO_REV))
    import gurobipy as gp
    import aro_core as ac
    import engine as ae
    import durable as ad
    return gp, ac, ae, ad

def aro_old_state(ad):
    pointer = read(ARO_SRC / "REPAIR_STATE.json")["state"]
    verify_ref(pointer)
    state = read(pointer["path"])
    model_id = ad.digest([row["fingerprint"] for row in state["cuts"]])
    model = ARO_SRC / "WORKING_PENDING" / model_id / "MODEL.mps.gz"
    if not model.is_file():
        raise ValueError("ARO_PENDING_MODEL_MISSING")
    return state, model_id, model, pointer

def aro_validate_records(records, point_id):
    if len(records) != 400 or {int(r["sid"]) for r in records} != set(range(400)):
        raise ValueError("ARO_PARTIAL400")
    values = []
    for record in records:
        if record["point_sha"] != point_id or not math.isfinite(float(record["qU_audited"])):
            raise ValueError("ARO_MIXED_POINT_OR_Q")
        for source in record["sources"]:
            verify_ref(source)
        values.append(float(record["qU_audited"]))
    return values

def aro_cached_records(point_id):
    old = ARO_SRC / "evaluations" / point_id / "COMPLETE.json"
    inherited = OLD_JOB / "ARO" / "evaluations" / point_id / "COMPLETE.json"
    if inherited.exists():
        old = inherited
    new = JOB / "ARO" / "evaluations" / point_id / "COMPLETE.json"
    path = new if new.exists() else old if old.exists() else None
    if path is None:
        return None
    payload = read(path)
    aro_validate_records(payload["records"], point_id)
    return payload["records"]

def aro_evaluate_new(ac, ae, run, deadline, attempt, x, point_id):
    out = JOB / "ARO"; session = attempt / "ARO_WORKERS" / point_id / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=True)
    policy = read(ARO_SRC / "PATCH_POLICY.json")
    policy = dict(policy, run_id=run["execution_id"], original_deadline_utc=run["deadline_utc"])
    seed_partial_cache(ac, ae, out, x, point_id)
    ae.configure(out, dict(run, source=str(ARO_SRC)), deadline)
    workers = ae.Workers(out, dict(run, source=str(ARO_SRC)), deadline, session,
                         oracle_cls=ac.AROOracle, policy=policy)
    try:
        records = workers.evaluate(x, list(range(400)), sha(ARO_SRC / "SOURCE_LOCK.json"),
                                   lambda count: atomic(out / "PROGRESS.json",
                                                        {"phase": "EVALUATE400", "completed": count,
                                                         "point_id": point_id, "epoch": time.time()}))
    finally:
        workers.close()
    aro_validate_records(records, point_id)
    folder = out / "evaluations" / point_id; folder.mkdir(parents=True, exist_ok=True)
    atomic(folder / "COMPLETE.json", {"records": records, "point_sha": point_id,
                                      "scientific_id": sha(ARO_SRC / "SOURCE_LOCK.json"),
                                      "full400": True}, True)
    return records

def aro_start_vector(model, ac, x, records, folder):
    A, a, names = ac.first()
    if len(x) != len(names) or model.NumVars != len(names) + 1:
        raise ValueError("ARO_START_DIMENSION")
    if model.getAttr("VarName")[:len(names)] != names or model.getVars()[-1].VarName != "ARO_ETA":
        raise ValueError("ARO_START_NAMES")
    audit = ac.e.b.row_audit(A, a["rhs"], a["sense"], a["lo"], a["hi"], x)
    binary = a["vtype"] == "B"
    distance = float(np.max(np.abs(x[binary] - np.rint(x[binary])), initial=0))
    if max(audit.values()) > 1e-9 or distance > 1e-9:
        raise ValueError("ARO_BEST_START_FIRST_STAGE")
    matrix = model.getA().tocsr(); rhs = np.asarray(model.getAttr("RHS"), float)
    sense = np.asarray(model.getAttr("Sense")); row_names = model.getAttr("ConstrName")
    eta_col = len(names); requirement = 0.0
    for i in range(A.shape[0], model.NumConstrs):
        ae = float(matrix[i, eta_col])
        if ae == 0:
            continue
        lhs = float(np.asarray(matrix[i, :eta_col] @ x).reshape(-1)[0])
        if sense[i] == "<" and ae < 0:
            requirement = max(requirement, (rhs[i] - lhs) / ae)
        elif sense[i] == ">" and ae > 0:
            requirement = max(requirement, (rhs[i] - lhs) / ae)
        else:
            raise ValueError("ARO_SUPPORT_ETA_ORIENTATION " + row_names[i])
    qmax = max(float(r["qU_audited"]) for r in records)
    eta = float(np.nextafter(max(0.0, qmax, requirement) + 1e-6, np.inf))
    start = np.r_[x, eta]
    residual = ac.e.b.row_audit(matrix, rhs, sense, np.asarray(model.getAttr("LB")),
                                np.asarray(model.getAttr("UB")), start)
    if max(residual.values()) > 1e-7:
        raise ValueError(("ARO_COMPLETE_START_INFEASIBLE", residual))
    model.setAttr("Start", model.getVars(), start.tolist())
    result = {"passed": True, "eta_start": eta, "qmax": qmax,
              "max_support_activity": requirement, "first_stage": audit,
              "complete_master": residual, "variables": len(start),
              "only_Start_changed": True}
    atomic(folder / "BEST_START_AUDIT.json", result, True)
    return result

def aro_stage(run, attempt, deadline):
    out = JOB / "ARO"; out.mkdir(parents=True, exist_ok=True)
    (out / "NODEFILES").mkdir(exist_ok=True)
    gp, ac, ae, ad = aro_imports()
    old, old_model_id, old_model, old_pointer = aro_old_state(ad)
    state_path = out / "STATE.json"
    if state_path.exists():
        state = read(state_path); model_path = verify_ref(state["model"])
    else:
        state = {"phase": "SOLVE", "iteration": 0,
                 "best_LB": float(old["best_LB"]),
                 "best_LB_source": ref(ARO_SRC / "outer_00054" /
                                             next(p.name for p in (ARO_SRC / "outer_00054").glob("MIP_END_*.json"))),
                 "best_UB_audited": float(old["best_UB_audited"]),
                 "best_UB_certified": None, "best_plan": old["best_plan"],
                 "best_plan_id": old["best_plan_id"],
                 "base_model_id": old_model_id, "model": ref(old_model),
                 "new_cut_fingerprints": [], "separated_points": list(old["processed"].keys()),
                 "historical_evaluations_reused": 71, "branch_tree_restored": False,
                 "external_scope": ARO_SCOPE, "history": []}
        atomic(state_path, state)
        model_path = old_model
    gap = evaluate_gap(state["best_LB"], state["best_UB_audited"], expected_scope=ARO_SCOPE,
                       lb_scope=ARO_SCOPE, ub_scope=ARO_SCOPE,
                       lower_evidence_verified=True, upper_evaluation_complete=True,
                       upper_evidence_verified=True)
    atomic(attempt / "INITIAL_GAP.json", gap.to_dict(), True)
    state.pop("status", None)
    state["phase"] = "SOLVE"
    atomic(state_path, state)
    model = gp.read(str(model_path))
    MODEL_REGISTRY.append(model)
    if (model.NumConstrs, model.NumVars, model.NumBinVars) != (96914 + len(state["new_cut_fingerprints"]), 60781, 25770):
        raise ValueError(("ARO_MASTER_SHAPE", model.NumConstrs, model.NumVars, model.NumBinVars))
    xvars = model.getVars()[:60780]; eta = model.getVars()[-1]
    best_x = np.load(verify_ref(state["best_plan"]))["accepted"]
    best_records = aro_cached_records(state["best_plan_id"])
    if best_records is None:
        raise ValueError("ARO_BEST400_MISSING")
    aro_start_vector(model, ac, best_x, best_records, out / "START_AUDITS" / str(time.time_ns()))
    known_rows = set(model.getAttr("ConstrName"))
    evaluated = set(state["separated_points"])
    first_A, first_a, first_names = ac.first(); binary = first_a["vtype"] == "B"
    while remaining(run) > 180:
        iteration = state["iteration"]; folder = out / f"ITER_{iteration:04d}"; folder.mkdir(exist_ok=True)
        log = folder / "NATIVE.log"
        parameters = {"Threads": 4, "MIPFocus": 0, "Method": 1, "NodeMethod": 1,
                      "PreSparsify": 1, "FeasibilityTol": 1e-9,
                      "OptimalityTol": 1e-9, "IntFeasTol": 1e-9,
                      "MIPGap": 0.001, "MIPGapAbs": 1.0, "SoftMemLimit": 24, "NodefileStart": 0.5, "NodefileDir": str(out / "NODEFILES"),
                      "Seed": 0, "TimeLimit": max(0.01, remaining(run) - 120),
                      "OutputFlag": 1, "LogFile": str(log)}
        for key, value in parameters.items():
            setattr(model.Params, key, value)
        begin = {"epoch": time.time(), "iteration": iteration,
                 "parameters": parameters, "rows": model.NumConstrs,
                 "columns": model.NumVars, "nonzeros": model.NumNZs,
                 "branch_tree_restored": False if iteration == 0 else "same_process_model_when_unchanged"}
        if not state.get("pending_evaluation"):
            atomic(folder / "BEGIN.json", begin, True)
        cb = {"root_complete_at": None, "first_root_relaxation_at": None,
              "pending": None, "last_publish": 0.0, "terminated_for_handoff": False,
              "mipsol_count": 0}
        allvars = model.getVars()
        def callback(m, where):
            now = time.monotonic()
            if (out / "PAUSE_RESOURCE.json").exists() or (out / "USER_STOP.json").exists() or now >= run["deadline_monotonic"] - 120:
                m.terminate(); return
            if where == gp.GRB.Callback.MIPNODE:
                try:
                    if int(m.cbGet(gp.GRB.Callback.MIPNODE_STATUS)) == gp.GRB.OPTIMAL and cb["first_root_relaxation_at"] is None:
                        cb["first_root_relaxation_at"] = now
                except gp.GurobiError:
                    pass
            if where == gp.GRB.Callback.MIP:
                nodes = float(m.cbGet(gp.GRB.Callback.MIP_NODCNT))
                if nodes > 0 and cb["root_complete_at"] is None:
                    cb["root_complete_at"] = now
                if now - cb["last_publish"] >= 60:
                    cb["last_publish"] = now
                    atomic(out / "PROGRESS.json", {"phase": "ARO_MASTER", "iteration": iteration,
                           "epoch": time.time(), "nodes": nodes,
                           "incumbent": native_number(m.cbGet(gp.GRB.Callback.MIP_OBJBST), solver_infinity=float(gp.GRB.INFINITY)),
                           "bound": native_number(m.cbGet(gp.GRB.Callback.MIP_OBJBND), solver_infinity=float(gp.GRB.INFINITY)),
                           "root_processing_completed": cb["root_complete_at"] is not None,
                           "remaining_global_s": remaining(run)})
                if (cb["root_complete_at"] is not None and now - cb["root_complete_at"] >= 300 and
                        cb["pending"] is not None):
                    cb["terminated_for_handoff"] = True; m.terminate()
            if where == gp.GRB.Callback.MIPSOL:
                values = np.asarray(m.cbGetSolution(allvars), float)
                rawx = values[:60780]
                if np.max(np.abs(rawx[binary] - np.rint(rawx[binary])), initial=0) <= 1e-8:
                    xx = rawx.copy(); xx[binary] = np.rint(xx[binary])
                    pid = hashlib.sha256(xx.tobytes()).hexdigest(); cb["mipsol_count"] += 1
                    if pid not in evaluated:
                        cb["pending"] = {"x": xx, "eta": float(values[-1]), "point_id": pid}
        saved_pending = state.get("pending_evaluation")
        if saved_pending and saved_pending["point_id"] not in evaluated:
            stored = np.load(verify_ref(saved_pending["payload"]))
            xx = np.asarray(stored[saved_pending["array_key"]], float).copy()
            if hashlib.sha256(xx.tobytes()).hexdigest() != saved_pending["point_id"]:
                raise ValueError("PENDING_POINT_IDENTITY")
            cb["pending"] = {"x": xx, "eta": saved_pending["eta"], "point_id": saved_pending["point_id"]}
            status = gp.GRB.INTERRUPTED
            bound_observation = {"classification": "MISSING", "value": None}
            atomic(folder / "PENDING_REUSED_BEFORE_OPTIMIZE.json", saved_pending, True)
        else:
            model.optimize(callback)
            status = int(model.Status)
            try:
                raw_bound = model.ObjBoundC
            except (gp.GurobiError, AttributeError):
                raw_bound = None
            bound_observation = native_number(raw_bound, solver_infinity=float(gp.GRB.INFINITY))
            end = {"epoch": time.time(), "status": status, "runtime": float(model.Runtime),
                   "nodes": float(model.NodeCount), "iterations": float(model.IterCount),
                   "work": float(model.Work), "mem_used_GB": float(model.MemUsed),
                   "max_mem_GB": float(model.MaxMemUsed), "SolCount": int(model.SolCount),
                   "ObjBoundC": bound_observation, "callback": {k: v for k, v in cb.items() if k != "pending"}}
            atomic(folder / "END.json", end, True)
            snapshot = save_native_exit(model, cb, binary, state, state_path, folder, status, bound_observation)
            if status == gp.GRB.MEM_LIMIT:
                # Snapshot and bound are committed before resource pause; never grow/evaluate here.
                break
            if (out / "PAUSE_RESOURCE.json").exists() or (out / "USER_STOP.json").exists():
                state["status"] = "PAUSED_RESOURCE_OR_USER_SAVED"
                atomic(state_path, state)
                break
        if status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED, gp.GRB.NUMERIC):
            raise ValueError("ARO_MASTER_CORRECTNESS_STATUS_" + str(status))
        if bound_observation["classification"] == "FINITE":
            candidate_lb = float(np.nextafter(bound_observation["value"] - 1e-6, -np.inf))
            if candidate_lb > state["best_LB"]:
                state["best_LB"] = candidate_lb; state["best_LB_source"] = ref(folder / "END.json")
        gap = evaluate_gap(state["best_LB"], state["best_UB_audited"], expected_scope=ARO_SCOPE,
                           lb_scope=ARO_SCOPE, ub_scope=ARO_SCOPE,
                           lower_evidence_verified=True, upper_evaluation_complete=True,
                           upper_evidence_verified=True)
        if gap.reached:
            state["status"] = "NUMERICAL_GAP_TARGET_REACHED_CERTIFICATION_PENDING"
            state["external_gap"] = gap.to_dict(); atomic(state_path, state); break
        pending = cb["pending"]
        if pending is None and model.SolCount:
            values = np.asarray(model.X, float); xx = values[:60780].copy(); xx[binary] = np.rint(xx[binary])
            pid = hashlib.sha256(xx.tobytes()).hexdigest()
            if pid not in evaluated:
                pending = {"x": xx, "eta": float(values[-1]), "point_id": pid}
        if pending is None:
            state["iteration"] += 1; state["history"].append({"iteration": iteration,
                "status": status, "LB": state["best_LB"], "UB": state["best_UB_audited"],
                "new_candidate": False}); atomic(state_path, state)
            if status in (gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED) or remaining(run) <= 180:
                state["status"] = "GLOBAL_WALL_LIMIT"; atomic(state_path, state); break
            if status == gp.GRB.OPTIMAL:
                state["status"] = "NO_NEW_CANDIDATE_CONSISTENCY_DIAGNOSTIC"; atomic(state_path, state); break
            continue
        x = pending["x"]; point_id = pending["point_id"]
        state["phase"] = "EVALUATE_PENDING"
        atomic(state_path, state)
        first_audit = ac.e.b.row_audit(first_A, first_a["rhs"], first_a["sense"],
                                       first_a["lo"], first_a["hi"], x)
        if max(first_audit.values()) > 1e-8:
            raise ValueError(("ARO_CANDIDATE_FIRST_STAGE", first_audit))
        point_dir = folder / ("POINT_" + point_id); point_dir.mkdir(exist_ok=True)
        np.savez_compressed(point_dir / "POINT.npz", accepted=x)
        atomic(point_dir / "AUDIT.json", {"passed": True, "point_id": point_id,
                                          "residual": first_audit}, True)
        records = aro_cached_records(point_id)
        cache_hit = records is not None
        if records is None:
            records = aro_evaluate_new(ac, ae, run, run["deadline_monotonic"] - 120,
                                       attempt, x, point_id)
        qvalues = aro_validate_records(records, point_id)
        f = float(np.nextafter(np.asarray(first_a["cost"], np.longdouble) @ np.asarray(x, np.longdouble), np.inf))
        ub = float(np.nextafter(np.longdouble(f) + np.longdouble(max(qvalues)), np.inf))
        if ub < state["best_UB_audited"]:
            state["best_UB_audited"] = ub
            state["best_plan"] = ref(point_dir / "POINT.npz")
            state["best_plan_id"] = point_id
            atomic(out / "BEST_REVISIONS" / (point_id + ".json"), {"point_id": point_id, "F": f,
                                          "maxQ": max(qvalues), "UB": ub,
                                          "cache_hit": cache_hit}, True)
        added = 0; new_fps = []
        for record in sorted(records, key=lambda item: item["sid"]):
            cut = np.load(verify_ref({"path": record["cut"], "sha256": sha(record["cut"]),
                                      "bytes": Path(record["cut"]).stat().st_size}))
            alpha = float(cut["alpha"]); beta = np.asarray(cut["beta"], float)
            fp = hashlib.sha256(str(record["sid"]).encode() + np.array([alpha]).tobytes() + beta.tobytes()).hexdigest()
            name = f"aro_support/{record['sid']}/{fp}"
            value = float(np.longdouble(alpha) + np.asarray(beta, np.longdouble) @ np.asarray(x, np.longdouble))
            if value > pending["eta"] + 1e-6 and name not in known_rows:
                ac.add_support(model, xvars, eta, alpha, beta, name); known_rows.add(name)
                added += 1; new_fps.append(fp)
        model.update(); evaluated.add(point_id)
        state["separated_points"] = sorted(evaluated)
        state["pending_evaluation"] = None
        state["phase"] = "SOLVE"
        state["new_cut_fingerprints"].extend(new_fps)
        state["iteration"] += 1
        state["history"].append({"iteration": iteration, "LB": state["best_LB"],
            "UB": state["best_UB_audited"], "point_id": point_id,
            "cache_hit": cache_hit, "new_cuts": added})
        checkpoint = out / "CHECKPOINTS" / f"ITER_{state['iteration']:04d}.mps.gz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True); model.write(str(checkpoint))
        state["model"] = ref(checkpoint); atomic(state_path, state)
        gap = evaluate_gap(state["best_LB"], state["best_UB_audited"], expected_scope=ARO_SCOPE,
                           lb_scope=ARO_SCOPE, ub_scope=ARO_SCOPE,
                           lower_evidence_verified=True, upper_evaluation_complete=True,
                           upper_evidence_verified=True)
        atomic(out / "GAP.json", gap.to_dict())
        if gap.reached:
            state["status"] = "NUMERICAL_GAP_TARGET_REACHED_CERTIFICATION_PENDING"
            atomic(state_path, state); break
        if added == 0 and status == gp.GRB.OPTIMAL:
            state["status"] = "NO_PROGRESS_DIAGNOSTIC"; atomic(state_path, state); break
        best_x = np.load(verify_ref(state["best_plan"]))["accepted"]
        best_records = aro_cached_records(state["best_plan_id"])
        aro_start_vector(model, ac, best_x, best_records, out / "START_AUDITS" / str(time.time_ns()))
    if "status" not in state:
        state["status"] = "GLOBAL_WALL_LIMIT"
    final_gap = evaluate_gap(state["best_LB"], state["best_UB_audited"], expected_scope=ARO_SCOPE,
                             lb_scope=ARO_SCOPE, ub_scope=ARO_SCOPE,
                             lower_evidence_verified=True, upper_evaluation_complete=True,
                             upper_evidence_verified=True)
    state["external_gap"] = final_gap.to_dict(); atomic(state_path, state)
    model.dispose()
    return state

def publish_arrays(path, **arrays):
    path = Path(path)
    if JOB.resolve() not in path.resolve().parents:
        raise ValueError('ARRAY_WRITE_OUTSIDE_REPAIR')
    if path.exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    with np.load(path) as z:
        for key,value in arrays.items():
            if not np.array_equal(z[key],value,equal_nan=True):
                raise ValueError('ARRAY_READBACK_'+key)
    return ref(path)


def save_native_exit(model, cb, binary, state, state_path, folder, status, bound):
    """Used by production AND tests. Commit payload before checkpoint; never evaluate/add rows."""
    import gurobipy as gp
    arrays={};pending=None;entries=[]
    n=len(binary)
    if model.SolCount:
        raw=np.asarray(model.X,float)
        if len(raw)!=n+1 or not np.isfinite(raw).all():
            raise ValueError('NATIVE_SNAPSHOT_DIMENSION_OR_FINITE')
        arrays['native_raw']=raw
        xx=raw[:n].copy()
        distance=float(np.max(abs(xx[binary]-np.rint(xx[binary])),initial=0))
        if distance<=1e-8:
            xx[binary]=np.rint(xx[binary]);arrays['native_rounded_first']=xx
            pid=hashlib.sha256(xx.tobytes()).hexdigest()
            entries.append({'kind':'NATIVE_INCUMBENT_UNAUDITED','point_id':pid,
                            'array_key':'native_rounded_first','eta':float(raw[-1]),
                            'binary_rounding_distance':distance})
    saved=cb.get('pending')
    if saved is not None:
        xx=np.asarray(saved['x'],float)
        if (len(xx)!=n or not np.isfinite(xx).all() or
            hashlib.sha256(xx.tobytes()).hexdigest()!=saved['point_id']):
            raise ValueError('PENDING_SNAPSHOT_IDENTITY')
        arrays['callback_pending_first']=xx
        entries.append({'kind':'CALLBACK_PENDING_UNAUDITED','point_id':saved['point_id'],
                        'array_key':'callback_pending_first','eta':float(saved['eta'])})
    payload=publish_arrays(folder/'NATIVE_CANDIDATES.npz',**arrays)
    for item in entries:
        item['payload']=payload
        if item['point_id'] not in state['separated_points']:
            pending=item
    old_ub=state['best_UB_audited'];old_best=state['best_plan'];old_lb=state['best_LB']
    valid_statuses={gp.GRB.OPTIMAL,gp.GRB.TIME_LIMIT,gp.GRB.INTERRUPTED,gp.GRB.MEM_LIMIT,
                    gp.GRB.NODE_LIMIT,gp.GRB.ITERATION_LIMIT,gp.GRB.USER_OBJ_LIMIT}
    used_bound=None
    if status in valid_statuses and bound.get('classification')=='FINITE':
        candidate=float(np.nextafter(float(bound['value'])-1e-6,-np.inf))
        if candidate>old_ub+1e-5:
            raise ValueError('NATIVE_BOUND_EXCEEDS_AUDITED_UB')
        if candidate>old_lb:
            state['best_LB']=candidate;used_bound=candidate
            state['best_LB_source']=ref(folder/'END.json')
    snapshot={'status':int(status),'epoch':time.time(),'model':state['model'],
              'rows':model.NumConstrs,'columns':model.NumVars,'nonzeros':model.NumNZs,
              'varname_sha256':hashlib.sha256(encoded(model.getAttr('VarName'))).hexdigest(),
              'entries':entries,'raw_payload':payload,'bound_observation':bound,
              'retained_LB':state['best_LB'],'new_bound_accepted':used_bound,
              'best_UB_audited':old_ub,'strict_UB':state.get('best_UB_certified'),
              'no_full400_UB_from_native_incumbent':True,'branch_tree_saved':False}
    atomic(folder/'NATIVE_SNAPSHOT.json',snapshot,True)
    state['last_native_snapshot']=ref(folder/'NATIVE_SNAPSHOT.json')
    state['pending_evaluation']=pending
    state['preserved_candidates']=entries
    state['phase']='EVALUATE_PENDING' if pending else 'SOLVE'
    if status==gp.GRB.MEM_LIMIT:
        state['status']='PAUSED_MEMORY_SAVED'
        state['automatic_retry']=False
        state['model_growth_after_memory_exit']=False
    assert state['best_UB_audited']==old_ub and state['best_plan']==old_best
    atomic(state_path,state)
    return snapshot


def seed_partial_cache(ac,ae,out,x,point_id):
    """Copy only RESULT metadata; immutable original primal/cut references stay at source."""
    science=sha(ARO_SRC/'SOURCE_LOCK.json');hits=[]
    for sid in range(400):
        task=ae.task_id(science,sid,x,'RECOURSE')
        dest=out/'EVALUATION_CACHE'/task/'RESULT.json'
        if dest.exists():continue
        for root in (OLD_JOB/'ARO',ARO_SRC):
            source=root/'EVALUATION_CACHE'/task/'RESULT.json'
            if not source.exists():continue
            rec=read(source)
            if rec['point_sha']!=point_id or int(rec['sid'])!=sid or rec['task_id']!=task:
                raise ValueError('PARTIAL_CACHE_IDENTITY')
            for item in rec['sources']:verify_ref(item)
            # Workers.evaluate independently replays the primal and dual before reuse.
            atomic(dest,rec,True);hits.append({'sid':sid,'source':ref(source)});break
    atomic(out/'CACHE_IMPORTS'/(point_id+'.json'),{'point_id':point_id,'hits':hits,
           'solve_calls':0,'certificate_replay_required_in_worker':True},True)


def controller():
    sys.path.insert(0,str(BASE/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'))
    import r23_common
    return r23_common


def protected_snapshot():
    entries=[]
    for name in ('R38_DET_PRODUCTION.json','R36_INTEGER_FINAL_V1.json'):
        p=CONTROL/'leases'/name;v=read(p);owner=v.get('owner') or v.get('supervisor')
        if not same(owner,proc(owner['pid'])):raise ValueError('PROTECTED_OWNER_CHANGED_'+name)
        code=[ref(x) for x in owner['argv'] if x.endswith('.py') and Path(x).is_file()]
        entries.append({'lease':str(p),'owner':proc(owner['pid']),'code':code,
                        'reserved_bytes':v['reserved_bytes']})
    det=BASE/'L08_DET_R38_CAPACITY_CONTINUATION_V1/CURRENT_CONTINUATION.json'
    value=read(det)
    if not same(value['owner'],proc(value['owner']['pid'])):raise ValueError('DET_WORKER_NOT_MATCHED')
    return {'epoch':time.time(),'entries':entries,'det_pointer':ref(det),
            'det_worker':proc(value['owner']['pid']),'production_signals_sent':[]}


def compare_protection(before):
    results=[]
    for entry in before['entries']:
        current=proc(entry['owner']['pid'])
        for x in entry['code']:verify_ref(x)
        nowlease=read(entry['lease'])
        if nowlease.get('reserved_bytes',0)!=entry['reserved_bytes']:
            raise ValueError('PROTECTED_RESERVATION_CHANGED')
        results.append({'owner_before':entry['owner'],'owner_now':current,
                        'same_live_owner':same(entry['owner'],current),'source_unchanged':True})
    verify_ref(before['det_pointer'])
    return {'epoch':time.time(),'owners':results,'production_signals_sent':[],
            'old_source_or_model_writes':False,'parameters_modified':False}


def safety_gate():
    w=read(OLD_JOB/'ATTEMPT_03/WAIT.json')
    if not w.get('actual_parent_wait') or w.get('returncode') is None:
        raise ValueError('NO_REAL_OLD_WAIT')
    if same(w['child'],proc(w['child']['pid'])):raise ValueError('OLD_ARO_STILL_LIVE')
    oldparent=read(OLD_JOB/'LAUNCHER.json')['owner']
    if same(oldparent,proc(oldparent['pid'])):raise ValueError('OLD_PARENT_STILL_LIVE')
    releases=[p for p in (OLD_JOB/'RESOURCE').glob('RELEASED_*.json')
              if read(p).get('actual_release') and
              read(p).get('lease',{}).get('owner',{}).get('pid')==oldparent['pid']]
    if not releases:raise ValueError('OLD_LEASE_RELEASE_MISSING')
    conflicts=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        q=proc(int(p.name))
        if not q:continue
        scripts=[x for x in q['argv'] if x.endswith('.py')]
        if any((str(ARO_SRC) in x or str(OLD_JOB) in x) and str(JOB) not in x for x in scripts):
            conflicts.append(q)
    if conflicts:raise ValueError('CONCURRENT_ARO_OWNER '+repr(conflicts))
    return {'old_wait':ref(OLD_JOB/'ATTEMPT_03/WAIT.json'),'old_child_exited':True,
            'old_parent_exited':True,'release':[ref(p) for p in releases],
            'conflicting_aro':[],'protection':protected_snapshot(),'signals_sent':[]}


def inventory_and_inherit():
    gate=safety_gate();atomic(JOB/'PREFLIGHT/SAFETY.json',gate,True)
    state=read(OLD_JOB/'ARO/STATE.json')
    for key in ('model','best_plan','best_LB_source'):verify_ref(state[key])
    lock=read(ARO_SRC/'SOURCE_LOCK.json')
    for item in lock['files']+lock['gates']:verify_ref(item)
    oldsources=[ref(OLD_JOB/'ARO/STATE.json'),ref(OLD_JOB/'ARO/ITER_0000/END.json'),
                ref(OLD_JOB/'RUN.json'),ref(OLD_JOB/'CONFIG.json'),ref(ARO_SRC/'SOURCE_LOCK.json')]
    manifest=[]
    for root in (ARO_SRC/'evaluations',OLD_JOB/'ARO/evaluations'):
        for p in sorted(root.glob('*/COMPLETE.json')):
            d=read(p)
            if len(d['records'])!=400 or {int(r['sid']) for r in d['records']}!=set(range(400)):
                raise ValueError('HISTORICAL_COMPLETE_NOT_400')
            pid=d.get('point_sha',p.parent.name)
            if pid!=p.parent.name or any(r['point_sha']!=pid for r in d['records']):
                raise ValueError('HISTORICAL_POINT_MIX')
            manifest.append({'point_id':pid,'complete':ref(p),'scenarios':400,
                             'source_payload_sha_rechecked_on_use':True})
    best=aro_cached_records(state['best_plan_id'])
    if best is None:raise ValueError('BEST_MISSING_COMPLETE400')
    gp,ac,ae,ad=aro_imports()
    x=np.load(verify_ref(state['best_plan']))['accepted']
    if hashlib.sha256(np.asarray(x,float).tobytes()).hexdigest()!=state['best_plan_id']:
        raise ValueError('BEST_PLAN_HASH')
    _,a,_=ac.first()
    F=float(np.nextafter(np.asarray(a['cost'],np.longdouble)@np.asarray(x,np.longdouble),np.inf))
    upper=float(np.nextafter(np.longdouble(F)+np.longdouble(max(r['qU_audited'] for r in best)),np.inf))
    if abs(upper-state['best_UB_audited'])>1e-7:raise ValueError('BEST_UB_REPLAY')
    state['previous_status']=state.pop('status',None)
    state['branch_tree_restored']=False;state['continuation_reason']='USER_AUTHORIZED_MEMORY_REPAIR'
    state['pending_evaluation']=None
    state['historical_evaluations_reused']=len({r['point_id'] for r in manifest})
    atomic(JOB/'ARO/STATE.json',state,True)
    atomic(JOB/'PREFLIGHT/INHERITANCE.json',{'sources':oldsources,'complete_evaluations':manifest,
           'best400_payload_references_verified':True,'best_UB_replayed':upper,
           'LB':state['best_LB'],'optimizer_calls':0,'branch_tree_restored':False},True)
    return gate


def test_child():
    import gurobipy as gp
    root=JOB/'TESTS';root.mkdir(parents=True,exist_ok=True)
    tests=[]
    # Native solve provides actual X; status injection below is explicitly a branch test.
    m=gp.Model('MEMORY_EXIT_SAVING_TEST');m.Params.OutputFlag=1
    m.Params.LogFile=str(root/'REAL_SMALL.log');m.Params.Threads=1
    x=m.addVars(5,vtype=gp.GRB.BINARY,name='x');eta=m.addVar(lb=0,name='ARO_ETA')
    m.addConstr(x.sum()>=2);m.setObjective(sum((i+1)*x[i] for i in range(5))+eta)
    m.optimize();assert m.Status==gp.GRB.OPTIMAL
    source=root/'TEST_MODEL.mps.gz';m.write(str(source))
    bestsource=root/'TEST_BEST.npz';publish_arrays(bestsource,accepted=np.asarray(m.X)[:5])
    base={'model':ref(source),'best_LB':0.,'best_UB_audited':9.,'best_UB_certified':None,
          'best_plan':ref(bestsource),'separated_points':[]}
    f=root/'SIMULATED_MEM_WITH_REAL_NATIVE_X';f.mkdir()
    atomic(f/'END.json',{'actual_native_status':int(m.Status),'injected_exit_status':17,
                        'test_type':'FAULT_INJECTION_AFTER_REAL_OPTIMAL_MIP'},True)
    xx=np.asarray(m.X)[:5];pid=hashlib.sha256(xx.tobytes()).hexdigest()
    cb={'pending':{'x':xx,'eta':0.,'point_id':pid}}
    state=json.loads(json.dumps(base))
    save_native_exit(m,cb,np.ones(5,dtype=bool),state,f/'STATE.json',f,17,
                     {'classification':'FINITE','value':float(m.ObjBoundC)})
    z=read(f/'STATE.json');assert z['status']=='PAUSED_MEMORY_SAVED'
    assert z['best_UB_audited']==9 and z['pending_evaluation']['point_id']==pid
    assert len(z['preserved_candidates'])==2 and z['best_LB']>2.99
    assert not z['model_growth_after_memory_exit']
    tests.append('memory_exit_saves_real_incumbent_pending_bound_without_false_UB')
    # Genuine native MEM_LIMIT, not simulated. Tiny cap only applies to this test model.
    mm=gp.Model('REAL_MEM_LIMIT_TEST');mm.Params.OutputFlag=1
    mm.Params.LogFile=str(root/'REAL_MEM_LIMIT.log');mm.Params.Threads=1
    y=mm.addVars(3000,vtype=gp.GRB.BINARY);mm.addConstr(y.sum()>=1501)
    mm.setObjective(sum((i%101+1)*y[i] for i in range(3000)))
    mm.Params.SoftMemLimit=1e-9;mm.optimize()
    if mm.Status!=gp.GRB.MEM_LIMIT:raise ValueError('REAL_MEMORY_EXIT_NOT_OBSERVED')
    ff=root/'REAL_MEM_LIMIT';ff.mkdir()
    atomic(ff/'END.json',{'actual_native_status':int(mm.Status),'runtime':mm.Runtime,
                         'max_mem_GB':mm.MaxMemUsed,'SolCount':mm.SolCount},True)
    state2=json.loads(json.dumps(base))
    save_native_exit(mm,{},np.ones(mm.NumVars-1,dtype=bool),state2,ff/'STATE.json',ff,17,
                     {'classification':'MISSING','value':None})
    assert state2['best_LB']==0 and state2['best_UB_audited']==9
    tests.append('actual_native_MEM_LIMIT_without_incumbent_preserves_old_bounds')
    bad=root/'REJECT_BAD_ID';bad.mkdir();atomic(bad/'END.json',{'test':True},True)
    try:
        save_native_exit(m,{'pending':dict(cb['pending'],point_id='wrong')},np.ones(5,bool),
                         json.loads(json.dumps(base)),bad/'STATE.json',bad,17,{'classification':'MISSING'})
    except ValueError:pass
    else:raise AssertionError('bad pending identity accepted')
    assert not (bad/'STATE.json').exists();tests.append('wrong_candidate_identity_rejected_without_checkpoint')
    # Publication failure must propagate; no falsely successful state/ACK.
    fail=root/'INJECT_WRITE_ERROR';fail.mkdir();atomic(fail/'END.json',{'test':True},True)
    (fail/'NATIVE_CANDIDATES.npz').mkdir()
    try:save_native_exit(m,cb,np.ones(5,bool),json.loads(json.dumps(base)),fail/'STATE.json',fail,17,{'classification':'MISSING'})
    except FileExistsError:pass
    else:raise AssertionError('write error swallowed')
    assert not (fail/'STATE.json').exists();tests.append('write_error_is_non_success_without_state_commit')
    # Check all configured production params by applying them to an actual native Model.
    m.Params.SoftMemLimit=24;m.Params.Threads=4;m.Params.NodefileStart=.5
    assert m.Params.SoftMemLimit==24 and m.Params.Threads==4
    tests.append('native_parameter_24_decimalGB_threads4_nodefile_applied')
    # Real cache reuse: optimize() must never be called while checking prior COMPLETE.
    old=gp.Model.optimize
    def forbidden(*a,**k):raise AssertionError('CACHE_REUSE_OPTIMIZE_FORBIDDEN')
    gp.Model.optimize=forbidden
    try:
        st=read(OLD_JOB/'ARO/STATE.json');records=aro_cached_records(st['best_plan_id'])
        assert len(records)==400
        try:aro_validate_records(records[:-1],st['best_plan_id'])
        except ValueError:pass
        else:raise AssertionError('partial400 accepted')
        try:aro_validate_records(records,'wrong')
        except ValueError:pass
        else:raise AssertionError('mixed point accepted')
    finally:gp.Model.optimize=old
    tests.extend(['real_best400_reuse_zero_optimizer_calls','partial400_and_mixed_plan_rejected'])
    m.dispose();mm.dispose()
    atomic(root/'RESULT.json',{'passed':True,'checks':tests,'native_optimizer_calls':2,
           'production_optimizer_calls':0,'real_native_memory_exit':True,
           'candidate_memory_branch':'fault injection using actual solved native X',
           'scientific_model_or_tolerance_changes':False},True)


def worker():
    started=time.monotonic();code=0
    atomic(JOB/'EXECUTION/BEGIN.json',{'owner':proc(os.getpid()),'epoch':time.time(),
           'remaining_s':remaining(read(JOB/'RUN.json'))},True)
    try:
        for item in read(JOB/'FROZEN_CODE.json')['files']:verify_ref(item)
        aro=aro_stage(read(JOB/'RUN.json'),JOB/'EXECUTION',0)
        result={'ARO':aro,'status':aro.get('status'),'production_signals_sent':[],
                'wall':time.monotonic()-started}
    except BaseException as exc:
        code=70;result={'status':'FAILED_PRESERVED','error':repr(exc),'traceback':traceback.format_exc(),
                        'wall':time.monotonic()-started,'production_signals_sent':[]}
    finally:
        for model in MODEL_REGISTRY:
            try:model.dispose()
            except Exception:pass
    atomic(JOB/'EXECUTION/RESULT.json',result,True)
    atomic(JOB/'EXECUTION/END.json',{'epoch':time.time(),'returncode':code,'status':result['status']},True)
    return code


def run_wait(command,folder,lease_path=None,lease=None):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'STDOUT.log').open('xb') as stdout,(folder/'STDERR.log').open('xb') as stderr:
        p=subprocess.Popen(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL)
        ident=None
        for _ in range(100):
            ident=proc(p.pid)
            if ident and ident['argv']:break
            time.sleep(.01)
        atomic(folder/'PROCESS.json',{'child':ident,'parent':proc(os.getpid())},True)
        while p.poll() is None:
            if lease_path and lease:
                # PSS monitor only requests cooperative pause of this ARO, never signals production.
                rc=controller();procs=rc.ctrl.allproc()
                family=rc.ctrl.descendants(rc.ctrl.proc(os.getpid()),procs)
                pss=sum(v['pss'] or 0 for v in family)
                atomic(JOB/'RESOURCE/PROGRESS.json',{'epoch':time.time(),'tree_pss_bytes':pss,
                       'reserved_bytes':40*1024**3,'child':proc(p.pid)})
                if pss>40*1024**3:
                    atomic(JOB/'ARO/PAUSE_RESOURCE.json',{'reason':'PROCESS_TREE_RESERVATION_EXCEEDED',
                           'pss_bytes':pss,'epoch':time.time()})
            time.sleep(5)
        code=p.wait()
    atomic(folder/'WAIT.json',{'returncode':code,'actual_parent_wait':True,
           'child':ident,'epoch':time.time()},True)
    return code


def parent():
    lease_path=CONTROL/'leases/ARO_MEMORY_EXIT_REPAIR_V1.json';lease=None;gate=None
    rc=controller()
    # Existing owner lock prevents another continuation of the same ARO instance.
    with (OLD_JOB/'OWNER.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            gate=inventory_and_inherit()
            run=read(OLD_JOB/'RUN.json')
            if remaining(run)<=600:raise ValueError('ORIGINAL_GLOBAL_BUDGET_INSUFFICIENT')
            atomic(JOB/'RUN.json',run,True)
            with rc.ctrl.lock(CONTROL/'DISPATCH.lock'):
                safety_gate();res=rc.resources();need=40*1024**3
                passed=not res['unknown_scheduler_limits'] and res['available_for_new_bytes']>=need
                atomic(JOB/'RESOURCE/ADMISSION.json',{'admitted':passed,'requested_bytes':need,
                       'resources':res,'solver_decimal_GB':24,'epoch':time.time()},True)
                if not passed:
                    atomic(JOB/'STATUS.json',{'status':'WAITING_FOR_RESOURCES','retryable':True,
                           'missing_bytes':max(0,need-res['available_for_new_bytes'])});return 75
                if lease_path.exists() and read(lease_path).get('reserved_bytes',0):
                    raise ValueError('EXISTING_ARO_REPAIR_LEASE')
                lease={'lease_id':uuid.uuid4().hex,'owner':rc.ctrl.proc(os.getpid()),
                       'reserved_bytes':need,'threads':4,'method':ARO_SCOPE,
                       'root':str(JOB),'epoch':time.time(),'production_signals_sent':[]}
                rc.ctrl.dump(lease_path,lease)
            # Reuse the successful real tests; only the parent scope guard/path changed.
            prior=OLD_JOB/'MEMORY_EXIT_REPAIR_V1'
            import ast
            previous_tree=ast.parse((prior/'tools/run_aro_memory_repair.py').read_text())
            current_tree=ast.parse(Path(__file__).read_text())
            funcs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
            aa,bb=funcs(previous_tree),funcs(current_tree)
            changed=[n for n in aa if aa[n]!=bb.get(n)]
            if changed!=['parent']:raise ValueError(('UNTESTED_FUNCTION_CHANGE',changed))
            if not read(prior/'TESTS/RESULT.json')['passed'] or read(prior/'TEST_PARENT/WAIT.json')['returncode']!=0:
                raise ValueError('PREVIOUS_TEST_NOT_PASSED')
            atomic(JOB/'TEST_REUSE.json',{'passed':True,'result':ref(prior/'TESTS/RESULT.json'),
                   'wait':ref(prior/'TEST_PARENT/WAIT.json'),'tested_code':ref(prior/'tools/run_aro_memory_repair.py'),
                   'all_tested_functions_AST_identical':True,'changed_functions':changed,
                   'new_native_test_calls':0},True)
            # Verify full patched execution AST, including the memory-save-before-pause ordering.
            import ast
            source=Path(__file__).read_text();ast.parse(source)
            def has_dro_call(text):
                return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dro_check'
                           for n in ast.walk(ast.parse(text)))
            assert not has_dro_call("x='dro_check()'")
            assert has_dro_call('dro_check(attempt)')
            if has_dro_call(source):raise ValueError('DRO_EXECUTION_PATH_PRESENT')
            atomic(JOB/'PREFLIGHT/SCOPE_GUARD_TEST.json',{'passed':True,'AST_actual_calls_only':True,
                   'string_literal_not_a_call':True,'actual_DRO_call_rejected':True,'optimizer_calls':0},True)
            atomic(JOB/'FROZEN_CODE.json',{'files':[ref(p) for p in sorted(TOOLS.glob('*.py'))]},True)
            safety_gate()
            protection=compare_protection(gate['protection'])
            if not all(x['same_live_owner'] for x in protection['owners']):raise ValueError('PROTECTED_OWNER_NOT_LIVE')
            atomic(JOB/'PREFLIGHT/PROTECTION_BEFORE_NATIVE.json',protection,True)
            atomic(JOB/'STATUS.json',{'status':'TESTED_STARTING_ARO','epoch':time.time()})
            code=run_wait([str(PYTHON),'-B',str(Path(__file__)),'worker'],JOB/'EXECUTION',lease_path,lease)
            atomic(JOB/'STATUS.json',{'status':read(JOB/'EXECUTION/RESULT.json')['status'],
                   'actual_parent_wait':True,'returncode':code,'epoch':time.time()})
            return code
        except BaseException as exc:
            atomic(JOB/'PARENT_FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc(),
                   'epoch':time.time(),'production_signals_sent':[]},True)
            return 70
        finally:
            if lease:
                with rc.ctrl.lock(CONTROL/'DISPATCH.lock'):
                    actual=read(lease_path)
                    if actual['lease_id']!=lease['lease_id']:raise ValueError('LEASE_RELEASE_MISMATCH')
                    atomic(JOB/'RESOURCE/RELEASED.json',{'lease':actual,'epoch':time.time(),
                           'actual_release':True},True)
                    rc.ctrl.dump(lease_path,dict(actual,reserved_bytes=0,owner=None,released_epoch=time.time()))
            if gate:
                atomic(JOB/'PROTECTION_AFTER.json',compare_protection(gate['protection']),True)


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'status'
    if mode=='tests':test_child();return 0
    if mode=='worker':return worker()
    if mode=='parent':return parent()
    if mode=='status':
        print(json.dumps({p.name:read(p) for p in (JOB/'STATUS.json',JOB/'EXECUTION/RESULT.json') if p.exists()}));return 0
    raise ValueError(mode)


if __name__=='__main__':
    raise SystemExit(main())
