import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REV = Path(__file__).resolve().parent
R = REV.parents[1]
TOOLS = R / "tools"
sys.path.insert(0, str(REV))
sys.path.insert(1, str(TOOLS))

from r36_common import B, V, ctrl, admission, read, dump, sha
import r36_common

OUT = R / "INTEGER_FINAL_V1"
LEASE = V / "leases/R36_INTEGER_FINAL_V1.json"
r36_common.LEASE = LEASE


def proc(pid):
    d = Path("/proc") / str(pid)
    if not d.exists():
        return None
    raw = (d / "stat").read_text()
    cut = raw.rfind(") ")
    rest = raw[cut + 2 :].split()
    return {
        "pid": pid,
        "state": rest[0],
        "ppid": int(rest[1]),
        "start_ticks": int(rest[19]),
        "argv": (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"),
    }


def preflight():
    OUT.mkdir(exist_ok=True)
    det = proc(2850521)
    aro = proc(3006841)
    assert det and "L08_DET_R38_CAPACITY_CONTINUATION_V1" in det["argv"]
    assert aro and "L08_ARO_DRO_GAP1_ACCELERATION_V1" in aro["argv"]
    old = read(R / "formal/CHECKPOINT.json")
    assert old["science"] == "L08_K4L4_FULL_ST_DRO400_R36"
    assert len(old["pool"]) == 93
    model = Path(old["attempt"]) / "LATEST_MODEL.mps.gz"
    assert model.exists()
    root_result = read(B / "L08_ARO_DRO_GAP1_ACCELERATION_V1/DRO/RESULT.json")
    assert root_result["support"] == {"probability": 400, "recourse": 93, "remaining_307_Q_lower": 0}
    prior = ctrl.read(LEASE)
    assert not ctrl.same(prior.get("owner")), "DUPLICATE_INTEGER_DRO_OWNER"
    oldlease = ctrl.read(V / "leases/R36_K4L4_FULL_ST.json")
    assert not ctrl.same(oldlease.get("owner")), "OLD_R36_OWNER_STILL_ALIVE"
    rec = {
        "passed": True,
        "epoch": time.time(),
        "method": "INTEGER_DRO_SEPARATE_FROM_ARO",
        "source_checkpoint": {"path": str(R / "formal/CHECKPOINT.json"), "sha256": sha(R / "formal/CHECKPOINT.json")},
        "source_model": {"path": str(model), "sha256": sha(model), "bytes": model.stat().st_size},
        "pool_size": 93,
        "probability_support": 400,
        "inherited_continuous_root_LB": root_result["LB"],
        "DET": det,
        "ARO": aro,
        "ARO_bounds_inherited": False,
        "production_signals_sent": [],
        "resource_request_GiB": 40,
        "threads": 4,
    }
    dump(OUT / "PREFLIGHT.json", rec)
    return rec


def child():
    preflight()
    from r36_integer_master import run

    begin = {"epoch": time.time(), "owner": proc(os.getpid()), "method": "INTEGER_DRO_SEPARATE_FROM_ARO"}
    dump(OUT / "BEGIN.json", begin)
    status = "UNKNOWN"
    try:
        with ctrl.lock(R / "INTEGER_FINAL_V1_OWNER.lock"), admission("INTEGER_FINAL_V1", 40, 4):
            run()
        status = "RETURNED"
    except BaseException as exc:
        status = "FAILED"
        dump(OUT / "FAILURE.json", {"epoch": time.time(), "error": repr(exc), "type": type(exc).__name__})
        raise
    finally:
        dump(OUT / "END.json", {"epoch": time.time(), "status": status})


def parent():
    preflight()
    logs = OUT / "PARENT"
    logs.mkdir(exist_ok=True)
    out = (logs / "stdout.log").open("ab")
    err = (logs / "stderr.log").open("ab")
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--child"]
    p = subprocess.Popen(cmd, stdout=out, stderr=err)
    dump(logs / "PROCESS.json", {"epoch": time.time(), "parent": os.getpid(), "child": proc(p.pid), "cmd": cmd})
    rc = p.wait()
    dump(logs / "WAIT.json", {"epoch": time.time(), "actual_parent_wait": True, "returncode": rc, "child_pid": p.pid})
    out.close()
    err.close()
    raise SystemExit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--parent", action="store_true")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.preflight:
        print(json.dumps(preflight(), ensure_ascii=False))
    elif a.child:
        child()
    elif a.parent:
        parent()
    else:
        raise SystemExit("choose --preflight, --parent, or --child")
