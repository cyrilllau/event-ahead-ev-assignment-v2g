"""Case0-only transient cgroup wrapper; never changes the user's parent slice."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

PROPERTIES = {"CPUQuota": "100%", "MemoryHigh": "6G", "MemoryMax": "8G",
              "MemorySwapMax": "0", "TasksMax": "64", "Nice": "10",
              "KillMode": "control-group"}


def unit_name(value):
    if not re.fullmatch(r"case0-[a-zA-Z0-9_.-]{1,64}", value):
        raise ValueError("Only explicitly named Case0 child units may be controlled")
    return value.removesuffix(".service") + ".service"


def isolated_command(unit, command, seconds, cpu, *, retain_failed_unit=False):
    unit = unit_name(unit)
    if not 0 < seconds <= 7200 or cpu < 0 or not command:
        raise ValueError("Invalid bounded Case0 job")
    properties = dict(PROPERTIES, RuntimeMaxSec=str(seconds), CPUAffinity=str(cpu))
    argv = ["systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
            "--unit", unit, "--working-directory", os.getcwd()]
    if retain_failed_unit:
        if not unit.startswith('case0-expanded-'):
            raise ValueError('Retained diagnostics require an expanded Case0 unit')
        argv.remove('--collect')
    for key, value in properties.items():
        argv += ["--property", f"{key}={value}"]
    for key in ("EVCS_CONFIG", "EVCS_JOB_ID", "EVCS_JOB_DIRECTORY"):
        if key in os.environ:
            argv += ["--setenv", f"{key}={os.environ[key]}"]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        argv += ["--setenv", f"{key}=1"]
    return [*argv, "--", *command]


def verify_current_cgroup():
    """Check actual kernel controls and membership before importing a solver."""
    raw = Path("/proc/self/cgroup").read_text().strip()
    relative = raw.split("0::", 1)[1]
    if not Path(relative).name.startswith("case0-") or not relative.endswith(".service"):
        raise RuntimeError("Case0 must run in its own child service cgroup")
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    values = {name: (root / name).read_text().strip() for name in
              ("cpu.max", "memory.high", "memory.max", "memory.swap.max", "pids.max")}
    quota, period = map(int, values["cpu.max"].split())
    if (quota > period or int(values["memory.high"]) > 6 * 1024**3 or
            int(values["memory.max"]) > 8 * 1024**3 or values["memory.swap.max"] != "0"
            or int(values["pids.max"]) > 64 or len(os.sched_getaffinity(0)) != 1):
        raise RuntimeError(f"Case0 resource controls not effective: {values}")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("Case0 low priority not effective")
    return {"cgroup": relative, "effective_controls": values,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit")
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--retain-failed-unit", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.probe:
        record = verify_current_cgroup()
        child = subprocess.check_output([sys.executable, "-c",
            "from pathlib import Path; print(Path('/proc/self/cgroup').read_text().strip())"], text=True)
        if record["cgroup"] not in child:
            raise RuntimeError("Child escaped Case0 resource cgroup")
        record["child_in_same_cgroup"] = True
        print(json.dumps(record))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    unit = unit_name(args.unit or "")
    proc = subprocess.Popen(isolated_command(unit, command, args.seconds, args.cpu,
        retain_failed_unit=args.retain_failed_unit))
    def stop_owned_unit(signum, frame):
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    signal.signal(signal.SIGTERM, stop_owned_unit)
    signal.signal(signal.SIGINT, stop_owned_unit)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
