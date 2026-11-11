#!/usr/bin/env python3
"""Auditable, detached, single-job execution. Python 3.12+, Linux/macOS stdlib."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED", "LAUNCH_FAILED"}
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def launch_lock(outroot):
    with (Path(outroot) / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def process_identity(pid):
    """Linux start ticks distinguish PID reuse; macOS falls back to ps start time."""
    try:
        if sys.platform.startswith("linux"):
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return {
                "pid": int(pid), "token": f"linux:{boot}:{fields[19]}",
                "pgid": int(fields[2]), "sid": int(fields[3]),
                "state": fields[0],
            }
        text = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat="],
            check=False, capture_output=True, text=True,
        ).stdout.strip()
        if not text:
            return None
        start, state = text.rsplit(None, 1)
        return {
            "pid": int(pid), "token": f"ps:{start}",
            "pgid": os.getpgid(pid), "sid": os.getsid(pid), "state": state,
        }
    except (ProcessLookupError, FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def identity_alive(identity):
    if not identity:
        return False
    current = process_identity(identity["pid"])
    return bool(current and current["token"] == identity["token"]
                and not current["state"].startswith("Z"))


def group_members(pgid, sid):
    """Return live members of this exact group AND supervisor-owned session."""
    if not pgid or not sid:
        return []
    if sys.platform.startswith("linux"):
        pids = (int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit())
    else:
        output = subprocess.run(
            ["ps", "-ax", "-o", "pid=", "-o", "pgid="],
            capture_output=True, text=True, check=True,
        ).stdout
        pids = [int(parts[0]) for line in output.splitlines()
                if len(parts := line.split()) == 2 and int(parts[1]) == pgid]
    members = []
    for pid in pids:
        identity = process_identity(pid)
        if (identity and identity["pgid"] == pgid and identity["sid"] == sid
                and not identity["state"].startswith("Z")):
            members.append(identity)
    return members


def observation(status):
    supervisor = status.get("supervisor")
    command = status.get("command_process")
    supervisor_alive = identity_alive(supervisor)
    command_alive = identity_alive(command)
    # A session ID can eventually be reused after its old session disappears.
    # Never attribute a new session with that numeric ID to an old job.
    session_generation_matches = bool(supervisor)
    if supervisor:
        current_supervisor = process_identity(supervisor["pid"])
        if current_supervisor and current_supervisor["token"] != supervisor["token"]:
            session_generation_matches = False
        if supervisor["token"].startswith("linux:"):
            current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if supervisor["token"].split(":")[1] != current_boot:
                session_generation_matches = False
    members = (group_members(status.get("command_pgid"), status.get("supervisor_sid"))
               if session_generation_matches else [])
    terminal = status.get("state") in TERMINAL
    if terminal:
        health = "COMPLETE" if not members else "UNEXPECTED_LIVE_GROUP"
    elif supervisor_alive:
        health = "SUPERVISED"
    elif command_alive or members:
        health = "SUPERVISOR_LOST_JOB_STILL_ALIVE"
    else:
        health = "SUPERVISOR_MISSING_REQUIRES_REVIEW"
    return {
        "observed_at_utc": utcnow(), "health": health,
        "supervisor_alive": supervisor_alive, "command_alive": command_alive,
        "live_group_members": members,
        # An interrupted supervisor must be reviewed, never silently retried.
        "blocks_new_launch": not terminal or bool(members),
    }


def checked_request(request):
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    allowed = {"job_id", "argv", "cwd", "outroot", "threads", "timeout_seconds",
               "grace_seconds", "spec_id", "purpose"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    job_id = request.get("job_id", "")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", job_id):
        raise ValueError("job_id must be 1-80 letters/digits/dots/dashes/underscores, starting alphanumeric")
    argv = request.get("argv")
    if (not isinstance(argv, list) or not argv or not argv[0]
            or any(not isinstance(v, str) or "\0" in v for v in argv)):
        raise ValueError("argv must be a nonempty JSON list of strings; no shell command string")
    for key in ("cwd", "outroot"):
        if not isinstance(request.get(key), str) or not Path(request[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    cwd = Path(request["cwd"]).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("cwd is not a directory")
    threads = request.get("threads", 1)
    if type(threads) is not int or not 1 <= threads <= 256:
        raise ValueError("threads must be an integer from 1 through 256")
    settings = dict(request, cwd=str(cwd), outroot=str(Path(request["outroot"]).resolve()), threads=threads)
    for key, default in (("timeout_seconds", 7200), ("grace_seconds", 5)):
        value = request.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
        settings[key] = value
    return settings


def launch(request):
    settings = checked_request(request)
    outroot = Path(settings["outroot"])
    outroot.mkdir(parents=True, exist_ok=True)
    jobdir = outroot / settings["job_id"]
    with launch_lock(outroot):
        if jobdir.exists():
            raise ValueError(f"job_id already exists and will not be overwritten: {jobdir}")
        for candidate in sorted(outroot.iterdir()):
            if not candidate.is_dir() or not (candidate / "manifest.json").exists():
                continue
            status_path = candidate / "status.json"
            if not status_path.exists():
                raise ValueError(f"Incomplete job requires review before a new launch: {candidate.name}")
            existing = read_json(status_path)
            if observation(existing)["blocks_new_launch"]:
                raise ValueError(f"One active or unreviewed job per outroot; blocked by {candidate.name}")
        jobdir.mkdir(mode=0o700)
        manifest = {
            "schema_version": 1, "created_at_utc": utcnow(),
            **settings, "max_active_jobs": 1,
            "thread_environment": {key: str(settings["threads"]) for key in THREAD_VARS},
            "supervisor_python": sys.executable, "controller_path": str(Path(__file__).resolve()),
            "host": os.uname().nodename, "uid": os.getuid(),
        }
        atomic_json(jobdir / "manifest.json", manifest)
        state = {"job_id": settings["job_id"], "state": "LAUNCHING", "updated_at_utc": utcnow()}
        atomic_json(jobdir / "status.json", state)
        for name in ("stdout.log", "stderr.log"):
            (jobdir / name).touch()
        atomic_json(jobdir / "result.json", {"complete": False, "job_id": settings["job_id"]})
        try:
            with (jobdir / "supervisor.stdout.log").open("ab") as stdout, (jobdir / "supervisor.stderr.log").open("ab") as stderr:
                supervisor = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "_supervise", str(jobdir / "manifest.json")],
                    cwd=outroot, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    start_new_session=True, close_fds=True,
                )
            state.update(supervisor=process_identity(supervisor.pid), supervisor_pid=supervisor.pid,
                         supervisor_sid=supervisor.pid, updated_at_utc=utcnow())
            atomic_json(jobdir / "status.json", state)
        except Exception as error:
            state.update(state="LAUNCH_FAILED", error=str(error), updated_at_utc=utcnow())
            atomic_json(jobdir / "result.json", dict(state, complete=True))
            atomic_json(jobdir / "status.json", state)
            raise
    return {"accepted": True, "job_id": settings["job_id"], "job_directory": str(jobdir),
            "supervisor_pid": supervisor.pid, "status": state["state"]}


def supervise(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    jobdir = manifest_path.parent
    manifest = read_json(manifest_path)
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda sig, frame: stop_requested.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stop_requested.append(sig))
    # Parent holds this lock until it has written the supervisor identity.
    with launch_lock(manifest["outroot"]):
        state = read_json(jobdir / "status.json")
        state.update(supervisor=process_identity(os.getpid()), supervisor_pid=os.getpid(),
                     supervisor_sid=os.getsid(0), updated_at_utc=utcnow())
        atomic_json(jobdir / "status.json", state)
    if os.getsid(0) != os.getpid():
        raise RuntimeError("Supervisor must own its detached session")
    process = None
    actions = []
    started = time.monotonic()

    def update(**values):
        state.update(values, updated_at_utc=utcnow())
        atomic_json(jobdir / "status.json", state)

    def signal_owned_group(sig):
        if process is None or os.getsid(0) != os.getpid():
            return
        # All matching members belong to the session created by this supervisor.
        # An unrelated reused PID/session cannot match while this supervisor lives.
        members = group_members(process.pid, os.getpid())
        if not members:
            return
        try:
            os.killpg(process.pid, sig)
            actions.append({"signal": signal.Signals(sig).name, "at_utc": utcnow(),
                            "pgid": process.pid, "sid": os.getpid(), "members": members})
        except ProcessLookupError:
            pass

    def terminate_group():
        signal_owned_group(signal.SIGTERM)
        deadline = time.monotonic() + manifest["grace_seconds"]
        while time.monotonic() < deadline:
            if process is not None:
                process.poll()
            if process is None or not group_members(process.pid, os.getpid()):
                break
            time.sleep(0.05)
        if process is not None and group_members(process.pid, os.getpid()):
            signal_owned_group(signal.SIGKILL)
        if process is not None:
            process.wait(timeout=5)

    try:
        env = os.environ.copy()
        env.update(manifest["thread_environment"])
        env["EVCS_JOB_ID"] = manifest["job_id"]
        env["EVCS_JOB_DIRECTORY"] = str(jobdir)
        with (jobdir / "stdout.log").open("ab", buffering=0) as stdout, (jobdir / "stderr.log").open("ab", buffering=0) as stderr:
            process = subprocess.Popen(
                manifest["argv"], cwd=manifest["cwd"], env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True, process_group=0,
            )
            # Child has its own process group within this supervisor's session.
            update(state="RUNNING", started_at_utc=utcnow(), command_pid=process.pid,
                   command_pgid=process.pid, command_process=process_identity(process.pid))
            reason = None
            while process.poll() is None:
                if stop_requested:
                    reason = "CANCELLED"
                    break
                if time.monotonic() - started >= manifest["timeout_seconds"]:
                    reason = "TIMED_OUT"
                    break
                time.sleep(0.05)
            if reason:
                update(state="STOPPING", stop_reason=reason)
                terminate_group()
                final_state = reason
            else:
                final_state = "SUCCEEDED" if process.returncode == 0 else "FAILED"
                if group_members(process.pid, os.getpid()):
                    reason = "CHILDREN_REMAINED_AFTER_COMMAND_EXIT"
                    update(state="STOPPING", stop_reason=reason)
                    terminate_group()
                    final_state = "FAILED"
        final = dict(state, state=final_state, complete=True, returncode=process.returncode,
                     reason=reason, elapsed_seconds=round(time.monotonic() - started, 6),
                     finished_at_utc=utcnow(), updated_at_utc=utcnow(), termination_actions=actions)
    except Exception as error:
        cleanup_error = None
        if process is not None:
            try:
                terminate_group()
            except Exception as cleanup:
                cleanup_error = repr(cleanup)
        final = dict(state, state="LAUNCH_FAILED" if process is None else "FAILED", complete=True,
                     error=repr(error), cleanup_error=cleanup_error,
                     returncode=None if process is None else process.returncode,
                     elapsed_seconds=round(time.monotonic() - started, 6),
                     finished_at_utc=utcnow(), updated_at_utc=utcnow(), termination_actions=actions)
    atomic_json(jobdir / "result.json", final)
    atomic_json(jobdir / "status.json", final)
    return 0


def status(outroot, job_id=None):
    root = Path(outroot).resolve(strict=True)
    if job_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", job_id):
        raise ValueError("Invalid job_id")
    paths = [root / job_id] if job_id else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists())
    jobs = []
    for path in paths:
        manifest = read_json(path / "manifest.json")
        state = read_json(path / "status.json")
        jobs.append({"job_directory": str(path), "manifest": manifest, "status": state,
                     "observation": observation(state), "result": read_json(path / "result.json")})
    return {"outroot": str(root), "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch_parser = sub.add_parser("launch", help="Launch once; structured JSON request, '-' means stdin")
    launch_parser.add_argument("--request", required=True)
    status_parser = sub.add_parser("status", help="Read state and process identities without altering jobs")
    status_parser.add_argument("--outroot", required=True)
    status_parser.add_argument("--job-id")
    supervise_parser = sub.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise_parser.add_argument("manifest")
    args = parser.parse_args()
    try:
        if args.action == "_supervise":
            return supervise(args.manifest)
        if args.action == "launch":
            request = json.load(sys.stdin) if args.request == "-" else read_json(args.request)
            result = launch(request)
        else:
            result = status(args.outroot, args.job_id)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
