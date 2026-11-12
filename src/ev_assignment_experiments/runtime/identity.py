"""Local process identity: host/boot/create-time/argv, never PID alone."""

import hashlib
import json
import os
import socket
import time
import psutil


def observe(pid):
    try:
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return None
        return dict(
            host=socket.gethostname(),
            boot=psutil.boot_time(),
            pid=pid,
            created=p.create_time(),
            argv_sha256=hashlib.sha256(json.dumps(p.cmdline()).encode()).hexdigest(),
        )
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied:
        return {"unknown": True, "pid": pid}


def matches(identity):
    if not identity or identity.get("host") != socket.gethostname():
        return None
    actual = observe(identity["pid"])
    if actual is None:
        return False
    if actual.get("unknown"):
        return None
    return actual == identity


def current():
    value = observe(os.getpid())
    if not value or value.get("unknown"):
        raise RuntimeError("Cannot establish process identity")
    return value


def wait_identity(pid, seconds=3.0):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        value = observe(pid)
        if value and not value.get("unknown"):
            return value
        time.sleep(0.01)
    raise RuntimeError("Child exited before process identity could be observed")
