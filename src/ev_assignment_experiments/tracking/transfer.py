"""Streaming, resumable, additive transfer of verified sealed dependency closures."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from .store import Ref, Store, encode, sha256

CHUNK = 1024 * 1024
MAX_RECORD = 16 * CHUNK


def file_digest(path, length=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        remaining = length
        while remaining is None or remaining:
            chunk = handle.read(CHUNK if remaining is None else min(CHUNK, remaining))
            if not chunk:
                if remaining:
                    raise ValueError("Truncated prefix")
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


class FilesystemSource:
    def __init__(self, store):
        self.store = store

    def _path(self, ref):
        path = self.store.path(ref)
        if path.is_symlink() or not path.resolve().is_relative_to(self.store.root):
            raise ValueError("Store paths must not escape through symlinks")
        return path

    def size(self, ref):
        return self._path(ref).stat().st_size

    def digest(self, ref, length):
        return file_digest(self._path(ref), length)

    @contextmanager
    def open(self, ref, offset=0):
        with self._path(ref).open("rb") as handle:
            handle.seek(offset)
            yield handle


class SSHSource:
    """Read-only transfer; caller chooses the host and absolute sealed store."""

    def __init__(self, host, root):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*", host):
            raise ValueError("Invalid SSH host/alias")
        if not root.startswith("/") or "\x00" in root:
            raise ValueError("An absolute remote store is required")
        from ev_assignment_experiments.runtime import remote_reader

        self.program = Path(remote_reader.__file__).read_text()
        self.host, self.root = host, root

    def command(self, action, ref, count):
        # OpenSSH joins its remote argv through the remote shell. Quote every
        # argument as shell text; no user value becomes a command or option.
        command = shlex.join(
            ["python3", "-c", self.program, self.root, action, ref.kind, ref.id, str(count)]
        )
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            self.host,
            command,
        ]

    def size(self, ref):
        output = subprocess.check_output(self.command("size", ref, 0), timeout=45)
        size = json.loads(output)["size"]
        if type(size) is not int or size < 0:
            raise ValueError("Invalid remote size")
        return size

    def digest(self, ref, length):
        value = subprocess.check_output(self.command("digest", ref, length)).decode().strip()
        if not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("Invalid remote digest")
        return value

    @contextmanager
    def open(self, ref, offset=0):
        process = subprocess.Popen(self.command("read", ref, offset), stdout=subprocess.PIPE)
        try:
            yield process.stdout
            if process.wait() != 0:
                raise OSError("Read-only SSH transfer failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()  # Only our local SSH child, never a solver.
            process.wait()


def record(source, ref):
    if ref.kind == "objects" or source.size(ref) > MAX_RECORD:
        raise ValueError("Expected bounded record")
    with source.open(ref) as handle:
        data = handle.read(MAX_RECORD + 1)
    if len(data) > MAX_RECORD or sha256(data) != ref.id:
        raise ValueError("Record digest mismatch or record too large")
    value = json.loads(data)
    if (
        set(value) != {"schema_version", "kind", "payload", "refs"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["kind"] != ref.kind
        or not isinstance(value["payload"], dict)
    ):
        raise ValueError("Invalid record schema")
    for item in value["refs"]:
        Ref(**item)
    return value


def closure(source, roots):
    pending, visited = list(roots), set()
    while pending:
        ref = pending.pop()
        if ref in visited:
            continue
        if ref.kind != "objects":
            pending.extend(Ref(**r) for r in record(source, ref)["refs"])
        else:
            source.size(ref)
        visited.add(ref)
    return sorted(visited)


def copy_one(source, destination, ref):
    path = destination.path(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.parent.resolve().is_relative_to(destination.root):
        raise ValueError("Destination escapes store")
    area = destination.root / ".transfers"
    area.mkdir(exist_ok=True)
    if area.is_symlink():
        raise ValueError("Transfer staging must not be a symlink")
    part = area / (ref.kind + "-" + ref.id + ".part")
    lock_path = area / (ref.kind + "-" + ref.id + ".lock")
    if part.is_symlink() or lock_path.is_symlink():
        raise ValueError("Transfer staging must not be a symlink")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if file_digest(path) != ref.id:
                raise ValueError("Digest mismatch in existing destination; refusing overwrite")
            return {"reused": True, "resumed_bytes": 0, "copied_bytes": 0}
        size = source.size(ref)
        offset = part.stat().st_size if part.exists() else 0
        if offset > size or (offset and file_digest(part) != source.digest(ref, offset)):
            raise ValueError("Partial transfer differs from sealed source")
        with part.open("ab") as target:
            try:
                with source.open(ref, offset) as handle:
                    while chunk := handle.read(CHUNK):
                        target.write(chunk)
            finally:
                target.flush()
                os.fsync(target.fileno())
        if part.stat().st_size != size or file_digest(part) != ref.id:
            raise ValueError("Transferred digest/size mismatch; not published")
        try:
            os.link(part, path)
        except FileExistsError:
            if path.is_symlink() or file_digest(path) != ref.id:
                raise ValueError("Conflicting sealed destination")
        part.unlink()
        return {"reused": False, "resumed_bytes": offset, "copied_bytes": size - offset}


def transfer(source, destination: Store, roots):
    roots = list(roots)
    if not roots:
        raise ValueError("Choose explicit sealed roots")
    refs = closure(source, roots)
    counts = {"copied_bytes": 0, "resumed_bytes": 0, "reused_records": 0}
    for ref in refs:
        done = copy_one(source, destination, ref)
        counts["copied_bytes"] += done["copied_bytes"]
        counts["resumed_bytes"] += done["resumed_bytes"]
        counts["reused_records"] += int(done["reused"])
    if closure(FilesystemSource(destination), roots) != refs:
        raise ValueError("Destination closure differs")
    # All closure members were independently SHA verified above. A missing or
    # interrupted member prevents the success receipt from being returned.
    return {
        "schema_version": 1,
        "complete": True,
        "records": len(refs),
        "roots": [r.as_dict() for r in roots],
        "closure_sha256": sha256(encode([r.as_dict() for r in refs])),
        **counts,
    }
