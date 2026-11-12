"""Immutable content-addressed records and verified, additive archive transfer.

Each record declares its complete references. Derived indexes are disposable.
Only sealed artifacts are registered here; live logs remain attempt-owned until
a supervisor publishes a snapshot. This module never talks to a running solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

KINDS = frozenset(
    {
        "studies",
        "experiments",
        "runs",
        "attempts",
        "audits",
        "decisions",
        "publications",
        "objects",
        "inputs",
        "checkpoints",
        "events",
        "observations",
    }
)


def encode(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, order=True)
class Ref:
    kind: str
    id: str

    def __post_init__(self):
        if self.kind not in KINDS or not re.fullmatch(r"[0-9a-f]{64}", self.id):
            raise ValueError("Invalid record reference")

    def as_dict(self):
        return asdict(self)


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self._verified_objects = {}

    def path(self, ref: Ref) -> Path:
        return self.root / ref.kind / ref.id

    def _read(self, ref: Ref) -> bytes:
        path = self.path(ref)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Store paths must not escape through symlinks")
        data = path.read_bytes()
        if sha256(data) != ref.id:
            raise ValueError(f"Digest mismatch: {ref.kind}/{ref.id}")
        return data

    def _put(self, ref: Ref, data: bytes):
        if sha256(data) != ref.id:
            raise ValueError("Digest mismatch before write")
        path = self.path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.parent.resolve().is_relative_to(self.root):
            raise ValueError("Destination escapes store")
        fd, name = tempfile.mkstemp(prefix=".incoming-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(name, path)  # exclusive publication, never overwrite
            except FileExistsError:
                self._read(ref)
        finally:
            Path(name).unlink(missing_ok=True)

    def put_object(self, data: bytes) -> Ref:
        ref = Ref("objects", sha256(data))
        self._put(ref, data)
        return ref

    def put_file(self, path) -> Ref:
        """Seal a large artifact with bounded memory; a changing file is refused."""
        from contextlib import contextmanager
        from .transfer import copy_one, file_digest

        path = Path(path)
        ref = Ref("objects", file_digest(path))

        class Source:
            def size(self, ref):
                return path.stat().st_size

            def digest(self, ref, length):
                return file_digest(path, length)

            @contextmanager
            def open(self, ref, offset=0):
                with path.open("rb") as handle:
                    handle.seek(offset)
                    yield handle

        copy_one(Source(), self, ref)
        return ref

    def put(self, kind: str, payload: dict, refs=()) -> Ref:
        if kind == "objects":
            raise ValueError("Use put_object for artifacts")
        references = sorted(set(refs))
        for ref in references:
            self.closure([ref])
        data = encode(
            {
                "schema_version": 1,
                "kind": kind,
                "payload": payload,
                "refs": [r.as_dict() for r in references],
            }
        )
        ref = Ref(kind, sha256(data))
        self._put(ref, data)
        return ref

    def read(self, ref: Ref) -> dict:
        if ref.kind == "objects":
            raise ValueError("Objects are bytes, not records")
        value = json.loads(self._read(ref))
        if (
            set(value) != {"schema_version", "kind", "payload", "refs"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["kind"] != ref.kind
            or not isinstance(value["payload"], dict)
        ):
            raise ValueError("Invalid or unsupported record schema")
        for item in value["refs"]:
            Ref(**item)
        return value

    def object(self, ref: Ref) -> bytes:
        if ref.kind != "objects":
            raise ValueError("Expected artifact reference")
        return self._read(ref)

    def closure(self, roots) -> list[Ref]:
        pending = list(roots)
        visited = set()
        while pending:
            ref = pending.pop()
            if ref in visited:
                continue
            if ref.kind != "objects":
                pending.extend(Ref(**r) for r in self.read(ref)["refs"])
            else:
                from .transfer import FilesystemSource, file_digest

                path = FilesystemSource(self)._path(ref)
                stat = path.stat()
                stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if self._verified_objects.get(ref) != stamp:
                    if file_digest(path) != ref.id:
                        raise ValueError("Digest mismatch in closure")
                    self._verified_objects[ref] = stamp
            visited.add(ref)
        return sorted(visited)

    def sync_to(self, destination: Store, roots) -> dict:
        """Copy a sealed closure with bounded memory and verified range resume."""
        from .transfer import FilesystemSource, transfer

        receipt = transfer(FilesystemSource(self), destination, roots)
        # Stable content receipt; transport statistics belong to the individual
        # transfer operation and must not change a bundle's content identity.
        return {
            k: v
            for k, v in receipt.items()
            if k not in {"copied_bytes", "resumed_bytes", "reused_records"}
        }

    def records(self, kind: str):
        if kind not in KINDS - {"objects"}:
            raise ValueError("Unknown record kind")
        folder = self.root / kind
        for path in sorted(folder.glob("*")):
            if path.name.startswith(".incoming-"):
                continue
            ref = Ref(kind, path.name)
            yield ref, self.read(ref)

    def rebuild_index(self):
        """Local-only projection, rebuilt atomically from immutable records."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".index-", dir=self.root)
        os.close(fd)
        try:
            with sqlite3.connect(temporary) as db:
                db.execute(
                    "CREATE TABLE records (kind TEXT, id TEXT, study TEXT, payload TEXT, "
                    "PRIMARY KEY(kind,id))"
                )
                for kind in sorted(KINDS - {"objects"}):
                    for ref, record in self.records(kind):
                        payload = record["payload"]
                        db.execute(
                            "INSERT INTO records VALUES (?,?,?,?)",
                            (kind, ref.id, payload.get("study"), encode(payload).decode()),
                        )
                db.execute("CREATE INDEX by_study ON records(study,kind)")
            os.replace(temporary, self.root / "index.sqlite")
        finally:
            Path(temporary).unlink(missing_ok=True)
