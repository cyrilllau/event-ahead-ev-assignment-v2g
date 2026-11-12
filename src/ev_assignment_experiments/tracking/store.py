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
    {"studies", "experiments", "runs", "attempts", "audits", "decisions", "publications", "objects"}
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

    def put(self, kind: str, payload: dict, refs=()) -> Ref:
        if kind == "objects":
            raise ValueError("Use put_object for artifacts")
        references = sorted(set(refs))
        for ref in references:
            self._read(ref)
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
            self._read(ref)
            if ref.kind != "objects":
                pending.extend(Ref(**r) for r in self.read(ref)["refs"])
            visited.add(ref)
        return sorted(visited)

    def sync_to(self, destination: Store, roots) -> dict:
        """Copy a sealed closure, validate every byte; retry is idempotent.

        Interrupted transfers can leave unreferenced complete objects, never a
        false success receipt. Streaming/SSH transport is a separate later phase.
        """
        roots = list(roots)
        closure = self.closure(roots)
        for ref in closure:
            destination._put(ref, self._read(ref))
        verified = destination.closure(roots)
        if closure != verified:
            raise ValueError("Destination closure differs")
        return {
            "schema_version": 1,
            "complete": True,
            "records": len(verified),
            "roots": [r.as_dict() for r in roots],
            "closure_sha256": sha256(encode([r.as_dict() for r in verified])),
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
