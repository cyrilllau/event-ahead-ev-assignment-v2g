"""Read only explicit historical files into CAS; never adopt old PIDs or results."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

from .store import Ref, encode
from .transfer import SSHSource, copy_one, file_digest
from ev_assignment_experiments.workflows.probability import utcnow

# This program is sent over SSH and only reads the named allowlisted file.
READER = """import sys,json,pathlib,hashlib
root=pathlib.Path(sys.argv[1]).resolve();action=sys.argv[2];name=sys.argv[3];count=int(sys.argv[5])
p=root/name
if p.is_symlink() or not p.resolve().is_relative_to(root): raise ValueError('path escape')
if action=='size': print(json.dumps({'size':p.stat().st_size}))
elif action=='read':
 with p.open('rb') as h:
  h.seek(count)
  while True:
   b=h.read(1048576)
   if not b: break
   sys.stdout.buffer.write(b)
elif action=='digest':
 d=hashlib.sha256()
 with p.open('rb') as h:
  while count:
   b=h.read(min(1048576,count))
   if not b: raise ValueError('truncated')
   d.update(b);count-=len(b)
 print(d.hexdigest())
else: raise ValueError('unknown read operation')
"""


class NamedSource:
    def __init__(self, root, entries, host=None):
        if not Path(root).is_absolute():
            raise ValueError("Explicit absolute historical root required")
        self.root, self.entries = Path(root), entries
        self.remote = SSHSource(host, root) if host else None
        if self.remote:
            self.remote.program = READER

    def path(self, ref):
        name = self.entries[ref.id]["path"]
        p = self.root / name
        if not self.remote and (
            p.is_symlink() or not p.resolve().is_relative_to(self.root.resolve())
        ):
            raise ValueError("Historical path escape")
        return p

    def command(self, action, ref, count):
        # SSHSource quotes path and host; use an explicit manifest name as reader argument.
        from types import SimpleNamespace

        named = SimpleNamespace(kind=self.entries[ref.id]["path"], id=ref.id)
        return self.remote.command(action, named, count)

    def size(self, ref):
        if self.remote:
            return json.loads(subprocess.check_output(self.command("size", ref, 0), timeout=45))[
                "size"
            ]
        return self.path(ref).stat().st_size

    def digest(self, ref, length):
        if self.remote:
            return (
                subprocess.check_output(self.command("digest", ref, length), timeout=45)
                .decode()
                .strip()
            )
        return file_digest(self.path(ref), length)

    @contextmanager
    def open(self, ref, offset=0):
        if not self.remote:
            with self.path(ref).open("rb") as h:
                h.seek(offset)
                yield h
            return
        p = subprocess.Popen(self.command("read", ref, offset), stdout=subprocess.PIPE)
        try:
            yield p.stdout
            if p.wait() != 0:
                raise OSError("Historical read-only transfer failed")
        finally:
            p.stdout.close()
            if p.poll() is None:
                p.terminate()  # Our local transport child only.
            p.wait()


def import_manifest(store, definition, *, root, host=None):
    required = {
        "schema_version",
        "study",
        "title",
        "origin",
        "observed_at",
        "files",
        "state",
        "scope",
    }
    if set(definition) != required or definition["schema_version"] != 1 or not definition["files"]:
        raise ValueError("Invalid historical import manifest")
    if definition["scope"] not in {"selected_reference_files", "declared_evidence_closure"}:
        raise ValueError("Explicit import coverage required")
    entries, names = {}, set()
    for row in definition["files"]:
        if set(row) != {"path", "sha256", "bytes", "role", "stability"}:
            raise ValueError("Unknown historical file schema")
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or str(path) in names:
            raise ValueError("Duplicate or escaping historical path")
        if row["stability"] not in {"sealed", "observed_snapshot"}:
            raise ValueError("Active files must be explicitly marked snapshots")
        if row["role"] not in {
            "source",
            "config",
            "input",
            "result",
            "audit",
            "log",
            "checkpoint",
            "state",
        }:
            raise ValueError("Unknown provenance role")
        ref = Ref("objects", row["sha256"])
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("Invalid file size")
        names.add(str(path))
        entries[ref.id] = row
    source = NamedSource(root, entries, host)
    refs = []
    for key, row in entries.items():
        ref = Ref("objects", key)
        if source.size(ref) != row["bytes"]:
            raise ValueError("Historical file changed size")
        copy_one(source, store, ref)
        refs.append(ref)
    manifest = store.put_object(encode(definition))
    refs.append(manifest)
    return store.put(
        "observations",
        dict(
            study=definition["study"],
            kind="legacy_import",
            title=definition["title"],
            origin=definition["origin"],
            observed_at=definition["observed_at"],
            imported_at=utcnow(),
            manifest=manifest.as_dict(),
            declared_state=definition["state"],
            scope=definition["scope"],
            files=len(definition["files"]),
            scientific="unchecked",
            adoption="not_adopted",
            execution="unknown",
            process_adopted=False,
            caveat="Historical state is an observation, not live status or new-source acceptance",
        ),
        refs,
    )
