"""Standard-library-only, read-only sealed-store protocol executed over SSH.

This program reads sealed content-addressed files. Inventory scans only a named
record namespace, never campaigns, source scripts or active logs/checkpoints.
"""

import hashlib
import json
from pathlib import Path
import re
import sys


def main(argv):
    root, action, kind, identity, count = argv
    allowed = {
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
    if kind not in allowed or not re.fullmatch("[0-9a-f]{64}", identity):
        raise ValueError("Invalid sealed reference")
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("An absolute remote store is required")
    root = root.resolve()
    if action == "inventory":
        if kind == "objects" or not 1 <= int(count) <= 100000:
            raise ValueError("Bounded record inventory required")
        rows = []
        for candidate in sorted((root / kind).glob("*")):
            if not re.fullmatch("[0-9a-f]{64}", candidate.name):
                continue
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
                raise ValueError("Inventory escapes sealed store")
            if candidate.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Oversized metadata")
            raw = candidate.read_bytes()
            if hashlib.sha256(raw).hexdigest() != candidate.name:
                raise ValueError("Corrupt inventory record")
            value = json.loads(raw)
            study = value.get("payload", {}).get("study")
            if isinstance(study, str) and hashlib.sha256(study.encode()).hexdigest() == identity:
                rows.append(dict(kind=kind, id=candidate.name))
                if len(rows) > int(count):
                    raise ValueError("Inventory exceeds explicit record limit")
        print(json.dumps(rows))
        return
    path = root / kind / identity
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Reference escapes remote store")
    count = int(count)
    if count < 0:
        raise ValueError("Negative range")
    if action == "size":
        print(json.dumps({"size": path.stat().st_size}))
    elif action == "digest":
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            remaining = count
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Truncated prefix")
                digest.update(chunk)
                remaining -= len(chunk)
        print(digest.hexdigest())
    elif action == "read":
        with path.open("rb") as handle:
            handle.seek(count)
            while chunk := handle.read(1024 * 1024):
                sys.stdout.buffer.write(chunk)
    else:
        raise ValueError("Unknown read operation")


if __name__ == "__main__":
    main(sys.argv[1:])
