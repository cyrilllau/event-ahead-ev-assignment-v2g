"""Standard-library-only, read-only sealed-store protocol executed over SSH.

This program reads exactly one content-addressed file per call. It never scans a
campaign, imports research scripts, writes files or opens active logs/checkpoints.
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
    }
    if kind not in allowed or not re.fullmatch("[0-9a-f]{64}", identity):
        raise ValueError("Invalid sealed reference")
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("An absolute remote store is required")
    root = root.resolve()
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
