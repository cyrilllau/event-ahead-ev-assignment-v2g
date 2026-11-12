"""Build a verified development reproduction bundle without publishing it."""

import json
from pathlib import Path

from ev_assignment_experiments.tracking.store import Ref, Store, encode
from ev_assignment_experiments.workflows.probability import reaudit


def build(store: Store, attempts: list[Ref], destination) -> dict:
    if not attempts or len(set(attempts)) != len(attempts):
        raise ValueError("Choose a nonempty, unique list of attempts explicitly")
    selected = []
    for ref in attempts:
        if ref.kind != "attempts":
            raise ValueError("Publication must select attempts")
        payload = store.read(ref)["payload"]
        if payload["execution"] != "exited" or not reaudit(store, ref)["passed"]:
            raise ValueError("Selected result failed independent audit")
        selected.append(ref)
    destination = Path(destination).resolve()
    if destination == store.root or destination.is_relative_to(store.root):
        raise ValueError("Bundle must be separate from the source archive")
    destination.mkdir(parents=True, exist_ok=False)
    receipt = store.sync_to(Store(destination / "archive"), selected)
    manifest = {
        "schema_version": 1,
        "scope": "development_probability_oracle",
        "paper_facing": False,
        "selected_attempts": [r.as_dict() for r in selected],
        "transfer": receipt,
    }
    # Publication marker is last; partial output cannot look complete.
    (destination / "manifest.json").write_bytes(encode(manifest))
    return manifest


def verify(destination) -> dict:
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    store = Store(destination / "archive")
    roots = [Ref(**r) for r in manifest["selected_attempts"]]
    from ev_assignment_experiments.tracking.store import sha256

    closure = store.closure(roots)
    digest = sha256(encode([r.as_dict() for r in closure]))
    if digest != manifest["transfer"]["closure_sha256"]:
        raise ValueError("Bundle closure changed")
    reports = [reaudit(store, ref) for ref in roots]
    return {
        "passed": all(r["passed"] for r in reports),
        "reports": reports,
        "paper_facing": False,
        "scope": manifest["scope"],
    }
