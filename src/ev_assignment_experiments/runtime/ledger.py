"""Append-only cumulative budgets, exclusive admissions and conservative recovery."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from .identity import current, matches
from ev_assignment_experiments.tracking.store import encode, sha256


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode(value)
    fd, tmp = tempfile.mkstemp(prefix=".new-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as h:
            h.write(data)
            h.flush()
            os.fsync(h.fileno())
        os.link(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(tmp).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


class BudgetLedger:
    def __init__(self, store, group):
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Missing budget group")
        self.root = Path(store) / "runtime" / "budgets" / sha256(group.encode())
        self.group = group

    def create(self, study, limit, *, inherited=0.0, evidence=()):
        if (
            any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in (limit, inherited))
            or limit <= 0
            or inherited > limit
        ):
            raise ValueError("Invalid cumulative budget")
        if inherited and not evidence:
            raise ValueError("Inherited cost requires explicit evidence")
        definition = dict(
            schema_version=1,
            group=self.group,
            study=study,
            limit_seconds=limit,
            inherited_seconds=inherited,
            inherited_evidence=list(evidence),
        )
        with lock(self.root / "lock"):
            path = self.root / "definition.json"
            if path.exists():
                if json.loads(path.read_text()) != definition:
                    raise ValueError("Budget cannot be reset or replaced")
            else:
                publish(path, definition)
        return definition

    def read(self):
        definition = json.loads((self.root / "definition.json").read_text())
        previous = sha256(encode(definition))
        events = []
        active = None
        spent = definition["inherited_seconds"]
        closed = set()
        for i, path in enumerate(sorted((self.root / "events").glob("*.json"))):
            row = json.loads(path.read_text())
            if (
                path.stem != f"{i:08d}"
                or row["previous"] != previous
                or row["sha256"] != sha256(encode(row["payload"]))
            ):
                raise ValueError("Budget event chain is corrupt or incomplete")
            previous = sha256(encode(row))
            event = row["payload"]
            kind = event["kind"]
            if kind == "claim":
                if active is not None or event["attempt"] in closed:
                    raise ValueError("Duplicate active claim")
                if event["reserved_seconds"] > max(0.0, definition["limit_seconds"] - spent) + 1e-9:
                    raise ValueError("Budget over-admission")
                active = event
            elif kind == "child":
                if active is None or event["attempt"] != active["attempt"]:
                    raise ValueError("Child without claim")
                active = dict(active, child=event["identity"])
            elif kind == "settlement":
                if active is None or event["attempt"] != active["attempt"]:
                    raise ValueError("Settlement without claim")
                charged = event["charged_seconds"]
                if not math.isfinite(charged) or charged < 0:
                    raise ValueError("Invalid settled cost")
                spent += charged
                closed.add(active["attempt"])
                active = None
            else:
                raise ValueError("Unknown budget event")
            events.append(row)
        return dict(
            definition=definition,
            events=events,
            previous=previous,
            spent_seconds=spent,
            remaining_seconds=max(0.0, definition["limit_seconds"] - spent),
            active=active,
        )

    def _append(self, snapshot, event):
        row = dict(previous=snapshot["previous"], payload=event, sha256=sha256(encode(event)))
        publish(self.root / "events" / f"{len(snapshot['events']):08d}.json", row)
        return event

    def claim(self, run, requested, attempt=None, *, charge_process_start=False):
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("Invalid requested budget")
        with lock(self.root / "lock"):
            snap = self.read()
            if snap["active"] is not None:
                raise ValueError("Unsettled claim; inspect actual identities before recovery")
            available = min(requested, snap["remaining_seconds"])
            if available <= 0:
                raise ValueError("Cumulative budget exhausted")
            owner = current()
            observed_epoch, observed_monotonic = time.time(), time.monotonic()
            startup = max(0.0, observed_epoch - owner["created"]) if charge_process_start else 0.0
            return self._append(
                snap,
                dict(
                    kind="claim",
                    attempt=attempt or uuid.uuid4().hex,
                    run=run,
                    reserved_seconds=available,
                    owner=owner,
                    charged_from_process_start=charge_process_start,
                    startup_seconds=startup,
                    started_epoch=observed_epoch - startup,
                    started_monotonic=observed_monotonic - startup,
                ),
            )

    def child(self, attempt, identity):
        with lock(self.root / "lock"):
            snap = self.read()
            if (
                not snap["active"]
                or snap["active"]["attempt"] != attempt
                or snap["active"]["owner"] != current()
            ):
                raise ValueError("Only the admitted parent can register its child")
            return self._append(snap, dict(kind="child", attempt=attempt, identity=identity))

    def settle(self, attempt, returncode, receipt):
        with lock(self.root / "lock"):
            snap = self.read()
            active = snap["active"]
            if not active or active["attempt"] != attempt or active["owner"] != current():
                raise ValueError("Only the admitted parent can settle its wait")
            elapsed = max(0.0, time.monotonic() - active["started_monotonic"])
            return self._append(
                snap,
                dict(
                    kind="settlement",
                    attempt=attempt,
                    charged_seconds=elapsed,
                    actual_parent_wait=True,
                    returncode=returncode,
                    receipt=receipt,
                    observed_epoch=time.time(),
                ),
            )

    def recover_dead_claim(self):
        with lock(self.root / "lock"):
            snap = self.read()
            active = snap["active"]
            if not active:
                return None
            identities = [active["owner"]] + ([active["child"]] if active.get("child") else [])
            if any(matches(i) is not False for i in identities):
                raise ValueError("Owner/child is live or unknown; recovery forbidden")
            # Unknown lost parent overhead cannot silently become free time.
            return self._append(
                snap,
                dict(
                    kind="settlement",
                    attempt=active["attempt"],
                    charged_seconds=active["reserved_seconds"],
                    actual_parent_wait=False,
                    reason="dead_owner_unknown_overhead_conservatively_charged_full_reservation",
                    observed_epoch=time.time(),
                ),
            )
