"""Metadata snapshots and complete result/full study archives; no remote mutation."""

import time
import subprocess
from .transfer import FilesystemSource, transfer, record
from ev_assignment_experiments.workflows.probability import utcnow


def synchronize(source, destination, roots, *, mode="results"):
    if mode not in {"metadata", "results", "full"}:
        raise ValueError("Unknown sync mode")
    if mode != "metadata":
        roots = include_settlements(source, roots)
        receipt = transfer(source, destination, roots)
        receipt.update(mode=mode, archived=True, observed_at=utcnow())
        return receipt
    rows = []
    for ref in roots:
        value = record(source, ref)
        # References in a metadata observation are external links, not a claim of local closure.
        p = value["payload"]
        summary = {
            k: p.get(k)
            for k in (
                "study",
                "observed_at",
                "execution",
                "scientific",
                "convergence",
                "decision",
                "scope",
                "run",
                "native_status",
                "checkpoint",
            )
        }
        rows.append(dict(reference=ref.as_dict(), summary=summary))
    study = rows[0]["summary"]["study"] if rows else None
    observation = destination.put(
        "observations",
        dict(
            study=study,
            origin="sealed_remote_metadata",
            observed_at=utcnow(),
            archived=False,
            external_links=rows,
            live_status="not_observed",
        ),
    )
    return dict(
        mode="metadata", archived=False, observation=observation.as_dict(), records=len(rows)
    )


def study_roots(store, study, mode):
    kinds = (
        ("attempts", "audits", "decisions", "publications", "observations")
        if mode == "results"
        else tuple(
            sorted(
                {
                    "studies",
                    "experiments",
                    "runs",
                    "attempts",
                    "audits",
                    "decisions",
                    "publications",
                    "events",
                    "checkpoints",
                    "observations",
                }
            )
        )
    )
    return [
        ref
        for kind in kinds
        for ref, record in store.records(kind)
        if record["payload"].get("study") == study
    ]


def watch_local(source, destination, study, *, seconds=60, interval=5, mode="results"):
    if not 0 < seconds <= 3600 or not 0 < interval <= 60:
        raise ValueError("Watch must be bounded")
    deadline = time.monotonic() + seconds
    receipts = []
    last = None
    while True:
        roots = study_roots(source, study, mode)
        if roots != last:
            receipts.append(synchronize(FilesystemSource(source), destination, roots, mode=mode))
            last = roots
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    return dict(receipts=receipts, observed_at=utcnow(), remote_mutations=0)


def watch_ssh(source, destination, study, *, seconds=60, interval=5, mode="results"):
    if not 0 < seconds <= 3600 or not 0 < interval <= 60:
        raise ValueError("Watch must be bounded")
    kinds = (
        ("attempts",)
        if mode == "metadata"
        else ("attempts", "audits", "decisions", "publications", "observations")
    )
    if mode == "full":
        kinds += ("studies", "experiments", "runs", "events", "checkpoints", "observations")
    deadline = time.monotonic() + seconds
    last = None
    receipts = []
    error = None
    while True:
        try:
            roots = source.inventory(study, kinds)
            if roots != last:
                receipts.append(synchronize(source, destination, roots, mode=mode))
                last = roots
            error = None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = repr(exc)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    observation = destination.put(
        "observations",
        dict(
            study=study,
            origin="ssh_watch",
            observed_at=utcnow(),
            receipts=receipts,
            connection="unavailable" if error else "observed",
            error=error,
            solver_liveness="not_inferred",
        ),
    )
    return dict(
        observation=observation.as_dict(), receipts=receipts, error=error, remote_mutations=0
    )


def include_settlements(source, roots):
    """A sealed attempt precedes its parent's wait; pull that later receipt as a root."""
    roots = list(roots)
    attempts = {}
    for ref in roots:
        if ref.kind == "attempts":
            value = record(source, ref)["payload"]
            attempts.setdefault(value["study"], []).append(ref.as_dict())
    for study, selected in attempts.items():
        if isinstance(source, FilesystemSource):
            candidates = [
                r for r, v in source.store.records("events") if v["payload"].get("study") == study
            ]
        else:
            candidates = source.inventory(study, ("events",))
        for ref in candidates:
            p = record(source, ref)["payload"]
            if p.get("kind") == "parent_settlement" and p.get("attempt") in selected:
                roots.append(ref)
    return sorted(set(roots))
