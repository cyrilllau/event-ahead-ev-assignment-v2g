"""Explicit research choices; immutable outcomes and native gaps are never relabelled."""

from .store import Ref
from ev_assignment_experiments.workflows.audit import reaudit
from ev_assignment_experiments.workflows.probability import utcnow


def decide(store, attempt, outcome, reason, *, actor, supersedes=None):
    if (
        outcome not in {"adopted", "approximate", "rejected", "superseded"}
        or not reason.strip()
        or not actor.strip()
    ):
        raise ValueError("Explicit outcome, reason and named decision maker required")
    payload = store.read(attempt)["payload"]
    refs = [attempt]
    if outcome in {"adopted", "approximate"}:
        checked = reaudit(store, attempt, register=True)
        if not checked["passed"]:
            raise ValueError("Scientific audit failed")
        if outcome == "adopted" and payload["convergence"] != "met":
            raise ValueError("Unmet convergence requires an explicit approximate decision")
        if "audit_ref" in checked:
            audit = Ref(**checked["audit_ref"])
        else:
            audit = store.put(
                "audits",
                dict(study=payload["study"], attempt=attempt.as_dict(), report=checked),
                [attempt],
            )
        refs.append(audit)
    else:
        audit = None
    if supersedes:
        prior = store.read(supersedes)["payload"]
        if prior["attempt"] != attempt.as_dict():
            raise ValueError("Decision correction changed attempt")
        refs.append(supersedes)
    return store.put(
        "decisions",
        dict(
            study=payload["study"],
            attempt=attempt.as_dict(),
            outcome=outcome,
            reason=reason,
            actor=actor,
            observed_at=utcnow(),
            audit=audit.as_dict() if audit else None,
            supersedes=supersedes.as_dict() if supersedes else None,
            original_execution=payload["execution"],
            original_convergence=payload["convergence"],
            native_status=payload.get("native_status"),
        ),
        refs,
    )


def next_action(store, study, action, *, dependencies=(), reason, actor):
    if not all(str(v).strip() for v in (study, action, reason, actor)):
        raise ValueError("Incomplete next action")
    return store.put(
        "decisions",
        dict(
            study=study,
            kind="next_action",
            action=action,
            reason=reason,
            actor=actor,
            dependencies=[r.as_dict() for r in dependencies],
            observed_at=utcnow(),
        ),
        dependencies,
    )
