"""Lock exact adopted plans before opening paired evaluation samples."""

from ev_assignment_experiments.tracking.store import Ref, encode, sha256
from ev_assignment_experiments.workflows.probability import utcnow


def lock(store, study, decisions, *, methods, sample_ids, random_stream, protocol):
    if (
        not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("Unique locked methods and samples required")
    if not protocol or not random_stream or len(decisions) != len(methods):
        raise ValueError("Incomplete evaluation lock")
    plans = {}
    for decision in decisions:
        row = store.read(decision)["payload"]
        if row.get("outcome") not in {"adopted", "approximate"} or row["study"] != study:
            raise ValueError("Evaluation requires explicit accepted plans for this study")
        attempt = store.read(Ref(**row["attempt"]))["payload"]
        result = __import__("json").loads(store.object(Ref(**attempt["result"])))
        method = result["method"]
        if method not in methods or method in plans or not result["plan"]:
            raise ValueError("Missing/duplicate plan")
        plans[method] = dict(
            decision=decision.as_dict(),
            attempt=row["attempt"],
            plan_sha256=sha256(encode(result["plan"])),
        )
    return store.put(
        "experiments",
        dict(
            kind="evaluation_lock",
            study=study,
            plans=plans,
            methods=list(methods),
            sample_ids=list(sample_ids),
            random_stream=random_stream,
            protocol=protocol,
            observed_at=utcnow(),
        ),
        decisions,
    )


def open_sample(store, lock_ref, sid, reader):
    locked = store.read(lock_ref)["payload"]
    if locked.get("kind") != "evaluation_lock" or sid not in locked["sample_ids"]:
        raise ValueError("OOS remains locked or sample is outside paired protocol")
    store.closure([lock_ref])
    return reader(sid)


def paired_summary(locked, observations):
    import numpy as np

    methods, ids = locked["methods"], locked["sample_ids"]
    values = {}
    for method in methods:
        rows = observations[method]
        if [r["sample_id"] for r in rows] != ids:
            raise ValueError("Paired sample order mismatch")
        a = np.array([r["cost"] for r in rows], float)
        if not np.isfinite(a).all():
            raise ValueError("Missing/nonfinite evaluation is not zero")
        values[method] = a
    base = methods[0]
    return dict(
        protocol=locked["protocol"],
        sample_count=len(ids),
        baseline=base,
        methods={
            m: dict(
                mean=float(a.mean()),
                paired_difference=float((a - values[base]).mean()),
                paired_standard_error=float((a - values[base]).std(ddof=1) / np.sqrt(len(a)))
                if len(a) > 1
                else None,
            )
            for m, a in values.items()
        },
    )
