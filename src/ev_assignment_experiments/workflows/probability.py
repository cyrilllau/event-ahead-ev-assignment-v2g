"""A real, bounded probability-LP vertical slice, not L08 planning admission."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import time
import uuid
import zipfile

from ev_assignment.methods.probability import ProbabilitySet, solve_worst_distribution
from ev_assignment.validation.probability import check_probability_solution
from ev_assignment_experiments.config.experiment import Experiment
from ev_assignment_experiments.tracking.store import Ref, Store, encode, sha256


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot() -> bytes:
    """Archive actual executing sources, including uncommitted development edits."""
    import ev_assignment
    import ev_assignment_experiments

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for module in (ev_assignment, ev_assignment_experiments):
            root = Path(module.__file__).parent
            for source in sorted(root.rglob("*.py")):
                name = module.__name__ + "/" + source.relative_to(root).as_posix()
                archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), source.read_bytes())
        versions = {
            name: metadata.version(name)
            for name in ("numpy", "scipy", "networkx", "psutil", "ev-assignment")
        }
        try:
            versions["gurobipy"] = metadata.version("gurobipy")
        except metadata.PackageNotFoundError:
            versions["gurobipy"] = None
        versions.update(python=platform.python_version(), platform=platform.platform())
        archive.writestr(
            zipfile.ZipInfo("environment.json", (1980, 1, 1, 0, 0, 0)), encode(versions)
        )
    return buffer.getvalue()


def plan(store: Store, experiment: Experiment) -> Ref:
    source = store.put_object(source_snapshot())
    definition = store.put("experiments", experiment.as_dict())
    return store.put(
        "runs",
        {
            "study": experiment.study,
            "scope": "development_probability_oracle",
            "experiment": definition.as_dict(),
            "source": source.as_dict(),
        },
        [source, definition],
    )


def problem(experiment: Experiment):
    return ProbabilitySet(
        experiment.support_ids,
        experiment.feature_ids,
        experiment.features,
        experiment.lower,
        experiment.upper,
    )


def execute(store: Store, run: Ref) -> Ref:
    if run.kind != "runs":
        raise ValueError("Expected a planned run")
    record = store.read(run)["payload"]
    source = Ref(**record["source"])
    if sha256(source_snapshot()) != source.id:
        raise ValueError("Executing source/environment changed since plan; create a new Run")
    experiment = Experiment.parse(store.read(Ref(**record["experiment"]))["payload"])
    lock_path = store.root / "locks" / sha256(experiment.study.encode())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # This first vertical slice has no recovery/budget controller. Refuse a
        # second execution instead of silently giving it a new budget.
        claim = store.root / "claims" / run.id
        claim.parent.mkdir(parents=True, exist_ok=True)
        with claim.open("x") as handle:
            handle.write(
                encode(
                    {"study": experiment.study, "run": run.as_dict(), "observed_at": utcnow()}
                ).decode()
            )
        started = time.monotonic()
        payload = {
            "study": experiment.study,
            "run": run.as_dict(),
            "attempt_id": uuid.uuid4().hex,
            "observed_at": utcnow(),
            "scope": "development_probability_oracle",
            "scientific": "unchecked",
            "convergence": "unknown",
            "decision": "pending",
        }
        refs = [run]
        try:
            result = solve_worst_distribution(
                problem(experiment),
                experiment.q,
                q_support_ids=experiment.support_ids,
                time_limit_seconds=experiment.seconds,
            )
            raw = store.put_object(encode(result.as_dict()))
            refs.append(raw)
            report = check_probability_solution(
                problem(experiment), experiment.q, result, q_support_ids=experiment.support_ids
            )
            audit_ref = store.put(
                "audits",
                {
                    "study": experiment.study,
                    "run": run.as_dict(),
                    "raw": raw.as_dict(),
                    "report": report.as_dict(),
                },
                [run, raw],
            )
            refs.append(audit_ref)
            payload.update(
                execution="exited",
                native_status=result.status,
                scientific="pass" if report.passed else "fail",
                convergence="met" if result.certified_optimal else "unmet",
                raw=raw.as_dict(),
                audit=audit_ref.as_dict(),
            )
        except Exception as exc:
            payload.update(execution="failed", error_type=type(exc).__name__, error=str(exc))
        payload.update(elapsed_seconds=time.monotonic() - started, observed_at=utcnow())
        return store.put("attempts", payload, refs)


def reaudit(store: Store, attempt: Ref) -> dict:
    if attempt.kind != "attempts":
        raise ValueError("Expected attempt")
    payload = store.read(attempt)["payload"]
    run = store.read(Ref(**payload["run"]))["payload"]
    experiment = Experiment.parse(store.read(Ref(**run["experiment"]))["payload"])
    result = json.loads(store.object(Ref(**payload["raw"])))
    report = check_probability_solution(
        problem(experiment), experiment.q, result, q_support_ids=experiment.support_ids
    )
    return report.as_dict()
