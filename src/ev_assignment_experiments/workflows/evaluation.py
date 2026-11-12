"""Complete fixed-plan paired evaluation, incremental receipts, and solver-free replay."""

from dataclasses import replace
import json
from pathlib import Path
import uuid
import fcntl
import numpy as np
from ev_assignment.domain.planning import Control
from ev_assignment.validation.l08 import audit_plan, check_first_vector
from ev_assignment_experiments.analysis.evaluation import lock, paired_summary
from ev_assignment_experiments.datasets.evaluation import (
    domain_identity,
    validate_manifest,
    seal_sample,
    scene,
)
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.runtime.identity import current
from ev_assignment_experiments.tracking.store import Ref, encode, sha256
from ev_assignment_experiments.workflows.l08 import problem_for, recipe_for, plain
from ev_assignment_experiments.workflows.probability import source_snapshot, utcnow

METRICS = (
    "cost",
    "first_cost",
    "recourse_cost",
    "critical_ENS",
    "other_ENS",
    "actual_V2G_ac",
    "shortfall_ac",
)


def prepare(store, *, study, decisions, manifest, budget_group, budget_seconds):
    """Read a metadata manifest only. No OOS physical or replay file is touched."""
    ids = validate_manifest(manifest)
    methods = []
    for ref in decisions:
        d = store.read(ref)["payload"]
        a = store.read(Ref(**d["attempt"]))["payload"]
        result = json.loads(store.object(Ref(**a["result"])))
        methods.append(result["method"])
    if set(methods) != {"DET", "DRO", "ARO"} or len(methods) != 3:
        raise ValueError("L08 comparison locks exactly DET/DRO/ARO")
    ordered = [decisions[methods.index(m)] for m in ("DET", "DRO", "ARO")]
    locked = lock(
        store,
        study,
        ordered,
        methods=["DET", "DRO", "ARO"],
        sample_ids=ids,
        random_stream=manifest["random_stream"],
        protocol=manifest["protocol"],
    )
    spec = store.put_object(encode(manifest))
    source = store.put_object(source_snapshot())
    plans = store.read(locked)["payload"]["plans"]
    for method, entry in plans.items():
        p, result = planning(store, entry)
        if domain_identity(p) != manifest["domain_sha256"]:
            raise ValueError("Methods/evaluation do not share the same scientific domain")
        if manifest["scope"] == "oos" and p.scope == "development":
            raise ValueError("Development plans cannot be relabelled as paper OOS")
        check_first_vector(p, result["plan"]["entries"], result["plan"]["first_stage"])
    BudgetLedger(store.root, budget_group).create(study, budget_seconds)
    run = store.put(
        "runs",
        dict(
            schema_version=3,
            kind="paired_evaluation",
            study=study,
            lock=locked.as_dict(),
            manifest=spec.as_dict(),
            source=source.as_dict(),
            scope=manifest["scope"],
            budget_group=budget_group,
            budget_seconds=budget_seconds,
            metric_units=dict(
                cost="USD",
                first_cost="USD",
                recourse_cost="USD",
                critical_ENS="kWh",
                other_ENS="kWh",
                actual_V2G_ac="kWh AC",
                shortfall_ac="kWh AC",
            ),
        ),
        [locked, spec, source],
    )
    return run


def planning(store, entry):
    a = store.read(Ref(**entry["attempt"]))["payload"]
    result = json.loads(store.object(Ref(**a["result"])))
    if sha256(encode(result["plan"])) != entry["plan_sha256"]:
        raise ValueError("Accepted plan changed")
    run = store.read(Ref(**a["run"]))["payload"]
    recipe = recipe_for(store, run)
    p = problem_for(store, run, recipe, for_evaluation=True)
    return p, result


def rows_for(store, run_ref):
    rows = {}
    for ref, value in store.records("events"):
        p = value["payload"]
        if p.get("kind") == "evaluation_sample" and p.get("run") == run_ref.as_dict():
            key = (p["method"], p["sample_id"])
            if key in rows:
                raise ValueError("Duplicate evaluation cell")
            rows[key] = (ref, p)
    return rows


def describe(store, run_ref):
    run = store.read(run_ref)["payload"]
    if run.get("kind") != "paired_evaluation":
        raise ValueError("Expected locked evaluation run")
    locked = store.read(Ref(**run["lock"]))["payload"]
    manifest = json.loads(store.object(Ref(**run["manifest"])))
    if (
        validate_manifest(manifest) != locked["sample_ids"]
        or manifest["random_stream"] != locked["random_stream"]
        or manifest["protocol"] != locked["protocol"]
    ):
        raise ValueError("Evaluation lock/manifest mismatch")
    return run, locked, manifest


def metrics(report):
    c = report.checks
    row = c["scenarios"][0]
    first, recourse = float(c["first_cost"]), float(row["primal"])
    return dict(
        cost=first + recourse,
        first_cost=first,
        recourse_cost=recourse,
        **{k: row[k] for k in METRICS[3:]},
        first_cost_parts=c["first_cost_parts"],
        recourse_cost_parts=row["cost_parts"],
    )


def verify_cell(store, run, locked, descriptor, payload, p, result):
    method, sid = payload["method"], descriptor["sample_id"]
    entry = locked["plans"][method]
    if (
        payload["sample_id"] != sid
        or payload["plan_sha256"] != entry["plan_sha256"]
        or payload["source"] != run["source"]
        or payload["manifest"] != run["manifest"]
        or payload["lock"] != run["lock"]
        or payload["protocol"] != locked["protocol"]
        or payload["random_stream"] != locked["random_stream"]
    ):
        raise ValueError("Evaluation evidence identity mismatch")
    sc = scene(store, descriptor, p)
    ep = replace(p, scenario_ids=(sid,), scenario=lambda s: sc, ambiguity=None, det_proof=None)
    witness = json.loads(store.object(Ref(**payload["witness"])))
    report = audit_plan(
        ep, result["plan"]["entries"], Control(60), witness_provider=lambda s: witness
    )
    expected = metrics(report)
    for key, value in expected.items():
        if not np.allclose(value, payload["metrics"][key], rtol=0, atol=1e-8):
            raise ValueError("Stored evaluation metric differs from original physics: " + key)
    return expected


def execute(store, run_ref, *, input_root, seconds=60, max_cells=None):
    from ev_assignment_experiments.workflows.l08 import atomic_pointer

    run = store.read(run_ref)["payload"]
    area = store.root / "runtime/studies" / sha256(run["study"].encode())
    area.mkdir(parents=True, exist_ok=True)
    with (area / "lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = area / "CURRENT.json"
        if pointer.exists():
            prior = json.loads(pointer.read_text())
            if BudgetLedger(store.root, prior["budget_group"]).read()["active"] is not None:
                raise ValueError(
                    "Study has an unsettled task; explicit dead-owner recovery required"
                )

        def admitted(claim):
            atomic_pointer(
                pointer,
                dict(
                    kind="paired_evaluation",
                    attempt=claim["attempt"],
                    run=run_ref.as_dict(),
                    budget_group=run["budget_group"],
                    owner=current(),
                ),
            )

        return _execute(
            store,
            run_ref,
            input_root=input_root,
            seconds=seconds,
            max_cells=max_cells,
            admitted=admitted,
        )


def _execute(store, run_ref, *, input_root, seconds, max_cells, admitted):
    if (
        not Path(input_root).is_absolute()
        or not 0 < seconds <= 60
        or (max_cells is not None and max_cells < 1)
    ):
        raise ValueError("Explicit input root and bounded evaluation slice (<=60s) required")
    run, locked, manifest = describe(store, run_ref)
    if sha256(source_snapshot()) != run["source"]["id"]:
        raise ValueError(
            "Evaluation source/environment changed; new run, same cumulative budget required"
        )
    old = rows_for(store, run_ref)
    total = len(locked["methods"]) * len(locked["sample_ids"])
    if len(old) == total:
        return audit(store, run_ref, register=True)
    ledger = BudgetLedger(store.root, run["budget_group"])
    claim = ledger.claim(run_ref.as_dict(), seconds, attempt=uuid.uuid4().hex)
    admitted(claim)
    control = Control(claim["reserved_seconds"])
    refs = []
    complete = 0
    error = None
    try:
        for method in locked["methods"]:
            control.check()
            p, result = planning(store, locked["plans"][method])
            if domain_identity(p) != manifest["domain_sha256"]:
                raise ValueError("Evaluation domain changed")
            for descriptor in manifest["samples"]:
                control.check()
                sid = descriptor["sample_id"]
                if (method, sid) in old:
                    verify_cell(store, run, locked, descriptor, old[method, sid][1], p, result)
                    continue
                objects = seal_sample(store, input_root, descriptor)
                sc = scene(store, descriptor, p)
                ep = replace(
                    p, scenario_ids=(sid,), scenario=lambda s: sc, ambiguity=None, det_proof=None
                )
                report = audit_plan(ep, result["plan"]["entries"], control, retain_witness=True)
                witness = store.put_object(encode(plain(report.checks["scenarios"][0])))
                row = dict(
                    kind="evaluation_sample",
                    study=run["study"],
                    run=run_ref.as_dict(),
                    method=method,
                    sample_id=sid,
                    plan_sha256=locked["plans"][method]["plan_sha256"],
                    source=run["source"],
                    manifest=run["manifest"],
                    lock=run["lock"],
                    protocol=locked["protocol"],
                    random_stream=locked["random_stream"],
                    metrics=metrics(report),
                    witness=witness.as_dict(),
                    scientific="pass",
                    observed_at=utcnow(),
                    claim=claim["attempt"],
                )
                verify_cell(store, run, locked, descriptor, row, p, result)
                ref = store.put("events", row, [run_ref, witness, *objects.values()])
                refs.append(ref)
                complete += 1
                if max_cells is not None and complete >= max_cells:
                    raise InterruptedError("Explicit slice cell boundary")
    except (TimeoutError, InterruptedError) as exc:
        error = str(exc)
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        event = store.put(
            "events",
            dict(
                kind="evaluation_slice",
                study=run["study"],
                run=run_ref.as_dict(),
                new_cells=complete,
                error=error,
                observed_at=utcnow(),
                claim=claim["attempt"],
            ),
            [run_ref, *refs],
        )
        settlement = ledger.settle(claim["attempt"], 0 if error is None else 2, event.as_dict())
        store.put(
            "events",
            dict(
                kind="evaluation_settlement",
                study=run["study"],
                run=run_ref.as_dict(),
                settlement=settlement,
                observed_at=utcnow(),
            ),
            [run_ref, event],
        )
    return audit(store, run_ref, register=True)


def audit(store, run_ref, *, register=False):
    run, locked, manifest = describe(store, run_ref)
    store.closure([run_ref])
    rows = rows_for(store, run_ref)
    expected = {(m, s) for m in locked["methods"] for s in locked["sample_ids"]}
    if set(rows) - expected:
        raise ValueError("Unexpected evaluation sample/method")
    observations = {m: [] for m in locked["methods"]}
    refs = [run_ref]
    for method in locked["methods"]:
        p, result = planning(store, locked["plans"][method])
        if domain_identity(p) != manifest["domain_sha256"]:
            raise ValueError("Evaluation domain changed")
        for descriptor in manifest["samples"]:
            key = method, descriptor["sample_id"]
            if key not in rows:
                continue
            ref, row = rows[key]
            store.closure([ref])
            values = verify_cell(store, run, locked, descriptor, row, p, result)
            observations[method].append(dict(sample_id=key[1], **values))
            refs.append(ref)
    settlements = [
        r
        for r, v in store.records("events")
        if v["payload"].get("kind") == "evaluation_settlement"
        and v["payload"].get("run") == run_ref.as_dict()
    ]
    refs += settlements
    accounted = {}
    for ref in settlements:
        value = store.read(ref)["payload"]["settlement"]
        claim = value["attempt"]
        if (
            claim in accounted
            or value["charged_seconds"] < 0
            or not np.isfinite(value["charged_seconds"])
        ):
            raise ValueError("Duplicate or invalid evaluation settlement")
        receipt = Ref(**value["receipt"])
        event = store.read(receipt)["payload"]
        if event["run"] != run_ref.as_dict() or event["claim"] != claim:
            raise ValueError("Evaluation settlement receipt mismatch")
        receipt_refs = {Ref(**r) for r in store.read(receipt)["refs"]}
        if any(r not in receipt_refs for r, row in rows.values() if row["claim"] == claim):
            raise ValueError("Evaluation cell missing from its settlement receipt")
        accounted[claim] = value["charged_seconds"]
    settled = all(row["claim"] in accounted for _, row in rows.values())
    finished = set(rows) == expected
    stats = {}
    if finished:
        for metric in METRICS:
            selected = {
                m: [dict(sample_id=v["sample_id"], cost=v[metric]) for v in values]
                for m, values in observations.items()
            }
            stats[metric] = paired_summary(locked, selected)
    report = dict(
        kind="paired_evaluation_audit",
        study=run["study"],
        run=run_ref.as_dict(),
        passed=finished and bool(settlements) and settled,
        charged_seconds=sum(accounted.values()),
        all_cells_settled=settled,
        complete=finished,
        cells=len(rows),
        required_cells=len(expected),
        scope=manifest["scope"],
        metrics=stats,
        solver_calls=0,
        missing=[list(k) for k in sorted(expected - set(rows))],
        source=run["source"],
        sample_count=len(locked["sample_ids"]),
        protocol=locked["protocol"],
    )
    if register:
        ref = store.put("audits", report, refs)
        report = dict(report, audit_ref=ref.as_dict())
    return report


def recover(store, run_ref):
    """Explicitly settle a proven dead evaluation owner and archive its exact ledger event."""
    run, _, _ = describe(store, run_ref)
    ledger = BudgetLedger(store.root, run["budget_group"])
    active = ledger.read()["active"]
    if active and active["run"] != run_ref.as_dict():
        raise ValueError("Cannot recover another run in this budget group")
    ledger.recover_dead_claim()  # Rejects a live/unknown process, including PID reuse.
    snapshot = ledger.read()
    sealed = {
        v["payload"]["settlement"]["attempt"]
        for _, v in store.records("events")
        if v["payload"].get("kind") == "evaluation_settlement"
        and v["payload"].get("run") == run_ref.as_dict()
    }
    claims = {
        e["payload"]["attempt"]
        for e in snapshot["events"]
        if e["payload"]["kind"] == "claim" and e["payload"]["run"] == run_ref.as_dict()
    }
    rows = rows_for(store, run_ref)
    added = []
    for event in snapshot["events"]:
        value = event["payload"]
        if (
            value["kind"] != "settlement"
            or value["attempt"] not in claims
            or value["attempt"] in sealed
        ):
            continue
        claim = value["attempt"]
        proof = store.put_object(
            encode(dict(definition=snapshot["definition"], events=snapshot["events"]))
        )
        cellrefs = [r for r, row in rows.values() if row["claim"] == claim]
        receipt = store.put(
            "events",
            dict(
                kind="evaluation_recovery",
                study=run["study"],
                run=run_ref.as_dict(),
                claim=claim,
                original_settlement=value,
                ledger=proof.as_dict(),
                observed_at=utcnow(),
            ),
            [run_ref, proof, *cellrefs],
        )
        ref = store.put(
            "events",
            dict(
                kind="evaluation_settlement",
                study=run["study"],
                run=run_ref.as_dict(),
                settlement=dict(value, receipt=receipt.as_dict()),
                recovered=True,
                observed_at=utcnow(),
            ),
            [run_ref, receipt],
        )
        added.append(ref.as_dict())
    return dict(
        recovered=added,
        spent_seconds=snapshot["spent_seconds"],
        remaining_seconds=snapshot["remaining_seconds"],
    )
