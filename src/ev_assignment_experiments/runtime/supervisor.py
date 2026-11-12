"""Local isolated parent/worker supervision. Never attaches to a legacy campaign."""

from __future__ import annotations
import argparse
import fcntl
import json
import subprocess
import sys
import time
import uuid

from .identity import current, matches, wait_identity
from .ledger import BudgetLedger, publish
from ev_assignment_experiments.tracking.store import Store, Ref, encode, sha256
from ev_assignment_experiments.workflows.l08 import execute, recipe_for, atomic_pointer


def launch(store, run_ref, *, resume=None):
    run = store.read(run_ref)["payload"]
    recipe = recipe_for(store, run)
    if resume:
        old = store.read(resume)["payload"]
        if old["run"] != run_ref.as_dict():
            raise ValueError("Resume must use the exact run")
    token = uuid.uuid4().hex
    folder = store.root / "runtime" / "launches" / token
    folder.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "ev_assignment_experiments.runtime.supervisor",
        "--role",
        "parent",
        "--store",
        str(store.root),
        "--run",
        run_ref.id,
        "--token",
        token,
    ]
    if resume:
        command += ["--resume", resume.id]
    with (folder / "supervisor.log").open("xb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True
        )
    until = time.monotonic() + 10
    while time.monotonic() < until:
        receipt = folder / "ADMISSION.json"
        if receipt.exists():
            return json.loads(receipt.read_text())
        rejected = folder / "REJECTED.json"
        if rejected.exists():
            raise ValueError(json.loads(rejected.read_text())["error"])
        if process.poll() is not None:
            raise RuntimeError("Supervisor exited before admission; inspect " + str(folder))
        time.sleep(0.02)
    return dict(
        status="starting", token=token, owner=wait_identity(process.pid), study=recipe.study
    )


def parent(store, run_ref, token, resume):
    run = store.read(run_ref)["payload"]
    recipe = recipe_for(store, run)
    folder = store.root / "runtime" / "launches" / token
    study_root = store.root / "runtime" / "studies" / sha256(recipe.study.encode())
    study_root.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(store.root, recipe.budget_group)
    lock = (study_root / "lock").open("a")
    claim = None
    p = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = study_root / "CURRENT.json"
        if pointer.exists():
            prior = json.loads(pointer.read_text())
            previous_ledger = BudgetLedger(store.root, prior["budget_group"]).read()
            if previous_ledger["active"] is not None:
                raise ValueError(
                    "Study has an unsettled task; explicit dead-owner recovery required"
                )
        # Same run cannot be silently restarted after a completed/failed attempt.
        prior_runs = [
            r for _, r in store.records("attempts") if r["payload"].get("run") == run_ref.as_dict()
        ]
        if prior_runs and resume is None:
            raise ValueError("Existing run requires an explicit checkpoint resume")
        if resume:
            payload = store.read(resume)["payload"]
            if payload["run"] != run_ref.as_dict():
                raise ValueError("Wrong resume run")
        claim = ledger.claim(run_ref.as_dict(), recipe.options.seconds, attempt=token)
        area = store.root / "runtime" / "attempts" / token
        area.mkdir(parents=True, exist_ok=False)
        publish(area / "CLAIM.json", claim)
        atomic_pointer(
            pointer,
            dict(
                attempt=token,
                run=run_ref.as_dict(),
                budget_group=recipe.budget_group,
                owner=current(),
            ),
        )
        command = [
            sys.executable,
            "-m",
            "ev_assignment_experiments.runtime.supervisor",
            "--role",
            "worker",
            "--store",
            str(store.root),
            "--run",
            run_ref.id,
            "--token",
            token,
        ]
        if resume:
            command += ["--resume", resume.id]
        with (area / "stdout.log").open("xb") as output:
            p = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ready_deadline = time.monotonic() + 5
            while not (area / "READY.json").exists():
                if p.poll() is not None or time.monotonic() >= ready_deadline:
                    raise RuntimeError("Worker failed before ready handshake")
                time.sleep(0.01)
            identity = json.loads((area / "READY.json").read_text())
            if identity["pid"] != p.pid or matches(identity) is not True:
                raise ValueError("Ready process is not our live worker")
            ledger.child(token, identity)
            publish(area / "GO.json", dict(child=identity, parent=current()))
            publish(
                folder / "ADMISSION.json",
                dict(
                    status="running",
                    study=recipe.study,
                    run=run_ref.as_dict(),
                    attempt=token,
                    child=identity,
                    owner=current(),
                    budget_group=recipe.budget_group,
                    admitted_seconds=claim["reserved_seconds"],
                ),
            )
            deadline = claim["started_monotonic"] + claim["reserved_seconds"]
            while p.poll() is None:
                if time.monotonic() >= deadline and not (area / "STOP.json").exists():
                    publish(
                        area / "STOP.json",
                        dict(reason="cumulative_budget_deadline", epoch=time.time()),
                    )
                if time.monotonic() >= deadline + 5 and matches(identity) is True:
                    p.terminate()  # Our verified child only; never a campaign process.
                if time.monotonic() >= deadline + 10 and matches(identity) is True:
                    p.kill()
                time.sleep(0.05)
            rc = p.wait()
        wait = dict(
            actual_parent_wait=True,
            returncode=rc,
            owner=current(),
            child=identity,
            elapsed_seconds=time.monotonic() - claim["started_monotonic"],
            observed_epoch=time.time(),
        )
        publish(area / "WAIT.json", wait)
        if not (area / "END.json").exists():
            error = store.put_object(
                encode(dict(error="Worker exited without sealed outcome", returncode=rc))
            )
            cp = (
                json.loads((area / "CHECKPOINT.json").read_text())
                if (area / "CHECKPOINT.json").exists()
                else None
            )
            refs = [run_ref, error] + ([Ref(**cp)] if cp else [])
            record = store.put(
                "attempts",
                dict(
                    schema_version=2,
                    study=recipe.study,
                    run=run_ref.as_dict(),
                    source=run["source"],
                    scope=run["scope"],
                    attempt_id=token,
                    execution="failed",
                    scientific="unchecked",
                    convergence="unknown",
                    decision="pending",
                    result=error.as_dict(),
                    checkpoint=cp,
                    observed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    accepted_for_comparison=False,
                ),
                refs,
            )
            publish(area / "END.json", dict(attempt=record.as_dict(), returncode=rc))
        end = json.loads((area / "END.json").read_text())
        attempt = Ref(**end["attempt"])
        wait_ref = store.put_object(encode(wait))
        stdout_ref = store.put_file(area / "stdout.log")
        ledger.settle(token, rc, wait_ref.as_dict())
        ledger_ref = store.put_object(encode(ledger.read()))
        evidence = store.put(
            "events",
            dict(
                study=recipe.study,
                kind="parent_settlement",
                attempt=end["attempt"],
                run=run_ref.as_dict(),
                budget_group=recipe.budget_group,
                ledger=ledger_ref.as_dict(),
                wait=wait_ref.as_dict(),
                stdout=stdout_ref.as_dict(),
            ),
            [attempt, ledger_ref, wait_ref, stdout_ref],
        )
        publish(
            folder / "COMPLETE.json",
            dict(attempt=end["attempt"], settlement=evidence.as_dict(), returncode=rc),
        )
    except BaseException as exc:
        if p is not None and p.poll() is None:
            # Keep the worker protected by the durable claim. No blind cleanup.
            error = "Parent failed with live child; inspect identities before recovery: " + repr(
                exc
            )
        else:
            error = repr(exc)
        if not (folder / "ADMISSION.json").exists():
            publish(folder / "REJECTED.json", dict(error=error))
        else:
            publish(folder / "PARENT_FAILURE.json", dict(error=error))
        raise
    finally:
        lock.close()


def worker(store, run_ref, token, resume):
    area = store.root / "runtime" / "attempts" / token
    publish(area / "READY.json", current())
    until = time.monotonic() + 5
    while not (area / "GO.json").exists():
        if time.monotonic() > until:
            raise RuntimeError("Parent admission handshake missing")
        time.sleep(0.01)
    go = json.loads((area / "GO.json").read_text())
    claim = json.loads((area / "CLAIM.json").read_text())
    if go["child"] != current() or matches(go["parent"]) is not True:
        raise ValueError("Worker is not the admitted child")
    remaining = claim["reserved_seconds"] - (time.monotonic() - claim["started_monotonic"])
    if remaining <= 0:
        raise TimeoutError("Budget exhausted before worker entry")
    ref = execute(
        store, run_ref, token, remaining, resume=resume, stop=lambda: (area / "STOP.json").exists()
    )
    return 0 if store.read(ref)["payload"]["execution"] == "exited" else 2


def status(store, token):
    folder = store.root / "runtime" / "launches" / token
    if not token.isalnum():
        raise ValueError("Invalid launch identity")
    if (folder / "COMPLETE.json").exists():
        return dict(status="settled", **json.loads((folder / "COMPLETE.json").read_text()))
    if (folder / "REJECTED.json").exists():
        return dict(status="rejected", **json.loads((folder / "REJECTED.json").read_text()))
    if (folder / "ADMISSION.json").exists():
        row = json.loads((folder / "ADMISSION.json").read_text())
        owner, child = matches(row["owner"]), matches(row["child"])
        return dict(
            row,
            status="running" if owner is True or child is True else "unknown",
            owner_observed=owner,
            child_observed=child,
            observed_epoch=time.time(),
        )
    return dict(status="unknown", token=token, observed_epoch=time.time())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["parent", "worker"], required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--resume")
    a = p.parse_args(argv)
    store = Store(a.store)
    run = Ref("runs", a.run)
    resume = Ref("checkpoints", a.resume) if a.resume else None
    return (
        parent(store, run, a.token, resume)
        if a.role == "parent"
        else worker(store, run, a.token, resume)
    )


if __name__ == "__main__":
    raise SystemExit(main())
