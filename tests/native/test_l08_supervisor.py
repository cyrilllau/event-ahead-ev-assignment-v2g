import json
import time
import pytest
from ev_assignment_experiments.config.run import RunPlan
from ev_assignment_experiments.runtime.supervisor import launch, status
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.tracking.store import Store, Ref
from ev_assignment_experiments.workflows.l08 import plan
from ev_assignment_experiments.workflows.audit import reaudit


def recipe(method):
    return RunPlan.parse(
        dict(
            schema_version=2,
            study="native-" + method,
            title="Real native small",
            method=method,
            input_kind="synthetic_l08",
            budget_group="native-" + method,
            budget_seconds=60,
            solver=dict(seconds=25, absolute_gap=1e-5),
        )
    )


def await_done(store, admission):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        value = status(store, admission["attempt"])
        if value["status"] == "settled":
            return value
        time.sleep(0.05)
    raise AssertionError("Own test supervisor did not settle")


@pytest.mark.parametrize("method", ["DET", "DRO", "ARO"])
def test_real_parent_worker_audit_and_offline_archive(tmp_path, method):
    store = Store(tmp_path / "source")
    r = recipe(method)
    run = plan(store, r)
    admission = launch(store, run)
    done = await_done(store, admission)
    attempt = Ref(**done["attempt"])
    payload = store.read(attempt)["payload"]
    assert done["returncode"] == 0, payload
    assert payload["execution"] == "exited" and payload["scientific"] == "pass"
    assert payload["convergence"] == "met" and payload["decision"] == "pending"
    audit = reaudit(store, attempt, register=True)
    assert audit["passed"] and audit["solver_calls"] == 0
    ledger = BudgetLedger(store.root, r.budget_group).read()
    assert ledger["active"] is None
    claim = ledger["events"][0]["payload"]
    assert claim["charged_from_process_start"] and claim["startup_seconds"] > 0
    assert claim["started_epoch"] == claim["owner"]["created"]
    assert ledger["spent_seconds"] >= payload["elapsed_seconds"] > 0
    with pytest.raises(ValueError, match="explicit checkpoint"):
        launch(store, run)
    destination = Store(tmp_path / "offline")
    store.sync_to(destination, [attempt])
    assert reaudit(destination, attempt)["passed"]
    assert not payload["accepted_for_comparison"]
    if method == "ARO":
        checkpoint = Ref(**payload["checkpoint"])
        second = await_done(store, launch(store, run, resume=checkpoint))
        assert second["returncode"] == 0
        assert second["attempt"] != done["attempt"]
        assert (
            BudgetLedger(store.root, r.budget_group).read()["spent_seconds"]
            > ledger["spent_seconds"]
        )
        assert reaudit(store, Ref(**second["attempt"]))["passed"]
    raw = json.loads(store.object(Ref(**payload["result"])))
    row = raw["audit"]["checks"]["scenarios"][0]
    evidence = json.loads(store.object(Ref(**row["witness"])))
    evidence["evidence"]["y"][0] += 100
    # Tampered evidence gets a new identity and still fails the physical audit.
    from ev_assignment_experiments.tracking.store import encode

    new_witness = store.put_object(encode(evidence))
    row["witness"] = new_witness.as_dict()
    bad_result = store.put_object(encode(raw))
    bad = store.put(
        "attempts", dict(payload, result=bad_result.as_dict()), [run, bad_result, new_witness]
    )
    with pytest.raises(ValueError):
        reaudit(store, bad)
