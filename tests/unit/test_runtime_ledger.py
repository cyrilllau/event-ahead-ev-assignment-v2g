import json
import time
import pytest
from ev_assignment_experiments.runtime.identity import current, matches
from ev_assignment_experiments.runtime.ledger import BudgetLedger


def test_budget_inheritance_real_parent_wait_and_no_reset(tmp_path):
    ledger = BudgetLedger(tmp_path, "same-science-budget")
    ledger.create("study", 10, inherited=2, evidence=["historical-receipt-sha"])
    claim = ledger.claim({"id": "run1"}, 9, "attempt1")
    assert claim["reserved_seconds"] == 8
    with pytest.raises(ValueError, match="Unsettled"):
        ledger.claim({"id": "run2"}, 1)
    with pytest.raises(ValueError, match="live or unknown"):
        ledger.recover_dead_claim()
    time.sleep(0.01)
    settled = ledger.settle("attempt1", 0, {"receipt": "wait"})
    assert settled["actual_parent_wait"] and settled["charged_seconds"] >= 0.01
    assert ledger.read()["spent_seconds"] == 2 + settled["charged_seconds"]
    with pytest.raises(ValueError, match="reset"):
        ledger.create("study", 10)
    with pytest.raises(ValueError, match="Only"):
        ledger.settle("attempt1", 0, {})
    second = ledger.claim({"id": "new-code-run"}, 10, "attempt2")
    assert second["reserved_seconds"] < 8
    ledger.settle("attempt2", 2, {})


def test_budget_chain_tampering_detected(tmp_path):
    ledger = BudgetLedger(tmp_path, "g")
    ledger.create("s", 2)
    ledger.claim({}, 1)
    path = ledger.root / "events/00000000.json"
    row = json.loads(path.read_text())
    row["payload"]["reserved_seconds"] = 100
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="corrupt"):
        ledger.read()


def test_pid_reuse_and_remote_observation_are_not_liveness():
    identity = current()
    assert matches(identity) is True
    assert matches(dict(identity, created=identity["created"] - 10)) is False
    assert matches(dict(identity, host="a-different-machine")) is None


def test_dead_owner_recovery_charges_reservation_and_refuses_live_child(tmp_path, monkeypatch):
    import ev_assignment_experiments.runtime.ledger as module

    ledger = BudgetLedger(tmp_path, "g")
    ledger.create("s", 10)
    claim = ledger.claim({}, 7, "a")
    ledger.child("a", current())
    monkeypatch.setattr(module, "matches", lambda i: None)
    with pytest.raises(ValueError, match="unknown"):
        ledger.recover_dead_claim()
    monkeypatch.setattr(module, "matches", lambda i: False)
    ledger.recover_dead_claim()
    assert ledger.read()["spent_seconds"] == claim["reserved_seconds"] == 7
    assert ledger.read()["remaining_seconds"] == 3


def test_real_dead_parent_claim_preserves_budget(tmp_path):
    import subprocess
    import sys

    source = 'from ev_assignment_experiments.runtime.ledger import BudgetLedger; import sys; b=BudgetLedger(sys.argv[1],"dead"); b.create("study",10); b.claim({},7,"dead-parent")'
    subprocess.run([sys.executable, "-c", source, str(tmp_path)], check=True)
    ledger = BudgetLedger(tmp_path, "dead")
    assert ledger.read()["active"] is not None
    settlement = ledger.recover_dead_claim()
    assert settlement["charged_seconds"] == 7 and not settlement["actual_parent_wait"]
    assert ledger.claim({"new": "run"}, 10)["reserved_seconds"] == 3


def test_evaluation_recovery_rejects_live_owner_and_archives_real_dead_claim(tmp_path):
    import subprocess
    import sys
    from ev_assignment_experiments.tracking.store import Store, Ref, encode
    from ev_assignment_experiments.workflows.evaluation import recover

    store = Store(tmp_path / "evaluation")
    sample = dict(
        sample_id="s0",
        origin="runtime-only fixture",
        physical=dict(path="not-opened", sha256="1" * 64),
        route_states=dict(path="not-opened", sha256="2" * 64),
    )
    spec = store.put_object(
        encode(
            dict(
                schema_version=1,
                scope="development",
                domain_sha256="3" * 64,
                protocol="L08_PAIRED_FIXED_PLAN_V1",
                random_stream="runtime-fixture",
                samples=[sample],
            )
        )
    )
    locked = store.put(
        "experiments",
        dict(
            sample_ids=["s0"], random_stream="runtime-fixture", protocol="L08_PAIRED_FIXED_PLAN_V1"
        ),
    )
    ledger = BudgetLedger(store.root, "eval")
    ledger.create("s", 10)
    run = store.put(
        "runs",
        dict(
            kind="paired_evaluation",
            study="s",
            budget_group="eval",
            lock=locked.as_dict(),
            manifest=spec.as_dict(),
        ),
        [locked, spec],
    )
    live = ledger.claim(run.as_dict(), 2)
    with pytest.raises(ValueError, match="live or unknown"):
        recover(store, run)
    ledger.settle(live["attempt"], 0, None)
    # Use another fresh ledger/run to exercise actual process death, not a mocked PID.
    BudgetLedger(store.root, "dead-eval").create("s", 10)
    run = store.put(
        "runs",
        dict(
            kind="paired_evaluation",
            study="s",
            budget_group="dead-eval",
            lock=locked.as_dict(),
            manifest=spec.as_dict(),
        ),
        [locked, spec],
    )
    program = """import sys,json,os
from ev_assignment_experiments.runtime.ledger import BudgetLedger
l=BudgetLedger(sys.argv[1],'dead-eval');l.create('s',10);l.claim(json.loads(sys.argv[2]),4);os._exit(0)
"""
    import json

    subprocess.run(
        [sys.executable, "-c", program, str(store.root), json.dumps(run.as_dict())], check=True
    )
    result = recover(store, run)
    assert (
        result["spent_seconds"] == 4
        and result["remaining_seconds"] == 6
        and len(result["recovered"]) == 1
    )
    assert recover(store, run)["recovered"] == []
    event = next(
        v["payload"]
        for _, v in store.records("events")
        if v["payload"].get("kind") == "evaluation_recovery"
    )
    assert not event["original_settlement"]["actual_parent_wait"]
    store.closure([Ref(**result["recovered"][0])])


def test_archived_run_cannot_silently_recreate_zero_cost_budget(tmp_path):
    from ev_assignment_experiments.tracking.store import Store

    source, mirror = Store(tmp_path / "source"), Store(tmp_path / "mirror")
    original = BudgetLedger(source.root, "scientific-budget")
    original.create("study", 10, inherited=3, evidence=["recorded-prior-cost"])
    run = source.put("runs", dict(study="study", budget_group="scientific-budget"))
    source.sync_to(mirror, [run])
    with pytest.raises(ValueError, match="refuse cost reset"):
        BudgetLedger(mirror.root, "scientific-budget").create("study", 10)
    assert not (BudgetLedger(mirror.root, "scientific-budget").root / "definition.json").exists()
    assert original.read()["spent_seconds"] == 3
    original.create("study", 10, inherited=3, evidence=["recorded-prior-cost"])
    # Explicit independent reproduction is possible, never aliases the historical group.
    BudgetLedger(mirror.root, "new-explicit-reproduction").create("study", 10)
    assert source.read(run) == mirror.read(run)
