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
