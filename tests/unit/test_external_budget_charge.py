import pytest

from ev_assignment_experiments.runtime.ledger import BudgetLedger


def test_external_compute_persists_and_reduces_future_admission(tmp_path):
    ledger = BudgetLedger(tmp_path, "shared")
    ledger.create("study", 100)
    ledger.charge_external("local-preflight", {"stage": "A2"}, 40,
                           {"measured": 11, "startup_reserve": 29, "receipt": "explicit"})
    assert ledger.read()["spent_seconds"] == 40
    with pytest.raises(ValueError, match="already charged"):
        ledger.charge_external("local-preflight", {"stage": "A2"}, 40, {"receipt": "explicit"})
    reopened = BudgetLedger(tmp_path, "shared")
    claim = reopened.claim({"stage": "A2"}, 100)
    assert claim["reserved_seconds"] == 60
    with pytest.raises(ValueError, match="active worker"):
        reopened.charge_external("another", {}, 1, {"receipt": "explicit"})


def test_external_charge_never_hides_overrun(tmp_path):
    ledger = BudgetLedger(tmp_path, "shared")
    ledger.create("study", 10)
    ledger.charge_external("observed", {}, 11, {"receipt": "explicit"})
    assert ledger.read()["remaining_seconds"] == 0
    assert ledger.read()["spent_seconds"] == 11
    with pytest.raises(ValueError, match="exhausted"):
        ledger.claim({}, 1)
