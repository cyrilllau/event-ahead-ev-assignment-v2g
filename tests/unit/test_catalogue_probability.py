from types import SimpleNamespace
import numpy as np

from ev_assignment.domain.planning import Control
from ev_assignment.methods.catalogue_probability import calibrated


def test_probability_identity_binds_bounds_rule_and_order(monkeypatch):
    # Stub the feasibility oracle: this unit test checks domain identity, not
    # whether an LP was solved. Actual probability LPs belong to remote stages.
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k:
                        SimpleNamespace(success=True, status=0, x=np.array([.25] * 4 + [.1])))
    features = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    def create(ids=("a", "b", "c", "d"), lower=(.1, .1, .1), rule="preregistered"):
        return calibrated(ids, features, features, Control(1), lower=lower,
                          upper=[.9] * 3, calibration_rule=rule)[0]
    domain = create()
    assert domain.identity == create().identity
    assert domain.identity != create(lower=(.11, .1, .1)).identity
    assert domain.identity != create(ids=("d", "c", "b", "a")).identity
    assert domain.identity != create(rule="different-calibration").identity
    assert not domain.check_original(np.array([0., 0., 0., 1.]))["passed"]
