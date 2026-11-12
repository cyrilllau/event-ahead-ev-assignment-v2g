from types import SimpleNamespace
import numpy as np
import pytest

from ev_assignment.domain.planning import Control
from ev_assignment.methods.catalogue_probability import calibrated, ProbabilityDomainInfeasible, finite_domain


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


@pytest.mark.parametrize("status,error", [(1, TimeoutError), (2, ProbabilityDomainInfeasible), (4, RuntimeError)])
def test_probability_oracle_limit_is_not_domain_infeasibility(monkeypatch, status, error):
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k:
                        SimpleNamespace(success=False, status=status, message="synthetic termination"))
    features = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    with pytest.raises(error):
        calibrated(("a", "b", "c", "d"), features, features, Control(1),
                   lower=[.1] * 3, upper=[.9] * 3, calibration_rule="mock-only")


def test_finite_singleton_and_simplex_keep_different_scientific_domains():
    saa, record = finite_domain(("a", "b", "c"), weights=[.2, .3, .5])
    aro, _ = finite_domain(("a", "b", "c"))
    assert saa.check_original(saa.feasible_anchor)["passed"]
    assert record["rounding_l1"] < 1e-12
    assert not saa.check_original(np.array([1., 0., 0.]))["passed"]
    assert aro.check_original(np.array([1., 0., 0.]))["passed"]
    assert saa.identity != aro.identity
