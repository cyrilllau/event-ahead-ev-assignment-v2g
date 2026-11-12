import numpy as np
import pytest
from scipy import sparse as sp

from ev_assignment_experiments.analysis.diagnostic import (
    affine_bounds,
    fixed_master_risk,
    read_linear_mps,
    subtract_intervals,
)


def test_full_continuous_vector_and_protection():
    low, high = affine_bounds(sp.csr_matrix([[2.0, -3.0]]), [1.0], [0.25, 0.75])
    assert low[0] <= -0.75 <= high[0]
    with pytest.raises(ValueError, match="mapping"):
        affine_bounds(sp.eye(2), [1.0, 2.0], [0.5])


def test_interval_difference_does_not_hide_violation():
    assert subtract_intervals([3.0, 5.0], [2.0, 4.0]) == [-1.0, 3.0]
    assert subtract_intervals([3.0, 5.0], [2.0, 4.0], monotone=True) == [0.0, 3.0]
    with pytest.raises(ValueError, match="Negative"):
        subtract_intervals([1.0, 2.0], [3.0, 4.0], monotone=True)


def test_fixed_risk_minimizes_aux_not_saved_slack():
    # theta0 >= 2*x+1; theta1>=3; rho>=.25*theta0+.75*theta1;
    # legacy rho>=4*x. At x=.5 min risk=2.75, not an arbitrary saved rho=20.
    A = sp.csr_matrix(
        [
            [-2.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, -0.25, -0.75],
            [-4.0, 1.0, 0.0, 0.0],
        ]
    )
    data = dict(rhs=np.array([1.0, 3.0, 0.0, 0.0]), sense=np.array([">"] * 4), lo=np.zeros(4))
    names = dict(
        cols=["x", "DRO_RHO", "THETA_SCENARIO_0", "THETA_SCENARIO_1"],
        rows=["theta_support/0/a", "adaptive/1/b", "theta_probability/p", "old"],
    )
    bound, _, counts = fixed_master_risk(A, data, names, [0.5], 1, "DRO", 0)
    assert bound[0] <= 2.75 <= bound[1]
    assert counts == dict(scenario=2, probability=1, aggregate=1, floor=0)
    A[2, 2] = 0.25
    with pytest.raises(ValueError, match="Non-monotone"):
        fixed_master_risk(A, data, names, [0.5], 1, "DRO", 0)


def test_mps_preserves_names_order_bounds_objective(tmp_path):
    p = tmp_path / "tiny.mps"
    p.write_text(
        "NAME T\nROWS\n N OBJ\n G row\nCOLUMNS\n"
        " x OBJ 2 row -3\n y OBJ 1 row 1\nRHS\n b row 4\n"
        "BOUNDS\n BV b x\n FR b y\nENDATA\n"
    )
    A, data, names = read_linear_mps(p)
    assert names == dict(rows=["row"], cols=["x", "y"])
    np.testing.assert_array_equal(A.toarray(), [[-3.0, 1.0]])
    np.testing.assert_array_equal(data["vtype"], ["B", "C"])
    assert data["hi"][0] == 1 and np.isneginf(data["lo"][1])


def test_parent_charges_worker_import_failure_without_invalid_main_entry(tmp_path):
    import json
    import subprocess
    import sys

    out = tmp_path / "attempt"
    # The explicit probe fails before any solver import, even if a license is
    # installed: actual module launch and failed-child budget settlement.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ev_assignment_experiments.analysis.diagnostic_native",
            "--root",
            str(tmp_path / "legacy"),
            "--replay",
            str(tmp_path / "replay"),
            "--out",
            str(out),
            "--store",
            str(tmp_path / "store"),
            "--stage",
            "ENTRYPOINT_CHECK",
            "--code-commit",
            "test",
            "--seconds",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "__main__.__spec__" not in (out / "worker.log").read_text()
    assert json.loads((out / "ENTRYPOINT.json").read_text())["solver_calls"] == 0
    budget = json.loads((out / "BUDGET_AFTER.json").read_text())
    assert budget["active"] is None
    assert 0 < budget["spent_seconds"] < 5
    assert json.loads((out / "WAIT.json").read_text())["own_child_wait"]
