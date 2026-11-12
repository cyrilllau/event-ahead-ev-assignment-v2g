import copy
import gurobipy as gp
import numpy as np
import pytest
from ev_assignment.domain.planning import SolverOptions, Control
from ev_assignment.l08 import builders
from ev_assignment.methods.l08 import solve
from ev_assignment_experiments.datasets.synthetic import problem


@pytest.mark.parametrize("method", ["DET", "DRO", "ARO"])
def test_complete_method_against_direct_formulation_and_original_physics(method):
    p = problem(method)
    options = SolverOptions(seconds=30, absolute_gap=1e-5)
    events = []
    r = solve(
        p, method, options, control=Control(30, event=lambda e: events.append(copy.deepcopy(e)))
    )
    assert r.plan is not None and r.audit.passed and r.gap_passed
    assert r.upper.level == "numerical_audit"
    assert not r.accepted_for_comparison
    assert r.lower.value <= r.upper.value
    assert len(r.audit.checks["scenarios"]) == len(p.scenario_ids)
    assert len(r.plan.entries) == 1
    assert r.audit.checks["scenarios"][0]["actual_V2G_ac"] == pytest.approx(60.0)
    assert r.audit.checks["scenarios"][0]["critical_ENS"] == pytest.approx(0.0)
    with gp.Env(params={"OutputFlag": 0}) as env:
        m, _, _, _ = builders.extensive(p, method, options, env=env)
        try:
            m.optimize()
            assert m.Status == gp.GRB.OPTIMAL
            assert r.lower.value <= m.ObjVal + 1e-7 <= r.upper.value + 1e-6
            assert r.upper.value == pytest.approx(m.ObjVal, abs=1e-4)
        finally:
            m.dispose()
    assert any(e["kind"] == "checkpoint" for e in events)


def test_resume_after_complete_scenario_does_not_repeat_its_solve():
    p = problem("ARO")
    events = []
    stopped = [False]
    checkpoints = []

    def record(e):
        events.append(copy.deepcopy(e))
        if e["kind"] == "checkpoint":
            checkpoints.append(copy.deepcopy(e["state"]))
            if len(e["state"]["partial"]) == 1:
                stopped[0] = True

    r = solve(
        p,
        "ARO",
        SolverOptions(seconds=30),
        control=Control(30, stop=lambda: stopped[0], event=record),
    )
    assert r.upper is None and r.audit is None and not r.gap_passed
    pending = checkpoints[-1]
    assert len(pending["partial"]) == 1
    second = []
    out = solve(
        p,
        "ARO",
        SolverOptions(seconds=30),
        checkpoint=pending,
        control=Control(30, event=lambda e: second.append(copy.deepcopy(e))),
    )
    assert out.audit.passed and out.gap_passed
    evidence = [e["evidence"] for e in second if e["kind"] == "recourse_evidence"]
    first_point = np.asarray(pending["point"])
    at_first = [e for e in evidence if np.array_equal(e["point"], first_point)]
    assert [e["sid"] for e in at_first] == ["train/0001"]


def test_checkpoint_cannot_change_problem_or_method():
    p = problem("ARO")
    with pytest.raises(ValueError, match="Checkpoint"):
        solve(p, "ARO", SolverOptions(), checkpoint={"problem": "wrong"})
