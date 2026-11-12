import json

import pytest

from ev_assignment_experiments.config.experiment import Experiment
from ev_assignment_experiments.publication.bundle import build, verify
from ev_assignment_experiments.tracking.context import context
from ev_assignment_experiments.tracking.store import Ref, Store
from ev_assignment_experiments.workflows.probability import execute, plan, reaudit


def example(**overrides):
    return Experiment.parse(
        {
            "schema_version": 1,
            "method": "development_probability_oracle",
            "study": "example",
            "title": "Hand-computed objective 5",
            "support_ids": ["s0", "s1"],
            "feature_ids": ["f0"],
            "features": [[0], [1]],
            "lower": [0.25],
            "upper": [0.5],
            "q": [0, 10],
            "seconds": 5,
            **overrides,
        }
    )


def test_real_solve_independent_audit_transfer_and_bundle(tmp_path):
    store = Store(tmp_path / "remote")
    run = plan(store, example())
    attempt = execute(store, run)
    result = store.read(attempt)["payload"]
    assert result["execution"] == "exited"
    assert result["scientific"] == "pass"
    assert result["decision"] == "pending"
    raw = json.loads(store.object(Ref(**result["raw"])))
    assert raw["primal_objective_usd"] == pytest.approx(5)
    assert reaudit(store, attempt)["passed"]
    local = Store(tmp_path / "local")
    store.sync_to(local, [attempt])
    bundle = tmp_path / "bundle"
    build(local, [attempt], bundle)
    assert verify(bundle)["passed"]
    assert verify(bundle)["paper_facing"] is False
    summary = context(local, "example")
    assert summary["server_live_status"] == "not_observed"
    assert summary["adopted_results"] == []
    assert summary["attempts"][0]["audit"] == result["audit"]
    with pytest.raises(FileExistsError):
        execute(store, run)


def test_changed_execution_source_rejects_plan_before_claim(tmp_path, monkeypatch):
    from ev_assignment_experiments.workflows import probability

    store = Store(tmp_path)
    run = plan(store, example())
    monkeypatch.setattr(probability, "source_snapshot", lambda: b"changed")
    with pytest.raises(ValueError, match="changed since plan"):
        execute(store, run)
    assert not (tmp_path / "claims").exists()


def test_solver_failure_is_preserved_and_never_published(tmp_path, monkeypatch):
    from ev_assignment_experiments.workflows import probability

    store = Store(tmp_path / "store")
    run = plan(store, example())

    def fail(*args, **kwargs):
        raise RuntimeError("license unavailable")

    monkeypatch.setattr(probability, "solve_worst_distribution", fail)
    attempt = execute(store, run)
    result = store.read(attempt)["payload"]
    assert result["execution"] == "failed"
    assert result["scientific"] == "unchecked"
    assert result["convergence"] == "unknown"
    with pytest.raises(ValueError, match="failed independent audit"):
        build(store, [attempt], tmp_path / "bundle")


@pytest.mark.parametrize(
    "change",
    [
        {"method": "DRO"},
        {"seconds": float("nan")},
        {"seconds": True},
        {"seconds": 61},
        {"unknown": 1},
        {"q": [1]},
        {"support_ids": ["s0", "s0"]},
    ],
)
def test_configuration_fails_closed(change):
    with pytest.raises(ValueError):
        example(**change)


def test_context_reports_truncation_and_keeps_failed_attempts(tmp_path):
    store = Store(tmp_path)
    for i in range(13):
        store.put(
            "attempts",
            {
                "study": "s",
                "observed_at": f"2026-01-{i + 1:02d}",
                "execution": "failed",
                "scientific": "unchecked",
            },
        )
    result = context(store, "s", 3)
    assert result["total"] == 13 and result["omitted"] == 10
    assert len(result["attempts"]) == 3
    assert all(r["execution"] == "failed" for r in result["attempts"])


def test_interrupted_claim_is_visible_without_guessing_process_health(tmp_path, monkeypatch):
    from ev_assignment_experiments.workflows import probability

    store = Store(tmp_path)
    run = plan(store, example())

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(probability, "solve_worst_distribution", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(store, run)
    summary = context(store, "example")
    assert summary["total"] == 0
    assert summary["unsettled_claims"][0]["run"] == run.as_dict()
    assert summary["unsettled_claims"][0]["execution"] == "unknown"
    with pytest.raises(FileExistsError):
        execute(store, run)
