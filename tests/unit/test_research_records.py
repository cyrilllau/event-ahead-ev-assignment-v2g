import pytest
from ev_assignment.domain.planning import Bound, BoundScope, combine_bounds
from ev_assignment_experiments.analysis.evaluation import open_sample, paired_summary
from ev_assignment_experiments.tracking.store import Store
from ev_assignment_experiments.tracking.context import context
from ev_assignment_experiments.tracking.decisions import next_action
from ev_assignment_experiments.tracking.sync import synchronize
from ev_assignment_experiments.tracking.transfer import FilesystemSource
from ev_assignment_experiments.config.run import RunPlan


def test_bounds_cannot_cross_methods_or_scenarios():
    a = Bound(1, BoundScope("p", "DRO", ("s0",)), "proof", "native_solver")
    for scope in (
        BoundScope("p", "ARO", ("s0",)),
        BoundScope("p", "DRO", ("s1",)),
        BoundScope("q", "DRO", ("s0",)),
    ):
        with pytest.raises(ValueError, match="different"):
            combine_bounds(a, Bound(2, scope, "audit", "numerical_audit"))
    assert combine_bounds(a, None) is None
    with pytest.raises(ValueError, match="Negative"):
        combine_bounds(a, Bound(0.5, a.scope, "audit", "numerical_audit"))


def test_oos_is_not_opened_by_a_missing_lock(tmp_path):
    store = Store(tmp_path)
    fake = store.put("experiments", dict(kind="planning", sample_ids=["secret"]))
    opened = []
    with pytest.raises(ValueError, match="locked"):
        open_sample(store, fake, "secret", lambda x: opened.append(x))
    assert opened == []


def test_pairing_order_missing_values_and_separation():
    lock = dict(
        methods=["DET", "DRO"], sample_ids=["s1", "s0"], protocol="same-outages-fixed-routes"
    )
    a = [dict(sample_id="s1", cost=2), dict(sample_id="s0", cost=4)]
    b = [dict(sample_id="s1", cost=1), dict(sample_id="s0", cost=5)]
    assert paired_summary(lock, {"DET": a, "DRO": b})["methods"]["DRO"]["paired_difference"] == 0
    with pytest.raises(ValueError, match="order"):
        paired_summary(lock, {"DET": a, "DRO": b[::-1]})
    b[0]["cost"] = float("nan")
    with pytest.raises(ValueError, match="not zero"):
        paired_summary(lock, {"DET": a, "DRO": b})


def test_context_pagination_explicit_next_and_metadata_not_archived(tmp_path):
    source = Store(tmp_path / "source")
    dest = Store(tmp_path / "dest")
    raw = source.put_object(b"private-large-evidence")
    attempts = [
        source.put(
            "attempts",
            dict(
                study="s",
                execution="failed",
                scientific="unchecked",
                observed_at=str(i),
                result=raw.as_dict(),
            ),
            [raw],
        )
        for i in range(4)
    ]
    action = next_action(
        source,
        "s",
        "Inspect failed physics",
        dependencies=[attempts[-1]],
        reason="recorded failure",
        actor="researcher",
    )
    result = context(source, "s", 2)
    second = context(source, "s", 2, offset=result["next_offset"])
    assert len({r["id"] for r in result["attempts"] + second["attempts"]}) == 4
    assert (
        result["next_action"]["ref"] == action.as_dict()
        and "Inspect failed physics" in result["markdown"]
    )
    receipt = synchronize(FilesystemSource(source), dest, attempts, mode="metadata")
    assert not receipt["archived"] and not dest.path(raw).exists()
    synchronize(FilesystemSource(source), dest, [attempts[-1]], mode="results")
    assert dest.closure([attempts[-1]]) == source.closure([attempts[-1]])


def test_recipes_reject_budget_reset_and_hidden_machine_inputs():
    base = dict(
        schema_version=2,
        study="s",
        title="t",
        method="DET",
        input_kind="synthetic_l08",
        budget_group="g",
        budget_seconds=60,
        solver=dict(seconds=30),
    )
    for change in (
        {"budget_seconds": 10},
        {"input_root": "/hidden"},
        {"method": "SAA"},
        {"schema_version": True},
        {"input_kind": "frozen_l08"},
    ):
        with pytest.raises(ValueError):
            RunPlan.parse(dict(base, **change))


def test_cached_objects_still_detect_changed_bytes(tmp_path):
    store = Store(tmp_path)
    obj = store.put_object(b"original")
    store.closure([obj])
    store.path(obj).write_bytes(b"changed!")
    with pytest.raises(ValueError, match="Digest"):
        store.closure([obj])


def test_handoff_bounds_nested_remote_snapshots_and_large_human_notes(tmp_path):
    store = Store(tmp_path)
    ref = store.put(
        "observations",
        dict(study="s", observed_at="now", external_links=[{"note": "x" * 10000}] * 1000),
    )
    summary = context(store, "s")
    import json

    assert len(json.dumps(summary)) < 16000
    assert summary["observations"][0]["ref"] == ref.as_dict()
    assert summary["observations"][0]["external_links"][-1]["items_omitted"] == 990
