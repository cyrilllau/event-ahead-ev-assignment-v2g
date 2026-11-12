import subprocess
import sys
import pytest
from ev_assignment_experiments.config.experiment import Experiment
from ev_assignment_experiments.tracking.store import Store
from ev_assignment_experiments.workflows.probability import plan, execute
from ev_assignment_experiments.tracking.decisions import decide
from ev_assignment_experiments.publication.bundle import build, verify, select, render
from ev_assignment_experiments.runtime.release import verify as verify_release


def test_selected_bundle_has_exact_installable_source_and_reproducible_tables(
    tmp_path, monkeypatch
):
    store = Store(tmp_path / "store")
    exp = Experiment.parse(
        dict(
            schema_version=1,
            method="development_probability_oracle",
            study="s",
            title="t",
            support_ids=["a", "b"],
            feature_ids=["f"],
            features=[[0], [1]],
            lower=[0.25],
            upper=[0.5],
            q=[0, 10],
            seconds=5,
        )
    )
    attempt = execute(store, plan(store, exp))
    original = store.path(attempt).read_bytes()
    decision = decide(
        store, attempt, "adopted", "Independent numerical audit passed", actor="test-researcher"
    )
    item = dict(
        label="table-smoke",
        attempt=attempt.as_dict(),
        decision=decision.as_dict(),
        generator="objective_table",
    )
    publication = select(store, "s", "Explicit demonstration", [item])
    out = tmp_path / "bundle"
    manifest = build(store, [attempt], out, publication=publication)
    assert verify(out)["passed"] and store.path(attempt).read_bytes() == original
    assert len(manifest["source_releases"]) == 1
    release = out / "code" / next(iter(manifest["source_releases"]))
    assert verify_release(release)["passed"]
    before = (out / "tables/objectives.csv").read_bytes()
    render(out)
    assert (out / "tables/objectives.csv").read_bytes() == before
    subprocess.run(
        [sys.executable, str(out / "reproduce.py"), "figures"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert manifest["generator_source"]["id"] in manifest["source_releases"]
    import io, zipfile

    bad_source = io.BytesIO()
    with zipfile.ZipFile(bad_source, "w") as z:
        z.writestr("ev_assignment/__init__.py", "changed generator")
    with monkeypatch.context() as patcher:
        from ev_assignment_experiments.workflows import probability

        patcher.setattr(probability, "source_snapshot", lambda: bad_source.getvalue())
        with pytest.raises(ValueError, match="exact generator_source"):
            render(out)
    (release / "src/ev_assignment/__init__.py").write_text("changed")
    with pytest.raises(ValueError, match="release changed"):
        verify(out)


def test_failed_or_unaccepted_selection_rejected(tmp_path):
    store = Store(tmp_path)
    a = store.put("attempts", dict(study="s", execution="failed", convergence="unknown"))
    d = decide(store, a, "rejected", "Native failure", actor="researcher")
    with pytest.raises(ValueError, match="accepted decision"):
        select(
            store,
            "s",
            "x",
            [
                dict(
                    label="fig1",
                    attempt=a.as_dict(),
                    decision=d.as_dict(),
                    generator="objective_table",
                )
            ],
        )
