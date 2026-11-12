import json
from pathlib import Path
import pytest
from ev_assignment_experiments.config.catalog import bind
from ev_assignment_experiments.tracking.store import Store
from ev_assignment_experiments.tracking.transfer import file_digest
from ev_assignment_experiments.tracking.legacy import import_manifest
from ev_assignment_experiments.datasets.evaluation import validate_manifest


@pytest.mark.parametrize("method", ["det", "dro", "aro"])
def test_frozen_recipe_requires_explicit_resources_and_preserves_science(pytestconfig, method):
    path = Path(pytestconfig.rootpath) / f"experiments/l08/frozen-{method}.toml"
    r = bind(
        path,
        input_root="/explicit/data",
        study="new-study",
        budget_group="new-budget",
        budget_seconds=120,
        seconds=60,
        master_seconds=20,
    )
    assert r.method == method.upper() and r.input_kind == "frozen_l08"
    assert r.options.absolute_gap == 1 and r.options.threads == 1
    with pytest.raises(ValueError, match="cumulative"):
        bind(
            path,
            input_root="/explicit/data",
            study="new-study",
            budget_group="new-budget",
            budget_seconds=10,
            seconds=60,
            master_seconds=20,
        )


def test_old_results_imported_as_unchecked_snapshot_not_new_acceptance(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    file = root / "result.json"
    file.write_text('{"status":"PASS"}')
    spec = dict(
        schema_version=1,
        study="s",
        title="old result",
        origin="old-release",
        observed_at="2026-11-11T00:00:00Z",
        scope="selected_reference_files",
        state="PASS in historical file",
        files=[
            dict(
                path=file.name,
                sha256=file_digest(file),
                bytes=file.stat().st_size,
                role="result",
                stability="observed_snapshot",
            )
        ],
    )
    store = Store(tmp_path / "new")
    ref = import_manifest(store, spec, root=str(root))
    p = store.read(ref)["payload"]
    assert (
        p["scientific"] == "unchecked" and p["execution"] == "unknown" and not p["process_adopted"]
    )
    other = Store(tmp_path / "offline")
    store.sync_to(other, [ref])
    file.unlink()
    assert other.read(ref) == store.read(ref)
    assert (
        json.loads(
            other.object(
                next(
                    r
                    for r in other.closure([ref])
                    if r.kind == "objects" and r.id == spec["files"][0]["sha256"]
                )
            )
        )["status"]
        == "PASS"
    )
    # Exact re-import reads from existing files, never pretends that removed source is live.
    with pytest.raises(FileNotFoundError):
        import_manifest(store, spec, root=str(root))


def test_legacy_import_bad_digest_and_paths_never_publish_observation(tmp_path):
    f = tmp_path / "f"
    f.write_text("data")
    store = Store(tmp_path / "store")
    spec = dict(
        schema_version=1,
        study="s",
        title="t",
        origin="old",
        observed_at="now",
        scope="selected_reference_files",
        state="unknown",
        files=[dict(path="f", sha256="0" * 64, bytes=4, role="source", stability="sealed")],
    )
    with pytest.raises(ValueError):
        import_manifest(store, spec, root=str(tmp_path))
    assert list(store.records("observations")) == []
    spec["files"][0]["path"] = "../f"
    with pytest.raises(ValueError, match="escaping"):
        import_manifest(store, spec, root=str(tmp_path))


def test_evaluation_manifest_does_not_open_samples_and_rejects_pair_changes():
    sample = dict(
        sample_id="a",
        origin="locked-stream",
        physical=dict(path="missing.npz", sha256="1" * 64),
        route_states=dict(path="missing.json", sha256="2" * 64),
    )
    spec = dict(
        schema_version=1,
        scope="oos",
        domain_sha256="3" * 64,
        protocol="L08_PAIRED_FIXED_PLAN_V1",
        random_stream="locked",
        samples=[sample],
    )
    assert validate_manifest(spec) == ["a"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(dict(spec, samples=[sample, sample]))
    with pytest.raises(ValueError, match="scope"):
        validate_manifest(dict(spec, random_stream=""))
