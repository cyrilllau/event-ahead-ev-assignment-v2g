import sqlite3

import pytest

from ev_assignment_experiments.tracking.store import Ref, Store


def test_verified_transitive_sync_dedup_and_offline_rebuild(tmp_path):
    source, local = Store(tmp_path / "server"), Store(tmp_path / "local")
    data = source.put_object(b"sealed evidence")
    run = source.put("runs", {"study": "s", "code": "example"}, [data])
    audit = source.put("audits", {"passed": True}, [run, data])
    assert source.put_object(b"sealed evidence") == data
    first = source.sync_to(local, [audit])
    assert first == source.sync_to(local, [audit])
    assert first["complete"] and first["records"] == 3
    # Local archive remains self-contained after the remote files disappear.
    for path in source.root.rglob("*"):
        if path.is_file():
            path.unlink()
    assert local.closure([audit]) == sorted([run, data, audit])
    local.rebuild_index()
    with sqlite3.connect(local.root / "index.sqlite") as db:
        assert db.execute("select study from records where kind='runs'").fetchone() == ("s",)


def test_corruption_is_not_overwritten_or_silently_trusted(tmp_path):
    source, local = Store(tmp_path / "source"), Store(tmp_path / "local")
    blob = source.put_object(b"correct")
    source.sync_to(local, [blob])
    local.path(blob).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Digest mismatch"):
        source.sync_to(local, [blob])
    assert local.path(blob).read_bytes() == b"corrupt"


def test_missing_reference_cannot_register_a_complete_record(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.put("runs", {}, [Ref("objects", "0" * 64)])
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "kind,identity",
    [("../outside", "a" * 64), ("runs", "../outside"), ("runs", "A" * 64), ("runs", "short")],
)
def test_refs_reject_path_traversal(kind, identity):
    with pytest.raises(ValueError):
        Ref(kind, identity)


def test_symlink_escape_cannot_write_outside_archive(tmp_path):
    store = Store(tmp_path / "store")
    store.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "objects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.put_object(b"payload")
    assert not list(outside.iterdir())


def test_record_order_and_nan_are_not_ambiguous(tmp_path):
    store = Store(tmp_path)
    assert store.put("runs", {"a": 1, "b": 2}) == store.put("runs", {"b": 2, "a": 1})
    with pytest.raises(ValueError):
        store.put("runs", {"objective": float("nan")})


def test_missing_dependency_is_detected_after_sync(tmp_path):
    source, local = Store(tmp_path / "s"), Store(tmp_path / "l")
    obj = source.put_object(b"needed")
    run = source.put("runs", {"study": "s"}, [obj])
    source.sync_to(local, [run])
    local.path(obj).unlink()
    with pytest.raises(FileNotFoundError):
        local.closure([run])
