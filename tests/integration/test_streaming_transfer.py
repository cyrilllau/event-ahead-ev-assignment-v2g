from contextlib import contextmanager
import shlex
import sys

import pytest

from ev_assignment_experiments.tracking.store import Store
from ev_assignment_experiments.tracking.transfer import (
    CHUNK,
    FilesystemSource,
    SSHSource,
    transfer,
)


def source(tmp_path):
    store = Store(tmp_path / "source")
    large = tmp_path / "large"
    with large.open("wb") as handle:
        for i in range(5):
            handle.write(bytes([i]) * CHUNK)
    artifact = store.put_file(large)
    root = store.put("attempts", {"raw": artifact.as_dict()}, [artifact])
    return store, artifact, root


def test_interruption_resumes_verified_prefix_without_overwriting(tmp_path):
    store, artifact, root = source(tmp_path)
    destination = Store(tmp_path / "local")

    class Interrupted(FilesystemSource):
        @contextmanager
        def open(self, ref, offset=0):
            with super().open(ref, offset) as handle:
                if ref != artifact:
                    yield handle
                    return

                class Reader:
                    calls = 0

                    def read(self, count):
                        assert count <= CHUNK
                        self.calls += 1
                        if self.calls > 2:
                            raise OSError("simulated disconnect")
                        return handle.read(count)

                yield Reader()

    with pytest.raises(OSError, match="disconnect"):
        transfer(Interrupted(store), destination, [root])
    assert not destination.path(artifact).exists()
    done = transfer(FilesystemSource(store), destination, [root])
    assert done["complete"] and done["resumed_bytes"] == 2 * CHUNK
    assert destination.closure([root]) == store.closure([root])
    again = transfer(FilesystemSource(store), destination, [root])
    assert again["copied_bytes"] == 0 and again["reused_records"] == 2


def test_complete_corruption_is_never_replaced(tmp_path):
    store, artifact, root = source(tmp_path)
    destination = Store(tmp_path / "local")
    store.sync_to(destination, [root])
    destination.path(artifact).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="refusing overwrite"):
        store.sync_to(destination, [root])
    assert destination.path(artifact).read_bytes() == b"corrupt"


def test_remote_protocol_in_separate_process_with_quoted_paths(tmp_path):
    store, artifact, root = source(tmp_path)

    class Loopback(SSHSource):
        def command(self, action, ref, count):
            # Execute exactly the generated remote command locally; no SSH
            # credentials or remote writes. Tests quoting plus the actual reader.
            command = super().command(action, ref, count)
            argv = shlex.split(command[-1])
            assert argv[0] == "python3"
            return [sys.executable, *argv[1:]]

    weird = tmp_path / "spaces ' quote $(false)"
    store.root.rename(weird)
    remote = Loopback("powerai", str(weird))
    destination = Store(tmp_path / "local")
    receipt = transfer(remote, destination, [root])
    assert receipt["complete"] and destination.path(artifact).stat().st_size == 5 * CHUNK
    assert not (tmp_path / "false").exists()


def test_remote_reader_rejects_symlink_escape(tmp_path):
    store, artifact, _ = source(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(store.path(artifact).read_bytes())
    store.path(artifact).unlink()
    store.path(artifact).symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        transfer(FilesystemSource(store), Store(tmp_path / "local"), [artifact])


def test_modified_partial_is_rejected(tmp_path):
    store, artifact, root = source(tmp_path)
    destination = Store(tmp_path / "local")
    area = destination.root / ".transfers"
    area.mkdir(parents=True)
    partial = area / ("objects-" + artifact.id + ".part")
    partial.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Partial transfer"):
        store.sync_to(destination, [root])
    assert not destination.path(artifact).exists()


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "x;true", "x\nx", "$(false)"])
def test_ssh_host_is_data(host):
    with pytest.raises(ValueError):
        SSHSource(host, "/archive")


def test_remote_watch_reads_only_sealed_namespaces_and_tracks_offline(tmp_path):
    from ev_assignment_experiments.tracking.sync import watch_ssh

    store = Store(tmp_path / "remote")
    raw = store.put_object(b"evidence")
    attempt = store.put("attempts", dict(study="watched", execution="exited"), [raw])

    class Loopback(SSHSource):
        def command(self, action, ref, count):
            argv = shlex.split(super().command(action, ref, count)[-1])
            return [sys.executable, *argv[1:]]

    destination = Store(tmp_path / "local")
    remote = Loopback("powerai", str(store.root))
    result = watch_ssh(remote, destination, "watched", seconds=0.01, interval=0.01)
    assert result["error"] is None and destination.object(raw) == b"evidence"
    assert destination.closure([attempt]) == store.closure([attempt])

    class Offline:
        def inventory(self, *args):
            raise OSError("offline")

    failed = watch_ssh(Offline(), destination, "watched", seconds=0.01, interval=0.01)
    assert "offline" in failed["error"] and destination.object(raw) == b"evidence"
