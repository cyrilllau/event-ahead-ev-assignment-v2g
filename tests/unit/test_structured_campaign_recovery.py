"""The 30 campaign recovery regressions: artificial metadata and in-memory IO only.

No research store, live ledger, process identity, worker, or solver is consulted.
Ledger snapshots stand for already chain-checked BudgetLedger.read() output;
the ledger's hash-chain implementation has its own test_runtime_ledger coverage.
"""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def prohibit_compute_and_live_ledger(monkeypatch):
    import scipy.optimize
    from ev_assignment_experiments.runtime import identity, ledger

    def forbidden(*args, **kwargs):
        pytest.fail("Recovery fixtures must not access a live ledger/process or execute compute")

    monkeypatch.setitem(sys.modules, "gurobipy", None)
    monkeypatch.setattr(scipy.optimize, "linprog", forbidden)
    monkeypatch.setattr(scipy.optimize, "milp", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(ledger, "BudgetLedger", forbidden)
    monkeypatch.setattr(identity, "matches", forbidden)
    monkeypatch.setattr(identity, "current", forbidden)


@pytest.fixture
def campaign(pytestconfig, prohibit_compute_and_live_ledger):
    api = runpy.run_path(str(pytestconfig.rootpath / "scripts/structured/validation_campaign.py"))
    # runpy returns a copy. Patch the functions' actual globals for main() fixtures.
    return api["main"].__globals__


def ref(name):
    return dict(path="/fixture/" + name, bytes=10, sha256="a" * 64)


@pytest.fixture
def recovery_case(campaign):
    expected = dict(
        stage="V0", run_id="interrupted", config=ref("EXECUTION_CONFIG.json"),
        source=ref("worker.py"), release=ref("RELEASE.json"), freeze=ref("FREEZE.json"),
        wrapper=ref("stage.py"), argv=["/frozen/python", "/frozen/worker.py", "1101"],
    )
    owner = dict(host="fixture", boot=1, pid=101, created=11, argv_sha256="owner")
    child = dict(host="fixture", boot=1, pid=102, created=12, argv_sha256="child")
    claim = dict(kind="claim", attempt="attempt-a", run=deepcopy(expected),
                 reserved_seconds=1000, owner=owner)
    settlement = dict(kind="settlement", attempt="attempt-a", charged_seconds=1000,
                      actual_parent_wait=False, reason=campaign["RECOVERY_REASON"], observed_epoch=30)
    events = [dict(payload=claim),
              dict(payload=dict(kind="child", attempt="attempt-a", identity=child)),
              dict(payload=settlement)]
    refs = {key: ref(name) for key, name in (
        ("claim", "CLAIM.json"), ("progress", "PROGRESS.json"),
        ("settlement_event", "events/00000002.json"), ("evidence", "OBSERVATION.json"),
    )}
    recovery = dict(schema=campaign["RECOVERY_SCHEMA"], status=campaign["RECOVERY_STATUS"],
                    attempt="attempt-a", actual_parent_wait=False, charged_seconds=1000,
                    **deepcopy(refs))
    observation = dict(
        identities=dict(owner=dict(identity=deepcopy(owner), matches=False),
                        child=dict(identity=deepcopy(child), matches=False)),
        unit="ActiveState=failed\nResult=oom-kill\n", journal="fixture systemd-oomd record",
        progress=deepcopy(refs["progress"]), wait_exists=False, result_exists=False,
    )
    return dict(claim=deepcopy(claim), expected=expected,
                snapshot=dict(events=events, active=None), config_sha256="a" * 64,
                recovery=recovery, recovery_refs=refs, observation=observation,
                settlement_event=deepcopy(events[-1]))


@pytest.fixture
def normal_wait_case(recovery_case):
    case = deepcopy(recovery_case)
    for key in ("recovery", "recovery_refs", "observation", "settlement_event"):
        case.pop(key)
    case.update(wait_ref=ref("WAIT.json"), wait=dict(returncode=0, elapsed_seconds=12))
    case["snapshot"]["events"][-1]["payload"].update(
        actual_parent_wait=True, returncode=0, receipt=ref("WAIT.json"), charged_seconds=12,
    )
    case["snapshot"]["events"][-1]["payload"].pop("reason")
    return case


def reject(campaign, case):
    with pytest.raises(RuntimeError):
        campaign["validate_settled_run"](**case)


def test_normal_wait(campaign, normal_wait_case):
    outcome = campaign["validate_settled_run"](**normal_wait_case)
    assert outcome["kind"] == "parent_wait"
    assert outcome["charged_seconds"] == 12


def test_normal_failed_wait_still_settled(campaign, normal_wait_case):
    normal_wait_case["wait"]["returncode"] = 124
    normal_wait_case["snapshot"]["events"][-1]["payload"]["returncode"] = 124
    assert campaign["validate_settled_run"](**normal_wait_case)["kind"] == "parent_wait"


def test_recovery_is_failed_and_full_charge(campaign, recovery_case):
    before = deepcopy(recovery_case)
    outcome = campaign["validate_settled_run"](**recovery_case)
    assert outcome == dict(kind="dead_claim_recovery", status=campaign["RECOVERY_STATUS"],
                           passed=False, attempt="attempt-a", charged_seconds=1000,
                           actual_parent_wait=False)
    assert recovery_case == before


def test_missing_chain(campaign, recovery_case):
    for key in ("recovery", "recovery_refs", "observation", "settlement_event"):
        case = deepcopy(recovery_case)
        case.pop(key)
        reject(campaign, case)


def test_missing_refs(campaign, recovery_case):
    for key in ("claim", "progress", "evidence", "settlement_event"):
        case = deepcopy(recovery_case)
        case["recovery"].pop(key)
        reject(campaign, case)


def test_tampered_refs(campaign, recovery_case):
    for key in ("claim", "progress", "evidence", "settlement_event"):
        case = deepcopy(recovery_case)
        case["recovery"][key]["sha256"] = "b" * 64
        reject(campaign, case)


def test_wrong_attempt(campaign, recovery_case):
    for target in ("claim", "recovery", "settlement_event"):
        case = deepcopy(recovery_case)
        doc = case[target]["payload"] if target == "settlement_event" else case[target]
        doc["attempt"] = "other"
        reject(campaign, case)


def test_active_and_unknown_ledger(campaign, recovery_case):
    for active in ({"attempt": "attempt-a"}, {"attempt": "unrelated"}, True):
        case = deepcopy(recovery_case)
        case["snapshot"]["active"] = active
        reject(campaign, case)
    case = deepcopy(recovery_case)
    del case["snapshot"]["active"]
    reject(campaign, case)


def test_no_ledger_claim_or_settlement(campaign, recovery_case):
    for index in (0, 2):
        case = deepcopy(recovery_case)
        case["snapshot"]["events"].pop(index)
        reject(campaign, case)


def test_duplicate_settlement(campaign, recovery_case):
    recovery_case["snapshot"]["events"].append(deepcopy(recovery_case["settlement_event"]))
    reject(campaign, recovery_case)


def test_detached_settlement(campaign, recovery_case):
    recovery_case["settlement_event"]["payload"]["observed_epoch"] = 40
    reject(campaign, recovery_case)


def test_forged_receipt_or_returncode(campaign, recovery_case):
    for key in ("receipt", "returncode"):
        for target in ("recovery", "settlement"):
            case = deepcopy(recovery_case)
            if target == "recovery":
                case["recovery"][key] = None
            else:
                case["snapshot"]["events"][-1]["payload"][key] = None
                case["settlement_event"]["payload"][key] = None
            reject(campaign, case)


def test_fake_wait_cannot_launder_recovery(campaign, normal_wait_case, recovery_case):
    normal_wait_case["snapshot"]["events"][-1]["payload"]["actual_parent_wait"] = False
    reject(campaign, normal_wait_case)
    recovery_case.update(wait_ref=ref("WAIT.json"), wait=dict(returncode=0))
    reject(campaign, recovery_case)


def test_normal_wait_receipt_and_rc(campaign, normal_wait_case):
    case = deepcopy(normal_wait_case)
    case["wait_ref"]["sha256"] = "b" * 64
    reject(campaign, case)
    case = deepcopy(normal_wait_case)
    case["wait"]["returncode"] = 1
    reject(campaign, case)


def test_wrong_frozen_job_or_config(campaign, recovery_case):
    for key, value in (("argv", ["different"]), ("release", ref("different-release")),
                       ("freeze", ref("different-freeze"))):
        case = deepcopy(recovery_case)
        case["expected"][key] = value
        reject(campaign, case)
    case = deepcopy(recovery_case)
    case["config_sha256"] = "b" * 64
    reject(campaign, case)


def test_full_claim_not_just_run(campaign, recovery_case):
    recovery_case["claim"]["owner"]["created"] = 99
    reject(campaign, recovery_case)


def test_undercharged_or_nonfinite(campaign, recovery_case):
    for value in (999, 0, True, float("nan"), float("inf")):
        case = deepcopy(recovery_case)
        case["recovery"]["charged_seconds"] = value
        reject(campaign, case)
    case = deepcopy(recovery_case)
    case["claim"]["reserved_seconds"] = 900
    reject(campaign, case)


def test_live_unknown_or_wrong_identity(campaign, recovery_case):
    for role in ("owner", "child"):
        for value in (True, None, 0, "false"):
            case = deepcopy(recovery_case)
            case["observation"]["identities"][role]["matches"] = value
            reject(campaign, case)
        case = deepcopy(recovery_case)
        case["observation"]["identities"][role]["identity"]["created"] = 99
        reject(campaign, case)


def test_child_and_order_required(campaign, recovery_case):
    case = deepcopy(recovery_case)
    case["snapshot"]["events"].pop(1)
    reject(campaign, case)
    case = deepcopy(recovery_case)
    events = case["snapshot"]["events"]
    events[0], events[2] = events[2], events[0]
    reject(campaign, case)


def test_oom_evidence(campaign, recovery_case):
    for value in ("ActiveState=failed Result=exit-code", "NotActiveState=failed Result=oom-kill", None):
        case = deepcopy(recovery_case)
        case["observation"]["unit"] = value
        reject(campaign, case)
    case = deepcopy(recovery_case)
    case["observation"]["journal"] = ""
    reject(campaign, case)


def test_observation_progress_and_missing_outputs(campaign, recovery_case):
    case = deepcopy(recovery_case)
    case["observation"]["progress"]["sha256"] = "b" * 64
    reject(campaign, case)
    for field in ("wait_exists", "result_exists"):
        case = deepcopy(recovery_case)
        case["observation"][field] = True
        reject(campaign, case)
    case = deepcopy(recovery_case)
    case["result_exists"] = True
    reject(campaign, case)
    case = deepcopy(recovery_case)
    case["recovery"]["passed"] = True
    reject(campaign, case)


def test_schema_status_and_reason(campaign, recovery_case):
    for key in ("schema", "status"):
        case = deepcopy(recovery_case)
        case["recovery"][key] = "wrong"
        reject(campaign, case)
    case = deepcopy(recovery_case)
    case["snapshot"]["events"][-1]["payload"]["reason"] = "manual"
    reject(campaign, case)


@pytest.fixture
def run_campaign(campaign, monkeypatch):
    def run(*, remaining=False, active=None, final_active=None, invalid=False):
        jobs = [dict(stage="V0", id="recovered", worker="validation_worker.py", args=["1101"])]
        if remaining:
            jobs.append(dict(stage="V0", id="remaining", worker="validation_worker.py", args=["2202"]))
        launched, writes = [], []

        class Log(io.StringIO):
            def close(self):
                pass  # Retain fake progress after the controller's with block.

        progress = Log()

        def exists(path):
            if str(path) == "/fixture/root/V0/recovered":
                return True
            if str(path) == "/fixture/root/V0/remaining":
                return bool(launched)
            if path.name == "RESULT.json":
                return False
            pytest.fail("Unexpected filesystem probe: " + str(path))

        def read_text(path, *args, **kwargs):
            if path.name == "FREEZE.json":
                return json.dumps(dict(jobs=jobs))
            if path.name == "DESIGN_CONFIG.json":
                return json.dumps(dict(ledger_store="/fixture/store", budget_group="g"))
            pytest.fail("Scientific result or unexpected read: " + str(path))

        def opened(path, *args, **kwargs):
            if path.name == "CAMPAIGN_PROGRESS.jsonl":
                return progress
            if path.name == "CAMPAIGN.log":
                return Log()
            pytest.fail("Unexpected write: " + str(path))

        def checked(root, release, job, ledger):
            if invalid:
                raise RuntimeError("fixture missing recovery chain")
            if job["id"] == "recovered":
                return dict(kind="dead_claim_recovery", status=campaign["RECOVERY_STATUS"],
                            passed=False, charged_seconds=1000)
            return dict(kind="parent_wait", charged_seconds=12)

        ledger = Mock()
        snapshots = [dict(active=active, previous="head")]
        if remaining:
            snapshots.append(dict(active=None, previous="head"))
        snapshots.append(dict(active=final_active, previous="settled-head"))
        ledger.read.side_effect = snapshots
        verifier = Mock(side_effect=checked)
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["controller-new.py", "/fixture/root", "/fixture/frozen-r08"])
            patch.setitem(campaign, "BudgetLedger", Mock(return_value=ledger))
            patch.setitem(campaign, "reference", lambda path: ref(str(path)))
            patch.setitem(campaign, "check_existing_run", verifier)
            patch.setattr(Path, "exists", exists)
            patch.setattr(Path, "read_text", read_text)
            patch.setattr(Path, "open", opened)
            patch.setattr(Path, "write_text", lambda p, s: writes.append((str(p), json.loads(s))))
            patch.setattr(subprocess, "run", lambda command, **kwargs: launched.append(command))
            error = None
            try:
                campaign["main"]()
            except RuntimeError as exc:
                error = exc
        events = [json.loads(line) for line in progress.getvalue().splitlines()]
        return error, launched, writes, events, verifier.call_count

    return run


def test_resume_skips_recovery_and_only_submits_remaining_frozen_job(campaign, run_campaign):
    error, launched, writes, events, checked = run_campaign(remaining=True)
    assert error is None
    assert launched == [["/fixture/frozen-r08/.venv/bin/python", "/fixture/frozen-r08/stage.py",
                         "/fixture/root", "V0", "remaining", "/fixture/frozen-r08/validation_worker.py",
                         "/fixture/frozen-r08/.venv/bin/python", "2202"]]
    assert checked == 2
    assert events[0]["status"] == campaign["RECOVERY_STATUS"]
    assert events[0]["passed"] is False
    assert writes[0][1]["status"] == "ALL_FROZEN_JOBS_SETTLED_REVIEW_ACCEPTANCE_SEPARATELY"


def test_active_restart_submits_nothing_and_never_completes(run_campaign):
    error, launched, writes, events, checked = run_campaign(active={"attempt": "still-active"})
    assert error is not None
    assert (launched, writes, events, checked) == ([], [], [], 0)


def test_incomplete_chain_never_completes_or_resubmits(run_campaign):
    error, launched, writes, events, _ = run_campaign(invalid=True)
    assert error is not None
    assert (launched, writes, events) == ([], [], [])


def test_final_active_cannot_publish_completion(run_campaign):
    error, launched, writes, _, _ = run_campaign(final_active={"attempt": "active"})
    assert error is not None
    assert (launched, writes) == ([], [])


@pytest.fixture
def memory_artifacts(campaign):
    """Fresh byte-bound execution metadata; paths are keys, never real files."""
    def build(recovery=True):
        root, release = Path("/fixture/root"), Path("/fixture/frozen-r08")
        job = dict(stage="V0", id="interrupted", worker="worker.py", args=["1101"])
        dest = root / "V0/interrupted"
        files = {}

        def put(path, value):
            files[str(path)] = json.dumps(value, sort_keys=True).encode()

        def reference(path):
            data = files[str(path)]
            return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

        for path in (release / "worker.py", release / "RELEASE.json", release / "stage.py",
                     root / "FREEZE.json"):
            put(path, dict(fixture=str(path)))
        put(root / "DESIGN_CONFIG.json", dict(fixture="fixed config"))
        files[str(dest / "EXECUTION_CONFIG.json")] = files[str(root / "DESIGN_CONFIG.json")]
        expected = dict(
            stage=job["stage"], run_id=job["id"], source=reference(release / "worker.py"),
            argv=[str(release / ".venv/bin/python"), str(release / "worker.py"), "1101"],
            release=reference(release / "RELEASE.json"), freeze=reference(root / "FREEZE.json"),
            wrapper=reference(release / "stage.py"), config=reference(dest / "EXECUTION_CONFIG.json"),
        )
        owner = dict(host="fixture", boot=1, pid=12, created=2, argv_sha256="owner")
        child = dict(host="fixture", boot=1, pid=13, created=3, argv_sha256="child")
        claim = dict(kind="claim", attempt="a", run=expected, owner=owner, reserved_seconds=1000)
        put(dest / "CLAIM.json", claim)
        settle = dict(kind="settlement", attempt="a", charged_seconds=1000,
                      actual_parent_wait=False, reason=campaign["RECOVERY_REASON"])
        if not recovery:
            put(dest / "WAIT.json", dict(returncode=0, elapsed_seconds=10))
            settle = dict(kind="settlement", attempt="a", charged_seconds=10, actual_parent_wait=True,
                          receipt=reference(dest / "WAIT.json"), returncode=0)
        events = [dict(payload=claim), dict(payload=dict(kind="child", attempt="a", identity=child)),
                  dict(payload=settle)]
        ledger = Mock(root=Path("/fixture/ledger"))
        ledger.read.return_value = dict(events=deepcopy(events), active=None)
        event_path = ledger.root / "events/00000002.json"
        put(event_path, events[-1])
        if recovery:
            put(dest / "PROGRESS.json", dict(fixture="opaque progress, not parsed"))
            observation = dict(
                identities=dict(owner=dict(identity=owner, matches=False),
                                child=dict(identity=child, matches=False)),
                unit="Result=oom-kill\nActiveState=failed", journal="fixture journal",
                progress=reference(dest / "PROGRESS.json"), wait_exists=False, result_exists=False,
            )
            put(dest / "OBSERVATION.json", observation)
            put(dest / "RECOVERY.json", dict(
                schema=campaign["RECOVERY_SCHEMA"], status=campaign["RECOVERY_STATUS"],
                attempt="a", actual_parent_wait=False, charged_seconds=1000,
                claim=reference(dest / "CLAIM.json"), progress=reference(dest / "PROGRESS.json"),
                settlement_event=reference(event_path), evidence=reference(dest / "OBSERVATION.json"),
            ))
        return dict(root=root, release=release, job=job, ledger=ledger, files=files,
                    reference=reference, dest=dest)

    return build


@pytest.fixture
def check_memory_artifacts(campaign, monkeypatch):
    def check(artifacts):
        files = artifacts["files"]
        with monkeypatch.context() as patch:
            patch.setitem(campaign, "reference", artifacts["reference"])
            patch.setattr(Path, "read_bytes", lambda path: files[str(path)])
            patch.setattr(Path, "exists", lambda path: str(path) in files)
            return campaign["check_existing_run"](
                artifacts["root"], artifacts["release"], artifacts["job"], artifacts["ledger"],
            )

    return check


def test_reads_bound_recovery_metadata_without_results(campaign, memory_artifacts,
                                                      check_memory_artifacts):
    answer = check_memory_artifacts(memory_artifacts())
    assert answer["status"] == campaign["RECOVERY_STATUS"]
    assert answer["passed"] is False
    assert answer["settlement_event"]["path"] == "/fixture/ledger/events/00000002.json"


def test_reads_normal_wait_without_results(memory_artifacts, check_memory_artifacts):
    assert check_memory_artifacts(memory_artifacts(False))["kind"] == "parent_wait"


def test_changed_progress_bytes_rejected(memory_artifacts, check_memory_artifacts):
    artifacts = memory_artifacts()
    artifacts["files"][str(artifacts["dest"] / "PROGRESS.json")] = b'{"changed": true}'
    with pytest.raises(RuntimeError, match="reference: progress"):
        check_memory_artifacts(artifacts)


def test_detached_event_copy_rejected(memory_artifacts, check_memory_artifacts):
    artifacts = memory_artifacts()
    recovery_path = str(artifacts["dest"] / "RECOVERY.json")
    doc = json.loads(artifacts["files"][recovery_path])
    doc["settlement_event"]["path"] = "/fixture/copied-event.json"
    artifacts["files"][recovery_path] = json.dumps(doc).encode()
    with pytest.raises(RuntimeError, match="reference: settlement_event"):
        check_memory_artifacts(artifacts)
