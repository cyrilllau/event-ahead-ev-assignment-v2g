"""Sequential invocation of the existing ledger wrapper from a frozen job list.

No existing job is adopted or resubmitted. A missing WAIT is accepted only with
the verified, conservatively settled dead-claim recovery chain below. Deploy this
controller separately: the release argument still identifies the frozen science
workers and wrapper, not this controller. Only that wrapper admits compute.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import math
import subprocess
import sys
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.analysis.diagnostic import reference


RECOVERY_SCHEMA = "round2-dead-claim-recovery-v1"
RECOVERY_STATUS = "RESOURCE_INTERRUPTED_OOM"
RECOVERY_REASON = "dead_owner_unknown_overhead_conservatively_charged_full_reservation"


def validate_settled_run(claim, expected, snapshot, *, config_sha256,
                         wait_ref=None, wait=None, recovery=None, recovery_refs=None,
                         observation=None, settlement_event=None, result_exists=False):
    """Pure metadata checks; snapshot must come from BudgetLedger.read().

    Recovery refs are freshly measured claim/progress/evidence/ledger-event refs.
    No liveness probing, ledger writes, result adoption, or scientific evaluation
    occurs here. Recovery proves settlement of a failed run, never a parent wait.
    """
    def require(ok, message):
        if not ok:
            raise RuntimeError("Existing output/recovery rejected: " + message)

    require(claim.get("run") == expected and
            expected["config"]["sha256"] == config_sha256, "frozen job identity/config")
    attempt = claim.get("attempt")
    require(isinstance(attempt, str) and bool(attempt), "missing attempt")
    events = snapshot["events"]
    claims = [(i, e) for i, e in enumerate(events)
              if e["payload"].get("kind") == "claim" and e["payload"].get("attempt") == attempt]
    settlements = [(i, e) for i, e in enumerate(events)
                   if e["payload"].get("kind") == "settlement" and e["payload"].get("attempt") == attempt]
    require(len(claims) == len(settlements) == 1, "missing/duplicate ledger claim or settlement")
    ci, cr = claims[0]
    si, sr = settlements[0]
    require(cr["payload"] == claim and ci < si, "CLAIM differs from original ledger claim")
    settled = sr["payload"]
    if wait_ref is not None:
        require(recovery is None, "WAIT and RECOVERY cannot coexist")
        require(settled.get("actual_parent_wait") is True and
                settled.get("receipt") == wait_ref and isinstance(wait, dict) and
                type(wait.get("returncode")) is int and
                type(settled.get("returncode")) is int and settled["returncode"] == wait["returncode"],
                "matching actual parent WAIT/receipt required")
        return dict(kind="parent_wait", attempt=attempt,
                    charged_seconds=settled["charged_seconds"])

    require(isinstance(recovery, dict), "missing WAIT and recovery chain; do not resubmit")
    require(snapshot.get("active", True) is None, "ledger still has an active/unknown claim")
    require(recovery.get("schema") == RECOVERY_SCHEMA and
            recovery.get("status") == RECOVERY_STATUS and recovery.get("attempt") == attempt,
            "recovery schema/status/attempt")
    require(recovery.get("actual_parent_wait") is False and
            settled.get("actual_parent_wait") is False and settled.get("reason") == RECOVERY_REASON,
            "not an original conservative dead-claim settlement")
    require(all(k not in doc for doc in (settled, recovery) for k in ("receipt", "returncode")),
            "recovery must not fabricate a receipt or returncode")
    charges = [claim.get("reserved_seconds"), settled.get("charged_seconds"),
               recovery.get("charged_seconds")]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                math.isfinite(v) and v > 0 for v in charges) and
            charges[0] == charges[1] == charges[2], "full original reservation must be charged")
    require(isinstance(recovery_refs, dict) and
            set(recovery_refs) == {"claim", "progress", "settlement_event", "evidence"},
            "missing recovery references")
    for key, ref in recovery_refs.items():
        require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"} and
                recovery.get(key) == ref, "unmatched recovery reference: " + key)
    require(settlement_event == sr, "settlement event is not the matching original ledger event")
    children = [(i, e["payload"]) for i, e in enumerate(events)
                if e["payload"].get("kind") == "child" and e["payload"].get("attempt") == attempt]
    require(len(children) == 1 and ci < children[0][0] < si, "missing/ambiguous original child")
    require(isinstance(observation, dict), "missing observation")
    identities = observation.get("identities", {})
    for role, identity in (("owner", claim.get("owner")), ("child", children[0][1].get("identity"))):
        observed = identities.get(role, {})
        require(isinstance(identity, dict) and bool(identity) and isinstance(observed, dict) and
                observed.get("identity") == identity and observed.get("matches") is False,
                "observation must prove the original " + role + " absent")
    unit = observation.get("unit", "")
    require(isinstance(unit, str) and "Result=oom-kill" in unit.split() and "ActiveState=failed" in unit.split()
            and isinstance(observation.get("journal"), str) and bool(observation["journal"].strip()),
            "missing recorded systemd OOM evidence")
    require(observation.get("progress") == recovery_refs["progress"] and
            observation.get("wait_exists") is False and observation.get("result_exists") is False
            and not result_exists and recovery.get("passed", False) is False,
            "observation/progress or missing-WAIT/RESULT identity mismatch")
    require("attempt" not in observation or observation["attempt"] == attempt,
            "observation attempt mismatch")
    return dict(kind="dead_claim_recovery", status=RECOVERY_STATUS, passed=False,
                attempt=attempt, charged_seconds=charges[1], actual_parent_wait=False)


def _json_ref(path):
    """Bind parsed metadata and its reference to the same bytes."""
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check_existing_run(root, release, job, ledger):
    """Read only execution/recovery metadata; never read a scientific result."""
    dest = root / job["stage"] / job["id"]
    python = release / ".venv/bin/python"
    expected = dict(stage=job["stage"], run_id=job["id"], source=reference(release / job["worker"]),
        argv=[str(python), str(release / job["worker"]), *job["args"]],
        release=reference(release / "RELEASE.json"), freeze=reference(root / "FREEZE.json"),
        wrapper=reference(release / "stage.py"), config=reference(dest / "EXECUTION_CONFIG.json"))
    claim, claim_ref = _json_ref(dest / "CLAIM.json")
    snapshot = ledger.read()  # Verifies the complete existing hash chain; never settles it.
    kwargs = dict(config_sha256=reference(root / "DESIGN_CONFIG.json")["sha256"])
    recovery_path = dest / "RECOVERY.json"
    if (dest / "WAIT.json").exists():
        kwargs["wait"], kwargs["wait_ref"] = _json_ref(dest / "WAIT.json")
        if recovery_path.exists():
            raise RuntimeError(f"WAIT and RECOVERY cannot coexist: {dest}")
    elif recovery_path.exists():
        recovery, recovery_ref = _json_ref(recovery_path)
        matches = [(i, e) for i, e in enumerate(snapshot["events"])
                   if e["payload"].get("kind") == "settlement" and
                   e["payload"].get("attempt") == claim["attempt"]]
        if len(matches) != 1:
            raise RuntimeError(f"Recovery lacks unique original ledger settlement: {dest}")
        event, event_ref = _json_ref(ledger.root / "events" / f"{matches[0][0]:08d}.json")
        observation, evidence_ref = _json_ref(recovery["evidence"]["path"])
        refs = dict(claim=claim_ref, progress=reference(dest / "PROGRESS.json"),
                    evidence=evidence_ref, settlement_event=event_ref)
        kwargs.update(recovery=recovery, recovery_refs=refs, observation=observation,
                      settlement_event=event, result_exists=(dest / "RESULT.json").exists())
    outcome = validate_settled_run(claim, expected, snapshot, **kwargs)
    if outcome["kind"] == "dead_claim_recovery":
        outcome.update(recovery=recovery_ref, settlement_event=event_ref)
    return outcome


def main():
    root, release = Path(sys.argv[1]), Path(sys.argv[2])
    freeze = json.loads((root / "FREEZE.json").read_text())
    config = json.loads((root / "DESIGN_CONFIG.json").read_text())
    ledger = BudgetLedger(config["ledger_store"], config["budget_group"])
    if ledger.read()["active"] is not None:
        raise RuntimeError("Campaign restart requires an already settled ledger; no PID adoption")
    python = release / ".venv/bin/python"
    for job in freeze["jobs"]:
        dest = root / job["stage"] / job["id"]
        if not dest.exists():
            if ledger.read()["active"] is not None:
                raise RuntimeError("Active ledger claim; do not submit another frozen job")
            command = [str(python), str(release / "stage.py"), str(root), job["stage"], job["id"],
                       str(release / job["worker"]), str(python), *job["args"]]
            with (root / "CAMPAIGN.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        settlement = check_existing_run(root, release, job, ledger)
        path = dest / "RESULT.json"
        recovered = settlement["kind"] == "dead_claim_recovery"
        result = dict(status=RECOVERY_STATUS, passed=False) if recovered else (
            json.loads(path.read_text()) if path.exists() else {})
        event = dict(time=datetime.now(timezone.utc).isoformat(), id=job["id"], stage=job["stage"],
                     status=result.get("status", "NO_RESULT"), passed=result.get("passed", False),
                     settlement=settlement, controller=reference(__file__))
        with (root / "CAMPAIGN_PROGRESS.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        if job.get("required_preparation") and not result.get("passed"):
            raise RuntimeError("Frozen preparation failed; stop without redesigning validation")
        if result.get("status") in {"IMPLEMENTATION_ERROR", "IMPLEMENTATION_OR_IDENTITY_ERROR",
                                   "INVALID_CERTIFICATE", "INCOMPATIBLE_CERTIFIED_INTERVAL"}:
            raise RuntimeError(f"Implementation/certificate review required before dependent jobs: {dest}")
    snapshot = ledger.read()
    if snapshot["active"] is not None:
        raise RuntimeError("Campaign cannot complete with an active ledger claim")
    (root / "CAMPAIGN_COMPLETE.json").write_text(json.dumps(dict(
        status="ALL_FROZEN_JOBS_SETTLED_REVIEW_ACCEPTANCE_SEPARATELY",
        ledger_head=snapshot["previous"], controller=reference(__file__),
        completed_utc=datetime.now(timezone.utc).isoformat())))


if __name__ == "__main__":
    main()
