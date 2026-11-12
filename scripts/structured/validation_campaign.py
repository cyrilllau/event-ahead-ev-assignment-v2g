"""Sequential invocation of the existing ledger wrapper from a frozen job list.

No job is adopted or resubmitted when its directory exists without WAIT.json.
Restarting the campaign reuses completed runs, including failed and timed-out
ones, instead of resetting budgets. Only the existing wrapper admits compute.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.analysis.diagnostic import reference


def main():
    root, release = Path(sys.argv[1]), Path(sys.argv[2])
    freeze = json.loads((root / "FREEZE.json").read_text())
    config = json.loads((root / "DESIGN_CONFIG.json").read_text())
    ledger = BudgetLedger(config["ledger_store"], config["budget_group"])
    python = release / ".venv/bin/python"
    for job in freeze["jobs"]:
        dest = root / job["stage"] / job["id"]
        if dest.exists():
            if not (dest / "WAIT.json").exists():
                raise RuntimeError(f"Unsettled existing run: inspect ledger and child identity, do not resubmit {dest}")
            claim = json.loads((dest / "CLAIM.json").read_text())
            snapshot = ledger.read()
            settlements = [e["payload"] for e in snapshot["events"] if e["payload"]["kind"] == "settlement"
                           and e["payload"]["attempt"] == claim["attempt"]]
            expected = dict(stage=job["stage"], run_id=job["id"], source=reference(release / job["worker"]),
                argv=[str(python), str(release / job["worker"]), *job["args"]],
                release=reference(release / "RELEASE.json"), freeze=reference(root / "FREEZE.json"),
                wrapper=reference(release / "stage.py"), config=reference(dest / "EXECUTION_CONFIG.json"))
            if (claim["run"] != expected or len(settlements) != 1
                    or expected["config"]["sha256"] != reference(root / "DESIGN_CONFIG.json")["sha256"]
                    or settlements[0]["receipt"] != reference(dest / "WAIT.json")):
                raise RuntimeError(f"Existing output lacks matching frozen job identity and ledger settlement: {dest}")
        else:
            command = [str(python), str(release / "stage.py"), str(root), job["stage"], job["id"],
                       str(release / job["worker"]), str(python), *job["args"]]
            with (root / "CAMPAIGN.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        path = dest / "RESULT.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        event = dict(time=datetime.now(timezone.utc).isoformat(), id=job["id"], stage=job["stage"],
                     status=result.get("status", "NO_RESULT"), passed=result.get("passed", False))
        with (root / "CAMPAIGN_PROGRESS.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        if job.get("required_preparation") and not result.get("passed"):
            raise RuntimeError("Frozen preparation failed; stop without redesigning validation")
        if result.get("status") in {"IMPLEMENTATION_ERROR", "IMPLEMENTATION_OR_IDENTITY_ERROR",
                                   "INVALID_CERTIFICATE", "INCOMPATIBLE_CERTIFIED_INTERVAL"}:
            raise RuntimeError(f"Implementation/certificate review required before dependent jobs: {dest}")
    (root / "CAMPAIGN_COMPLETE.json").write_text(json.dumps(dict(
        status="ALL_FROZEN_JOBS_SETTLED_REVIEW_ACCEPTANCE_SEPARATELY", completed_utc=datetime.now(timezone.utc).isoformat())))


if __name__ == "__main__":
    main()
