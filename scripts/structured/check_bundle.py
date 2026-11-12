"""Read-only artifact integrity and stop-gate consistency; zero solver calls."""
from pathlib import Path
import json
import sys
from ev_assignment_experiments.analysis.diagnostic import digest, read
from ev_assignment_experiments.runtime.ledger import BudgetLedger

root = Path(sys.argv[1]).resolve()
manifest = read(root / "ARTIFACT_MANIFEST.json")
for entry in manifest["files"]:
    path = root / entry["path"]
    if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
        raise ValueError(f"Artifact changed: {entry['path']}")
r = read(root / "CASE_DESIGN_RESULTS.json")
b = BudgetLedger(root / "store", "structured-design-development-v1").read()
assert b["active"] is None and b["spent_seconds"] == r["budget"]["development_charged_seconds"]
assert r["status"] == "NO_COMPUTABLE_FAMILY_FOUND_WITHIN_BUDGET"
assert not r["found_admitted_full400_family"] and not r["full400"]["attempted"]
assert r["full400"]["LB"] is r["full400"]["UB"] is r["full400"]["relative_gap"] is None
assert r["stages"]["validation"] == "NOT_OPENED" and r["revision_rounds"] == 1
assert len(r["candidates"]) == 14 and len({c["id"] for c in r["candidates"]}) == 12
assert len(r["A5"]["all_shedding_audits"]) == 130
assert all(x["residual"] <= 1e-9 for x in r["A5"]["all_shedding_audits"])
assert all(x["original_row_residual"] <= 1e-9 for x in r["A5"]["zero_EV_LPs"])
assert r["A1"]["passed"] and not r["A3"]["passed"]
assert all(c["global_LB"] is None and c["global_UB"] is None for c in r["candidates"])
print(json.dumps(dict(artifact_hashes_verified=len(manifest["files"]),
                      status="CLOSED_FAILURE_REPORT_CONSISTENT", solver_calls=0)))
