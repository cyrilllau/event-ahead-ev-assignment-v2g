"""One-time conservative accounting for observed local generator preflight."""

from pathlib import Path
import sys
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.analysis.diagnostic import read, reference, save

root = Path(sys.argv[1]).resolve()
receipt = root / "LOCAL_GENERATOR_PREFLIGHT_ACCOUNTING.json"
evidence = dict(kind="reported_local_real_source_generator_checks",
                solver_calls=0, measured_generation_seconds=11.507,
                unmeasured_startup_reserve_seconds=108.493,
                conservative_total_seconds=120.0,
                actual_parent_wait=False, accounting_stage="A2",
                source="generator subagent completion and follow-up audit, 2026-11-12",
                validation_or_oos_read=False,
                note="Observed deviation from intended remote-only build accounting; "
                     "no fabricated solver results or wait time. Reserve is conservative, not measured wall time.")
if receipt.exists():
    if read(receipt) != evidence:
        raise ValueError("Previous preflight receipt differs")
else:
    save(receipt, evidence)
ledger = BudgetLedger(root / "store", "structured-design-development-v1")
ledger.charge_external("generator-local-preflight-20261112", dict(stage="A2"), 120,
                       reference(receipt))
print(ledger.read()["spent_seconds"])
