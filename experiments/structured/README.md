# Structured fleet design, 2026-11-12

This is a separate synthetic business domain for the existing L08 physical kernel.
It does not replace the frozen historical problem, production releases or records.
The source plan is `design-v1.json`; the sole permitted development revision is
`revision-1.json`. External inputs require their own hash manifest. Never place
input data, reports, result JSON, wheels, logs or licenses in this checkout.

## Observed status

The round stopped at the A3 **design suitability** screen, after A0/A1 and the
revised A2 finite-policy screen passed. The load-scaling rule caused substantial
healthy-grid congestion. A5 explicitly established that the original LP remains
feasible with load shedding. Failure of a connected-full-supply constructive
witness is **not** a proof of original-model infeasibility or nonconvergence.

No A4/full400 optimization, frozen validation or OOS evaluation was performed.
The count/sparse-Benders adapter is implemented but was not exercised by A3:
the design screen returned before invoking it. Do not claim it is end-to-end
validated. Tiny integer equivalence and fixed-policy recourse were validated.
The closed round is not eligible for an A4-only fallback or another revision.

Local evidence root:

```text
/Users/shixinliu/Desktop/Research/with Lusha/event-ahead-ev-assignment-v2g-store/validation/structured-fleet-design-20261112
```

Remote evidence root:

```text
/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112
```

Read `CASE_DESIGN_STATUS.md` there first. `CASE_DESIGN_RESULTS.json` retains the
failed attempts and missing results as null. `FROZEN_DEVELOPMENT_MANIFEST.json`
identifies failed development snapshots, **not an admitted validation family**.

## Read-only verification

From the installed development package in this checkout:

```sh
.venv/bin/python scripts/structured/check_bundle.py \
  '/Users/shixinliu/Desktop/Research/with Lusha/event-ahead-ev-assignment-v2g-store/validation/structured-fleet-design-20261112'
```

Software regression uses `python -m pytest tests/unit tests/integration`. Some
legacy portable fixture tests solve small HiGHS LPs; they are software tests,
not real-source experimental runs or evidence of full-scale convergence.

## Recorded execution interfaces

These are replay interfaces, **not instructions to resubmit the closed round**.
The per-Run `CLAIM.json`, `CHILD.json`, `WAIT.json`, log, immutable release wheel
and source hashes bind actual attempts. A new authorized reproduction requires
a fresh explicit research store and budget while retaining this round's history.

```sh
python scripts/structured/a0.py EXTERNAL_ROOT
python scripts/structured/stage.py EXTERNAL_ROOT A1 RUN_ID A1_WORKER PYTHON
python scripts/structured/stage.py EXTERNAL_ROOT A1 REPAIR_RUN A1_WORKER PYTHON PREVIOUS_A1_OUT
python scripts/structured/stage.py EXTERNAL_ROOT A2 RUN_ID A2_WORKER PYTHON INPUTS
python scripts/structured/stage.py EXTERNAL_ROOT A2 REVISION_RUN A2_WORKER PYTHON INPUTS 1
python scripts/structured/stage.py EXTERNAL_ROOT A3 RUN_ID OPT_WORKER PYTHON INPUTS A2_RESULT A3
python scripts/structured/stage.py EXTERNAL_ROOT A5 RUN_ID A5_WORKER PYTHON A3_RUN_DIRECTORY
```

Only the wrapper admits computation. It uses the common cumulative `BudgetLedger`,
charges process startup/build/audit/failure/parent wait, and refuses overlapping
claims or a stage whose cumulative allocation is exhausted. Limits are A0 600,
A1 1200, A2 1200, A3 1800, A4 1800, A5 600 seconds and overall 7200 seconds.
The observed out-of-wrapper generator preflight was conservatively debited using
`book_preflight.py`; its 120 seconds include a separately identified reserve.
The added `charge_external` API makes this cost persistent and prevents duplication.

Actual jobs used separate systemd user units, CPUQuota=400%, MemoryMax=24G,
MemorySwapMax=0, Nice=10, Gurobi 12.0.3, Python 3.12.3, single-thread recourse,
and OMP/OPENBLAS/MKL thread limits of one. No production process was adopted.

`report.py EXTERNAL_ROOT` assembles the stopped round's two final deliverables
from existing receipts without solver calls; it writes them once. It is specific
to this observed failure path, not a generic success-report generator.

## Scientific boundaries

- The original full task domain → finite package catalogue is a business change.
  Integer count aggregation inside the same catalogue is a separate exact claim.
- Equal-share canonical expansion fixes contracts across scenarios. Unequal
  individual-contract policies cannot share Q/cuts by totals alone.
- Continuous evaluations are never integer upper bounds. Complete-support
  probability dual upper bounds and original per-vehicle audits are required.
- Sparse cuts/probability-row certificates reuse the existing core primitives.
  Changed probability bounds/support order have different scientific identities.
- Generator probabilities retain only three registered coordinates and use new
  CAL2000 Wilson intervals. This is not the old 322,311-row ambiguity set.
- The next research improvement should constrain affected-area load design by
  feeder capacities before scaling the fleet; do not hide healthy-grid ENS by
  changing line limits or treating failed design checks as solver nonconvergence.
