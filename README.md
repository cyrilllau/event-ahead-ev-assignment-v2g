# Event-Ahead EV Assignment and V2G Reserve Procurement

Research software for event-ahead electric-delivery-fleet task assignment and V2G
reserve procurement under coupled grid/road disruptions (IEEE33 v54-Q1).

**Architecture v2 is being implemented in stages.** Read the
[implementation status](docs/architecture/IMPLEMENTATION_STATUS.md) and
[approved complete plan](docs/architecture/REFACTOR_PLAN_V2_CN.md).
The development probability workflow is executable. Formal L08 DET/DRO/ARO
execution, automatic remote synchronization and paper publication are not yet
admitted in v2. Existing PowerAI experiments use their original releases.

## Install

Use Python 3.12. Numerical dependency versions are preserved.

```bash
uv sync --locked --group dev
```

For pip, create/activate a Python 3.12 environment, then:

```bash
python -m pip install -r requirements.lock -e .
```

Gurobi is optional (`uv sync --extra gurobi`) and needs a separately configured
license. Importing the new packages does not initialize it.

## Executable development workflow

Choose an archive outside the code checkout. This example is a two-support
probability LP with a hand-computed optimum of 5 USD, not a full planning run.

```bash
ev-assignment --store /tmp/ev-assignment-example plan --experiment experiments/examples/probability-oracle.toml
ev-assignment --store /tmp/ev-assignment-example run --id RUN_ID
ev-assignment --store /tmp/ev-assignment-example audit --id ATTEMPT_ID
ev-assignment --store /tmp/ev-assignment-example context --study development-probability
ev-assignment --store /tmp/ev-assignment-example sync --attempt ATTEMPT_ID --destination /tmp/ev-assignment-local-copy
ev-assignment --store /tmp/ev-assignment-local-copy bundle --attempt ATTEMPT_ID --destination /tmp/ev-assignment-bundle
ev-assignment --store /tmp/ev-assignment-local-copy verify-bundle --destination /tmp/ev-assignment-bundle
```

Replace RUN_ID / ATTEMPT_ID with the IDs printed by the preceding commands.
Planning seals the actual source snapshot and environment. Source changes require
a new Run. Re-executing a claimed Run is refused: budget-aware recovery belongs
to the runtime stage. Context labels local snapshots and unsettled claims; it
does not infer server liveness or scientific adoption.
Sync streams sealed closures with bounded memory. Interrupted transfers retain a
partial file, verify its prefix against the source, and resume at that offset.
Existing complete artifacts are never overwritten. A read-only SSH transport is
also available for stores created by this new workflow:

```bash
ev-assignment --store /absolute/local/archive pull --host powerai --remote-store /absolute/remote/sealed-store --attempt ATTEMPT_ID
```

This command requires an explicit sealed store and attempt. It does not import
old campaign folders, watch active files, deploy code or restart jobs. The remote
reader protocol is tested in a separate local process; no new store or release
has been deployed to PowerAI during this refactor.

## Package boundaries

| Package / directory | Role |
|---|---|
| ev_assignment | Migrated models, probability methods, physics and audits |
| ev_assignment_experiments | Config, immutable records, local transfer, development bundles |
| evcs_experiments | Transitional legacy callers, not a new formal release |
| powerai_l08 | Source references; no direct execution during refactoring |
| experiments | Experiment definitions, no generated outputs |

Eleven numerical modules were extracted; their scientific AST is checked against
the baseline after removing import nodes only. Legacy sources, scripts and tests
remain byte-identical to the baseline while entire workflows are migrated. New
scientific tests exercise the extracted modules. Historical source-bound
acceptance gates are not changed or treated as new-release admission.

The `ev_assignment.l08` subpackage additionally extracts L08's full-dwell energy,
islanding, explicit task matrices and exact power-factor reduction. Five scenario
categories compare original and extracted matrices, LP solutions and lifted
physical residuals. This is component equivalence, not controller admission.

## Validation

```bash
uv run pytest tests/unit tests/integration -q
uv run ruff check src/ev_assignment_experiments tests/unit tests/integration
uv build --wheel
```

Original scientific tests remain under tests/. Their full suite requires
separately managed private inputs, source-bound evidence and explicit paths
(EVCS_CONFIG); see implementation status for current results. CI runs only
self-contained local checks, never SSH or a commercial solver.

## Development, compute and publication

Local development and private GitHub own code history. PowerAI executes fixed
versions and temporarily stores artifacts. Completed research is verified and
archived locally. A separate public repository will be assembled from explicitly
selected, audited paper results and exact source versions.

Generated data, results, checkpoints, logs, manuscripts and credentials are
excluded from tracked files. Synthetic definitions and source migration
manifests are tracked.
