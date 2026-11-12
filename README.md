# Event-Ahead EV Assignment and V2G Reserve Procurement

IEEE33 event-ahead electric-delivery-fleet assignment and V2G reserve procurement
under coupled grid/road disruptions. The active implementation separates scientific
models from experiments, execution and research records.

Start with the [acceptance status](docs/architecture/IMPLEMENTATION_STATUS.md),
[approved design](docs/architecture/REFACTOR_PLAN_V2_CN.md) and
[operating guide](docs/operations/WORKFLOW_CN.md). Existing PowerAI experiments keep
using their original immutable releases; this refactor does not take them over.

## Install and validate

Python 3.12, with pinned numerical dependencies:

```bash
uv sync --locked --group dev
uv run pytest tests/unit tests/integration -q
uv run ruff check src/ev_assignment_experiments tests/unit tests/integration
uv build --wheel
```

Pip alternative: `python -m pip install -r requirements.lock -e .`.
For actual optimization, add `uv sync --locked --group dev --extra gurobi` and a
Gurobi 12.0.3 license, then run `uv run --extra gurobi pytest tests/native -q`.
CI does not use a commercial license or connect to PowerAI.

## Run an experiment

Use an absolute storage path outside this checkout. Definitions contain no outputs.

```bash
ev-assignment --store /absolute/research-store plan --experiment experiments/l08/smoke-det.toml
ev-assignment --store /absolute/research-store job launch --id RUN_SHA
ev-assignment --store /absolute/research-store job status --attempt LAUNCH_TOKEN
ev-assignment --store /absolute/research-store audit --id ATTEMPT_SHA
ev-assignment --store /absolute/research-store context --study l08-smoke-det --refresh --markdown
```

DRO and ARO have corresponding small physical examples. Frozen400 runs require an
explicit input manifest and their own scientific/resource admission. A successful
small example or a matrix identity check does not establish full-size convergence.

| Location | Responsibility |
|---|---|
| `src/ev_assignment/` | Typed problems, physics, models, DET/DRO/ARO, numerical certificates |
| `src/ev_assignment_experiments/` | Input adapters, RunPlan, supervision, budgets, immutable records, analysis/export |
| `experiments/` | Explicit experiment definitions and small examples |
| `tests/unit`, `tests/integration` | Portable engineering and numerical checks |
| `tests/native` | Explicit real Gurobi validation |
| `tests/scientific` | Preserved scientific assertions against extracted kernels |
| `docs/`, `skills/` | Architecture, operating rules and task-specific agent instructions |
| `src/evcs_experiments`, `powerai_l08`, original tests/scripts | Byte-preserved historical reference; no new-code dependency |

New core imports neither experiment infrastructure nor legacy code. Input roots are
explicit. Optional Gurobi loads only when requested. Original source-bound acceptance
is preserved as historical evidence; it is never silently reassigned to new code.

## Keep track as experiments grow

Study → ExperimentRevision → Run → Attempt → Audit/Decision → Publication. Each Run
pins actual source, environment, recipe and input identity. Failed attempts remain
visible. Checkpoint recovery retains cumulative costs; source changes require a new
Run. Execution, scientific validation, convergence and research adoption are separate.

`context` produces bounded JSON and Markdown with exact evidence links, recorded next
actions, unresolved claims and pagination. It labels observations by time. Missing or
stale metadata never means a process has died. The SQLite index is disposable.

## Compute, sync and publish

Git synchronizes code with the private repository. PowerAI installs a fixed version
into a new release directory. Never rsync over an active release or its working files.
Artifacts live in a separate content-addressed store, with verified resumable transfers.

```bash
ev-assignment --store /absolute/research-store sync --study STUDY --mode full --destination /absolute/local-archive
ev-assignment --store /absolute/local-archive pull --host powerai --remote-store /absolute/remote-store --attempt ATTEMPT_SHA
ev-assignment --store /absolute/local-archive bundle --attempt ATTEMPT_SHA --destination /absolute/reproduction-bundle
ev-assignment --store /absolute/local-archive verify-bundle --destination /absolute/reproduction-bundle
```

Metadata-only sync is explicitly **not** an archived result. Complete bundles contain
the exact installable source, input/evidence closure, environment and commands to audit,
regenerate result tables, smoke-test and explicitly rerun a selected experiment.
Publication selection requires recorded decisions. DET/DRO/ARO training objectives are
not interchangeable evaluation metrics; paired evaluation has a separate plan/sample lock.

The final paper repository will be created separately, after final results are selected
and audited. This private development repository excludes results, artifacts, logs,
checkpoints, private data and credentials. No public repository is created by the refactor.
