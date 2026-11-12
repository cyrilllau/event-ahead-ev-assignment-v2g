# EV assignment development

Scope: IEEE33 event-ahead EV assignment and V2G reserve; active L08 DET/DRO/ARO.
Read `docs/architecture/IMPLEMENTATION_STATUS.md` and the relevant operating section in
`docs/operations/WORKFLOW_CN.md`. Design: `docs/architecture/REFACTOR_PLAN_V2_CN.md`.

- Core `src/ev_assignment/`; experiment IO/control `src/ev_assignment_experiments/`.
  Core must not import experiments/legacy, mutate sys.path, infer data roots or compute on import.
- Python 3.12, installed package: `uv sync --locked --group dev`. Real native tests need
  `--extra gurobi`. CI: `pytest tests/unit tests/integration` plus portable scientific checks.
- Scientific inputs use explicit manifests. Preserve units, stream addresses, pairing,
  fixed routes/H, initial states, physical tolerances and OOS locks.
- Historical sources/scripts/tests are byte-preserved under `reference/baseline/`,
  excluded from the active install and default test collection. Install its reference
  distribution only for explicit scientific cross-version tests. Never repoint historical
  acceptance hashes or loosen assertions. Private scientific fixtures use explicit roots.
- Source/config changes create a new Run; the same budget group retains old costs.
  Execution, physics, convergence and adoption are distinct. Near-optimal acceptance
  retains original gap/native status and requires a recorded authorized decision.
- PowerAI jobs retain their original releases. Do not modify active sources, current links,
  campaigns, locks, budgets or unrelated processes. New runtime never adopts legacy PIDs.
- Git contains source/config/tests/docs. Research stores must be outside the checkout.
  Do not commit results/artifacts/data/logs/checkpoints/licenses or credentials.
- Read `context --study ID` from the specified store; use `--refresh` only for observations
  on that host. Offline/unknown does not mean dead. Consult precise evidence references.
- Public reproduction uses a separate repository after explicit paper selection. A bundle
  or passing small test is not formal full-scale convergence or permission to publish.

Task instructions: `skills/ev-develop/SKILL.md`, `skills/ev-run-recover/SKILL.md`,
`skills/ev-audit-publish/SKILL.md`. These call the common CLI, not duplicate controllers.

- Full frozen recipes: `experiments/l08/frozen-{det,dro,aro}.toml`, bound via `plan-frozen`.
  Evaluation: `evaluate lock/run/audit/recover`; preserve exact accepted plans, sample
  manifest, random stream and source. Each slice <=60s; resume same Run/budget.
- Existing files enter through `import-legacy` with an explicit manifest. Imported
  historical status is unchecked observation; do not reinterpret it as live/adopted.
- Completion gate: `docs/architecture/DEFG_REMAINING_CN.md`. Historical small-test
  acceptance does not prove full-scale convergence or production replacement admission.
