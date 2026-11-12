# EV assignment development

Read `docs/architecture/IMPLEMENTATION_STATUS.md` before choosing work. The approved
design is `docs/architecture/REFACTOR_PLAN_V2_CN.md`; a plan is not acceptance evidence.

- Scope: IEEE33 event-ahead EV assignment / V2G reserve; L08 DET/DRO/ARO mainline.
- Core: `src/ev_assignment/`; orchestration: `src/ev_assignment_experiments/`.
- `evcs_experiments`, `powerai_l08`, and old scripts are transitional reference/legacy
  implementation. Their existence does not admit a new production dispatcher.
- Core must not import experiments or legacy modules, mutate sys.path, infer dataset
  roots from source locations, or start solvers/filesystem writes on import.
- Use Python 3.12 and the installed package: `uv sync --group dev`.
- Unit/integration checks: `uv run pytest tests/unit tests/integration`.
  Historical tests require explicit `EVCS_PROJECT_ROOT` / `EVCS_CONFIG` and private
  fixtures. Report their failures; never remove assertions to obtain PASS.
- Lint: `uv run ruff check src/ev_assignment_experiments tests/unit tests/integration`.
- Preserve physical tolerances, random streams, sample pairing, bound scope,
  inherited budget and original approximate-acceptance flags.
- PowerAI currently runs independently. Never modify its active code, locks,
  campaigns, symlinks, process parameters or processes for a refactor.
- Source uses Git; artifacts use a separately configured store. Do not commit live
  results/logs/data/checkpoints or credentials to this private development repo.
- Public paper reproduction is a separate repository, built from audited local
  archive manifests after selection. Do not automatically publish or change visibility.
- Keep factual phase status and remaining scientific admission gaps explicit.
