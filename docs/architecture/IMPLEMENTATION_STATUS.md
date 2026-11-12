# Architecture v2 implementation and acceptance

Updated 2026-11-11 (America/Los_Angeles). **The planned engineering refactor is
implemented and locally accepted.** All active source changes are in the new packages.
Main may advance only after the matching private GitHub CI succeeds. The final Git
branch/commit and its Actions checks are the delivery authority.

Read the [operating guide](../operations/WORKFLOW_CN.md) for actual commands and the
[acceptance report](ACCEPTANCE_20261111.md) for evidence and scientific limits.
The [approved plan](REFACTOR_PLAN_V2_CN.md) is retained unchanged. The previous
[milestone report](IMPLEMENTATION_MILESTONES_1_2.md) is preserved as historical text.

| Stage | Delivered and validated |
|---|---|
| A baseline/closure | Baseline tag, 691 original Python files preserved; current L08 input closure sealed locally: 2,047 files, 401 scenario identities; exact source maps |
| B foundation | Installed `ev_assignment` core and `ev_assignment_experiments` orchestration; explicit typed inputs/results/config; lazy Gurobi; Python 3.12 / pinned numerical dependencies |
| C full small workflow | DET/DRO/ARO planning → original physical audit → immutable journal → settlement → sync → offline readback |
| D methods | Native compact formulation, original numerical certificates, DET R38 screening/cuts, DRO theta/probability separation, ARO max controller; real Gurobi small tests and frozen full-size structural checks |
| E runtime | Immutable source releases, exclusive study parent, process identity handshake, whole-parent cumulative costs, explicit checkpoint resume, conservative dead-owner recovery; metadata/results/full sync and bounded local/SSH watch |
| F publication | Explicit decisions/plan locks, paired evaluation guards, bounded JSON/Markdown context, selected source-complete bundles, solver-free audit/table regeneration, smoke and explicit full reproduction |
| G delivery | Acceptance recorded; update private main only after matching CI success; no results/artifacts tracked |

Current-source acceptance: **71 tests passed**, including **11 actual Gurobi tests**;
**111 scientific regressions passed** with explicit historical fixture roots. All seven
structure checks pass, including source bytes/AST and no Gurobi import side effects.
Clean wheel and exact exported-source installation, offline figures/audit, smoke and
full reproduction of a development example pass. A 10,000-record context is bounded.

## Scientific and operational boundary

This is acceptance of the refactored implementation and research infrastructure.
It does not relabel historical results or assert convergence of a new full 30-EV,
400-support campaign. The full frozen model was constructed and compared without
optimization; a one-EV subset of the real IEEE33 data was solved and independently
audited. Native global lower bounds remain `native_solver`; numerical LP/physics
certificates remain `numerical_audit`. No exact-rational global MIP proof is claimed.

The three development methods agree with direct formulations. The reusable controller
preserves the model/certificate semantics, but does not claim identical historical
solver trajectories, basis/node trees or S07 benchmark lineage. Production replacement
requires its own explicit scientific/resource admission and a new immutable release.
Original PowerAI jobs continue with their original sources and budgets.

The old 1,887-test full suite had 4 failures/49 errors on both the isolated baseline
and the earlier preserved checkout because of private evidence/configuration/source
bindings. Those historical results remain unchanged and are not called PASS. No legacy
assertion/source hash was relaxed. This slice revalidated the applicable 111 scientific
cases using explicit original data roots; CI uses the portable subset and local tests.

## Where to resume

For development, read `AGENTS.md`, then the task skill in `skills/`. For a study, use
`ev-assignment --store ABS context --study ID`; inspect the referenced records before
acting. No OOS sample is opened without a plan/sample/protocol lock. No observation
implies a server task is dead. No engineering example is automatically a paper result.

Future work is research-specific: register authorized production recipes/releases,
run the scientific protocol, select final paper evidence, and publish a separate public
reproduction repository. These are not unfinished engineering migration stages and
were not executed against the active PowerAI jobs during this refactor.
