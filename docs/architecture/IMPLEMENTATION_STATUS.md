# Architecture v2 implementation status

Updated: 2026-11-11, America/Los_Angeles.

**Milestone 1 implemented and locally validated; full refactor NOT complete.**
Do not deploy this branch over an active PowerAI release. Do not merge main until
the remaining stages pass. See [approved plan](REFACTOR_PLAN_V2_CN.md).

## Delivered in this milestone

- Saved the complete approved plan before implementation; baseline tag
  `pre-refactor-v2-20261111` pins commit `0f8b537`.
- Created an independent local checkout and `codex/research-architecture-v2` branch.
- Inventoried all 691 baseline Python files with hashes/imports. Retrieved two
  missing L08 source references read-only (certificate_core, r41_common).
- Extracted 11 numerical modules into `ev_assignment`: random streams, probability
  LP/phase-I/audit, first-stage/recourse/extensive models, zero reduction,
  descriptive compaction, native HiGHS and Gurobi adapters. Also extracted pure
  reactive capability functions. Core does not import legacy or orchestration.
- Preserve all 691 baseline Python files byte-for-byte. Their source-bound
  scientific acceptance must not be repointed silently to refactored files.
- Installed-package layout, pinned numerical dependencies, uv lock + pip export,
  pytest/Ruff and local-only Linux/macOS CI configuration.
- Executable probability-LP development workflow: validated TOML → frozen source
  archive/environment → real solve → independent numerical audit → immutable
  records → verified local transfer → offline reproduction bundle.
- Content-addressed artifacts/records, declared transitive dependencies,
  no-overwrite publication, corruption and missing-reference rejection, local
  rebuildable SQLite projection, bounded factual context and unsettled-claim
  visibility. Context never treats a local observation as server liveness.

## Validation actually performed

| Check | Result |
|---|---|
| New unit/integration tests | 25 PASS |
| Extracted scientific tests | 111 PASS, original numerical assertions retained |
| Numerical AST identity | 11 modules equal to baseline except imports |
| Legacy Python byte identity | 691 files unchanged |
| Legacy full suite, isolated baseline environment | 1887 tests; 4 failures, 49 errors |
| Legacy full suite, current environment | Same 1887 tests, same 4 failures/49 errors; no new failing test IDs |
| Frozen dataset manifest | PASS, 74 files checked read-only |
| Ruff / syntax / diff whitespace | PASS |
| Wheel install outside checkout | PASS; solve/audit/transfer/bundle verified from `/tmp` |
| Synthetic record scale | 10,000 records; context ~0.96s, index rebuild ~0.91s, 3193-byte summary |
| PowerAI interaction | Read-only process/resource/source checks; no new solver run or deployment |

Historical full-suite failures are NOT relabeled as PASS. The source-only export
lacks private provenance/configuration evidence expected by legacy tests. The
unchanged baseline was run in a separate environment with the same declared data
roots and available fixtures to establish the comparison. No test assertion or
historical hash gate was removed or loosened.

The new scientific suite includes nine small SAA/DRO/ARO extensive-form hand
goldens, source-compatible HiGHS starts and exact-zero reduction checks. The
checkpoint integration case uses the legacy controller's own typed inputs and
explicitly injects the extracted array-level HiGHS adapter. This does not claim
that the CCG controller was migrated. Gurobi adapter tests use mocks, not a
commercial solver; formal L08/Gurobi admission remains pending.

## Why parallel extraction

An initial direct-move candidate exposed legacy file-location and exact-source
bindings. That candidate was tested, then replaced with parallel extraction.
Baseline sources/scripts/tests were restored byte-identically; the new packages
and their scientific tests stand separately. This preserves historical evidence
and allows each entire workflow to acquire its own new admission before retiring
the old implementation. Intermediate failed test logs are retained locally.

## Remaining stages and next work

| Plan stage | Current status / remaining work |
|---|---|
| A baseline | Inventory and tag done; complete L08 recursive input/source closure and map exact accepted implementations |
| B foundation | Package boundary and install validated; final typed planning/result interfaces and explicit dataset adapters still required |
| C vertical slice | Probability-oracle component complete; full EV planning→physical audit→journal pipeline still required |
| D methods | Reusable numerical extraction only; actual L08 DET/DRO/ARO controllers/data adapters not migrated or admitted |
| E runtime | Local sealed-store transfer tested; SSH streaming/resume, immutable remote releases, process identities, cumulative-budget recovery, watcher and full archive gate pending |
| F publication | Development bundle tested; paper selection/decisions, figures, public repo export and full reproduction modes pending |
| G final delivery | Not ready to merge main or declare full refactor complete |

Next implementation unit: map L08 model/input/source dependencies without importing
side-effectful scripts; introduce explicit adapters and compare reconstructed
matrix coefficients/variable types/ordering with the frozen references. Then
validate an isolated true EV planning/physical-audit workflow. Keep active PowerAI
jobs/releases/campaigns untouched. Only after isolated admission may the planned
single-thread, <=60s-per-solve Gurobi checks be run in a separate directory.

Current limitations: the transfer reads objects into memory, operates on local
filesystem stores, and only supports sealed closures. No SSH watch, budget-aware
resume, adoption decisions or live process query is implemented in the new CLI.
Claimed runs cannot be automatically restarted. Development bundles explicitly
carry `paper_facing=False`; no public repository has been created.

## Local evidence location

Outside Git: sibling `event-ahead-ev-assignment-v2g-store/validation/stage1-20261111/`.
Key files: `migrated-scientific-tests-final.log`, `baseline-suite.log`,
`legacy-suite-preserved-baseline.log`, `baseline-comparison.json`,
`wheel-validation-final.json`, `scale-validation.json`.
Generated research artifacts, private fixtures, logs, environments and wheels
are excluded from the development Git history.
