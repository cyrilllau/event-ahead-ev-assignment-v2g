# Refactor acceptance — 2026-11-11

The remaining engineering stages were implemented in isolated checkout
`event-ahead-ev-assignment-v2g`. The original `ev-assignment` workspace and active
PowerAI release were not edited. Source/config/tests/docs are versioned; all input,
result, log, package and acceptance artifacts are in the sibling external store.

## Implementation checkpoints

| Commit | Scope |
|---|---|
| `7f2faff` | Typed planning interfaces, explicit frozen inputs, original native L08 kernels, DET/DRO/ARO controllers, real native tests |
| `186b523` | Supervision, process identity, cumulative budgets, checkpoint/record evidence, read-only archive transfer/watch |
| `5f5019f` | Explicit decisions, evaluation locks, publication selection/export, CLI and agent operating instructions |
| `e148cb9` | Git provenance alongside actual source hashes; full frozen native variable-type check |
| `12ba039` | Charge parent import/startup overhead as well as worker/wait time |

Final executing-source archive:
`90a51c91e8623de9b50f5828fd3d512801fc286c3fd8d09426b2d2ecb5df3ba6`.
Later acceptance/documentation commits do not change that source archive.

## Gates and evidence

| Gate | Actual result |
|---|---|
| Unit/integration + actual native tests | 71 PASS (60 portable, 11 Gurobi 12.0.3) |
| Applicable scientific regression suite | 111 PASS with explicit original fixture root; assertions unchanged |
| Portable scientific CI subset | 77 PASS; no commercial solver/SSH |
| Structure/source invariants | Seven checks PASS; original 691 Python files unchanged; original numerical function AST preserved except documented imports/local constants |
| Full input closure | 2,047 files / 2,511,584,071 bytes; all hashes validated; no OOS content |
| Scene provenance | forecast44 + 400 train scenario/replay identities checked |
| Original ambiguity | All 322,311 original rows verified against exact compression into 45,139 effective rows; original strict feasible anchor passed |
| Native first stage | 60,780 columns / 90,768 rows / 25,770 binaries; exact coefficients, objective, bounds, row/column names/order and variable types match frozen R41 |
| DET R38 | 99,234 columns / 140,611 rows; exact 13,104-route screening and 84 capacity cuts checked, including every integer secant case |
| Real frozen physical solve | One-EV IEEE33 development subset, actual Gurobi and original physical audit PASS; not a full-scale experiment |
| Method workflow | DET/DRO/ARO agree with direct small formulations; complete physical evidence, real parent/worker receipts and sealed independent readback |
| Recovery/accounting | Real parent cost, startup inclusion, explicit same-run resume, preserved old cost, PID reuse/unknown rejection, real dead-parent recovery and no-reset checks PASS |
| Archive | Streaming interruption/resume, digest corruption, missing dependencies, symlink/quoting protection, read-only remote protocol and offline watch checks PASS |
| Full offline inputs | 2,053-member run closure materialized from archived objects; all 401 scenes reread, zero remote/solver calls, zero attempts launched |
| Publication | Actual selected development bundle; exact exported source install and clean wheel; solver-free figure/table regeneration and audit; real smoke/full example entrypoints PASS |
| Context scale | 10,000 records; 10 returned, 9,990 omitted explicitly; 8,872-byte response, ~1.15 s query and ~1.17 s index rebuild |
| Git scope | No tracked results/artifacts/checkpoints/data/logs/licenses/binary outputs |
| PowerAI protection | Original parent/worker PIDs still observed and entry SHA unchanged; no remote source/state/release/process mutations |

Linux and macOS CI runs are available at the
[private Actions workflow](https://github.com/cyrilllau/event-ahead-ev-assignment-v2g/actions/workflows/tests.yml).
Run [43137813766](https://github.com/cyrilllau/event-ahead-ev-assignment-v2g/actions/runs/43137813766)
passed for the main implementation checkpoint. Final main integration additionally
requires success on the exact integration commit; GitHub records that check separately.

## Local evidence and reproducibility

Relative to sibling `event-ahead-ev-assignment-v2g-store/validation/remaining-20261111/`:

- `accepted-final/acceptance.json`: acceptance scope and content hashes of receipts.
- `final-parent-accounting-tests.log`, `all-scientific-explicit-root.log`.
- `frozen-final.json`, `frozen-subset.json`, `frozen-input-identity.json`.
- `accepted-final/frozen-archive-acceptance.json`: current-source full offline closure.
- `accepted-final/delivery-acceptance.json` and `delivery-bundle/`.
- `accepted-final/clean-export-acceptance.json`: `/tmp`, clean environments, no original checkout dependency.
- `scale-acceptance.json`, `git-scope.json`, `powerai-protection-final.json`.

The frozen input manifest SHA is
`310d39bcafa28a095460aa81d593cfef83e7dcb6f6a4009bdef06ce074258d7a`.
The final full-input run closure SHA is
`581c5db2caa4908baf0f5e9c34a388cb1ffe547b18a4fd18d0c4d9f586a4abdf`.
Intermediate failed runs/logs and earlier accepted source snapshots are retained; only
`accepted-final/` is the final current-source delivery. No raw evidence is pushed to Git.

## Limits preserved deliberately

The refactor does not take over or rerun the active PowerAI campaign, change original
mathematics, unlock OOS, renew production budgets or publish a public repository.
Research decisions remain explicit. Approximate acceptance does not rewrite the native
gap. Global native MIP lower bounds are identified as such; offline physical/LP audit
does not claim an exact global rational proof. Full-size convergence and final paper
selection belong to subsequent authorized research execution.

Historical full-suite failures are documented in the previous milestone report. The
first attempt to run all 111 scientific tests without an explicit private fixture root
failed with missing inputs; the explicit-root rerun passed. Both logs are preserved.
No scientific tolerance was relaxed, no failing assertion deleted, and no private
fixture copied into Git to manufacture a green result.
