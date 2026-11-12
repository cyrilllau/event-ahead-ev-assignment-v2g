---
name: ev-run-recover
description: Define, launch, inspect or resume an EV assignment run using sealed recipes and cumulative budgets; applies to the new runtime rather than legacy PowerAI processes.
---

Read `docs/operations/WORKFLOW_CN.md` sections on runs and recovery. Inspect
`ev-assignment --store ABS context --study ID --refresh` and the exact run definition.
Refresh observes processes on the host running the command. A mirrored observation is
not a fresh PowerAI observation.

Use `job launch --id RUN` and `job status --attempt TOKEN`. Resume requires
`--resume CHECKPOINT_SHA` for the exact source, recipe and input identity. Actual parent
waits charge the cumulative budget; changing the attempt or Run cannot clear it.
`job recover` refuses live or unknown owners/children and conservatively charges a dead
owner's reservation when actual overhead is unknown. Do not edit ledger files.

Code releases are immutable. Build/install into a new explicit release path. Existing
legacy jobs have their own authority, budgets and releases; this skill does not authorize
adopting their process, installing over their code or restarting them. Stop the current
inspection after a healthy bounded background launch; subsequent inspection verifies
its receipt and scientific evidence separately.

Use `plan-frozen` to bind a catalog recipe and explicit resources. Paired evaluation
slices use `evaluate run` (<=60s) and share the study lock with planning. Only explicit
`evaluate recover` can settle a proven dead evaluator; it archives the original ledger
and retains previously completed cells. Imported old process IDs are never adopted.
