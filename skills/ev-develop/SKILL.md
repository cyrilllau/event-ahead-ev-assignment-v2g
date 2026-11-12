---
name: ev-develop
description: Change or validate EV assignment scientific code and experiment infrastructure in this repository, preserving frozen L08 semantics and evidence.
---

Read `AGENTS.md` and `docs/architecture/IMPLEMENTATION_STATUS.md`. Core is
`ev_assignment`; configuration, IO and supervision belong to `ev_assignment_experiments`.
Import the installed package. Bind data by explicit manifest, never the source directory.

Choose tests by scientific impact. Unit/integration CI needs no license; `tests/native`
uses real local Gurobi. Frozen400 structural validation is an explicit script with an
external input root. Its zero optimization calls do not prove full-size convergence.
Record the distinction between coefficient identity, small solve, physical audit and
production outcome. Preserve baseline source bytes and scientific assertions.

Use `docs/operations/WORKFLOW_CN.md` for commands and current limitations. A change to
source/configuration creates a new Run under the existing cumulative budget; never
relabel old evidence as a run of the changed source.
