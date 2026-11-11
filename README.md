# Event-Ahead EV Assignment and V2G Reserve Procurement

Research code for assigning electric delivery fleet tasks and procuring vehicle-to-grid (V2G) reserve before storm-induced grid and road disruptions. The research combines a coupled IEEE 33-bus distribution and road network with route replay, battery state of charge, and reserve deliverability checks.

| Directory | Contents |
| --- | --- |
| `src/evcs_experiments/` | IEEE33 v54-Q1 experiment models, validation, and command-line tools |
| `scripts/` | Audit, reporting, and experiment-support code |
| `tests/` | Python test code |
| `powerai_l08/` | Selected PowerAI L08 code for deterministic, distributionally robust, and adjustable robust optimization |

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock -e .
evcs --help
```

Use `configs/local.example.toml` to point the project at separately managed inputs and experiment specifications. Full validation and some tests require those inputs and their evidence fixtures. The PowerAI L08 scripts retain their original server paths. Optimization with Gurobi requires a separately configured environment and license.

This repository contains source code and a configuration template. Research datasets, experiment results, checkpoints, logs, archives, manuscripts, and credentials are managed separately.
