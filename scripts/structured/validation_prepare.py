"""Materialize all frozen TRAIN/OOS IDs before any validation outcome exists."""
from pathlib import Path
import json
import pickle
import sys
import time
import traceback

from ev_assignment.domain.planning import Control
from ev_assignment_experiments.datasets.structured_round2 import make_fixed_load_case
from ev_assignment_experiments.analysis.diagnostic import reference
from round2_worker import dump, healthy_fallback


def main():
    out, seconds, inputs, freeze = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    cc, started = Control(seconds), time.monotonic()
    plan = json.loads(freeze.read_text())
    result = dict(status="PREPARATION_INCOMPLETE", passed=False, freeze=reference(freeze), cases=[])
    try:
        for seed in plan["validation_seeds"]:
            for split, count in [("validation", 400), ("oos", 1000)]:
                cc.check()
                began = time.monotonic()
                case = make_fixed_load_case(inputs, split, seed, count)
                healthy_fallback(case, cc)
                path = out / f"{split}_{seed}.pkl"
                with path.open("xb") as f:
                    pickle.dump(case, f, protocol=5)
                record = dict(split=split, seed=seed, case=reference(path), identity=case["identity"],
                              ordered_ids=[s["identity"] for s in case["scenarios"]],
                              weights=case["weights"], physical_audit=case["physical_audit"],
                              elapsed_seconds=time.monotonic() - began)
                dump(out / f"{split}_{seed}_MANIFEST.json", record)
                result["cases"].append(record)
                dump(out / "PROGRESS.json", result, True)
                if not case["physical_audit"]["passed"]:
                    raise ValueError(f"Frozen {split} {seed} physical audit failed; no parameter retuning")
        result.update(status="ALL_SAMPLE_IDS_FROZEN_BEFORE_OPTIMIZATION", passed=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        (out / "ERROR.txt").write_text(traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    dump(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
