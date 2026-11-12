"""Remote A3/A4 worker for the same catalogue-certified decomposition adapter."""

from pathlib import Path
import json
import pickle
import sys
import time
import traceback

import numpy as np
from ev_assignment.domain.planning import Control
from ev_assignment.methods.catalogue import solve_structured
from ev_assignment.methods.catalogue_probability import calibrated
from ev_assignment_experiments.datasets.structured import make_case, _jsonable
from ev_assignment_experiments.analysis.diagnostic import read, save, reference


def dump(path, value, *, replace=False):
    if replace:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False))
        temp.replace(path)
    else:
        save(path, _jsonable(value))


def main():
    out, seconds, inputs, selected_path, stage = (Path(sys.argv[1]), float(sys.argv[2]),
        Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
    control = Control(seconds)
    selection = read(selected_path)
    if stage == "A3":
        chosen = [c for c in selection["candidates"] if c["id"] in selection["finalists"]]
        n, support = 12, 64
    elif stage == "A4":
        chosen = [c for c in selection["candidates"] if c.get("passed")][:1]
        n, support = 30, 400
    else:
        raise ValueError("Only A3/A4 allowed")
    if not chosen or len(chosen) > 2 or not selection["passed"]:
        raise ValueError("Previous stage did not pass / shortlist size invalid")
    result = dict(stage=stage, passed=False, status="RUNNING", candidates=[],
                  selection=reference(selected_path), validation_opened=False,
                  fleet_size=n, support=support, groups=6, full400=support == 400)
    for i, selected in enumerate(chosen):
        path = out / selected["id"]
        path.mkdir()
        cc = Control(max(.01, (control.remaining - 2) / (len(chosen) - i)))
        started = time.monotonic()
        row = dict(id=selected["id"], passed=False, physical_config=selected["physical_config"],
                    inverse=selected["inverse"], status="RUNNING")
        try:
            case = make_case(row["physical_config"], inputs, "development", 101,
                             scenario_count=support, fleet_size=n, groups=6)
            with (path / "CASE.pkl").open("xb") as f:
                pickle.dump(case, f, protocol=5)
            row.update(science_identity=case["identity"], physical_audit=case["physical_audit"],
                        case_file=reference(path / "CASE.pkl"))
            if not case["physical_audit"]["passed"]:
                row["status"] = "PHYSICAL_GATE_FAILED"
            else:
                ambiguity, calibration = calibrated([s["identity"] for s in case["scenarios"]],
                    case["features"].T, case["calibration_features"].T, cc,
                    lower=case["calibration_lower"], upper=case["calibration_upper"],
                    calibration_rule=case["calibration_rule"])
                dump(path / "CALIBRATION.json", calibration)
                def witness(kind, phase, iteration, sid, x, answer):
                    dest = path / "witnesses"
                    dest.mkdir(exist_ok=True)
                    stem = f"{kind}_{phase}_{iteration:03d}_{sid:04d}"
                    if kind == "risk":
                        dump(dest / (stem + ".json"), dict(answer, x=x))
                        return reference(dest / (stem + ".json"))
                    if kind == "master_bound":
                        A, a = answer["A"], answer["arrays"]
                        np.savez_compressed(dest / (stem + ".npz"), A_data=A.data,
                            A_indices=A.indices, A_indptr=A.indptr, A_shape=A.shape,
                            raw_pi=answer["raw_pi"] if answer["raw_pi"] is not None else np.zeros(0), **a)
                        dump(dest / (stem + ".json"), dict(names=answer["names"], certificate=answer["certificate"],
                                                           arrays=reference(dest / (stem + ".npz"))))
                        return reference(dest / (stem + ".json"))
                    data = {k: answer[k] for k in ["y", "raw_pi", "alpha", "beta"] if k in answer}
                    np.savez_compressed(dest / (stem + ".npz"), x=x, **data)
                evidence = solve_structured(case, ambiguity, cc, weights=row["inverse"]["weights"],
                    event=lambda s: dump(path / "PROGRESS.json", s, replace=True), witness=witness)
                row["optimization"] = evidence
                if evidence["best"] is not None:
                    H = len(case["packages"])
                    counts = np.asarray(evidence["best"][:H])
                    selected_count = float(counts.sum())
                    row["selection_fraction"] = selected_count / n
                    row["nondegenerate"] = .1 <= selected_count / n <= .9
                    row["selected_packages"] = [dict(case["packages"][h], count=float(v))
                                                 for h, v in enumerate(counts) if v > 0]
                else:
                    row["nondegenerate"] = False
                row.update(passed=bool(evidence["passed"] and row["nondegenerate"]),
                           status="PASS" if evidence["passed"] and row["nondegenerate"] else
                           "NONDEGENERACY_GATE_FAILED" if evidence["passed"] else evidence["status"])
        except Exception as exc:
            row.update(status="IMPLEMENTATION_OR_BUDGET_FAILURE", error=f"{type(exc).__name__}: {exc}")
            (path / "ERROR.txt").write_text(traceback.format_exc())
        row["elapsed_seconds"] = time.monotonic() - started
        dump(path / "RESULT.json", row)
        result["candidates"].append(row)
        print(json.dumps(dict(stage=stage, id=row["id"], status=row["status"])), flush=True)
    result.update(passed=any(c["passed"] for c in result["candidates"]),
                  status="PASS" if any(c["passed"] for c in result["candidates"]) else "NO_PASSING_CANDIDATE",
                  elapsed_seconds=seconds - control.remaining)
    dump(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
