"""One preregistered G stress, preserving failures and the shared wall ledger."""
from pathlib import Path
import pickle
import sys
import time
import traceback

from ev_assignment.domain.planning import Control
from ev_assignment.methods.catalogue import solve_structured
from ev_assignment.methods.catalogue_probability import calibrated
from ev_assignment_experiments.datasets.structured_stress import make_stress_case
from ev_assignment_experiments.analysis.diagnostic import reference
from round2_worker import dump, healthy_fallback
from validation_worker import actual_sizes, witness_logger, deployment, service_summary


def main():
    out, seconds, inputs, groups = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    cc, begun = Control(seconds), time.monotonic()
    result = dict(status="STRESS_INCOMPLETE", passed=False, G=groups, N=30, support=400,
                  source=reference(__file__), seed=2101)
    try:
        case = make_stress_case(inputs, groups)
        healthy_fallback(case, cc)
        with (out / "CASE.pkl").open("xb") as f:
            pickle.dump(case, f, protocol=5)
        result.update(case_identity=case["identity"], physical_screen=case["physical_audit"],
                      protocol=case["stress_protocol"], soc_report=case["soc_report"], case=reference(out / "CASE.pkl"))
        if not case["physical_audit"]["passed"]:
            result["status"] = "STRESS_PHYSICAL_GATE_FAILED_OR_UNRESOLVED"
        else:
            ids = [s["identity"] for s in case["scenarios"]]
            ambiguity, info = calibrated(ids, case["features"].T, case["calibration_features"].T, cc,
                lower=case["calibration_lower"], upper=case["calibration_upper"], calibration_rule=case["calibration_rule"])
            dump(out / "PROBABILITY_DOMAIN.json", info)
            first, sizes = actual_sizes(case, False, cc)
            result["model_sizes"] = sizes
            child = Control(max(.01, cc.remaining - 3), stop=lambda: cc.stop() or cc.remaining <= 3)
            solved = solve_structured(case, ambiguity, child,
                event=lambda r: dump(out / "PROGRESS.json", r, True), witness=witness_logger(out))
            result.update(optimization=solved, status=solved["status"], passed=solved["passed"])
            if solved.get("best") is not None:
                view = dict(case, identity=dict(case["identity"], first_stage_restriction="all_packages"))
                result["deployment"] = deployment(first, view, solved["best"])
            result["service"] = service_summary(case, solved)
    except Exception as exc:
        result.update(status="STRESS_BUDGET_OR_ORACLE_INCOMPLETE" if isinstance(exc, (TimeoutError, RuntimeError)) else
                      "IMPLEMENTATION_OR_IDENTITY_ERROR", error=f"{type(exc).__name__}: {exc}")
        (out / "ERROR.txt").write_text(traceback.format_exc())
    result["total_seconds"] = time.monotonic() - begun
    dump(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
