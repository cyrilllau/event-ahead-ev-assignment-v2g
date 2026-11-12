"""One charged validation job; no generation/solving occurs on import.

CLI: OUT SECONDS INPUTS METHOD SEED [individual|count] [prepass|no-prepass]
METHOD: DET, SAA, ARO, P3, P2, P3-noadvance. Seeds: 1101, 1102, 1103.
INPUTS is an explicitly supplied trusted frozen TRAIN CASE.pkl, or the source
directory for make_fixed_load_case. Saved cases avoid repeated generation. No
directory scan, sibling manifest/OOS read, old cuts/bounds, or new ledger occurs.
The parent admits this job only after the validation entry/freeze gate; this
worker does not decide that gate. All actual work uses the parent's wall budget.

All methods start from N30/G6/S400 fixed-load validation physics. DET optimizes
only the declared forecast, with its OWN subproblem identity. P3-noadvance fixes
the earlier choices through the existing core allow_advance=False API, retaining
the complete catalogue and all original Big-M audit rows. SAA uses declared
stratified weights through finite_domain, which records exact-dyadic rounding.
Service summaries concern the optimization support, not paired OOS performance.
"""

import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

from ev_assignment.domain.planning import Control
from ev_assignment.l08 import catalogue as cat
from ev_assignment.methods.catalogue import solve_structured
from ev_assignment.methods.catalogue_probability import calibrated, finite_domain, ProbabilityDomainInfeasible
from ev_assignment_experiments.analysis.diagnostic import reference, save
from ev_assignment_experiments.datasets.structured import _digest, _jsonable


METHODS = ("DET", "SAA", "ARO", "P3", "P2", "P3-noadvance")
SEEDS = (1101, 1102, 1103)


def dump(path, value, replace=False):
    if replace:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False,
                                         allow_nan=False, indent=2))
        temporary.replace(path)
    else:
        save(path, _jsonable(value))


def verify_case(case, seed):
    """Validate frozen TRAIN identity without reading any other artifact."""
    identity = case["identity"]
    if (seed not in SEEDS or identity.get("split") != "validation" or identity.get("seed") != seed
            or identity.get("scenario_count") != 400 or identity.get("fleet_size") != 30
            or identity.get("groups") != 6 or case.get("type_counts") != [5] * 6
            or not case.get("fixed_load_round2") or case.get("fleet_scale") != 1.):
        raise ValueError("Require matching validation seed and fixed-load N30/G6/S400 case")
    if identity.get("sha256") != _digest({k: v for k, v in identity.items() if k != "sha256"}):
        raise ValueError("Frozen case scientific identity hash mismatch")
    scenes = case["scenarios"]
    ids = [s["identity"] for s in scenes]
    weights = np.asarray(case["weights"], float)
    if (len(ids) != 400 or len(set(ids)) != 400 or identity.get("scenarios") != ids
            or weights.shape != (400,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or abs(weights.sum() - 1) > 1e-12
            or not np.array_equal(np.asarray(identity.get("weights")), weights)):
        raise ValueError("Frozen ordered TRAIN support/weights mismatch")
    if (identity.get("forecast") != case["forecast"]["identity"] or
            identity.get("catalogue") != _digest(case["packages"]) or
            len(case["packages"]) != 72):
        raise ValueError("Frozen forecast/catalogue identity mismatch")
    if any(len(s["route_states"]) != 72 for s in [case["forecast"], *scenes]):
        raise ValueError("Frozen route-state/package cardinality mismatch")
    manifest = dict(case_identity=identity["sha256"], split="validation", seed=seed,
                    ordered_ids=ids, weights=weights.tolist(), forecast_id=case["forecast"]["identity"],
                    N=30, G=6, S=400, catalogue_sha256=identity["catalogue"])
    return dict(manifest, sha256=_digest(manifest))


def prepare_method(base, method, control):
    """Derive only the declared support/domain/restriction; never change physics."""
    if method not in METHODS:
        raise ValueError("Unknown validation method")
    case = dict(base)
    if method == "DET":
        case["scenarios"] = [base["forecast"]]
        case["weights"] = np.ones(1)
        case["features"] = np.asarray([base["forecast"]["features"]])
    ids = [s["identity"] for s in case["scenarios"]]
    if method == "DET":
        ambiguity, info = finite_domain(ids, weights=[1.])
    elif method == "SAA":
        ambiguity, info = finite_domain(ids, weights=base["weights"])
    elif method == "ARO":
        ambiguity, info = finite_domain(ids)
    else:
        control.check()
        cc = Control(min(60., control.remaining),
                     stop=lambda: control.stop() or control.remaining <= 0, event=control.event)
        ambiguity, info = calibrated(ids, base["features"].T, base["calibration_features"].T, cc,
            marginal=method == "P2", lower=base["calibration_lower"], upper=base["calibration_upper"],
            calibration_rule=base["calibration_rule"])
    control.check()
    name = {"P3": "P3-DRO", "P2": "P2-DRO", "P3-noadvance": "P3-DRO/no-advance"}.get(method, method)
    view = dict(schema="fixed_load_validation_subproblem_v1", parent_case_identity=base["identity"]["sha256"],
                method=name, requested_method=method, split="validation", seed=base["identity"]["seed"],
                N=30, G=6, source_support_count=len(base["scenarios"]),
                source=base["identity"]["source"], catalogue=base["identity"]["catalogue"],
                type_counts=list(base["type_counts"]), fleet_size=30, groups=6,
                scenario_count=len(ids), scenarios=ids, forecast=base["forecast"]["identity"],
                optimization_support_count=len(ids), ordered_optimization_ids=ids,
                forecast_only=method == "DET", probability_identity=ambiguity.identity,
                catalogue_sha256=base["identity"]["catalogue"],
                first_stage_restriction="latest_departure_only" if method == "P3-noadvance" else "all_packages",
                noadvance_rule="departure_choice == 2 (d=64); no other constraints changed" if method == "P3-noadvance" else None,
                first_stage_weights=[1., 1., 1., 1.], prior_bound_or_cut_reuse=False)
    if method == "P3-noadvance" and any(p["departure_choice"] not in (0, 1, 2) or
            (p["departure_choice"] == 2 and p["d"] != 64) for p in base["packages"]):
        raise ValueError("Frozen no-advance rule no longer matches the catalogue")
    case["identity"] = dict(view, sha256=_digest(view))
    info = dict(info, name=name, identity=ambiguity.identity, ordered_support=ids,
                moment_rows=ambiguity.phi.shape[0], original_322311=False,
                anchor_check=ambiguity.check_original(ambiguity.feasible_anchor),
                feature_scope=base["feature_scope"], optimization_support_count=len(ids))
    return case, ambiguity, info


def actual_sizes(case, individual, control):
    """Actual CSR sizes on one named scene; do not extrapolate its nnz to S400."""
    control.check()
    args = case["packages"], case["type_counts"], len(case["network"]["station_buses"])
    count = cat.build_first(*args, ports=case.get("ports", 20))
    indiv = cat.build_first(*args, ports=case.get("ports", 20), individual=True)
    first = indiv if individual else count
    scene = case["scenarios"][0]
    template = cat.build_recourse(case["network"], case["packages"], scene["physical"], scene["route_states"], first)
    original = cat.build_original_adapter(case["network"], case["packages"], scene["physical"], scene["route_states"], indiv)
    m = original["matrix"]

    def first_size(fs):
        return dict(vars=fs["A"].shape[1], rows=fs["A"].shape[0], nnz=fs["A"].nnz,
                    discrete_vars=int(np.count_nonzero(fs["vtype"] != "C")))

    sizes = dict(count_first=first_size(count), individual_first=first_size(indiv),
        original_first=first_size(original["first"]), representative_scenario=scene["identity"],
        representation="individual" if individual else "count",
        catalogue_recourse=dict(vars=len(template.names), rows=template.W.shape[0],
                                 W_nnz=template.W.nnz, T_nnz=template.T.nnz),
        original_recourse=dict(vars=len(m.cost), rows=m.A.shape[0] + m.H.shape[0],
                               W_nnz=m.A.nnz + m.H.nnz, T_nnz=m.B.nnz + m.G.nnz),
        scope="actual first stages and one named physical block; complete live master size reported by solver")
    control.check()
    return first, sizes


def witness_logger(out):
    """Same six-argument protocol as round2_worker, retaining numerical evidence."""
    def witness(kind, phase, iteration, sid, x, answer):
        dest = out / "witnesses"
        dest.mkdir(exist_ok=True)
        stem = f"{kind}_{phase}_{iteration:03d}_{sid:04d}"
        if kind == "risk":
            dump(dest / (stem + ".json"), dict(answer, x=x))
            return reference(dest / (stem + ".json"))
        if kind == "master_bound":
            A, a = answer["A"], answer["arrays"]
            np.savez_compressed(dest / (stem + ".npz"), A_data=A.data, A_indices=A.indices,
                A_indptr=A.indptr, A_shape=A.shape,
                raw_pi=answer["raw_pi"] if answer["raw_pi"] is not None else np.zeros(0), **a)
            dump(dest / (stem + ".json"), dict(names=answer["names"], certificate=answer["certificate"],
                                              arrays=reference(dest / (stem + ".npz"))))
            return reference(dest / (stem + ".json"))
        keys = {k: answer[k] for k in ("y", "raw_pi", "Pi", "alpha", "beta") if k in answer}
        np.savez_compressed(dest / (stem + ".npz"), x=x, **keys)
        dump(dest / (stem + ".json"), dict(
            {k: v for k, v in answer.items() if k not in keys},
            arrays=reference(dest / (stem + ".npz"))))
        return reference(dest / (stem + ".json"))
    return witness


def deployment(first, case, x):
    """Report count sums without changing individual contracts or policy identity."""
    x = np.asarray(x, float)
    D, H = len(first["package_map"]), len(case["packages"])
    if x.shape != (2 * D,) or not np.isfinite(x).all():
        raise ValueError("Deployment vector does not match the selected representation")
    n, energy = np.zeros(H), np.zeros(H)
    np.add.at(n, first["package_map"], x[:D])
    np.add.at(energy, first["package_map"], x[D:])
    chosen = [dict(case["packages"][h], count=float(v), total_contract_kwh=float(energy[h]))
              for h, v in enumerate(n) if v > 1e-9]
    if (case["identity"]["first_stage_restriction"] == "latest_departure_only" and
            any(p["departure_choice"] != 2 for p in chosen)):
        raise ValueError("No-advance incumbent uses an earlier departure")
    return dict(selected_vehicles=float(n.sum()), selected_fraction=float(n.sum() / 30),
                contract_kwh=float(energy.sum()), effective_selected_types=len({p["g"] for p in chosen}),
                selected_stations=sorted({p["c"] for p in chosen}), packages=chosen,
                total_advance_vehicle_hours=float(sum((64 - p["d"]) / 4 * v for p, v in zip(case["packages"], n))),
                signature=hashlib.sha256(x.tobytes()).hexdigest(),
                representation="individual" if first["individual"] else "count",
                individual_contracts_preserved=True, package_sums_for_reporting_only=True)


def service_summary(case, solved):
    audits = solved.get("original_audit", [])
    expected = [s["identity"] for s in case["scenarios"]]
    if not solved.get("original_expansion_audited") or [a["sid"] for a in audits] != expected:
        return dict(status="INCOMPLETE_ORIGINAL_SUPPORT_AUDIT", completed=len(audits), expected=len(expected))
    weights = np.asarray(case["weights"], float)
    if weights.shape != (len(expected),) or abs(weights.sum() - 1) > 1e-12:
        raise ValueError("Service summary weight/support mismatch")
    fields = ("discharging_vehicles", "discharge_AC_kwh", "total_ENS_kwh", "critical_ENS_kwh", "shortfall_kwh")
    metrics = {}
    for key in fields:
        values = np.asarray([a[key] for a in audits], float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite original-audit service evidence")
        metrics[key] = dict(nominal_weighted=float(weights @ values), minimum=float(values.min()),
                            maximum=float(values.max()))
    return dict(status="COMPLETE_OPTIMIZATION_SUPPORT_AUDIT", scenarios=len(expected),
                support="forecast_only" if len(expected) == 1 else "validation_TRAIN400",
                weights="declared nominal stratified TRAIN weights; singleton one for forecast",
                weights_sum=float(weights.sum()), metrics=metrics, paired_oos=False,
                cross_method_common_support_comparison=len(expected) == 400)


def main(argv=None):
    begun = time.monotonic()
    argv = sys.argv[1:] if argv is None else argv
    if not 5 <= len(argv) <= 7:
        raise ValueError("Usage: validation_worker.py OUT SECONDS INPUTS METHOD SEED [individual|count] [prepass|no-prepass]")
    out, seconds, inputs, method, seed = Path(argv[0]).resolve(), float(argv[1]), Path(argv[2]).resolve(), argv[3], int(argv[4])
    representation = argv[5] if len(argv) > 5 else "count"
    prepass = argv[6] if len(argv) > 6 else "prepass"
    if (method not in METHODS or seed not in SEEDS or representation not in ("individual", "count")
            or prepass not in ("prepass", "no-prepass") or not np.isfinite(seconds) or seconds <= 0
            or not out.is_dir()):
        raise ValueError("Invalid frozen validation job arguments")
    if any((p / ".git").exists() for p in (out, *out.parents)):
        raise ValueError("Validation outputs must be outside Git")
    if (out / "RESULT.json").exists():
        raise FileExistsError("RESULT.json already exists")
    control = Control(seconds)
    result = dict(stage="VALIDATION", method=method, seed=seed, passed=False, status="PREPARING",
                  representation=representation, prepass=prepass == "prepass", source=reference(__file__),
                  validation_opened=True, oos_opened=False, original_322311=False,
                  N=30, G=6, source_support=400, initial_information="same frozen catalogue; no old bounds/cuts/warm starts",
                  timing=dict(generation=0., loading=0., healthy_fallback=0., probability=0., structural_sizes=0.))
    latest_optimization = None

    def progress(row):
        nonlocal latest_optimization
        # Serialize the latest evidence now: the controller mutates nested lists
        # after callbacks. Keep valid bounds even if its stop hook then raises.
        latest_optimization = _jsonable(row)
        dump(out / "PROGRESS.json", dict(job_method=method, seed=seed, optimization=latest_optimization), True)

    dump(out / "PROGRESS.json", result, True)
    try:
        started = time.monotonic()
        if inputs.is_file():
            raw = inputs.read_bytes()
            case = pickle.loads(raw)  # only the explicitly supplied trusted preparation artifact
            result["frozen_case"] = dict(path=str(inputs), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            del raw
            result["timing"]["loading"] = time.monotonic() - started
            result["shared_generation"] = dict(charged_to_this_job=False, seconds=None,
                                                accounting="preparation job; this job charges loading and checks only")
        else:
            if not inputs.is_dir():
                raise ValueError("INPUTS must be a frozen TRAIN pickle or explicit source directory")
            from ev_assignment_experiments.datasets.structured_round2 import make_fixed_load_case
            case = make_fixed_load_case(inputs, "validation", seed, 400, fleet_size=30, groups=6)
            result["timing"]["generation"] = time.monotonic() - started
            result["shared_generation"] = dict(charged_to_this_job=True, seconds=result["timing"]["generation"])
        control.check()
        manifest = verify_case(case, seed)
        dump(out / "SUPPORT_MANIFEST.json", manifest)
        result.update(base_case_identity=case["identity"], support_manifest=reference(out / "SUPPORT_MANIFEST.json"))
        if not case["physical_audit"]["passed"]:
            # Python's script directory resolves this sibling; no sys.path edits.
            from round2_worker import healthy_fallback
            started = time.monotonic()
            healthy_fallback(case, control)
            result["timing"]["healthy_fallback"] = time.monotonic() - started
        result["physical_screen"] = case["physical_audit"]
        if not case["physical_audit"]["passed"]:
            result["status"] = "PHYSICAL_DESIGN_GATE_FAILED_OR_UNRESOLVED"
        else:
            if "frozen_case" not in result:
                with (out / "CASE.pkl").open("xb") as stream:
                    pickle.dump(case, stream, protocol=5)
                result["frozen_case"] = reference(out / "CASE.pkl")
            started = time.monotonic()
            view, ambiguity, info = prepare_method(case, method, control)
            result["timing"]["probability"] = time.monotonic() - started
            dump(out / "MODEL_VIEW.json", view["identity"])
            phi = ambiguity.phi.tocsr()
            np.savez_compressed(out / "PROBABILITY_DOMAIN.npz", phi_data=phi.data, phi_indices=phi.indices,
                phi_indptr=phi.indptr, phi_shape=phi.shape, lower=ambiguity.lower, upper=ambiguity.upper,
                feasible_anchor=ambiguity.feasible_anchor)
            dump(out / "PROBABILITY_DOMAIN.json", dict(info, arrays=reference(out / "PROBABILITY_DOMAIN.npz")))
            result.update(model_identity=view["identity"], probability_identity=ambiguity.identity,
                          probability_domain=reference(out / "PROBABILITY_DOMAIN.json"),
                          optimization_support=len(view["scenarios"]), moment_rows=phi.shape[0])
            started = time.monotonic()
            first, sizes = actual_sizes(view, representation == "individual", control)
            result["timing"]["structural_sizes"] = time.monotonic() - started
            result["model_sizes"] = sizes
            dump(out / "MODEL_SIZES.json", sizes)
            control.check()
            # Parent Control still includes generation/loading/calibration/audits.
            cc = Control(max(.001, control.remaining - 2), stop=control.stop,
                         event=control.event)
            solved = solve_structured(view, ambiguity, cc, individual=representation == "individual",
                prepass=prepass == "prepass", method_name=info["name"], allow_advance=method != "P3-noadvance",
                event=progress,
                witness=witness_logger(out))
            result.update(optimization=solved, status=solved["status"], passed=bool(solved.get("passed")))
            result["model_sizes"]["actual_final_master"] = solved.get("master_size")
            if solved.get("best") is not None:
                result["deployment"] = deployment(first, view, solved["best"])
                np.savez_compressed(out / "PLAN.npz", x=np.asarray(solved["best"]),
                                    package_map=first["package_map"], owner_map=first["owner_map"])
                result["plan"] = reference(out / "PLAN.npz")
            result["service"] = service_summary(view, solved)
            if result["passed"] and result["service"]["status"] != "COMPLETE_OPTIMIZATION_SUPPORT_AUDIT":
                raise ValueError("Solver acceptance lacks complete matching original support audits")
    except Exception as exc:
        if latest_optimization is not None and "optimization" not in result:
            result["optimization"] = latest_optimization
            result["optimization"]["passed"] = False
            result["optimization"]["status"] = "INTERRUPTED_BEFORE_FINAL_RETURN"
        status = ("PROBABILITY_DESIGN_GATE_FAILED" if isinstance(exc, ProbabilityDomainInfeasible) else
                  "BUDGET_OR_ORACLE_INCOMPLETE" if isinstance(exc, (TimeoutError, RuntimeError, InterruptedError)) else
                  "IMPLEMENTATION_OR_IDENTITY_ERROR")
        result.update(passed=False, status=status, error=f"{type(exc).__name__}: {exc}")
        (out / "ERROR.txt").write_text(traceback.format_exc())
    finally:
        result["total_seconds"] = time.monotonic() - begun
        result["remaining_seconds"] = control.remaining
        dump(out / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("method", "seed", "status", "passed", "total_seconds")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
