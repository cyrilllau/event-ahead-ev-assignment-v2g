"""Summarize complete reaudited parity cells and replay full-support risk bounds.

No planning/probability optimization. Old native lower bounds retain their original
evidence grade. --bounds names DET result/native, DRO best/final, ARO state/native
files from the same immutable evidence manifest; results remain outside Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from scipy import sparse

from ev_assignment.l08.certificates import enclosure
from ev_assignment.validation.l08 import check_risk
from ev_assignment.validation.parity import interval_parity
from ev_assignment_experiments.datasets.l08 import L08Dataset, R41
from ev_assignment_experiments.workflows.probability import source_snapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "bounds", "validation", "checker", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    validation = Path(a.validation)
    run = json.loads((validation / "RUN.json").read_text())
    if run["spec_sha"] != digest(a.spec) or run["checker_sha"] != digest(a.checker):
        raise ValueError("Validation source/spec changed")
    if run["source_sha"] != hashlib.sha256(source_snapshot()).hexdigest():
        raise ValueError("Executing package differs from validated source")
    api = runpy.run_path(a.checker)
    spec, bounds = json.loads(Path(a.spec).read_text()), json.loads(Path(a.bounds).read_text())
    evidence = api["Evidence"](
        spec["evidence_root"], spec["evidence_manifest"], spec["evidence_manifest_sha"]
    )
    d = L08Dataset(spec["inputs"], spec["input_manifest"])
    inputs = d.inputs.verify()
    cells = {}
    metrics = ("primal", "critical_ENS", "other_ENS", "actual_V2G_ac", "shortfall_ac")
    methods = {}
    for method, n in (("DET", 1), ("DRO", 400), ("ARO", 400)):
        rows = []
        for i in range(n):
            path = validation / method / f"{i:04d}.json"
            row = json.loads(path.read_text())
            readback = json.loads(path.with_suffix(".readback.json").read_text())
            if (
                not row["passed"]
                or not readback["passed"]
                or readback["cell_sha"] != digest(path)
                or readback["witness_sha"] != digest(path.with_suffix(".npz"))
                or row["method"] != method
                or row["sid"] != ("forecast44" if method == "DET" else f"train/{i:04d}")
            ):
                raise ValueError("Incomplete/corrupt/misordered independent readback")
            rows.append(row)
        cells[method] = rows
        first = rows[0]["first_cost"]
        if any(r["first_cost"] != first for r in rows):
            raise ValueError("Scenario evaluation changed the fixed first stage")
        comparable = [r for r in rows if r["old_dispatch"] is not None]
        methods[method] = dict(
            scenarios=n,
            fixed_plan_objective_parity=True,
            first_cost=first,
            old_primal_physics_audited=len(comparable),
            old_analytic_zero_checked_by_fresh_LP=n - len(comparable),
            max_old_primal_physical_residual=max(
                r["old_dispatch"]["original_residual"] for r in comparable
            ),
            max_new_primal_physical_residual=max(r["new"]["primal_residual"] for r in rows),
            max_abs_dispatch_metric_deltas={
                k: max(abs(r["new"][k] - r["old_dispatch"][k]) for r in comparable) for k in metrics
            },
            max_abs_cost_component_delta=max(
                abs(x - y)
                for r in comparable
                for x, y in zip(r["new"]["cost_parts"], r["old_dispatch"]["cost_parts"])
            ),
            max_first_cost_component_delta=max(
                abs(x - y)
                for r in comparable
                for x, y in zip(
                    r["old_dispatch"]["first_cost_parts"],
                    comparable[0]["old_dispatch"]["first_cost_parts"],
                )
            ),
        )
    det = evidence.read(bounds["det_result"])
    det_native = evidence.read(bounds["det_native"])
    dro = evidence.read(bounds["dro_best"])
    dro_final = evidence.read(bounds["dro_final"])
    aro = evidence.read(bounds["aro_state"])
    aro_native = evidence.read(bounds["aro_native"])
    dro_sources = evidence.read(bounds["dro_sources"])
    if det["native"] != det_native:
        raise ValueError("DET native evidence does not match the selected result")
    if digest(evidence.path(bounds["aro_native"])) != aro["best_LB_source"]["sha256"]:
        raise ValueError("ARO native evidence is not the selected lower-bound source")
    if (
        dro_sources["history"]["LB"] != dro_final["bounds"]["LB"]
        or dro["UB_audited"] != dro_final["bounds"]["UB"]
    ):
        raise ValueError("DRO final bounds do not match the selected historical sources")
    if dro["science"] != d.input_identity["identity"]:
        raise ValueError("DRO historical science changed")
    by_method = {c["method"]: c for c in spec["cases"]}
    if (
        dro["point_sha"] != by_method["DRO"]["point_sha"]
        or aro["best_plan_id"] != by_method["ARO"]["point_sha"]
    ):
        raise ValueError("Bounds are from a different plan")
    first_arrays = d.inputs.npz(R41 + "/identity/FIRST_ARRAYS.npz")
    for method in ("DRO", "ARO"):
        case = by_method[method]
        x = evidence.npz(case["point"])[case["point_key"]]
        original_f = float(
            np.asarray(first_arrays["cost"], np.longdouble) @ np.asarray(x, np.longdouble)
        )
        delta = methods[method]["first_cost"] - original_f
        if abs(delta) > 1e-7:
            raise ValueError("First-stage objective differs from original frozen coefficients")
        methods[method]["first_cost_delta_from_original_coefficients"] = delta
    ambiguity = d.ambiguity()
    certificate = evidence.npz(dro["risk"]["certificate"]["path"])
    if (
        digest(evidence.path(dro["risk"]["certificate"]["path"]))
        != dro["risk"]["certificate"]["sha256"]
    ):
        raise ValueError("Original probability certificate changed")
    old = dict(
        certificate, risk_U=dro["risk"]["risk_U"], all_support_ids=list(ambiguity.support_ids)
    )
    check_risk(ambiguity, certificate["qU"], old)
    q = evidence.npz(dro["Q"]["path"])
    if hashlib.sha256(q["x"].tobytes()).hexdigest() != by_method["DRO"]["point_sha"]:
        raise ValueError("Risk vector is from a different first-stage plan")
    if not all(np.array_equal(certificate[k], q[k]) for k in ("qL", "qU")):
        raise ValueError("Risk certificate uses a different scenario vector")
    lower = np.array([r["new"]["qL"] for r in cells["DRO"]])
    upper = np.array([r["new"]["qU"] for r in cells["DRO"]])
    plane = certificate["beta0"] + ambiguity.phi.T @ (certificate["plus"] - certificate["minus"])
    # Extend the same verified plane only by the observed new numerical guard.
    shift = max(0.0, float(np.max(upper - plane))) + 1e-7
    beta0 = float(certificate["beta0"] + shift)
    risk_u = float(
        np.nextafter(
            beta0
            + ambiguity.upper @ certificate["plus"]
            - ambiguity.lower @ certificate["minus"]
            + 1e-7,
            np.inf,
        )
    )
    risk_l = float(enclosure(sparse.csr_matrix(lower[None, :]), certificate["pstar"])[0][0])
    new = dict(old, beta0=beta0, risk_U=risk_u)
    check_risk(ambiguity, upper, new)
    overlap = interval_parity(dro["risk"]["risk_L"], dro["risk"]["risk_U"], risk_l, risk_u)
    if not overlap["passed"]:
        raise ValueError("Full-ambiguity risk intervals disagree")
    methods["DRO"]["risk"] = dict(
        overlap,
        full_original_rows=322311,
        effective_rows=45139,
        probability_solver_calls=0,
        dual_plane_shift=shift,
    )
    methods["DRO"]["old_total_UB"] = dro["UB_audited"]
    methods["DRO"]["new_total_interval"] = [
        methods["DRO"]["first_cost"] + risk_l,
        methods["DRO"]["first_cost"] + risk_u,
    ]
    aro_records = evidence.read(by_method["ARO"]["records"])["records"]
    old_aro_q = max(r["qU_audited"] for r in aro_records)
    new_aro = [
        max(r["new"][key] for r in cells["ARO"]) + methods["ARO"]["first_cost"]
        for key in ("qL", "qU")
    ]
    old_aro = [
        max(r["qL"] for r in aro_records) + methods["ARO"]["first_cost"],
        aro["best_UB_audited"],
    ]
    if abs(methods["ARO"]["first_cost"] + old_aro_q - aro["best_UB_audited"]) > 1e-7:
        raise ValueError("ARO original full400 max aggregation mismatch")
    if not interval_parity(*old_aro, *new_aro)["passed"]:
        raise ValueError("ARO minmax objective differs")
    methods["ARO"].update(old_total_UB=aro["best_UB_audited"], new_total_interval=new_aro)
    methods["DET"].update(
        old_total=det["native"]["native_UB"],
        new_total=methods["DET"]["first_cost"] + cells["DET"][0]["new"]["primal"],
        original_primal_delta=cells["DET"][0]["parity"]["total_objective_delta"],
    )
    native = dict(
        DET=dict(
            status=det["status"],
            LB=det["bounds"]["LB"],
            UB=det["bounds"]["UB"],
            original_native=det_native,
            source_grade="historical_native_MIP_bound",
        ),
        DRO=dict(
            status=dro_final["status"],
            **dro_final["bounds"],
            source_grade=dro_sources["history"]["source_scope"],
            lower_bound_reference=dro_sources["history"]["certificate"],
        ),
        ARO=dict(
            LB=aro["best_LB"],
            UB=aro["best_UB_audited"],
            original_native=aro_native,
            phase=aro["phase"],
            source_grade="historical_native_MIP_bound",
            stored_external_gap=aro["external_gap"],
        ),
    )
    for method, item in native.items():
        item["recomputed_absolute_gap"] = item["UB"] - item["LB"]
        item["recomputed_relative_gap"] = (item["UB"] - item["LB"]) / max(1.0, abs(item["UB"]))
        item["lower_bound_reproved_by_this_check"] = False
    native["DET"]["stored_gap_consistent"] = (
        abs(det["bounds"]["signed_gap"] - native["DET"]["recomputed_absolute_gap"]) <= 1e-7
    )
    native["DRO"]["stored_gap_consistent"] = (
        abs(dro_final["bounds"]["absolute_gap"] - native["DRO"]["recomputed_absolute_gap"]) <= 1e-7
    )
    native["ARO"]["stored_gap_consistent"] = (
        abs(aro["external_gap"]["absolute_gap"] - native["ARO"]["recomputed_absolute_gap"]) <= 1e-7
    )
    issues = [
        dict(
            method=m,
            issue="STALE_HISTORICAL_GAP_SUMMARY",
            action="retain_original_and_report_recomputed_arithmetic",
        )
        for m, x in native.items()
        if not x["stored_gap_consistent"]
    ]
    report = dict(
        fixed_plan_parity_passed=True,
        cells=801,
        solver_free_readbacks=801,
        inputs=inputs,
        methods=methods,
        native_bounds=native,
        metadata_issues=issues,
        all_metadata_consistent=not issues,
        source=run,
        report_generator_sha=digest(__file__),
        bounds_spec_sha=digest(a.bounds),
        no_global_reoptimization=True,
        no_new_plan_adoption=True,
        oos_opened=False,
    )
    api["write_new"](Path(a.output), report)
    print(
        json.dumps({k: report[k] for k in ("fixed_plan_parity_passed", "cells", "metadata_issues")})
    )


if __name__ == "__main__":
    main()
