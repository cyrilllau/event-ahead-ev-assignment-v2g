"""Three original-P risk LPs for one fractional point, with directed certificates."""

from fractions import Fraction
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog, OptimizeWarning

from ev_assignment.l08.envelope import enclosure
from ev_assignment.l08.probability_certificate import certify_with_feasible_anchor
from ev_assignment_experiments.analysis.diagnostic import (
    affine_bounds,
    npz,
    read,
    reference,
    save,
    subtract_intervals,
)
from ev_assignment_experiments.analysis.diagnostic_replay import AROOT
from ev_assignment_experiments.runtime.ledger import BudgetLedger


def run(root, round_root, out, deadline):
    started = time.monotonic()
    base = root / "L08_K4L4_CALIBRATION_AND_FORECAST_R29/05_DRO_MOMENTS"
    dense = np.load(base / "SUPPORT_UNIQUE.npy")
    bounds = npz(base / "EFFECTIVE_INTERVALS.npz")
    lower, upper = bounds["lower"], bounds["upper"]
    scale = np.maximum(
        np.max(abs(dense), axis=1), np.maximum(np.maximum(abs(lower), abs(upper)), 1e-3)
    )
    Phi = sp.csr_matrix(dense)
    phi = sp.csr_matrix(dense / scale[:, None])
    lo, hi = lower / scale, upper / scale
    dmin, dmax = dense.min(axis=1), dense.max(axis=1)

    def check(p):
        exact = bool(p.min() >= 0 and sum(map(Fraction.from_float, p), Fraction(0)) == 1)
        pl, pu = enclosure(Phi, p)
        sl, su = enclosure(phi, p)
        raw_margin = float(
            min(np.min(np.maximum(pl, dmin) - lower), np.min(upper - np.minimum(pu, dmax)))
        )
        scaled_margin = float(
            min(
                np.min(np.maximum(sl, dmin / scale) - lo), np.min(hi - np.minimum(su, dmax / scale))
            )
        )
        return dict(
            passed=exact and raw_margin >= 0 and scaled_margin >= 0,
            exact_simplex=exact,
            raw_margin=raw_margin,
            scaled_margin=scaled_margin,
            original_rows=322311,
            effective_rows=45139,
        )

    # Bind the independently replayed full compression and probability matrix.
    validation = read(round_root / "origins-r1/PROBABILITY_REPLAY.json")
    from ev_assignment_experiments.analysis.diagnostic import digest

    for ref in validation["inputs"]:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Original probability matrix changed")
    anchor_ref = read(round_root / "E4_CONFIG.json")["feasible_probability_certificates"][0]
    if digest(anchor_ref["path"]) != anchor_ref["sha256"]:
        raise ValueError("Probability anchor changed")
    anchor = npz(anchor_ref["path"])["pstar"]
    assert check(anchor)["passed"]
    full = read(round_root / "E4_FULL/RESULT.json")
    if not full["coverage_complete"]:
        raise ValueError("Full scenario upper vector unavailable")
    q = npz(round_root / "E4_FULL/SCENARIO_BOUNDS.npz")
    qbars = np.asarray(
        [
            read(root / AROOT / "oracles" / f"{s:04d}" / "QBAR_PROOF.json")["Qbar"]
            for s in range(400)
        ]
    )
    assert np.all(q["q_lower"] <= q["q_upper"]) and np.all(q["q_upper"] <= qbars)
    scan = read(round_root / "E2_SCAN/SCAN.json")
    saved = npz(round_root / "E2_SCAN/POINT_AND_BOUNDS.npz")
    active = np.asarray([s["theta"] for s in scan["scenarios"]])
    # E2 lift's theta is an outward upper value of the active envelope.
    # Recompute lower endpoint rather than pretending that upper is exact.
    from ev_assignment_experiments.analysis.diagnostic import envelope, scenario_rows

    A = sp.load_npz(round_root / "replay-r1/DRO/A.npz")
    aa = npz(round_root / "replay-r1/DRO/ARRAYS.npz")
    names = read(round_root / "replay-r1/DRO/NAMES.json")
    active_low, active_high = envelope(scenario_rows(A, aa, names, 60780, "DRO"), saved["x"])
    del A
    objectives = [
        (
            "active",
            np.asarray(active_low, float),
            np.maximum(active, np.asarray(active_high, float)),
        ),
        ("archive", saved["archive_lower"], saved["archive_upper"]),
        ("Q", q["q_lower"], q["q_upper"]),
    ]
    matrix = sp.vstack([phi, -phi], format="csc")
    rhs = np.r_[hi, -lo]
    results = []
    for label, ql, qu in objectives:
        begin = time.monotonic()
        remaining = min(290.0, deadline - time.time() - 10)
        if remaining <= 0:
            raise TimeoutError("Risk LP cumulative deadline")
        normal = max(1.0, float(abs(ql).max()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            lp = linprog(
                -ql / normal,
                A_ub=matrix,
                b_ub=rhs,
                A_eq=np.ones((1, 400)),
                b_eq=[1.0],
                bounds=(0, None),
                method="highs",
                options=dict(
                    time_limit=remaining,
                    threads=1,
                    primal_feasibility_tolerance=1e-9,
                    dual_feasibility_tolerance=1e-9,
                ),
            )
        if not lp.success:
            save(
                out / f"{label}_FAILURE.json",
                dict(native_status=int(lp.status), message=lp.message),
            )
            raise RuntimeError("Probability LP not optimal")
        m = len(lo)
        bp = np.maximum(-lp.ineqlin.marginals[:m] * normal, 0.0)
        bm = np.maximum(-lp.ineqlin.marginals[m:] * normal, 0.0)
        b0 = float(-lp.eqlin.marginals[0] * normal)
        pstar, repair = certify_with_feasible_anchor(
            lp.x,
            anchor,
            qbars,
            list(range(400)),
            check,
            nontrivial_margin_hint=2.9694066476472115e-6,
        )
        # Rebind the scaled solver dual to ORIGINAL raw rows. Directed
        # intercept repair covers all 400 upper costs, not only active rows.
        ld = np.longdouble
        eps = np.finfo(ld).eps
        wp, wm = (
            np.asarray(bp, ld) / np.asarray(scale, ld),
            np.asarray(bm, ld) / np.asarray(scale, ld),
        )
        planes, _ = enclosure(Phi.T, wp - wm)
        planes -= abs(Phi.T).astype(ld) @ (4 * eps * (abs(wp) + abs(wm)))
        planes += ld(b0)
        planes -= eps * (abs(b0) + abs(planes))
        shift = max(ld(0), np.max(np.asarray(qu, ld) - planes))
        newb = float(np.nextafter(float(ld(b0) + shift), np.inf))
        terms = np.asarray(upper, ld) * wp - np.asarray(lower, ld) * wm
        err = 8 * eps * (abs(upper * wp) + abs(lower * wm))
        ru = ld(newb) + sum(terms) + sum(err) + (8 * len(terms) + 8) * eps * sum(abs(terms))
        rl, _ = affine_bounds(sp.csr_matrix(pstar[None, :]), [0], ql)
        interval = [float(rl[0]), float(np.nextafter(float(ru), np.inf))]
        if interval[1] < interval[0]:
            raise ValueError("Negative risk certificate interval")
        np.savez_compressed(
            out / f"{label}_CERTIFICATE.npz",
            pstar=pstar,
            raw_probability=lp.x,
            beta0=newb,
            plus=bp,
            minus=bm,
            qL=ql,
            qU=qu,
        )
        result = dict(
            label=label,
            risk_interval=interval,
            native_status=int(lp.status),
            native_objective=float(-lp.fun * normal),
            native_iterations=int(lp.nit),
            correction_shift=float(shift),
            probability_repair=repair,
            original_probability_check=check(pstar),
            relative_interval_width=(interval[1] - interval[0]) / max(1.0, abs(interval[1])),
            wrapper_seconds=time.monotonic() - begin,
            certificate=reference(out / f"{label}_CERTIFICATE.npz"),
        )
        save(out / f"{label}_RESULT.json", result)
        results.append(result)
        print(label, interval, flush=True)
    ar, br, qr = [r["risk_interval"] for r in results]
    from ev_assignment_experiments.analysis.diagnostic import fixed_master_risk

    A = sp.load_npz(round_root / "replay-r1/DRO/A.npz")
    fixed, _, _ = fixed_master_risk(A, aa, names, saved["x"], 60780, "DRO", 90768)
    f = full["first_cost"]
    save(
        out / "RESULT.json",
        dict(
            point_sha256=full["point_sha256"],
            results=results,
            first_cost=f,
            point_DRO_interval=[f + qr[0], f + qr[1]],
            point_ARO_interval=[f + float(q["q_lower"].max()), f + float(q["q_upper"].max())],
            active_fixed_master_risk=fixed,
            delta_pool=subtract_intervals(br, ar, monotone=True),
            delta_recourse=subtract_intervals(qr, br, monotone=True),
            delta_representation=subtract_intervals(ar, fixed),
            representation_nonnegativity_not_assumed=True,
            fractional_batch_count=1,
            full400=True,
            solver_calls=3,
            evidence_grade="ORIGINAL_P_DIRECTED_DUAL_UPPER_AND_EXACT_FEASIBLE_PROBABILITY_LOWER; Q_PRIMALS_NUMERICALLY_AUDITED",
            wrapper_seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    root, round_root, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(round_root / "store", "diagnostic-round1-20261111")
    claim = ledger.claim(
        dict(stage="E4_ORIGINAL_P_RISK", source=reference(__file__)),
        900.0,
        charge_process_start=True,
    )
    save(out / "CLAIM.json", claim)
    rc = 1
    try:
        run(root, round_root, out, time.time() + claim["reserved_seconds"] - 2)
        rc = 0
    finally:
        save(out / "WAIT.json", dict(returncode=rc))
        ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
        save(out / "BUDGET_AFTER.json", ledger.read())
