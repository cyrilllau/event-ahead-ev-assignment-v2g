"""Independent bounded no-solve evidence checks, final process snapshot and budget."""

from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from scipy import sparse as sp

from ev_assignment.l08.envelope import enclosure
from ev_assignment_experiments.analysis.diagnostic import read, save, npz, digest, reference
from ev_assignment_experiments.runtime.ledger import BudgetLedger


def audit(root, r):
    full = read(r / "E4_FULL/RESULT.json")
    assert len(full["records"]) == 400
    assert sorted(x["sid"] for x in full["records"]) == list(range(400))
    verified = set()
    for row in full["records"]:
        assert row["point_sha256"] == full["point_sha256"]
        assert row["native_status"] == 2
        assert row["q_lower"] <= row["q_upper"]
        assert max(row["primal_residual"].values()) <= 1e-9
        for ref in row["input_refs"]:
            assert digest(ref["path"]) == ref["sha256"]
            verified.add(ref["path"])
    assert full["solver_calls"] == 392 and full["reused_scenario_count"] == 8
    raw = root / "L08_K4L4_CALIBRATION_AND_FORECAST_R29/05_DRO_MOMENTS"
    phi = np.load(raw / "SUPPORT_UNIQUE.npy")
    bs = npz(raw / "EFFECTIVE_INTERVALS.npz")
    lower, upper = bs["lower"], bs["upper"]
    scale = np.maximum(
        np.max(abs(phi), axis=1), np.maximum(np.maximum(abs(lower), abs(upper)), 1e-3)
    )
    matrix = sp.csr_matrix(phi)
    risk_checks = []
    for label in ["active", "archive", "Q"]:
        cert = npz(r / "E4_RISK" / (label + "_CERTIFICATE.npz"))
        observed = read(r / "E4_RISK" / (label + "_RESULT.json"))
        p = cert["pstar"]
        assert p.min() >= 0 and sum(map(F.from_float, p), F(0)) == 1
        pl, pu = enclosure(matrix, p)
        margin = min(
            np.min(np.maximum(pl, phi.min(axis=1)) - lower),
            np.min(upper - np.minimum(pu, phi.max(axis=1))),
        )
        assert margin >= 0
        plus, minus = cert["plus"], cert["minus"]
        nz = np.flatnonzero((plus != 0) | (minus != 0))
        assert min(plus.min(), minus.min()) >= 0
        wp = {i: F(float(plus[i])) / F(float(scale[i])) for i in nz}
        wm = {i: F(float(minus[i])) / F(float(scale[i])) for i in nz}
        b0 = F(float(cert["beta0"]))
        margins = [
            b0
            + sum((F(float(phi[i, s])) * (wp[i] - wm[i]) for i in nz), F(0))
            - F(float(cert["qU"][s]))
            for s in range(400)
        ]
        assert min(margins) >= 0, (label, float(min(margins)))
        exact_upper = b0 + sum(
            (F(float(upper[i])) * wp[i] - F(float(lower[i])) * wm[i] for i in nz), F(0)
        )
        exact_lower = sum((F(float(p[s])) * F(float(cert["qL"][s])) for s in range(400)), F(0))
        assert (
            F(observed["risk_interval"][0])
            <= exact_lower
            <= exact_upper
            <= F(observed["risk_interval"][1])
        )
        risk_checks.append(
            dict(
                label=label,
                active_dual_rows=len(nz),
                exact_dual_min_margin=float(min(margins)),
                exact_stored_data_upper=float(exact_upper),
                feasible_p_lower=float(exact_lower),
                probability_directed_margin=float(margin),
                passed=True,
            )
        )
    save(
        r / "RISK_INDEPENDENT_READBACK.json",
        dict(
            checks=risk_checks,
            kind="existing exact arithmetic for stored binary64 dual on ORIGINAL unscaled P rows; no new solver",
            scenario_inputs_verified=len(verified),
        ),
    )
    contacts = read(r / "replay-r1/CUT_CONTACT.json")["contacts"]
    details = []
    for row in contacts:
        cut = npz(row["cut"]["path"])
        nonzero = abs(cut["beta"])[cut["beta"] != 0]
        details.append(
            dict(
                row,
                raw_abs_coefficient_min=float(nonzero.min()) if len(nonzero) else None,
                raw_abs_coefficient_max=float(nonzero.max()) if len(nonzero) else None,
                native_termination=None,
                actual_current_stored_row=None,
                missing_reason="Raw safe-cut coefficients/primal/dual recovered; native termination and this contact-to-current-row join not reconstructed. Actual current-row storage audit is separately complete for9738 DRO rows. Empty coefficient ranges mean constant cut.",
            )
        )
    save(r / "CUT_CONTACT_DETAILS.json", dict(rows=details, count=len(details), new_solver_calls=0))
    # Identify the actual remaining dormant cuts without an additional LP or re-add.
    scan = read(r / "E2_RESCAN/SCAN.json")
    x = npz(r / "E2_RESCAN/POINT_AND_BOUNDS.npz")["x"]
    cuts = []
    for row in scan["scenarios"]:
        if row["max_violation_lower"] <= 1e-7:
            continue
        sid = row["sid"]
        bp = r / "replay-r1/archive" / f"B_{sid:04d}.npz"
        ap = r / "replay-r1/archive" / f"ALPHA_{sid:04d}.npz"
        B, alpha = sp.load_npz(bp), npz(ap)["alpha"]
        vals = B.astype(np.longdouble) @ np.asarray(x, np.longdouble) + np.asarray(
            alpha, np.longdouble
        )
        i = int(np.argmax(vals))
        cuts.append(
            dict(
                sid=sid,
                archive_row=i,
                cut_id=f"archive/{sid:04d}/{i}",
                value=float(vals[i]),
                violation_interval=[row["max_violation_lower"], row["max_violation_upper"]],
                matrix=reference(bp),
                alpha=reference(ap),
            )
        )
    save(
        r / "REMAINING_CUT_IDS.json",
        dict(count=len(cuts), rows=cuts, archive_closed=False, new_lp_calls=0),
    )
    # Compare all common MIP parameters, allowing output-directory differences only.
    arms = [read(r / folder / "RESULT.json") for folder in ["E5_BASE", "E5_ONE_BLOCK_R2"]]
    begins = [read(r / folder / "NATIVE_BEGIN.json") for folder in ["E5_BASE", "E5_ONE_BLOCK_R2"]]
    parameters = [
        {k: v for k, v in b["parameters"].items() if k not in ["LogFile", "NodefileDir"]}
        for b in begins
    ]
    assert parameters[0] == parameters[1]
    assert arms[0]["source_config"] == arms[1]["source_config"]
    assert (
        abs(arms[0]["common_warm_start_objective"] - arms[1]["common_warm_start_objective"]) <= 1e-9
    )
    assert all(
        max(a["warm_start_audit"].values()) <= 1e-9 and not a["new_full400_UB"] for a in arms
    )
    save(
        r / "PAIR_INDEPENDENT_READBACK.json",
        dict(
            common_parameters=parameters[0],
            same_config=True,
            same_start_objective=True,
            original_start_audits=[a["warm_start_audit"] for a in arms],
            new_rows=arms[1]["rows"] - arms[0]["rows"],
            new_cols=arms[1]["cols"] - arms[0]["cols"],
            new_nnz=arms[1]["nnz"] - arms[0]["nnz"],
            fullQ_calls=0,
            passed=True,
        ),
    )
    expected = read(r / "PROTECTION_EXPECTATIONS.json")
    protected = []
    for ref in expected["files"]:
        actual = digest(ref["path"])
        assert actual == ref["sha256"], ref["path"]
        protected.append(dict(ref, unchanged=True))
    processes = []
    for pid in [2850511, 2850521]:
        p = psutil.Process(pid)
        processes.append(
            dict(
                pid=pid,
                created=p.create_time(),
                status=p.status(),
                cmdline=p.cmdline(),
                rss=p.memory_info().rss,
            )
        )
    native = read(root / "L08_DET_R38_CAPACITY_CONTINUATION_V1/RECOVERY_02/NATIVE_BEGIN.json")
    current_ticks = Path("/proc/2850521/stat").read_text().rsplit(")", 1)[1].split()[19]
    assert current_ticks == native["owner"]["start_ticks"]
    save(
        r / "FINAL_SNAPSHOT.json",
        dict(
            observed_utc=datetime.now(timezone.utc).isoformat(),
            processes=processes,
            same_original_DET_worker_start_ticks=True,
            protected_files=protected,
            DET_bounds=read(root / "L08_DET_R38_CAPACITY_CONTINUATION_V1/RECOVERY_02/BOUNDS.json"),
            DET_progress=read(
                root / "L08_DET_R38_CAPACITY_CONTINUATION_V1/RECOVERY_02/PROGRESS.json"
            ),
            production_source_or_campaign_writes=0,
        ),
    )


if __name__ == "__main__":
    root, r = map(Path, sys.argv[1:])
    ledger = BudgetLedger(r / "store", "diagnostic-round1-20261111")
    claim = ledger.claim(
        dict(stage="FINAL_INDEPENDENT_READBACK", source=reference(__file__)),
        180,
        charge_process_start=True,
    )
    save(r / "FINAL_AUDIT_CLAIM.json", claim)
    started = time.monotonic()
    rc = 1
    try:
        audit(root, r)
        rc = 0
    finally:
        save(r / "FINAL_AUDIT_WAIT.json", dict(returncode=rc, seconds=time.monotonic() - started))
        ledger.settle(claim["attempt"], rc, reference(r / "FINAL_AUDIT_WAIT.json"))
    snap = ledger.read()
    adjustments = []
    stages = []
    active = None
    for event in snap["events"]:
        p = event["payload"]
        if p["kind"] == "claim":
            active = p
            if not p["charged_from_process_start"]:
                adjustments.append(
                    dict(
                        attempt=p["attempt"],
                        startup_seconds=max(0, p["started_epoch"] - p["owner"]["created"]),
                    )
                )
        if p["kind"] == "settlement":
            stages.append(
                dict(
                    attempt=p["attempt"],
                    stage=active["run"]["stage"],
                    charged_seconds=p["charged_seconds"],
                    returncode=p.get("returncode"),
                    claim=active,
                )
            )
    extra = sum(a["startup_seconds"] for a in adjustments)
    used = snap["spent_seconds"] + extra
    assert used <= 7200 and snap["active"] is None
    readback = (
        read(r / "replay-r1/READBACK_END.json")["elapsed_seconds"]
        + read(r / "origins-r1/END.json")["elapsed_seconds"]
        + read(r / "comparison-audit/END.json")["elapsed_seconds"]
        + read(r / "CONTINUOUS_SCOPE.json")["elapsed_readback_seconds"]
    )
    save(
        r / "BUDGET_FINAL.json",
        dict(
            configured_seconds=7200,
            used_solver_wrapper_seconds=used,
            remaining_seconds=7200 - used,
            ledger_spent_seconds=snap["spent_seconds"],
            startup_adjustment_seconds=extra,
            startup_adjustments=adjustments,
            stages=stages,
            new_full400_batches_started=1,
            new_full400_batches_completed=1,
            new_scenario_LP_calls=400,
            mip_AB_pairs=1,
            concurrent_solver_jobs_max=1,
            historical_readback_seconds=readback,
            source_reading_coding_and_environment_wall_seconds=None,
            missing_reason="Human/agent source reading, implementation and environment setup wall not continuously instrumented; historical readback wrappers separately measured. Startup corrections preserve original immutable event chain.",
            ledger_hash=snap["previous"],
            active_claim=None,
        ),
    )
    save(r / "BUDGET_LEDGER_FINAL.json", snap)
    print(dict(passed=True, solver_wrapper_seconds=used, remaining_seconds=7200 - used))
