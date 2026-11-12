"""One full archived-cut scan at an already solved continuous master point.

ROOT REPLAY ARM OUTPUT STORE. Re-add at most one strongest cut per scenario.
This is separation/build time and is charged to the round's existing ledger.
"""

from pathlib import Path
from fractions import Fraction
import hashlib
import math
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.certificates import row_audit
from ev_assignment.l08.theta import compensate, lift
from ev_assignment_experiments.analysis.diagnostic import (
    affine_bounds,
    digest,
    npz,
    read,
    reference,
    save,
    scenario_rows,
)
from ev_assignment_experiments.runtime.ledger import BudgetLedger


def scan(root, replay, arm, out, prepare=True):
    start = time.monotonic()
    base = Path(read(arm / "RESULT.json")["source_refs"][0]["path"]).parent
    A, a, names = (
        sp.load_npz(base / "A.npz"),
        npz(base / "ARRAYS.npz"),
        read(base / "NAMES.json"),
    )
    raw = npz(arm / "RAW.npz")["x"]
    x = raw[:60780].copy()
    block = scenario_rows(A, a, names, 60780, "DRO")
    stored = {
        s: (block["B"][block["sids"] == s], block["alpha"][block["sids"] == s])
        for s in block["theta"]
    }
    lifted = lift(A, a, x, 60780, 60780, block["theta"], stored)
    check = row_audit(A, a["rhs"], a["sense"], a["lo"], a["hi"], lifted)
    if max(check.values()) > 1e-9:
        raise ValueError(("Existing lift fails unchanged numeric audit", check))
    selected, summaries, all_lo, all_hi = [], [], np.zeros(400), np.zeros(400)
    extra, rhs, rownames = [], [], []
    for sid in range(400):
        B = sp.load_npz(replay / "archive" / f"B_{sid:04d}.npz")
        alpha = npz(replay / "archive" / f"ALPHA_{sid:04d}.npz")["alpha"]
        lo, hi = affine_bounds(B, alpha, x)
        best = int(np.argmax(lo))
        all_lo[sid], all_hi[sid] = max(0, float(lo[best])), max(0, float(hi.max()))
        theta = float(lifted[block["theta"][sid]]) if sid in block["theta"] else 0.0
        summaries.append(
            dict(
                sid=sid,
                rows=B.shape[0],
                theta=theta,
                archive_lower=all_lo[sid],
                archive_upper=all_hi[sid],
                max_violation_lower=float(lo.max() - theta),
                max_violation_upper=float(hi.max() - theta),
            )
        )
        if prepare and lo[best] - theta > 1e-6:
            if sid not in block["theta"]:
                raise ValueError("Positive archived support conflicts with universal-zero scenario")
            source = B.getrow(best)
            actual = source.copy()
            actual.data[abs(actual.data) < 1e-13] = 0
            actual.eliminate_zeros()
            safe, pr = compensate(
                float(alpha[best]), source, actual, a["lo"][:60780], a["hi"][:60780]
            )
            row = sp.hstack([-actual, sp.csr_matrix((1, A.shape[1] - 60780))], format="lil")
            row[0, block["theta"][sid]] = 1
            extra.append(row.tocsr())
            rhs.append(safe)
            key = hashlib.sha256(
                source.data.tobytes()
                + source.indices.tobytes()
                + np.asarray([alpha[best]]).tobytes()
            ).hexdigest()
            rownames.append(f"adaptive/{sid}/{key}")
            selected.append(
                dict(
                    sid=sid,
                    archive_row=best,
                    source_alpha=float(alpha[best]),
                    stored_alpha=safe,
                    nnz=actual.nnz,
                    **pr,
                )
            )
    # Complete eligible saved probability archive: no probability is dropped
    # because it currently assigns zero mass to an interesting scenario.
    active_ids = {n.rsplit("/", 1)[-1] for n in names["rows"] if "probability" in n}
    saved_probs = {}
    for ref in read(replay / "ARCHIVE_INPUTS.json")["evaluations"]:
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Eligible archive manifest changed")
        e = read(ref["path"])
        if "risk" not in e:
            continue
        cert = e["risk"]["certificate"]
        if digest(cert["path"]) != cert["sha256"]:
            raise ValueError("Saved risk certificate changed")
        if e["P_identity"] != read(replay / "E0.json")["probability_identity"]:
            raise ValueError("Probability set mismatch")
        if not e["risk"]["probability_check"]["passed"]:
            raise ValueError("Probability feasibility has not been certified")
        p = npz(cert["path"])["pstar"]
        if p.min() < 0 or sum(map(Fraction.from_float, p), Fraction(0)) != 1:
            raise ValueError("Invalid exact simplex")
        key = hashlib.sha256(p.tobytes()).hexdigest()
        saved_probs.setdefault(key, (p, cert))
    added_p = []
    for key, (p, ref) in sorted(saved_probs.items()):
        if key in active_ids or not prepare:
            continue
        k = 0
        while p[p > 0].min() * 2**k < 1e-12:
            k += 1
        scale = math.ldexp(1.0, k)
        if scale > 1e6:
            raise ValueError("Probability scaling is outside existing faithful-storage rule")
        row = sp.lil_matrix((1, A.shape[1]))
        row[0, 60780] = scale
        for sid, col in block["theta"].items():
            row[0, col] = -math.ldexp(float(p[sid]), k)
        extra.append(row.tocsr())
        rhs.append(0.0)
        rownames.append("adaptive_probability/" + key)
        added_p.append(dict(id=key, source=ref, power2=k))
    folder = out / "DRO"
    folder.mkdir()
    newA = sp.vstack([A, *extra], format="csr") if extra else A
    newa = dict(a, rhs=np.r_[a["rhs"], rhs], sense=np.r_[a["sense"], [">"] * len(extra)])
    sp.save_npz(folder / "A.npz", newA)
    np.savez_compressed(folder / "ARRAYS.npz", **newa)
    save(folder / "NAMES.json", dict(cols=names["cols"], rows=names["rows"] + rownames))
    np.savez_compressed(
        out / "POINT_AND_BOUNDS.npz",
        x=x,
        raw=raw,
        lifted=lifted,
        archive_lower=all_lo,
        archive_upper=all_hi,
    )
    save(
        out / "SCAN.json",
        dict(
            source_arm=reference(arm / "RESULT.json"),
            source_raw=reference(arm / "RAW.npz"),
            source_archive=reference(replay / "ARCHIVE_INPUTS.json"),
            raw_primal_residual=read(arm / "RESULT.json")["primal_residual"],
            lifted_residual=check,
            lifted_objective=float(np.asarray(a["cost"], np.longdouble) @ lifted),
            full_rows_scanned=sum(s["rows"] for s in summaries),
            scenarios=summaries,
            selected=selected,
            probability_archive_unique=len(saved_probs),
            probability_added=added_p,
            any_first_coordinate_changed=False,
            rows_added=len(extra),
            preparation_enabled=prepare,
            new_rows=newA.shape[0],
            new_nnz=newA.nnz,
            separation_build_seconds=time.monotonic() - start,
            solver_calls=0,
        ),
    )


if __name__ == "__main__":
    root, replay, arm, out, store = map(Path, sys.argv[1:6])
    out.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(store, "diagnostic-round1-20261111")
    claim = ledger.claim(dict(stage="E2_ARCHIVE_SCAN", source=reference(__file__)), 300.0)
    save(out / "CLAIM.json", claim)
    rc = 1
    try:
        scan(root, replay, arm, out, prepare="--scan-only" not in sys.argv)
        rc = 0
    finally:
        save(out / "WAIT.json", dict(returncode=rc, solver_calls=0))
        ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
        save(out / "BUDGET_AFTER.json", ledger.read())
