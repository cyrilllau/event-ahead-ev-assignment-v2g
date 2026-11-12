"""Independent no-solve follow-up to the round-one replay.

Pass explicit legacy root, completed replay directory, new output directory.
Uses only the installed package; never imports a historical launcher.
"""

from fractions import Fraction
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.envelope import enclosure
from ev_assignment.l08.theta import compensate
from ev_assignment_experiments.analysis.diagnostic import (
    affine_bounds,
    digest,
    fixed_master_risk,
    npz,
    read,
    read_linear_mps,
    reference,
    save,
)
from ev_assignment_experiments.analysis.diagnostic_replay import ACONT, BANK, DROOT, FIRST


def origins(root, replay, out):
    first = npz(root / FIRST / "FIRST_ARRAYS.npz")
    current = sp.load_npz(replay / "ARO/A.npz")
    current_a = npz(replay / "ARO/ARRAYS.npz")
    current_n = read(replay / "ARO/NAMES.json")
    points = read(replay / "E1_ENVELOPES.json")["points"]
    rows = []
    for iteration in [0, 11]:
        base = root / ACONT / f"ITER_{iteration:04d}"
        snapshot = read(base / "NATIVE_SNAPSHOT.json")
        entry = next(e for e in snapshot["entries"] if e["kind"] == "NATIVE_INCUMBENT_UNAUDITED")
        p = next(r for r in points if r["method"] == "ARO" and r["point"] == entry["point_id"])
        x = npz(replay / "points/ARO" / p["point"] / "POINT.npz")["x"]
        for ref in [snapshot["model"], entry["payload"]]:
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("Candidate origin source changed")
        native = npz(entry["payload"]["path"])[entry["array_key"]]
        if not np.array_equal(native, x):
            raise ValueError("Native incumbent is not this fully evaluated point")
        A, a, names = read_linear_mps(snapshot["model"]["path"])
        nr = A.shape[0]
        prefix_exact = (
            names["cols"] == current_n["cols"]
            and names["rows"] == current_n["rows"][:nr]
            and (A != current[:nr]).nnz == 0
            and np.array_equal(a["rhs"], current_a["rhs"][:nr])
        )
        if not prefix_exact:
            raise ValueError("Historical/current matrix ancestry differs")
        risk, _, _ = fixed_master_risk(A, a, names, x, 60780, "ARO", 90768)
        f = float(np.asarray(first["cost"], np.longdouble) @ x)
        bound = read(base / "END.json")["ObjBoundC"]["value"]
        if bound != snapshot["bound_observation"]["value"]:
            raise ValueError("Native bound source mismatch")
        q = p["true_risk"]
        rows.append(
            dict(
                method="ARO",
                iteration=iteration,
                point=p["point"],
                first_cost=f,
                A_then_risk=risk,
                A_now_risk=p["fixed_master_risk"],
                native_saved_eta=entry["eta"],
                same_call_native_LB=bound,
                surrogate_incumbent_interval=[f + risk[0], f + risk[1]],
                search_term_interval=[f + risk[0] - bound, f + risk[1] - bound],
                recourse_underestimate_interval=[q[0] - risk[1], q[1] - risk[0]],
                full_candidate_gap_interval=[f + q[0] - bound, f + q[1] - bound],
                candidate_audited_UB=f + q[1],
                prefix_coefficient_identity=prefix_exact,
                matrix=snapshot["model"],
                native_snapshot=reference(base / "NATIVE_SNAPSHOT.json"),
                native_end=reference(base / "END.json"),
                retrospective=True,
                interpretation="Same point/call decomposition; later cuts are not missing current-pool information",
            )
        )
        print("paired origin", iteration, rows[-1]["recourse_underestimate_interval"], flush=True)
    save(out / "A_THEN.json", rows)


def guards(root, replay, out):
    A = sp.load_npz(replay / "DRO/A.npz")
    a, names = npz(replay / "DRO/ARRAYS.npz"), read(replay / "DRO/NAMES.json")
    first = npz(root / FIRST / "FIRST_ARRAYS.npz")
    bank, new = {}, {}
    for folder in sorted((root / BANK).glob("[0-9][0-9][0-9][0-9]")):
        sid = int(folder.name)
        bank[sid] = (
            sp.load_npz(folder / "B.npz"),
            npz(folder / "ALPHA.npz")["alpha"],
            {v: i for i, v in enumerate(read(folder / "IDS.json"))},
        )
    paths = list((root / DROOT / "ATTEMPT_05/INCREMENTS").glob("*/MANIFEST.json"))
    paths.append(
        root
        / "L08_DRO400_THETA_MULTICUT_CACHE_AND_ADAPTIVE_V1/RESUME_01/INCREMENTS/R00001/MANIFEST.json"
    )
    for path in paths:
        for rec in read(path)["new_supports"]:
            new[rec["sid"], rec["id"]] = rec
    checked = []
    for i, name in enumerate(names["rows"]):
        if not name.startswith(("theta_support/", "adaptive/")):
            continue
        family, sid, key = name.split("/")
        sid = int(sid)
        actual = -A[i, :60780]
        if family == "theta_support":
            B, alpha, ids = bank[sid]
            j = ids[key]
            raw_beta, raw_alpha = B.getrow(j), float(alpha[j])
        else:
            rec = new[sid, key]
            ref = rec["source"]
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("Raw incremental support changed")
            z = npz(ref["path"])
            raw_beta, raw_alpha = sp.csr_matrix(z["beta"][None, :]), float(z["alpha"])
        rhs, proof = compensate(raw_alpha, raw_beta, actual, first["lo"], first["hi"])
        if rhs != a["rhs"][i]:
            raise ValueError("Actual stored row compensation mismatch")
        checked.append(
            dict(
                row=i,
                sid=sid,
                id=key,
                raw_alpha=raw_alpha,
                stored_alpha=rhs,
                raw_nnz=raw_beta.nnz,
                stored_nnz=actual.nnz,
                **proof,
            )
        )
    save(
        out / "STORAGE_GUARDS.json",
        dict(
            rows=checked,
            checked=len(checked),
            all_matched=True,
            max_alpha_loss=max(v["alpha_loss"] for v in checked),
            changed_coefficient_rows=sum(v["changed_coefficients"] > 0 for v in checked),
        ),
    )
    print("stored rows replayed", len(checked), flush=True)


def probability(root, replay, out):
    base = root / "L08_K4L4_CALIBRATION_AND_FORECAST_R29/05_DRO_MOMENTS"
    co = read(base / "COMPRESSION_COMPLETE.json")
    for file, key in [
        ("SUPPORT_UNIQUE.npy", "rows_sha"),
        ("SUPPORT_INVERSE.npy", "inverse_sha"),
        ("EFFECTIVE_INTERVALS.npz", "bounds_sha"),
    ]:
        if digest(base / file) != co[key]:
            raise ValueError("Probability matrix changed")
    dense = np.load(base / "SUPPORT_UNIQUE.npy")
    inv = np.load(base / "SUPPORT_INVERSE.npy")
    raw = np.load(base / "X_SUPPORT.npy", mmap_mode="r")
    bounds = npz(base / "EFFECTIVE_INTERVALS.npz")
    rb = npz(base / "CALIBRATED_RAW_BOUNDS_K4L4.npz")
    lower, upper = np.full(len(dense), -np.inf), np.full(len(dense), np.inf)
    np.maximum.at(lower, inv, rb["lower"])
    np.minimum.at(upper, inv, rb["upper"])
    assert np.array_equal(lower, bounds["lower"]) and np.array_equal(upper, bounds["upper"])
    for i in range(0, raw.shape[1], 2048):
        assert np.array_equal(raw[:, i : i + 2048].T, dense[inv[i : i + 2048]])
    scale = np.maximum(
        np.max(abs(dense), axis=1), np.maximum(np.maximum(abs(lower), abs(upper)), 1e-3)
    )
    Phi = sp.csr_matrix(dense)
    scaled = sp.csr_matrix(dense / scale[:, None])
    dmin, dmax = dense.min(axis=1), dense.max(axis=1)
    results = []
    for row in read(replay / "E1_ENVELOPES.json")["points"]:
        if row["method"] != "DRO":
            continue
        cert = npz(row["risk_certificate"]["path"])
        p = cert["pstar"]
        exact = sum(map(Fraction.from_float, p), Fraction(0)) == 1 and p.min() >= 0
        assert exact
        pl, pu = enclosure(Phi, p)
        sl, su = enclosure(scaled, p)
        raw_margin = float(
            min(np.min(np.maximum(pl, dmin) - lower), np.min(upper - np.minimum(pu, dmax)))
        )
        scaled_margin = float(
            min(
                np.min(np.maximum(sl, dmin / scale) - lower / scale),
                np.min(upper / scale - np.minimum(su, dmax / scale)),
            )
        )
        assert raw_margin >= 0 and scaled_margin >= 0
        z = npz(replay / "points/DRO" / row["point"] / "POINT.npz")
        bz = npz(replay / "points/DRO" / row["point"] / "ENVELOPES.npz")
        point_result = dict(
            point=row["point"],
            raw_rows=raw.shape[1],
            effective_rows=len(dense),
            exact_simplex=True,
            raw_margin=raw_margin,
            scaled_margin=scaled_margin,
            risks={},
        )
        ld = np.longdouble
        eps = np.finfo(ld).eps
        bp, bm = np.maximum(cert["plus"], 0), np.maximum(cert["minus"], 0)
        wp, wm = (
            np.asarray(bp, ld) / np.asarray(scale, ld),
            np.asarray(bm, ld) / np.asarray(scale, ld),
        )
        plane_low, _ = enclosure(Phi.T, wp - wm)
        plane_low -= abs(Phi.T).astype(ld) @ (4 * eps * (abs(wp) + abs(wm)))
        b0 = ld(cert["beta0"])
        plane_low += b0
        plane_low -= eps * (abs(b0) + abs(plane_low))
        terms = np.asarray(upper, ld) * wp - np.asarray(lower, ld) * wm
        err = 8 * eps * (abs(upper * wp) + abs(lower * wm))
        for label, ql, qu in [
            ("Q", z["qL"], z["qU"]),
            ("active", z["active_lower"], z["active_upper"]),
            ("archive", bz["archive_lower"], bz["archive_upper"]),
        ]:
            low, _ = affine_bounds(sp.csr_matrix(p[None, :]), [0], ql)
            shift = max(ld(0), np.max(np.asarray(qu, ld) - plane_low))
            newb = np.nextafter(float(b0 + shift), np.inf)
            ru = ld(newb) + sum(terms) + sum(err) + (8 * len(terms) + 8) * eps * sum(abs(terms))
            point_result["risks"][label] = [float(low[0]), float(np.nextafter(float(ru), np.inf))]
        results.append(point_result)
    save(
        out / "PROBABILITY_REPLAY.json",
        dict(
            points=results,
            compression_replayed_all_raw=True,
            inputs=[
                reference(base / n)
                for n in [
                    "COMPRESSION_COMPLETE.json",
                    "SUPPORT_UNIQUE.npy",
                    "SUPPORT_INVERSE.npy",
                    "EFFECTIVE_INTERVALS.npz",
                    "CALIBRATED_RAW_BOUNDS_K4L4.npz",
                    "X_SUPPORT.npy",
                ]
            ],
            solver_calls=0,
            upper_from="saved dual transferred with full400 intercept correction; not feasible-p upper",
        ),
    )
    print("original probability set and transferred dual bounds verified", flush=True)


if __name__ == "__main__":
    root, replay, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    origins(root, replay, out)
    guards(root, replay, out)
    probability(root, replay, out)
    save(out / "END.json", dict(solver_calls=0, elapsed_seconds=time.monotonic() - start))
