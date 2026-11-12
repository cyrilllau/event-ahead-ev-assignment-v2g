"""Round-one L08 evidence adapter; reads the supplied legacy root only.

Usage: python -m ev_assignment_experiments.analysis.diagnostic_replay ROOT OUT
The output directory must be new. No native optimizer is called.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.certificates import row_audit, replay_cut
from .diagnostic import (
    affine_bounds,
    digest,
    fixed_master_risk,
    matrix_stats,
    npz,
    read,
    read_linear_mps,
    reference,
    save,
    scenario_rows,
)

DROOT = "L08_DRO400_THETA_MULTICUT_REPAIR02_NATIVE_RESUME_V1"
AROOT = "L08_ARO400_SHARED_INTEGER_AUTONOMOUS_V1"
ACONT = "L08_ARO_DRO_GAP1_ACCELERATION_V1/MEMORY_EXIT_REPAIR_V1/CONTINUE_01/ARO"
FIRST = "L08_DRO_R41_NESTED_BENDERS_ROOT/REPAIR_01/identity"
BANK = "L08_DRO400_CACHED_NONNEGATIVE_ENVELOPE_PILOT_V1/BANK"
R45 = "L08_DET_R45_VEHICLE_HULL_PROJECTED_CUTS/INFRASTRUCTURE_REPAIR_01"
SCIENCE = "c1fe0c77564e0b350782b0375447b2f637f2e5241871b28601a68cf6aa543765"


def records(evaluation):
    rec = evaluation["records"]
    if isinstance(rec, dict):
        if digest(rec["path"]) != rec["sha256"]:
            raise ValueError("Record manifest hash mismatch")
        rec = read(rec["path"])
    if sorted(r["sid"] for r in rec) != list(range(400)):
        raise ValueError("Not a complete full400 evaluation")
    return sorted(rec, key=lambda r: r["sid"])


def selected_points(root, ds, ars):
    dro = root / DROOT / "ATTEMPT_04/evaluations"
    dpaths = [
        Path(ds["best"]["Q"]["path"]).parent / "COMPLETE.json",
        Path(ds["evaluation"]["Q"]["path"]).parent / "COMPLETE.json",
        min(dro.glob("*/COMPLETE.json"), key=lambda p: p.stat().st_mtime_ns),
    ]
    aids = [ars["best_plan_id"], ars["history"][-1]["point_id"], ars["history"][0]["point_id"]]
    apaths = []
    for pid in aids:
        options = [
            root / ACONT / "evaluations" / pid / "COMPLETE.json",
            root / AROOT / "evaluations" / pid / "COMPLETE.json",
        ]
        apaths.append(next(p for p in options if p.exists()))
    result, seen = [], set()
    for method, paths in [("DRO", dpaths), ("ARO", apaths)]:
        for role, path in zip(["best", "latest_complete", "stalled"], paths):
            e = read(path)
            rec = records(e)
            first_record = next(r for r in rec if "primal" in r)
            x = npz(first_record["primal"])["x"]
            point = hashlib.sha256(x.tobytes()).hexdigest()
            if point != e["point_sha"]:
                raise ValueError("Full point digest mismatch")
            if (method, point) in seen:
                continue
            seen.add((method, point))
            result.append(
                dict(
                    method=method,
                    role=role,
                    point=point,
                    x=x,
                    evaluation=e,
                    records=rec,
                    source=reference(path),
                )
            )
    return result


def run(root, out):
    started = time.monotonic()
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.is_relative_to(root):
        raise ValueError("Diagnostic output must be outside the legacy project")
    out.mkdir(parents=True, exist_ok=False)
    ds = read(root / DROOT / "ATTEMPT_05/STATE.json")
    ars = read(root / ACONT / "STATE.json")
    if ds["science"] != SCIENCE:
        raise ValueError("Scientific identity changed")
    first_a = npz(root / FIRST / "FIRST_ARRAYS.npz")
    first_A = sp.load_npz(root / FIRST / "FIRST_A.npz")
    first_names = read(root / FIRST / "FIRST_NAMES.json")
    nfirst, nrows = len(first_names), first_A.shape[0]
    points = selected_points(root, ds, ars)
    snap = read(root / DROOT / "ATTEMPT_05/LATEST_SNAPSHOT.json")
    # The pointer schema is recorded; choose its final immutable directory.
    snapshots = sorted((root / DROOT / "ATTEMPT_05/SNAPSHOTS").glob("FINAL_*_R00008"))
    if len(snapshots) != 1:
        raise ValueError("Ambiguous current snapshot")
    dpath = snapshots[0]
    da = npz(dpath / "ARRAYS.npz")
    dA = sp.load_npz(dpath / "A.npz")
    dn = read(dpath / "NAMES.json")
    print("E0: reading actual ARO checkpoint without Gurobi", flush=True)
    aroref = ars["model"]
    if digest(aroref["path"]) != aroref["sha256"]:
        raise ValueError("ARO model hash mismatch")
    aA, aa, an = read_linear_mps(aroref["path"])
    models = {"DRO": (dA, da, dn), "ARO": (aA, aa, an)}
    identities, matrices = {}, {}
    for method, (A, a, names) in models.items():
        if names["cols"][:nfirst] != first_names or A.shape[1] < nfirst:
            raise ValueError("First variable mapping differs")
        if (A[:nrows, :nfirst] != first_A).nnz:
            raise ValueError("Common first-stage matrix differs")
        for key in ["cost", "lo", "hi", "vtype"]:
            if not np.array_equal(a[key][:nfirst], first_a[key]):
                raise ValueError("Common first-stage arrays differ: " + key)
        if not np.array_equal(a["rhs"][:nrows], first_a["rhs"]):
            raise ValueError("Common first-stage RHS differs")
        matrices[method] = matrix_stats(A, a, names, nrows)
        folder = out / method
        folder.mkdir()
        sp.save_npz(folder / "A.npz", A)
        np.savez_compressed(folder / "ARRAYS.npz", **a)
        save(folder / "NAMES.json", names)
        identities[method] = dict(
            matrix=reference(folder / "A.npz"),
            arrays=reference(folder / "ARRAYS.npz"),
            names=reference(folder / "NAMES.json"),
        )
    save(
        out / "E0.json",
        dict(
            science=SCIENCE,
            probability_identity=ds["P_identity"],
            first_mapping=reference(root / FIRST / "FIRST_NAMES.json"),
            first_matrix=reference(root / FIRST / "FIRST_A.npz"),
            common_first_identity=True,
            matrices=matrices,
            model_refs=identities,
            dro_snapshot_pointer=snap,
            original_dro_snapshot=reference(dpath / "IDENTITY.json"),
            original_aro_model=aroref,
            r45_domain_text=reference(root / R45 / "VEHICLE_DOMAIN_COMPLETENESS.md"),
            elapsed_readback_seconds=time.monotonic() - started,
        ),
    )
    print("E0 complete", matrices, flush=True)
    point_rows = []
    for p in points:
        folder = out / "points" / p["method"] / p["point"]
        folder.mkdir(parents=True)
        x = p["x"]
        ql = np.asarray([r["qL"] for r in p["records"]])
        qu = np.asarray([r["qU_audited"] for r in p["records"]])
        A, a, names = models[p["method"]]
        fixed, (alo, ahi), counts = fixed_master_risk(A, a, names, x, nfirst, p["method"], nrows)
        p.update(qL=ql, qU=qu, active_low=alo, active_high=ahi)
        np.savez_compressed(
            folder / "POINT.npz",
            x=x,
            qL=ql,
            qU=qu,
            active_lower=np.asarray(alo, float),
            active_upper=np.asarray(ahi, float),
        )
        row = dict(
            method=p["method"],
            role=p["role"],
            point=p["point"],
            source=p["source"],
            first_cost=float(np.asarray(first_a["cost"], np.longdouble) @ x),
            point_file=reference(folder / "POINT.npz"),
            fixed_master_risk=fixed,
            families=counts,
            first_stage_residual=row_audit(
                first_A, first_a["rhs"], first_a["sense"], first_a["lo"], first_a["hi"], x
            ),
            full400=True,
            evidence_grade="NUMERICALLY_AUDITED_ONLY",
            true_risk=(
                [p["evaluation"]["risk"]["risk_L"], p["evaluation"]["risk"]["risk_U"]]
                if p["method"] == "DRO"
                else [float(ql.max()), float(qu.max())]
            ),
        )
        if p["method"] == "DRO":
            ref = p["evaluation"]["risk"]["certificate"]
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("Risk certificate changed")
            cert = npz(ref["path"])
            row["risk_certificate"] = ref
            p["probability"] = cert["pstar"]
            # A feasible p supplies ONLY a risk lower bound. Q's audited upper
            # also bounds every valid underestimator at this same point.
            pl, _ = affine_bounds(sp.csr_matrix(cert["pstar"][None, :]), [0], alo)
            row["active_risk_interval"] = [float(pl[0]), row["true_risk"][1]]
        else:
            row["active_risk_interval"] = [float(alo.max()), float(ahi.max())]
        point_rows.append(row)
    save(out / "E1_ACTIVE.json", point_rows)
    print(
        "E1 active envelopes",
        [(r["method"], r["role"], r["fixed_master_risk"], r["true_risk"]) for r in point_rows],
        flush=True,
    )
    # Freeze the eligible archive before any new optimization. Include the
    # certified cached bank and subsequent full-domain COMPLETE records only.
    # Raw integer-pruned historical evaluations are deliberately ineligible.
    files = list((root / DROOT).glob("ATTEMPT_*/evaluations/*/COMPLETE.json"))
    for prefix in [
        AROOT,
        ACONT,
        "L08_DRO400_AGGREGATED_CUTS_INTEGER_REPAIR01_FULL_CYCLE_V1/RESUME_01",
        "L08_DRO400_THETA_MULTICUT_CACHE_AND_ADAPTIVE_V1/RESUME_01",
    ]:
        files.extend((root / prefix / "evaluations").glob("*/COMPLETE.json"))
    cut_refs, archive_sources, exclusions = {}, [], []
    aro_science = digest(root / AROOT / "SOURCE_LOCK.json")
    for path in sorted(set(files)):
        e = read(path)
        ok = e.get("science") == SCIENCE or e.get("scientific_id") == aro_science
        if not ok or not e.get("full400"):
            exclusions.append(
                dict(path=str(path), reason="scientific identity/full400 not admitted")
            )
            continue
        archive_sources.append(reference(path))
        for r in records(e):
            if r.get("universal_zero"):
                continue
            if not r.get("global_cut_valid"):
                exclusions.append(
                    dict(path=str(path), sid=r["sid"], reason="no global cut evidence")
                )
                continue
            ref = next(v for v in r["sources"] if v["path"] == r["cut"])
            cut_refs.setdefault((r["sid"], ref["sha256"]), ref)
    # Certified bank contains the older compatible history, including cuts
    # whose original COMPLETE manifest is outside these recent directories.
    banks, bank_sources = {}, []
    for folder in sorted((root / BANK).glob("[0-9][0-9][0-9][0-9]")):
        sid = int(folder.name)
        bb = sp.load_npz(folder / "B.npz")
        az = npz(folder / "ALPHA.npz")
        alpha = az["alpha"]
        banks[sid] = ([bb], [alpha])
        for name in ["B.npz", "ALPHA.npz", "SOURCE_RECORDS.json", "DOMAIN_PROOF.json"]:
            bank_sources.append(reference(folder / name))
    # Read each unique certified cut once; retain exact doubles without drops.
    print("E1 archive unique source payloads", len(cut_refs), flush=True)
    for k, ((sid, sha), ref) in enumerate(sorted(cut_refs.items())):
        if digest(ref["path"]) != sha:
            raise ValueError("Archive cut hash changed")
        z = npz(ref["path"])
        if z["beta"].shape != (nfirst,):
            raise ValueError("Archive coordinate mismatch")
        bs, al = banks.setdefault(sid, ([], []))
        bs.append(sp.csr_matrix(z["beta"][None, :]))
        al.append(np.asarray([float(z["alpha"])]))
        if k % 2000 == 0:
            print("archive", k, flush=True)
    save(
        out / "ARCHIVE_INPUTS.json",
        dict(
            evaluations=archive_sources,
            bank_files=bank_sources,
            unique_source_cuts=[dict(sid=s, **v) for (s, _), v in cut_refs.items()],
            excluded=exclusions,
            validity="inherited full-domain finite-box certificates; not freshly proved for every historical cut",
        ),
    )
    # Union with current actual rows ensures A_now is truly a subset of B_all.
    for method, (A, a, names) in models.items():
        block = scenario_rows(A, a, names, nfirst, method)
        for sid in sorted(set(block["sids"])):
            keep = block["sids"] == sid
            bs, al = banks.setdefault(int(sid), ([], []))
            bs.append(block["B"][keep])
            al.append(block["alpha"][keep])
    archive_counts = {}
    for p in points:
        p["archive_low"] = np.zeros(400, np.longdouble)
        p["archive_high"] = np.zeros(400, np.longdouble)
    (out / "archive").mkdir()
    for sid, (bs, al) in sorted(banks.items()):
        B, alpha = sp.vstack(bs, format="csr"), np.concatenate(al)
        sp.save_npz(out / "archive" / f"B_{sid:04d}.npz", B)
        np.savez_compressed(out / "archive" / f"ALPHA_{sid:04d}.npz", alpha=alpha)
        archive_counts[sid] = B.shape[0]
        for p in points:
            lo, hi = affine_bounds(B, alpha, p["x"])
            p["archive_low"][sid] = max(0, lo.max(initial=0))
            p["archive_high"][sid] = max(0, hi.max(initial=0))
    for p, row in zip(points, point_rows):
        blo, bhi = p["archive_low"], p["archive_high"]
        if p["method"] == "ARO":
            br = [float(blo.max()), float(bhi.max())]
        else:
            pl, _ = affine_bounds(sp.csr_matrix(p["probability"][None, :]), [0], blo)
            br = [float(pl[0]), row["true_risk"][1]]
        row["archive_risk_interval"] = br
        row["max_active_above_Q_upper"] = float(np.max(p["active_low"] - p["qU"]))
        row["max_archive_above_Q_upper"] = float(np.max(blo - p["qU"]))
        row["max_scenario_Q_upper_minus_archive_lower"] = float(np.max(p["qU"] - blo))
        row["max_scenario_archive_minus_active"] = float(np.max(bhi - p["active_low"]))
        np.savez_compressed(
            out / "points" / p["method"] / p["point"] / "ENVELOPES.npz",
            archive_lower=np.asarray(blo, float),
            archive_upper=np.asarray(bhi, float),
        )
    save(
        out / "E1_ENVELOPES.json",
        dict(
            points=point_rows,
            archive_counts=archive_counts,
            archive_inputs=reference(out / "ARCHIVE_INPUTS.json"),
        ),
    )
    print("E1 archive envelopes complete", flush=True)
    # Independently replay raw dual coefficients at the six selected anchors.
    # Archive-wide contact coverage is explicitly not claimed.
    contacts, suspicious = [], []
    for sid in range(400):
        nonzero = [p for p in points if not p["records"][sid].get("universal_zero")]
        if not nonzero:
            continue
        base = root / AROOT / "oracles" / f"{sid:04d}"
        W, T, data = (
            sp.load_npz(base / "W.npz"),
            sp.load_npz(base / "T.npz"),
            npz(base / "ARRAYS.npz"),
        )
        for p in nonzero:
            r = p["records"][sid]
            for ref in r["sources"]:
                if digest(ref["path"]) != ref["sha256"]:
                    raise ValueError("Selected anchor source hash changed")
            z, cut = npz(r["primal"]), npz(r["cut"])
            if not np.array_equal(z["x"], p["x"]):
                raise ValueError("Selected anchor changed")
            alpha, beta, _ = replay_cut(W, T, data, z["Pi"])
            lo, hi = affine_bounds(
                sp.csr_matrix(cut["beta"][None, :]), [float(cut["alpha"])], z["x"]
            )
            residual = row_audit(
                W, data["h"] - T @ z["x"], data["sense"], data["lo"], data["hi"], z["y"]
            )
            primal = float(np.asarray(data["c"], np.longdouble) @ np.asarray(z["y"], np.longdouble))
            c = dict(
                method=p["method"],
                point=p["point"],
                sid=sid,
                alpha=float(cut["alpha"]),
                nnz=int(np.count_nonzero(cut["beta"])),
                raw_contact=[float(lo[0]), float(hi[0])],
                qL=r["qL"],
                qU=r["qU_audited"],
                primal_objective=primal,
                residual=residual,
                coefficient_replay_exact=bool(
                    alpha == float(cut["alpha"]) and np.array_equal(beta, cut["beta"])
                ),
                upper_minus_contact=float(r["qU_audited"] - lo[0]),
                contact_minus_saved_qL=float(hi[0] - r["qL"]),
                strict_upper_certified=False,
                cut=reference(r["cut"]),
            )
            if (
                not c["coefficient_replay_exact"]
                or max(residual.values()) > 1e-8
                or lo[0] > r["qU_audited"] + 1e-7
            ):
                suspicious.append(c)
            contacts.append(c)
        if sid % 40 == 0:
            print("contact audit scenario", sid, flush=True)
    save(
        out / "CUT_CONTACT.json",
        dict(
            contacts=contacts,
            suspicious=suspicious,
            audited_count=len(contacts),
            archive_wide_contact=False,
        ),
    )
    save(
        out / "READBACK_END.json",
        dict(
            elapsed_seconds=time.monotonic() - started,
            solver_calls=0,
            point_count=len(points),
            source_root=str(root),
            output=str(out),
        ),
    )
    print("REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
