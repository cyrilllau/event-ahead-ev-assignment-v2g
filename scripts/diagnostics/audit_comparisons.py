"""No-solve independent checks of DET pairing and newly selected archive cuts."""

from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.certificates import replay_cut
from ev_assignment_experiments.analysis.diagnostic import npz, read, reference, save, digest
from ev_assignment_experiments.analysis.diagnostic_replay import AROOT, BANK, FIRST, R45


def main(root, round_root, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    base = root / R45
    A = sp.load_npz(base / "INPUT_A.npz")
    B = sp.load_npz(base / "PROJECTED_LP_A.npz")
    a = npz(base / "INPUT_ARRAYS.npz")
    b = npz(base / "PROJECTED_LP_ARRAYS.npz")
    names = read(base / "INPUT_NAMES.json")["columns"]
    assert (A != B[: A.shape[0]]).nnz == 0
    for key in ["cost", "lo", "hi"]:
        assert np.array_equal(a[key], b[key])
    for key in ["b", "sense"]:
        assert np.array_equal(a[key], b[key][: A.shape[0]])
    support = sp.load_npz(base / "SUPPORT_CUTS.npz")
    rhs = npz(base / "SUPPORT_RHS.npz")["rhs"]
    assert (support != B[A.shape[0] :]).nnz == 0 and np.array_equal(rhs, b["b"][A.shape[0] :])
    assert np.all(b["sense"][A.shape[0] :] == ">")
    validity = read(base / "SUPPORT_VALIDITY.json")
    assert len(validity) == len(rhs) == 30
    assert all(
        v["pricing"]["closed_full_traversal"] and v["pricing"]["lower"] == rhs[i]
        for i, v in enumerate(validity)
    )
    original_binary = a["types"] == "B"
    rows = []
    for arm in ["E3_B0", "E3_R45"]:
        result = read(round_root / arm / "RESULT.json")
        x = npz(round_root / arm / "RAW.npz")["x"]
        c = a["cost"]
        sums = {
            prefix: float(x[[n.startswith(prefix + "[") for n in names]].sum())
            for prefix in ["J", "K", "selected", "E"]
        }
        route = np.array([n.startswith("route[") for n in names])
        tail = np.arange(len(x)) >= 60780
        short = tail & (c == 1)
        assert short.sum() == 30
        parts = dict(
            activation=25 * sums["selected"],
            departure=30 * sums["J"] + float(c[route] @ x[route]),
            occupancy=sums["J"] - sums["K"],
            procurement=0.02 * sums["E"],
            shortfall=float(c[short] @ x[short]),
            other_recourse=float(c[tail & ~short] @ x[tail & ~short]),
        )
        assert abs(sum(parts.values()) - result["primal_value"]) < 1e-7
        frac = np.minimum(abs(x - np.floor(x)), abs(np.ceil(x) - x)) > 1e-6
        fractions = {
            prefix: int(np.sum(frac & np.array([n.startswith(prefix + "[") for n in names])))
            for prefix in ["route", "start", "end", "window", "selected"]
        }
        rows.append(
            dict(
                arm=arm,
                components=parts,
                fractional_variables=fractions,
                fractional_original_binary=int(np.sum(frac & original_binary)),
                original_binary_count=int(original_binary.sum()),
                added_family_max_violation=float(np.max(rhs - support @ x)),
                native_original_binary_metric_needs_base_mask=arm == "E3_R45",
                result=reference(round_root / arm / "RESULT.json"),
            )
        )
    save(
        out / "E3_MATCHED.json",
        dict(
            common_matrix_objective_bounds_exact=True,
            added_rows=30,
            added_nnz=support.nnz,
            added_variables=0,
            R32_DET_screen_used=True,
            validity_domain="INTEGER_ONLY over DET R32-screened domain; strengthened relaxation",
            not_transferable_without_full_domain_validation=True,
            exact_lower_pricing_rhs_replayed=True,
            complete_hull_pricing_closed=False,
            validity=reference(base / "SUPPORT_VALIDITY.json"),
            rows=rows,
        ),
    )
    replay = round_root / "replay-r1"
    sources = read(replay / "ARCHIVE_INPUTS.json")["unique_source_cuts"]
    bysid = {
        s: sorted([r for r in sources if r["sid"] == s], key=lambda r: r["sha256"])
        for s in range(400)
    }
    checked = []
    first = npz(root / FIRST / "FIRST_ARRAYS.npz")
    for selected in read(round_root / "E2_SCAN/SCAN.json")["selected"]:
        sid = selected["sid"]
        initial = root / BANK / f"{sid:04d}/ALPHA.npz"
        offset = len(npz(initial)["alpha"]) if initial.exists() else 0
        index = selected["archive_row"] - offset
        if not 0 <= index < len(bysid[sid]):
            raise ValueError("Selected archive row needs another provenance mapping")
        ref = bysid[sid][index]
        assert digest(ref["path"]) == ref["sha256"]
        cut = npz(ref["path"])
        oracle = root / AROOT / "oracles" / f"{sid:04d}"
        W, T, data = (
            sp.load_npz(oracle / "W.npz"),
            sp.load_npz(oracle / "T.npz"),
            npz(oracle / "ARRAYS.npz"),
        )
        alpha, beta, _ = replay_cut(W, T, data, cut["Pi"])
        assert alpha == float(cut["alpha"]) and np.array_equal(beta, cut["beta"])
        archive = sp.load_npz(replay / "archive" / f"B_{sid:04d}.npz")
        assert np.array_equal(archive.getrow(selected["archive_row"]).toarray().ravel(), beta)
        assert alpha == selected["source_alpha"]
        box_contains_common = bool(
            np.all(data["xlo"] <= first["lo"]) and np.all(data["xhi"] >= first["hi"])
        )
        if not box_contains_common:
            raise ValueError("Newly used cut box does not contain common first-stage box")
        family = "ARO" if "ARO" in ref["path"] else "DRO"
        checked.append(
            dict(
                sid=sid,
                source=ref,
                source_method=family,
                global_coefficient_replay_exact=True,
                finite_box_contains_common=box_contains_common,
                continuous_valid=True,
            )
        )
    save(
        out / "E2_NEW_CUTS.json",
        dict(
            checked=len(checked),
            all_passed=True,
            sources=dict(Counter(r["source_method"] for r in checked)),
            cuts=checked,
        ),
    )
    save(out / "END.json", dict(elapsed_seconds=time.monotonic() - start, solver_calls=0))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
