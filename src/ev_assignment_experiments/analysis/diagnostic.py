"""Solver-free, full-coordinate diagnostics for explicitly supplied L08 evidence.

No legacy launcher is imported. Intervals concern the given point, never an
unobserved optimum. A replay is not a new global cut-validity certificate.
"""

from __future__ import annotations

from array import array
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp

from ev_assignment.l08.envelope import enclosure


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reference(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    """Write once: a new source/run must not replace a prior measurement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def affine_bounds(B, alpha, x):
    if B.shape[1] != len(x) or B.shape[0] != len(alpha):
        raise ValueError("Full-coordinate mapping mismatch")
    low, high = enclosure(B, x)
    aa = np.asarray(alpha, np.longdouble)
    error = 4 * np.finfo(np.longdouble).eps * (abs(low) + abs(high) + abs(aa))
    return (np.nextafter(low + aa - error, -np.inf), np.nextafter(high + aa + error, np.inf))


def subtract_intervals(left, right, *, monotone=False):
    low, high = left[0] - right[1], left[1] - right[0]
    if monotone:
        if high < 0:
            raise ValueError("Negative upper bound contradicts claimed monotonicity")
        low = max(0.0, low)
    return [float(low), float(high)]


def scenario_rows(A, data, names, nfirst, method):
    """Read the actually stored coefficients, including storage compensation."""
    rows, sids, scales = [], [], []
    theta = {
        int(n.rsplit("_", 1)[1]): j
        for j, n in enumerate(names["cols"])
        if n.startswith("THETA_SCENARIO_")
    }
    for i, name in enumerate(names["rows"]):
        if name.startswith(("theta_support/", "adaptive/", "aro_support/")):
            sid = int(name.split("/")[1])
            expected = theta[sid] if method == "DRO" else nfirst
            tail = A.getrow(i)[:, nfirst:].tocoo()
            if (
                tail.nnz != 1
                or tail.col[0] + nfirst != expected
                or not (
                    (data["sense"][i] == ">" and tail.data[0] > 0)
                    or (data["sense"][i] == "<" and tail.data[0] < 0)
                )
            ):
                raise ValueError("Unsupported scenario row structure: " + name)
            rows.append(i)
            sids.append(sid)
            scales.append(1 / tail.data[0])
    return dict(
        rows=np.asarray(rows, int),
        sids=np.asarray(sids, int),
        B=sp.diags(scales) @ -A[rows, :nfirst],
        alpha=data["rhs"][rows] * scales,
        theta=theta,
    )


def envelope(block, x, size=400):
    low, high = affine_bounds(block["B"], block["alpha"], x)
    lower, upper = np.zeros(size, np.longdouble), np.zeros(size, np.longdouble)
    np.maximum.at(lower, block["sids"], low)
    np.maximum.at(upper, block["sids"], high)
    return lower, upper


def fixed_master_risk(A, data, names, x, nfirst, method, first_rows):
    """Exact monotone epigraph minimization at fixed x, with outward dot bounds.

    Reject any unrecognized coupling rather than silently treating a feasible
    overfilled rho as the fixed-point minimum.
    """
    block = scenario_rows(A, data, names, nfirst, method)
    qlo, qhi = envelope(block, x)
    lower = np.zeros(A.shape[1], np.longdouble)
    upper = lower.copy()
    lower[:nfirst] = upper[:nfirst] = x
    for sid, col in block["theta"].items():
        lower[col], upper[col] = qlo[sid], qhi[sid]
    support_set = set(block["rows"])
    rlo = rhi = np.longdouble(max(0, data["lo"][nfirst]))
    counts = dict(scenario=len(support_set), probability=0, aggregate=0, floor=0)
    for i in range(first_rows, A.shape[0]):
        if i in support_set:
            if method == "ARO":
                rlo, rhi = max(rlo, qlo.max()), max(rhi, qhi.max())
            continue
        row = A.getrow(i)
        coeff = float(row[0, nfirst])
        if data["sense"][i] != ">" or coeff <= 0:
            raise ValueError("Unknown fixed-risk row: " + names["rows"][i])
        tail = row[:, nfirst + 1 :]
        if tail.nnz and (np.max(tail.data) > 0 or np.any(row.indices < nfirst)):
            raise ValueError("Non-monotone or mixed probability row")
        other = row.copy()
        other.data[other.indices == nfirst] = 0
        # Every auxiliary coefficient is nonpositive; high theta gives high rho.
        lo, _ = affine_bounds(-other / coeff, [data["rhs"][i] / coeff], lower)
        _, hi = affine_bounds(-other / coeff, [data["rhs"][i] / coeff], upper)
        rlo, rhi = max(rlo, lo[0]), max(rhi, hi[0])
        key = "probability" if tail.nnz else ("aggregate" if row[:, :nfirst].nnz else "floor")
        counts[key] += 1
    return [float(rlo), float(rhi)], (qlo, qhi), counts


def matrix_stats(A, data, names, first_rows):
    families = {}
    for i, name in enumerate(names["rows"]):
        if i < first_rows:
            family = "first_stage"
        elif name.startswith(("theta_support/", "adaptive/", "aro_support/")):
            family = "scenario_cut"
        elif "probability" in name:
            family = "probability_layer"
        else:
            family = "other_requires_scope_review"
        group = families.setdefault(family, dict(rows=0, nnz=0))
        group["rows"] += 1
        group["nnz"] += int(A.indptr[i + 1] - A.indptr[i])
    values = abs(A.data[A.data != 0])
    return dict(
        rows=A.shape[0],
        cols=A.shape[1],
        nnz=A.nnz,
        binary_count=int(np.sum(data["vtype"] == "B")),
        families=families,
        coefficient_min=float(values.min()),
        coefficient_max=float(values.max()),
    )


def read_linear_mps(path):
    """Strict linear MPS reader for archive inspection without a solver/license.

    Fail closed on extensions. Preserve file variable order and every real
    coefficient; no integer rounding or scenario pruning.
    """
    rows, senses, rowmap = [], [], {}
    cols, colmap, costs, lows, highs, types = [], {}, [], [], [], []
    ii, jj, vv = array("i"), array("i"), array("d")
    rhs, section, integer, objective, objcon = {}, None, False, None, 0.0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("*"):
                continue
            parts = line.split()
            if not line[0].isspace():
                section = parts[0]
                if section not in {"NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"}:
                    raise ValueError("Unsupported MPS section: " + section)
                continue
            if section == "ROWS":
                kind, name = parts
                if kind == "N":
                    if objective is not None:
                        raise ValueError("Multiple MPS objectives")
                    objective = name
                else:
                    rowmap[name] = len(rows)
                    rows.append(name)
                    senses.append({"E": "=", "L": "<", "G": ">"}[kind])
            elif section == "COLUMNS":
                if "'MARKER'" in parts:
                    integer = "'INTORG'" in parts
                    continue
                col = parts[0]
                if col not in colmap:
                    colmap[col] = len(cols)
                    cols.append(col)
                    costs.append(0.0)
                    lows.append(0.0)
                    highs.append(float("inf"))
                    types.append("I" if integer else "C")
                j = colmap[col]
                for k in range(1, len(parts), 2):
                    row, value = parts[k], float(parts[k + 1])
                    if row == objective:
                        costs[j] += value
                    else:
                        ii.append(rowmap[row])
                        jj.append(j)
                        vv.append(value)
            elif section == "RHS":
                for k in range(1, len(parts), 2):
                    row, value = parts[k], float(parts[k + 1])
                    if row == objective:
                        objcon = -value
                    else:
                        if row in rhs:
                            raise ValueError("Duplicate MPS RHS")
                        rhs[row] = value
            elif section == "BOUNDS":
                kind, _, col = parts[:3]
                j = colmap[col]
                value = float(parts[3]) if len(parts) > 3 else None
                if kind in {"LO", "LI"}:
                    lows[j] = value
                elif kind in {"UP", "UI"}:
                    highs[j] = value
                elif kind == "FX":
                    lows[j] = highs[j] = value
                elif kind == "BV":
                    types[j], lows[j], highs[j] = "B", 0.0, 1.0
                elif kind == "FR":
                    lows[j], highs[j] = -float("inf"), float("inf")
                elif kind == "MI":
                    lows[j] = -float("inf")
                elif kind == "PL":
                    highs[j] = float("inf")
                else:
                    raise ValueError("Unsupported MPS bound: " + kind)
    for j in range(len(cols)):
        if types[j] == "I" and lows[j] == 0 and highs[j] == 1:
            types[j] = "B"
    A = sp.coo_matrix(
        (np.asarray(vv), (np.asarray(ii), np.asarray(jj))), shape=(len(rows), len(cols))
    ).tocsr()
    data = dict(
        cost=np.asarray(costs),
        lo=np.asarray(lows),
        hi=np.asarray(highs),
        vtype=np.asarray(types),
        rhs=np.asarray([rhs.get(r, 0.0) for r in rows]),
        sense=np.asarray(senses),
        objcon=np.asarray(objcon),
    )
    return A, data, dict(rows=rows, cols=cols)
