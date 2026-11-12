"""Sparse cache envelope; no native optimizer in this module."""

import hashlib, json, math
from fractions import Fraction as F
import numpy as np
from scipy import sparse as sp

ld = np.longdouble
eps = np.finfo(ld).eps


def digest(a, b):
    return hashlib.sha256(
        np.asarray([a], dtype=float).tobytes() + np.asarray(b, dtype=float).tobytes()
    ).hexdigest()


def simplex(p):
    p = np.asarray(p, float)
    if (
        p.ndim != 1
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or sum(map(F.from_float, p), F(0)) != 1
    ):
        raise ValueError("INVALID_COMPLETE_PROBABILITY")
    return p


def enclosure(A, x):
    A = A.astype(ld, copy=False)
    x = np.asarray(x, ld)
    val = A @ x
    mag = abs(A) @ abs(x)
    n = A.shape[1]
    g = ld(2 * n + 16) * eps / (1 - ld(2 * n + 16) * eps)
    err = g * mag + np.finfo(ld).tiny
    return np.nextafter(val - err, -np.inf), np.nextafter(val + err, np.inf)


def protect(alpha, beta, p, lo, hi, stored=None):
    """Rows of beta are selected supports, one complete p. Enclose source products."""
    p = simplex(p)
    lo = np.asarray(lo, ld)
    hi = np.asarray(hi, ld)
    if (
        beta.shape != (len(p), len(lo))
        or len(lo) != len(hi)
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(lo > hi)
    ):
        raise ValueError("INVALID_FINITE_BOX_OR_DIMENSION")
    bl, bu = enclosure(beta.T, p)
    al, au = enclosure(sp.csr_matrix(np.asarray(alpha, float)[None, :]), p)
    bh = np.asarray((bl + bu) / 2, float) if stored is None else np.asarray(stored, float).copy()
    if stored is None:
        bh[abs(bh) < 1e-13] = 0
    if not np.isfinite(bh).all():
        raise ValueError("NONFINITE_STORED_ROW")
    dl = np.nextafter(np.asarray(bh, ld) - bu, -np.inf)
    du = np.nextafter(np.asarray(bh, ld) - bl, np.inf)
    terms = np.maximum.reduce([dl * lo, dl * hi, du * lo, du * hi])
    E = max(
        ld(0), np.sum(terms, dtype=ld) + (8 * len(lo) + 32) * eps * np.sum(abs(terms), dtype=ld)
    )
    safe = al[0] - E - 32 * eps * (abs(al[0]) + E + 1)
    ah = float(np.nextafter(float(safe), -np.inf))
    return (
        ah,
        bh,
        dict(
            compensation=float(E),
            alpha_lower=float(al[0]),
            gamma_kernel="longdouble_sparse_outward",
            drop_before_protection=True,
        ),
    )


class Bank:
    def __init__(self, blocks, probs, pids, lo, hi):
        self.blocks = blocks
        self.p = np.asarray(probs, float)
        if self.p.ndim != 2 or self.p.shape[1] != len(blocks):
            raise ValueError("PROBABILITY_DIMENSION")
        for p in self.p:
            simplex(p)
        if len(pids) != len(self.p):
            raise ValueError("P_ID_DIMENSION")
        order = np.argsort(pids, kind="stable")
        self.p = self.p[order]
        self.pids = [pids[i] for i in order]
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        for b in blocks:
            if (
                b["B"].shape[1] != len(lo)
                or len(b["alpha"]) != b["B"].shape[0]
                or not np.isfinite(b["B"].data).all()
                or not np.isfinite(b["alpha"]).all()
            ):
                raise ValueError("SUPPORT_DIMENSION_OR_NONFINITE")
            order = np.argsort(b["ids"], kind="stable")
            b["B"] = b["B"][order].tocsr()
            b["alpha"] = np.asarray(b["alpha"])[order]
            b["ids"] = [b["ids"][i] for i in order]

    def separate(self, x, rho=0):
        x = np.asarray(x, float)
        if x.shape != self.lo.shape or not np.isfinite(x).all() or not np.isfinite(rho):
            raise ValueError("POINT_DIMENSION_OR_NONFINITE")
        vl = []
        vu = []
        choice = []
        for b in self.blocks:
            if not len(b["alpha"]):
                vl.append(ld(0))
                vu.append(ld(0))
                choice.append(-1)
                continue
            low, up = enclosure(b["B"], x)
            aa = np.asarray(b["alpha"], ld)
            low = np.nextafter(low + aa - 2 * eps * (abs(low) + abs(aa)), -np.inf)
            up = np.nextafter(up + aa + 2 * eps * (abs(up) + abs(aa)), np.inf)
            mid = (low + up) / 2
            best = int(np.argmax(mid))
            k = best if mid[best] > 0 else -1
            # Stable exact refinement for ambiguous maximizers/ties; only shortlisted rows.
            candidates = np.flatnonzero(up >= max(ld(0), np.max(low)))
            if len(candidates) > 1 or (up[best] >= 0 and low[best] <= 0):
                vals = [(F(0), -1)]
                for i in candidates:
                    row = b["B"].getrow(int(i))
                    v = F(float(b["alpha"][i])) + sum(
                        (F(float(c)) * F(float(x[j])) for j, c in zip(row.indices, row.data)), F(0)
                    )
                    vals.append((v, int(i)))
                val, k = max(vals, key=lambda t: (t[0], -t[1]))
            vl.append(max(ld(0), np.max(low)))
            vu.append(max(ld(0), np.max(up)))
            choice.append(k)
        pl, pu = (
            enclosure(sp.csr_matrix(self.p), np.asarray(vl, ld)),
            enclosure(sp.csr_matrix(self.p), np.asarray(vu, ld)),
        )
        lower = pl[0]
        upper = pu[1]
        order = sorted(
            range(len(self.p)), key=lambda j: (-float((lower[j] + upper[j]) / 2), self.pids[j])
        )
        H = max(upper)
        tau = 1e-6 + 1e-9 * max(1, abs(rho), abs(H))
        return dict(
            choice=choice,
            order=order,
            H_lower=float(max(lower)),
            H_upper=float(np.nextafter(float(H), np.inf)),
            tau=float(tau),
            closed=bool(H - ld(rho) <= tau),
        )

    def row(self, sep, pindex, stored=None):
        rows = []
        alpha = []
        ids = []
        for b, k in zip(self.blocks, sep["choice"]):
            if k < 0:
                rows.append(sp.csr_matrix((1, len(self.lo))))
                alpha.append(0.0)
                ids.append("ZERO")
            else:
                rows.append(b["B"].getrow(k))
                alpha.append(b["alpha"][k])
                ids.append(b["ids"][k])
        B = sp.vstack(rows, format="csr")
        a, be, proof = protect(alpha, B, self.p[pindex], self.lo, self.hi, stored)
        return (
            a,
            be,
            dict(
                proof,
                support_ids=ids,
                p_id=self.pids[pindex],
                p_index=int(pindex),
                row_id=digest(a, be),
            ),
        )

    def activity(self, a, b, x):
        l, u = enclosure(sp.csr_matrix(np.asarray(b)[None, :]), x)
        g = 4 * eps * (abs(a) + max(abs(l[0]), abs(u[0])) + 1)
        return float(np.nextafter(float(l[0] + ld(a) - g), -np.inf)), float(
            np.nextafter(float(u[0] + ld(a) + g), np.inf)
        )
