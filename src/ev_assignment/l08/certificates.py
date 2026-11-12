"""Original finite-bound, outward-rounded L08 numerical certificates."""

import numpy as np
from scipy import sparse as sp


def enclosure(A, x):
    ld = np.longdouble
    eps = np.finfo(ld).eps
    n = A.shape[1]
    AA = A.astype(ld)
    xx = np.asarray(x, ld)
    value = AA @ xx
    mass = abs(AA) @ abs(xx)
    gamma = ld(2 * n + 8) * eps / (1 - ld(2 * n + 8) * eps)
    err = gamma * mass + np.finfo(ld).tiny
    return (value - err, value + err)


def row_audit(A, rhs, sense, lo, hi, x):
    ld = np.longdouble
    value = A.astype(ld) @ np.asarray(x, ld) - np.asarray(rhs, ld)
    rv = np.where(sense == "=", abs(value), np.where(sense == "<", value, -value))
    return {
        "row": float(max(0, np.max(rv, initial=0))),
        "bound": float(
            max(
                0,
                np.max(np.asarray(lo, ld) - x, initial=0),
                np.max(np.asarray(x, ld) - hi, initial=0),
            )
        ),
    }


def lagrangian(A, rhs, sense, cost, lo, hi, pi, objcon=0.0):
    ld = np.longdouble
    eps = np.finfo(ld).eps
    cert_pi = np.where(
        sense == "<", np.minimum(pi, 0), np.where(sense == ">", np.maximum(pi, 0), pi)
    )
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    lower_at, upper_at = enclosure(A.T, cert_pi)
    lower_rc = np.asarray(cost, ld) - upper_at
    upper_rc = np.asarray(cost, ld) - lower_at
    err = eps * (abs(cost) + np.maximum(abs(lower_at), abs(upper_at)))
    lower_rc -= err
    upper_rc += err
    terms = np.minimum.reduce([lower_rc * lo, lower_rc * hi, upper_rc * lo, upper_rc * hi])
    guard = ld(8 * len(terms) + 8) * eps * sum(abs(terms))
    guard += ld(8 * len(cert_pi) + 8) * eps * sum(abs(np.asarray(cert_pi, ld) * rhs))
    value = ld(objcon) + np.asarray(cert_pi, ld) @ np.asarray(rhs, ld) + sum(terms) - guard
    return (float(np.nextafter(float(value), -np.inf)), np.asarray(cert_pi, float))


def replay_cut(W, T, a, pi):
    pi = np.asarray(pi, float)
    pi = np.where(
        a["sense"] == "<", np.minimum(pi, 0), np.where(a["sense"] == ">", np.maximum(pi, 0), pi)
    )
    lower_at, upper_at = enclosure(W.T, pi)
    ld = np.longdouble
    eps = np.finfo(ld).eps
    rrl = np.asarray(a["c"], ld) - upper_at
    rru = np.asarray(a["c"], ld) - lower_at
    er = eps * (abs(a["c"]) + np.maximum(abs(lower_at), abs(upper_at)))
    rrl -= er
    rru += er
    terms = np.minimum.reduce(
        [rrl * a["finite_lo"], rrl * a["finite_hi"], rru * a["finite_lo"], rru * a["finite_hi"]]
    )
    bl, bu = enclosure(T.T, -pi)
    beta = np.asarray((bl + bu) / 2, float)
    err = np.maximum(abs(bl - beta), abs(bu - beta))
    correction = sum(err * np.maximum(abs(a["xlo"]), abs(a["xhi"])))
    hl, _ = enclosure(sp.csr_matrix(a["h"][None, :]), pi)
    alpha = hl[0] + sum(terms) - correction
    alpha -= eps * (8 * len(terms) + 8) * sum(abs(terms)) + eps * (8 * len(err) + 8) * correction
    return (float(np.nextafter(float(alpha), -np.inf)), beta, pi)


def finite_bounds(A, b, sense, lo, hi, names):
    from fractions import Fraction as F

    A = A.tocsr()
    lo = lo.copy()
    hi = hi.copy()
    proof = []
    for sweep in range(4):
        if np.isfinite(lo).all() and np.isfinite(hi).all():
            break
        for i in range(A.shape[0]):
            if sense[i] == "=":
                directions = [1, -1]
            else:
                directions = [1 if sense[i] == "<" else -1]
            js = A.indices[A.indptr[i] : A.indptr[i + 1]]
            co = A.data[A.indptr[i] : A.indptr[i + 1]]
            for sign in directions:
                cc = sign * co
                missing = [
                    k
                    for k, j in enumerate(js)
                    if cc[k] > 0
                    and (not np.isfinite(hi[j]))
                    or (cc[k] < 0 and (not np.isfinite(lo[j])))
                ]
                for k in missing:
                    terms = []
                    valid = True
                    for z, j in enumerate(js):
                        if z == k:
                            continue
                        endpoint = lo[j] if cc[z] > 0 else hi[j]
                        if not np.isfinite(endpoint):
                            valid = False
                            break
                        terms.append(F(float(cc[z])) * F(float(endpoint)))
                    if not valid:
                        continue
                    bd = (F(float(sign * b[i])) - sum(terms, F(0))) / F(float(cc[k]))
                    j = js[k]
                    val = float(bd)
                    if cc[k] > 0:
                        hi[j] = float(np.nextafter(val, np.inf))
                        kind = "upper"
                    else:
                        lo[j] = float(np.nextafter(val, -np.inf))
                        kind = "lower"
                    proof.append(
                        dict(
                            variable=names[j],
                            row=int(i),
                            direction=sign,
                            kind=kind,
                            exact=[bd.numerator, bd.denominator],
                            outward=float(hi[j] if kind == "upper" else lo[j]),
                        )
                    )
    return (lo, hi, proof)
