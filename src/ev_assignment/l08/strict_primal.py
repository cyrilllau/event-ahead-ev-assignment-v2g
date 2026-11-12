"""Independent exact binary64 primal checks, including original h - T*x RHS.

This proves supplied feasible primals; it does not pretend to repair arbitrary
rounded equalities or to be an exact LP solver. Failures retain audited labels.
"""

from fractions import Fraction as F
import math
import time
import numpy as np


def fraction(v):
    if not math.isfinite(float(v)):
        raise ValueError("NONFINITE_EXACT_INPUT")
    return F.from_float(float(v))


def upper(v):
    r = float(v)
    return np.nextafter(r, np.inf).item() if F.from_float(r) < v else r


def check(A, rhs, sense, lo, hi, x, deadline, T=None, first=None):
    A = A.tocsr()
    xx = [fraction(v) for v in x]
    ff = [fraction(v) for v in first] if first is not None else None
    for j, v in enumerate(xx):
        if (math.isfinite(float(lo[j])) and v < fraction(lo[j])) or (
            math.isfinite(float(hi[j])) and v > fraction(hi[j])
        ):
            return {"passed": False, "reason": "EXACT_BOUND", "column": j}
    for i in range(A.shape[0]):
        if i % 64 == 0 and time.monotonic() >= deadline:
            raise TimeoutError("FINAL_CERTIFICATION_DEADLINE")
        lhs = sum(
            (
                fraction(a) * xx[j]
                for j, a in zip(
                    A.indices[A.indptr[i] : A.indptr[i + 1]], A.data[A.indptr[i] : A.indptr[i + 1]]
                )
            ),
            F(0),
        )
        r = fraction(rhs[i])
        if T is not None:
            r -= sum(
                (
                    fraction(a) * ff[j]
                    for j, a in zip(
                        T.indices[T.indptr[i] : T.indptr[i + 1]],
                        T.data[T.indptr[i] : T.indptr[i + 1]],
                    )
                ),
                F(0),
            )
        ok = lhs <= r if sense[i] == "<" else lhs >= r if sense[i] == ">" else lhs == r
        if not ok:
            return {
                "passed": False,
                "reason": "EXACT_ORIGINAL_ROW",
                "row": i,
                "sense": str(sense[i]),
                "lhs_minus_original_rhs": str(lhs - r),
                "RHS_binding": "original h-T*x" if T is not None else "original first-stage rhs",
            }
    return {
        "passed": True,
        "rows": A.shape[0],
        "columns": A.shape[1],
        "arithmetic": "exact rational interpretation of binary64 data",
        "RHS_binding": "original h-T*x" if T is not None else "original first-stage rhs",
    }


def cost_upper(c, x):
    return upper(sum((fraction(a) * fraction(v) for a, v in zip(c, x)), F(0)))
