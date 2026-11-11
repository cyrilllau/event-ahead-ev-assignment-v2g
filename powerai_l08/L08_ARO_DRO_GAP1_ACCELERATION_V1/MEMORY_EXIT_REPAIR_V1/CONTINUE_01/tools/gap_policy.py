from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Any


@dataclass(frozen=True)
class GapReport:
    state: str
    reached: bool
    absolute_gap: float | None = None
    relative_gap: float | None = None
    trigger: str | None = None
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def evaluate_gap(lb: Any, ub: Any, *, expected_scope: str, lb_scope: str,
                 ub_scope: str, lower_evidence_verified: bool,
                 upper_evaluation_complete: bool,
                 upper_evidence_verified: bool,
                 relative_target: str = "0.01",
                 absolute_target: str = "1.0") -> GapReport:
    if not expected_scope or lb_scope != expected_scope or ub_scope != expected_scope:
        return GapReport("SCOPE_MISMATCH", False)
    if not (lower_evidence_verified and upper_evidence_verified and upper_evaluation_complete):
        return GapReport("EVIDENCE_INCOMPLETE", False, scope=expected_scope)
    if not (finite_number(lb) and finite_number(ub)):
        return GapReport("BOUNDS_UNAVAILABLE", False, scope=expected_scope)
    rel_tol, abs_tol = Fraction(relative_target), Fraction(absolute_target)
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("Gap targets must be nonnegative")
    lower, upper = Fraction.from_float(float(lb)), Fraction.from_float(float(ub))
    delta = upper - lower
    if delta < 0:
        return GapReport("BOUND_INCONSISTENCY", False, scope=expected_scope)
    relative = delta / max(Fraction(1), abs(upper))
    by_abs, by_rel = delta <= abs_tol, relative <= rel_tol
    trigger = ("BOTH" if by_abs and by_rel else
               "RELATIVE_1PCT" if by_rel else
               "ABSOLUTE_1USD" if by_abs else None)
    gap_float, rel_float = float(delta), float(relative)
    if not (math.isfinite(gap_float) and math.isfinite(rel_float)):
        return GapReport("ARITHMETIC_RANGE_DIAGNOSTIC", False, scope=expected_scope)
    return GapReport("TARGET_REACHED" if by_abs or by_rel else "CONTINUE",
                     by_abs or by_rel, gap_float, rel_float, trigger,
                     expected_scope)


def native_number(value: Any, *, solver_infinity: float) -> dict[str, Any]:
    if value is None:
        return {"value": None, "classification": "UNAVAILABLE"}
    if isinstance(value, bool):
        return {"value": None, "classification": "INVALID_TYPE"}
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return {"value": None, "classification": "UNAVAILABLE", "raw_text": repr(value)}
    if math.isnan(v):
        return {"value": None, "classification": "NAN_DIAGNOSTIC", "raw_text": repr(v)}
    if math.isinf(v):
        return {"value": None, "classification": "POS_INF" if v > 0 else "NEG_INF", "raw_text": repr(v)}
    if not math.isfinite(solver_infinity) or solver_infinity <= 0:
        raise ValueError("invalid solver infinity")
    if abs(v) >= solver_infinity:
        return {"value": None, "classification": "SOLVER_INFINITY_SENTINEL", "raw_text": repr(v)}
    return {"value": v, "classification": "FINITE"}
