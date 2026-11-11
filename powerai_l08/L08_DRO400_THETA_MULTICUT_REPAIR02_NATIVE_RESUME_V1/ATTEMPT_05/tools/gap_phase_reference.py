"""Solver-free reference for control decisions, not a server patch or certificate.

Input bounds must already have independent provenance and scope validation.
This module checks arithmetic/routing only, not mathematical validity of bounds.
"""
from fractions import Fraction
import math
from typing import Optional


def _bound(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("BOOLEAN_IS_NOT_BOUND")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("NONFINITE_BOUND")
    return value


def gap(lb: Optional[float], ub: Optional[float], *, scope_matches: bool = True) -> dict:
    if not scope_matches:
        raise ValueError("BOUND_SCOPE_MISMATCH")
    lb, ub = _bound(lb), _bound(ub)
    if lb is None or ub is None:
        return {"reached": False, "reason": "MISSING_BOUND", "absolute": None, "relative": None}
    lower, upper = Fraction(lb), Fraction(ub)
    diff = upper - lower
    if diff < 0:
        raise ValueError("NEGATIVE_EXTERNAL_GAP")
    denominator = max(Fraction(1), abs(upper))
    absolute_ok, relative_ok = diff <= 1, diff / denominator <= Fraction(1, 100)
    reason = ("BOTH" if absolute_ok and relative_ok else
              "ABSOLUTE_1USD" if absolute_ok else
              "RELATIVE_1PCT" if relative_ok else "CONTINUE")
    return {"reached": absolute_ok or relative_ok, "reason": reason,
            "absolute": float(diff), "relative": float(diff / denominator)}


def route_after_bound_update(lb: Optional[float], audited_ub: Optional[float],
                             certified_primal_ub: Optional[float] = None, *,
                             certification_done_for_audited_point: bool = False) -> str:
    """Certification of a feasible upper bound does not certify a solver-based LB."""
    strict_gap = gap(lb, certified_primal_ub)
    if strict_gap["reached"]:
        return "TARGET_WITH_CERTIFIED_PRIMAL_UB_CHECK_LB_LEVEL"
    audited_gap = gap(lb, audited_ub)
    if not audited_gap["reached"]:
        return "CONTINUE_SCIENCE"
    if not certification_done_for_audited_point:
        return "FINAL_CERTIFY"
    if certified_primal_ub is None:
        return "NUMERICAL_TARGET_CERTIFICATION_PENDING"
    return "RESUME_SCIENCE_STRICT_GAP_NOT_REACHED"


def candidate_route(*, complete_cached: bool, certified_cut_violates: bool,
                    oracle_tight: bool, master_proof_complete: bool) -> str:
    """Called AFTER global-gap check; deliberately no 'previous_cut_new' input."""
    if not complete_cached:
        return "VALIDATE_AND_EVALUATE_FULL400"
    if certified_cut_violates:
        return "SEPARATE_CACHED_SUPPORT"
    if not oracle_tight:
        return "REFINE_EXISTING_ORACLE_EVIDENCE"
    if not master_proof_complete:
        return "CONTINUE_SAME_MASTER"
    return "CONSISTENCY_DIAGNOSTIC"


def request_handover(*, root_completed: bool, seconds_after_root: float,
                     pending_evaluation_or_separation: bool, global_stop: bool = False) -> bool:
    if global_stop:
        return True
    return (root_completed and seconds_after_root >= 120.0
            and pending_evaluation_or_separation)
