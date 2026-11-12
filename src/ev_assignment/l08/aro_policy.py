"""Solver-independent ARO transition policy. Not a replacement for certification.
Integration must call this after each legitimate lower/upper-bound update.
"""

from __future__ import annotations
from copy import deepcopy
import math
from typing import Any


def next_action(
    *,
    lower: float | None,
    upper_audited: float | None,
    has_incumbent: bool,
    cached_evaluation: bool,
    cache_separated_for_current_master: bool = False,
    native_optimal: bool = False,
    abs_target: float = 1.0,
) -> str:
    if not math.isfinite(abs_target) or abs_target < 0:
        raise ValueError("INVALID_GAP_TARGET")
    if lower is not None and not math.isfinite(lower):
        raise ValueError("INVALID_LOWER_BOUND")
    if upper_audited is not None and not math.isfinite(upper_audited):
        raise ValueError("INVALID_AUDITED_UPPER")
    if lower is not None and upper_audited is not None:
        gap = upper_audited - lower
        if gap < 0:
            return "BOUND_INCONSISTENCY_REVIEW"
        if gap <= abs_target:
            return "FINAL_CERTIFY"  # Audited UB is never upgraded by this helper.
    if not has_incumbent:
        return "CONTINUE_MASTER"
    if cached_evaluation and not cache_separated_for_current_master:
        return "REPLAY_CACHED_EVALUATION"
    if cached_evaluation:
        return "DIAGNOSE_UNCLOSED_GAP" if native_optimal else "CONTINUE_MASTER"
    return "EVALUATE400"


def candidate_transition(
    state: dict[str, Any], *, point_id: str, point_ref: dict[str, Any], eta: float
) -> dict[str, Any]:
    """Return one complete new state; publish the return value once, atomically."""
    if state.get("phase") != "SOLVE":
        raise ValueError("CANDIDATE_TRANSITION_REQUIRES_SOLVE")
    if not math.isfinite(eta):
        raise ValueError("NONFINITE_ETA")
    new = deepcopy(state)
    new.update(
        outer=int(state["outer"]) + 1,
        phase="EVALUATE",
        point_id=point_id,
        point=deepcopy(point_ref),
        eta_at_candidate=float(eta),
    )
    return new
