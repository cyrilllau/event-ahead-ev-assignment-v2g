"""Typed scientific interfaces, independent of files, campaigns and solver state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Mapping, Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Scenario:
    identity: str
    physical: Mapping[str, np.ndarray]
    route_states: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class AmbiguitySet:
    support_ids: tuple[str, ...]
    phi: sparse.csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    identity: str
    # Full original rows, not only the compressed/active solver matrix.
    check_original: Callable[[np.ndarray], Mapping[str, Any]]
    feasible_anchor: np.ndarray | None = None

    def __post_init__(self):
        if (
            not self.support_ids
            or len(set(self.support_ids)) != len(self.support_ids)
            or self.phi.shape != (len(self.lower), len(self.support_ids))
            or self.lower.shape != self.upper.shape
            or not np.isfinite(self.phi.data).all()
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.lower > self.upper)
        ):
            raise ValueError("Invalid or misordered ambiguity set")
        if self.feasible_anchor is not None:
            if np.asarray(self.feasible_anchor).shape != (len(self.support_ids),):
                raise ValueError("Anchor support mismatch")
            if not self.check_original(self.feasible_anchor).get("passed"):
                raise ValueError("Anchor is not in the original ambiguity set")


@dataclass(frozen=True)
class PlanningProblem:
    identity: str
    protocol: str
    routes: tuple[Mapping[str, Any], ...]
    fleet_size: int
    network: Mapping[str, Any]
    energy_bounds: np.ndarray
    scenario_ids: tuple[str, ...]
    scenario: Callable[[str], Scenario]
    ambiguity: AmbiguitySet | None = None
    starts: tuple[int, ...] = tuple(range(93))
    ends: tuple[int, ...] = tuple(range(1, 97))
    scope: str = "development"
    det_proof: Mapping[str, Any] | None = None

    def __post_init__(self):
        if self.protocol != "L08_FULL_DWELL_FORECAST44_V1":
            raise ValueError("Unsupported scientific protocol")
        if (
            not self.identity
            or not self.scenario_ids
            or len(set(self.scenario_ids)) != len(self.scenario_ids)
            or type(self.fleet_size) is not int
            or self.fleet_size < 1
            or np.asarray(self.energy_bounds).shape != (self.fleet_size,)
            or not np.isfinite(self.energy_bounds).all()
            or np.any(self.energy_bounds < 0)
        ):
            raise ValueError("Invalid planning problem identity or dimensions")
        if len({r["route_id"] for r in self.routes}) != len(self.routes):
            raise ValueError("Duplicate route identity")
        for r in self.routes:
            if not 0 <= r["e"] < self.fleet_size or not 0 <= r["c"] < len(
                self.network["station_buses"]
            ):
                raise ValueError("Route fleet/station mismatch")
            if not all(math.isfinite(r[k]) for k in ("departure_dc", "distance", "arrival", "B")):
                raise ValueError("Nonfinite route")
        if self.ambiguity and self.ambiguity.support_ids != self.scenario_ids:
            raise ValueError("Ambiguity/scenario ordering mismatch")
        if len(set(self.starts)) != len(self.starts) or len(set(self.ends)) != len(self.ends):
            raise ValueError("Duplicate time-domain entry")
        if any(type(t) is not int or not 0 <= t <= 92 for t in self.starts):
            raise ValueError("Invalid start domain")
        if any(type(t) is not int or not 1 <= t <= 96 for t in self.ends):
            raise ValueError("Invalid end domain")

    def load_scenario(self, sid):
        if sid not in self.scenario_ids:
            raise ValueError("Scenario outside locked planning domain")
        scene = self.scenario(sid)
        if scene.identity != sid or len(scene.route_states) != len(self.routes):
            raise ValueError("Scenario/replay identity mismatch")
        return scene


@dataclass(frozen=True)
class SolverOptions:
    seconds: float = 60.0
    threads: int = 1
    seed: int = 0
    mip_gap: float = 0.0
    absolute_gap: float = 1.0
    soft_memory_gb: float = 4.0
    master_seconds: float = 60.0

    def __post_init__(self):
        if any(
            isinstance(v, bool) or not math.isfinite(v) or v <= 0
            for v in (self.seconds, self.soft_memory_gb, self.master_seconds)
        ):
            raise ValueError("Invalid finite resource budget")
        if type(self.threads) is not int or self.threads < 1 or type(self.seed) is not int:
            raise ValueError("Invalid solver thread/seed")
        if any(not math.isfinite(v) or v < 0 for v in (self.mip_gap, self.absolute_gap)):
            raise ValueError("Invalid convergence target")


@dataclass(frozen=True)
class BoundScope:
    problem: str
    method: str
    scenarios: tuple[str, ...]
    integer_first_stage: bool = True


@dataclass(frozen=True)
class Bound:
    value: float
    scope: BoundScope
    evidence: str
    level: str  # native_solver, numerical_audit, strict_certificate

    def __post_init__(self):
        if not math.isfinite(self.value) or not self.evidence:
            raise ValueError("Missing or nonfinite bound evidence")
        if self.level not in {"native_solver", "numerical_audit", "strict_certificate"}:
            raise ValueError("Unknown bound evidence level")


def combine_bounds(lower: Bound | None, upper: Bound | None):
    if lower is None or upper is None:
        return None
    if lower.scope != upper.scope:
        raise ValueError("Cannot combine bounds from different scientific domains")
    gap = upper.value - lower.value
    if gap < 0:
        raise ValueError("Negative external gap requires scientific review")
    return gap


@dataclass(frozen=True)
class DecisionPlan:
    problem: str
    entries: tuple[Mapping[str, Any], ...]
    first_stage: np.ndarray


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    checks: Mapping[str, Any]
    level: str = "numerical_audit"


@dataclass(frozen=True)
class SolveResult:
    method: str
    native_status: int | None
    termination: str
    plan: DecisionPlan | None
    lower: Bound | None
    upper: Bound | None
    audit: AuditReport | None
    elapsed_seconds: float
    gap_passed: bool
    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    accepted_for_comparison: bool = False


class Control:
    """Cooperative budget/stop hook; callbacks do not grant scientific acceptance."""

    def __init__(self, seconds, *, stop=lambda: False, event=lambda value: None):
        self.start = time.monotonic()
        self.deadline = self.start + seconds
        self.stop = stop
        self.event = event

    def check(self):
        if self.stop():
            raise InterruptedError("Explicit stop requested")
        if self.remaining <= 0:
            raise TimeoutError("Cumulative execution budget exhausted")

    @property
    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())
