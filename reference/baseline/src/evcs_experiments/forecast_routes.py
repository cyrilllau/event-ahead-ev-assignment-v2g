"""Complete forecast route search over all directed node-simple paths.

Arc travel time is fixed on entry into its half-open replay slot. No waiting,
rerouting, cycles, or earliest-prefix label dominance is permitted. Independent
static free-flow and distance lower bounds are safe even for non-FIFO travel.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from numbers import Integral
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError
from .random_streams import canonical_hash


@dataclass(frozen=True)
class Arc:
    a: int
    source: int
    target: int
    distance_km: float
    freeflow_h: float

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in
               (self.a, self.source, self.target)):
            raise InputValidationError("Arc and node identifiers must be integers")
        if (self.source == self.target or not math.isfinite(self.distance_km)
                or not math.isfinite(self.freeflow_h) or self.distance_km < 0
                or self.freeflow_h < 0):
            raise InputValidationError("Arc must join distinct nodes with nonnegative finite costs")


@dataclass(frozen=True)
class ForecastNetwork:
    arcs: tuple[Arc, ...]
    slot_starts_h: np.ndarray
    travel_hours: np.ndarray
    horizon_end_h: float = 24.0
    network_hash: str = ""

    def __post_init__(self):
        arcs = tuple(self.arcs)
        if not all(isinstance(a, Arc) for a in arcs) or len({a.a for a in arcs}) != len(arcs):
            raise InputValidationError("Network requires distinct Arc identifiers")
        starts = np.array(self.slot_starts_h, dtype=float, copy=True)
        travel = np.array(self.travel_hours, dtype=float, copy=True)
        if (starts.ndim != 1 or not len(starts) or not np.isfinite(starts).all()
                or np.any(np.diff(starts) <= 0) or not math.isfinite(self.horizon_end_h)
                or self.horizon_end_h <= starts[-1]):
            raise InputValidationError("Slot starts must increase strictly before a finite horizon end")
        if (travel.shape != (len(starts), len(arcs)) or np.isnan(travel).any()
                or np.isneginf(travel).any() or np.any(travel < 0)
                or np.any(travel < np.array([a.freeflow_h for a in arcs])[None, :])):
            raise InputValidationError("Travel times require (slots,arcs), each at least free-flow; +inf closes an arc")
        starts.setflags(write=False)
        travel.setflags(write=False)
        content_hash = canonical_hash({"arcs": [dict(a=int(a.a), source=int(a.source),
            target=int(a.target), distance_km=float(a.distance_km), freeflow_h=float(a.freeflow_h))
            for a in arcs], "slot_starts_h": starts.tolist(),
            "travel_hours": [[None if math.isinf(v) else float(v) for v in row] for row in travel],
            "horizon_end_h": float(self.horizon_end_h), "source_context_hash": self.network_hash})
        object.__setattr__(self, "arcs", arcs)
        object.__setattr__(self, "slot_starts_h", starts)
        object.__setattr__(self, "travel_hours", travel)
        object.__setattr__(self, "network_hash", content_hash)

    def slot_index(self, entry_h):
        if not math.isfinite(entry_h) or entry_h < self.slot_starts_h[0] or entry_h >= self.horizon_end_h:
            return None
        return int(np.searchsorted(self.slot_starts_h, entry_h, side="right") - 1)


@dataclass(frozen=True)
class Route:
    arc_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    departure_h: float
    arrival_h: float
    distance_km: float
    freeflow_h: float
    travel_energy_kwh: float
    arrival_soc: float
    entry_times_h: tuple[float, ...]

    def as_dict(self):
        return {"arc_ids": list(self.arc_ids), "node_ids": list(self.node_ids),
            "departure_h": self.departure_h, "arrival_h": self.arrival_h,
            "distance_km": self.distance_km, "freeflow_h": self.freeflow_h,
            "travel_energy_kwh": self.travel_energy_kwh, "arrival_soc": self.arrival_soc,
            "entry_times_h": list(self.entry_times_h)}


@dataclass(frozen=True)
class RouteSearchResult:
    status: str
    route: Route | None
    certificate: Mapping[str, object]

    def as_dict(self):
        certificate = dict(self.certificate)
        certificate["pruned"] = dict(certificate["pruned"])
        certificate["query"] = dict(certificate["query"])
        return {"status": self.status, "route": None if self.route is None else self.route.as_dict(),
                "certificate": certificate}


def _vehicle(departure_h, initial_soc, capacity_kwh, energy_kwh_per_km, auxiliary_kw):
    if (not all(math.isfinite(v) for v in
                (departure_h, initial_soc, capacity_kwh, energy_kwh_per_km, auxiliary_kw))
            or not 0 <= initial_soc <= 1 or capacity_kwh <= 0
            or energy_kwh_per_km < 0 or auxiliary_kw < 0):
        raise InputValidationError("Invalid departure, SOC, capacity, or nonnegative travel energy parameters")


def _route(arc_ids, nodes, departure, arrival, distance, freeflow, entries,
           initial_soc, capacity, energy_per_km, auxiliary):
    energy = energy_per_km * distance + auxiliary * max(0.0, arrival - departure - freeflow)
    return Route(tuple(arc_ids), tuple(nodes), float(departure), float(arrival), float(distance),
                 float(freeflow), float(energy), max(0.0, initial_soc - energy / capacity), tuple(entries))


def replay_route(network, arc_ids, departure_h, initial_soc, capacity_kwh,
                 energy_kwh_per_km, auxiliary_kw, *, origin=None, destination=None):
    """Replay precisely the supplied path; infeasible entry returns None.

    Invalid identifiers, noncontiguous paths and cycles raise an input error.
    The empty path is valid at its supplied origin and consumes no energy.
    Qualification and service deadlines are separate from physical replay.
    """
    _vehicle(departure_h, initial_soc, capacity_kwh, energy_kwh_per_km, auxiliary_kw)
    lookup = {a.a: (i, a) for i, a in enumerate(network.arcs)}
    ids = tuple(arc_ids)
    if any(i not in lookup for i in ids):
        raise InputValidationError("Route contains an unknown arc")
    if origin is None:
        if not ids:
            raise InputValidationError("Empty route requires an explicit origin")
        origin = lookup[ids[0]][1].source
    nodes = [origin]
    for aid in ids:
        arc = lookup[aid][1]
        if arc.source != nodes[-1] or arc.target in nodes:
            raise InputValidationError("Route must be contiguous and node-simple")
        nodes.append(arc.target)
    if destination is not None and destination != nodes[-1]:
        raise InputValidationError("Route does not end at the requested destination")
    arrival, distance, freeflow, entries = float(departure_h), 0.0, 0.0, []
    for aid in ids:
        index, arc = lookup[aid]
        slot = network.slot_index(arrival)
        if slot is None or not math.isfinite(network.travel_hours[slot, index]):
            return None
        entries.append(arrival)
        arrival += float(network.travel_hours[slot, index])
        distance += arc.distance_km
        freeflow += arc.freeflow_h
    return _route(ids, nodes, departure_h, arrival, distance, freeflow, entries,
                  initial_soc, capacity_kwh, energy_kwh_per_km, auxiliary_kw)


def _lower_bounds(arcs, destination, attribute):
    reverse = {}
    for arc in arcs:
        reverse.setdefault(arc.target, []).append((arc.source, getattr(arc, attribute)))
    best, heap = {destination: 0.0}, [(0.0, destination)]
    while heap:
        value, node = heapq.heappop(heap)
        if value != best[node]:
            continue
        for parent, cost in reverse.get(node, ()):
            candidate = value + cost
            if candidate < best.get(parent, math.inf):
                best[parent] = candidate
                heapq.heappush(heap, (candidate, parent))
    return best


def search_route(network, origin, destination, departure_h, initial_soc, capacity_kwh,
                 energy_kwh_per_km, auxiliary_kw, qualification_soc,
                 latest_arrival_h=23.0, max_expansions=1000000, time_limit_seconds=30.0):
    """Find the lexicographically best qualified route within the full path domain.

    Objective key is (arrival_h, distance_km, arc_ids). Static lower bounds may
    admit visited nodes and closed arcs; this makes them optimistic and safe.
    They never compare arrival labels of two different prefixes. A capped
    search returns its incumbent with SEARCH_INCOMPLETE, never a complete H.
    """
    _vehicle(departure_h, initial_soc, capacity_kwh, energy_kwh_per_km, auxiliary_kw)
    if (not math.isfinite(qualification_soc) or not 0 <= qualification_soc <= 1
            or not math.isfinite(latest_arrival_h)
            or isinstance(max_expansions, bool) or not isinstance(max_expansions, Integral)
            or max_expansions < 0 or not math.isfinite(time_limit_seconds) or time_limit_seconds < 0
            or any(isinstance(n, bool) or not isinstance(n, Integral) for n in (origin, destination))):
        raise InputValidationError("Invalid route query qualification, endpoint, deadline, or search limits")
    started = time.monotonic()
    outgoing = {}
    for index, arc in enumerate(network.arcs):
        outgoing.setdefault(arc.source, []).append((arc.a, index, arc))
    for values in outgoing.values():
        values.sort(key=lambda item: item[0])
    min_time = _lower_bounds(network.arcs, destination, "freeflow_h")
    min_distance = _lower_bounds(network.arcs, destination, "distance_km")
    # Stack entries retain exact full prefixes; the reversed insertion order
    # yields deterministic increasing-arc DFS.
    stack = [(origin, float(departure_h), 0.0, 0.0, (), (origin,), ())]
    best = None
    expansions = complete_paths = feasible_paths = 0
    pruned = {"cycle": 0, "blocked_or_horizon": 0, "unreachable": 0,
              "deadline_or_incumbent": 0, "energy": 0}
    termination = "exhausted"
    # At a zero qualification threshold the paper's clipped SOC admits zero
    # SOC, even if the algebraic pre-clipping SOC is negative.
    budget = (capacity_kwh * (initial_soc - qualification_soc)
              if qualification_soc > 0 else math.inf)
    while stack:
        if expansions >= max_expansions:
            termination = "max_expansions"
            break
        if time.monotonic() - started >= time_limit_seconds:
            termination = "time_limit"
            break
        node, arrival, distance, freeflow, ids, nodes, entries = stack.pop()
        expansions += 1
        if node == destination:
            complete_paths += 1
            route = _route(ids, nodes, departure_h, arrival, distance, freeflow, entries,
                           initial_soc, capacity_kwh, energy_kwh_per_km, auxiliary_kw)
            if route.arrival_h <= latest_arrival_h and route.arrival_soc >= qualification_soc:
                feasible_paths += 1
                if best is None or (route.arrival_h, route.distance_km, route.arc_ids) < (
                        best.arrival_h, best.distance_km, best.arc_ids):
                    best = route
            continue
        if node not in min_time:
            pruned["unreachable"] += 1
            continue
        # Roundoff in static sums could otherwise prune an exact boundary.
        # The slack weakens pruning only; all final eligibility is evaluated
        # directly above with the requested inequalities.
        slack = 1e-12 * max(1.0, abs(arrival), abs(latest_arrival_h), capacity_kwh)
        cutoff = min(latest_arrival_h, best.arrival_h if best is not None else math.inf)
        if arrival + min_time[node] > cutoff + slack:
            pruned["deadline_or_incumbent"] += 1
            continue
        lower_energy = (energy_kwh_per_km * (distance + min_distance[node])
            + auxiliary_kw * max(0.0, arrival - departure_h - freeflow))
        if lower_energy > budget + slack:
            pruned["energy"] += 1
            continue
        slot = network.slot_index(arrival)
        if slot is None:
            pruned["blocked_or_horizon"] += 1
            continue
        for aid, index, arc in reversed(outgoing.get(node, ())):
            if arc.target in nodes:
                pruned["cycle"] += 1
                continue
            travel = float(network.travel_hours[slot, index])
            if not math.isfinite(travel):
                pruned["blocked_or_horizon"] += 1
                continue
            stack.append((arc.target, arrival + travel, distance + arc.distance_km,
                freeflow + arc.freeflow_h, ids + (aid,), nodes + (arc.target,), entries + (arrival,)))
    status = "COMPLETE" if not stack else "SEARCH_INCOMPLETE"
    query = {"network_hash": network.network_hash, "origin": int(origin),
        "destination": int(destination), "departure_h": float(departure_h),
        "initial_soc": float(initial_soc), "capacity_kwh": float(capacity_kwh),
        "energy_kwh_per_km": float(energy_kwh_per_km), "auxiliary_kw": float(auxiliary_kw),
        "qualification_soc": float(qualification_soc), "latest_arrival_h": float(latest_arrival_h)}
    query_hash = canonical_hash({"query": query, "contract": generator_contract()})
    certificate = MappingProxyType({"domain": "all directed node-simple paths", "expansions": expansions,
        "complete_paths": complete_paths, "feasible_paths": feasible_paths,
        "pruned": MappingProxyType(pruned), "frontier_states_remaining": len(stack),
        "termination_reason": termination, "complete": status == "COMPLETE",
        "network_hash": network.network_hash, "query_hash": query_hash,
        "query": MappingProxyType(query),
        "pruning_margin": "1e-12*max(1,abs(prefix_arrival),abs(latest_arrival),capacity_kwh); final inequalities strict",
        "frontier_checkpoint_saved": False,
        "max_expansions": int(max_expansions), "time_limit_seconds": float(time_limit_seconds),
        "elapsed_seconds": time.monotonic() - started})
    return RouteSearchResult(status, best, certificate)


def generator_contract():
    return {"version": "S01-I06-route-v1", "domain": "all directed node-simple paths",
        "objective": ["arrival_h", "distance_km", "lexicographic arc ID sequence"],
        "entry_slot": "half-open [slot_start,next_slot_start); final slot ends at horizon_end_h",
        "travel": "fix travel time at entry; do not re-evaluate while on an arc",
        "no_waiting": True, "no_cycles": True, "no_prefix_label_dominance": True,
        "energy": "Ekm*distance + Paux*max(0,arrival-departure-sum(freeflow_h))",
        "safe_bounds": "nonnegative future freeflow, distance and delay; no FIFO assumption",
        "cap_status": "SEARCH_INCOMPLETE with incumbent and frontier count; cannot certify candidate H"}
