"""Fixed-route, left-held road replay, with no routing or energy reset on failure.

The first arc may be a residual part of the normal business trip. Fractions
scale distance and travel time; passability is still checked on the parent arc.
Times are hours, arc distances km. No solver, data loading, or global data root.
"""

from __future__ import annotations

import math

import numpy as np


def replay_route(arc_ids, departure_h, arcs, travel_hours, blocked, *,
                 fractions=None, step_h=.25, horizon_h=24., origin=None,
                 destination=None):
    """Replay a declared route at actual entry times; never wait or reroute.

    ``arcs`` is an ordered sequence of {id, source, target, distance_km}.
    Return ``success``, ``arrival`` (positive infinity on failure), and a
    step-by-step witness. Reaching the last arc after the horizon is recorded
    as a physical arrival; the existing qualification kernel rejects service.
    """
    ids = tuple(arc_ids)
    frac = tuple(1. for _ in ids) if fractions is None else tuple(fractions)
    if (not math.isfinite(departure_h) or not 0 <= departure_h < horizon_h
            or not math.isfinite(step_h) or step_h <= 0
            or not math.isfinite(horizon_h) or horizon_h <= 0
            or len(frac) != len(ids)
            or any(not math.isfinite(f) or not 0 < f <= 1 for f in frac)):
        raise ValueError("Invalid route clock or residual fractions")
    tt = np.asarray(travel_hours, dtype=float)
    bb = np.asarray(blocked, dtype=bool)
    count = int(round(horizon_h / step_h))
    if (abs(count * step_h - horizon_h) > 1e-12 or tt.shape != (count, len(arcs))
            or bb.shape != tt.shape or np.isnan(tt).any() or (tt < 0).any()
            or np.any(~np.isfinite(tt[~bb]))):
        raise ValueError("Invalid left-held road arrays")
    by_id = {a["id"]: (j, a) for j, a in enumerate(arcs)}
    if len(by_id) != len(arcs) or any(a not in by_id for a in ids):
        raise ValueError("Duplicate or unknown road arc")
    path = [by_id[a][1] for a in ids]
    if (any(a["target"] != b["source"] for a, b in zip(path, path[1:]))
            or (path and origin is not None and path[0]["source"] != origin)
            or (path and destination is not None and path[-1]["target"] != destination)
            or (not path and origin != destination)
            or any(f != 1 for f in frac[1:])):
        raise ValueError("Route is not a connected path with at most one residual prefix")
    if any(not math.isfinite(a["distance_km"]) or a["distance_km"] < 0 for a in path):
        raise ValueError("Invalid road distance")
    current = float(departure_h)
    steps = []
    distance = sum(a["distance_km"] * f for a, f in zip(path, frac))
    for aid, fraction in zip(ids, frac):
        j, arc = by_id[aid]
        slot = math.floor(current / step_h)
        if not 0 <= slot < count:
            return dict(success=False, arrival=math.inf, distance=distance,
                        steps=steps, failed_arc_id=aid, reason="ENTRY_OUTSIDE_HORIZON")
        stopped = bool(bb[slot, j])
        duration = float(tt[slot, j]) * fraction
        steps.append(dict(arc_id=aid, entry_h=current, slot=slot,
                          blocked=stopped, fraction=fraction, travel_h=duration))
        if stopped:
            return dict(success=False, arrival=math.inf, distance=distance,
                        steps=steps, failed_arc_id=aid, reason="ROAD_BLOCKED_AT_ENTRY")
        current += duration
    return dict(success=True, arrival=current, distance=distance, steps=steps,
                failed_arc_id=None, reason="ROUTE_COMPLETE")
