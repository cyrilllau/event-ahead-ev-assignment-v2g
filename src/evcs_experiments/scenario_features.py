"""Five decision-independent scenario features on their explicit physical domains.

Wind uses every point of the native weather grid; line and road averages use
the canonical 32 closed lines and 80 directed road arcs. Deliverability gives
each vehicle with candidate tasks equal total weight, independent of H size.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError, VALID_LEADS_H
from .forecast_tasks import CandidateTaskBank
from .random_streams import canonical_hash
from .wind_field import latlon_to_xy

FEATURE_NAMES = ("Phi_W", "Phi_L", "Phi_R", "Phi_D", "Phi_load")
LINE_IDS = tuple(range(1, 33))
ROAD_IDS = tuple(range(1, 81))
GRID_IDS = tuple(range(441))
BUS_IDS = tuple(range(1, 34))


def _readonly(values, dtype=float):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _array_hash(values):
    array = np.ascontiguousarray(values)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def _finite(values, name):
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputValidationError(f"{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise InputValidationError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class FeatureGrid:
    latitude: np.ndarray
    longitude: np.ndarray
    points_latlon: np.ndarray
    points_xy_km: np.ndarray
    point_ids: tuple[int, ...]
    grid_sha256: str


def native_weather_grid(wind_bank):
    """Build all 441 native grid points in latitude-major ij meshgrid order.

    Fields have already normalized native axes in the accepted wind module.
    Requiring all eight bases to agree prevents event-specific feature domains.
    No boundary point is discarded when transformed weather needs projection.
    """
    fields = getattr(wind_bank, "fields", None)
    if not isinstance(fields, Mapping) or len(fields) != 8:
        raise InputValidationError("Native feature grid requires all eight base weather fields")
    latitude = longitude = None
    for key in sorted(fields):
        field = fields[key]
        lat = _finite(field.latitude, "Native latitude")
        lon = _finite(field.longitude, "Native longitude")
        if (lat.shape != (21,) or lon.shape != (21,)
                or np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0)):
            raise InputValidationError("Expected strictly increasing native 21 by 21 weather axes")
        if latitude is None:
            latitude, longitude = lat, lon
        elif not np.array_equal(latitude, lat) or not np.array_equal(longitude, lon):
            raise InputValidationError("Base weather fields disagree on the complete feature grid")
    mesh = np.meshgrid(latitude, longitude, indexing="ij")
    latlon = np.stack(mesh, axis=-1).reshape(-1, 2)
    xy = latlon_to_xy(latlon)
    digest = canonical_hash({"latitude": latitude.tolist(), "longitude": longitude.tolist(),
        "points_latlon_sha256": _array_hash(latlon), "points_xy_sha256": _array_hash(xy),
        "point_order": "latitude-major ij meshgrid, longitude varying fastest"})
    return FeatureGrid(_readonly(latitude), _readonly(longitude), _readonly(latlon),
                       _readonly(xy), GRID_IDS, digest)


@dataclass(frozen=True)
class PreparedFeatureTasks:
    task_ids: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    forecast_usable_dc_kwh: np.ndarray
    task_weights: np.ndarray
    task_sha256: str
    prepared_sha256: str
    vehicle_count: int
    empty_H: bool
    vehicle_groups: tuple[np.ndarray, ...]


def prepare_feature_tasks(frozen_bank):
    """Validate the full, unselected H and positive forecast energy denominators."""
    bank = frozen_bank.as_dict() if isinstance(frozen_bank, CandidateTaskBank) else frozen_bank
    if (not isinstance(bank, Mapping) or bank.get("status") != "COMPLETE"
            or bank.get("certified") is not True or bank.get("tasks_provisional", False)):
        raise InputValidationError("Features require a certified, complete, frozen task bank")
    tasks = bank["tasks"]
    task_hash = canonical_hash({"dependency_sha256": bank["dependency_sha256"],
        "config": dict(bank["config"]), "certified": True, "tasks": tasks})
    if task_hash != bank["task_sha256"]:
        raise InputValidationError("Frozen task semantic hash mismatch")
    ids = tuple(row["task_id"] for row in tasks)
    vehicles = tuple(row["vehicle_id"] for row in tasks)
    if (len(ids) != len(set(ids)) or any(not isinstance(v, str) or not v for v in ids + vehicles)):
        raise InputValidationError("Candidate tasks require unique nonempty IDs and explicit vehicle IDs")
    denominators = _finite([row["usable_energy_kwh"] for row in tasks], "Forecast usable DC energy")
    if np.any(denominators <= 0):
        raise InputValidationError("Every candidate task must have strictly positive forecast usable DC energy")
    groups = tuple(_readonly([i for i, vehicle in enumerate(vehicles) if vehicle == key], np.int64)
                   for key in sorted(set(vehicles)))
    weights = np.zeros(len(ids), dtype=float)
    for group in groups:
        weights[group] = 1. / (len(groups) * len(group))
    digest = canonical_hash({"task_sha256": task_hash, "task_ids": ids, "vehicle_ids": vehicles,
        "forecast_usable_dc_kwh": denominators.tolist(), "task_weights": weights.tolist(),
        "contract": generator_contract()})
    return PreparedFeatureTasks(ids, vehicles, _readonly(denominators), _readonly(weights),
        task_hash, digest, len(groups), not bool(ids), groups)


@dataclass(frozen=True)
class FeatureSample:
    values: np.ndarray
    feature_sha256: str
    diagnostics: Mapping

    def as_dict(self):
        return {"feature_names": list(FEATURE_NAMES), "values": self.values.tolist(),
                "feature_sha256": self.feature_sha256, "diagnostics": dict(self.diagnostics)}


def _ids(values, expected, name, allow_permutation=False):
    ids = tuple(values)
    if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in ids)
            or len(ids) != len(set(ids)) or (set(ids) != set(expected) if allow_permutation else ids != expected)):
        raise InputValidationError(f"{name} IDs do not match the explicit canonical domain/order")
    return ids


def compute_features(grid_hazard, line_outages, road_blocked, task_usable_dc,
                     load_p_kw, base_p_kw, *, prepared_tasks, times_h, task_ids,
                     road_ids, line_ids=LINE_IDS, grid_ids=GRID_IDS, bus_ids=BUS_IDS,
                     grid_sha256=None, scenario_identity=None):
    """Compute (W,L,R,D,load) without changing samples, H, or physical domains.

    All hazard/state rows must have the same declared actual replay grid.
    Connector columns may be supplied only with all 88 explicit arc IDs and
    are excluded from Phi_R. Passing the movement mask as blockage would be a
    caller error; this API intentionally accepts only the named blocked input.
    """
    if not isinstance(prepared_tasks, PreparedFeatureTasks):
        raise InputValidationError("Expected a validated PreparedFeatureTasks")
    actual_task_ids = tuple(task_ids)
    if actual_task_ids != prepared_tasks.task_ids:
        raise InputValidationError("Realized task energies are not aligned to the frozen H IDs/order")
    times = _finite(times_h, "Actual replay times")
    if times.ndim != 1 or not len(times):
        raise InputValidationError("Replay times must be a nonempty vector")
    lead = -float(times[0])
    if (lead not in VALID_LEADS_H
            or not np.array_equal(times, np.arange(-lead, 24., .25))):
        raise InputValidationError("Features require the actual assignment-to-24h quarter-hour grid")
    _ids(line_ids, LINE_IDS, "Closed line")
    _ids(grid_ids, GRID_IDS, "Weather grid")
    _ids(bus_ids, BUS_IDS, "Load bus")
    roads = tuple(road_ids)
    if len(roads) not in (80, 88):
        raise InputValidationError("Road blockage requires either canonical 80 arcs or complete 88-arc network")
    _ids(roads, tuple(range(1, len(roads) + 1)), "Road", allow_permutation=True)
    wind = _finite(grid_hazard, "Native-grid hazard")
    lines = _finite(line_outages, "Closed-line outages")
    blocked = _finite(road_blocked, "Road blockage")
    energy = _finite(task_usable_dc, "Realized usable DC energy")
    realized = _finite(load_p_kw, "Realized active load")
    baseline = _finite(base_p_kw, "Baseline active load")
    if (wind.shape != (len(times), 441) or lines.shape != (len(times), 32)
            or blocked.shape != (len(times), len(roads))
            or energy.shape != (len(prepared_tasks.task_ids),)
            or realized.shape != (24, 33) or baseline.shape != (24, 33)):
        raise InputValidationError("Feature input shape does not match time and explicit asset/task domains")
    if (np.any((wind < 0) | (wind > 1)) or not np.isin(lines, (0, 1)).all()
            or not np.isin(blocked, (0, 1)).all() or np.any(energy < 0)
            or np.any(realized < 0) or np.any(baseline < 0) or baseline.sum() <= 0
            or np.any(realized[baseline == 0] != 0)):
        raise InputValidationError("Feature inputs violate physical hazard, binary state, energy or load bounds")
    canonical_columns = [roads.index(arc) for arc in ROAD_IDS]
    canonical_blocked = blocked[:, canonical_columns]
    if prepared_tasks.empty_H:
        phi_d = 0.
    else:
        losses = 1 - np.minimum(1., energy / prepared_tasks.forecast_usable_dc_kwh)
        # The two explicit averages give exact 0/1 at the two physical extremes
        # and avoid a rounded sum of per-task weights slightly exceeding one.
        phi_d = float(np.mean([np.mean(losses[group]) for group in prepared_tasks.vehicle_groups]))
    phi_load = float(realized.sum() / baseline.sum())
    if not .91 - 1e-12 <= phi_load <= 1.09 + 1e-12:
        raise InputValidationError("Load feature is outside the frozen joint-truncation physical range")
    values = _readonly([wind.mean(), lines.mean(), canonical_blocked.mean(), phi_d, phi_load])
    lower, upper = [0., 0., 0., 0., .91], [1., 1., 1., 1., 1.09]
    if prepared_tasks.empty_H:
        upper[3] = 0.
    input_hashes = {name: _array_hash(array) for name, array in
        (("grid_hazard", wind), ("line_outages", lines), ("canonical_road_blocked", canonical_blocked),
         ("task_usable_dc", energy), ("load_p_kw", realized), ("base_p_kw", baseline), ("times_h", times))}
    diagnostics = {"feature_names": FEATURE_NAMES, "time_count": len(times), "lead_h": lead,
        "weather_grid_count": 441, "closed_line_count": 32, "road_feature_arc_count": 80,
        "road_input_arc_count": len(roads), "excluded_connector_count": len(roads) - 80,
        "task_count": len(prepared_tasks.task_ids), "candidate_vehicle_count": prepared_tasks.vehicle_count,
        "empty_H": prepared_tasks.empty_H, "phi_D_structurally_zero": prepared_tasks.empty_H,
        "physical_lower_bounds": lower, "physical_upper_bounds": upper,
        "phi_load_exceeds_one": phi_load > 1., "grid_sha256": grid_sha256,
        "task_sha256": prepared_tasks.task_sha256, "prepared_tasks_sha256": prepared_tasks.prepared_sha256,
        "component_array_hashes": input_hashes,
        "scenario_identity": {} if scenario_identity is None else dict(scenario_identity)}
    digest = canonical_hash({"values": values.tolist(), "diagnostics": diagnostics,
                             "contract": generator_contract()})
    return FeatureSample(values, digest, MappingProxyType(diagnostics))


def generator_contract():
    return {"version": "S01-I08-feature-v1", "feature_order": FEATURE_NAMES,
        "Phi_W": "Mean hazard over all 441 native weather-grid points and actual assignment replay slots",
        "Phi_L": "Mean outage of canonical 32 closed lines over actual replay slots",
        "Phi_R": "Mean persistent blocked state of canonical 80 directed arcs; exclude connectors and do not substitute unsafe-movement mask",
        "Phi_D": "For each candidate vehicle, mean_h[1-min(1,Euse_real_DC/Euse_forecast_DC)], then equal mean across vehicles with H",
        "empty_H": "Phi_D=0 with explicit structural flag; do not infer physical constant domains from observed finite samples",
        "Phi_load": "Total realized active load / total already-scaled baseline active load; may exceed one",
        "bounds": [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [.91, 1.09]],
        "time_domain": "Target's complete [-lead,24) grid at 0.25h, not the longer wind reference grid",
        "source_policy": "Frozen unselected H with verified semantic hash and explicit ordered task IDs"}
