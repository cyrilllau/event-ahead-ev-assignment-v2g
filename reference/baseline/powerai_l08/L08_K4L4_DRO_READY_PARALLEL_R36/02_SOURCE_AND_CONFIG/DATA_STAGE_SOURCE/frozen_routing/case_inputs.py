"""Forecast-only case inputs for S01; no realized-weather or scenario generation.

Operating period t=1 is [0,1), and t=24 is [23,24). Weather
interpolation holds endpoint values, including the counterfactual L=2 h
extension before the supplied -1 h weather state. Daily fleet profiles
interpolate SOC but hold the preceding integer clock-hour location.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

FORECAST_KEYS = (
    "event_ids", "model_hours", "latitude", "longitude",
    "forecast_u10_grid", "forecast_v10_grid", "forecast_gust_grid",
)
INPUT_PATHS = (
    "07_weather/xiW_base__registered_wind_8events_25states.npz",
    "07_weather/theta_asg__event_clock.csv",
    "06_fleet/E__vehicle_parameters.csv",
    "06_fleet/loc_SOC__E_clockhour.csv",
    "06_fleet/loc_ini_SOC_ini__event_E.csv",
    "05_evcs/C__evcs_bus_road_mapping.csv",
)
VALID_LEADS_H = (0.0, 0.5, 1.0, 2.0)


class InputValidationError(ValueError):
    """Input violates the frozen case contract."""


def _readonly(value: object, dtype: object = float) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid number: {label}") from exc
    if not math.isfinite(number):
        raise InputValidationError(f"Nonfinite number: {label}")
    return number


def _integer(value: str, label: str) -> int:
    number = _finite(value, label)
    if number != int(number):
        raise InputValidationError(f"Not an integer: {label}")
    return int(number)


def _rows(path: Path, keys: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise InputValidationError(f"Empty table: {path.name}")
    seen = set()
    for row in rows:
        try:
            key = tuple(row[k] for k in keys)
        except KeyError as exc:
            raise InputValidationError(f"Missing key column: {path.name}") from exc
        if any(not value for value in key) or key in seen:
            raise InputValidationError(f"Empty/duplicate primary key in {path.name}: {key}")
        seen.add(key)
    return rows


@dataclass(frozen=True, slots=True)
class VehicleState:
    vehicle_id: str
    depot_id: str
    road_node: int
    soc: float
    x_km: float
    y_km: float
    longitude: float
    latitude: float


def sample_clock_profile(profile: Mapping[int, VehicleState], clock_hour: float) -> VehicleState:
    """Sample the cyclic 24-hour profile, including the 23 -> 0 boundary."""
    if not math.isfinite(clock_hour):
        raise InputValidationError("Nonfinite clock hour")
    hour = clock_hour % 24.0
    left = math.floor(hour)
    fraction = hour - left
    a, b = profile[left], profile[(left + 1) % 24]
    return VehicleState(a.vehicle_id, a.depot_id, a.road_node,
                        a.soc + fraction * (b.soc - a.soc),
                        a.x_km, a.y_km, a.longitude, a.latitude)


@dataclass(frozen=True, slots=True)
class ForecastSamples:
    times_h: np.ndarray
    u10: np.ndarray
    v10: np.ndarray
    gust: np.ndarray


@dataclass(frozen=True, slots=True)
class ForecastEvent:
    event_id: str
    model_hours: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    u10: np.ndarray
    v10: np.ndarray
    gust: np.ndarray

    def at_times(self, times_h: Sequence[float]) -> ForecastSamples:
        times = np.atleast_1d(np.asarray(times_h, dtype=float))
        if times.ndim != 1 or not np.isfinite(times).all():
            raise InputValidationError("Weather query times must be a finite vector")
        clipped = np.clip(times, self.model_hours[0], self.model_hours[-1])
        right = np.searchsorted(self.model_hours, clipped, side="right")
        right = np.clip(right, 1, len(self.model_hours) - 1)
        left = right - 1
        fraction = ((clipped - self.model_hours[left]) /
                    (self.model_hours[right] - self.model_hours[left]))[:, None, None]
        def interpolate(values: np.ndarray) -> np.ndarray:
            return _readonly(values[left] + fraction * (values[right] - values[left]))
        return ForecastSamples(_readonly(times), interpolate(self.u10),
                               interpolate(self.v10), interpolate(self.gust))


@dataclass(frozen=True, slots=True)
class ForecastBank:
    event_ids: tuple[str, ...]
    events: Mapping[str, ForecastEvent]

    def for_event(self, event_id: str) -> ForecastEvent:
        try:
            return self.events[event_id]
        except KeyError as exc:
            raise InputValidationError(f"Unknown event: {event_id}") from exc


def _load_forecast(path: Path) -> ForecastBank:
    # Do not enumerate values/items or retain the NpzFile: only these keys are read.
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in FORECAST_KEYS}
    ids = arrays["event_ids"]
    if ids.ndim != 1 or len(ids) != 8 or len(set(ids.tolist())) != 8:
        raise InputValidationError("Weather requires eight unique event IDs")
    event_ids = tuple(str(value) for value in ids)
    if any(not value for value in event_ids):
        raise InputValidationError("Empty weather event ID")
    hours = arrays["model_hours"]
    if not np.array_equal(hours, np.arange(-1, 24)):
        raise InputValidationError("Expected weather state hours -1,...,23")
    for key in ("latitude", "longitude"):
        axis = arrays[key]
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise InputValidationError(f"Invalid {key} axis")
        if not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0)):
            raise InputValidationError(f"Nonmonotonic {key} axis")
    shape = (8, 25, len(arrays["latitude"]), len(arrays["longitude"]))
    for key in FORECAST_KEYS[4:]:
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise InputValidationError(f"Invalid shape/nonfinite weather: {key}")
    if np.any(arrays["forecast_gust_grid"] < 0):
        raise InputValidationError("Negative forecast gust")
    axes = [_readonly(arrays[key]) for key in ("model_hours", "latitude", "longitude")]
    events = {}
    for index, event_id in enumerate(event_ids):
        fields = [_readonly(arrays[key][index]) for key in FORECAST_KEYS[4:]]
        events[event_id] = ForecastEvent(event_id, *axes, *fields)
    return ForecastBank(event_ids, MappingProxyType(events))


@dataclass(frozen=True, slots=True)
class TargetInputs:
    event_id: str
    target_id: str
    lead_h: float
    assignment_time_local: datetime
    service_start_time_local: datetime
    forecast_run_time_utc: datetime
    forecast_issued_after_assignment: bool
    vehicle_states: tuple[VehicleState, ...]
    station_ids: tuple[str, ...]
    state_sha256: str
    input_hashes: Mapping[str, str]
    forecast: ForecastEvent
    operating_intervals_h: tuple[tuple[int, int], ...]
    replay_times_h: np.ndarray


@dataclass(frozen=True, slots=True)
class CaseInputs:
    event_ids: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    vehicle_parameters: Mapping[str, Mapping[str, str]]
    station_parameters: Mapping[str, Mapping[str, str]]
    input_hashes: Mapping[str, str]
    forecast: ForecastBank
    _event_clocks: Mapping[str, Mapping[str, str]]
    _profiles: Mapping[str, Mapping[int, VehicleState]]

    def target(self, event_id: str, lead_h: float = 1.0) -> TargetInputs:
        if isinstance(lead_h, bool) or lead_h not in VALID_LEADS_H:
            raise InputValidationError(f"Lead must be one of {VALID_LEADS_H}")
        if event_id not in self._event_clocks:
            raise InputValidationError(f"Unknown target event: {event_id}")
        clock = self._event_clocks[event_id]
        service = datetime.fromisoformat(clock["service_start_time_local_eastern"])
        assignment = service - timedelta(hours=float(lead_h))
        hour = assignment.hour + assignment.minute / 60 + assignment.second / 3600
        states = tuple(sample_clock_profile(self._profiles[e], hour) for e in self.vehicle_ids)
        target_id = f"{event_id}__lead_{int(lead_h * 60):03d}min"
        payload = {"target_id": target_id, "assignment_time_local": assignment.isoformat(),
                   "states": [[s.vehicle_id, s.depot_id, s.road_node, s.soc,
                               s.x_km, s.y_km, s.longitude, s.latitude] for s in states]}
        state_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
        forecast_run = datetime.fromisoformat(clock["forecast_run_time_utc"])
        return TargetInputs(event_id, target_id, float(lead_h), assignment, service,
                            forecast_run, forecast_run > assignment, states, self.station_ids,
                            state_hash, self.input_hashes, self.forecast.for_event(event_id),
                            tuple((t - 1, t) for t in range(1, 25)),
                            _readonly(np.arange(-int(lead_h * 4), 96) / 4))


def load_case_inputs(case_root: str | Path) -> CaseInputs:
    root = Path(case_root)
    hashes = MappingProxyType({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in INPUT_PATHS})
    forecast = _load_forecast(root / INPUT_PATHS[0])
    clocks = {r["event_id"]: MappingProxyType(r) for r in _rows(root / INPUT_PATHS[1], ("event_id",))}
    vehicles = {r["e"]: MappingProxyType(r) for r in _rows(root / INPUT_PATHS[2], ("e",))}
    stations = {r["c"]: MappingProxyType(r) for r in _rows(root / INPUT_PATHS[5], ("c",))}
    if set(clocks) != set(forecast.event_ids) or len(vehicles) != 80 or len(stations) != 8:
        raise InputValidationError("Expected identical eight events, fixed 80 vehicles and eight stations")
    for event, row in clocks.items():
        service = datetime.fromisoformat(row["service_start_time_local_eastern"])
        service_utc = datetime.fromisoformat(row["service_start_time_utc"])
        assignment = datetime.fromisoformat(row["theta_asg_local_eastern"])
        run = datetime.fromisoformat(row["forecast_run_time_utc"])
        if any(dt.utcoffset() is None for dt in (service, service_utc, assignment, run)):
            raise InputValidationError(f"Naive event timestamp: {event}")
        if (service != service_utc or (service - assignment).total_seconds() != 3600 or
                assignment != datetime.fromisoformat(row["theta_asg_utc"]) or
                service.hour != _integer(row["service_start_hour_local"], event) or
                assignment.hour != _integer(row["assignment_hour_local"], event)):
            raise InputValidationError(f"Inconsistent event clock: {event}")
        if (service.minute or service.second or service.microsecond or
                _integer(row["candidate_vehicle_count"], event) != 80 or
                row["candidate_fleet_fixed"] != "1" or row["availability_filter_used"] != "0"):
            raise InputValidationError(f"Invalid fixed-fleet event contract: {event}")
    for vehicle, row in vehicles.items():
        if row["is_in_E"] != "1" or row["is_contracted_e"] != "1":
            raise InputValidationError(f"Vehicle not in fixed contracted fleet: {vehicle}")
        for key in ("E_cap_e_kWh", "P_dis_bar_e_kW", "eta_dis_e", "E_km_e_kWh_per_km",
                    "P_aux_e_kW", "SOC_dep_under_e", "P_blk_under_e_kW", "S_inv_bar_e_kVA"):
            if _finite(row[key], f"{vehicle}/{key}") < 0:
                raise InputValidationError(f"Negative vehicle parameter: {vehicle}/{key}")
        if not 0 < float(row["eta_dis_e"]) <= 1 or not 0 <= float(row["SOC_dep_under_e"]) <= 1:
            raise InputValidationError(f"Invalid efficiency/SOC: {vehicle}")
        if any(float(row[key]) <= 0 for key in
               ("E_cap_e_kWh", "P_dis_bar_e_kW", "P_blk_under_e_kW", "S_inv_bar_e_kVA")):
            raise InputValidationError(f"Nonpositive capacity/rating: {vehicle}")
    for station, row in stations.items():
        for key in ("n_c", "road_node_c", "N_plug_c"):
            if _integer(row[key], f"{station}/{key}") <= 0:
                raise InputValidationError(f"Invalid station index/count: {station}")
        for key in ("P_int_bar_c_kW", "S_int_bar_c_kVA"):
            if _finite(row[key], f"{station}/{key}") <= 0:
                raise InputValidationError(f"Invalid station rating: {station}")
    profiles: dict[str, dict[int, VehicleState]] = {e: {} for e in vehicles}
    for row in _rows(root / INPUT_PATHS[3], ("e", "clock_hour_local")):
        e = row["e"]
        hour = _integer(row["clock_hour_local"], "clock_hour_local")
        if e not in vehicles or hour not in range(24) or hour in profiles[e]:
            raise InputValidationError(f"Unknown vehicle or invalid/duplicate clock key: {e}/{hour}")
        if row["d_e"] != vehicles[e]["d_e"]:
            raise InputValidationError(f"Depot mismatch: {e}")
        node = _integer(row["loc_e_road_node"], f"{e}/road_node")
        numbers = [_finite(row[key], f"{e}/{key}") for key in
                   ("SOC_e", "loc_e_x_km", "loc_e_y_km", "loc_e_longitude", "loc_e_latitude")]
        if node <= 0 or not 0 <= numbers[0] <= 1:
            raise InputValidationError(f"Invalid node/SOC: {e}")
        profiles[e][hour] = VehicleState(e, row["d_e"], node, *numbers)
    if any(set(profile) != set(range(24)) for profile in profiles.values()):
        raise InputValidationError("Every vehicle requires all 24 clock-hour states")
    case = CaseInputs(forecast.event_ids, tuple(sorted(vehicles)), tuple(sorted(stations)),
                      MappingProxyType(vehicles), MappingProxyType(stations), hashes, forecast,
                      MappingProxyType(clocks), MappingProxyType({e: MappingProxyType(p) for e, p in profiles.items()}))
    baseline_rows = _rows(root / INPUT_PATHS[4], ("event_id", "e"))
    expected = {(event, e) for event in case.event_ids for e in case.vehicle_ids}
    if {(r["event_id"], r["e"]) for r in baseline_rows} != expected:
        raise InputValidationError("Baseline snapshots must cover all eight events x 80 vehicles")
    derived = {(event, s.vehicle_id): s for event in case.event_ids for s in case.target(event).vehicle_states}
    for row in baseline_rows:
        state = derived[row["event_id"], row["e"]]
        actual = (state.road_node, state.soc, state.x_km, state.y_km, state.longitude, state.latitude)
        reference = tuple(_finite(row[k], k) for k in ("loc_ini_e_road_node", "SOC_ini_e",
                          "loc_ini_e_x_km", "loc_ini_e_y_km", "loc_ini_e_longitude", "loc_ini_e_latitude"))
        if actual != reference or row["d_e"] != state.depot_id or row["is_in_E"] != "1":
            raise InputValidationError(f"Derived L=1 state differs from baseline: {row['event_id']}/{row['e']}")
    return case
