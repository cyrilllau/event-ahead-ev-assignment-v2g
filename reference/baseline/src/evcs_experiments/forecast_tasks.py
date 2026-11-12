"""Forecast-only road adapter and certified fixed-route candidate tasks.

No realized wind, residual bank, line outage, or road resistance draw enters
this module. Exact route queries are completed before nearest-station filtering;
an incomplete query makes the whole returned task bank provisional.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import CaseInputs, InputValidationError, TargetInputs, load_case_inputs
from .forecast_routes import Arc, ForecastNetwork, search_route
from .random_streams import canonical_hash
from .road_connectors import derive_road_connectors
from .road_disruptions import load_road_model
from .wind_field import forecast_vector_field


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _array_hash(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def qualification_soc(basic_soc, departure_soc, block_power_kw, tau_blk_h,
                      efficiency, capacity_kwh):
    values = (basic_soc, departure_soc, block_power_kw, tau_blk_h, efficiency, capacity_kwh)
    if (not all(math.isfinite(float(v)) for v in values)
            or not 0 <= basic_soc <= 1 or not 0 <= departure_soc <= 1
            or block_power_kw <= 0 or tau_blk_h <= 0
            or not 0 < efficiency <= 1 or capacity_kwh <= 0):
        raise InputValidationError("Invalid SOC qualification or reserve hardware parameters")
    return max(float(basic_soc), float(departure_soc) +
               float(block_power_kw) * float(tau_blk_h) / (float(efficiency) * float(capacity_kwh)))


def service_periods(k, tau_blk_h, horizon_h=24):
    if (type(k) is not int or not 1 <= k <= 24 or isinstance(tau_blk_h, bool)
            or tau_blk_h not in (1, 2) or horizon_h != 24):
        raise InputValidationError("Frozen service grid requires k=1..24, tau=1/2 hours, horizon=24")
    occupancy = tuple(t >= k and t - 1 < horizon_h for t in range(1, 25))
    reserve = tuple(t >= k and t - 1 + tau_blk_h <= horizon_h for t in range(1, 25))
    block = tuple(range(k, k + int(tau_blk_h)))
    compatible = all(1 <= t <= 24 and reserve[t - 1] for t in block)
    return occupancy, reserve, block, compatible


@dataclass(frozen=True)
class CandidateTask:
    task_id: str
    vehicle_id: str
    station_id: str
    start_period: int
    arc_ids: tuple[int, ...]
    arrival_h: float
    distance_km: float
    freeflow_h: float
    travel_time_h: float
    travel_energy_kwh: float
    arrival_soc: float
    usable_energy_kwh: float
    qualification_soc: float
    occupancy: tuple[bool, ...]
    reserve: tuple[bool, ...]
    block_periods: tuple[int, ...]


def route_tasks(vehicle_id, station_id, route, vehicle_parameters,
                basic_soc=.3, tau_blk_h=1):
    """Reuse one earliest SOC-qualified route across all compatible start k."""
    if route is None:
        return ()
    p = vehicle_parameters
    qualification = qualification_soc(basic_soc, float(p["SOC_dep_under_e"]),
                                      float(p["P_blk_under_e_kW"]), tau_blk_h,
                                      float(p["eta_dis_e"]), float(p["E_cap_e_kWh"]))
    if not math.isfinite(route.arrival_h) or route.arrival_soc < qualification:
        return ()
    usable = max(0., route.arrival_soc - float(p["SOC_dep_under_e"])) * float(p["E_cap_e_kWh"])
    tasks = []
    for k in range(1, 25):
        occupancy, reserve, block, compatible = service_periods(k, tau_blk_h)
        if route.arrival_h <= k - 1 and 24 - (k - 1) >= tau_blk_h and compatible:
            tasks.append(CandidateTask(
                f"{vehicle_id}__{station_id}__k{k:02d}", vehicle_id, station_id, k,
                tuple(route.arc_ids), float(route.arrival_h), float(route.distance_km),
                float(route.freeflow_h), float(route.arrival_h - route.departure_h),
                float(route.travel_energy_kwh), float(route.arrival_soc), float(usable), qualification,
                occupancy, reserve, block))
    return tuple(tasks)


@dataclass(frozen=True)
class ForecastTaskInputs:
    case: CaseInputs
    target: TargetInputs
    network: ForecastNetwork
    forecast_speed: np.ndarray
    forecast_travel_hours: np.ndarray
    input_hashes: Mapping[str, str]
    dependency_sha256: str


def load_forecast_task_inputs(case_root, event_id, lead_h=1.0):
    root = Path(case_root)
    case = load_case_inputs(root)
    target = case.target(event_id, lead_h)
    road = load_road_model(root, include_connectors=True)
    connectors = derive_road_connectors(root)
    with (root / "04_transportation/A__directed_road_arcs.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f)) + list(connectors.arc_rows)
    rows = sorted(rows, key=lambda row: int(row["a"]))
    if tuple(int(row["a"]) for row in rows) != road.arc_ids:
        raise InputValidationError("Forecast network must contain canonical arc IDs 1..88")
    arcs = tuple(Arc(int(row["a"]), int(row["m_road"]), int(row["n_road"]),
                     float(row["D_a_km"]), float(row["T0_a_min"]) / 60.) for row in rows)
    vectors = forecast_vector_field(target.forecast).sample(target.replay_times_h, road.points_xy_km)
    speed = np.linalg.norm(vectors, axis=-1)
    hazard = np.clip((speed - road.w_thr_m_s) / (road.w_sev_m_s - road.w_thr_m_s), 0, 1)
    travel = np.where(speed <= road.w_mov_m_s, road.t0_min[None, :] *
                      (1 + road.kappa_tt[None, :] * hazard) / 60., np.inf)
    # Semantic identity uses only forecast arrays actually consumed. Raw source
    # file hashes remain provenance, but an unused analysis-array edit cannot
    # alter the route/task dependency identity.
    dependency = canonical_hash({
        "target_id": target.target_id, "state_sha256": target.state_sha256,
        "arcs": [asdict(arc) for arc in arcs], "points_sha256": _array_hash(road.points_xy_km),
        "travel_hours_sha256": _array_hash(travel), "times_sha256": _array_hash(target.replay_times_h),
        "hardware": {key: dict(value) for key, value in case.vehicle_parameters.items()},
        "stations": {key: dict(value) for key, value in case.station_parameters.items()},
        "contract": generator_contract()})
    network = ForecastNetwork(arcs, target.replay_times_h, travel,
                              horizon_end_h=24., network_hash=dependency)
    hashes = dict(case.input_hashes)
    hashes.update(road.input_hashes)
    return ForecastTaskInputs(case, target, network, _readonly(speed), _readonly(travel),
                              MappingProxyType(hashes), dependency)


@dataclass(frozen=True)
class CandidateTaskBank:
    status: str
    certified: bool
    tasks: tuple[CandidateTask, ...]
    queries: tuple[dict, ...]
    vehicle_summaries: tuple[dict, ...]
    task_sha256: str
    dependency_sha256: str
    config: Mapping

    def as_dict(self):
        return {"status": self.status, "certified": self.certified,
                "tasks_provisional": not self.certified,
                "tasks": [asdict(task) for task in self.tasks],
                "queries": list(self.queries), "vehicle_summaries": list(self.vehicle_summaries),
                "task_sha256": self.task_sha256, "dependency_sha256": self.dependency_sha256,
                "config": dict(self.config), "task_count": len(self.tasks)}


def build_candidate_tasks(inputs, basic_soc=.3, tau_blk_h=1,
                          max_expansions=1000000, time_limit_seconds=30.,
                          overall_time_limit_seconds=540.):
    """Query all EV/station pairs, retain nearest three, then build H.

    Runtime limits affect certification and diagnostics, never the physical
    route domain or tie order. Unfinished results remain provisional.
    """
    if not isinstance(inputs, ForecastTaskInputs):
        raise InputValidationError("Expected forecast-only task inputs")
    compatible_starts = [k - 1 for k in range(1, 25) if service_periods(k, tau_blk_h)[3]]
    latest = float(max(compatible_starts))
    if (not math.isfinite(float(overall_time_limit_seconds)) or overall_time_limit_seconds <= 0
            or not math.isfinite(float(time_limit_seconds)) or time_limit_seconds <= 0):
        raise InputValidationError("Search runtime limits must be positive and finite")
    config = {"basic_soc": float(basic_soc), "tau_blk_h": float(tau_blk_h),
              "nearest_stations": 3, "latest_useful_arrival_h": latest,
              "route_domain": "all node-simple paths", "station_order": "arrival,distance,station_id"}
    deadline = time.monotonic() + float(overall_time_limit_seconds)
    cache, queries, tasks, summaries = {}, [], [], []
    all_complete = True
    for state in inputs.target.vehicle_states:
        parameters = inputs.case.vehicle_parameters[state.vehicle_id]
        qualification = qualification_soc(basic_soc, float(parameters["SOC_dep_under_e"]),
                                          float(parameters["P_blk_under_e_kW"]), tau_blk_h,
                                          float(parameters["eta_dis_e"]), float(parameters["E_cap_e_kWh"]))
        feasible, vehicle_complete = [], True
        for station_id in inputs.target.station_ids:
            station = inputs.case.station_parameters[station_id]
            arguments = {
                "origin": state.road_node, "destination": int(station["road_node_c"]),
                "departure_h": -inputs.target.lead_h, "initial_soc": state.soc,
                "capacity_kwh": float(parameters["E_cap_e_kWh"]),
                "energy_kwh_per_km": float(parameters["E_km_e_kWh_per_km"]),
                "auxiliary_kw": float(parameters["P_aux_e_kW"]),
                "qualification_soc": qualification, "latest_arrival_h": latest}
            query_id = canonical_hash({"dependency_sha256": inputs.dependency_sha256, "arguments": arguments})
            cached = query_id in cache
            remaining = deadline - time.monotonic()
            if cached:
                result = cache[query_id]
            elif remaining > 0:
                result = search_route(inputs.network, **arguments, max_expansions=max_expansions,
                                      time_limit_seconds=min(float(time_limit_seconds), remaining))
                cache[query_id] = result
            else:
                result = None
            status = "SEARCH_NOT_STARTED" if result is None else result.status
            route = None if result is None else result.route
            complete = status == "COMPLETE"
            vehicle_complete &= complete
            all_complete &= complete
            queries.append({"vehicle_id": state.vehicle_id, "station_id": station_id,
                            "query_id": query_id, "arguments": arguments, "cache_hit": cached,
                            "status": status, "route": None if route is None else asdict(route),
                            "certificate": {} if result is None else result.as_dict()["certificate"]})
            eligible = route_tasks(state.vehicle_id, station_id, route, parameters, basic_soc, tau_blk_h)
            if eligible:
                feasible.append((route.arrival_h, route.distance_km, station_id, eligible))
        retained = sorted(feasible, key=lambda item: item[:3])[:3]
        vehicle_tasks = [task for row in retained for task in row[3]]
        tasks.extend(vehicle_tasks)
        summaries.append({"vehicle_id": state.vehicle_id, "origin_node": state.road_node,
                          "initial_soc": state.soc, "qualification_soc": qualification,
                          "initial_soc_below_qualification": state.soc < qualification,
                          "search_complete": vehicle_complete, "tasks_provisional": not vehicle_complete,
                          "feasible_station_count": len(feasible),
                          "retained_station_ids": [row[2] for row in retained],
                          "task_count": len(vehicle_tasks),
                          "zero_distance_task_count": sum(task.distance_km == 0 for task in vehicle_tasks),
                          "empty_H": not vehicle_tasks})
    semantic_payload = {"dependency_sha256": inputs.dependency_sha256, "config": config,
                        "certified": all_complete, "tasks": [asdict(task) for task in tasks]}
    return CandidateTaskBank("COMPLETE" if all_complete else "SEARCH_INCOMPLETE", all_complete,
                             tuple(tasks), tuple(queries), tuple(summaries), canonical_hash(semantic_payload),
                             inputs.dependency_sha256, MappingProxyType(config))


def generator_contract():
    return {
        "version": "S01-I06-v1",
        "forecast_only": "Assignment-epoch states and native gust vectors from baseline forecast; no analysis/residual/scenario outcomes",
        "road": "88 directed arcs including own depot connectors; travel hours=T0_minutes*(1+kappa_tt*clip((speed-10)/15,0,1))/60, speed>Wmov gives infinity",
        "route_domain": "All node-simple paths; no waiting, cycles or realized rerouting; entry slot owns entire traversal",
        "energy": "Ekm*distance+Paux*max(0,arrival-departure-sum(freeflow_hours))",
        "route_order": "Minimum arrival among SOC-qualified routes; ties by distance then arc-ID tuple",
        "station_order": "Among forecast-feasible stations, minimum arrival then distance then station ID; retain first three",
        "start_reuse": "Departure/SOC qualification do not depend on k; globally earliest qualified route is reused, then each start deadline checked",
        "H": "Arrival<=k-1, qualified SOC, remaining>=tau, and every compulsory block period reserve-eligible; tau2 last k=22",
        "certification": "Any incomplete/not-started pair query makes entire H provisional; partial bank must not feed formal optimization",
        "identity": "Semantic route/H hash uses consumed forecast travel, network, assignment state, hardware, settings; source-file hashes are separate provenance",
    }
