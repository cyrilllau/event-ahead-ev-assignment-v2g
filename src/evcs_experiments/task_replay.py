"""Physical deliverability of certified, previously fixed forecast tasks.

No route search is performed. Failed routes use the manuscript's zero
Troute/Etravel/SOC accounting convention, not a claim of zero partial-trip
battery consumption. Output usable energy is battery-side DC energy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError
from .evaluation_access import require_oos_access
from .forecast_routes import ForecastNetwork
from .forecast_tasks import CandidateTaskBank, qualification_soc, service_periods
from .random_streams import canonical_hash

HARDWARE = ("E_cap_e_kWh", "E_km_e_kWh_per_km", "P_aux_e_kW", "SOC_dep_under_e",
            "P_blk_under_e_kW", "eta_dis_e")


def _readonly(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _value(row, key):
    return row[key] if isinstance(row, Mapping) else getattr(row, key)


def _array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


@dataclass(frozen=True)
class PreparedTaskBank:
    task_ids: tuple[str, ...]
    task_sha256: str
    dependency_sha256: str
    prepared_sha256: str
    departure_h: float
    tau_blk_h: float
    basic_soc: float
    groups: tuple[Mapping, ...]
    task_group_indices: np.ndarray
    start_periods: np.ndarray
    qualification_soc: np.ndarray
    capacity_kwh: np.ndarray
    departure_soc: np.ndarray
    metadata: Mapping


@dataclass(frozen=True)
class TaskReplay:
    task_ids: tuple[str, ...]
    arrival_h: np.ndarray
    gamma_route: np.ndarray
    route_time_h: np.ndarray
    travel_energy_kwh: np.ndarray
    arrival_soc: np.ndarray
    varsigma: np.ndarray
    mu_full: np.ndarray
    usable_energy_dc_kwh: np.ndarray
    status: tuple[str, ...]
    failed_arc_ids: np.ndarray
    departure_h: float
    task_sha256: str
    network_sha256: str
    replay_sha256: str
    unique_routes_replayed: int
    metadata: Mapping

    def as_dict(self):
        """Small JSON-ready index/provenance; numerical arrays belong in NPZ."""
        return {"task_ids": list(self.task_ids), "task_count": len(self.task_ids),
            "departure_h": self.departure_h, "task_sha256": self.task_sha256,
            "network_sha256": self.network_sha256, "replay_sha256": self.replay_sha256,
            "unique_routes_replayed": self.unique_routes_replayed,
            "status_counts": {s: self.status.count(s) for s in sorted(set(self.status))},
            "metadata": dict(self.metadata)}


def task_deliverability(arrival_h, gamma_route, arrival_soc, start_period,
                       capacity_kwh, departure_soc, qualification_soc, tau_blk_h):
    """Scalar manuscript success, 24 hourly availability, and battery DC energy."""
    if (type(start_period) is not int or not 1 <= start_period <= 24
            or tau_blk_h not in (1, 2) or isinstance(tau_blk_h, bool)
            or not math.isfinite(capacity_kwh) or capacity_kwh <= 0
            or not math.isfinite(arrival_soc) or not 0 <= arrival_soc <= 1
            or not math.isfinite(departure_soc) or not 0 <= departure_soc <= 1
            or not math.isfinite(qualification_soc) or qualification_soc < 0
            or math.isnan(arrival_h) or arrival_h == -math.inf
            or bool(gamma_route) != math.isfinite(arrival_h)):
        raise InputValidationError("Invalid task deliverability inputs")
    available_from = max(arrival_h, float(start_period - 1))
    success = bool(gamma_route and arrival_soc >= qualification_soc
                   and 24.0 - available_from >= tau_blk_h)
    starts = np.arange(24, dtype=float)
    mu = success & (available_from <= starts) & (starts + tau_blk_h <= 24)
    usable = max(arrival_soc - departure_soc, 0.) * capacity_kwh if success else 0.
    return success, _readonly(mu, bool), float(usable)


def prepare_task_bank(frozen_bank, vehicle_parameters, initial_states):
    """Validate and index H once, then reuse it across physical scenarios.

    The I06 semantic task hash is checked before any grouping. Frozen query
    inputs bind every task to its original origin, SOC and vehicle hardware.
    """
    bank = frozen_bank.as_dict() if isinstance(frozen_bank, CandidateTaskBank) else frozen_bank
    if (not isinstance(bank, Mapping) or bank.get("status") != "COMPLETE"
            or bank.get("certified") is not True or bank.get("tasks_provisional", False)):
        raise InputValidationError("Physical replay requires a certified complete frozen H")
    tasks, config = bank["tasks"], dict(bank["config"])
    digest = canonical_hash({"dependency_sha256": bank["dependency_sha256"],
        "config": config, "certified": True, "tasks": tasks})
    if digest != bank["task_sha256"]:
        raise InputValidationError("Frozen task semantic hash mismatch")
    tau, basic = float(config["tau_blk_h"]), float(config["basic_soc"])
    if tau not in (1, 2) or not math.isfinite(basic) or not 0 <= basic <= 1:
        raise InputValidationError("Unsupported frozen qualification configuration")
    state_source = getattr(initial_states, "vehicle_states", initial_states)
    if isinstance(state_source, Mapping):
        states = dict(state_source)
    else:
        states = {_value(row, "vehicle_id"): row for row in state_source}
        if len(states) != len(state_source):
            raise InputValidationError("Duplicate initial vehicle state")
    queries, departures = {}, set()
    for query in bank["queries"]:
        key = (query["vehicle_id"], query["station_id"])
        if key in queries or query["status"] != "COMPLETE":
            raise InputValidationError("Frozen H has duplicate or incomplete source queries")
        args = query["arguments"]
        if key[0] not in states or key[0] not in vehicle_parameters:
            raise InputValidationError("Missing original vehicle state or hardware")
        state, hardware = states[key[0]], vehicle_parameters[key[0]]
        expected = {"origin": int(_value(state, "road_node")), "initial_soc": float(_value(state, "soc")),
            "capacity_kwh": float(hardware["E_cap_e_kWh"]),
            "energy_kwh_per_km": float(hardware["E_km_e_kWh_per_km"]),
            "auxiliary_kw": float(hardware["P_aux_e_kW"]),
            "qualification_soc": qualification_soc(basic, float(hardware["SOC_dep_under_e"]),
                float(hardware["P_blk_under_e_kW"]), tau, float(hardware["eta_dis_e"]),
                float(hardware["E_cap_e_kWh"]))}
        if any(not math.isfinite(float(args[k])) or float(args[k]) != v for k, v in expected.items()):
            raise InputValidationError("Replay state/hardware differs from the frozen forecast query")
        departure = float(args["departure_h"])
        if not math.isfinite(departure) or departure not in (-2., -1., -.5, 0.):
            raise InputValidationError("Invalid frozen assignment departure")
        departures.add(departure)
        queries[key] = query
    if len(departures) != 1:
        raise InputValidationError("Frozen task bank must have exactly one assignment epoch")
    original_departure = departures.pop()
    groups, group_lookup, indices, ids, periods, quals, caps, dep_socs = [], {}, [], [], [], [], [], []
    state_hardware = {}
    for task in tasks:
        vehicle_id, station_id, k = task["vehicle_id"], task["station_id"], task["start_period"]
        if (type(k) is not int or task["task_id"] != f"{vehicle_id}__{station_id}__k{k:02d}"
                or task["task_id"] in ids or (vehicle_id, station_id) not in queries):
            raise InputValidationError("Task identity is duplicate or inconsistent with frozen queries")
        query, state = queries[(vehicle_id, station_id)], states[vehicle_id]
        p = {key: float(vehicle_parameters[vehicle_id][key]) for key in HARDWARE}
        if (not all(math.isfinite(v) for v in p.values()) or p["E_cap_e_kWh"] <= 0
                or p["E_km_e_kWh_per_km"] < 0 or p["P_aux_e_kW"] < 0
                or not 0 <= float(_value(state, "soc")) <= 1):
            raise InputValidationError("Invalid frozen hardware or initial SOC")
        occupancy, reserve, block, compatible = service_periods(k, tau)
        qual = qualification_soc(basic, p["SOC_dep_under_e"], p["P_blk_under_e_kW"],
                                  tau, p["eta_dis_e"], p["E_cap_e_kWh"])
        route = query["route"]
        if (route is None or tuple(task["arc_ids"]) != tuple(route["arc_ids"])
                or tuple(task["occupancy"]) != occupancy or tuple(task["reserve"]) != reserve
                or tuple(task["block_periods"]) != block or not compatible
                or task["qualification_soc"] != qual or task["arrival_soc"] < qual
                or task["arrival_h"] > k - 1):
            raise InputValidationError("Frozen task violates its certified route or qualification")
        group = {"vehicle_id": vehicle_id, "station_id": station_id,
            "arc_ids": tuple(task["arc_ids"]), "origin": int(_value(state, "road_node")),
            "destination": int(query["arguments"]["destination"]),
            "initial_soc": float(_value(state, "soc")), "hardware": MappingProxyType(p),
            "distance_km": float(task["distance_km"]), "freeflow_h": float(task["freeflow_h"])}
        key = (vehicle_id, station_id, group["arc_ids"], group["origin"], group["destination"],
               group["initial_soc"], tuple(p.items()), group["distance_km"], group["freeflow_h"])
        if key not in group_lookup:
            group_lookup[key] = len(groups)
            groups.append(MappingProxyType(group))
        indices.append(group_lookup[key]); ids.append(task["task_id"]); periods.append(k)
        quals.append(qual); caps.append(p["E_cap_e_kWh"]); dep_socs.append(p["SOC_dep_under_e"])
        state_hardware[vehicle_id] = {"origin": group["origin"], "initial_soc": group["initial_soc"],
                                      "hardware": p}
    prepared_hash = canonical_hash({"task_sha256": digest, "state_hardware": state_hardware,
        "departure_h": original_departure, "contract": generator_contract()})
    return PreparedTaskBank(tuple(ids), digest, bank["dependency_sha256"], prepared_hash,
        original_departure, tau, basic, tuple(groups), _readonly(indices, np.int64),
        _readonly(periods, np.int64), _readonly(quals), _readonly(caps), _readonly(dep_socs),
        MappingProxyType({"state_hardware_sha256": canonical_hash(state_hardware),
                          "config": config, "task_count": len(ids)}))


def _physical(network, group, lookup, departure):
    ids = group["arc_ids"]
    node, seen, distance, freeflow = group["origin"], {group["origin"]}, 0., 0.
    for aid in ids:
        if aid not in lookup:
            raise InputValidationError("Fixed route arc is absent from the physical network")
        _, arc = lookup[aid]
        if arc.source != node or arc.target in seen:
            raise InputValidationError("Frozen physical route is not contiguous and node-simple")
        node = arc.target; seen.add(node); distance += arc.distance_km; freeflow += arc.freeflow_h
    if (node != group["destination"] or abs(distance - group["distance_km"]) > 1e-10
            or abs(freeflow - group["freeflow_h"]) > 1e-10):
        raise InputValidationError("Realized network changed frozen endpoints, distance, or free-flow geometry")
    arrival = departure
    for aid in ids:
        index, arc = lookup[aid]
        slot = network.slot_index(arrival)
        if slot is None:
            return math.inf, False, 0., 0., 0., "ENTRY_OUTSIDE_HORIZON", aid
        travel = float(network.travel_hours[slot, index])
        if not math.isfinite(travel):
            return math.inf, False, 0., 0., 0., "ENTRY_IMPASSABLE", aid
        arrival += travel
    p = group["hardware"]
    route_time = arrival - departure
    energy = p["E_km_e_kWh_per_km"] * distance + p["P_aux_e_kW"] * max(route_time - freeflow, 0.)
    soc = max(group["initial_soc"] - energy / p["E_cap_e_kWh"], 0.)
    return arrival, True, route_time, energy, soc, "ROUTE_COMPLETE", -1


def replay_prepared(network, prepared, departure_h=None, *, split="planning",
                    development=False, scenario_identity=None):
    if not isinstance(network, ForecastNetwork) or not isinstance(prepared, PreparedTaskBank):
        raise InputValidationError("Expected physical ForecastNetwork and validated PreparedTaskBank")
    if split not in ("planning", "calibration", "oos") or type(development) is not bool:
        raise InputValidationError("Invalid split or development domain")
    if split == "oos" and not development:
        identity = {} if scenario_identity is None else dict(scenario_identity)
        required = ("target_event", "base_weather_event", "scenario_id", "regime")
        if (any(not isinstance(identity.get(key), str) or not identity[key] for key in required)
                or type(identity.get("variant_index")) is not int):
            raise InputValidationError("Formal OOS replay requires complete target/base/variant/sample/regime identity")
        require_oos_access(target_event=identity.get("target_event"),
            base_event=identity.get("base_weather_event"),
            variant_index=identity.get("variant_index"),
            sample_id=identity.get("scenario_id"), regime=identity.get("regime"))
    departure = prepared.departure_h if departure_h is None else float(departure_h)
    if not math.isfinite(departure) or departure not in (prepared.departure_h, 0.):
        raise InputValidationError("Replay allows frozen EA departure or DSP departure zero using the same initial state")
    if (network.horizon_end_h != 24 or np.any(np.abs(np.diff(network.slot_starts_h) - .25) > 1e-10)
            or network.slot_starts_h[0] > departure or network.slot_starts_h[-1] != 23.75):
        raise InputValidationError("Physical replay requires contiguous quarter-hour slots covering departure through 23.75")
    lookup = {arc.a: (index, arc) for index, arc in enumerate(network.arcs)}
    raw = [_physical(network, group, lookup, departure) for group in prepared.groups]
    indices = prepared.task_group_indices
    def field(column, dtype=float):
        return _readonly([raw[int(i)][column] for i in indices], dtype)
    arrival, gamma, route_time, energy, soc = (field(0), field(1, bool), field(2), field(3), field(4))
    available_from = np.maximum(arrival, prepared.start_periods - 1.)
    success = gamma & (soc >= prepared.qualification_soc) & (24 - available_from >= prepared.tau_blk_h)
    starts = np.arange(24, dtype=float)
    mu = success[:, None] & (available_from[:, None] <= starts) & (starts + prepared.tau_blk_h <= 24)
    usable = np.where(success, np.maximum(soc - prepared.departure_soc, 0.) * prepared.capacity_kwh, 0.)
    statuses = tuple(raw[int(i)][5] if not gamma[j] else
        "TASK_QUALIFIED" if success[j] else "SOC_UNQUALIFIED" if soc[j] < prepared.qualification_soc[j]
        else "INSUFFICIENT_SERVICE_TIME" for j, i in enumerate(indices))
    failed = field(6, np.int64)
    arrays = {"arrival_h": arrival, "gamma_route": gamma, "route_time_h": route_time,
        "travel_energy_kwh": energy, "arrival_soc": soc, "varsigma": _readonly(success, bool),
        "mu_full": _readonly(mu, bool), "usable_energy_dc_kwh": _readonly(usable), "failed_arc_ids": failed}
    scenario = {} if scenario_identity is None else dict(scenario_identity)
    metadata = {"prepared_sha256": prepared.prepared_sha256, "dependency_sha256": prepared.dependency_sha256,
        "split": split, "development": development, "scenario_identity": scenario,
        "array_hashes": {key: _array_hash(value) for key, value in arrays.items()},
        "tau_blk_h": prepared.tau_blk_h, "basic_soc": prepared.basic_soc,
        "failed_route_accounting": "Troute/Etravel/SOC zero is task accounting, not zero partial-trip consumption",
        "usable_energy_side": "battery DC; discharge efficiency is applied later by the physical power model"}
    digest = canonical_hash({"prepared_sha256": prepared.prepared_sha256, "network_sha256": network.network_hash,
        "departure_h": departure, "metadata": metadata, "status": statuses})
    return TaskReplay(prepared.task_ids, **arrays, status=statuses, departure_h=departure,
        task_sha256=prepared.task_sha256, network_sha256=network.network_hash, replay_sha256=digest,
        unique_routes_replayed=len(raw), metadata=MappingProxyType(metadata))


def replay_task_bank(network, frozen_bank, vehicle_parameters, initial_states, departure_h=None,
                     *, split="planning", development=False, scenario_identity=None):
    return replay_prepared(network, prepare_task_bank(frozen_bank, vehicle_parameters, initial_states),
        departure_h, split=split, development=development, scenario_identity=scenario_identity)


def generator_contract():
    return {"version": "S01-I07-v1", "frozen_H": "Require COMPLETE/certified and recompute I06 semantic task hash",
        "replay": "Fixed node-simple route, no search/wait/rerouting; half-open quarter-hour entry fixes traversal",
        "failure": "arrival=inf, Gamma=0, Troute=Etravel=SOC=0 by failed-task accounting convention",
        "energy": "Ekm*distance+Paux*max(0,arrival-departure-freeflow); usable energy is battery DC",
        "success": "Gamma and SOCarr>=SOCqual and 24-max(arrival,k-1)>=tau",
        "availability": "varsigma and max(arrival,k-1)<=t-1 and t-1+tau<=24",
        "dsp": "Only departure changes to zero; same original EA state/H/route; no extra waiting-energy charge",
        "oos": "Formal physical replay locked until S04"}
