"""Sparse task assignment and AC reserve procurement, paper first-stage equations.

Columns are x[H], then C-order r[H,T]. Rows use lower <= A @ z <= upper;
all powers are kW AC, and forecast energy is battery-side kWh DC. This module
neither selects routes nor introduces forecast dispatch or recourse variables.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix


def _readonly(value, dtype=float):
    array = np.asarray(value, dtype=dtype)
    # Immutable bytes prevent callers from re-enabling WRITEABLE on plan data.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _identity(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _ids(value, name, *, unique=False):
    result = tuple(value)
    for item in result:
        _identity(item, name)
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _numeric(value, shape, name, *, positive=False):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    if np.any(result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a binary array with shape {shape}")
    return np.array(array, dtype=bool, copy=True)


@dataclass(frozen=True)
class FirstStagePlan:
    x: np.ndarray
    r: np.ndarray
    task_sha256: str
    config_hash: str

    def __post_init__(self):
        x, r = _readonly(self.x), _readonly(self.r)
        if x.ndim != 1 or r.ndim != 2 or r.shape[0] != len(x):
            raise ValueError("A plan requires x[H] and r[H,T]")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(r)):
            raise ValueError("Plan values must be finite")
        _identity(self.task_sha256, "task_sha256")
        _identity(self.config_hash, "config_hash")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "r", r)


@dataclass(frozen=True)
class PlanEvaluation:
    feasible: bool
    objective_usd: float
    c_act_usd: float
    c_cap_usd: float
    max_row_violation: float
    max_bound_violation: float
    max_integrality_violation: float
    row_activity: np.ndarray
    row_violation: np.ndarray
    bound_violation: np.ndarray
    integrality_violation: np.ndarray
    tolerance: float

    def as_dict(self):
        return {"feasible": self.feasible, "objective_usd": self.objective_usd,
            "objective_components": {"F_act": self.c_act_usd, "F_cap": self.c_cap_usd,
                                     "F_pre": self.objective_usd},
            "max_row_violation": self.max_row_violation,
            "max_bound_violation": self.max_bound_violation,
            "max_integrality_violation": self.max_integrality_violation,
            "tolerance": self.tolerance,
            "violating_row_indices": np.flatnonzero(self.row_violation > self.tolerance).tolist(),
            "violating_bound_indices": np.flatnonzero(self.bound_violation > self.tolerance).tolist(),
            "violating_integrality_indices": np.flatnonzero(self.integrality_violation > self.tolerance).tolist()}


@dataclass(frozen=True)
class SparseFirstStage:
    A: csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    integrality: np.ndarray
    c: np.ndarray
    variable_map: tuple
    constraint_map: tuple
    task_ids: tuple[str, ...]
    periods: tuple[int, ...]
    task_sha256: str
    config_hash: str
    matrix_sha256: str
    units: object

    def vector(self, plan: FirstStagePlan) -> np.ndarray:
        if plan.task_sha256 != self.task_sha256 or plan.config_hash != self.config_hash:
            raise ValueError("Plan task/config identity does not match the assembled first stage")
        if plan.x.shape != (len(self.task_ids),) or plan.r.shape != (len(self.task_ids), len(self.periods)):
            raise ValueError("Plan shape does not match the assembled task and period axes")
        return _readonly(np.concatenate((plan.x, plan.r.ravel(order="C"))))

    def evaluate(self, plan: FirstStagePlan, tolerance: float = 1e-8) -> PlanEvaluation:
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("The absolute evaluation tolerance must be finite and nonnegative")
        vector = self.vector(plan)
        activity = self.A @ vector
        row_violation = np.maximum(0., np.maximum(self.row_lower - activity, activity - self.row_upper))
        bound_violation = np.maximum(0., np.maximum(self.lower_bounds - vector, vector - self.upper_bounds))
        integer_violation = np.where(self.integrality == 1, np.abs(vector - np.rint(vector)), 0.)
        maxima = [float(np.max(values, initial=0.)) for values in (row_violation, bound_violation, integer_violation)]
        h = len(self.task_ids)
        activation, capacity = float(self.c[:h] @ vector[:h]), float(self.c[h:] @ vector[h:])
        if not np.all(np.isfinite(activity)) or not np.isfinite(activation + capacity):
            raise ValueError("Plan evaluation overflowed its original-unit numeric representation")
        return PlanEvaluation(all(value <= tolerance for value in maxima), activation + capacity,
            activation, capacity, *maxima, _readonly(activity), _readonly(row_violation),
            _readonly(bound_violation), _readonly(integer_violation), float(tolerance))

    def as_dict(self):
        """JSON metadata; the numeric CSR arrays are exported separately by callers."""
        return {"task_ids": list(self.task_ids), "periods": list(self.periods),
            "task_sha256": self.task_sha256, "config_hash": self.config_hash,
            "matrix_sha256": self.matrix_sha256, "shape": list(self.A.shape), "nnz": self.A.nnz,
            "column_order": "x[H], then r[H,T] flattened in C order",
            "constraint_form": "row_lower <= A @ z <= row_upper",
            "objective_eq_label": "eq:first_stage_cost", "units": dict(self.units),
            "variable_map": [dict(row) for row in self.variable_map],
            "constraint_map": [dict(row) for row in self.constraint_map]}


def assemble_first_stage(case, tasks) -> SparseFirstStage:
    """Assemble the six first-stage constraint families without changing H."""
    periods = tuple(case.periods)
    if not periods or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or t <= 0 for t in periods):
        raise ValueError("periods must contain positive integers")
    if any(b <= a for a, b in zip(periods, periods[1:])):
        raise ValueError("periods must be strictly increasing")
    periods = tuple(int(t) for t in periods)
    stations = _ids(case.station_ids, "station_ids", unique=True)
    if not stations or len(tuple(case.station_bus_ids)) != len(stations):
        raise ValueError("Each station requires a bus mapping")
    task_ids = _ids(tasks.task_ids, "task_ids", unique=True)
    vehicles = _ids(tasks.vehicle_ids, "vehicle_ids")
    task_stations = _ids(tasks.station_ids, "task station_ids")
    h_count, t_count, c_count = len(task_ids), len(periods), len(stations)
    if len(vehicles) != h_count or len(task_stations) != h_count or not set(task_stations) <= set(stations):
        raise ValueError("Task vehicle/station mappings do not match the frozen task axis")
    dt = float(case.dt_h)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_h must be positive and finite")
    plugs = _numeric(case.station_plugs, (c_count,), "station_plugs")
    if not np.all(plugs == np.rint(plugs)):
        raise ValueError("station_plugs must be integer counts")
    interface = _numeric(case.station_interface_kw, (c_count,), "station_interface_kw")
    occ = _mask(tasks.chi_occ, (h_count, t_count), "chi_occ")
    res = _mask(tasks.chi_res, (h_count, t_count), "chi_res")
    block = _mask(tasks.block_mask, (h_count, t_count), "block_mask")
    if np.any(res & ~occ) or np.any(block & ~res) or np.any(~block.any(axis=1)):
        raise ValueError("Every minimum block must be nonempty and a subset of reserve eligibility and occupancy")
    p_dis = _numeric(tasks.p_dis_kw, (h_count,), "p_dis_kw", positive=True)
    p_block = _numeric(tasks.p_block_kw, (h_count,), "p_block_kw", positive=True)
    eta = _numeric(tasks.eta_dis, (h_count,), "eta_dis", positive=True)
    if np.any(p_block > p_dis) or np.any(eta > 1):
        raise ValueError("Pblock must not exceed Pdis, and eta_dis must not exceed one")
    energy = _numeric(tasks.e_forecast_dc_kwh, (h_count,), "e_forecast_dc_kwh")
    c_act = _numeric(tasks.c_act_usd, (h_count,), "c_act_usd")
    c_cap = _numeric(tasks.c_cap_usd_per_kwh, (h_count, t_count), "c_cap_usd_per_kwh")
    task_hash = _identity(tasks.task_sha256, "task_sha256")
    config_hash = _identity(tasks.config_hash, "config_hash")
    variable_count = h_count * (1 + t_count)
    by_vehicle, by_station = {}, {station: [] for station in stations}
    for h, (vehicle, station) in enumerate(zip(vehicles, task_stations)):
        by_vehicle.setdefault(vehicle, []).append(h)
        by_station[station].append(h)
    variable_map = []
    for h, task in enumerate(task_ids):
        variable_map.append(MappingProxyType({"index": h, "name": "x", "task_id": task,
            "vehicle_id": vehicles[h], "station_id": task_stations[h], "period": None,
            "unit": "dimensionless", "eq_label": "eq:one_task_per_ev"}))
    for h, task in enumerate(task_ids):
        for t, period in enumerate(periods):
            variable_map.append(MappingProxyType({"index": h_count + h*t_count + t,
                "name": "r", "task_id": task, "vehicle_id": vehicles[h],
                "station_id": task_stations[h], "period": period, "unit": "kW_AC",
                "eq_label": "eq:reserve_procurement_upper_bound"}))
    rows, columns, values, upper, constraint_map = [], [], [], [], []

    def add(family, eq_label, unit, coefficients, rhs, **index):
        row = len(upper)
        for column, value in coefficients:
            if value != 0:
                rows.append(row)
                columns.append(column)
                values.append(float(value))
        upper.append(float(rhs))
        constraint_map.append(MappingProxyType({"index": row, "name": family,
            "unit": unit, "eq_label": eq_label, "task_id": None, "vehicle_id": None,
            "station_id": None, "period": None, **index}))

    def r_col(h, t):
        return h_count + h*t_count + t

    for vehicle, indices in by_vehicle.items():
        add("one_task_per_ev", "eq:one_task_per_ev", "task_count", ((h, 1.) for h in indices), 1., vehicle_id=vehicle)
    for h, task in enumerate(task_ids):
        for t, period in enumerate(periods):
            add("reserve_upper", "eq:reserve_procurement_upper_bound", "kW_AC",
                ((h, -p_dis[h]*res[h, t]), (r_col(h, t), 1.)), 0., task_id=task, period=period)
    for h, task in enumerate(task_ids):
        for t in np.flatnonzero(block[h]):
            add("minimum_block", "eq:minimum_block_reserve", "kW_AC",
                ((h, p_block[h]), (r_col(h, t), -1.)), 0., task_id=task, period=periods[t])
    for h, task in enumerate(task_ids):
        add("forecast_energy", "eq:forecast_reserve_energy_envelope", "kWh_DC",
            [(h, -energy[h]), *((r_col(h, t), dt/eta[h]) for t in range(t_count))], 0., task_id=task)
    for c, station in enumerate(stations):
        for t, period in enumerate(periods):
            add("plug_occupancy", "eq:plug_reservation", "plug_count",
                ((h, occ[h, t]) for h in by_station[station]), plugs[c], station_id=station, period=period)
    for c, station in enumerate(stations):
        for t, period in enumerate(periods):
            add("station_interface", "eq:station_interface_limit", "kW_AC",
                ((r_col(h, t), 1.) for h in by_station[station]), interface[c], station_id=station, period=period)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(upper), variable_count), dtype=float)
    matrix.sum_duplicates()
    matrix.sort_indices()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.setflags(write=False)
    row_lower, row_upper = _readonly(np.full(len(upper), -np.inf)), _readonly(upper)
    lower_bounds = _readonly(np.zeros(variable_count))
    upper_bounds = _readonly(np.r_[np.ones(h_count), np.full(h_count*t_count, np.inf)])
    integrality = _readonly(np.r_[np.ones(h_count), np.zeros(h_count*t_count)], np.int8)
    objective = _readonly(np.r_[c_act, (c_cap*dt).ravel(order="C")])
    units = MappingProxyType({"x": "dimensionless", "r": "kW_AC", "energy": "kWh_DC",
                             "dt": "h", "objective": "USD", "c_act": "USD/task", "c_cap": "USD/kWh_AC"})
    digest = hashlib.sha256(json.dumps({"task_sha256": task_hash, "config_hash": config_hash,
        "task_ids": task_ids, "periods": periods, "variable_map": [dict(v) for v in variable_map],
        "constraint_map": [dict(v) for v in constraint_map], "units": dict(units)}, sort_keys=True).encode())
    for array in (matrix.data, matrix.indices, matrix.indptr, row_lower, row_upper,
                  lower_bounds, upper_bounds, integrality, objective):
        canonical = np.asarray(array, dtype="<i8" if array.dtype.kind in "iu" else "<f8")
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes(order="C"))
    return SparseFirstStage(matrix, row_lower, row_upper, lower_bounds, upper_bounds,
        integrality, objective, tuple(variable_map), tuple(constraint_map), task_ids, periods,
        task_hash, config_hash, digest.hexdigest(), units)
