"""Independent physical checks for the paper's first-stage task/reserve plan.

This module does not import the optimization assembler or inspect its matrix.
It evaluates the seven physical inequalities and procurement payments directly
in kW, battery-side kWh, task selections and occupied plugs. Inputs are read
without rounding, clipping, changing selection or reordering the frozen H.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import get_paths


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    _require(result.shape == shape, f"{name} has shape {result.shape}; expected {shape}")
    _require(np.isfinite(result).all(), f"{name} must contain finite values")
    return result


def _mask(value, shape, name):
    result = _array(value, shape, name)
    _require(np.all((result == 0) | (result == 1)), f"{name} must be binary")
    return result.astype(bool)


def _policy(value):
    if value is None:
        path = get_paths().spec_root / "runtime_policy.json"
        policy = json.loads(path.read_text())
        source = str(path)
    elif isinstance(value, Mapping):
        policy, source = value, "explicit_mapping"
    else:
        path = Path(value)
        policy, source = json.loads(path.read_text()), str(path)
    result = {key: float(policy[key]) for key in (
        "LP_scaled_feasibility_tolerance", "physical_power_residual_tolerance_kW_kvar")}
    _require(all(np.isfinite(v) and v > 0 for v in result.values()), "Invalid feasibility tolerance")
    return result, source


def check_first_stage(case, tasks, plan, *, runtime_policy=None):
    """Return a JSON-serializable, independent physical feasibility report.

    Malformed/nonfinite input or an identity mismatch raises ValueError;
    finite but infeasible decisions return ``passed=False``. Each inequality
    must satisfy both the runtime policy's normalized LP tolerance and its
    original-unit tolerance. The normalization denominator is the relevant
    fixed physical capacity, floored at 1 in its stated unit; it does not
    depend on a solver's row scaling or a candidate's magnitude.

    No standalone energy tolerance is specified by runtime_policy. The DC
    envelope's original-unit budget is the sum of per-period power tolerance
    converted to battery energy: T * dt / eta * power_tolerance. The stricter
    normalized LP test still applies to every envelope. Task/plug counts and
    binary decisions use the normalized LP tolerance also in original units.
    """
    policy, policy_source = _policy(runtime_policy)
    scaled_tol = policy["LP_scaled_feasibility_tolerance"]
    power_tol = policy["physical_power_residual_tolerance_kW_kvar"]
    task_ids = tuple(tasks.task_ids)
    vehicle_ids = tuple(tasks.vehicle_ids)
    task_station_ids = tuple(tasks.station_ids)
    station_ids = tuple(case.station_ids)
    periods = tuple(case.periods)
    h, t, c = len(task_ids), len(periods), len(station_ids)
    _require(t > 0 and periods == tuple(range(1, t + 1)) and
             all(isinstance(p, (int, np.integer)) and not isinstance(p, (bool, np.bool_)) for p in periods),
             "Periods must be consecutive one-based operating periods")
    _require(c > 0 and len(set(station_ids)) == c, "Station IDs must be nonempty and unique")
    _require(len(set(task_ids)) == h and len(vehicle_ids) == h and len(task_station_ids) == h,
             "Frozen task identities must be unique and aligned")
    _require(all(isinstance(v, str) and v for v in (*task_ids, *vehicle_ids, *task_station_ids, *station_ids)),
             "Task/vehicle/station IDs must be nonempty strings")
    _require(set(task_station_ids) <= set(station_ids), "Task references an unknown station")
    for key in ("task_sha256", "config_hash"):
        expected = getattr(tasks, key)
        _require(isinstance(expected, str) and expected and getattr(plan, key) == expected,
                 f"Plan {key} does not match the frozen task bank")
    dt = float(case.dt_h)
    _require(np.isfinite(dt) and dt > 0, "dt_h must be positive and finite")
    plugs = _array(case.station_plugs, (c,), "station_plugs")
    interface = _array(case.station_interface_kw, (c,), "station_interface_kw")
    _require(np.all(plugs >= 0) and np.all(plugs == np.floor(plugs)), "Plug capacities must be nonnegative integers")
    _require(np.all(interface >= 0), "Interface capacities must be nonnegative")
    occ = _mask(tasks.chi_occ, (h, t), "chi_occ")
    eligible = _mask(tasks.chi_res, (h, t), "chi_res")
    block = _mask(tasks.block_mask, (h, t), "block_mask")
    _require(np.all(~eligible | occ) and np.all(~block | eligible),
             "Block/reserve masks must be contained in occupancy")
    _require(h == 0 or np.all(np.any(block, axis=1)), "Every retained task requires a nonempty minimum block")
    values = {name: _array(getattr(tasks, name), (h,), name) for name in (
        "p_dis_kw", "p_block_kw", "eta_dis", "e_forecast_dc_kwh", "c_act_usd")}
    p_dis, p_block, eta, energy, c_act = (values[name] for name in (
        "p_dis_kw", "p_block_kw", "eta_dis", "e_forecast_dc_kwh", "c_act_usd"))
    c_cap = _array(tasks.c_cap_usd_per_kwh, (h, t), "c_cap_usd_per_kwh")
    _require(np.all(p_dis >= 0) and np.all(p_block >= 0) and np.all(p_block <= p_dis),
             "Discharge/block powers must obey 0 <= Pblock <= Pdis")
    _require(np.all((eta > 0) & (eta <= 1)), "Discharge efficiencies must be in (0,1]")
    _require(np.all(energy >= 0) and np.all(c_act >= 0) and np.all(c_cap >= 0),
             "Energy and procurement prices must be nonnegative")
    # When adapter timing fields are present, independently verify the paper's
    # mask definitions instead of silently trusting stale service masks.
    if hasattr(tasks, "start_periods") or hasattr(tasks, "tau_blk_h"):
        _require(hasattr(tasks, "start_periods") and hasattr(tasks, "tau_blk_h"),
                 "Task timing metadata must contain both start_periods and tau_blk_h")
        starts = _array(tasks.start_periods, (h,), "start_periods")
        tau = float(tasks.tau_blk_h)
        _require(np.isfinite(tau) and tau > 0 and round(tau / dt) >= 1 and
                 np.isclose(tau / dt, round(tau / dt), atol=1e-12, rtol=0),
                 "Minimum reserve block must span a positive integer number of periods")
        _require(np.all((starts >= 1) & (starts <= t) & (starts == np.floor(starts))),
                 "Invalid task start periods")
        hours = np.arange(t, dtype=float) * dt
        expected_occ = np.arange(1, t + 1)[None, :] >= starts[:, None]
        expected_res = expected_occ & (hours[None, :] + tau <= t * dt + 1e-12)
        expected_block = expected_occ & (np.arange(1, t + 1)[None, :] <= starts[:, None] - 1 + round(tau / dt))
        _require(np.array_equal(occ, expected_occ) and np.array_equal(eligible, expected_res) and
                 np.array_equal(block, expected_block), "Task masks disagree with paper service timing")

    x = _array(plan.x, (h,), "plan.x")
    reserve = _array(plan.r, (h, t), "plan.r")
    vehicle_order = tuple(dict.fromkeys(vehicle_ids))
    vehicle_selection = np.array([sum(float(x[j]) for j, ev in enumerate(vehicle_ids) if ev == vehicle)
                                  for vehicle in vehicle_order])
    occupancy = np.zeros((c, t))
    station_reserve = np.zeros((c, t))
    station_lookup = {station: j for j, station in enumerate(station_ids)}
    for j, station in enumerate(task_station_ids):
        occupancy[station_lookup[station]] += occ[j] * x[j]
        station_reserve[station_lookup[station]] += reserve[j]
    reserved_dc = np.sum(reserve, axis=1) * dt / eta
    residuals = {}

    def location(kind, index):
        if kind == "task_period":
            return {"task_id": task_ids[index[0]], "period": int(periods[index[1]])}
        if kind == "station_period":
            return {"station_id": station_ids[index[0]], "period": int(periods[index[1]])}
        if kind == "vehicle":
            return {"vehicle_id": vehicle_order[index[0]]}
        return {"task_id": task_ids[index[0]]}

    def record(name, excess, scale, original_tol, unit, label, kind, active=None):
        excess = np.asarray(excess, dtype=float)
        _require(np.isfinite(excess).all(), f"Nonfinite arithmetic in {name}")
        positive = np.maximum(excess, 0.)
        scales = np.broadcast_to(np.maximum(np.asarray(scale, dtype=float), 1.), positive.shape)
        tolerance = np.broadcast_to(np.asarray(original_tol, dtype=float), positive.shape)
        normalized = positive / scales
        if active is not None:
            positive = np.where(active, positive, 0.)
            normalized = np.where(active, normalized, 0.)
        worst = float(np.max(positive, initial=0.))
        worst_normalized = float(np.max(normalized, initial=0.))
        failed = (positive > tolerance) | (normalized > scaled_tol)
        row_count = int(np.count_nonzero(active)) if active is not None else positive.size
        residuals[name] = {
            "passed": not bool(np.any(failed)), "unit": unit, "paper_equation": label,
            "constraint_count": int(row_count), "violation_count": int(np.count_nonzero(failed)),
            "max_violation": worst, "max_normalized_violation": worst_normalized,
            "normalized_tolerance": scaled_tol,
            "original_tolerance_min": float(np.min(tolerance)) if tolerance.size else None,
            "original_tolerance_max": float(np.max(tolerance)) if tolerance.size else None,
            "worst_location": location(kind, np.unravel_index(np.argmax(positive), positive.shape)) if worst else None,
            "worst_normalized_location": location(kind, np.unravel_index(np.argmax(normalized), normalized.shape)) if worst_normalized else None,
        }

    record("binary", np.minimum(np.abs(x), np.abs(x - 1)), 1., scaled_tol, "selection",
           "eq:one_task_per_ev", "task")
    record("r_nonnegative", -reserve, p_dis[:, None], power_tol, "kW",
           "eq:reserve_procurement_upper_bound", "task_period")
    record("one_task_per_vehicle", vehicle_selection - 1., 1., scaled_tol, "selection",
           "eq:one_task_per_ev", "vehicle")
    record("reserve_eligibility", reserve - p_dis[:, None] * eligible * x[:, None],
           p_dis[:, None], power_tol, "kW", "eq:reserve_procurement_upper_bound", "task_period")
    record("minimum_block", p_block[:, None] * x[:, None] - reserve, p_block[:, None], power_tol,
           "kW", "eq:minimum_block_reserve", "task_period", active=block)
    energy_tolerance = t * dt / eta * power_tol
    record("forecast_energy", reserved_dc - energy * x, energy, energy_tolerance, "kWh_DC",
           "eq:forecast_reserve_energy_envelope", "task")
    record("plug_capacity", occupancy - plugs[:, None], plugs[:, None], scaled_tol, "plugs",
           "eq:plug_reservation", "station_period")
    record("interface_capacity", station_reserve - interface[:, None], interface[:, None], power_tol, "kW",
           "eq:station_interface_limit", "station_period")
    activation_cost = float(np.sum(c_act * x))
    reservation_cost = float(np.sum(c_cap * reserve) * dt)
    _require(np.isfinite(activation_cost) and np.isfinite(reservation_cost), "Nonfinite procurement cost")
    result = {
        "passed": all(row["passed"] for row in residuals.values()),
        "task_sha256": tasks.task_sha256, "config_hash": tasks.config_hash,
        "dimensions": {"tasks": h, "periods": t, "stations": c, "vehicles_with_tasks": len(vehicle_order)},
        "residuals": residuals,
        "max_normalized_violation": max(row["max_normalized_violation"] for row in residuals.values()),
        "costs": {"activation_usd": activation_cost, "reservation_usd": reservation_cost,
                  "Fpre_usd": activation_cost + reservation_cost},
        "aggregates": {"vehicle_selection": dict(zip(vehicle_order, vehicle_selection.tolist())),
                       "station_ids": list(station_ids), "periods": [int(v) for v in periods],
                       "station_plug_occupancy": occupancy.tolist(),
                       "station_reserved_kw": station_reserve.tolist(),
                       "task_reserved_energy_dc_kwh": reserved_dc.tolist()},
        "tolerances": {**policy, "source": policy_source,
                       "energy_original_tolerance_rule": "T * dt_h / eta_dis * power_tolerance",
                       "normalization": "max(1, fixed physical row capacity); binary and one-task rows use 1",
                       "acceptance": "Every original-unit and normalized inequality tolerance must pass"},
    }
    # Enforce the promised report format even in unusual numeric edge cases.
    json.dumps(result, allow_nan=False)
    return result
