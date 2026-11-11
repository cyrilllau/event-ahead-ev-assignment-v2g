"""Immutable, hash-bound v54-Q1 inputs for assignment and physical recourse.

Power is kW/kvar, battery energy is DC kWh, voltage is squared per unit.
This adapter never draws a scenario, changes a route, or applies an efficiency
or load scaling factor to a saved physical coefficient.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .config import get_paths
from .random_streams import canonical_hash

FROZEN_MANIFEST_SHA256 = "245a940345d6a01dc7eddaba3d0048ed88350eb637dd75a78b851354477cc461"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _bound_file(path, expected):
    _check(isinstance(expected, str) and len(expected) == 64, "An expected file SHA256 is required")
    actual = _sha(path)
    _check(actual == expected, f"Accepted file SHA256 mismatch: {path}")
    return actual


def _freeze(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        _check(not array.dtype.hasobject, "Object arrays are not physical inputs")
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


class _Immutable:
    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))


@dataclass(frozen=True)
class PhysicalCase(_Immutable):
    periods: tuple[int, ...]
    bus_ids: tuple[int, ...]
    line_ids: tuple[int, ...]
    fixed_open_ties: tuple[int, ...]
    station_ids: tuple[str, ...]
    station_bus_ids: tuple[int, ...]
    station_plugs: np.ndarray
    station_interface_kw: np.ndarray
    station_rating_kva: np.ndarray
    line_from_bus_ids: tuple[int, ...]
    line_to_bus_ids: tuple[int, ...]
    line_m_p_per_kw: np.ndarray
    line_m_q_per_kvar: np.ndarray
    line_p_limit_kw: np.ndarray
    line_q_limit_kvar: np.ndarray
    tan_phi_load: np.ndarray
    critical_buses: np.ndarray
    c_shed_usd_per_kwh: np.ndarray
    baseline_p_kw: np.ndarray
    baseline_q_kvar: np.ndarray
    archived_baseline_q_kvar: np.ndarray
    dt_h: float
    u_min: float
    u_max: float
    u_ref: float
    m_voltage: float
    s_base_kva: float
    polygon_sides: int
    c_short_usd_per_kwh: float
    vehicle_parameters: Mapping
    station_parameters: Mapping
    parameters: Mapping
    input_hashes: Mapping
    manifest_sha256: str
    case_sha256: str
    model_revision: str
    capability_active_coordinate: str
    max_archived_q_adjustment_kvar: float


@dataclass(frozen=True)
class FrozenTaskInputs(_Immutable):
    task_ids: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    station_indices: np.ndarray
    bus_indices: np.ndarray
    start_periods: np.ndarray
    chi_occ: np.ndarray
    chi_res: np.ndarray
    block_mask: np.ndarray
    p_dis_kw: np.ndarray
    p_block_kw: np.ndarray
    eta_dis: np.ndarray
    e_forecast_dc_kwh: np.ndarray
    c_act_usd: np.ndarray
    c_cap_usd_per_kwh: np.ndarray
    e_cap_kwh: np.ndarray
    soc_departure_min: np.ndarray
    s_inv_kva: np.ndarray
    c_v2g_usd_per_kwh: np.ndarray
    c_deg_usd_per_kwh: np.ndarray
    qualification_soc: np.ndarray
    forecast_arrival_soc: np.ndarray
    forecast_arrival_h: np.ndarray
    tau_blk_h: float
    vehicle_task_indices: Mapping
    station_task_indices: Mapping
    bus_task_indices: Mapping
    task_sha256: str
    config_hash: str
    config: Mapping
    physical_case_sha256: str
    source_sha256: str
    dependency_sha256: str
    source_path: str


def load_physical_case(case_root=None):
    """Load the frozen 74-file case, checking its accepted manifest and units."""
    root = Path(case_root) if case_root is not None else get_paths().data_root
    manifest_hash = _bound_file(root / "MANIFEST_SHA256.csv", FROZEN_MANIFEST_SHA256)
    def rows(relative):
        with (root / relative).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    manifest = rows("MANIFEST_SHA256.csv")
    _check(len(manifest) == 74, "Expected complete frozen 74-file case manifest")
    hashes = {}
    for row in manifest:
        name = row["relative_path"]
        _check(name not in hashes and not Path(name).is_absolute() and ".." not in Path(name).parts,
               "Invalid or duplicate manifest identity")
        hashes[name] = _bound_file(root / name, row["sha256"])
        _check((root / name).stat().st_size == int(row["size_bytes"]), "Frozen input byte count mismatch")
    symbols = rows("01_config/model_parameters__paper_symbols.csv")
    params = {}
    for row in symbols:
        key = row["code_key"]
        _check(key not in params, "Duplicate model parameter")
        try:
            params[key] = float(row["value"])
        except ValueError:
            params[key] = row["value"]
    settings = json.loads((root / "01_config/reactive_support__v54_q1.json").read_text())
    _check(settings["revision"] == "v54-q1" and settings["mode"] == "controllable_polygon"
           and settings["capability_active_coordinate"] == "accredited_reserve_r_av"
           and settings["polygon_sides"] == 16, "Unsupported Q1 model identity or capability coordinate")
    _check(settings["load_Q_mapping"] == "P_base * Q_rated / P_rated; zero at zero-load bus",
           "Unsupported Q1 load mapping")
    buses = rows("02_power_network/N_P_rated_Q_rated__bus_parameters.csv")
    branches = rows("02_power_network/L_z__branch_parameters.csv")
    limits = {int(r["ell"]): r for r in rows("02_power_network/Pbar_line_Qbar_line__line_limits.csv")}
    stations = rows("05_evcs/C__evcs_bus_road_mapping.csv")
    fleet = rows("06_fleet/E__vehicle_parameters.csv")
    bus_ids = tuple(int(r["n"]) for r in buses)
    _check(bus_ids == tuple(range(1, 34)), "Canonical bus order must be 1..33")
    _check(tuple(int(r["ell"]) for r in branches) == tuple(range(1, 38)), "Canonical branch order must be 1..37")
    closed = [r for r in branches if int(r["chi_closed_ell"]) == 1]
    ties = tuple(int(r["ell"]) for r in branches if int(r["chi_closed_ell"]) == 0)
    line_ids = tuple(int(r["ell"]) for r in closed)
    _check(line_ids == tuple(range(1, 33)) and ties == tuple(range(33, 38)), "Baseline closed/tie topology changed")
    _check(tuple(int(r["ell"]) for r in rows("02_power_network/Lclosed__radial_line_set.csv")) == line_ids
           and tuple(int(r["ell"]) for r in rows("02_power_network/Ltie__fixed_open_tie_set.csv")) == ties,
           "Closed/tie sets disagree with branch table")
    station_ids = tuple(r["c"] for r in stations)
    _check(station_ids == tuple(f"CS{i}" for i in range(1, 9)), "Canonical station order must be CS1..CS8")
    def values(source, key):
        a = np.array([float(row[key]) for row in source])
        _check(np.all(np.isfinite(a)) and np.all(a >= 0), f"Invalid nonnegative physical coefficient: {key}")
        return a
    sbase = float(params["S_base_MVA"]) * 1000.
    mp, mq = values(closed, "mP_ell_per_kW"), values(closed, "mQ_ell_per_kvar")
    # The CSV coefficients and pu impedances have independent decimal rounding.
    _check(np.allclose(mp, 2 * values(closed, "r_ell_pu") / sbase, rtol=1e-9, atol=1e-15)
           and np.allclose(mq, 2 * values(closed, "x_ell_pu") / sbase, rtol=1e-9, atol=1e-15),
           "Voltage-drop coefficient units disagree with per-unit base")
    for r in closed:
        lim = limits[int(r["ell"])]
        _check((r["i_bus"], r["j_bus"]) == (lim["i_bus"], lim["j_bus"])
               and int(lim["active_in_baseline"]) == 1, "Line limit identity mismatch")
    rated_p, rated_q = values(buses, "P_rated_n_kW"), values(buses, "Q_rated_n_kvar")
    _check(np.all(rated_q[rated_p == 0] == 0), "Zero active-load bus has nonzero rated Q")
    tanphi = np.divide(rated_q, rated_p, out=np.zeros(33), where=rated_p > 0)
    base_p, archive_q = np.zeros((24, 33)), np.zeros((24, 33))
    keys = set()
    for row in rows("03_load/P_base_Q_base__N_T.csv"):
        t, n = int(row["t"]), int(row["n"])
        _check((t, n) not in keys and 1 <= t <= 24 and n in bus_ids, "Invalid baseline load key")
        keys.add((t, n)); base_p[t-1, n-1] = float(row["P_base_n_t_kW"])
        archive_q[t-1, n-1] = float(row["Q_base_n_t_kvar"])
    _check(len(keys) == 792 and np.all(np.isfinite(base_p)) and np.all(base_p >= 0)
           and np.all(np.isfinite(archive_q)) and np.all(archive_q >= 0), "Incomplete or invalid baseline load")
    base_q = base_p * tanphi[None, :]
    adjustment = float(np.max(np.abs(base_q - archive_q)))
    _check(adjustment <= .00080000001, "Archived Q1 correction exceeds the accepted baseline bound")
    hardware_keys = ("E_cap_e_kWh", "P_dis_bar_e_kW", "eta_dis_e", "E_km_e_kWh_per_km",
                     "P_aux_e_kW", "SOC_dep_under_e", "P_blk_under_e_kW", "S_inv_bar_e_kVA")
    hardware = {}
    for row in fleet:
        _check(row["e"] not in hardware, "Duplicate fleet identity")
        p = {key: float(row[key]) for key in hardware_keys}
        _check(all(np.isfinite(v) and v >= 0 for v in p.values()) and p["E_cap_e_kWh"] > 0
               and 0 < p["eta_dis_e"] <= 1 and p["SOC_dep_under_e"] <= 1
               and 0 < p["P_blk_under_e_kW"] <= p["P_dis_bar_e_kW"]
               and row["reactive_support_mode"] == "controllable_polygon"
               and np.isclose(p["S_inv_bar_e_kVA"], settings["apparent_power_multiplier"]*p["P_dis_bar_e_kW"], rtol=0, atol=1e-9),
               "Invalid vehicle hardware/rating")
        hardware[row["e"]] = p
    station_map = {r["c"]: {"bus_id": int(r["n_c"]), "road_node": int(r["road_node_c"])} for r in stations}
    plugs = values(stations, "N_plug_c")
    interface, rating = values(stations, "P_int_bar_c_kW"), values(stations, "S_int_bar_c_kVA")
    _check(np.all(plugs == plugs.astype(int)) and np.all(plugs > 0)
           and np.all(interface > 0) and np.allclose(rating, interface*settings["apparent_power_multiplier"], rtol=0, atol=1e-9)
           and all(r["reactive_support_mode"] == "controllable_polygon" for r in stations), "Invalid station hardware/rating")
    _check(params["Delta_t_h"] == 1 and params["tau_end_h"] == 24 and sbase == 10000,
           "Unsupported operating horizon or unit base")
    return PhysicalCase(periods=tuple(range(1,25)), bus_ids=bus_ids, line_ids=line_ids, fixed_open_ties=ties,
        station_ids=station_ids, station_bus_ids=tuple(int(r["n_c"]) for r in stations), station_plugs=plugs.astype(int),
        station_interface_kw=interface, station_rating_kva=rating,
        line_from_bus_ids=tuple(int(r["i_bus"]) for r in closed), line_to_bus_ids=tuple(int(r["j_bus"]) for r in closed),
        line_m_p_per_kw=mp, line_m_q_per_kvar=mq, line_p_limit_kw=values([limits[i] for i in line_ids], "P_line_bar_ell_kW"),
        line_q_limit_kvar=values([limits[i] for i in line_ids], "Q_line_bar_ell_kvar"), tan_phi_load=tanphi,
        critical_buses=values(buses, "is_critical_n").astype(bool), c_shed_usd_per_kwh=values(buses, "C_shed_n_USD_per_kWh"),
        baseline_p_kw=base_p, baseline_q_kvar=base_q, archived_baseline_q_kvar=archive_q, dt_h=1.,
        u_min=params["V_under_pu"]**2, u_max=params["V_bar_pu"]**2, u_ref=params["V_ref_pu"]**2,
        m_voltage=params["M_V_ell"], s_base_kva=sbase, polygon_sides=16,
        c_short_usd_per_kwh=params["C_short_USD_per_kWh"], vehicle_parameters=hardware, station_parameters=station_map,
        parameters=params, input_hashes=hashes, manifest_sha256=manifest_hash,
        case_sha256=canonical_hash({"manifest_sha256": manifest_hash, "revision": settings["revision"], "input_hashes": hashes}),
        model_revision=settings["revision"], capability_active_coordinate=settings["capability_active_coordinate"],
        max_archived_q_adjustment_kvar=adjustment)


def load_frozen_tasks(path, case, expected_sha256, expected_config_hash=None):
    """Read accepted H; expected_sha256 is the JSON FILE hash, not H's hash.

    config_hash is exactly canonical_hash(bank['config']). The physical case
    identity and frozen semantic task hash remain separate identities.
    """
    path = Path(path)
    source_hash = _bound_file(path, expected_sha256)
    bank = json.loads(path.read_text())
    _check(bank.get("status") == "COMPLETE" and bank.get("certified") is True
           and bank.get("tasks_provisional") is False, "Tasks must be a certified complete frozen H")
    tasks, config = bank["tasks"], bank["config"]
    _check(bank["task_count"] == len(tasks), "Frozen task count mismatch")
    semantic = canonical_hash({"dependency_sha256": bank["dependency_sha256"], "config": config,
                               "certified": True, "tasks": tasks})
    _check(semantic == bank["task_sha256"], "Frozen task semantic hash mismatch")
    config_hash = canonical_hash(config)
    if expected_config_hash is not None:
        _check(config_hash == expected_config_hash, "Frozen task config hash mismatch")
    tau, basic = float(config["tau_blk_h"]), float(config["basic_soc"])
    _check(tau in (1., 2.) and np.isfinite(basic) and 0 <= basic <= 1, "Unsupported task configuration")
    queries = {}
    for query in bank["queries"]:
        key = (query["vehicle_id"], query["station_id"])
        _check(key not in queries and query["status"] == "COMPLETE", "Incomplete or duplicate frozen query")
        _check(key[0] in case.vehicle_parameters and key[1] in case.station_ids, "Unknown query vehicle/station")
        p, args = case.vehicle_parameters[key[0]], query["arguments"]
        q = max(basic, p["SOC_dep_under_e"] + p["P_blk_under_e_kW"]*tau/(p["eta_dis_e"]*p["E_cap_e_kWh"]))
        expected = {"capacity_kwh":p["E_cap_e_kWh"], "energy_kwh_per_km":p["E_km_e_kWh_per_km"],
                    "auxiliary_kw":p["P_aux_e_kW"], "qualification_soc":q,
                    "destination":case.station_parameters[key[1]]["road_node"]}
        _check(all(args.get(k) == v for k, v in expected.items()), "Frozen query hardware/station differs from case")
        queries[key] = query
    ids, vehicles, stations, rows = [], [], [], []
    seen = set()
    starts = np.arange(1,25)
    for task in tasks:
        e, c, k = task["vehicle_id"], task["station_id"], task["start_period"]
        _check(type(k) is int and 1 <= k <= 24 and task["task_id"] == f"{e}__{c}__k{k:02d}"
               and task["task_id"] not in seen and (e,c) in queries, "Invalid or duplicate task identity")
        seen.add(task["task_id"])
        p, query = case.vehicle_parameters[e], queries[e,c]
        occ = starts >= k
        reserve = occ & (starts-1+tau <= 24)
        block = np.isin(starts, np.arange(k,k+int(tau)))
        _check(all(type(x) is bool for x in task["occupancy"]+task["reserve"])
               and np.array_equal(task["occupancy"],occ) and np.array_equal(task["reserve"],reserve)
               and task["block_periods"] == list(range(k,k+int(tau)))
               and int(block.sum()) == int(tau) and np.all(reserve[block]), "Frozen occupancy/reserve/block mismatch")
        route = query["route"]
        _check(route is not None and task["arc_ids"] == route["arc_ids"], "Task route differs from certified query")
        for key in ("arrival_h", "distance_km", "freeflow_h", "travel_energy_kwh", "arrival_soc"):
            _check(np.isfinite(task[key]) and task[key] == route[key], "Task physical field differs from frozen route")
        qual = query["arguments"]["qualification_soc"]
        usable = max(task["arrival_soc"]-p["SOC_dep_under_e"],0.)*p["E_cap_e_kWh"]
        _check(0 <= task["arrival_soc"] <= 1 and task["arrival_soc"] >= qual
               and task["qualification_soc"] == qual and task["arrival_h"] <= k-1
               and task["usable_energy_kwh"] == usable and usable > 0
               and task["travel_time_h"] == task["arrival_h"]-query["arguments"]["departure_h"]
               and min(task["distance_km"],task["freeflow_h"],task["travel_time_h"],task["travel_energy_kwh"]) >= 0,
               "Invalid forecast physical task or energy")
        ids.append(task["task_id"]); vehicles.append(e); stations.append(c); rows.append((task,p,occ,reserve,block))
    h = len(rows)
    def col(key): return np.array([r[1][key] for r in rows],dtype=float)
    def taskcol(key): return np.array([r[0][key] for r in rows],dtype=float)
    si = np.array([case.station_ids.index(c) for c in stations],dtype=int)
    bi = np.array([case.bus_ids.index(case.station_bus_ids[i]) for i in si],dtype=int)
    def groups(values):
        return {key:tuple(i for i,v in enumerate(values) if v == key) for key in dict.fromkeys(values)}
    cact = case.parameters["C_act_e_USD"]+case.parameters["c_mob_e_USD_per_km"]*taskcol("distance_km")
    return FrozenTaskInputs(task_ids=tuple(ids), vehicle_ids=tuple(vehicles), station_ids=tuple(stations),
        station_indices=si, bus_indices=bi, start_periods=taskcol("start_period").astype(int),
        chi_occ=np.array([r[2] for r in rows],dtype=bool).reshape(h,24),
        chi_res=np.array([r[3] for r in rows],dtype=bool).reshape(h,24),
        block_mask=np.array([r[4] for r in rows],dtype=bool).reshape(h,24),
        p_dis_kw=col("P_dis_bar_e_kW"), p_block_kw=col("P_blk_under_e_kW"), eta_dis=col("eta_dis_e"),
        e_forecast_dc_kwh=taskcol("usable_energy_kwh"), c_act_usd=cact,
        c_cap_usd_per_kwh=np.full((h,24),case.parameters["C_cap_USD_per_kW_h"]),
        e_cap_kwh=col("E_cap_e_kWh"), soc_departure_min=col("SOC_dep_under_e"), s_inv_kva=col("S_inv_bar_e_kVA"),
        c_v2g_usd_per_kwh=np.full(h,case.parameters["C_V2G_USD_per_kWh"]),
        c_deg_usd_per_kwh=np.full(h,case.parameters["C_deg_USD_per_kWh"]),
        qualification_soc=taskcol("qualification_soc"), forecast_arrival_soc=taskcol("arrival_soc"),
        forecast_arrival_h=taskcol("arrival_h"), tau_blk_h=tau, vehicle_task_indices=groups(vehicles),
        station_task_indices=groups(stations), bus_task_indices=groups([case.bus_ids[i] for i in bi]),
        task_sha256=semantic, config_hash=config_hash, config=config, physical_case_sha256=case.case_sha256,
        source_sha256=source_hash, dependency_sha256=bank["dependency_sha256"], source_path=str(path.resolve()))


@dataclass(frozen=True)
class ScenarioPhysicalInputs(_Immutable):
    target_event: str
    split: str
    base_weather_event: str
    variant: int
    regime: str
    development: bool
    scenario_id: str
    scenario_sha256: str
    task_sha256: str
    config_hash: str
    physical_case_sha256: str
    task_ids: tuple[str, ...]
    bus_ids: tuple[int, ...]
    line_ids: tuple[int, ...]
    periods: tuple[int, ...]
    times_h: np.ndarray
    line_operating_outages: np.ndarray
    load_p_kw: np.ndarray
    load_q_kvar: np.ndarray
    load_archived_q_kvar: np.ndarray
    task_mu_full: np.ndarray
    task_varsigma: np.ndarray
    task_arrival_soc: np.ndarray
    task_arrival_h: np.ndarray
    task_gamma_route: np.ndarray
    task_usable_energy_dc_kwh: np.ndarray
    source_hashes: Mapping
    metadata: Mapping


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def load_scenario_physical_inputs(npz_path, ledger_path, scenario_index, case, tasks, *,
                                 expected_npz_sha256, expected_ledger_sha256,
                                 expected_scenario_sha256=None):
    """Read one accepted I08 scenario without regenerating its physical arrays.

    Both expected file hashes must come from accepted evidence. config_hash is
    the H config hash; I08 has no separately saved scenario-config hash. The
    complete saved scenario provenance remains in metadata. This I01 adapter
    accepts development planning/calibration only, leaving formal OOS gated.
    """
    _check(tasks.physical_case_sha256 == case.case_sha256, "Task/case physical identity differs")
    source_hashes = {str(Path(npz_path).resolve()):_bound_file(npz_path, expected_npz_sha256),
                     str(Path(ledger_path).resolve()):_bound_file(ledger_path, expected_ledger_sha256)}
    ledger = json.loads(Path(ledger_path).read_text())
    scenarios = ledger["scenarios"]
    _check(type(scenario_index) is int and 0 <= scenario_index < len(scenarios), "Invalid scenario index")
    row = scenarios[scenario_index]
    payload = {k:v for k,v in row.items() if k != "scenario_sha256"}
    _check(canonical_hash(payload) == row["scenario_sha256"], "Scenario semantic hash mismatch")
    if expected_scenario_sha256 is not None:
        _check(row["scenario_sha256"] == expected_scenario_sha256, "Wrong requested scenario identity")
    _check(row["task_sha256"] == tasks.task_sha256, "Scenario uses another frozen H")
    _check(row["development"] is True and row["split"] in ("planning", "calibration")
           and row["regime"] == "R1" and row["lead_h"] == 1., "I01 accepts the development I08 R1 bank only")
    _check(type(row["variant"]) is int and row["variant"] == row["variant_index"]+1
           and row["variant"] >= 1, "Invalid scenario variant identity")
    expected_id = f'{row["target_event"]}/development/{row["split"]}/{row["base_weather_event"]}/v{row["variant"]:03d}/R1'
    _check(row["scenario_id"] == expected_id, "Scenario ID does not match its coordinates")
    names = ("line_operating_outages", "load_p_kw", "load_q_kvar", "load_archived_q_kvar", "load_epsilon",
             "task_mu_full", "task_varsigma", "task_arrival_soc", "task_arrival_h", "task_gamma_route",
             "task_usable_energy_dc_kwh", "task_route_time_h", "task_travel_energy_kwh")
    arrays = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        for key in ("task_ids", "scenario_ids", "times_h"):
            value = archive[key]
            _check(_array_hash(value) == ledger["array_payload_hashes"][key], "NPZ identity axis hash mismatch")
        _check(tuple(archive["task_ids"].tolist()) == tasks.task_ids, "Scenario task order differs from frozen H")
        _check(archive["scenario_ids"].tolist() == [x["scenario_id"] for x in scenarios], "Scenario ledger order mismatch")
        times = archive["times_h"]
        _check(np.array_equal(times, np.arange(-4,96)/4.), "I08 replay time axis changed")
        for name in names:
            bank_array = archive[name]
            _check(bank_array.shape[0] == len(scenarios), "Scenario count differs from ledger")
            value = bank_array[scenario_index]
            _check(_array_hash(value) == row["array_hashes"][name], f"Saved scenario field hash mismatch: {name}")
            arrays[name] = value.copy()
    h, t, n, ell = len(tasks.task_ids), len(case.periods), len(case.bus_ids), len(case.line_ids)
    shapes = {"line_operating_outages":(t,ell), "load_p_kw":(t,n), "load_q_kvar":(t,n),
              "load_archived_q_kvar":(t,n), "load_epsilon":(t,), "task_mu_full":(h,t)}
    for name, value in arrays.items():
        _check(value.shape == shapes.get(name,(h,)), f"Physical array shape mismatch: {name}")
        if name in ("line_operating_outages", "task_mu_full", "task_varsigma", "task_gamma_route"):
            _check(value.dtype == np.dtype(bool), f"Expected Boolean physical mask: {name}")
        elif name != "task_arrival_h":
            _check(np.all(np.isfinite(value)), f"Nonfinite physical coefficient: {name}")
    epsilon = arrays["load_epsilon"]
    _check(np.all(np.abs(epsilon) <= .09), "Load error outside frozen support")
    factor = 1+epsilon[:,None]
    for key, expected in (("load_p_kw",case.baseline_p_kw*factor),
                          ("load_q_kvar",case.baseline_q_kvar*factor),
                          ("load_archived_q_kvar",case.archived_baseline_q_kvar*factor)):
        _check(np.all(arrays[key] >= 0) and np.allclose(arrays[key],expected,rtol=0,atol=2e-12),
               "Saved load differs from frozen baseline/error mapping")
    _check(np.allclose(arrays["load_q_kvar"], arrays["load_p_kw"]*case.tan_phi_load[None,:], rtol=0, atol=2e-12),
           "Scenario Q is not the exact rated-PF Q1 mapping")
    arrival, gamma, soc = arrays["task_arrival_h"], arrays["task_gamma_route"], arrays["task_arrival_soc"]
    _check(not np.any(np.isnan(arrival)) and not np.any(arrival == -np.inf)
           and np.array_equal(np.isfinite(arrival),gamma) and np.all((soc >= 0)&(soc <= 1)),
           "Invalid arrival/Gamma/SOC accounting")
    for name in ("task_arrival_soc", "task_route_time_h", "task_travel_energy_kwh"):
        _check(np.all(arrays[name] >= 0) and np.all(arrays[name][~gamma] == 0), "Failed-task zero accounting changed")
    available = np.maximum(arrival,tasks.start_periods-1)
    success = gamma & (soc >= tasks.qualification_soc) & (24-available >= tasks.tau_blk_h)
    mu = success[:,None] & (available[:,None] <= np.arange(24)[None,:]) & (np.arange(24)[None,:]+tasks.tau_blk_h <= 24)
    energy = success*np.maximum(soc-tasks.soc_departure_min,0)*tasks.e_cap_kwh
    _check(np.array_equal(arrays["task_varsigma"],success) and np.array_equal(arrays["task_mu_full"],mu)
           and np.allclose(arrays["task_usable_energy_dc_kwh"],energy,rtol=0,atol=1e-12),
           "Saved task deliverability or DC energy differs from frozen physical definition")
    exposed = {name:arrays[name] for name in names if name not in ("load_epsilon","task_route_time_h","task_travel_energy_kwh")}
    return ScenarioPhysicalInputs(target_event=row["target_event"], split=row["split"],
        base_weather_event=row["base_weather_event"], variant=row["variant"], regime=row["regime"], development=True,
        scenario_id=row["scenario_id"], scenario_sha256=row["scenario_sha256"], task_sha256=tasks.task_sha256,
        config_hash=tasks.config_hash, physical_case_sha256=case.case_sha256, task_ids=tasks.task_ids,
        bus_ids=case.bus_ids, line_ids=case.line_ids, periods=case.periods, times_h=times,
        source_hashes=source_hashes, metadata={"scenario_provenance":row,"config_hash_semantics":"canonical_hash(frozen H config)",
            "scenario_config_hash_saved_by_I08":False,"formal_OOS_locked":True}, **exposed)
