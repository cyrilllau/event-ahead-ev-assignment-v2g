"""Independent direct-physics audit of a fixed-plan v54-Q1 recourse solution.

No LP matrix, capability helper, solver or fallback builder is used here.
All arrays retain their frozen task/scenario axes. A feasible candidate need
not be optimal: the analytic *optimal-value* upper bound is opt-in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import get_paths
from .first_stage_check import check_first_stage


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _array(value, shape, name):
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    _require(a.shape == shape, f"{name} has shape {a.shape}; expected {shape}")
    _require(np.isfinite(a).all(), f"{name} contains nonfinite values")
    return a


def _mask(value, shape, name):
    a = _array(value, shape, name)
    _require(np.all((a == 0) | (a == 1)), f"{name} must be binary")
    return a.astype(bool)


def _policy(value):
    if value is None:
        path = get_paths().spec_root / "runtime_policy.json"
        data, source = json.loads(path.read_text()), str(path)
    elif isinstance(value, Mapping):
        data, source = value, "explicit_mapping"
    else:
        path = Path(value)
        data, source = json.loads(path.read_text()), str(path)
    names = ("LP_scaled_feasibility_tolerance", "physical_power_residual_tolerance_kW_kvar",
             "SOC_residual_tolerance", "voltage_squared_residual_tolerance_pu2",
             "objective_comparison_USD_absolute", "objective_comparison_relative")
    result = {name: float(data[name]) for name in names}
    _require(all(np.isfinite(v) and v > 0 for v in result.values()), "Invalid runtime tolerance")
    return result, source


def check_recourse(case, tasks, scene, plan, solution, *, runtime_policy=None,
                   check_objective_bound=False):
    """Audit physical feasibility and reconstruct all four operating costs.

    Identity/shape/nonfinite or inconsistent physical inputs raise ValueError.
    Finite infeasible decisions return passed=False, without clipping or
    changing x/r. Every physical row must satisfy both its original-unit
    tolerance and normalized LP tolerance. Energy tolerance propagates the
    power tolerance through the actual AC-to-DC coefficients; the normalized
    energy check is retained. check_objective_bound is appropriate when a
    caller claims optimality, but passing this check does not certify it.
    """
    _require(isinstance(check_objective_bound, (bool, np.bool_)), "check_objective_bound must be boolean")
    policy, policy_source = _policy(runtime_policy)
    first_stage = check_first_stage(case, tasks, plan, runtime_policy=policy)
    st = policy["LP_scaled_feasibility_tolerance"]
    pt = policy["physical_power_residual_tolerance_kW_kvar"]
    soct = policy["SOC_residual_tolerance"]
    vt = policy["voltage_squared_residual_tolerance_pu2"]
    ids = tuple(tasks.task_ids); periods = tuple(case.periods)
    buses = tuple(case.bus_ids); lines = tuple(case.line_ids); stations = tuple(case.station_ids)
    h, t, n, ell, c = len(ids), len(periods), len(buses), len(lines), len(stations)
    _require(n > 0 and len(set(buses)) == n and len(set(lines)) == ell, "Invalid bus/line identities")
    _require(all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)) for i in buses),
             "Bus identities must be integers")
    _require(not set(lines).intersection(case.fixed_open_ties), "Fixed-open ties cannot be operating lines")
    _require(case.model_revision == "v54-q1" and
             case.capability_active_coordinate == "accredited_reserve_r_av",
             "Unsupported physical model or capability coordinate")
    for name in ("task_sha256", "config_hash"):
        expected = getattr(tasks, name)
        _require(getattr(scene, name) == expected and getattr(solution, name) == expected,
                 f"Scenario/solution {name} mismatch")
    _require(isinstance(scene.scenario_sha256, str) and bool(scene.scenario_sha256) and
             solution.scenario_sha256 == scene.scenario_sha256, "Solution scenario_sha256 mismatch")
    for name, expected in (("task_ids", ids), ("bus_ids", buses), ("line_ids", lines), ("periods", periods)):
        _require(tuple(getattr(scene, name)) == expected, f"Scenario {name} axis mismatch")
    _require(tasks.physical_case_sha256 == case.case_sha256 == scene.physical_case_sha256,
             "Physical case identity mismatch")
    lookup = {bus: j for j, bus in enumerate(buses)}
    station_bus_ids = tuple(case.station_bus_ids)
    _require(len(station_bus_ids) == c and set(station_bus_ids) <= set(buses), "Unknown station bus")
    source_ids, dest_ids = tuple(case.line_from_bus_ids), tuple(case.line_to_bus_ids)
    _require(len(source_ids) == ell and len(dest_ids) == ell and
             set(source_ids + dest_ids) <= set(buses) and
             all(a != b for a, b in zip(source_ids, dest_ids)), "Invalid branch endpoints")
    source = np.array([lookup[v] for v in source_ids], dtype=int)
    dest = np.array([lookup[v] for v in dest_ids], dtype=int)
    station_lookup = {s: i for i, s in enumerate(stations)}
    task_station = np.array([station_lookup[s] for s in tasks.station_ids], dtype=int)
    task_bus = np.array([lookup[station_bus_ids[j]] for j in task_station], dtype=int)
    _require(np.array_equal(_array(tasks.station_indices, (h,), "station_indices"), task_station) and
             np.array_equal(_array(tasks.bus_indices, (h,), "bus_indices"), task_bus),
             "Task bus/station index disagrees with identities")
    dt, tau = float(case.dt_h), float(tasks.tau_blk_h)
    eta = _array(tasks.eta_dis, (h,), "eta_dis")
    pdis = _array(tasks.p_dis_kw, (h,), "p_dis_kw")
    ecap = _array(tasks.e_cap_kwh, (h,), "e_cap_kwh")
    dep_min = _array(tasks.soc_departure_min, (h,), "soc_departure_min")
    sinv = _array(tasks.s_inv_kva, (h,), "s_inv_kva")
    station_s = _array(case.station_rating_kva, (c,), "station_rating_kva")
    pbar = _array(case.line_p_limit_kw, (ell,), "line_p_limit_kw")
    qbar = _array(case.line_q_limit_kvar, (ell,), "line_q_limit_kvar")
    mp = _array(case.line_m_p_per_kw, (ell,), "line_m_p_per_kw")
    mq = _array(case.line_m_q_per_kvar, (ell,), "line_m_q_per_kvar")
    phi = _array(case.tan_phi_load, (n,), "tan_phi_load")
    _require(np.all(ecap > 0) and np.all((dep_min >= 0) & (dep_min <= 1)) and
             all(np.all(a >= 0) for a in (sinv, station_s, pbar, qbar, mp, mq, phi)),
             "Invalid physical rating/coefficient")
    umin, umax, uref = float(case.u_min), float(case.u_max), float(case.u_ref)
    mv = np.broadcast_to(np.asarray(case.m_voltage, dtype=float), (ell,))
    _require(all(np.isfinite(v) for v in (umin, umax, uref)) and 0 < umin <= uref <= umax and
             np.isfinite(mv).all() and np.all(mv >= 0), "Invalid voltage settings")
    sides = case.polygon_sides
    _require(isinstance(sides, (int, np.integer)) and not isinstance(sides, (bool, np.bool_)) and sides == 16,
             "v54-Q1 requires 16 polygon facets")
    load_p = _array(scene.load_p_kw, (t,n), "load_p_kw")
    load_q = _array(scene.load_q_kvar, (t,n), "load_q_kvar")
    outage = _mask(scene.line_operating_outages, (t,ell), "line_operating_outages")
    mu = _mask(scene.task_mu_full, (h,t), "task_mu_full")
    qualified = _mask(scene.task_varsigma, (h,), "task_varsigma")
    socarr = _array(scene.task_arrival_soc, (h,), "task_arrival_soc")
    energy = _array(scene.task_usable_energy_dc_kwh, (h,), "task_usable_energy_dc_kwh")
    _require(np.all(load_p >= 0) and np.all(load_q >= 0) and
             np.allclose(load_q, load_p * phi[None,:], rtol=1e-12, atol=1e-10),
             "Scenario load does not preserve the exact rated Q/P mapping")
    _require(np.all((socarr >= 0) & (socarr <= 1)) and np.all(energy >= 0) and
             np.all(~mu | (qualified[:,None] & np.asarray(tasks.chi_res, dtype=bool))),
             "Invalid SOC/energy or full-period availability")
    expected_energy = qualified * np.maximum(socarr-dep_min, 0.) * ecap
    _require(np.allclose(energy, expected_energy, rtol=1e-12, atol=1e-10),
             "Usable DC energy disagrees with arrival SOC and qualification")
    x = _array(plan.x, (h,), "plan.x"); reserve = _array(plan.r, (h,t), "plan.r")
    shapes = {"rav":(h,t), "p":(h,t), "q":(h,t), "short":(c,t), "shed":(t,n),
              "line_p":(t,ell), "line_q":(t,ell), "u":(t,n), "soc_dep":(h,)}
    values = {name: _array(getattr(solution, name), shape, f"solution.{name}") for name, shape in shapes.items()}
    rav, p, q, short, shed, lp, lq, u, socdep = (values[name] for name in shapes)
    station_rav = np.zeros((c,t)); station_q = np.zeros((c,t)); station_r = np.zeros((c,t))
    node_p = np.zeros((t,n)); node_q = np.zeros((t,n))
    node_nameplate_p = np.zeros(n); node_nameplate_q = np.zeros(n)
    for j in range(h):
        station_rav[task_station[j]] += rav[j]
        station_q[task_station[j]] += q[j]
        station_r[task_station[j]] += reserve[j]
        node_p[:,task_bus[j]] += p[j]; node_q[:,task_bus[j]] += q[j]
        node_nameplate_p[task_bus[j]] += pdis[j]; node_nameplate_q[task_bus[j]] += sinv[j]
    net_p, net_q = np.zeros((t,n)), np.zeros((t,n))
    incident_p, incident_q = np.zeros(n), np.zeros(n)
    for j, (a,b) in enumerate(zip(source,dest)):
        net_p[:,a] -= lp[:,j]; net_p[:,b] += lp[:,j]
        net_q[:,a] -= lq[:,j]; net_q[:,b] += lq[:,j]
        incident_p[a] += pbar[j]; incident_p[b] += pbar[j]
        incident_q[a] += qbar[j]; incident_q[b] += qbar[j]

    residuals = {}
    axes = {"task":ids, "period":periods, "station":stations, "bus":buses,
            "line":lines, "facet":tuple(range(sides))}
    def record(name, excess, scale, tolerance, unit, equation, dimensions):
        a = np.asarray(excess, dtype=float)
        _require(np.isfinite(a).all(), f"Nonfinite arithmetic in {name}")
        violation = np.maximum(a, 0.)
        denominator = np.broadcast_to(np.maximum(np.asarray(scale, dtype=float), 1.), a.shape)
        original_tol = np.broadcast_to(np.asarray(tolerance, dtype=float), a.shape)
        scaled = violation / denominator
        failed = (violation > original_tol) | (scaled > st)
        def where(array):
            if not np.any(array > 0): return None
            index = np.unravel_index(int(np.argmax(array)), array.shape)
            return {axis: int(axes[axis][i]) if isinstance(axes[axis][i], (int,np.integer)) else axes[axis][i]
                    for axis,i in zip(dimensions,index)}
        residuals[name] = {"passed":not bool(np.any(failed)), "unit":unit, "paper_equation":equation,
            "constraint_count":int(a.size), "violation_count":int(np.count_nonzero(failed)),
            "max_violation":float(np.max(violation, initial=0.)),
            "max_normalized_violation":float(np.max(scaled, initial=0.)),
            "normalized_tolerance":st,
            "original_tolerance_min":float(np.min(original_tol)) if a.size else None,
            "original_tolerance_max":float(np.max(original_tol)) if a.size else None,
            "worst_location":where(violation), "worst_normalized_location":where(scaled)}

    hp = ("task", "period"); cp = ("station", "period"); pn = ("period", "bus"); pl = ("period", "line")
    chain = "eq:commitment_certification_dispatch_chain"
    record("p_nonnegative", -p, pdis[:,None], pt, "kW", chain, hp)
    record("rav_nonnegative", -rav, pdis[:,None], pt, "kW", chain, hp)
    record("p_dispatch", p-rav, pdis[:,None], pt, "kW", chain, hp)
    record("rav_contract", rav-reserve, pdis[:,None], pt, "kW", chain, hp)
    record("rav_available", rav-pdis[:,None]*mu*x[:,None], pdis[:,None], pt, "kW",
           "eq:certified_reserve_availability", hp)
    # Sum only previous actual discharge. Unactivated reserve consumes no DC energy.
    previous_p = np.cumsum(p, axis=1)-p
    used_dc = (dt*previous_p + tau*rav)/eta[:,None]
    energy_tol = (np.arange(t)[None,:]*dt+tau)/eta[:,None]*pt
    record("sustainability", used_dc-energy[:,None]*x[:,None], energy[:,None], energy_tol,
           "kWh_DC", "eq:reserve_sustainability", hp)
    expected_soc = socarr*x-np.sum(p, axis=1)*dt/(eta*ecap)
    record("soc_definition", np.abs(socdep-expected_soc), 1., soct, "SOC_fraction",
           "eq:departure_soc_balance", ("task",))
    record("soc_departure", dep_min*x*qualified-socdep, 1., soct, "SOC_fraction",
           "eq:departure_soc_requirement", ("task",))
    theta = (2*np.arange(sides)+1)*np.pi/sides
    cosine, sine, inset = np.cos(theta), np.sin(theta), np.cos(np.pi/sides)
    task_facets = rav[:,:,None]*cosine+q[:,:,None]*sine-sinv[:,None,None]*inset*mu[:,:,None]*x[:,None,None]
    record("task_capability", task_facets, sinv[:,None,None], pt, "kVA_facet",
           "eq:task_apparent_capability", hp+("facet",))
    station_facets = station_rav[:,:,None]*cosine+station_q[:,:,None]*sine-station_s[:,None,None]*inset
    record("station_capability", station_facets, station_s[:,None,None], pt, "kVA_facet",
           "eq:station_apparent_capability", cp+("facet",))
    interface = np.asarray(case.station_interface_kw, dtype=float)[:,None]
    record("short_nonnegative", -short, interface, pt, "kW", "eq:station_reserve_shortfall", cp)
    record("shortfall", station_r-station_rav-short, interface, pt, "kW", "eq:station_reserve_shortfall", cp)
    record("shed_lower", -shed, load_p, pt, "kW", "eq:load_shedding_bounds", pn)
    record("shed_upper", shed-load_p, load_p, pt, "kW", "eq:load_shedding_bounds", pn)
    pscale = load_p+incident_p[None,:]+node_nameplate_p[None,:]
    qscale = load_q+incident_q[None,:]+node_nameplate_q[None,:]+phi[None,:]*load_p
    record("p_balance", np.abs(net_p+node_p+shed-load_p), pscale, pt, "kW", "eq:active_power_balance", pn)
    record("q_balance", np.abs(net_q+node_q+phi[None,:]*shed-load_q), qscale, pt, "kvar", "eq:reactive_power_balance", pn)
    record("line_p_limit", np.abs(lp)-pbar[None,:]*(~outage), pbar[None,:], pt, "kW", "eq:scenario_line_limits", pl)
    record("line_q_limit", np.abs(lq)-qbar[None,:]*(~outage), qbar[None,:], pt, "kvar", "eq:scenario_line_limits", pl)
    record("voltage_bounds", np.maximum(umin-u,u-umax), 1., vt, "pu_squared", "eq:voltage_limits", pn)
    drop = u[:,dest]-u[:,source]+mp[None,:]*lp+mq[None,:]*lq
    record("voltage_drop", np.abs(drop)-mv[None,:]*outage, 1., vt, "pu_squared", "eq:scenario_voltage_drop", pl)
    # Independent graph traversal, not the assembler's component routine.
    reference_mask = np.zeros((t,n), dtype=bool)
    references = []
    evcs_buses = set(station_bus_ids)
    for ti in range(t):
        adjacency = {bus:[] for bus in buses}
        for j,(a,b) in enumerate(zip(source_ids,dest_ids)):
            if not outage[ti,j]: adjacency[a].append(b); adjacency[b].append(a)
        unseen = set(buses); period_refs = []
        while unseen:
            component = set(); stack = [min(unseen)]
            while stack:
                bus = stack.pop()
                if bus in component: continue
                component.add(bus); unseen.discard(bus)
                stack.extend(v for v in adjacency[bus] if v not in component)
            reference = min(component & evcs_buses) if component & evcs_buses else min(component)
            reference_mask[ti,lookup[reference]] = True
            period_refs.append(int(reference))
        references.append(sorted(period_refs))
    record("voltage_reference", np.where(reference_mask,np.abs(u-uref),0.), 1., vt, "pu_squared",
           "eq:reference_voltage", pn)
    residuals["voltage_reference"]["constraint_count"] = int(np.count_nonzero(reference_mask))

    cshed = _array(case.c_shed_usd_per_kwh, (n,), "c_shed_usd_per_kwh")
    cshort = np.broadcast_to(np.asarray(case.c_short_usd_per_kwh, dtype=float), (c,t))
    cv2g = _array(tasks.c_v2g_usd_per_kwh, (h,), "c_v2g_usd_per_kwh")
    cdeg = _array(tasks.c_deg_usd_per_kwh, (h,), "c_deg_usd_per_kwh")
    _require(all(np.isfinite(a).all() and np.all(a >= 0) for a in (cshed,cshort,cv2g,cdeg)), "Invalid recourse price")
    costs = {"Fshed":float(np.sum(shed*cshed[None,:])*dt),
             "Fshort":float(np.sum(short*cshort)*dt),
             "FV2G":float(np.sum(p*cv2g[:,None])*dt),
             "Fdeg":float(np.sum(p*cdeg[:,None]/eta[:,None])*dt)}
    costs["Qop"] = sum(costs.values())
    full_shed_cost = float(np.sum(load_p*cshed[None,:])*dt)
    cmax = full_shed_cost+float(np.sum(interface*cshort)*dt)
    fallback_cost = full_shed_cost+float(np.sum(station_r*cshort)*dt)
    _require(all(np.isfinite(v) for v in (*costs.values(),cmax,fallback_cost)), "Nonfinite cost arithmetic")
    oa, ore = policy["objective_comparison_USD_absolute"], policy["objective_comparison_relative"]
    objective_tol = oa+ore*max(1.,abs(cmax),abs(fallback_cost),abs(costs["Qop"]))
    lower_ok = costs["Qop"] >= -objective_tol
    cmax_ok = costs["Qop"] <= cmax+objective_tol
    fallback_ok = costs["Qop"] <= fallback_cost+objective_tol
    bound_ok = lower_ok and cmax_ok and fallback_ok
    physics_passed = first_stage["passed"] and all(row["passed"] for row in residuals.values())
    result = {"passed":bool(physics_passed and (bound_ok if check_objective_bound else True)),
        "physical_feasible":bool(physics_passed), "optimality_certified":False,
        "task_sha256":tasks.task_sha256, "config_hash":tasks.config_hash,
        "scenario_sha256":scene.scenario_sha256, "case_sha256":case.case_sha256,
        "first_stage":first_stage, "residuals":residuals, "costs":costs,
        "cmax_usd":cmax, "fallback_cost_usd":fallback_cost,
        "objective_bound":{"enforced":bool(check_objective_bound), "passed":bool(bound_ok),
            "nonnegative":bool(lower_ok), "within_cmax":bool(cmax_ok),
            "within_fixed_plan_fallback":bool(fallback_ok), "tolerance_usd":objective_tol,
            "scope":"Necessary bound on optimal value; not an optimality certificate"},
        "max_normalized_violation":max(first_stage["max_normalized_violation"],
                                        max(row["max_normalized_violation"] for row in residuals.values())),
        "reference_bus_ids_by_period":references,
        "dimensions":{"tasks":h,"periods":t,"buses":n,"lines":ell,"stations":c},
        "tolerances":{**policy,"source":policy_source,
            "energy_original_tolerance_rule":"((period-1)*dt_h + tau_blk_h)/eta_dis * power_tolerance",
            "normalization":"max(1, fixed physical capacity); SOC and squared voltage use 1",
            "acceptance":"Both original-unit and normalized physical tolerances must pass"}}
    json.dumps(result, allow_nan=False)
    return result
