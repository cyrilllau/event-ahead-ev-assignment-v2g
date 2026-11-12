"""v54-Q1 recourse LP with explicit first-stage RHS coefficient blocks.

No route, assignment, upstream injection, or switching decision is introduced.
All H tasks remain present. Powers are AC kW/kvar, energy is DC kWh, and
voltages are squared per unit. An island voltage reference supplies no power.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from types import MappingProxyType
import time
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csr_matrix

from .first_stage import FirstStagePlan, assemble_first_stage
from .reactive_support_model import capability_facets


def _array(value, dtype=float):
    source = np.asarray(value, dtype=dtype)
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


def _numbers(value, shape, name, *, nonnegative=True):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)) or (nonnegative and np.any(array < 0)):
        raise ValueError(f"Invalid {name}: expected finite {'nonnegative ' if nonnegative else ''}{shape}")
    return _array(array)


def _mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"Invalid Boolean {name} shape or values")
    return _array(array, bool)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _csr(values, rows, columns, shape):
    matrix = csr_matrix((values, (rows, columns)), shape=shape, dtype=float)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.data = _array(matrix.data)
    matrix.indices = _array(matrix.indices, matrix.indices.dtype)
    matrix.indptr = _array(matrix.indptr, matrix.indptr.dtype)
    return matrix


@dataclass(frozen=True)
class RecourseSolution:
    rav: np.ndarray
    p: np.ndarray
    q: np.ndarray
    short: np.ndarray
    shed: np.ndarray
    line_p: np.ndarray
    line_q: np.ndarray
    u: np.ndarray
    soc_dep: np.ndarray
    task_sha256: str
    config_hash: str
    scenario_sha256: str

    def __post_init__(self):
        for item in fields(self):
            if item.name.endswith("sha256") or item.name == "config_hash":
                if not isinstance(getattr(self, item.name), str) or not getattr(self, item.name):
                    raise ValueError("Solution identities must be nonempty strings")
            else:
                array = _array(getattr(self, item.name))
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"Nonfinite solution field: {item.name}")
                object.__setattr__(self, item.name, array)
        if self.rav.ndim != 2:
            raise ValueError("rav must have shape [H,T]")
        h, t = self.rav.shape
        if (self.p.shape != (h,t) or self.q.shape != (h,t) or self.soc_dep.shape != (h,)
                or self.short.ndim != 2 or self.short.shape[1] != t
                or self.shed.ndim != 2 or self.shed.shape[0] != t or self.u.shape != self.shed.shape
                or self.line_p.ndim != 2 or self.line_p.shape[0] != t or self.line_q.shape != self.line_p.shape):
            raise ValueError("Recourse solution field axes disagree")


@dataclass(frozen=True)
class RecourseSolveResult:
    status: str
    optimal: bool
    solution: RecourseSolution | None
    objective_usd: float | None
    objective_components: object
    solver_status: int
    message: str
    certificate: object
    wall_seconds: float

    def as_dict(self):
        return {"status": self.status, "optimal": self.optimal, "has_solution": self.solution is not None,
            "objective_usd": self.objective_usd, "objective_components": _json(self.objective_components),
            "solver_status": self.solver_status, "message": self.message,
            "wall_seconds": self.wall_seconds, "certificate": _json(self.certificate)}


@dataclass(frozen=True)
class RecourseTemplate:
    A_ub: csr_matrix
    b_ub: np.ndarray
    Bx_ub: csr_matrix
    Br_ub: csr_matrix
    A_eq: csr_matrix
    b_eq: np.ndarray
    Bx_eq: csr_matrix
    Br_eq: csr_matrix
    c: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    variable_map: tuple
    constraint_map_ub: tuple
    constraint_map_eq: tuple
    reference_bus_ids_by_period: tuple
    template_sha256: str
    cmax_usd: float
    task_ids: tuple
    station_ids: tuple
    bus_ids: tuple
    line_ids: tuple
    periods: tuple
    task_sha256: str
    config_hash: str
    scenario_sha256: str
    variable_slices: object
    variable_shapes: object
    component_objectives: np.ndarray
    first_stage: object
    load_p_kw: np.ndarray
    arrival_soc: np.ndarray
    station_task_indices: tuple
    u_ref: float
    units: object

    def fixed_rhs(self, plan):
        self.first_stage.vector(plan)  # Identity/shape check; no clipping or plan substitution.
        reserve = plan.r.ravel(order="C")
        return (_array(self.b_ub + self.Bx_ub @ plan.x + self.Br_ub @ reserve),
                _array(self.b_eq + self.Bx_eq @ plan.x + self.Br_eq @ reserve))

    def pack(self, solution):
        if (solution.task_sha256 != self.task_sha256 or solution.config_hash != self.config_hash
                or solution.scenario_sha256 != self.scenario_sha256):
            raise ValueError("Solution task/config/scenario identity mismatch")
        arrays = []
        for name, shape in self.variable_shapes.items():
            value = getattr(solution, name)
            if value.shape != shape:
                raise ValueError(f"Solution {name} shape differs from template")
            arrays.append(value.ravel(order="C"))
        return _array(np.concatenate(arrays))

    def unpack(self, vector):
        vector = _numbers(vector, self.c.shape, "solution vector", nonnegative=False)
        values = {name: vector[slice(*self.variable_slices[name])].reshape(shape)
                  for name, shape in self.variable_shapes.items()}
        return RecourseSolution(**values, task_sha256=self.task_sha256,
                                config_hash=self.config_hash, scenario_sha256=self.scenario_sha256)

    def objective_components(self, solution):
        values = self.component_objectives @ self.pack(solution)
        names = ("Fshed", "Fshort", "FV2G", "Fdeg")
        return {**dict(zip(names, map(float, values))), "Qop": float(np.sum(values))}

    def as_dict(self):
        return {"template_sha256": self.template_sha256, "task_sha256": self.task_sha256,
            "config_hash": self.config_hash, "scenario_sha256": self.scenario_sha256,
            "task_ids": list(self.task_ids), "station_ids": list(self.station_ids),
            "bus_ids": list(self.bus_ids), "line_ids": list(self.line_ids), "periods": list(self.periods),
            "shape_ub": list(self.A_ub.shape), "shape_eq": list(self.A_eq.shape),
            "rhs_form": "b + Bx @ x + Br @ r.ravel(order='C')", "cmax_usd": self.cmax_usd,
            "reference_bus_ids_by_period": _json(self.reference_bus_ids_by_period),
            "variable_slices": _json(self.variable_slices), "variable_shapes": _json(self.variable_shapes),
            "units": dict(self.units), "variable_map": _json(self.variable_map),
            "constraint_map_ub": _json(self.constraint_map_ub), "constraint_map_eq": _json(self.constraint_map_eq)}


def assemble_recourse(case, tasks, scene):
    first = assemble_first_stage(case, tasks)
    H, T = len(tasks.task_ids), len(case.periods)
    buses, lines, stations = tuple(case.bus_ids), tuple(case.line_ids), tuple(case.station_ids)
    N, L, C = len(buses), len(lines), len(stations)
    if not N or len(set(buses)) != N or len(set(lines)) != L:
        raise ValueError("Bus and line axes must be unique and buses nonempty")
    if (tuple(scene.task_ids) != tuple(tasks.task_ids) or tuple(scene.bus_ids) != buses
            or tuple(scene.line_ids) != lines or tuple(scene.periods) != tuple(case.periods)
            or scene.task_sha256 != tasks.task_sha256 or scene.config_hash != tasks.config_hash
            or scene.physical_case_sha256 != case.case_sha256 or tasks.physical_case_sha256 != case.case_sha256):
        raise ValueError("Case/task/scenario identities or axes disagree")
    if not isinstance(scene.scenario_sha256, str) or not scene.scenario_sha256:
        raise ValueError("Scenario identity is required")
    if case.model_revision != "v54-q1" or case.capability_active_coordinate != "accredited_reserve_r_av":
        raise ValueError("Q1 requires accredited reserve as the active capability coordinate")
    if set(lines) & set(case.fixed_open_ties):
        raise ValueError("Fixed-open ties cannot enter the recourse branch set")
    bus_index = {bus:n for n,bus in enumerate(buses)}
    station_index = {station:c for c,station in enumerate(stations)}
    endpoints = list(zip(case.line_from_bus_ids, case.line_to_bus_ids))
    if len(endpoints) != L or any(a not in bus_index or b not in bus_index or a == b for a,b in endpoints):
        raise ValueError("Invalid branch endpoints")
    if any(bus not in bus_index for bus in case.station_bus_ids):
        raise ValueError("Unknown station bus")
    station_tasks = tuple(tuple(h for h,c in enumerate(tasks.station_ids) if c == station) for station in stations)
    task_bus = [bus_index[case.station_bus_ids[station_index[c]]] for c in tasks.station_ids]
    bus_tasks = [tuple(h for h,n in enumerate(task_bus) if n == bus) for bus in range(N)]
    mu = _mask(scene.task_mu_full, (H,T), "task_mu_full")
    success = _mask(scene.task_varsigma, (H,), "task_varsigma")
    outages = _mask(scene.line_operating_outages, (T,L), "line_operating_outages")
    if np.any(mu & ~success[:,None]):
        raise ValueError("Unavailable task cannot have full-period availability")
    soc = _numbers(scene.task_arrival_soc, (H,), "arrival SOC")
    if np.any(soc > 1):
        raise ValueError("Arrival SOC exceeds one")
    energy = _numbers(scene.task_usable_energy_dc_kwh, (H,), "usable DC energy")
    load_p = _numbers(scene.load_p_kw, (T,N), "load P")
    load_q = _numbers(scene.load_q_kvar, (T,N), "load Q")
    tanphi = _numbers(case.tan_phi_load, (N,), "rated Q/P")
    if not np.allclose(load_q, load_p*tanphi[None,:], rtol=0, atol=2e-12):
        raise ValueError("Q demand must use the exact rated-PF mapping")
    eta = _numbers(tasks.eta_dis, (H,), "eta_dis")
    capacity = _numbers(tasks.e_cap_kwh, (H,), "battery capacity")
    if np.any(eta <= 0) or np.any(eta > 1) or np.any(capacity <= 0):
        raise ValueError("Invalid efficiency or battery capacity")
    pdis = _numbers(tasks.p_dis_kw, (H,), "Pdis")
    dep = _numbers(tasks.soc_departure_min, (H,), "departure SOC")
    if np.any(dep > 1):
        raise ValueError("Departure SOC exceeds one")
    sinv = _numbers(tasks.s_inv_kva, (H,), "inverter kVA")
    srating = _numbers(case.station_rating_kva, (C,), "station kVA")
    plimit = _numbers(case.line_p_limit_kw, (L,), "line P limit")
    qlimit = _numbers(case.line_q_limit_kvar, (L,), "line Q limit")
    mp = _numbers(case.line_m_p_per_kw, (L,), "voltage P slope")
    mq = _numbers(case.line_m_q_per_kvar, (L,), "voltage Q slope")
    cshed = _numbers(case.c_shed_usd_per_kwh, (N,), "shedding cost")
    cv2g = _numbers(tasks.c_v2g_usd_per_kwh, (H,), "V2G cost")
    cdeg = _numbers(tasks.c_deg_usd_per_kwh, (H,), "degradation cost")
    dt, tau, cshort = float(case.dt_h), float(tasks.tau_blk_h), float(case.c_short_usd_per_kwh)
    umin, umax, uref, bigm = map(float, (case.u_min, case.u_max, case.u_ref, case.m_voltage))
    if (not all(np.isfinite(v) for v in (dt,tau,cshort,umin,umax,uref,bigm))
            or dt <= 0 or tau < dt or cshort < 0 or not 0 < umin <= uref <= umax or bigm < 0):
        raise ValueError("Invalid operating time, voltage or cost constants")
    facets, facet_rhs = capability_facets(case.polygon_sides)
    shapes = {"rav":(H,T), "p":(H,T), "q":(H,T), "short":(C,T), "shed":(T,N),
              "line_p":(T,L), "line_q":(T,L), "u":(T,N), "soc_dep":(H,)}
    slices, count, variable_map = {}, 0, []
    unit_map = {"rav":"kW_AC", "p":"kW_AC", "q":"kvar", "short":"kW_AC", "shed":"kW_AC",
                "line_p":"kW_AC", "line_q":"kvar", "u":"pu_squared", "soc_dep":"fraction"}
    labels = {"rav":"eq:commitment_certification_dispatch_chain", "p":"eq:commitment_certification_dispatch_chain",
        "q":"eq:task_apparent_capability", "short":"eq:station_reserve_shortfall", "shed":"eq:load_shedding_bounds",
        "line_p":"eq:scenario_line_limits", "line_q":"eq:scenario_line_limits", "u":"eq:voltage_limits",
        "soc_dep":"eq:departure_soc_balance"}
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        slices[name] = (count, count+size)
        for coordinate in np.ndindex(shape):
            entry = {"index":count+int(np.ravel_multi_index(coordinate,shape)), "name":name,
                     "unit":unit_map[name], "eq_label":labels[name], "task_id":None, "station_id":None,
                     "bus_id":None, "line_id":None, "period":None}
            if name in ("rav","p","q"):
                h,t = coordinate; entry.update(task_id=tasks.task_ids[h], period=case.periods[t])
            elif name == "soc_dep":
                entry["task_id"] = tasks.task_ids[coordinate[0]]
            elif name == "short":
                c,t = coordinate; entry.update(station_id=stations[c], period=case.periods[t])
            elif name in ("shed","u"):
                t,n = coordinate; entry.update(bus_id=buses[n], period=case.periods[t])
            else:
                t,ell = coordinate; entry.update(line_id=lines[ell], period=case.periods[t])
            variable_map.append(MappingProxyType(entry))
        count += size

    def col(name, *coordinate):
        return slices[name][0] + int(np.ravel_multi_index(coordinate, shapes[name]))

    lower, upper = np.zeros(count), np.full(count,np.inf)
    for name in ("q","line_p","line_q"):
        lower[slice(*slices[name])] = -np.inf
    upper[slice(*slices["shed"])] = load_p.ravel()
    for name, limit in (("line_p",plimit),("line_q",qlimit)):
        bounds = ((~outages)*limit[None,:]).ravel()
        lower[slice(*slices[name])], upper[slice(*slices[name])] = -bounds, bounds
    lower[slice(*slices["u"])], upper[slice(*slices["u"])] = umin, umax
    upper[slice(*slices["soc_dep"])] = 1.
    costs = np.zeros((4,count))
    costs[0,slice(*slices["shed"])] = np.broadcast_to(cshed[None,:]*dt,(T,N)).ravel()
    costs[1,slice(*slices["short"])] = cshort*dt
    costs[2,slice(*slices["p"])] = np.broadcast_to(cv2g[:,None]*dt,(H,T)).ravel()
    costs[3,slice(*slices["p"])] = np.broadcast_to(cdeg[:,None]*dt/eta[:,None],(H,T)).ravel()
    stores = {sense:{"a":([],[],[]),"x":([],[],[]),"r":([],[],[]),"b":[],"map":[]} for sense in ("ub","eq")}

    def add(sense, name, label, unit, coefficients, rhs=0., bx=(), br=(), **indices):
        store = stores[sense]; row = len(store["b"])
        for kind, terms in (("a",coefficients),("x",bx),("r",br)):
            values,rows,columns = store[kind]
            for column,value in terms:
                if value != 0:
                    values.append(float(value)); rows.append(row); columns.append(int(column))
        store["b"].append(float(rhs))
        store["map"].append(MappingProxyType({"index":row,"name":name,"eq_label":label,"unit":unit,
            "task_id":None,"station_id":None,"bus_id":None,"line_id":None,"period":None,"facet":None,**indices}))

    for h,task in enumerate(tasks.task_ids):
        for t,period in enumerate(case.periods):
            identity = {"task_id":task,"period":period}
            add("ub","dispatch_certification","eq:commitment_certification_dispatch_chain","kW_AC",
                [(col("p",h,t),1),(col("rav",h,t),-1)], **identity)
            add("ub","certification_procurement","eq:commitment_certification_dispatch_chain","kW_AC",
                [(col("rav",h,t),1)], br=[(h*T+t,1)], **identity)
            add("ub","certified_availability","eq:certified_reserve_availability","kW_AC",
                [(col("rav",h,t),1)], bx=[(h,pdis[h]*mu[h,t])], **identity)
            add("ub","reserve_sustainability","eq:reserve_sustainability","kWh_DC",
                [(col("p",h,j),dt/eta[h]) for j in range(t)]+[(col("rav",h,t),tau/eta[h])],
                bx=[(h,energy[h])], **identity)
            for facet,(ac,qc) in enumerate(facets):
                add("ub","task_capability","eq:task_apparent_capability","kVA",
                    [(col("rav",h,t),ac),(col("q",h,t),qc)],
                    bx=[(h,sinv[h]*facet_rhs[facet]*mu[h,t])], facet=facet, **identity)
        add("eq","departure_soc_balance","eq:departure_soc_balance","fraction",
            [(col("soc_dep",h),1)]+[(col("p",h,t),dt/(eta[h]*capacity[h])) for t in range(T)],
            bx=[(h,soc[h])], task_id=task)
        add("ub","departure_soc_requirement","eq:departure_soc_requirement","fraction",
            [(col("soc_dep",h),-1)], bx=[(h,-dep[h]*success[h])], task_id=task)
    for c,station in enumerate(stations):
        for t,period in enumerate(case.periods):
            add("ub","station_shortfall","eq:station_reserve_shortfall","kW_AC",
                [(col("short",c,t),-1)]+[(col("rav",h,t),-1) for h in station_tasks[c]],
                br=[(h*T+t,-1) for h in station_tasks[c]], station_id=station,period=period)
            for facet,(ac,qc) in enumerate(facets):
                add("ub","station_capability","eq:station_apparent_capability","kVA",
                    [(col("rav",h,t),ac) for h in station_tasks[c]]+[(col("q",h,t),qc) for h in station_tasks[c]],
                    rhs=srating[c]*facet_rhs[facet],station_id=station,period=period,facet=facet)
    references = []
    for t,period in enumerate(case.periods):
        for n,bus in enumerate(buses):
            pterms, qterms = [(col("shed",t,n),1)], [(col("shed",t,n),tanphi[n])]
            pterms += [(col("p",h,t),1) for h in bus_tasks[n]]
            qterms += [(col("q",h,t),1) for h in bus_tasks[n]]
            for ell,(source,target) in enumerate(endpoints):
                sign = int(target == bus)-int(source == bus)
                if sign:
                    pterms.append((col("line_p",t,ell),sign)); qterms.append((col("line_q",t,ell),sign))
            add("eq","active_balance","eq:active_power_balance","kW_AC",pterms,load_p[t,n],bus_id=bus,period=period)
            add("eq","reactive_balance","eq:reactive_power_balance","kvar",qterms,load_q[t,n],bus_id=bus,period=period)
        adjacency = {bus:set() for bus in buses}
        for ell,(source,target) in enumerate(endpoints):
            terms = [(col("u",t,bus_index[target]),1),(col("u",t,bus_index[source]),-1),
                     (col("line_p",t,ell),mp[ell]),(col("line_q",t,ell),mq[ell])]
            for sign in (1,-1):
                add("ub","voltage_drop_upper" if sign == 1 else "voltage_drop_lower",
                    "eq:scenario_voltage_drop","pu_squared",[(c,sign*v) for c,v in terms],bigm*outages[t,ell],
                    line_id=lines[ell],period=period)
            if not outages[t,ell]:
                adjacency[source].add(target); adjacency[target].add(source)
        unseen, refs = set(buses), []
        while unseen:
            seed = min(unseen); component, frontier = set(), [seed]
            while frontier:
                bus = frontier.pop()
                if bus in component:
                    continue
                component.add(bus); frontier.extend(adjacency[bus]-component)
            unseen -= component
            candidates = component & set(case.station_bus_ids)
            reference = min(candidates if candidates else component)
            refs.append(reference)
            add("eq","reference_voltage","eq:reference_voltage","pu_squared",
                [(col("u",t,bus_index[reference]),1)],uref,bus_id=reference,period=period)
        references.append(tuple(sorted(refs)))
    assembled = {}
    for sense,store in stores.items():
        rows = len(store["b"])
        for kind,width in (("a",count),("x",H),("r",H*T)):
            assembled[kind+"_"+sense] = _csr(*store[kind],(rows,width))
        assembled["b_"+sense] = _array(store["b"])
    cmax = float(np.sum(load_p*cshed[None,:])*dt+cshort*np.sum(case.station_interface_kw)*T*dt)
    identity = {"task_sha256":tasks.task_sha256,"config_hash":tasks.config_hash,"scenario_sha256":scene.scenario_sha256,
        "first_stage_sha256":first.matrix_sha256,"shapes":shapes,"task_ids":tasks.task_ids,"bus_ids":buses,
        "line_ids":lines,"station_ids":stations,"periods":case.periods,"references":references,"cmax_usd":cmax}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode())
    for key in sorted(assembled):
        value = assembled[key]
        arrays = (value.data,value.indices,value.indptr) if isinstance(value,csr_matrix) else (value,)
        digest.update(key.encode())
        for array in arrays:
            canonical = np.asarray(array,dtype="<i8" if array.dtype.kind in "iu" else "<f8")
            digest.update(str(canonical.shape).encode()); digest.update(canonical.tobytes())
    for array in (lower,upper,costs):
        digest.update(np.asarray(array,dtype="<f8").tobytes())
    return RecourseTemplate(assembled["a_ub"],assembled["b_ub"],assembled["x_ub"],assembled["r_ub"],
        assembled["a_eq"],assembled["b_eq"],assembled["x_eq"],assembled["r_eq"],_array(costs.sum(axis=0)),
        _array(lower),_array(upper),tuple(variable_map),tuple(stores["ub"]["map"]),tuple(stores["eq"]["map"]),
        tuple(references),digest.hexdigest(),cmax,tuple(tasks.task_ids),stations,buses,lines,tuple(case.periods),
        tasks.task_sha256,tasks.config_hash,scene.scenario_sha256,MappingProxyType(slices),MappingProxyType(shapes),
        _array(costs),first,load_p,soc,station_tasks,uref,MappingProxyType({**unit_map,"objective":"USD"}))


def _valid_plan(template, plan):
    check = template.first_stage.evaluate(plan)
    if not check.feasible:
        raise ValueError("Recourse requires a feasible fixed first-stage plan")


def full_shedding_fallback(template, plan):
    _valid_plan(template,plan)
    values = {name:np.zeros(shape) for name,shape in template.variable_shapes.items()}
    values["shed"] = template.load_p_kw
    values["u"] = np.full(template.variable_shapes["u"],template.u_ref)
    values["soc_dep"] = template.arrival_soc*plan.x
    for c,indices in enumerate(template.station_task_indices):
        values["short"][c] = plan.r[list(indices)].sum(axis=0)
    return RecourseSolution(**values,task_sha256=template.task_sha256,config_hash=template.config_hash,
                            scenario_sha256=template.scenario_sha256)


def solve_fixed_plan(template, plan, time_limit_seconds=60):
    started = time.monotonic()
    _valid_plan(template,plan)
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("LP time limit must be positive and finite")
    bub,beq = template.fixed_rhs(plan)
    options = {"time_limit":float(time_limit_seconds),"threads":1,"primal_feasibility_tolerance":1e-8,
               "dual_feasibility_tolerance":1e-8,"ipm_optimality_tolerance":1e-8}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=OptimizeWarning,message="Unrecognized options detected.*")
        result = linprog(template.c,A_ub=template.A_ub,b_ub=bub,A_eq=template.A_eq,b_eq=beq,
            bounds=np.column_stack((template.lower_bounds,template.upper_bounds)),method="highs",options=options)
    statuses = {0:"OPTIMAL",1:"LIMIT_REACHED",2:"INFEASIBLE",3:"UNBOUNDED",4:"SOLVER_ERROR"}
    solution, objective, components = None,None,{}
    certificate = {"solver":"SciPy linprog/HiGHS","solver_status":int(result.status),
        "solver_message":str(result.message),"options":options,"iterations":int(result.nit or 0),
        "template_sha256":template.template_sha256,"cmax_usd":template.cmax_usd,
        "cost_bound_scope":"optimal recourse value only; arbitrary feasible shortfall may exceed Cmax",
        "optimal_cost_bound_valid":None,"primal_bound_usd":None,"dual_bound_usd":None,
        "certificate_status":"NEED_CERTIFICATE","optimal_certificate_valid":False,
        "primal_absolute_tolerance":1e-7,"dual_stationarity_tolerance":1e-8,
        "dual_sign_and_infinite_bound_tolerance":1e-8}
    if result.x is not None and np.all(np.isfinite(result.x)):
        solution = template.unpack(result.x)
        components = template.objective_components(solution)
        objective = components["Qop"]
        ub_violation = float(np.max(template.A_ub@result.x-bub,initial=0))
        eq_violation = float(np.max(np.abs(template.A_eq@result.x-beq),initial=0))
        bound_violation = float(np.max(np.maximum(template.lower_bounds-result.x,result.x-template.upper_bounds),initial=0))
        primal_valid = max(ub_violation,eq_violation,bound_violation) <= 1e-7
        certificate.update(max_inequality_violation=ub_violation,max_equality_violation=eq_violation,
            max_bound_violation=bound_violation,primal_feasible_within_tolerance=primal_valid,
            primal_bound_usd=objective if primal_valid else None)
        if result.status == 0:
            bound_tolerance = 1e-4+1e-7*max(abs(objective),template.cmax_usd)
            certificate["optimal_cost_bound_tolerance_usd"] = bound_tolerance
            certificate["optimal_cost_bound_valid"] = bool(-bound_tolerance <= objective <= template.cmax_usd+bound_tolerance)
            duals = [np.asarray(getattr(result,name).marginals) for name in ("ineqlin","eqlin","lower","upper")]
            if all(np.all(np.isfinite(array)) for array in duals):
                y,lam,lo,hi = duals
                stationarity = template.c-template.A_ub.T@y-template.A_eq.T@lam-lo-hi
                signs = max(float(np.max(y,initial=0)),float(np.max(-lo,initial=0)),float(np.max(hi,initial=0)))
                finite_lo,finite_hi = np.isfinite(template.lower_bounds),np.isfinite(template.upper_bounds)
                dual = float(bub@y+beq@lam+template.lower_bounds[finite_lo]@lo[finite_lo]+template.upper_bounds[finite_hi]@hi[finite_hi])
                stationarity_error = float(np.max(np.abs(stationarity),initial=0))
                infinite_marginal_error = max(float(np.max(np.abs(lo[~finite_lo]),initial=0)),
                                              float(np.max(np.abs(hi[~finite_hi]),initial=0)))
                gap_tolerance = 1e-4+1e-7*max(abs(objective),abs(dual),1.)
                certificate.update(dual_objective_usd=dual,duality_gap_usd=abs(objective-dual),
                    max_dual_stationarity_violation=stationarity_error,max_dual_sign_violation=signs,
                    max_invalid_infinite_bound_marginal=infinite_marginal_error,
                    duality_gap_tolerance_usd=gap_tolerance,
                    dual_inequality_marginals=_array(y),dual_equality_marginals=_array(lam),
                    dual_lower_marginals=_array(lo),dual_upper_marginals=_array(hi))
                dual_valid = stationarity_error <= 1e-8 and signs <= 1e-8 and infinite_marginal_error <= 1e-8
                certificate["dual_bound_usd"] = dual if dual_valid else None
                certificate["optimal_certificate_valid"] = bool(dual_valid and max(ub_violation,eq_violation,bound_violation)<=1e-7
                    and abs(objective-dual)<=gap_tolerance and certificate["optimal_cost_bound_valid"])
                if certificate["optimal_certificate_valid"]:
                    certificate["certificate_status"] = "PASS_NUMERIC_LP_CERTIFICATE"
    wall_seconds = time.monotonic()-started
    certificate["wall_seconds"] = wall_seconds
    return RecourseSolveResult(statuses.get(result.status,"UNKNOWN"),result.status==0,solution,objective,
        MappingProxyType(components),int(result.status),str(result.message),MappingProxyType(certificate),wall_seconds)
