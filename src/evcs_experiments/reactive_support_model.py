"""Reusable v54-Q1 input adapter and linear capability coefficients.

The polygon acts on (accredited ACTIVE reserve, scheduled REACTIVE injection),
not merely actual discharge. Optimization backends should insert its rows in
the task and station blocks. No additional recourse integer variables arise.
"""
from pathlib import Path
import csv
import json
import numpy as np
from .config import get_paths

DEFAULT_ROOT = get_paths().data_root

def capability_facets(sides=16):
    """Return (A, b) for an inscribed unit disk: A @ [R_av,Q] <= b.

    Scale b by S_inv * mu_full * x for a task, or S_station for a station.
    sides divisible by four ensures symmetry in both axes and the downward
    active-power property required by 0 <= P_dispatch <= R_av.
    """
    if not isinstance(sides, int) or sides < 4 or sides % 4:
        raise ValueError("polygon_sides must be a positive multiple of four >= 4")
    theta = (2*np.arange(sides)+1)*np.pi/sides
    return np.column_stack((np.cos(theta), np.sin(theta))), np.full(sides, np.cos(np.pi/sides))

def capability_inequalities(rating_kVA, availability=1.0, sides=16):
    """Return A,rhs. For a variable x, multiply rhs by x in the MILP.

    availability is mu_full for the physical task model. M6 instead supplies
    its frozen pooled availability coefficient and selected-cohort sum.
    """
    if rating_kVA < 0 or not 0 <= availability <= 1:
        raise ValueError("nonnegative rating and availability in [0,1] required")
    A,b=capability_facets(sides)
    return A,b*rating_kVA*availability

def load_case(root=None):
    """Read authoritative Q1 settings and exact-PF model loads.

    Original archived load CSV Q is retained in Q_archived_n_t_kvar. Its small
    rounding discrepancy is removed in the model, preserving complete recourse
    at full shedding. Do not reconstruct active Q injection using legacy PF=0.
    """
    root=Path(root) if root is not None else get_paths().data_root
    def read(name):
        with (root/name).open(newline="",encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    settings=json.loads((root/"01_config/reactive_support__v54_q1.json").read_text())
    if settings["mode"]!="controllable_polygon" or settings["capability_active_coordinate"]!="accredited_reserve_r_av":
        raise ValueError("unsupported reactive support mode or certification coordinate")
    capability_facets(settings["polygon_sides"])
    fleet=read("06_fleet/E__vehicle_parameters.csv")
    stations=read("05_evcs/C__evcs_bus_road_mapping.csv")
    multiplier=float(settings["apparent_power_multiplier"])
    for rows,scol,pcol in [(fleet,"S_inv_bar_e_kVA","P_dis_bar_e_kW"),(stations,"S_int_bar_c_kVA","P_int_bar_c_kW")]:
        for r in rows:
            if not np.isclose(float(r[scol]),multiplier*float(r[pcol]),rtol=0,atol=1e-9):
                raise ValueError("rating does not match the documented synthetic multiplier")
    buses=read("02_power_network/N_P_rated_Q_rated__bus_parameters.csv")
    pf={int(r["n"]):float(r["Q_rated_n_kvar"])/float(r["P_rated_n_kW"]) if float(r["P_rated_n_kW"]) else 0. for r in buses}
    loads=[]
    for row in read("03_load/P_base_Q_base__N_T.csv"):
        p=float(row["P_base_n_t_kW"]); n=int(row["n"])
        loads.append(dict(n=n,t=int(row["t"]),P_base_n_t_kW=p,
            Q_base_n_t_kvar=p*pf[n],Q_archived_n_t_kvar=float(row["Q_base_n_t_kvar"]),tan_phi_load_n=pf[n]))
    return dict(settings=settings,fleet=fleet,stations=stations,buses=buses,loads=loads,
                max_load_Q_rounding_adjustment_kvar=max(abs(r["Q_base_n_t_kvar"]-r["Q_archived_n_t_kvar"]) for r in loads))

if __name__=="__main__":
    case=load_case()
    print(json.dumps(dict(revision=case["settings"]["revision"],fleet_count=len(case["fleet"]),
        station_count=len(case["stations"]),load_rows=len(case["loads"]),
        polygon_sides=case["settings"]["polygon_sides"],
        max_load_Q_rounding_adjustment_kvar=case["max_load_Q_rounding_adjustment_kvar"]),indent=2))
