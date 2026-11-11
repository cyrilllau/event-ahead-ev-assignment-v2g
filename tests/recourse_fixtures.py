"""Explicit small physical fixtures, independent of LP/checker assembly.

These synthetic validation fixtures deliberately use small horizons and eta=1
unless a test requests another efficiency. They are not accepted case data or
planning scenarios. Each load uses one exact rated Q/P ratio per bus.
"""
from types import SimpleNamespace

import numpy as np

from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.random_streams import canonical_hash


def make_fixture(*, periods=1, bus_ids=(1,), station_bus_ids=(1,), line_edges=(),
                 task_station_indices=(0,), load_p=None, tan_phi=.5,
                 reserve=10., selected=1., arrival_soc=.8, available=True,
                 eta=1., tau=1., outages=None, start_periods=None):
    """Return namespace(case,tasks,scenario,plan) using production field names.

    Array orientations: H,T task arrays; T,N load/voltage; T,L branch flow;
    C,T station shortfall. Scalar arguments broadcast to their full axes.
    """
    t=int(periods); n=len(bus_ids); c=len(station_bus_ids); h=len(task_station_indices); ell=len(line_edges)
    station_ids=tuple(f'CS{i+1}' for i in range(c))
    task_ids=tuple(f'EV{i+1}__{station_ids[s]}__k01' for i,s in enumerate(task_station_indices))
    periods_axis=tuple(range(1,t+1))
    def array(value,shape,dtype=float):
        return np.broadcast_to(np.asarray(value,dtype=dtype),shape).copy()
    phi=array(tan_phi,(n,))
    p=np.zeros((t,n)) if load_p is None else array(load_p,(t,n))
    q=p*phi[None,:]
    cfg=canonical_hash({'fixture':True,'periods':t,'tau':tau,'eta':np.asarray(eta).tolist()})
    case_hash=canonical_hash({'fixture':True,'buses':list(bus_ids),'stations':list(station_bus_ids),'lines':list(line_edges)})
    task_hash=canonical_hash({'fixture':True,'task_ids':task_ids,'config_hash':cfg})
    starts=np.ones(h,dtype=int) if start_periods is None else array(start_periods,(h,),int)
    occ=np.arange(1,t+1)[None,:]>=starts[:,None]
    res=occ & (np.arange(t)[None,:]+tau<=t)
    block=occ & (np.arange(1,t+1)[None,:]<starts[:,None]+int(tau))
    ids=tuple(f'EV{i+1}__{station_ids[s]}__k{starts[i]:02d}' for i,s in enumerate(task_station_indices))
    assert ids==task_ids or start_periods is not None
    task_ids=ids
    task_hash=canonical_hash({'fixture':True,'task_ids':task_ids,'config_hash':cfg})
    si=np.array(task_station_indices,dtype=int)
    bi=np.array([bus_ids.index(station_bus_ids[i]) for i in si],dtype=int)
    eff=array(eta,(h,)); soc=array(arrival_soc,(h,)); success=array(available,(h,),bool)
    case=SimpleNamespace(periods=periods_axis,bus_ids=tuple(bus_ids),line_ids=tuple(range(1,ell+1)),fixed_open_ties=(),
        station_ids=station_ids,station_bus_ids=tuple(station_bus_ids),station_plugs=np.full(c,20),
        station_interface_kw=np.full(c,300.),station_rating_kva=np.full(c,375.),dt_h=1.,
        line_from_bus_ids=tuple(e[0] for e in line_edges),line_to_bus_ids=tuple(e[1] for e in line_edges),
        line_m_p_per_kw=np.full(ell,.001),line_m_q_per_kvar=np.full(ell,.001),
        line_p_limit_kw=np.full(ell,100.),line_q_limit_kvar=np.full(ell,100.),tan_phi_load=phi,
        c_shed_usd_per_kwh=np.full(n,3.),critical_buses=np.zeros(n,dtype=bool),
        baseline_p_kw=p.copy(),baseline_q_kvar=q.copy(),archived_baseline_q_kvar=q.copy(),
        u_min=.9025,u_max=1.1025,u_ref=1.,m_voltage=.2,s_base_kva=10000.,polygon_sides=16,
        c_short_usd_per_kwh=1.,case_sha256=case_hash,manifest_sha256='fixture',
        model_revision='v54-q1',capability_active_coordinate='accredited_reserve_r_av',
        input_hashes={},parameters={'fixture_only':True})
    def groups(keys): return {k:tuple(i for i,x in enumerate(keys) if x==k) for k in dict.fromkeys(keys)}
    vehicle_ids=tuple(f'EV{i+1}' for i in range(h));task_stations=tuple(station_ids[s] for s in si)
    tasks=SimpleNamespace(task_ids=task_ids,vehicle_ids=vehicle_ids,station_ids=task_stations,
        station_indices=si,bus_indices=bi,start_periods=starts,chi_occ=occ,chi_res=res,block_mask=block,
        p_dis_kw=np.full(h,15.),p_block_kw=np.full(h,1.),eta_dis=eff,e_forecast_dc_kwh=np.full(h,48.),
        c_act_usd=np.full(h,25.),c_cap_usd_per_kwh=np.full((h,t),.02),e_cap_kwh=np.full(h,80.),
        soc_departure_min=np.full(h,.2),s_inv_kva=np.full(h,18.75),c_v2g_usd_per_kwh=np.full(h,.12),
        c_deg_usd_per_kwh=np.full(h,.05),qualification_soc=np.maximum(.3,.2+tau/(eff*80)),
        forecast_arrival_soc=np.full(h,.8),forecast_arrival_h=starts.astype(float)-1.,tau_blk_h=float(tau),
        vehicle_task_indices=groups(vehicle_ids),station_task_indices=groups(task_stations),
        bus_task_indices=groups([bus_ids[i] for i in bi]),task_sha256=task_hash,config_hash=cfg,
        physical_case_sha256=case_hash,source_sha256='fixture',dependency_sha256='fixture',
        config={'fixture_only':True,'tau_blk_h':float(tau)},source_path='synthetic validation fixture')
    op=np.zeros((t,ell),dtype=bool) if outages is None else array(outages,(t,ell),bool)
    scene_hash=canonical_hash({'fixture':True,'task_sha256':task_hash,'p':p.tolist(),'q':q.tolist(),
                              'soc':soc.tolist(),'availability':success.tolist(),'outages':op.tolist()})
    scenario=SimpleNamespace(target_event='synthetic_fixture',split='development_fixture',
        base_weather_event='synthetic_fixture',variant=1,regime='R1',development=True,
        scenario_id='synthetic_fixture',scenario_sha256=scene_hash,task_sha256=task_hash,config_hash=cfg,
        physical_case_sha256=case_hash,task_ids=task_ids,bus_ids=tuple(bus_ids),line_ids=case.line_ids,
        periods=periods_axis,times_h=np.arange(t,dtype=float),line_operating_outages=op,
        load_p_kw=p,load_q_kvar=q,load_archived_q_kvar=q.copy(),task_mu_full=res&success[:,None],
        task_varsigma=success,task_arrival_soc=soc,
        task_arrival_h=np.where(success,starts.astype(float)-1.,float(t+1)),
        task_gamma_route=np.ones(h,dtype=bool),task_usable_energy_dc_kwh=success*np.maximum(soc-.2,0)*80.,
        source_hashes={},metadata={'fixture_only':True})
    r=array(reserve,(h,t))*res
    x=array(selected,(h,));r*=x[:,None]
    plan=FirstStagePlan(x,r,task_hash,cfg)
    return SimpleNamespace(case=case,tasks=tasks,scenario=scenario,plan=plan)


def manual_solution(fixture, *, rav, p, q, short=None, shed=None, line_p=None,
                    line_q=None, u=None, soc_dep=None):
    """Make explicit candidate arrays without consulting LP or checker code."""
    f=fixture; h=len(f.tasks.task_ids);t=len(f.case.periods);n=len(f.case.bus_ids)
    c=len(f.case.station_ids);ell=len(f.case.line_ids)
    def array(value,shape): return np.broadcast_to(np.asarray(value,dtype=float),shape).copy()
    discharge=array(p,(h,t))
    if soc_dep is None:
        soc_dep=f.scenario.task_arrival_soc*f.plan.x-np.sum(discharge,axis=1)*f.case.dt_h/(f.tasks.eta_dis*f.tasks.e_cap_kwh)
    return SimpleNamespace(rav=array(rav,(h,t)),p=discharge,q=array(q,(h,t)),
        short=np.zeros((c,t)) if short is None else array(short,(c,t)),
        shed=np.zeros((t,n)) if shed is None else array(shed,(t,n)),
        line_p=np.zeros((t,ell)) if line_p is None else array(line_p,(t,ell)),
        line_q=np.zeros((t,ell)) if line_q is None else array(line_q,(t,ell)),
        u=np.ones((t,n)) if u is None else array(u,(t,n)),soc_dep=array(soc_dep,(h,)),
        task_sha256=f.tasks.task_sha256,config_hash=f.tasks.config_hash,
        scenario_sha256=f.scenario.scenario_sha256)


def golden_single_bus():
    return make_fixture(load_p=[[6.]],tan_phi=.5,reserve=10.)


def golden_reverse_flow():
    return make_fixture(bus_ids=(1,2),station_bus_ids=(2,),line_edges=((1,2),),load_p=[[6.,0.]],tan_phi=.5)


def golden_islands(selected=1.):
    return make_fixture(bus_ids=(1,2,3),station_bus_ids=(1,),line_edges=((1,2),(2,3)),
                        load_p=[[0.,4.,5.]],tan_phi=0.,outages=[[False,True]],selected=selected)
