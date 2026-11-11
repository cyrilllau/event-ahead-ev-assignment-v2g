"""Persistent S02 validation toys, never production planning scenarios.

The explicit fixture constructors below were copied from the accepted test
fixture definitions on 2026-09-05 so production validation never imports tests.
The physical LP and independent checker remain the production implementations.
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


import csv
import hashlib
import json
import math
from pathlib import Path
import time
import warnings
from collections.abc import Mapping

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack

from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan, full_shedding_fallback
from evcs_experiments.recourse_check import check_recourse


SOLUTION_FIELDS = ('rav','p','q','short','shed','line_p','line_q','u','soc_dep')


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serial(value):
    if isinstance(value,np.ndarray): return _serial(value.tolist())
    if isinstance(value,np.generic): return _serial(value.item())
    if isinstance(value,Mapping): return {str(k):_serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_serial(x) for x in value]
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not math.isfinite(value): return 'Infinity' if value>0 else '-Infinity'
    if hasattr(value,'__dict__'): return _serial(vars(value))
    return value


def _json_file(path,value):
    Path(path).write_text(json.dumps(_serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _csv_file(path,rows):
    rows=list(rows)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(_serial(v),sort_keys=True) if isinstance(v,(Mapping,list,tuple,np.ndarray)) else v
                             for k,v in row.items()})


def _save_solution(path,solution):
    np.savez_compressed(path,**{name:getattr(solution,name) for name in SOLUTION_FIELDS},
        task_sha256=solution.task_sha256,config_hash=solution.config_hash,scenario_sha256=solution.scenario_sha256)


def _joint_milp(fixture,time_limit):
    """True joint x/r/recourse MILP; no fixed-roster or policy simplification."""
    f=fixture;first=assemble_first_stage(f.case,f.tasks);recourse=assemble_recourse(f.case,f.tasks,f.scenario)
    n0,n1=first.c.size,recourse.c.size
    A=vstack((hstack((first.A,csr_matrix((first.A.shape[0],n1)))),
              hstack((-recourse.Bx_ub,-recourse.Br_ub,recourse.A_ub)),
              hstack((-recourse.Bx_eq,-recourse.Br_eq,recourse.A_eq))),format='csc')
    lower=np.concatenate((first.row_lower,np.full(recourse.A_ub.shape[0],-np.inf),recourse.b_eq))
    upper=np.concatenate((first.row_upper,recourse.b_ub,recourse.b_eq))
    objective=np.concatenate((first.c,recourse.c))
    integrality=np.concatenate((first.integrality,np.zeros(n1,dtype=int)))
    bounds=Bounds(np.concatenate((first.lower_bounds,recourse.lower_bounds)),
                  np.concatenate((first.upper_bounds,recourse.upper_bounds)))
    began=time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*',category=RuntimeWarning)
        result=milp(objective,integrality=integrality,bounds=bounds,
                    constraints=LinearConstraint(A,lower,upper),
                    options={'time_limit':float(time_limit),'mip_rel_gap':1e-9,'threads':1})
    if result.status!=0 or result.x is None:
        raise ValueError(f'Joint toy MILP did not reach OPTIMAL: {result.message}')
    H,T=len(f.tasks.task_ids),len(f.case.periods)
    plan=FirstStagePlan(result.x[:H],result.x[H:n0].reshape(H,T),f.tasks.task_sha256,f.tasks.config_hash)
    solution=recourse.unpack(result.x[n0:])
    first_check=check_first_stage(f.case,f.tasks,plan)
    physical=check_recourse(f.case,f.tasks,f.scenario,plan,solution,check_objective_bound=True)
    total=first_check['costs']['Fpre_usd']+physical['costs']['Qop']
    if not first_check['passed'] or not physical['passed'] or abs(total-result.fun)>1e-7:
        raise ValueError('Joint toy MILP failed independent physical/cost reconstruction')
    certificate={'status':'OPTIMAL','solver':'scipy.optimize.milp/HiGHS','solver_status':int(result.status),
        'message':str(result.message),'wall_seconds':time.monotonic()-began,'objective_usd':float(result.fun),
        'dual_bound_usd':float(result.mip_dual_bound),'mip_gap':float(result.mip_gap),
        'mip_node_count':int(result.mip_node_count),'threads':1,'column_order':'x, r[H,T], production recourse',
        'joint_integer_variables':int(np.count_nonzero(integrality)),
        'first_stage_fixed_in_optimization':False}
    joint_problem={'A':A,'lower':lower,'upper':upper,'c':objective,'integrality':integrality,
                   'variable_lower':bounds.lb,'variable_upper':bounds.ub,'first_columns':n0}
    return plan,solution,certificate,first_check,physical,recourse,joint_problem


def export_toys(out,lp_time_limit=30):
    """Write a new persistent toy evidence pack; return a JSON-ready report.

    This function neither changes case data nor signs S02 acceptance. Positive
    cases use the production LP/checker. Explicit bad candidates are expected
    to fail named constraints and are never reported as feasible solutions.
    """
    if not np.isfinite(lp_time_limit) or lp_time_limit<=0:
        raise ValueError('Positive finite toy time limit required')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();records=[];objectives=[];maps=[]

    def persist(name,f,solution,expected,solver=None,template=None,physical=None,first_check=None):
        folder=out/name;folder.mkdir()
        if template is None: template=assemble_recourse(f.case,f.tasks,f.scenario)
        if physical is None: physical=check_recourse(f.case,f.tasks,f.scenario,f.plan,solution)
        if first_check is None: first_check=check_first_stage(f.case,f.tasks,f.plan)
        if not first_check['passed']:
            raise ValueError('Toy first stage is invalid: '+name)
        want=expected.get('physical_pass',True)
        if physical['passed'] is not want:
            raise ValueError('Unexpected physical verdict in '+name)
        for key in expected.get('failed_residuals',()):
            if physical['residuals'][key]['passed']: raise ValueError('Expected violation absent: '+name+'/'+key)
        if 'Qop' in expected and abs(physical['costs']['Qop']-expected['Qop'])>1e-7:
            raise ValueError('Hand-calculated cost differs in '+name)
        for field,value in expected.get('arrays',{}).items():
            if not np.allclose(getattr(solution,field),np.asarray(value),rtol=0,atol=1e-7):
                raise ValueError('Hand-calculated array differs: '+name+'/'+field)
        _json_file(folder/'fixture_inputs.json',f)
        np.savez_compressed(folder/'plan.npz',x=f.plan.x,r=f.plan.r,task_sha256=f.plan.task_sha256,config_hash=f.plan.config_hash)
        _save_solution(folder/('solution.npz' if want else 'rejected_candidate.npz'),solution)
        _json_file(folder/'expected.json',expected)
        _json_file(folder/'physical_check.json',physical)
        _json_file(folder/'first_stage_check.json',first_check)
        if solver is not None: _json_file(folder/'solver_result.json',solver)
        _csv_file(folder/'first_stage_variable_map.csv',template.first_stage.variable_map)
        _csv_file(folder/'first_stage_constraint_map.csv',template.first_stage.constraint_map)
        _csv_file(folder/'variable_map.csv',template.variable_map)
        _csv_file(folder/'constraint_map_ub.csv',template.constraint_map_ub)
        _csv_file(folder/'constraint_map_eq.csv',template.constraint_map_eq)
        _json_file(folder/'template_metadata.json',template.as_dict())
        row={'case_id':name,'passed':True,'expected_physical_pass':want,'observed_physical_pass':physical['passed'],
             'expectations_passed':True,'task_sha256':f.tasks.task_sha256,'scenario_sha256':f.scenario.scenario_sha256,
             'source':'explicit synthetic validation fixture','solver_optimal':None if solver is None else solver.get('optimal',solver.get('status')=='OPTIMAL')}
        records.append(row)
        objectives.append({'case_id':name,'feasible':want,**physical['costs'],'Fpre':first_check['costs']['Fpre_usd'],
                           'total_usd':first_check['costs']['Fpre_usd']+physical['costs']['Qop'],
                           'cmax_usd':physical['cmax_usd'],'fallback_cost_usd':physical['fallback_cost_usd']})
        maps.append({'case_id':name,'first_stage_variables':name+'/first_stage_variable_map.csv',
                     'first_stage_constraints':name+'/first_stage_constraint_map.csv',
                     'variables':name+'/variable_map.csv','inequalities':name+'/constraint_map_ub.csv',
                     'equalities':name+'/constraint_map_eq.csv','template_sha256':template.template_sha256})
        return physical

    def optimal(name,f,expected):
        template=assemble_recourse(f.case,f.tasks,f.scenario)
        solved=solve_fixed_plan(template,f.plan,time_limit_seconds=lp_time_limit)
        if not solved.optimal or not solved.certificate['optimal_certificate_valid']:
            raise ValueError('Toy requires numeric optimal certificate: '+name)
        persist(name,f,solved.solution,expected,solver=solved.as_dict(),template=template)
        fallback=full_shedding_fallback(template,f.plan)
        fallback_report=check_recourse(f.case,f.tasks,f.scenario,f.plan,fallback,check_objective_bound=True)
        if not fallback_report['passed']: raise ValueError('Toy fallback is infeasible: '+name)
        _save_solution(out/name/'fallback.npz',fallback);_json_file(out/name/'fallback_check.json',fallback_report)

    optimal('single_bus_paid_service',golden_single_bus(),{'Qop':1.02,'arrays':{'p':[[6.]],'q':[[3.]],'rav':[[10.]],'soc_dep':[.725]}})
    # An independent scalar facet evaluation gives the exact old Q1 boundary.
    angles=(2*np.arange(16)+1)*np.pi/16
    q_limit=float(np.min((18.75*np.cos(np.pi/16)-15*np.cos(angles))[np.sin(angles)>0]/np.sin(angles)[np.sin(angles)>0]))
    f=make_fixture(load_p=[[15.]],tan_phi=q_limit/15,reserve=15.)
    persist('positive_Q_rav15_boundary',f,manual_solution(f,rav=15.,p=15.,q=q_limit),{'Qop':2.55,'q_limit_kvar':q_limit})
    f=make_fixture(task_station_indices=(0,0),load_p=[[15.]],tan_phi=0.,reserve=[[15.],[1.]])
    persist('negative_Q_rav15_boundary',f,manual_solution(f,rav=[[15.],[1.]],p=[[15.],[0.]],q=[[-q_limit],[q_limit]]),{'Qop':2.55,'q_limit_kvar':q_limit})
    f=make_fixture(task_station_indices=(0,0),load_p=[[6.]],tan_phi=18.75/6,reserve=[[6.],[1.]])
    persist('Q_only_rav_zero',f,manual_solution(f,rav=[[6.],[0.]],p=[[6.],[0.]],q=[[0.],[18.75]],short=1.),
            {'Qop':2.02,'arrays':{'rav':[[6.],[0.]],'q':[[0.],[18.75]],'soc_dep':[.725,.8]}})
    optimal('no_selected_EV',make_fixture(load_p=[[6.]],tan_phi=.5,selected=0.),{'Qop':18.,'arrays':{'p':[[0.]],'q':[[0.]],'shed':[[6.]]}})
    optimal('unavailable_unqualified_EV',make_fixture(load_p=[[6.]],tan_phi=.5,available=False,arrival_soc=.1),
            {'Qop':28.,'arrays':{'p':[[0.]],'q':[[0.]],'soc_dep':[.1]}})
    f=make_fixture(task_station_indices=(0,0),load_p=[[6.]],tan_phi=15/6,reserve=[[15.],[6.]])
    persist('rejected_p0_rav15_q15',f,manual_solution(f,rav=[[15.],[6.]],p=[[0.],[6.]],q=[[15.],[0.]]),
            {'physical_pass':False,'failed_residuals':['task_capability'],'arrays':{'p':[[0.],[6.]]}})
    optimal('reserve_without_activation',make_fixture(periods=2,load_p=[[0.],[0.]],tan_phi=0.,reserve=15.,arrival_soc=.3875),
            {'Qop':0.,'arrays':{'rav':[[15.,15.]],'p':[[0.,0.]]}})
    f=make_fixture(periods=2,load_p=[[10.],[0.]],tan_phi=0.,reserve=15.,arrival_soc=.45)
    optimal('actual_discharge_reserve_chain',f,{'Qop':6.7,'arrays':{'rav':[[15.,10.]],'soc_dep':[.325]}})
    persist('rejected_actual_discharge_chain',f,manual_solution(f,rav=[[15.,15.]],p=[[10.,0.]],q=0.),
            {'physical_pass':False,'failed_residuals':['sustainability']})
    optimal('departure_SOC_binding',make_fixture(load_p=[[15.]],tan_phi=.6,reserve=15.,eta=.95,
            arrival_soc=.2+15/(.95*80)),{'Qop':1.8+.05*15/.95,'arrays':{'p':[[15.]],'soc_dep':[.2]}})
    optimal('tau2_efficiency_SOC',make_fixture(periods=3,tau=2.,eta=.8,arrival_soc=.7,reserve=[[15.,15.,0.]],
            load_p=[[8.],[0.],[0.]],tan_phi=0.),{'Qop':4.46,'arrays':{'rav':[[15.,12.,0.]],'soc_dep':[.575]}})
    f=golden_single_bus();f.case.station_rating_kva=np.array([10.])
    persist('rejected_station_polygon',f,manual_solution(f,rav=10.,p=6.,q=3.),
            {'physical_pass':False,'failed_residuals':['station_capability']})
    optimal('derated_station_polygon',f,{})
    optimal('reverse_signed_flow',golden_reverse_flow(),{'Qop':1.02,'arrays':{'line_p':[[-6.]],'line_q':[[-3.]],'u':[[.991,1.]]}})
    optimal('two_islands_one_without_EVCS',golden_islands(),{'Qop':15.68,'arrays':{'shed':[[0.,0.,5.]],'line_p':[[4.,0.]]}})
    optimal('two_islands_zero_selected',golden_islands(selected=0.),{'Qop':27.,'arrays':{'shed':[[0.,4.,5.]],'p':[[0.]]}})

    joint=[]
    for name,load,want_x,want_total in [('joint_MILP_useful',15.,1.,27.85),('joint_MILP_not_useful',1.,0.,3.)]:
        f=make_fixture(load_p=[[load]],tan_phi=.6,reserve=15.);f.tasks.p_block_kw[:]=15.
        f.tasks.qualification_soc=np.maximum(.3,f.tasks.soc_departure_min+15/(f.tasks.eta_dis*f.tasks.e_cap_kwh))
        plan,solution,certificate,first_check,physical,template,joint_problem=_joint_milp(f,lp_time_limit)
        if abs(plan.x[0]-want_x)>1e-8 or abs(certificate['objective_usd']-want_total)>1e-7:
            raise ValueError('Joint activation/payment golden result differs')
        f.plan=plan
        persist(name,f,solution,{'selected':want_x,'total_usd':want_total,'arrays':{'p':[[15. if want_x else 0.]]}},
                solver=certificate,template=template,physical=physical,first_check=first_check)
        from scipy.sparse import save_npz
        save_npz(out/name/'joint_matrix.npz',joint_problem['A'])
        np.savez_compressed(out/name/'joint_arrays.npz',**{k:v for k,v in joint_problem.items() if k!='A'})
        np.savez_compressed(out/name/'joint_primal.npz',vector=np.concatenate((plan.x,plan.r.ravel(),template.pack(solution))))
        joint.append({'case_id':name,'selected':float(plan.x[0]),'Fpre':first_check['costs']['Fpre_usd'],
                      'Qop':physical['costs']['Qop'],'total_usd':certificate['objective_usd'],**certificate})

    # Compare production helper coefficients with separately evaluated angles.
    from evcs_experiments.reactive_support_model import capability_facets
    actual_a,actual_b=capability_facets(16)
    reference_a=np.column_stack((np.cos(angles),np.sin(angles)));reference_b=np.full(16,np.cos(np.pi/16))
    facet_check={'passed':bool(np.allclose(actual_a,reference_a,rtol=0,atol=1e-14) and np.allclose(actual_b,reference_b,rtol=0,atol=1e-14)),
                 'max_normal_error':float(np.max(np.abs(actual_a-reference_a))),
                 'max_rhs_error':float(np.max(np.abs(actual_b-reference_b))),
                 'q_limit_at_rav15_kvar':q_limit,'active_coordinate':'rav'}
    if not facet_check['passed']: raise ValueError('Independent facet reference differs')
    _json_file(out/'facet_reference_check.json',facet_check)
    _json_file(out/'joint_milp_comparison.json',joint)
    _csv_file(out/'objective_components.csv',objectives)
    _csv_file(out/'variable_constraint_map.csv',maps)
    report={'status':'PASS_TOY_EXPECTATIONS_AUTOMATED','passed':True,'full_S02_pass':False,'fixture_count':len(records),
            'cases':records,'joint_activation_MILP':joint,'facet_reference':facet_check,
            'formal_scenarios_generated':0,'full_case_claim':False,'wall_seconds':time.monotonic()-start,
            'source_provenance':{'this_module_sha256':_sha_file(__file__),
                'fixture_constructor_origin':'explicit copy of tests/recourse_fixtures.py, 2026-09-05; no test import at runtime'},
            'independent_review_required':True}
    _json_file(out/'physical_check_report.json',report)
    _json_file(out/'artifact_sha256.json',{str(p.relative_to(out)):_sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return report


def read_fixture_inputs(path):
    """Restore saved raw toy inputs without assembling or solving any model.

    Independent auditors may also decode fixture_inputs.json themselves; this
    helper only restores tuple/ndarray representations from explicit JSON data.
    """
    data=json.loads(Path(path).read_text())
    tuple_fields={'periods','bus_ids','line_ids','fixed_open_ties','station_ids',
                  'station_bus_ids','line_from_bus_ids','line_to_bus_ids','task_ids','vehicle_ids'}
    result={}
    for kind in ('case','tasks','scenario'):
        values={}
        for key,value in data[kind].items():
            if isinstance(value,list):
                value=tuple(value) if key in tuple_fields else np.asarray(value)
            if key.endswith('_task_indices') and isinstance(value,dict):
                value={(int(k) if key=='bus_task_indices' else k):tuple(v) for k,v in value.items()}
            values[key]=value
        result[kind]=SimpleNamespace(**values)
    p=data['plan']
    result['plan']=FirstStagePlan(p['x'],p['r'],p['task_sha256'],p['config_hash'])
    return SimpleNamespace(**result)
