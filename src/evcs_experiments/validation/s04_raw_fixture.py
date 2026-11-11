"""Explicit one-vehicle development inputs, replayed by the production route kernel."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evcs_experiments.forecast_routes import Arc,ForecastNetwork
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.task_replay import replay_task_bank
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.validation.s02_toys import make_fixture


def raw_fixture():
    config=dict(basic_soc=.3,tau_blk_h=1.,nearest_stations=1)
    task=dict(task_id='EV1__CS1__k01',vehicle_id='EV1',station_id='CS1',start_period=1,
              arc_ids=[1],arrival_h=-.75,distance_km=1.,freeflow_h=.25,travel_time_h=.25,
              travel_energy_kwh=.3,arrival_soc=.8-.3/80,usable_energy_kwh=(.8-.3/80-.2)*80,
              qualification_soc=.3,occupancy=[True]*24,reserve=[True]*24,block_periods=[1])
    dependency=canonical_hash({'development_raw_geometry':[[1,0,1,1.,.25]],'initial_state':{'road_node':0,'soc':.8}})
    bank=dict(status='COMPLETE',certified=True,tasks=[task],config=config,dependency_sha256=dependency,
              task_sha256=canonical_hash(dict(dependency_sha256=dependency,config=config,certified=True,tasks=[task])),
              queries=[dict(vehicle_id='EV1',station_id='CS1',status='COMPLETE',arguments=dict(
                  origin=0,destination=1,departure_h=-1.,initial_soc=.8,capacity_kwh=80.,energy_kwh_per_km=.3,auxiliary_kw=1.,qualification_soc=.3),
                  route=dict(arc_ids=[1]))])
    hardware={'EV1':dict(E_cap_e_kWh=80.,E_km_e_kWh_per_km=.3,P_aux_e_kW=1.,SOC_dep_under_e=.2,P_blk_under_e_kW=1.,eta_dis_e=.95)}
    return dict(scope='development_hand_fixture',target_id='synthetic_fixture',bank=bank,hardware=hardware,
                states={'EV1':dict(road_node=0,soc=.8)},arcs=[[1,0,1,1.,.25]],arc_to_corridor={'1':'physical_1'},
                load_p_kw=[[.5]]*24,tan_phi=.5,
                samples={f'synthetic_fixture/development/oos/b{b}/v{v+1:03d}/R1':dict(base_id=f'b{b}',variant_id=f'v{v+1:03d}',travel_h=.25+b+v*.5) for b in range(2) for v in range(2)})


def frozen_fixture(raw):
    f=make_fixture(periods=24,load_p=raw['load_p_kw'],tan_phi=raw['tan_phi'],reserve=1,eta=.95)
    bank=raw['bank'];t=f.tasks;t.task_sha256=bank['task_sha256'];t.config_hash=canonical_hash(bank['config']);t.config=dict(bank['config'])
    t.forecast_arrival_h=np.array([bank['tasks'][0]['arrival_h']]);t.forecast_arrival_soc=np.array([bank['tasks'][0]['arrival_soc']]);t.e_forecast_dc_kwh=np.array([bank['tasks'][0]['usable_energy_kwh']]);t.qualification_soc=np.array([.3])
    f.case.critical_buses[:]=True
    f.plan=FirstStagePlan(f.plan.x,f.plan.r,t.task_sha256,t.config_hash)
    return f


def physical_fixture(raw,sample_id,departure_h):
    """Raw geometry, delays, vehicle state and load → true fixed-route replay."""
    if raw.get('scope')!='development_hand_fixture':raise ValueError('This constructor is development-only')
    sample=raw['samples'][sample_id];f=frozen_fixture(raw)
    times=np.arange(-1.,24.,.25);arcs=tuple(Arc(*v) for v in raw['arcs'])
    net=ForecastNetwork(arcs,times,np.full((len(times),len(arcs)),sample['travel_h']))
    states={k:SimpleNamespace(**v) for k,v in raw['states'].items()}
    replay=replay_task_bank(net,raw['bank'],raw['hardware'],states,departure_h,split='oos',development=True,scenario_identity=dict(scenario_id=sample_id))
    from evcs_experiments.evaluation_physical import scenario_from_replay
    identity=dict(scenario_id=sample_id,split='oos',development=True,target_event=raw['target_id'],base_weather_event=sample['base_id'],variant=int(sample['variant_id'][1:]),regime='R1')
    f.scenario=scenario_from_replay(f.case,f.tasks,replay,identity=identity,line_operating_outages=f.scenario.line_operating_outages,load_p_kw=f.scenario.load_p_kw,times_h=times,source_hashes={'raw_fixture':canonical_hash(raw)})
    return f,replay


def locked_factory(record,sample_id,meta):
    raw=json.loads(Path(record['references']['tasks']['path']).read_text())
    f,replay=physical_fixture(raw,sample_id,meta['departure_h'])
    return f.case,f.tasks,f.scenario
