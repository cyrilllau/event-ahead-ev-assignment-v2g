"""Independent lead0 physical/H checks without a full bank or OOS."""
from copy import deepcopy
import importlib.util
import unittest
import json
from types import SimpleNamespace
import numpy as np
from evcs_experiments.config import get_paths

spec=importlib.util.spec_from_file_location('onset_bank_independent_audit',get_paths().project_root/'scripts/audits/audit_formal_bank.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class OnsetAuditTests(unittest.TestCase):
    def test_onset_scalar_departure_and_feature_denominator(self):
        task=dict(vehicle_id='v',arc_ids=[1],distance_km=1.,freeflow_h=.25,start_period=1,usable_energy_kwh=10.)
        hw=dict(E_cap_e_kWh=80.,E_km_e_kWh_per_km=.3,P_aux_e_kW=1.,SOC_dep_under_e=.2,P_blk_under_e_kW=1.,eta_dis_e=.95)
        arcs=[dict(a=1,m_road=0,n_road=1,D_a_km=1.,T0_a_min=15.)]
        times=np.arange(96)/4
        fields,status,count=audit.scalar_task_arrays([task],{'v':SimpleNamespace(road_node=0,soc=.8)},{'v':hw},arcs,np.full((96,1),75.),times=times,departure_h=0.)
        self.assertEqual(fields['arrival_h'][0],1.25);self.assertAlmostEqual(fields['travel_energy_kwh'][0],1.3)
        arrays=dict(grid_hazard=np.zeros((96,441)),line_outages=np.zeros((96,32)),road_blocked=np.ones((96,88)),task_usable_energy_dc_kwh=np.array([10.]),load_p_kw=np.ones((24,33)))
        self.assertEqual(audit.feature_values(arrays,[task],np.ones((24,33)))[2],1.)

    def test_real_onset_H_independent_oracle_and_rehashed_omission(self):
        from evcs_experiments.forecast_tasks import load_forecast_task_inputs,build_candidate_tasks
        inputs=load_forecast_task_inputs(get_paths().data_root,'michael_2018',lead_h=0.)
        bank=json.loads(json.dumps(build_candidate_tasks(inputs,overall_time_limit_seconds=20.).as_dict()))
        result=audit.audit_onset_task_contract(bank,get_paths().data_root,inputs.case,inputs.target,lambda:None)
        self.assertEqual(result['queries'],640)
        bad=deepcopy(bank);bad['tasks'].pop();bad['task_sha256']=audit.canonical({k:bad[k] for k in ('dependency_sha256','config','certified','tasks')})
        with self.assertRaises(AssertionError):audit.audit_onset_task_contract(bad,get_paths().data_root,inputs.case,inputs.target,lambda:None)
        bad=deepcopy(bank);bad['dependency_sha256']='wrong'
        with self.assertRaisesRegex(AssertionError,'forecast dependency'):audit.audit_onset_task_contract(bad,get_paths().data_root,inputs.case,inputs.target,lambda:None)

    def test_real_onset_atom_full_chain_and_replay_hash(self):
        from evcs_experiments.onset_planning import OnsetPlanningContext
        from evcs_experiments.forecast_tasks import load_forecast_task_inputs,build_candidate_tasks
        inputs=load_forecast_task_inputs(get_paths().data_root,'michael_2018',lead_h=0.)
        bank=json.loads(json.dumps(build_candidate_tasks(inputs,overall_time_limit_seconds=20.).as_dict()))
        ctx=OnsetPlanningContext(get_paths().data_root,inputs.network.arcs)
        prepared=ctx.prepare_target('michael_2018',bank);atom=ctx.generate(prepared,ctx.events[1],0,'planning')
        arrays=dict(atom.arrays);ledger=dict(atom.ledger);times=np.arange(96)/4
        audit.physical_arrays(arrays,ledger,ctx.lines,ctx.roads,ctx.load,ctx.events,lead_h=0.)
        rows=[dict(a=a.a,m_road=a.source,n_road=a.target,D_a_km=a.distance_km,T0_a_min=a.freeflow_h*60.) for a in inputs.network.arcs]
        fields,status,_=audit.scalar_task_arrays(bank['tasks'],{x.vehicle_id:x for x in inputs.target.vehicle_states},inputs.case.vehicle_parameters,rows,arrays['road_travel_time_min'],times=times,departure_h=0.)
        for name,value in fields.items():audit.close(arrays['task_'+name],value,name,1e-12)
        identity={k:ledger[k] for k in ('scenario_id','target_event','base_weather_event','variant','variant_index','split','development','regime','lead_h','residual_donor_event')}
        self.assertEqual(audit.replay_semantic_hash(prepared.tasks,bank,identity,arrays,status,rows,times=times,departure_h=prepared.tasks.departure_h),ledger['task_replay_sha256'])
        audit.close(arrays['features'],audit.feature_values(arrays,bank['tasks'],ctx.load.baseline_p_kw),'onset features',1e-12)
        bad={k:np.array(v,copy=True) for k,v in arrays.items()};bad['line_operating_outages'][0,0]^=True
        with self.assertRaisesRegex(AssertionError,'Hourly line mapping'):audit.physical_arrays(bad,ledger,ctx.lines,ctx.roads,ctx.load,ctx.events,lead_h=0.)

    def test_oracle_honors_parent_deadline(self):
        q=dict(origin=0,destination=0,departure_h=0.,initial_soc=.8,capacity_kwh=80.,energy_kwh_per_km=.3,auxiliary_kw=1.,qualification_soc=.3,latest_arrival_h=23.)
        def stop():raise TimeoutError('parent budget')
        with self.assertRaises(TimeoutError):audit.onset_route_oracle([],np.arange(96)/4,np.empty((96,0)),q,stop)

if __name__=='__main__':unittest.main()
