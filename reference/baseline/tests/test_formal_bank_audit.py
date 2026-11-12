"""Small numerical and tamper tests; never build a formal bank or read OOS."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.moment_calibration import bootstrap_moments,calibrate_moments
from evcs_experiments.wind_scenarios import draw_parameters

spec=importlib.util.spec_from_file_location('formal_bank_independent_audit',get_paths().project_root/'scripts/audits/audit_formal_bank.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FormalBankAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration=np.tile([.6,.2,.3,.4,1.],(8,1))
        boot=bootstrap_moments(cls.calibration,development=False)
        cls.boot=dict(indices=boot.indices,means=boot.bootstrap_means,center=boot.center,halfwidth=boot.halfwidth)
        cls.planning=np.tile([.6,.2,.3,.4,1.],(3,1))
        cls.moment=calibrate_moments(cls.planning,boot).as_dict();cls.moment['bootstrap']=boot.as_dict()
        cls.repair_planning=cls.planning.copy();cls.repair_planning[:,0]=.1
        cls.repair_moment=calibrate_moments(cls.repair_planning,boot).as_dict();cls.repair_moment['bootstrap']=boot.as_dict()

    def test_minimal_zero_repair_and_all_rhos(self):
        result=audit.audit_moments(self.planning,self.calibration,self.moment,self.boot)
        self.assertEqual(result['delta'],0.);self.assertEqual(result['certificate_count'],5)
        self.assertEqual(result['solver_calls'],0);self.assertEqual(result['maximum_duality_gap'],0.)

    def test_positive_repair_primal_and_dual_certify_minimum(self):
        result=audit.audit_moments(self.repair_planning,self.calibration,self.repair_moment,self.boot)
        self.assertAlmostEqual(result['delta'],1.);self.assertEqual(result['certificate_count'],5)
        bad=deepcopy(self.repair_moment);bad['repair_certificate']['dual_certificate']['inequality_marginals'][0]=9.
        with self.assertRaises(AssertionError):audit.audit_moments(self.repair_planning,self.calibration,bad,self.boot)

    def test_fake_pass_cannot_hide_changed_probability_or_rho(self):
        bad=deepcopy(self.moment);bad['bounds_by_rho']['2.0']['feasibility']['witness']=[0.,0.,0.]
        with self.assertRaisesRegex(AssertionError,'primal'):audit.audit_moments(self.planning,self.calibration,bad,self.boot)
        bad=deepcopy(self.moment);bad['bounds_by_rho']['1.0']['lower'][0]-=.01
        with self.assertRaisesRegex(AssertionError,'Final lower'):audit.audit_moments(self.planning,self.calibration,bad,self.boot)

    def test_bootstrap_wrong_formal_stream_and_changed_indices_rejected(self):
        bad=deepcopy(self.moment);bad['bootstrap']['stream_identity']['development']=True
        with self.assertRaisesRegex(AssertionError,'namespace'):audit.audit_moments(self.planning,self.calibration,bad,self.boot)
        wrong={k:np.array(v,copy=True) for k,v in self.boot.items()};wrong['indices'][0,0]=(wrong['indices'][0,0]+1)%8
        with self.assertRaisesRegex(AssertionError,'address'):audit.audit_moments(self.planning,self.calibration,self.moment,wrong)

    def test_bootstrap_means_recomputed_despite_rehashed_payload(self):
        wrong={k:np.array(v,copy=True) for k,v in self.boot.items()};wrong['means'][0,0]+=.001
        bad=deepcopy(self.moment);bad['bootstrap']['bootstrap_means_sha256']=audit.bh(wrong['means'])
        with self.assertRaisesRegex(AssertionError,'bootstrap means'):audit.audit_moments(self.planning,self.calibration,bad,wrong)

    def test_scalar_route_departure_delay_and_failed_accounting(self):
        task=dict(task_id='one',vehicle_id='v',arc_ids=[1],distance_km=1.,freeflow_h=.25,start_period=1)
        states={'v':SimpleNamespace(road_node=0,soc=.8)}
        hardware={'v':dict(E_cap_e_kWh=80.,E_km_e_kWh_per_km=.3,P_aux_e_kW=1.,SOC_dep_under_e=.2,P_blk_under_e_kW=1.,eta_dis_e=.95)}
        arcs=[dict(a=1,m_road=0,n_road=1,D_a_km=1.,T0_a_min=15.)]
        travel=np.full((100,1),75.)
        arrays,statuses,count=audit.scalar_task_arrays([task],states,hardware,arcs,travel)
        self.assertEqual(count,1);self.assertEqual(statuses,['TASK_QUALIFIED']);self.assertAlmostEqual(arrays['arrival_h'][0],.25)
        self.assertAlmostEqual(arrays['travel_energy_kwh'][0],1.3);self.assertAlmostEqual(arrays['arrival_soc'][0],.8-1.3/80)
        self.assertFalse(arrays['mu_full'][0,0]);self.assertTrue(arrays['mu_full'][0,1:].all())
        travel[0,0]=np.inf
        arrays,statuses,count=audit.scalar_task_arrays([task],states,hardware,arcs,travel)
        self.assertEqual(statuses,['ENTRY_IMPASSABLE']);self.assertTrue(np.isinf(arrays['arrival_h'][0]))
        self.assertEqual(arrays['failed_arc_ids'][0],1)
        for name in ('route_time_h','travel_energy_kwh','arrival_soc','usable_energy_dc_kwh'):self.assertEqual(arrays[name][0],0.)
        self.assertFalse(arrays['mu_full'].any())

    def test_feature_equal_vehicle_weight_and_connector_exclusion(self):
        tasks=[dict(vehicle_id='a',usable_energy_kwh=10.),dict(vehicle_id='a',usable_energy_kwh=10.),dict(vehicle_id='b',usable_energy_kwh=10.)]
        roads=np.zeros((100,88),bool);roads[:,80:]=True
        arrays=dict(grid_hazard=np.full((100,441),.4),line_outages=np.zeros((100,32),bool),road_blocked=roads,
            task_usable_energy_dc_kwh=np.array([0.,0.,10.]),load_p_kw=np.full((24,33),1.05))
        features=audit.feature_values(arrays,tasks,np.ones((24,33)))
        np.testing.assert_allclose(features,[.4,0.,0.,.5,1.05],atol=1e-14,rtol=0)

    def test_formal_stream_reconstruction_and_anchors(self):
        events=tuple('event'+str(i) for i in range(8))
        for split in ('planning','calibration'):
            actual=audit.wind_parameters(events,events[0],events[2],3,split)
            expected=draw_parameters(events,events[0],events[2],3,split=split,development=False).as_dict()
            self.assertEqual(audit.canonical(actual),audit.canonical(expected));self.assertNotEqual(actual['donor_event'],events[0])
        anchors=audit.anchor_ids(events);self.assertEqual(len(anchors),16)
        self.assertEqual({p.split('/')[0] for p in anchors},set(events));self.assertEqual({p.split('/')[3] for p in anchors},set(events))
        with self.assertRaisesRegex(AssertionError,'OOS'):audit.stream('oos',0,0,7)

    def test_raw_physical_chain_matches_accepted_components_and_detects_tamper(self):
        from evcs_experiments.line_outages import load_line_model
        from evcs_experiments.road_disruptions import load_road_model
        from evcs_experiments.load_scenarios import load_load_model,draw_load_errors,realize_load
        root=get_paths().data_root;lines=load_line_model(root);roads=load_road_model(root,include_connectors=True);load=load_load_model(root)
        events=tuple('event'+str(i) for i in range(8));split='planning';base=events[2];v=0
        lt=lines.draw_thresholds(events,base,v,split=split,development=False)
        rt=roads.draw_thresholds(events,base,v,split=split,development=False)
        ld=draw_load_errors(load,2,v,split=split,development=False);realized=realize_load(load,ld)
        lw=np.zeros((100,32,2));lw[:,:,0]=np.linspace(5.,35.,100)[:,None]
        rw=np.zeros((100,88,2));rw[:,:,0]=np.linspace(5.,35.,100)[:,None]
        line=lines.replay(audit.TIMES,lw,lt,lead_h=1.);road=roads.replay(audit.TIMES,rw,rt,line,lead_h=1.,regime='R1')
        arrays=dict(line_wind_vectors_m_s=lw,road_wind_vectors_m_s=rw,line_outages=line.outages,
            line_operating_outages=line.operating_outages,road_blocked=road.blocked,road_movable=road.movable,
            road_travel_time_min=road.travel_time_min,line_standard_normals=lt.standard_normals,
            line_threshold_uniforms=lt.uniforms,road_corridor_uniforms=rt.corridor_uniforms,
            load_standard_innovation=ld.accepted_standard_innovation,load_epsilon=ld.epsilon,
            load_p_kw=realized.p_kw,load_q_kvar=realized.q_kvar,load_archived_q_kvar=realized.archived_q_kvar)
        ledger=dict(split=split,base_weather_event=base,variant_index=v,line_stream_id=lt.stream_id,
            road_stream_ids=list(rt.stream_ids),load_draw=ld.as_dict())
        audit.physical_arrays(arrays,ledger,lines,roads,load,events)
        bad={k:np.array(value,copy=True) for k,value in arrays.items()};bad['road_blocked'][0,0]=~bad['road_blocked'][0,0]
        with self.assertRaisesRegex(AssertionError,'road blockage'):audit.physical_arrays(bad,ledger,lines,roads,load,events)
        bad={k:np.array(value,copy=True) for k,value in arrays.items()};bad['load_p_kw'][0,0]+=.1
        with self.assertRaisesRegex(AssertionError,'load_p_kw'):audit.physical_arrays(bad,ledger,lines,roads,load,events)

    def test_real_88_arc_network_hash_binds_geometry_travel_and_source_context(self):
        from evcs_experiments.forecast_routes import ForecastNetwork
        from evcs_experiments.validation.validate_s01_replay import network_arcs
        arcs=network_arcs(get_paths().data_root)
        rows=[dict(a=a.a,m_road=a.source,n_road=a.target,D_a_km=a.distance_km,T0_a_min=a.freeflow_h*60.) for a in arcs]
        minutes=np.tile([float(r['T0_a_min']) for r in rows],(100,1));minutes[8,3]=np.inf
        identity=dict(scenario_id='hand-physical-network',split='planning',development=False)
        expected=ForecastNetwork(arcs,audit.TIMES,minutes/60.,network_hash=audit.canonical(identity))
        digest=audit.network_content_hash(rows,minutes,identity)
        self.assertEqual(digest,expected.network_hash);self.assertNotEqual(digest,audit.canonical(identity))
        changed=minutes.copy();changed[0,0]+=.01
        self.assertNotEqual(digest,audit.network_content_hash(rows,changed,identity))
        changed_rows=deepcopy(rows);changed_rows[0]['D_a_km']+=.01
        self.assertNotEqual(digest,audit.network_content_hash(changed_rows,minutes,identity))
        self.assertNotEqual(digest,audit.network_content_hash(rows,minutes,{**identity,'scenario_id':'other'}))
        for invalid in (np.nan,-np.inf):
            changed=minutes.copy();changed[0,0]=invalid
            with self.assertRaisesRegex(AssertionError,'invalid time'):audit.network_content_hash(rows,changed,identity)

    def test_real_88_arc_replay_semantic_hash_and_array_status_tamper(self):
        from evcs_experiments.forecast_routes import ForecastNetwork
        from evcs_experiments.forecast_tasks import service_periods
        from evcs_experiments.task_replay import prepare_task_bank,replay_prepared
        from evcs_experiments.validation.validate_s01_replay import network_arcs
        arcs=network_arcs(get_paths().data_root)
        rows=[dict(a=a.a,m_road=a.source,n_road=a.target,D_a_km=a.distance_km,T0_a_min=a.freeflow_h*60.) for a in arcs]
        first=arcs[0];state=SimpleNamespace(road_node=first.source,soc=.8)
        hw=dict(E_cap_e_kWh=80.,E_km_e_kWh_per_km=.3,P_aux_e_kW=1.,SOC_dep_under_e=.2,P_blk_under_e_kW=1.,eta_dis_e=.95)
        occupancy,reserve,block,_=service_periods(1,1.)
        task=dict(task_id='v__s__k01',vehicle_id='v',station_id='s',start_period=1,arc_ids=[first.a],
            distance_km=first.distance_km,freeflow_h=first.freeflow_h,arrival_h=-1.+first.freeflow_h,
            arrival_soc=.8-.3*first.distance_km/80.,qualification_soc=.3,
            occupancy=list(occupancy),reserve=list(reserve),block_periods=list(block))
        bank=dict(status='COMPLETE',certified=True,dependency_sha256='hand-fixture',config=dict(basic_soc=.3,tau_blk_h=1.),tasks=[task],
            queries=[dict(vehicle_id='v',station_id='s',status='COMPLETE',route=dict(arc_ids=[first.a]),arguments=dict(
                origin=first.source,destination=first.target,departure_h=-1.,initial_soc=.8,capacity_kwh=80.,
                energy_kwh_per_km=.3,auxiliary_kw=1.,qualification_soc=.3))])
        bank['task_sha256']=audit.canonical({k:bank[k] for k in ('dependency_sha256','config','certified','tasks')})
        prepared=prepare_task_bank(bank,{'v':hw},{'v':state})
        identity=dict(scenario_id='hand-physical-replay',split='planning',development=False)
        minutes=np.tile([r['T0_a_min'] for r in rows],(100,1));minutes[8,3]=np.inf
        network=ForecastNetwork(arcs,audit.TIMES,minutes/60.,network_hash=audit.canonical(identity))
        result=replay_prepared(network,prepared,split='planning',development=False,scenario_identity=identity)
        independent,statuses,_=audit.scalar_task_arrays([task],{'v':state},{'v':hw},rows,minutes)
        arrays={'task_'+k:getattr(result,k) for k in audit.TASK_FIELDS};arrays['road_travel_time_min']=minutes
        for name,value in independent.items():np.testing.assert_array_equal(value,arrays['task_'+name])
        self.assertEqual(statuses,list(result.status))
        self.assertEqual(audit.replay_semantic_hash(prepared,bank,identity,arrays,statuses,rows),result.replay_sha256)
        self.assertNotEqual(audit.replay_semantic_hash(prepared,bank,identity,arrays,['SOC_UNQUALIFIED'],rows),result.replay_sha256)
        changed={k:np.array(v,copy=True) for k,v in arrays.items()};changed['task_arrival_h'][0]+=.01
        self.assertNotEqual(audit.replay_semantic_hash(prepared,bank,identity,changed,statuses,rows),result.replay_sha256)

    def test_diagnostic_limit_cannot_issue_success_exit_or_bank_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for limit in (0,17,True,1.5):
                with self.assertRaisesRegex(AssertionError,'never a bank acceptance'):
                    audit.audit(root/'bank',root,root,root/'audit',expected_summary_sha256='0'*64,
                        expected_source_manifest_sha256='0'*64,diagnostic_atom_limit=limit)
        args=['audit','--bank-root','bank','--case-root','case','--source-root','source','--audit-output','out',
            '--expected-summary-sha256','0'*64,'--expected-source-manifest-sha256','0'*64,'--diagnostic-atom-limit','1']
        with patch.object(audit.sys,'argv',args),patch.object(audit,'audit',return_value=dict(status='DIAGNOSTIC_ONLY_NOT_BANK_PASS')) as call:
            self.assertEqual(audit.main(),2);self.assertEqual(call.call_args.kwargs['diagnostic_atom_limit'],1)

    def test_external_summary_binding_fails_before_scientific_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bank=root/'bank';bank.mkdir();(bank/'summary.json').write_text('{}')
            with self.assertRaisesRegex(AssertionError,'Externally bound summary'):
                audit.audit(bank,root,root,root/'audit',expected_summary_sha256='0'*64,expected_source_manifest_sha256='0'*64)
            report=json.loads((root/'audit/audit.json').read_text());self.assertEqual(report['status'],'FAIL_INDEPENDENT_FORMAL_BANK_AUDIT')
            self.assertFalse(report['full_S05_pass']);self.assertEqual(report['checked_atom_count'],0)

if __name__=='__main__':unittest.main()
