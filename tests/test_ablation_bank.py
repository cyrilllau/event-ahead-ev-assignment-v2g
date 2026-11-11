"""Complete development recombination bank with independently checked features.

Source arrays use canonical axes and an actual validated context/fixed H, but
are explicitly synthetic fixtures. No formal scenario acceptance or OOS use.
"""
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
import unittest
from unittest.mock import patch
import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.joint_scenarios import JointContext,JointScenario,array_hash
from evcs_experiments.forecast_tasks import qualification_soc,service_periods
from evcs_experiments.random_streams import DEVELOPMENT_ROOT,Stream,canonical_hash
from evcs_experiments.scenario_features import compute_features
from evcs_experiments.physical_ablations import BLOCK_FIELDS
from evcs_experiments.validation.validate_s01_replay import network_arcs


def fixed_unequal_task_bank(context,event):
    target=context.case.target(event,lead_h=1.)
    tasks=[];queries=[];qualified=[]
    for state in target.vehicle_states:
        hw={k:float(context.case.vehicle_parameters[state.vehicle_id][k]) for k in
            ('SOC_dep_under_e','P_blk_under_e_kW','eta_dis_e','E_cap_e_kWh','E_km_e_kWh_per_km','P_aux_e_kW')}
        qual=qualification_soc(.3,hw['SOC_dep_under_e'],hw['P_blk_under_e_kW'],1.,hw['eta_dis_e'],hw['E_cap_e_kWh'])
        if state.soc>=qual:qualified.append((state,hw,qual))
        if len(qualified)==2:break
    assert len(qualified)==2
    for number,(state,hw,qual) in enumerate(qualified):
        station='HAND_ZERO_ROUTE'
        queries.append(dict(vehicle_id=state.vehicle_id,station_id=station,status='COMPLETE',
            arguments=dict(origin=state.road_node,destination=state.road_node,departure_h=-1.,initial_soc=state.soc,
                capacity_kwh=hw['E_cap_e_kWh'],energy_kwh_per_km=hw['E_km_e_kWh_per_km'],auxiliary_kw=hw['P_aux_e_kW'],qualification_soc=qual),
            route=dict(arc_ids=[])))
        for k in ((1,2,3) if number==0 else (1,)):
            occupancy,reserve,block,_=service_periods(k,1.)
            tasks.append(dict(task_id=f'{state.vehicle_id}__{station}__k{k:02d}',vehicle_id=state.vehicle_id,station_id=station,
                start_period=k,arc_ids=[],arrival_h=-1.,distance_km=0.,freeflow_h=0.,travel_time_h=0.,travel_energy_kwh=0.,
                arrival_soc=state.soc,usable_energy_kwh=(state.soc-hw['SOC_dep_under_e'])*hw['E_cap_e_kWh'],
                qualification_soc=qual,occupancy=list(occupancy),reserve=list(reserve),block_periods=list(block)))
    bank=dict(status='COMPLETE',certified=True,dependency_sha256=canonical_hash({'fixture':'unequal H canonical axes','state':target.state_sha256}),
        config=dict(basic_soc=.3,tau_blk_h=1.,nearest_stations=3),tasks=tasks,queries=queries,vehicle_summaries=[])
    bank['task_sha256']=canonical_hash({k:bank[k] for k in ('dependency_sha256','config','certified','tasks')})
    return bank


def feature(context,prepared,arrays,identity):
    return compute_features(arrays['grid_hazard'],arrays['line_outages'],arrays['road_blocked'],
        arrays['task_usable_energy_dc_kwh'],arrays['load_p_kw'],context.load.baseline_p_kw,
        prepared_tasks=prepared.feature_tasks,times_h=prepared.target.replay_times_h,
        task_ids=prepared.tasks.task_ids,road_ids=context.roads.arc_ids,line_ids=context.lines.line_ids,
        grid_ids=context.grid.point_ids,bus_ids=context.load.bus_ids,grid_sha256=context.grid.grid_sha256,
        scenario_identity=identity)


def synthetic_sample(context,prepared,template,split,index):
    arrays={k:np.array(v,copy=True) for k,v in template.arrays.items()}
    arrays['grid_hazard'][:]=.1+.05*index
    arrays['line_wind_vectors_m_s'][:]=index+1.
    arrays['road_wind_vectors_m_s'][:]=2.*index+1.
    arrays['line_outages'][:]=False;arrays['line_outages'][40+index:,:4+index]=True
    arrays['line_operating_outages']=arrays['line_outages'][4:].reshape(24,4,32).max(axis=1)
    arrays['line_standard_normals'][:]=index/10.
    arrays['line_threshold_uniforms'][:]=.2+.03*index
    arrays['road_blocked'][:]=False;arrays['road_blocked'][:,:20+index]=True;arrays['road_blocked'][:,80:]=True
    arrays['road_movable']=~arrays['road_blocked']
    arrays['road_travel_time_min']=np.where(arrays['road_movable'],10.+index,np.inf)
    arrays['road_corridor_uniforms'][:]=.1+.04*index
    energy_fraction=np.array([1.,1.,1.,index/10.]) if prepared.tasks.task_ids else np.empty(0)
    arrays['task_usable_energy_dc_kwh']=prepared.feature_tasks.forecast_usable_dc_kwh*energy_fraction
    arrays['task_arrival_h'][:]=index/10.
    arrays['task_route_time_h'][:]=1.+index/10.
    arrays['task_travel_energy_kwh'][:]=2.+index
    arrays['task_arrival_soc'][:]=.5+index/100.
    arrays['task_gamma_route'][:]=True;arrays['task_varsigma'][:]=True
    arrays['task_mu_full'][:]=True;arrays['task_failed_arc_ids'][:]=-1
    factor=.96+.01*index+(0.002 if split=='calibration' else 0.)
    arrays['load_p_kw']=context.load.baseline_p_kw*factor
    arrays['load_q_kvar']=np.array(template.arrays['load_q_kvar'])*factor
    arrays['load_archived_q_kvar']=np.array(template.arrays['load_archived_q_kvar'])*factor
    arrays['load_epsilon'][:]=factor-1.;arrays['load_standard_innovation'][:]=index/10.
    ledger=dict(template.ledger);base=context.events[index]
    identity=dict(scenario_id=f'{prepared.target.event_id}/development/{split}/{base}/v001/R1',
        target_event=prepared.target.event_id,base_weather_event=base,variant=1,variant_index=0,split=split,
        development=True,regime='R1',lead_h=1.,residual_donor_event=ledger['residual_donor_event'])
    ledger.update(identity)
    f=feature(context,prepared,arrays,identity);arrays['features']=np.array(f.values)
    ledger['feature_sha256']=f.feature_sha256
    ledger['array_hashes']={k:array_hash(v) for k,v in arrays.items()}
    ledger.pop('scenario_sha256',None);sha=canonical_hash(ledger);ledger['scenario_sha256']=sha
    for array in arrays.values():array.setflags(write=False)
    return JointScenario(MappingProxyType(arrays),MappingProxyType(ledger),
        MappingProxyType({'explicit_synthetic_canonical_fixture':True,'feature_diagnostics':dict(f.diagnostics)}),sha)


class AblationBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=get_paths().data_root;cls.context=JointContext(root,network_arcs(root))
        cls.bank=fixed_unequal_task_bank(cls.context,cls.context.events[0])
        cls.prepared=cls.context.prepare_target(cls.context.events[0],cls.bank)
        cls.samples={}
        for split in ('planning','calibration'):
            template=cls.context.generate(cls.prepared,cls.context.events[0],0,split)
            cls.samples[split]=tuple(synthetic_sample(cls.context,cls.prepared,template,split,i) for i in range(8))
        cls.streams={split:{block:Stream(DEVELOPMENT_ROOT,(1,2,1 if split=='planning' else 2,0,0,0,711,i+1,0))
                    for i,block in enumerate(BLOCK_FIELDS)} for split in cls.samples}
        cls.indices={split:{block:stream.generator().permutation(8) for block,stream in ss.items()} for split,ss in cls.streams.items()}

    def arguments(self):
        return dict(planning_samples=self.samples['planning'],calibration_samples=self.samples['calibration'],
            expected_scenario_sha256={s:tuple(x.scenario_sha256 for x in samples) for s,samples in self.samples.items()},
            sample_indices=self.indices,streams=self.streams,domain='development',method_id='independent_recombination')

    def build(self,**changes):
        from evcs_experiments.ablation_bank import build_independent_recombination_bank
        args=self.arguments();args.update(changes)
        return build_independent_recombination_bank(self.context,self.prepared,**args)

    def test_complete_bank_preserves_whole_blocks_and_recomputes_five_features(self):
        before={s:[(x.scenario_sha256,{k:array_hash(v) for k,v in x.arrays.items()}) for x in samples] for s,samples in self.samples.items()}
        with patch('evcs_experiments.joint_scenarios.replay_prepared',side_effect=AssertionError('Recombination must not replay tasks')), \
             patch('evcs_experiments.road_disruptions.RoadModel.replay',side_effect=AssertionError('Recombination must not replay roads')):
            value=self.build()
        np.testing.assert_allclose(self.prepared.feature_tasks.task_weights,[1/6,1/6,1/6,1/2],rtol=0,atol=1e-15)
        for split,result in (('planning',value.planning),('calibration',value.calibration)):
            self.assertEqual(len(result),8)
            for i,scenario in enumerate(result):
                for block,fields in BLOCK_FIELDS.items():
                    donor=self.samples[split][self.indices[split][block][i]]
                    for field in fields:np.testing.assert_array_equal(scenario.arrays[field],donor.arrays[field])
                w=self.indices[split]['weather_grid'][i];r=self.indices[split]['road_vehicle'][i];load=self.indices[split]['load'][i]
                expected=np.array([.1+.05*w,(60-w)*(4+w)/(100*32),(20+r)/80.,.5*(1-r/10.),.96+.01*load+(0.002 if split=='calibration' else 0.)])
                np.testing.assert_allclose(scenario.arrays['features'],expected,rtol=0,atol=2e-14)
                self.assertNotAlmostEqual(expected[3],np.mean([0.,0.,0.,1-r/10.]))
                self.assertNotAlmostEqual(expected[2],(28+r)/88.)
                self.assertNotIn(scenario.scenario_sha256,{x.scenario_sha256 for x in self.samples[split]})
                self.assertEqual(dict(scenario.ledger)['scenario_sha256'],scenario.scenario_sha256)
                self.assertFalse(scenario.arrays['features'].flags.writeable)
                for block,columns in (('weather_grid',(0,1)),('road_vehicle',(2,3)),('load',(4,))):
                    donor=self.samples[split][self.indices[split][block][i]]
                    np.testing.assert_allclose(scenario.arrays['features'][list(columns)],donor.arrays['features'][list(columns)],rtol=0,atol=2e-14)
            np.testing.assert_allclose(getattr(value,split+'_features'),np.stack([x.arrays['features'] for x in result]),rtol=0,atol=0)
            baseline=np.stack([x.arrays['features'] for x in self.samples[split]])
            transformed=getattr(value,split+'_features')
            np.testing.assert_allclose(np.sort(transformed,axis=0),np.sort(baseline,axis=0),rtol=0,atol=2e-14)
            np.testing.assert_allclose(transformed.mean(axis=0),baseline.mean(axis=0),rtol=0,atol=2e-14)
            self.assertTrue(any(not np.array_equal(row,baseline[j]) for j,row in enumerate(transformed)))
        self.assertFalse(value.provenance['formal_bank_admitted'])
        self.assertFalse(value.provenance['formal_OOS'])
        self.assertFalse(value.provenance['scientific_acceptance'])
        self.assertTrue(value.provenance['all_five_features_recomputed'])
        self.assertEqual(before,{s:[(x.scenario_sha256,{k:array_hash(v) for k,v in x.arrays.items()}) for x in samples] for s,samples in self.samples.items()})

    def test_actual_whole_row_bootstrap_and_repaired_rho_half_and_one_witnesses(self):
        value=self.build()
        self.assertEqual(value.bootstrap.indices.shape,(2000,8))
        expected=np.array([value.calibration_features[row].mean(axis=0) for row in value.bootstrap.indices])
        np.testing.assert_allclose(value.bootstrap.bootstrap_means,expected,rtol=0,atol=1e-14)
        np.testing.assert_allclose(value.bootstrap.center,value.calibration_features.mean(axis=0),rtol=0,atol=1e-15)
        self.assertEqual(value.bootstrap.quantile_probability,.99)
        np.testing.assert_allclose(value.bootstrap.halfwidth,np.quantile(np.abs(expected-value.bootstrap.center),.99,axis=0,method='higher'),rtol=0,atol=1e-14)
        self.assertEqual(value.moments.status,'PASS_MOMENT_CALIBRATION')
        for rho in ('0.5','1.0'):
            bounds=value.moments.bounds_by_rho[rho];witness=np.array(bounds['feasibility']['witness'])
            self.assertTrue(bounds['feasibility']['valid_witness']);self.assertAlmostEqual(witness.sum(),1.,delta=1e-8)
            self.assertGreaterEqual(witness.min(),-1e-8)
            mean=witness@value.planning_features
            self.assertTrue(np.all(mean>=np.array(bounds['lower'])-1e-8));self.assertTrue(np.all(mean<=np.array(bounds['upper'])+1e-8))

    def test_method_split_order_donor_and_source_pins_cannot_be_relabelled(self):
        for change in (dict(domain='formal'),dict(domain='oos'),dict(method_id='no_line_dependence'),
                       dict(bootstrap_replications=64),dict(bootstrap_alpha=.1),
                       dict(planning_samples=tuple(reversed(self.samples['planning']))),
                       dict(calibration_samples=self.samples['planning'])):
            with self.subTest(change=list(change)),self.assertRaises(ValueError):self.build(**change)
        pins={s:list(rows) for s,rows in self.arguments()['expected_scenario_sha256'].items()};pins['planning'][0]='0'*64
        with self.assertRaises(ValueError):self.build(expected_scenario_sha256=pins)
        indices={s:{b:a.copy() for b,a in rows.items()} for s,rows in self.indices.items()};indices['planning']['load'][0]=indices['planning']['load'][1]
        with self.assertRaises(ValueError):self.build(sample_indices=indices)

    def test_stream_addresses_are_split_specific_and_determine_the_donors(self):
        indices={s:{b:a.copy() for b,a in rows.items()} for s,rows in self.indices.items()}
        indices['planning']['load']=np.roll(indices['planning']['load'],1)
        with self.assertRaises(ValueError):self.build(sample_indices=indices)
        for replacement in (self.streams['calibration']['load'],self.streams['planning']['weather_grid'],
                            Stream(DEVELOPMENT_ROOT,(1,2,1,1,0,0,711,3,0)),
                            Stream(DEVELOPMENT_ROOT,(1,2,1,0,0,0,10,3,0)),
                            Stream(DEVELOPMENT_ROOT+1,(1,2,1,0,0,0,711,3,0))):
            streams={s:dict(rows) for s,rows in self.streams.items()};streams['planning']['load']=replacement
            with self.subTest(replacement=replacement.spawn_key),self.assertRaises(ValueError):self.build(streams=streams)

    def test_physical_source_and_canonical_asset_or_sample_axes_are_bound(self):
        hashes=dict(self.context.input_hashes);hashes[next(iter(hashes))]='0'*64
        with patch.object(self.context,'input_hashes',hashes),self.assertRaises(ValueError):self.build()
        changed_roads=replace(self.context.roads,arc_ids=tuple(reversed(self.context.roads.arc_ids)))
        with patch.object(self.context,'roads',changed_roads),self.assertRaises(ValueError):self.build()
        original=self.samples['planning'][0]
        for field in ('grid_hazard','line_outages','road_wind_vectors_m_s','task_mu_full','load_p_kw'):
            arrays=dict(original.arrays);arrays[field]=arrays[field][:-1]
            ledger=dict(original.ledger);ledger['array_hashes']={k:array_hash(v) for k,v in arrays.items()}
            ledger.pop('scenario_sha256');digest=canonical_hash(ledger);ledger['scenario_sha256']=digest
            bad=replace(original,arrays=MappingProxyType(arrays),ledger=MappingProxyType(ledger),scenario_sha256=digest)
            pins={s:list(rows) for s,rows in self.arguments()['expected_scenario_sha256'].items()};pins['planning'][0]=digest
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.build(planning_samples=(bad,*self.samples['planning'][1:]),expected_scenario_sha256=pins)

    def test_empty_H_remains_structurally_zero_through_features_and_moments(self):
        from evcs_experiments.ablation_bank import build_independent_recombination_bank
        bank=deepcopy(self.bank);bank['tasks']=[]
        bank['task_sha256']=canonical_hash({k:bank[k] for k in ('dependency_sha256','config','certified','tasks')})
        prepared=self.context.prepare_target(self.context.events[0],bank)
        samples={}
        for split in ('planning','calibration'):
            template=self.context.generate(prepared,self.context.events[0],0,split)
            samples[split]=tuple(synthetic_sample(self.context,prepared,template,split,i) for i in range(8))
        args=self.arguments();args.update(planning_samples=samples['planning'],calibration_samples=samples['calibration'],
            expected_scenario_sha256={s:tuple(x.scenario_sha256 for x in rows) for s,rows in samples.items()})
        value=build_independent_recombination_bank(self.context,prepared,**args)
        for split in ('planning','calibration'):
            np.testing.assert_array_equal(getattr(value,split+'_features')[:,3],np.zeros(8))
            for scenario in getattr(value,split):self.assertEqual(scenario.arrays['task_mu_full'].shape,(0,24))
        self.assertEqual(value.bootstrap.center[3],0.);self.assertEqual(value.bootstrap.halfwidth[3],0.)
        for bounds in value.moments.bounds_by_rho.values():
            self.assertEqual(bounds['lower'][3],0.);self.assertEqual(bounds['upper'][3],0.)

    def test_target_state_H_time_axis_and_mutated_payload_are_rejected(self):
        from evcs_experiments.ablation_bank import build_independent_recombination_bank
        for changed in (replace(self.prepared,target=replace(self.prepared.target,state_sha256='0'*64)),
                        replace(self.prepared,tasks=replace(self.prepared.tasks,task_ids=tuple(reversed(self.prepared.tasks.task_ids)))),
                        replace(self.prepared,target=replace(self.prepared.target,replay_times_h=self.prepared.target.replay_times_h+.01))):
            with self.subTest(change=str(changed.target.state_sha256)[:8]),self.assertRaises(ValueError):
                build_independent_recombination_bank(self.context,changed,**self.arguments())
        original=self.samples['planning'][0];arrays=dict(original.arrays);arrays['road_blocked']=np.array(arrays['road_blocked'],copy=True);arrays['road_blocked'][0,0]=not arrays['road_blocked'][0,0]
        changed=replace(original,arrays=MappingProxyType(arrays))
        with self.assertRaises(ValueError):self.build(planning_samples=(changed,*self.samples['planning'][1:]))


if __name__=='__main__':unittest.main()
