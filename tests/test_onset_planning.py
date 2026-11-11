"""Real frozen forecast/H and one real atom; no solver or OOS."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.case_inputs import InputValidationError
from evcs_experiments.config import get_paths
from evcs_experiments.forecast_tasks import load_forecast_task_inputs, build_candidate_tasks
from evcs_experiments.formal_scenarios import PlanningContext
from evcs_experiments.onset_planning import OnsetPlanningContext


class OnsetPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=get_paths().data_root
        cls.event='michael_2018'
        cls.forecast=load_forecast_task_inputs(cls.root,cls.event,lead_h=0.)
        cls.bank=build_candidate_tasks(cls.forecast,overall_time_limit_seconds=10.)
        assert cls.bank.certified
        cls.context=OnsetPlanningContext(cls.root,cls.forecast.network.arcs)
        cls.prepared=cls.context.prepare_target(cls.event,cls.bank)
        cls.atom=cls.context.generate(cls.prepared,cls.context.events[1],0,'planning')
        ea_inputs=load_forecast_task_inputs(cls.root,cls.event,lead_h=1.)
        cls.ea_bank=build_candidate_tasks(ea_inputs,overall_time_limit_seconds=10.)
        cls.ea=PlanningContext(cls.root,ea_inputs.network.arcs)
        cls.ea_prepared=cls.ea.prepare_target(cls.event,cls.ea_bank)
        cls.ea_atom=cls.ea.generate(cls.ea_prepared,cls.context.events[1],0,'planning')

    def test_real_onset_rebuilds_all_fleet_queries_from_zero_state(self):
        p=self.prepared
        self.assertEqual(len(p.target.vehicle_states),80)
        self.assertEqual(len(p.bank['queries']),80*8)
        self.assertEqual(p.tasks.departure_h,0.)
        self.assertEqual(p.target.state_sha256,self.context.case.target(self.event,0.).state_sha256)
        self.assertNotEqual(p.target.state_sha256,self.ea_prepared.target.state_sha256)
        self.assertNotEqual(p.tasks.task_sha256,self.ea_prepared.tasks.task_sha256)
        self.assertEqual(len(p.tasks.task_ids),1128)
        self.assertEqual({q['arguments']['departure_h'] for q in p.bank['queries']},{0.})
        for q in p.bank['queries']:
            state={s.vehicle_id:s for s in p.target.vehicle_states}[q['vehicle_id']]
            self.assertEqual(q['arguments']['initial_soc'],state.soc)
            self.assertEqual(q['arguments']['origin'],state.road_node)

    def test_same_known_forecast_and_paired_primitive_addresses(self):
        np.testing.assert_array_equal(self.forecast.forecast_travel_hours,
            load_forecast_task_inputs(self.root,self.event,1.).forecast_travel_hours[4:])
        a,b=self.atom,self.ea_atom
        self.assertEqual(a.ledger['scenario_id'],b.ledger['scenario_id'])
        for name in ('wind_parameters','line_stream_id','road_stream_ids','load_draw'):
            self.assertEqual(a.ledger[name],b.ledger[name])
        for name in ('line_standard_normals','line_threshold_uniforms','road_corridor_uniforms',
                     'load_standard_innovation','load_epsilon','load_p_kw','load_q_kvar'):
            np.testing.assert_array_equal(a.arrays[name],b.arrays[name])
        np.testing.assert_array_equal(a.arrays['grid_hazard'],b.arrays['grid_hazard'][4:])
        self.assertNotEqual(a.scenario_sha256,b.scenario_sha256)

    def test_line_accumulation_starts_at_zero_and_replay_keeps_H(self):
        a=self.atom
        lt=self.context.primitives[('formal','planning',self.context.events[1],0)][0]
        expected=self.context.lines.replay(np.arange(96)/4,a.arrays['line_wind_vectors_m_s'],lt,lead_h=0.)
        np.testing.assert_array_equal(a.arrays['line_outages'],expected.outages)
        self.assertEqual(a.arrays['line_outages'].shape[0],96)
        self.assertEqual(a.arrays['task_mu_full'].shape,(1128,24))
        self.assertEqual(a.ledger['task_sha256'],self.prepared.tasks.task_sha256)
        self.assertEqual(a.ledger['lead_h'],0.)
        self.assertEqual(a.arrays['features'].shape,(5,))
        self.assertTrue(np.isfinite(a.arrays['features']).all())
        rows=self.prepared.bank['tasks']
        vehicles=sorted({r['vehicle_id'] for r in rows})
        losses=[np.mean([1-min(1.,a.arrays['task_usable_energy_dc_kwh'][i]/r['usable_energy_kwh'])
            for i,r in enumerate(rows) if r['vehicle_id']==v]) for v in vehicles]
        self.assertAlmostEqual(a.arrays['features'][3],float(np.mean(losses)),places=14)
        repeat=self.context.generate(self.prepared,self.context.events[1],0,'planning')
        self.assertEqual(repeat.scenario_sha256,a.scenario_sha256)

    def test_rejects_EA_provisional_missing_queries_and_stale_state(self):
        for mutation in ('ea','provisional','missing','state','destination','deadline','query_id'):
            b=deepcopy(self.bank.as_dict())
            if mutation=='ea':b=self.ea_bank.as_dict()
            if mutation=='provisional':b['certified']=False
            if mutation=='missing':b['queries'].pop()
            if mutation=='state':b['queries'][0]['arguments']['initial_soc']+=.01
            if mutation=='destination':b['queries'][0]['arguments']['destination']+=1
            if mutation=='deadline':b['queries'][0]['arguments']['latest_arrival_h']-=1
            if mutation=='query_id':b['queries'][0]['query_id']='0'*64
            with self.subTest(mutation=mutation), self.assertRaises(InputValidationError):
                self.context.prepare_target(self.event,b)

    def test_unknown_prepared_changed_bank_and_OOS_are_refused(self):
        with self.assertRaises(InputValidationError):
            self.context.generate(self.ea_prepared,self.context.events[1],0,'planning')
        p=self.context.prepare_target(self.event,self.bank)
        p.bank['tasks'][0]['arrival_soc']+=.01
        with self.assertRaises(InputValidationError):
            self.context.generate(p,self.context.events[1],0,'planning')
        with patch.object(type(self.context.wind),'sample',side_effect=AssertionError('Must not sample')):
            for kwargs in (dict(split='oos'),dict(regime='R2'),dict(variant_index=20)):
                args=dict(base_event=self.context.events[1],variant_index=0,split='planning');args.update(kwargs)
                with self.subTest(kwargs=kwargs),self.assertRaises(InputValidationError):
                    self.context.generate(self.prepared,**args)

    def test_factory_builds_H_without_scenario_generation(self):
        with patch.object(type(self.context.wind),'sample',side_effect=AssertionError('Future draw')):
            p=self.context.prepare_target(self.event,overall_time_limit_seconds=10.)
        self.assertEqual(p.tasks.task_sha256,self.prepared.tasks.task_sha256)
