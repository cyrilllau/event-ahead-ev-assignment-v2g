"""Case0 real IEEE33 input identity, independent streams and immutable reuse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.case0_inputs import Case0Context


class Case0InputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.ea=Case0Context(cls.root/'ea');cls.onset=Case0Context(cls.root/'or',lead_hours=0)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_full_network_and_deterministic_same_depot_cohort(self):
        c=self.ea
        self.assertEqual(len(c.case.bus_ids),33);self.assertEqual(len(c.case.line_ids),32)
        self.assertEqual(tuple(c.case.periods),tuple(range(1,25)))
        self.assertEqual(c.selected_vehicle_ids,tuple(f'D{d}-EV{i:03d}' for d in range(1,5) for i in range(1,4)))
        self.assertEqual(c.selected_vehicle_ids,self.onset.selected_vehicle_ids)
        self.assertEqual(len(c.bank['queries']),12*8)
        self.assertTrue(set(c.tasks.vehicle_ids)<=set(c.selected_vehicle_ids))
        self.assertNotEqual(c.tasks.task_sha256,self.onset.tasks.task_sha256)
        self.assertNotEqual(c.target.state_sha256,self.onset.target.state_sha256)
        self.assertEqual(len(c.planning_ids),16)

    def test_real_planning_calibration_and_test_are_separate_development_streams(self):
        a=self.ea.sample(0);b=self.ea.sample(0,split='calibration');c=self.ea.sample(0,split='oos')
        for sample,split in [(a,'planning'),(b,'calibration'),(c,'oos')]:
            self.assertTrue(sample.physical.development);self.assertFalse(sample.ledger['formal_OOS'])
            self.assertEqual(sample.physical.split,split);self.assertEqual(sample.ledger['variant_index'],10000)
            self.assertEqual(sample.physical.task_sha256,self.ea.tasks.task_sha256)
            self.assertEqual(sample.physical.load_p_kw.shape,(24,33))
            self.assertEqual(len(sample.physical.times_h),100)
        self.assertEqual(len({x.ledger['line_stream_id'] for x in (a,b,c)}),3)
        self.assertNotEqual(a.ledger['load_draw']['draw_sha256'],b.ledger['load_draw']['draw_sha256'])

    def test_DSP_and_OR_keep_matching_weather_addresses_and_own_task_state(self):
        ea=self.ea.sample(0,split='oos');dsp=self.ea.sample(0,split='oos',departure_h=0)
        onset=self.onset.sample(0,split='oos');r3=self.ea.sample(0,split='oos',regime='R3')
        self.assertEqual(ea.physical.task_sha256,dsp.physical.task_sha256)
        self.assertEqual(ea.ledger['assignment_state_sha256'],dsp.ledger['assignment_state_sha256'])
        self.assertEqual(dsp.replay.departure_h,0.);self.assertEqual(ea.replay.departure_h,-1.)
        for field in ('wind_parameters','line_stream_id','road_stream_ids','load_draw'):
            self.assertEqual(ea.ledger[field],onset.ledger[field]);self.assertEqual(ea.ledger[field],r3.ledger[field])
        self.assertEqual(len(onset.physical.times_h),96)
        np.testing.assert_array_equal(ea.physical.load_p_kw,onset.physical.load_p_kw)
        self.assertNotEqual(ea.physical.scenario_sha256,dsp.physical.scenario_sha256)

    def test_reopen_without_task_regeneration_rejects_changed_configuration(self):
        with patch('evcs_experiments.case0_inputs.build_candidate_tasks',side_effect=AssertionError('Do not rebuild')):
            again=Case0Context(self.root/'ea');self.assertEqual(again.identity,self.ea.identity)
            with self.assertRaisesRegex(ValueError,'configuration'):
                Case0Context(self.root/'ea',variant_start=20000)

    def test_calibration_is_own_subset_and_verified_reusable(self):
        p=self.ea.probability_set
        self.assertEqual(len(p.support_ids),16);self.assertEqual(p.features.shape,(16,5))
        self.assertEqual(self.ea.bootstrap['calibration_rows'],32)
        again=Case0Context(self.root/'ea')
        with patch.object(again,'sample',side_effect=AssertionError('Do not regenerate')):
            np.testing.assert_array_equal(again.probability_set.features,p.features)
        self.assertTrue((self.root/'ea/calibration_manifest.json').exists())

    def test_explicit_debug_and_invalid_domains(self):
        debug=Case0Context(self.root/'debug',fleet_per_depot=2,planning_variants=1,variant_start=20000)
        self.assertEqual(len(debug.selected_vehicle_ids),8);self.assertEqual(len(debug.planning_ids),8)
        self.assertIn('v20001',debug.planning_ids[0])
        for kwargs in ({'split':'formal'},{'split':'planning','regime':'R2'}):
            with self.assertRaises(ValueError):debug.scenario(0,**kwargs)
        with self.assertRaises(ValueError):debug.scenario(64,split='oos')
        with self.assertRaises(ValueError):debug.scenario(0,departure_h=0)

if __name__=='__main__':unittest.main()
