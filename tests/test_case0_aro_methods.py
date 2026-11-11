import unittest
from unittest.mock import patch
from tests.test_case0_remaining_methods import RemainingMethodsTests
from evcs_experiments.case0_aro_methods import run_remaining
class AROOnlyTests(unittest.TestCase):
 def setUp(self):
  RemainingMethodsTests.setUp(self)
  self.s['configuration_spent_seconds'].update({'size0/OR_DRO':1260.,'size0/SAA':1251.});self.s['configuration_master_spent_seconds'].update({'size0/OR_DRO':1136.,'size0/SAA':1004.});self.s['completed']['size0/SAA']={'gap_passed':False,'gap':.88597}
  self.api.save_state(self.root,self.s)
 def run_it(self):
  with patch('evcs_experiments.case0_aro_methods.load_accepted_or',return_value={'accepted_for_comparison':True}),patch('evcs_experiments.case0_aro_methods.check_saa_deferral',return_value={'accepted_for_comparison':False}):
   return run_remaining(self.root,self.op,seconds=1850,accepted_api=self.accepted,api=self.api)
 def test_only_aro_no_old_method_restart_and_no_OOS(self):
  s=self.run_it();self.assertEqual(self.calls,['ARO']);self.assertEqual(s['configuration_spent_seconds']['size0/SAA'],1251.);self.assertFalse(s['completed']['size0/SAA']['gap_passed']);self.assertFalse(s['evaluation_opened']);self.assertEqual(s['status'],'ARO_RESULT_PENDING_SAA_QUALITY_AND_ALL_PLAN_AUDIT')
 def test_aro_retains_normal_gap_gate(self):
  self.gap=.02;s=self.run_it();self.assertEqual(self.calls,['ARO']);self.assertEqual(s['status'],'DIAGNOSIS_REQUIRED');self.assertFalse(s['completed']['size0/ARO']['gap_passed'])
 def test_missing_or_acceptance_stops_before_worker(self):
  with self.assertRaisesRegex(ValueError,'accepted OR'):run_remaining(self.root,self.op,seconds=1850,accepted_api=self.accepted,api=self.api)
  self.assertEqual(self.calls,[])
