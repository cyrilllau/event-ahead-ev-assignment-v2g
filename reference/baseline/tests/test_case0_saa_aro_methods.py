import unittest
from tests.test_case0_remaining_methods import RemainingMethodsTests
from evcs_experiments.case0_saa_aro_methods import run_remaining
class RemainingTwoTests(unittest.TestCase):
 def setUp(self):
  RemainingMethodsTests.setUp(self);self.s['deferred_quality_plans']={'size0/OR_DRO':{'gap':.00635005465,'accepted_for_comparison':False,'gap_passed':False}};self.s['configuration_spent_seconds']['size0/OR_DRO']=1260.;self.s['configuration_master_spent_seconds']['size0/OR_DRO']=1136.;self.api.save_state(self.root,self.s)
 def run_it(self):return run_remaining(self.root,self.op,seconds=3700,accepted_api=self.accepted,api=self.api)
 def test_runs_only_saa_aro_without_accepting_or_or_opening_oos(self):
  s=self.run_it();self.assertEqual(self.calls,['SAA','ARO']);self.assertEqual(s['configuration_spent_seconds']['size0/OR_DRO'],1260.);self.assertFalse(s['deferred_quality_plans']['size0/OR_DRO']['accepted_for_comparison']);self.assertFalse(s['evaluation_opened']);self.assertNotIn('size0/OR_DRO',s['completed']);self.assertEqual(s['status'],'SAA_ARO_RESULTS_PENDING_OR_QUALITY_AND_ALL_PLAN_AUDIT')
 def test_saa_failure_still_stops_aro(self):
  self.gap=.02;s=self.run_it();self.assertEqual(self.calls,['SAA']);self.assertEqual(s['status'],'DIAGNOSIS_REQUIRED')
 def test_explicit_deferral_required(self):
  self.s['deferred_quality_plans']={};self.api.save_state(self.root,self.s)
  with self.assertRaisesRegex(ValueError,'deferral'):self.run_it()
