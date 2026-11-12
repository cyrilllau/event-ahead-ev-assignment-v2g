import time,unittest
from unittest.mock import patch
from tests.test_case0_saved_candidate_audit import SavedCandidateTests,audit
class ORSavedCandidateTests(SavedCandidateTests):
 def run_or(self,**changes):
  out=self.root/'or_audit';out.mkdir();args=dict(master=self.master,header=self.header,context=self.context,output_dir=out,deadline=time.monotonic()+20,binding={'method_id':'OR_DRO','master':'original'},method_id='OR_DRO');args.update(changes)
  with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No new master')):return audit.audit_candidate(**args)
 def test_or_uses_exact_context_without_creating_bound_or_acceptance(self):
  self.context.lead_hours=0;r=self.run_or();self.assertEqual(r['method_id'],'OR_DRO');self.assertAlmostEqual(r['upper_bound_usd'],self.expected);self.assertIsNone(r['gap']);self.assertFalse(r['planning_acceptance']);self.assertTrue(r['full_support_fixed_Q_and_KKT_passed'])
 def test_or_wrong_departure_rejected_before_q(self):
  self.context.lead_hours=1
  with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('No Q')):
   with self.assertRaisesRegex(ValueError,'lead0'):self.run_or()
 def test_or_without_explicit_source_method_rejected(self):
  self.context.lead_hours=0
  with self.assertRaisesRegex(ValueError,'binding'):self.run_or(binding={})
if __name__=='__main__':unittest.main()
