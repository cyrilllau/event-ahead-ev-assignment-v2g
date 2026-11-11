"""Independent producer/foreign-method/normal-result binding negatives."""
import copy,unittest
from evcs_experiments import case0_accepted_plan as bridge
from tests.test_case0_accepted_plan import AcceptedPlanTests

class IndependentAcceptedPlanTests(AcceptedPlanTests):
 def test_changed_original_source_cannot_inherit_acceptance(self):
  self.sp.write_text('# altered producer\n')
  with self.assertRaises(ValueError):self.load()
 def test_other_methods_cannot_claim_this_gap_override(self):
  for method in ('OR_DRO','SAA','ARO','EA_DRO_BAD'):
   self.assertIsNone(bridge.load_accepted_plan(self.root,self.state,method=method))
 def test_later_global_management_fee_is_allowed_without_ea_fee(self):
  self.state['compute_seconds']+=2.5
  r=self.load()['comparison_result'];self.assertEqual(r['configuration_wall_seconds'],1501.)
  self.assertFalse(r['gap_passed']);self.assertTrue(r['accepted_for_comparison'])

if __name__=='__main__':unittest.main()
