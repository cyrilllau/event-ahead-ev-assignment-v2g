"""Independent numerical LP diagnostics: signs, infinity and no formal gate."""
import copy
from tests.test_case0_saved_lp_bound import DualCertificateTests, SavedLPBoundTests, audit

class IndependentDualTests(DualCertificateTests):
 def test_wrong_lower_row_sign_with_same_shape_rejected(self):
  w=copy.deepcopy(self.w);w['ge'][0]=-.001
  with self.assertRaisesRegex(ValueError,'dual signs/stationarity'):self.check(w)
 def test_infinite_upper_bound_reduced_cost_cannot_be_discarded(self):
  w=copy.deepcopy(self.w);w['reduced_cost'][0]=-.001
  with self.assertRaisesRegex(ValueError,'dual signs/stationarity'):self.check(w)

class IndependentDualOnlyTests(DualCertificateTests):
 def test_dual_survives_infeasible_primal_without_claiming_optimality(self):
  w=copy.deepcopy(self.w);w['primal'][1]=100.
  r=self.check(w);self.assertTrue(r['passed']);self.assertEqual(r['dual_objective_usd'],4.)
 def test_dual_survives_missing_primal(self):
  w=copy.deepcopy(self.w);w['primal']=None
  r=self.check(w);self.assertTrue(r['passed']);self.assertEqual(r['dual_objective_usd'],4.)

class IndependentSavedLPTests(SavedLPBoundTests):
 def test_diagnostic_gap_does_not_become_formal_gap_pass(self):
  r=self.run_numeric();self.assertTrue(r['bound_available'])
  self.assertFalse(r['gap_passed']);self.assertFalse(r['planning_acceptance'])
  self.assertIn('diagnostic_gap_within_threshold',r)
