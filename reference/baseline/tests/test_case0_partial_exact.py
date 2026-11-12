import time,unittest
from tests.test_case0_aro_saved_arrays import AROArraysTests
from tests.test_case0_saa_saved_arrays import SavedArraysTests
from evcs_experiments.case0_aro_saved_arrays import derive_saved_aro_cuts
from evcs_experiments.case0_saa_saved_arrays import derive_saved_saa_cuts
from evcs_experiments.case0_anchor_refinement import build_cut_bundle
from evcs_experiments.case0_partial_exact import augment_exact_scenario
from evcs_experiments.relaxation_bound import solve_relaxation_bound
class PartialExactTests(unittest.TestCase):
 def test_exact_one_scenario_strengthens_ARO_to_hand_optimum(self):
  AROArraysTests.setUp(self);base,_,_=derive_saved_aro_cuts(self.A,self.d,self.h,self.u,self.e,deadline=time.monotonic()+3)
  model,r=augment_exact_scenario(base,self.A,self.d,self.h,0);sol,_=solve_relaxation_bound(model,time_limit_seconds=2);self.assertAlmostEqual(sol['lower_bound_usd'],1.1)
 def test_selected_index_and_changed_domain_rejected(self):
  AROArraysTests.setUp(self);base,_,_=derive_saved_aro_cuts(self.A,self.d,self.h,self.u,self.e,deadline=time.monotonic()+3)
  with self.assertRaisesRegex(ValueError,'scenario'):augment_exact_scenario(base,self.A,self.d,self.h,2)
  base['c'][0]=99
  with self.assertRaisesRegex(ValueError,'objective'):augment_exact_scenario(base,self.A,self.d,self.h,0)
