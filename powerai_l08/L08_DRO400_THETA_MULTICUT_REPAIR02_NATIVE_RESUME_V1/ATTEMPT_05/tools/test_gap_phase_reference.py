import unittest
from gap_phase_reference import gap, route_after_bound_update, candidate_route, request_handover

class PolicyTests(unittest.TestCase):
    def test_relative_boundary(self):
        self.assertEqual(gap(9900, 10000)["reason"], "RELATIVE_1PCT")
    def test_relative_outside(self):
        self.assertFalse(gap(9899, 10000)["reached"])
    def test_absolute_only(self):
        self.assertEqual(gap(9.5, 10)["reason"], "ABSOLUTE_1USD")
    def test_both(self):
        self.assertEqual(gap(199, 200)["reason"], "BOTH")
    def test_equal_zero(self):
        self.assertTrue(gap(0, 0)["reached"])
    def test_missing_bound(self):
        self.assertFalse(gap(None, 200)["reached"])
    def test_negative_gap(self):
        with self.assertRaises(ValueError): gap(201, 200)
    def test_nonfinite(self):
        for value in [float('inf'), float('-inf'), float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError): gap(100, value)
    def test_wrong_scope(self):
        with self.assertRaises(ValueError): gap(199, 200, scope_matches=False)
    def test_post_master_gap(self):
        self.assertEqual(route_after_bound_update(199, 200), "FINAL_CERTIFY")
    def test_old_gap_not_reached(self):
        self.assertEqual(route_after_bound_update(150, 200), "CONTINUE_SCIENCE")
    def test_wider_certified_upper(self):
        self.assertEqual(route_after_bound_update(100,100.5,150,
            certification_done_for_audited_point=True), "RESUME_SCIENCE_STRICT_GAP_NOT_REACHED")
    def test_certification_pending(self):
        self.assertEqual(route_after_bound_update(100,100.5,
            certification_done_for_audited_point=True), "NUMERICAL_TARGET_CERTIFICATION_PENDING")
    def test_valid_strict_target(self):
        self.assertEqual(route_after_bound_update(199,200,200,
            certification_done_for_audited_point=True), "TARGET_WITH_CERTIFIED_PRIMAL_UB_CHECK_LB_LEVEL")
    def test_new_point_needs_eval(self):
        self.assertEqual(candidate_route(complete_cached=False,certified_cut_violates=False,
            oracle_tight=True,master_proof_complete=True), "VALIDATE_AND_EVALUATE_FULL400")
    def test_cached_point_new_rho(self):
        self.assertEqual(candidate_route(complete_cached=True,certified_cut_violates=True,
            oracle_tight=True,master_proof_complete=True), "SEPARATE_CACHED_SUPPORT")
    def test_continue_master(self):
        self.assertEqual(candidate_route(complete_cached=True,certified_cut_violates=False,
            oracle_tight=True,master_proof_complete=False), "CONTINUE_SAME_MASTER")
    def test_refine_oracle(self):
        self.assertEqual(candidate_route(complete_cached=True,certified_cut_violates=False,
            oracle_tight=False,master_proof_complete=True), "REFINE_EXISTING_ORACLE_EVIDENCE")
    def test_start_only_no_handover(self):
        self.assertFalse(request_handover(root_completed=True,seconds_after_root=125,
            pending_evaluation_or_separation=False))
    def test_unfinished_root_no_handover(self):
        self.assertFalse(request_handover(root_completed=False,seconds_after_root=500,
            pending_evaluation_or_separation=True))
    def test_known_point_violated_can_handover(self):
        self.assertTrue(request_handover(root_completed=True,seconds_after_root=125,
            pending_evaluation_or_separation=True))
    def test_global_stop_priority(self):
        self.assertTrue(request_handover(root_completed=False,seconds_after_root=0,
            pending_evaluation_or_separation=False,global_stop=True))

if __name__ == '__main__':
    unittest.main(verbosity=2)
