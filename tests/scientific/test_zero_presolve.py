import unittest
from dataclasses import replace
import numpy as np
from scipy.sparse import csr_matrix
from ev_assignment.models.zero_presolve import reduce_zero_columns
from ev_assignment.models.extensive import assemble_extensive, solve_extensive
from ev_assignment.models.first_stage import assemble_first_stage
from ev_assignment.models.recourse import assemble_recourse
from s03_fixtures import build_golden_cases


class ZeroPresolveTests(unittest.TestCase):
    def test_chain_signed_singletons_and_contradictory_rows(self):
        a = csr_matrix([[1.,0,0],[-1,1,0],[0,-1,-2],[0,0,0]])
        r = reduce_zero_columns(a,[-np.inf]*3+[1.],[0,0,0,1.],[0,0,-np.inf],[1,1,0])
        np.testing.assert_array_equal(r.zero_columns,[0,1,2])
        self.assertEqual(r.matrix.shape,(1,0))
        self.assertEqual(r.row_indices.tolist(),[3])
        np.testing.assert_array_equal(r.expand([]),[0,0,0])

    def test_tiny_positive_rhs_and_nonzero_fixed_columns_never_zeroed(self):
        a=csr_matrix([[1.,0],[0,1.]])
        r=reduce_zero_columns(a,[-np.inf,-np.inf],[1e-30,2],[0,2],[1,2])
        self.assertEqual(r.zero_columns.tolist(),[])
        np.testing.assert_array_equal(r.matrix.toarray(),a.toarray())

    def test_signed_homogeneous_equality(self):
        r=reduce_zero_columns(csr_matrix([[-2.]]),[0],[0],[-1],[0])
        self.assertEqual(r.zero_columns.tolist(),[0])

    def test_opposite_facets_fix_free_variable_without_changing_feasible_set(self):
        r=reduce_zero_columns(csr_matrix([[1.],[-3.]]),[-np.inf]*2,[0,0],[-np.inf],[np.inf])
        self.assertEqual(r.zero_columns.tolist(),[0])

    def test_impossible_zero_intersection_is_left_for_solver(self):
        r=reduce_zero_columns(csr_matrix([[1.],[-1.]]),[-np.inf]*2,[0,0],[1],[2])
        self.assertEqual(r.zero_columns.tolist(),[])

    def test_no_zeroing_of_unbounded_homogeneous_singleton(self):
        r=reduce_zero_columns(csr_matrix([[1.]]),[-np.inf],[0],[-np.inf],[1])
        self.assertEqual(r.zero_columns.tolist(),[])

    def test_reduced_eager_models_match_all_hand_optima_and_original_rows(self):
        for fixture in build_golden_cases():
            f=assemble_first_stage(fixture.case,fixture.tasks)
            ts=tuple(assemble_recourse(fixture.case,fixture.tasks,s) for s in fixture.scenarios)
            for risk in ('DRO','ARO','SAA'):
                m=assemble_extensive(f,ts,risk_model=risk,scenario_ids=fixture.scenario_ids,
                    probability_set=fixture.probability_set,empirical_probabilities=fixture.empirical_probabilities,
                    scenario_cost_constants_usd=fixture.scenario_cost_constants_usd)
                original=solve_extensive(m,time_limit_seconds=5)
                reduced=solve_extensive(m,time_limit_seconds=5,zero_presolve=True)
                with self.subTest(fixture=fixture.name,risk=risk):
                    self.assertTrue(original.optimal);self.assertTrue(reduced.optimal)
                    self.assertAlmostEqual(original.objective_usd,reduced.objective_usd,places=7)
                    self.assertAlmostEqual(original.lower_bound_usd,reduced.lower_bound_usd,places=7)
                    self.assertEqual(reduced.solution.vector.shape,m.c.shape)
                    self.assertTrue(reduced.certificate['primal_feasible'])
                    cert=reduced.raw_solver['zero_reduction']
                    self.assertIsNone(cert['coefficient_threshold'])
                    z=np.array(cert['zero_columns'],dtype=int)
                    self.assertTrue(np.all(reduced.solution.vector[z]==0))

if __name__=='__main__':unittest.main()
