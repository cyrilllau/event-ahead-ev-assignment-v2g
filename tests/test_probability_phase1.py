"""Hand-computed initialization witnesses, support retention and rank checks."""
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.probability_phase1 import solve_phase1
from evcs_experiments.random_streams import canonical_hash


def ambiguity(phi, lower, upper, ids=None):
    phi = np.asarray(phi, dtype=float)
    return ProbabilitySet(tuple(ids or [f's{i}' for i in range(len(phi))]),
                          tuple(f'f{j}' for j in range(phi.shape[1])),
                          phi, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def solver_witness(p, status=0):
    vector = None if p is None else np.asarray(p, dtype=float)
    return (SimpleNamespace(status=status, x=vector),
            {'status': status, 'success': status == 0, 'raw_solution': vector,
             'raw_objective': None if p is None else 0.,
             'infeasibility_certificate_available': False})


class Phase1Tests(unittest.TestCase):
    def solve(self, a):
        result = solve_phase1(a, time_limit_seconds=5)
        self.assertTrue(result.feasible, result.as_dict())
        self.assertEqual(result.status, 'FEASIBLE_BASIC')
        self.assertTrue(result.basis_verified)
        self.assertEqual(result.support_ids, a.support_ids)
        self.assertEqual(result.ambiguity_sha256, a.ambiguity_sha256)
        lifted = np.zeros(len(a.support_ids))
        lifted[list(result.initial_support_indices)] = result.initial_probabilities
        np.testing.assert_array_equal(lifted, result.probabilities)
        np.testing.assert_array_equal(result.solver['raw_solution'], result.probabilities)
        self.assertTrue(result.certificate['restricted_witness_feasible'])
        self.assertTrue(result.certificate['restricted_lift_exactly_matches_full_witness'])
        self.assertLessEqual(abs(result.initial_probabilities.sum() - 1.), 1e-8)
        means = a.features[list(result.initial_support_indices)].T @ result.initial_probabilities
        self.assertTrue(np.all(means >= a.lower - 1e-8))
        self.assertTrue(np.all(means <= a.upper + 1e-8))
        self.assertEqual(result.certificate['basis']['full_active_rank'], len(a.support_ids))
        json.dumps(result.as_dict(), allow_nan=False)
        return result

    def test_singleton_with_constant_feature(self):
        a = ambiguity([[1.04]], [1.04], [1.04], ['only'])
        r = self.solve(a)
        np.testing.assert_array_equal(r.probabilities, [1.])
        self.assertEqual(r.initial_support_ids, ('only',))

    def test_fixed_moments_require_three_atoms(self):
        a = ambiguity([[0., 0.], [1., 0.], [0., 1.]], [.2, .3], [.2, .3])
        r = self.solve(a)
        np.testing.assert_allclose(r.probabilities, [.5, .2, .3], atol=1e-12)
        self.assertEqual(r.initial_support_ids, a.support_ids)
        self.assertEqual(r.certificate['basis']['compressed_rank'], 3)

    def test_nonempty_full_set_does_not_imply_arbitrary_subset_feasible(self):
        a = ambiguity([[0.], [1.]], [.4], [.6])
        r = self.solve(a)
        self.assertEqual(len(r.initial_support_ids), 2)
        for value in (0., 1.):
            single = ambiguity([[value]], [.4], [.6])
            rejected = solve_phase1(single)
            self.assertFalse(rejected.feasible)
            self.assertEqual(rejected.status, 'INFEASIBLE')

    def test_zero_feature_simplex_has_one_basic_atom(self):
        r = self.solve(ambiguity(np.empty((5, 0)), [], []))
        self.assertEqual(len(r.initial_support_indices), 1)
        self.assertEqual(r.initial_probabilities[0], 1.)

    def test_duplicate_and_redundant_moment_rows_do_not_inflate_rank(self):
        a = ambiguity([[0., 0., 1.04], [1., 1., 1.04]], [.25, .25, 1.04], [.25, .25, 1.04])
        r = self.solve(a)
        np.testing.assert_allclose(r.probabilities, [.75, .25], atol=1e-12)
        self.assertEqual(r.certificate['basis']['compressed_rank'], 2)
        self.assertGreater(len(r.certificate['basis']['active_moment_row_ids']), 2)

    def test_tiny_positive_probability_from_real_lp_is_not_thresholded(self):
        a = ambiguity([[0.], [1e6]], [1e-6], [1e-6])
        r = self.solve(a)
        self.assertGreater(r.probabilities[1], 0.)
        self.assertLess(r.probabilities[1], 1e-8)
        self.assertAlmostEqual(r.probabilities[1], 1e-12, places=20)
        self.assertEqual(r.initial_support_indices, (0, 1))
        self.assertEqual(r.certificate['retained_tiny_positive_probability_ids'], ('s1',))
        self.assertEqual(r.certificate['basis']['zero_constraint_indices'], ())

    def test_interior_distribution_is_feasible_but_not_basic(self):
        a = ambiguity(np.empty((3, 0)), [], [])
        with patch('evcs_experiments.probability_phase1._solve_lp', return_value=solver_witness([.2, .3, .5])):
            r = solve_phase1(a)
        self.assertTrue(r.feasible); self.assertFalse(r.basis_verified)
        self.assertEqual(r.status, 'FEASIBLE_NONBASIC')
        self.assertEqual(r.initial_support_indices, (0, 1, 2))
        self.assertEqual(r.certificate['basis']['full_active_rank'], 1)

    def test_nearly_active_moment_is_not_promoted_by_feasibility_tolerance(self):
        a = ambiguity([[0.], [1.]], [.5 - 1e-9], [.5 + 1e-9])
        with patch('evcs_experiments.probability_phase1._solve_lp', return_value=solver_witness([.5, .5])):
            r = solve_phase1(a)
        self.assertTrue(r.feasible); self.assertFalse(r.basis_verified)
        self.assertEqual(r.certificate['basis']['active_moment_row_ids'], ('normalization',))

    def test_negative_roundoff_is_retained_without_claiming_basic(self):
        a = ambiguity(np.empty((2, 0)), [], [])
        p = np.array([-1e-12, 1. + 1e-12])
        with patch('evcs_experiments.probability_phase1._solve_lp', return_value=solver_witness(p)):
            r = solve_phase1(a)
        self.assertTrue(r.feasible); self.assertFalse(r.basis_verified)
        np.testing.assert_array_equal(r.probabilities, p)
        np.testing.assert_array_equal(r.initial_probabilities, p)
        self.assertEqual(r.initial_support_indices, (0, 1))
        self.assertEqual(r.certificate['retained_negative_probability_ids'], ('s0',))

    def test_coupled_empty_set_has_no_fabricated_farkas_or_support(self):
        a = ambiguity([[0., 0.], [1., 1.]], [.8, 0.], [1., .2])
        r = solve_phase1(a)
        self.assertEqual(r.status, 'INFEASIBLE'); self.assertFalse(r.feasible)
        self.assertIsNone(r.probabilities); self.assertEqual(r.initial_support_ids, ())
        self.assertFalse(r.certificate['infeasibility_certified'])
        self.assertFalse(r.solver['infeasibility_certificate_available'])

    def test_tiny_constant_contradiction_is_not_tolerance_repaired(self):
        a = ambiguity([[0.], [0.]], [1e-12], [.2])
        r = solve_phase1(a)
        self.assertEqual(r.status, 'INFEASIBLE'); self.assertFalse(r.feasible)
        self.assertTrue(r.certificate['infeasibility_certified'])
        self.assertEqual(r.certificate['analytic_infeasibility']['kind'],
                         'constant_feature_outside_moment_interval')

    def test_timeout_without_witness_does_not_invent_initialization(self):
        a = ambiguity(np.empty((2, 0)), [], [])
        with patch('evcs_experiments.probability_phase1._solve_lp', return_value=solver_witness(None, 1)):
            r = solve_phase1(a)
        self.assertEqual(r.status, 'TIME_LIMIT'); self.assertFalse(r.feasible)
        self.assertFalse(r.basis_verified); self.assertIsNone(r.initial_probabilities)

    def test_timeout_with_verified_witness_preserves_timeout_status(self):
        a = ambiguity([[0.], [1.]], [.25], [.25])
        with patch('evcs_experiments.probability_phase1._solve_lp', return_value=solver_witness([.75, .25], 1)):
            r = solve_phase1(a)
        self.assertEqual(r.status, 'TIME_LIMIT'); self.assertTrue(r.feasible)
        self.assertTrue(r.basis_verified); self.assertFalse(r.certificate['solver_optimal'])
        self.assertEqual(r.initial_support_indices, (0, 1))

    def test_success_cannot_hide_infeasible_or_wrong_shape_witness(self):
        a = ambiguity([[0.], [1.]], [.5], [.5])
        for p in ([1., 0.], [1., 0., 0.], [-1e-5, 1. + 1e-5]):
            with self.subTest(p=p), patch('evcs_experiments.probability_phase1._solve_lp',
                                        return_value=solver_witness(p)):
                r = solve_phase1(a)
            self.assertEqual(r.status, 'NUMERICAL_CERTIFICATE_FAILED')
            self.assertFalse(r.feasible); self.assertFalse(r.basis_verified)
            self.assertEqual(r.initial_support_indices, ())
            self.assertIsNone(r.initial_probabilities)

    def test_immutable_identity_and_original_support_order(self):
        a = ambiguity([[0., 0.], [1., 0.], [0., 1.]], [.2, .3], [.2, .3], ['z', 'a', 'm'])
        r = self.solve(a)
        self.assertEqual(r.initial_support_ids, ('z', 'a', 'm'))
        expected = canonical_hash({'ambiguity_sha256': a.ambiguity_sha256,
                                   'support_ids': list(a.support_ids), 'probabilities': r.probabilities.tolist()})
        self.assertEqual(r.witness_sha256, expected)
        with self.assertRaises(ValueError): r.probabilities.setflags(write=True)
        with self.assertRaises(ValueError): r.initial_probabilities.setflags(write=True)
        with self.assertRaises(TypeError): r.certificate['support_threshold'] = 1e-8
        with self.assertRaises(FrozenInstanceError): r.feasible = False

    def test_invalid_set_shape_and_limits_rejected(self):
        with self.assertRaises(ValueError):
            ambiguity([[0., 1.]], [0.], [1.])
        with self.assertRaises(ValueError):
            solve_phase1({'features': [[0.]]})
        a = ambiguity([[0.]], [0.], [1.])
        for limit in (0., -1., True, np.inf, np.nan):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                solve_phase1(a, time_limit_seconds=limit)

    def test_lp_is_original_zero_objective_and_single_thread(self):
        import evcs_experiments.probability_phase1 as module
        a = ambiguity([[0.], [1.]], [.25], [.5])
        with patch.object(module, 'linprog', wraps=module.linprog) as lp:
            self.solve(a)
        self.assertEqual(lp.call_count, 1)
        args = lp.call_args
        np.testing.assert_array_equal(args.args[0], [0., 0.])
        np.testing.assert_array_equal(args.kwargs['A_ub'], [[0., 1.], [0., -1.]])
        np.testing.assert_array_equal(args.kwargs['b_ub'], [.5, -.25])
        np.testing.assert_array_equal(args.kwargs['A_eq'], [[1., 1.]])
        self.assertEqual(args.kwargs['method'], 'highs-ds')
        self.assertEqual(args.kwargs['options']['threads'], 1)
        self.assertEqual(args.kwargs['options']['primal_feasibility_tolerance'], 1e-8)


if __name__ == '__main__':
    unittest.main()
