"""Independent adversarial checks of exact streaming reduction (development only)."""
import unittest

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from evcs_experiments.saa_streaming import _reduce_recourse


class StreamingReductionIndependentReview(unittest.TestCase):
    def test_chain_zero_reduction_retains_contradiction_and_shared_first_stage(self):
        # z is a shared first-stage column: even a fixed zero may not be removed.
        # y1 = 0, y2 + y1 = 0 propagates; y2 = 1 must remain an infeasible row.
        matrix = csr_matrix([[0., 1., 0.], [0., 1., 1.], [0., 0., 1.]])
        lo = np.zeros(3); up = np.array([0., 2., 2.])
        rl = np.array([0., 0., 1.]); ru = rl.copy()
        reduced, rows, cols, proof = _reduce_recourse(matrix, rl, ru, lo, up, 1)
        self.assertEqual(cols.tolist(), [0])
        self.assertEqual(rows.tolist(), [2])
        self.assertEqual(reduced.shape, (1, 1))
        self.assertEqual(reduced.nnz, 0)
        self.assertGreater(len(proof), 1)

    def test_signed_bounds_and_opposite_singletons_force_only_zero(self):
        # Free y can become zero through independent lower/upper singleton rows.
        matrix = csr_matrix([[0., -2., 0.], [0., 3., 0.], [0., 0., 1.]])
        rl = np.array([-np.inf, -np.inf, -1.]); ru = np.array([0., 0., 2.])
        lo = np.array([0., -np.inf, -2.]); up = np.array([1., np.inf, 3.])
        reduced, rows, cols, _ = _reduce_recourse(matrix, rl, ru, lo, up, 1)
        self.assertEqual(cols.tolist(), [0, 2])
        self.assertEqual(rows.tolist(), [2])
        np.testing.assert_array_equal(reduced.toarray(), [[0., 1.]])

    def test_random_objectives_equal_original_with_negative_costs(self):
        # Multi-step homogeneous propagation must not remove a nonzero column,
        # and negative costs on retained columns remain exact.
        matrix = csr_matrix([[0., 1., 0., 0.], [0., 1., -1., 0.],
                             [1., 0., 1., 1.], [0., 0., 0., 1.]])
        rl = np.array([0., 0., 1., -1.]); ru = np.array([0., 0., 2., 2.])
        lo = np.array([0., 0., -3., -2.]); up = np.array([1., 4., 3., 3.])
        reduced, rows, cols, _ = _reduce_recourse(matrix, rl, ru, lo, up, 1)
        for seed in range(8):
            c = np.random.default_rng(seed).normal(size=4)
            original = linprog(c, A_ub=np.vstack([matrix.toarray(), -matrix.toarray()]),
                               b_ub=np.r_[ru, -rl], bounds=list(zip(lo, up)), method='highs')
            final = linprog(c[cols], A_ub=np.vstack([reduced.toarray(), -reduced.toarray()]),
                            b_ub=np.r_[ru[rows], -rl[rows]],
                            bounds=list(zip(lo[cols], up[cols])), method='highs')
            self.assertTrue(original.success); self.assertTrue(final.success)
            self.assertAlmostEqual(original.fun, final.fun, places=10)
            lifted = np.zeros(4); lifted[cols] = final.x
            self.assertTrue(np.all(matrix @ lifted >= rl - 1e-10))
            self.assertTrue(np.all(matrix @ lifted <= ru + 1e-10))


if __name__ == '__main__':
    unittest.main()
