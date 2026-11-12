import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from evcs_experiments.config import get_paths

spec = importlib.util.spec_from_file_location('root_lp_diagnostic', Path(os.environ.get('EVCS_ROOT_LP_DIAGNOSTIC_SCRIPT', get_paths().project_root/'scripts/diagnostics/profile_root_lp.py')))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class RootLPDiagnosticTests(unittest.TestCase):
    def test_fractional_relaxation_is_not_integer_plan(self):
        a = csc_matrix([[2.]])
        d = dict(c=np.array([1.]), lower_bounds=np.array([0.]), upper_bounds=np.array([1.]),
                 row_lower=np.array([1.]), row_upper=np.array([np.inf]), integrality=np.array([1]))
        with tempfile.TemporaryDirectory() as folder:
            r, v = diag.solve_relaxation(a, d, 5, Path(folder)/'ipm.log')
        self.assertTrue(r['certified_optimal_relaxation'])
        self.assertAlmostEqual(v['x'][0], .5)
        self.assertFalse(r['integer_plan'])
        self.assertFalse(r['accepted_campaign_lower_bound'])
        self.assertEqual(d['integrality'].tolist(), [1])
        self.assertFalse(diag.certificate(a, d, v['x'], v['row_dual']+1, v['col_dual'])['passed'])

    def test_free_equality_and_upper_bound_duals(self):
        a = csc_matrix([[1., 0.]])
        d = dict(c=np.array([2., -3.]), lower_bounds=np.array([-np.inf, 0.]), upper_bounds=np.array([np.inf, 4.]),
                 row_lower=np.array([1.]), row_upper=np.array([1.]), integrality=np.array([0, 0]))
        with tempfile.TemporaryDirectory() as folder:
            r, v = diag.solve_relaxation(a, d, 5, Path(folder)/'ipm.log')
        self.assertTrue(r['certified_optimal_relaxation'])
        self.assertAlmostEqual(r['certificate']['dual_objective'], -10.)
        self.assertFalse(diag.certificate(a, d, [1., 5.], v['row_dual'], v['col_dual'])['passed'])
        with self.assertRaises(ValueError):
            diag.certificate(a, d, [np.nan, 4.], v['row_dual'], v['col_dual'])

    def test_wrong_dual_sign_on_infinite_row_bound(self):
        a = csc_matrix([[1.]])
        d = dict(c=np.array([-1.]), lower_bounds=np.array([0.]), upper_bounds=np.array([np.inf]),
                 row_lower=np.array([0.]), row_upper=np.array([np.inf]))
        self.assertFalse(diag.certificate(a, d, [0.], [-1.], [0.])['passed'])
