"""Whole-row bootstrap reference and hand-solved normalized moment repair LPs."""
import importlib.util
import json
import math
import unittest

import numpy as np

from evcs_experiments.moment_calibration import bootstrap_moments, calibrate_moments

SCIPY_AVAILABLE = importlib.util.find_spec('scipy') is not None


def calibration_rows(x):
    x = np.asarray(x, dtype=float)
    return np.column_stack((x, np.full(len(x), .5), np.full(len(x), .5),
                            np.full(len(x), .5), np.ones(len(x))))


class MomentBootstrapTests(unittest.TestCase):
    def test_complete_row_resampling_preserves_dependence_and_higher_quantile(self):
        x = np.linspace(0., 1., 80)
        calibration = np.column_stack((x, x, 1-x, .1+.5*x, .91+.18*x))
        result = bootstrap_moments(calibration, development=True)
        self.assertEqual(result.indices.shape, (2000, 80))
        self.assertTrue(np.issubdtype(result.indices.dtype, np.integer))
        self.assertTrue(np.all((result.indices >= 0) & (result.indices < 80)))
        reference_means = np.array([calibration[index].mean(axis=0) for index in result.indices])
        np.testing.assert_allclose(result.bootstrap_means, reference_means, atol=1e-15, rtol=0)
        np.testing.assert_allclose(result.bootstrap_means[:, 1], result.bootstrap_means[:, 0], atol=1e-15)
        np.testing.assert_allclose(result.bootstrap_means[:, 2], 1-result.bootstrap_means[:, 0], atol=2e-15)
        np.testing.assert_allclose(result.bootstrap_means[:, 4], .91+.18*result.bootstrap_means[:, 0], atol=3e-15)
        np.testing.assert_array_equal(result.center, calibration.mean(axis=0))
        deviations = np.sort(np.abs(reference_means - calibration.mean(axis=0)), axis=0)
        order_index = math.ceil(.99 * (2000 - 1))
        np.testing.assert_allclose(result.halfwidth, deviations[order_index], atol=1e-15, rtol=0)
        self.assertEqual(result.quantile_probability, .99)

    def test_bootstrap_indices_repeat_across_values_targets_and_formal_domain(self):
        data = calibration_rows(np.linspace(.2, .7, 80))
        first = bootstrap_moments(data, development=True)
        repeat = bootstrap_moments(data, development=True)
        shifted = data.copy()
        shifted[:, 0] += .1
        other_target = bootstrap_moments(shifted, development=True)
        np.testing.assert_array_equal(first.indices, repeat.indices)
        np.testing.assert_array_equal(first.indices, other_target.indices)
        self.assertEqual(first.index_sha256, repeat.index_sha256)
        self.assertEqual(first.index_sha256, other_target.index_sha256)
        formal = bootstrap_moments(data, development=False)
        self.assertFalse(np.array_equal(first.indices, formal.indices))
        self.assertNotEqual(first.stream_identity, formal.stream_identity)
        for array in (first.center, first.halfwidth, first.bootstrap_means, first.indices):
            self.assertFalse(array.flags.writeable)

    def test_bad_physical_features_are_rejected_not_clipped(self):
        constant = np.tile([.3, .3, .3, .3, 1.09], (80, 1))
        exact = bootstrap_moments(constant, development=True)
        np.testing.assert_array_equal(exact.center, constant[0])
        np.testing.assert_array_equal(exact.halfwidth, np.zeros(5))
        np.testing.assert_array_equal(exact.bootstrap_means, np.tile(constant[0], (2000, 1)))
        for column, value in [(0, 1.01), (1, -.01), (4, 1.10), (4, .90), (3, np.nan)]:
            bad = calibration_rows([.3, .5, .7])
            bad[0, column] = value
            with self.assertRaises(ValueError):
                bootstrap_moments(bad, development=True)
        for bad in (np.empty((0, 5)), np.ones((80, 4))):
            with self.assertRaises(ValueError):
                bootstrap_moments(bad, development=True)


@unittest.skipUnless(SCIPY_AVAILABLE, 'SciPy is absent locally; PowerAI final validation must run these LP tests without skips')
class MomentFeasibilityTests(unittest.TestCase):
    def assert_witness(self, planning, bound):
        certificate = bound['feasibility']
        self.assertTrue(certificate['valid_witness'])
        p = np.asarray(certificate['witness'])
        self.assertAlmostEqual(float(p.sum()), 1., delta=1e-8)
        self.assertGreaterEqual(float(p.min()), -1e-8)
        mean = p @ planning
        self.assertTrue(np.all(mean >= np.asarray(bound['lower']) - 1e-8))
        self.assertTrue(np.all(mean <= np.asarray(bound['upper']) + 1e-8))

    def test_independent_calibration_center_outside_support_requires_known_delta(self):
        planning = calibration_rows([.2, .4])
        bootstrap = bootstrap_moments(calibration_rows([.6] * 8), development=True)
        result = calibrate_moments(planning, bootstrap)
        self.assertEqual(result.status, 'PASS_MOMENT_CALIBRATION')
        self.assertFalse(result.raw_feasibility['valid_witness'])
        # Closest feasible mean .4, center .6; .5*delta must cover .2.
        self.assertAlmostEqual(result.optimal_delta, .4, delta=1e-7)
        self.assertGreaterEqual(result.delta, result.optimal_delta - 1e-10)
        self.assertLess(result.numeric_margin_delta, 1e-6)
        np.testing.assert_array_equal(result.center, bootstrap.center)
        np.testing.assert_array_equal(result.original_halfwidth, bootstrap.halfwidth)
        scale = np.array([1., 1., 1., 1., .18])
        np.testing.assert_allclose(result.repaired_halfwidth, bootstrap.halfwidth + result.delta * scale, atol=1e-12)
        last = None
        for rho in (.5, 1., 1.5, 2.):
            bounds = result.bounds_by_rho[str(rho)]
            self.assert_witness(planning, bounds)
            lower, upper = np.asarray(bounds['lower']), np.asarray(bounds['upper'])
            np.testing.assert_allclose(lower, np.maximum([0, 0, 0, 0, .91], result.center-rho*result.repaired_halfwidth), atol=1e-12)
            np.testing.assert_allclose(upper, np.minimum([1, 1, 1, 1, 1.09], result.center+rho*result.repaired_halfwidth), atol=1e-12)
            if last is not None:
                self.assertTrue(np.all(lower <= last[0]))
                self.assertTrue(np.all(upper >= last[1]))
            last = lower, upper
        json.dumps(result.as_dict(), allow_nan=False)

    def test_positive_base_halfwidth_is_repaired_once_before_rho_scaling(self):
        planning = calibration_rows([.2, .4])
        bootstrap = bootstrap_moments(calibration_rows([.5, .5, .7, .7]), development=True)
        result = calibrate_moments(planning, bootstrap)
        expected = 2*(bootstrap.center[0]-.4) - bootstrap.halfwidth[0]
        self.assertAlmostEqual(result.optimal_delta, expected, delta=1e-7)
        self.assertAlmostEqual(result.optimal_delta, .3, delta=1e-7)

    def test_already_feasible_means_do_not_get_unsolicited_width_repair(self):
        planning = calibration_rows([.2, .4])
        bootstrap = bootstrap_moments(calibration_rows([.3] * 4), development=True)
        result = calibrate_moments(planning, bootstrap)
        self.assertTrue(result.raw_feasibility['valid_witness'])
        self.assertEqual(result.optimal_delta, 0.)
        self.assertEqual(result.delta, 0.)
        np.testing.assert_array_equal(result.repaired_halfwidth, result.original_halfwidth)
        for bounds in result.bounds_by_rho.values():
            self.assert_witness(planning, bounds)

    def test_explicit_constant_physical_dimension_is_not_artificially_widened(self):
        planning = calibration_rows([.2, .4])
        planning[:, 1], planning[:, 4] = .3, 1.09
        calibration = calibration_rows([.6] * 80)
        calibration[:, 1], calibration[:, 4] = .3, 1.09
        bootstrap = bootstrap_moments(calibration, development=True)
        lower, upper = (0., .3, 0., 0., .91), (1., .3, 1., 1., 1.09)
        result = calibrate_moments(planning, bootstrap, physical_lower=lower, physical_upper=upper)
        self.assertEqual(result.repaired_halfwidth[1], 0.)
        for bound in result.bounds_by_rho.values():
            self.assertEqual(bound['lower'][1], .3)
            self.assertEqual(bound['upper'][1], .3)
        inconsistent = planning.copy()
        inconsistent[0, 1] = .30001
        with self.assertRaises(ValueError):
            calibrate_moments(inconsistent, bootstrap, physical_lower=lower, physical_upper=upper)


if __name__ == '__main__':
    unittest.main()
