"""Independent checks of hourly conditional Gaussian load errors and source scaling."""
from __future__ import annotations

import csv
import types
from dataclasses import replace
import unittest

import numpy as np

from evcs_experiments.config import get_paths

from evcs_experiments.load_scenarios import (
    covariance_matrix, draw_load_errors, load_load_model, realize_load,
    reject_multivariate_normal,
)
from evcs_experiments.random_streams import stream

CASE_ROOT = get_paths().data_root


class ProposalSequence:
    def __init__(self, rows):
        self.rows = iter(np.asarray(rows, dtype=float))
        self.calls = 0

    def standard_normal(self, size):
        self.calls += 1
        value = next(self.rows)
        assert value.shape == (size,)
        return value.copy()


def ar1_reference(z, sigma=.03, rho=.8):
    """Independent AR recursion for the untruncated joint normal proposal."""
    z = np.asarray(z)
    result = np.empty_like(z)
    result[..., 0] = sigma * z[..., 0]
    for t in range(1, z.shape[-1]):
        result[..., t] = rho * result[..., t - 1] + sigma * np.sqrt(1 - rho**2) * z[..., t]
    return result


class LoadScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_load_model(CASE_ROOT)

    def draw(self, variant=0, **kwargs):
        return draw_load_errors(self.model, 5, variant, development=True, **kwargs)

    def test_archived_scaled_and_unscaled_cells_system_totals_and_clock(self):
        m = self.model
        self.assertEqual(m.bus_ids, tuple(range(1, 34)))
        np.testing.assert_array_equal(m.hours_h, np.arange(24))
        self.assertEqual(m.baseline_p_kw.shape, (24, 33))
        with (CASE_ROOT / '03_load/P_base_Q_base__N_T.csv').open() as f:
            rows = list(csv.DictReader(f))
        with (CASE_ROOT / '03_load/Ptilde_base_Qtilde_base__N_T.csv').open() as f:
            unscaled = {(int(r['t']), int(r['n'])): r for r in csv.DictReader(f)}
        for r in rows:
            t, n = int(r['t']), int(r['n'])
            p, q = float(r['P_base_n_t_kW']), float(r['Q_base_n_t_kvar'])
            self.assertEqual(m.baseline_p_kw[t - 1, n - 1], p)
            self.assertEqual(m.archived_baseline_q_kvar[t - 1, n - 1], q)
            old = unscaled[t, n]
            self.assertAlmostEqual(p, .8 * float(old['P_tilde_base_n_t_kW']), delta=1e-10)
            self.assertAlmostEqual(q, .8 * float(old['Q_tilde_base_n_t_kvar']), delta=1e-10)
        with (CASE_ROOT / '03_load/Pbase_system_Qbase_system_lambda_load__T.csv').open() as f:
            totals = list(csv.DictReader(f))
        np.testing.assert_allclose(m.baseline_p_kw.sum(axis=1),
            [float(r['P_base_system_t_kW']) for r in totals], atol=1e-9, rtol=0)
        np.testing.assert_allclose(m.archived_baseline_q_kvar.sum(axis=1),
            [float(r['Q_base_system_t_kvar']) for r in totals], atol=1e-9, rtol=0)
        self.assertEqual(m.baseline_p_kw[0, 1], 21.8336)
        self.assertEqual(m.archived_baseline_q_kvar[0, 1], 13.1)
        self.assertEqual(m.baseline_q_kvar[0, 1], 21.8336 * .6)

    def test_q1_rated_power_factor_and_authoritative_helper_crosscheck(self):
        reference_path = CASE_ROOT / 'tools/reactive_support_model.py'
        reference = types.ModuleType('frozen_reactive_support_reference')
        reference.__file__ = str(reference_path)
        exec(compile(reference_path.read_bytes(), str(reference_path), 'exec'), reference.__dict__)
        existing = reference.load_case(CASE_ROOT)
        for row in existing['loads']:
            cell = int(row['t']) - 1, int(row['n']) - 1
            self.assertEqual(self.model.baseline_q_kvar[cell], row['Q_base_n_t_kvar'])
            self.assertEqual(self.model.archived_baseline_q_kvar[cell], row['Q_archived_n_t_kvar'])
        self.assertLessEqual(np.max(np.abs(self.model.baseline_q_kvar - self.model.archived_baseline_q_kvar)), .000800000001)
        np.testing.assert_array_equal(self.model.baseline_q_kvar,
                                      self.model.baseline_p_kw * self.model.tan_phi_load)

    def test_hourly_covariance_and_cholesky_formula(self):
        lag = np.abs(np.arange(24)[:, None] - np.arange(24)[None, :])
        expected = .03**2 * .8**lag
        np.testing.assert_allclose(covariance_matrix(), expected, atol=1e-18, rtol=1e-14)
        np.testing.assert_allclose(self.model.covariance, expected, atol=1e-18, rtol=1e-14)
        np.testing.assert_allclose(self.model.cholesky @ self.model.cholesky.T,
                                   expected, atol=1e-18, rtol=1e-14)
        independent_columns = ar1_reference(np.eye(24)).T
        np.testing.assert_allclose(self.model.cholesky, independent_columns, atol=1e-17, rtol=1e-14)
        self.assertEqual(self.model.sigma_load, .03)
        self.assertEqual(self.model.rho_load, .8)

    def test_whole_vector_rejection_never_clips_or_keeps_valid_coordinates(self):
        rng = ProposalSequence([[.2, .011], [.022, .033]])
        z, epsilon, count = reject_multivariate_normal(rng, np.eye(2), -.09, .09)
        self.assertEqual(count, 2)
        self.assertEqual(rng.calls, 2)
        np.testing.assert_array_equal(z, [.022, .033])
        np.testing.assert_array_equal(epsilon, [.022, .033])
        self.assertNotIn(.011, epsilon)
        with self.assertRaises(ValueError):
            epsilon[0] = 0.

    def test_rejection_checks_correlated_error_not_standard_innovation(self):
        matrix = np.array([[.03, 0.], [.024, .018]])
        rng = ProposalSequence([[0., 6.], [2., -2.]])
        z, epsilon, count = reject_multivariate_normal(rng, matrix, -.09, .09)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(z, [2., -2.])
        np.testing.assert_allclose(epsilon, [.06, .012], rtol=0, atol=1e-16)

    def test_exact_stream_reference_rejections_counts_and_nonorder_dependence(self):
        expected_stream = stream('planning', 5, 0, 'load_innovation', development=True)
        rng = expected_stream.generator()
        count = 0
        while True:
            count += 1
            z = rng.standard_normal(24)
            eps = ar1_reference(z)
            if np.all(np.abs(eps) <= .09):
                break
        draw = self.draw()
        self.assertEqual(draw.stream_id, expected_stream.stream_id)
        self.assertEqual(draw.proposal_count, count)
        self.assertEqual(draw.rejection_count, count - 1)
        np.testing.assert_array_equal(draw.accepted_standard_innovation, z)
        np.testing.assert_allclose(draw.epsilon, eps, atol=2e-17, rtol=1e-14)
        first = {i: self.draw(i).epsilon for i in range(20)}
        for i in reversed(range(20)):
            np.testing.assert_array_equal(self.draw(i).epsilon, first[i])
        self.assertFalse(np.array_equal(draw.epsilon, self.draw(split='calibration').epsilon))
        self.assertNotEqual(draw.stream_id,
            stream('planning', 5, 0, 'line_gaussian', development=True).stream_id)
        self.assertNotEqual(draw.stream_id,
            stream('planning', 5, 0, 'road_uniform', development=True).stream_id)

    def test_all_bus_common_pq_factor_no_repeat_point8_scaling_and_zero_buses(self):
        d = self.draw()
        s = realize_load(self.model, d)
        factor = 1 + d.epsilon[:, None]
        np.testing.assert_array_equal(s.p_kw, self.model.baseline_p_kw * factor)
        np.testing.assert_array_equal(s.q_kvar, self.model.baseline_q_kvar * factor)
        positive = self.model.baseline_p_kw > 0
        np.testing.assert_allclose(s.q_kvar[positive] / s.p_kw[positive],
            np.broadcast_to(self.model.tan_phi_load, s.p_kw.shape)[positive], atol=2e-16, rtol=1e-14)
        np.testing.assert_array_equal(s.archived_q_kvar, self.model.archived_baseline_q_kvar * factor)
        np.testing.assert_array_equal(s.p_kw[:, 0], np.zeros(24))
        np.testing.assert_array_equal(s.q_kvar[:, 0], np.zeros(24))
        self.assertTrue(np.all(s.p_kw >= .91 * self.model.baseline_p_kw))
        self.assertTrue(np.all(s.p_kw <= 1.09 * self.model.baseline_p_kw))

    def test_phi_load_is_weighted_total_ratio_and_may_exceed_one(self):
        found_above = False
        nonuniform = False
        for i in range(30):
            d = self.draw(i)
            s = realize_load(self.model, d)
            weights = self.model.baseline_p_kw.sum(axis=1)
            reference = 1 + float(weights @ d.epsilon / weights.sum())
            self.assertAlmostEqual(s.phi_load, reference, delta=5e-15)
            self.assertAlmostEqual(s.phi_load, float(s.p_kw.sum() / self.model.baseline_p_kw.sum()), delta=5e-15)
            self.assertTrue(.91 <= s.phi_load <= 1.09)
            found_above |= s.phi_load > 1
            nonuniform |= abs(s.phi_load - (1 + d.epsilon.mean())) > 1e-5
        self.assertTrue(found_above, 'Phi_load must not be clipped to one')
        self.assertTrue(nonuniform, 'Use load-weighted error, not simple mean')

    def test_distribution_against_independent_conditioned_ar_proposal_reference(self):
        # Conditional covariance differs from the pre-truncation covariance.
        # A separate AR proposal simulator supplies the reference distribution.
        rng = np.random.default_rng(981507)
        proposals = ar1_reference(rng.standard_normal((60000, 24)))
        reference = proposals[np.all(np.abs(proposals) <= .09, axis=1)]
        draws = [self.draw(i) for i in range(5000)]
        samples = np.array([d.epsilon for d in draws])
        self.assertTrue(np.all(np.abs(samples) <= .09))
        self.assertFalse(np.any(np.abs(samples) == .09))
        np.testing.assert_allclose(samples.mean(axis=0), reference.mean(axis=0), atol=.002, rtol=0)
        np.testing.assert_allclose(np.cov(samples, rowvar=False),
                                   np.cov(reference, rowvar=False), atol=.000075, rtol=0)
        observed_acceptance = len(draws) / sum(d.proposal_count for d in draws)
        self.assertAlmostEqual(observed_acceptance, len(reference) / len(proposals), delta=.014)
        self.assertGreater(sum(d.rejection_count for d in draws), 0)
        correlation = np.corrcoef(samples, rowvar=False)
        self.assertTrue(.7 < correlation[0, 1] < .9)
        self.assertLess(abs(correlation[0, 12]), .16)

    def test_immutable_source_draw_and_realization_payloads(self):
        d = self.draw()
        s = realize_load(self.model, d)
        for array in (self.model.baseline_p_kw, self.model.baseline_q_kvar,
                      self.model.archived_baseline_q_kvar, self.model.tan_phi_load,
                      self.model.covariance, self.model.cholesky, self.model.hours_h,
                      d.epsilon, d.accepted_standard_innovation, s.p_kw, s.q_kvar):
            self.assertFalse(array.flags.writeable)
        self.assertTrue(self.model.input_hashes)
        self.assertTrue(all(len(h) == 64 for h in self.model.input_hashes.values()))
        self.assertEqual(s.draw_sha256, d.draw_sha256)
        self.assertEqual(s.model_sha256, self.model.model_sha256)

    def test_formal_oos_guard_formal_bank_boundaries_and_foreign_model(self):
        with self.assertRaises(ValueError):
            draw_load_errors(self.model, 5, 0, split='oos')
        oos_dev = self.draw(split='oos')
        self.assertEqual(realize_load(self.model, oos_dev).p_kw.shape, (24, 33))
        with self.assertRaises(ValueError):
            realize_load(self.model, replace(oos_dev, development=False))
        for split, limit in [('planning', 20), ('calibration', 10)]:
            self.assertEqual(draw_load_errors(self.model, 7, limit - 1, split=split).epsilon.shape, (24,))
            with self.assertRaises(ValueError):
                draw_load_errors(self.model, 7, limit, split=split)
        with self.assertRaises(ValueError):
            realize_load(self.model, replace(self.draw(), model_sha256='0' * 64))
        for base, variant in [(8, 0), (-1, 0), (0, -1), (True, 0), (0, .5)]:
            with self.assertRaises(ValueError):
                draw_load_errors(self.model, base, variant, development=True)

    def test_invalid_gaussian_parameters_and_invalid_realized_draws(self):
        for kwargs in ({'sigma_load': 0.}, {'sigma_load': float('nan')},
                       {'rho_load': 1.}, {'rho_load': -.1}, {'hours': 0}):
            with self.assertRaises(ValueError):
                covariance_matrix(**kwargs)
        for bad in (np.full(24, .091), np.full(24, np.nan), np.zeros(23)):
            with self.assertRaises(ValueError):
                realize_load(self.model, replace(self.draw(), epsilon=bad))


if __name__ == '__main__':
    unittest.main()
