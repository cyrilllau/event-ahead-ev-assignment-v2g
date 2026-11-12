"""Independent acceptance of line fragility, copula and replay time conventions.

Reference formulas below are taken from the manuscript; tests use the development
namespace and include analytic Gaussian-copula probabilities at median thresholds.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import math
import unittest

import numpy as np

from evcs_experiments.config import get_paths

from evcs_experiments.line_outages import (
    correlation_matrix, crosswind_components, failure_probabilities,
    load_line_model, operating_outages, persistent_outages,
)
from evcs_experiments.random_streams import stream

CASE_ROOT = get_paths().data_root
EVENTS = ("helene2024", "ian2022", "ida2021", "idalia2023", "laura2020",
          "michael2018", "milton2024", "zeta2020")
COEFFICIENTS = {
    2: ((-10.289, .0529, .0677), (-6.843, .0345, .0388)),
    4: ((-9.313, .0494, .0776), (-6.037, .0219, .0471)),
    5: ((-8.884, .0468, .0821), (-6.688, .0260, .0476)),
}


class LineOutageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_line_model(CASE_ROOT)
        cls.times = np.arange(-2., 24., .25)

    def thresholds(self, variant=0, **kwargs):
        return self.model.draw_thresholds(EVENTS, "michael2018", variant,
                                         development=True, **kwargs)

    def test_exact_case_assets_coefficients_and_geometry(self):
        m = self.model
        self.assertEqual(m.line_ids, tuple(range(1, 33)))
        self.assertEqual(m.tie_ids, tuple(range(33, 38)))
        with (CASE_ROOT / "07_weather/loc_L_loc_A__weather_sampling_points.csv").open() as f:
            rows = {int(r["index_id"]): r for r in csv.DictReader(f)
                    if r["index_type"] == "ell" and r["hazard_eligible_ell"] == "1"}
        np.testing.assert_array_equal(m.points_xy_km,
            [[float(rows[i]["loc_x_km"]), float(rows[i]["loc_y_km"])] for i in m.line_ids])
        np.testing.assert_array_equal(m.azimuth_deg_from_north,
            [float(rows[i]["phi_deg_from_north"]) for i in m.line_ids])
        np.testing.assert_array_equal(m.pole_coefficients, np.tile(COEFFICIENTS[4][0], (32, 1)))
        np.testing.assert_array_equal(m.wire_coefficients, np.tile(COEFFICIENTS[4][1], (32, 1)))
        self.assertTrue(m.input_hashes)
        self.assertTrue(all(len(s) == 64 for s in m.input_hashes.values()))
        with self.assertRaises(ValueError):
            m.points_xy_km[0, 0] = 0.

    def test_covariance_distance_units_diagonal_and_duplicate_assets(self):
        pts = np.array([[0., 0.], [4., 0.], [0., 0.]])
        expected = [[1., .95 / math.e, .95], [.95 / math.e, 1., .95 / math.e],
                    [.95, .95 / math.e, 1.]]
        np.testing.assert_allclose(correlation_matrix(pts), expected, rtol=1e-14, atol=1e-14)
        # Co-located lines remain separate asset resistances, with rho=.95, not 1.
        self.assertGreater(np.linalg.eigvalsh(correlation_matrix(pts))[0], .049999999)
        m = self.model
        distance = np.linalg.norm(m.points_xy_km[:, None] - m.points_xy_km[None], axis=-1)
        expected = .95 * np.exp(-distance / 4.) + .05 * np.eye(32)
        np.testing.assert_allclose(m.covariance, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(m.cholesky @ m.cholesky.T, expected, rtol=1e-13, atol=1e-13)

    def test_crosswind_north_clockwise_cardinals_direction_reversal(self):
        # Same east wind, line bearings north/east/south/west/NE.
        azimuth = np.array([0., 90., 180., 270., 45.])
        vectors = np.tile([3., 0.], (2, 5, 1))
        vectors[1] *= -1
        speed, cross = crosswind_components(vectors, azimuth)
        np.testing.assert_allclose(speed, np.full((2, 5), 3.), atol=1e-14)
        np.testing.assert_allclose(cross, np.tile([3., 0., 3., 0., 3 / math.sqrt(2)], (2, 1)), atol=1e-14)
        _, north_cross = crosswind_components(np.tile([0., 4.], (1, 5, 1)), azimuth)
        np.testing.assert_allclose(north_cross, [[0., 4., 0., 4., 4 / math.sqrt(2)]], atol=1e-14)
        _, zero_cross = crosswind_components(np.zeros((1, 5, 2)), azimuth)
        np.testing.assert_array_equal(zero_cross, np.zeros((1, 5)))

    def test_fragility_component_combination_and_paper_class_values(self):
        vectors = np.array([[[0., 0.], [30., 0.]], [[0., 40.], [30., 40.]]])
        azimuth = np.array([0., 90.])
        expected_speed = np.array([[0., 30.], [40., 50.]])
        expected_cross = np.array([[0., 0.], [0., 40.]])
        for pole_class, (pole, wire) in COEFFICIENTS.items():
            result = failure_probabilities(vectors, azimuth,
                                           np.tile(pole, (2, 1)), np.tile(wire, (2, 1)))
            expected = []
            for alpha in (pole, wire):
                linear = alpha[0] + alpha[1] * expected_speed + alpha[2] * expected_cross
                expected.append(1 / (1 + np.exp(-linear)))
            np.testing.assert_allclose(result.pole, expected[0], rtol=1e-13)
            np.testing.assert_allclose(result.wire, expected[1], rtol=1e-13)
            np.testing.assert_allclose(result.combined, 1 - (1 - expected[0]) * (1 - expected[1]), atol=2e-16)
        # Even zero wind has the small nonzero logistic probability specified by the paper.
        self.assertGreater(result.combined[0, 0], 0.)

    def test_persistent_threshold_not_independent_redraw_or_hazard_accumulation(self):
        probabilities = np.array([[.1, .2, .9], [.5, .1, .3], [.2, .8, .1], [.4, .7, .1]])
        uniforms = np.array([.5, .75, .95])
        cumulative, outages = persistent_outages(probabilities, uniforms)
        expected = [[.1, .2, .9], [.5, .2, .9], [.5, .8, .9], [.5, .8, .9]]
        np.testing.assert_array_equal(cumulative, expected)
        np.testing.assert_array_equal(outages,
                                      [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]])
        # Constant q over many slots must not become 1-(1-q)^T.
        repeated, repeated_outages = persistent_outages(np.full((104, 1), .1), [.2])
        np.testing.assert_array_equal(repeated, np.full((104, 1), .1))
        self.assertFalse(repeated_outages.any())

    def test_assignment_reset_excludes_prior_reference_damage(self):
        vectors = np.zeros((len(self.times), 32, 2))
        vectors[self.times == -2., :, 0] = 1000.
        threshold = replace(self.thresholds(), uniforms=np.full(32, .5))
        long = self.model.replay(self.times, vectors, threshold, lead_h=2.)
        short = self.model.replay(self.times, vectors, threshold, lead_h=1.)
        self.assertTrue(long.outages.all())
        self.assertFalse(short.outages.any())
        self.assertEqual(short.times_h[0], -1.)
        sliced = self.model.replay(self.times[4:], vectors[4:], threshold, lead_h=1.)
        np.testing.assert_array_equal(short.outages, sliced.outages)
        np.testing.assert_array_equal(short.failure_probability, sliced.failure_probability)
        for lead in (0., .5, 1., 2.):
            result = self.model.replay(self.times, vectors, threshold, lead_h=lead)
            self.assertEqual(result.times_h[0], -lead)
            self.assertEqual(result.outages.shape, (int((24 + lead) * 4), 32))
            np.testing.assert_array_equal(result.thresholds.uniforms, threshold.uniforms)

    def test_hourly_mapping_includes_last_slot_and_excludes_boundary_next_hour(self):
        state = np.zeros((len(self.times), 2), dtype=bool)
        state[self.times >= .75, 0] = True
        state[self.times >= 1., 1] = True
        actual = operating_outages(self.times, state)
        self.assertEqual(actual.shape, (24, 2))
        np.testing.assert_array_equal(actual[0], [True, False])
        self.assertTrue(actual[1:].all())
        vectors = np.zeros((len(self.times), 32, 2))
        vectors[self.times == .75, :, 0] = 1000.
        result = self.model.replay(self.times, vectors,
                                  replace(self.thresholds(), uniforms=np.full(32, .5)), lead_h=1.)
        self.assertFalse(result.outages[result.times_h == .5].any())
        self.assertTrue(result.outages[result.times_h >= .75].all())
        self.assertTrue(result.operating_outages.all())

    def test_crn_direct_stream_order_repeat_split_separation_and_immutability(self):
        threshold = self.thresholds()
        reverse = self.model.draw_thresholds(tuple(reversed(EVENTS)), "michael2018", 0,
                                             development=True)
        np.testing.assert_array_equal(threshold.uniforms, reverse.uniforms)
        independent_reference = stream("planning", 5, 0, "line_gaussian", development=True)
        expected_normals = independent_reference.generator().standard_normal(32)
        np.testing.assert_array_equal(threshold.standard_normals, expected_normals)
        self.assertEqual(threshold.stream_id, independent_reference.stream_id)
        expected_z = self.model.cholesky @ expected_normals
        np.testing.assert_allclose(threshold.correlated_normals, expected_z, rtol=0, atol=1e-15)
        expected_uniform = np.array([.5 * (1 + math.erf(z / math.sqrt(2))) for z in expected_z])
        np.testing.assert_allclose(threshold.uniforms, expected_uniform, rtol=0, atol=2e-16)
        self.assertFalse(np.array_equal(threshold.uniforms, self.thresholds(split="calibration").uniforms))
        with self.assertRaises(ValueError):
            threshold.uniforms[0] = .5

    def test_empirical_normal_covariance_uniform_marginals_and_copula_joint_probability(self):
        # Fixed development samples; compare to analytical probabilities independent of code.
        samples = [self.thresholds(v) for v in range(8000)]
        z = np.array([s.correlated_normals for s in samples])
        u = np.array([s.uniforms for s in samples])
        np.testing.assert_allclose(np.cov(z, rowvar=False), self.model.covariance, atol=.065, rtol=0)
        np.testing.assert_allclose(u.mean(axis=0), .5, atol=.014, rtol=0)
        np.testing.assert_allclose(u.var(axis=0), 1 / 12, atol=.005, rtol=0)
        for i, j in ((0, 1), (0, 16), (0, 31)):
            rho = self.model.covariance[i, j]
            expected_joint = .25 + math.asin(rho) / (2 * math.pi)
            joint = np.mean((u[:, i] <= .5) & (u[:, j] <= .5))
            self.assertAlmostEqual(joint, expected_joint, delta=.015)
        np.testing.assert_allclose((u <= .2).mean(axis=0), .2, atol=.018, rtol=0)

    def test_replay_class_sensitivity_exact_coefficients_and_finite_outputs(self):
        vectors = np.tile([30., 40.], (len(self.times), 32, 1))
        for pole_class, (pole, wire) in COEFFICIENTS.items():
            result = self.model.replay(self.times, vectors, self.thresholds(),
                                      lead_h=1., pole_class=pole_class)
            bearing = np.deg2rad(self.model.azimuth_deg_from_north)
            cross = np.abs(30 * np.cos(bearing) - 40 * np.sin(bearing))
            qp = 1 / (1 + np.exp(-(pole[0] + pole[1] * 50 + pole[2] * cross)))
            qw = 1 / (1 + np.exp(-(wire[0] + wire[1] * 50 + wire[2] * cross)))
            expected = 1 - (1 - qp) * (1 - qw)
            np.testing.assert_allclose(result.failure_probability,
                                       np.tile(expected, (100, 1)), rtol=1e-13, atol=2e-16)
            self.assertTrue(np.isfinite(result.failure_probability).all())

    def test_invalid_data_and_formal_oos_lock(self):
        with self.assertRaises(ValueError):
            self.model.draw_thresholds(EVENTS, "michael2018", 0, split="oos")
        self.assertEqual(self.thresholds(split="oos").uniforms.shape, (32,))
        forged_formal_oos = replace(self.thresholds(split="oos"), development=False)
        with self.assertRaises(ValueError):
            self.model.replay(self.times, np.zeros((104, 32, 2)), forged_formal_oos)
        foreign_model = replace(self.thresholds(), model_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.model.replay(self.times, np.zeros((104, 32, 2)), foreign_model)
        for q, u in (([[-.1]], [.5]), ([[1.1]], [.5]), ([[.2]], [math.nan]),
                     ([[math.nan]], [.5]), ([[.2]], [1.1])):
            with self.assertRaises(ValueError):
                persistent_outages(q, u)
        with self.assertRaises(ValueError):
            correlation_matrix([[math.nan, 0.]])
        with self.assertRaises(ValueError):
            correlation_matrix([[0., 0.]], d_corr_km=0.)
        with self.assertRaises(ValueError):
            operating_outages(self.times[self.times != .75], np.zeros((103, 32), dtype=bool))
        with self.assertRaises(ValueError):
            self.model.replay(self.times, np.zeros((104, 37, 2)), self.thresholds())
        with self.assertRaises(ValueError):
            self.model.replay(self.times, np.zeros((104, 32, 2)), self.thresholds(), pole_class=3)
        with self.assertRaises(ValueError):
            self.model.replay(self.times, np.full((104, 32, 2), math.nan), self.thresholds())


if __name__ == "__main__":
    unittest.main()
