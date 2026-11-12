"""Independent road-kernel checks from the manuscript equations and canonical CSVs."""
from __future__ import annotations

import csv
from dataclasses import replace
import math
import unittest

import numpy as np

from evcs_experiments.config import get_paths

from evcs_experiments.road_disruptions import (
    load_road_model, road_probabilities, persistent_blockages, road_travel_times,
)
from evcs_experiments.line_outages import load_line_model
from evcs_experiments.random_streams import stream

CASE_ROOT = get_paths().data_root
EVENTS = ("helene2024", "ian2022", "ida2021", "idalia2023", "laura2020",
          "michael2018", "milton2024", "zeta2020")


class RoadDisruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_road_model(CASE_ROOT)
        cls.line_model = load_line_model(CASE_ROOT)
        cls.times = np.arange(-2., 24., .25)

    def thresholds(self, variant=0, **kwargs):
        return self.model.draw_thresholds(EVENTS, "michael2018", variant,
                                         development=True, **kwargs)

    def lines(self, lead=1., states=None):
        threshold = self.line_model.draw_thresholds(EVENTS, "michael2018", 0,
                                                   development=True)
        threshold = replace(threshold, uniforms=np.ones(32))
        replay = self.line_model.replay(self.times, np.zeros((104, 32, 2)),
                                        threshold, lead_h=lead)
        return replay if states is None else replace(replay, outages=np.asarray(states, dtype=bool))

    def force_uniform(self, value):
        return replace(self.thresholds(), corridor_uniforms=np.full(40, value),
                       arc_uniforms=np.full(80, value))

    def test_canonical_assets_parameters_and_coupling_already_include_vulnerability(self):
        m = self.model
        self.assertEqual(m.arc_ids, tuple(range(1, 81)))
        self.assertEqual(m.corridor_ids, tuple(range(1, 41)))
        self.assertEqual(m.line_ids, tuple(range(1, 33)))
        with (CASE_ROOT / "04_transportation/A__directed_road_arcs.csv").open() as f:
            rows = list(csv.DictReader(f))
        with (CASE_ROOT / "08_grid_road_coupling/L_A_gamma_blk__line_road_pairs.csv").open() as f:
            pairs = list(csv.DictReader(f))
        expected = np.zeros((32, 80))
        self.assertEqual(len(pairs), 112)
        self.assertEqual(len({(r["ell"], r["corridor_id"]) for r in pairs}), 56)
        for r in pairs:
            self.assertIn(int(r["ell"]), range(1, 33))
            expected[int(r["ell"]) - 1, int(r["a"]) - 1] = float(r["gamma_blk_ell_a"])
        np.testing.assert_array_equal(m.gamma, expected)
        self.assertTrue(np.any(m.gamma == .42))  # Stored 1.2 vulnerable multiplier is final.
        for attr, col in (("t0_min", "T0_a_min"), ("kappa_tt", "kappa_tt_a"),
                          ("w_mov_m_s", "W_mov_type_a_m_s"),
                          ("kappa_blk", "kappa_blk_type_a"),
                          ("alpha0", "alpha_R_type_a_0"), ("alpha1", "alpha_R_type_a_1")):
            np.testing.assert_array_equal(getattr(m, attr), [float(r[col]) for r in rows])
        np.testing.assert_array_equal(m.arc_corridor_indices,
                                     [int(r["corridor_id"]) - 1 for r in rows])
        np.testing.assert_array_equal(m.points_xy_km,
            [[float(r["loc_a_x_km"]), float(r["loc_a_y_km"])] for r in rows])
        with self.assertRaises(ValueError):
            m.gamma[0, 0] = 1.

    def test_hand_calculated_direct_wind_and_product_coupling_once(self):
        speed = np.array([[0., 1., 2.], [2., 1., 0.]])
        kappa = np.array([.2, .35, .5])
        a0, a1 = np.array([-1., 0., 1.]), np.array([1., .5, .25])
        gamma = np.array([[.1, .2, .3], [.4, 0., .45]])
        states = np.array([[1, 0], [1, 1]])
        actual = road_probabilities(speed, kappa, a0, a1, gamma, states)
        direct = kappa / (1 + np.exp(-(a0 + a1 * speed)))
        expected = np.array([[1 - (1 - direct[0, j]) * (1 - gamma[0, j])
                              for j in range(3)],
                             [1 - (1 - direct[1, j]) * (1 - gamma[0, j]) * (1 - gamma[1, j])
                              for j in range(3)]])
        np.testing.assert_allclose(actual.wind, direct, rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(actual.combined, expected, rtol=1e-14, atol=1e-16)
        absent = road_probabilities(speed, kappa, a0, a1, gamma, np.zeros((2, 2)))
        np.testing.assert_allclose(absent.combined, direct, atol=1e-16)
        zero = road_probabilities(speed, kappa, a0, a1, np.zeros_like(gamma), states)
        np.testing.assert_allclose(zero.combined, direct, atol=1e-16)

    def test_persistent_max_and_inclusive_threshold_without_slot_hazard_accumulation(self):
        probability = np.array([[.1, .3], [.5, .2], [.2, .7], [.1, .1]])
        cumulative, blocked = persistent_blockages(probability, [.5, .6])
        np.testing.assert_array_equal(cumulative, [[.1, .3], [.5, .3], [.5, .7], [.5, .7]])
        np.testing.assert_array_equal(blocked, [[0, 0], [1, 0], [1, 1], [1, 1]])
        _, constant = persistent_blockages(np.full((104, 1), .1), [.2])
        self.assertFalse(constant.any())

    def test_driving_safety_equality_passes_and_wind_only_closure_recovers(self):
        speed = np.array([[25., 30.], [25.001, 29.], [24., 20.]])
        hazard = np.clip((speed - 10.) / 15., 0., 1.)
        blocked = np.array([[0, 0], [0, 1], [0, 1]])
        movable, travel = road_travel_times(speed, hazard, blocked,
                                            [10., 20.], [.4, .2], [25., 30.])
        np.testing.assert_array_equal(movable, [[1, 1], [0, 0], [1, 0]])
        self.assertEqual(travel[0, 0], 14.)
        self.assertEqual(travel[0, 1], 24.)
        self.assertAlmostEqual(travel[2, 0], 10 * (1 + .4 * 14 / 15))
        self.assertTrue(np.isinf(travel[~movable]).all())
        self.assertFalse(blocked[:, 0].any())
        with self.assertRaises(ValueError):
            road_travel_times([[0.]], [[0.]], [[0]], [1.], [.2], [0.])

    def test_corridor_threshold_crn_identity_direction_sharing_and_colocated_separation(self):
        t = self.thresholds()
        streams = [stream("planning", 5, 0, "road_uniform", development=True,
                          asset_index=c) for c in self.model.corridor_ids]
        expected = [s.generator().uniform() for s in streams]
        np.testing.assert_array_equal(t.corridor_uniforms, expected)
        self.assertEqual(tuple(t.stream_ids), tuple(s.stream_id for s in streams))
        np.testing.assert_array_equal(t.arc_uniforms, np.array(expected)[self.model.arc_corridor_indices])
        np.testing.assert_array_equal(t.arc_uniforms[::2], t.arc_uniforms[1::2])
        other = self.model.draw_thresholds(tuple(reversed(EVENTS)), "michael2018", 0,
                                          development=True)
        np.testing.assert_array_equal(t.arc_uniforms, other.arc_uniforms)
        points = self.model.points_xy_km[::2]
        colocated = [(i, j) for i in range(40) for j in range(i + 1, 40)
                     if np.array_equal(points[i], points[j])]
        self.assertTrue(colocated)
        for i, j in colocated:
            self.assertNotEqual(t.corridor_uniforms[i], t.corridor_uniforms[j])
            self.assertNotEqual(t.stream_ids[i], t.stream_ids[j])
        self.assertFalse(np.array_equal(t.arc_uniforms, self.thresholds(split="calibration").arc_uniforms))
        with self.assertRaises(ValueError):
            t.arc_uniforms[0] = 0.

    def test_empirical_uniform_marginals_and_conditional_residual_independence(self):
        values = np.array([self.thresholds(v).corridor_uniforms for v in range(2000)])
        np.testing.assert_allclose(values.mean(axis=0), .5, rtol=0, atol=.027)
        np.testing.assert_allclose((values <= .2).mean(axis=0), .2, rtol=0, atol=.031)
        for i, j in ((0, 1), (0, 39), (10, 20)):
            self.assertAlmostEqual(np.mean((values[:, i] <= .5) & (values[:, j] <= .5)),
                                   .25, delta=.03)
            self.assertAlmostEqual(np.cov(values[:, i], values[:, j])[0, 1], 0., delta=.008)

    def test_assignment_reset_and_all_lead_grids_preserve_primitive_threshold(self):
        wind = np.zeros((104, 80, 2))
        wind[0, :, 0] = 1000.
        t = self.force_uniform(.1)
        long = self.model.replay(self.times, wind, t, self.lines(2.), lead_h=2.)
        short = self.model.replay(self.times, wind, t, self.lines(1.), lead_h=1.)
        self.assertTrue(long.blocked.all())
        self.assertFalse(short.blocked.any())
        for lead in (0., .5, 1., 2.):
            replay = self.model.replay(self.times, wind, t, self.lines(lead), lead_h=lead)
            self.assertEqual(replay.times_h[0], -lead)
            self.assertEqual(replay.blocked.shape, (int((24 + lead) * 4), 80))
            np.testing.assert_array_equal(replay.thresholds.arc_uniforms, t.arc_uniforms)
            np.testing.assert_array_equal(replay.blocked[:, ::2], replay.blocked[:, 1::2])

    def test_coupling_uses_same_quarter_hour_line_state_without_hourly_lookahead(self):
        times = self.times[self.times >= -1]
        states = np.zeros((len(times), 32), dtype=bool)
        states[times >= .75, 0] = True
        lines = self.lines(states=states)
        t = self.force_uniform(.1)
        replay = self.model.replay(self.times, np.zeros((104, 80, 2)), t, lines)
        coupled = self.model.gamma[0] > 0
        self.assertTrue(coupled.any())
        self.assertFalse(replay.blocked[replay.times_h == .5][:, coupled].any())
        self.assertTrue(replay.blocked[replay.times_h >= .75][:, coupled].all())
        expected = 1 - (1 - replay.wind_probability) * np.prod(
            1 - states[:, :, None] * self.model.gamma[None], axis=1)
        np.testing.assert_allclose(replay.blockage_probability, expected, atol=1e-16)

    def test_r3_effective_parameters_and_no_coupling_share_resistance(self):
        wind = np.zeros((104, 80, 2)); wind[:, :, 0] = 25.
        lines = self.lines(states=np.ones((100, 32), dtype=bool))
        t = self.force_uniform(1.)
        r1 = self.model.replay(self.times, wind, t, lines, regime="R1")
        r3 = self.model.replay(self.times, wind, t, lines, regime="R3")
        zero = self.model.replay(self.times, wind, t, lines, regime="R3", no_coupling=True)
        np.testing.assert_array_equal(r3.effective_gamma, np.minimum(1.25 * self.model.gamma, .45))
        np.testing.assert_array_equal(r3.effective_kappa_tt, 1.25 * self.model.kappa_tt)
        self.assertGreater(np.max(r3.effective_kappa_tt), .43)
        np.testing.assert_allclose(r3.max_passable_travel_time_min,
                                   self.model.t0_min * (1 + 1.25 * self.model.kappa_tt), rtol=1e-14)
        self.assertTrue(np.all(r3.travel_time_min >= r1.travel_time_min))
        self.assertTrue(np.all(r3.blockage_probability >= r1.blockage_probability - 1e-16))
        np.testing.assert_array_equal(zero.effective_gamma, np.zeros((32, 80)))
        np.testing.assert_allclose(zero.blockage_probability, zero.wind_probability, atol=1e-16)
        for result in (r1, r3, zero):
            np.testing.assert_array_equal(result.thresholds.arc_uniforms, t.arc_uniforms)
        np.testing.assert_array_equal(zero.effective_kappa_tt, r3.effective_kappa_tt)

    def test_connector_expansion_preserves_canonical_assets_random_streams_and_replay(self):
        full = load_road_model(CASE_ROOT, include_connectors=True)
        self.assertEqual(full.arc_ids, tuple(range(1, 89)))
        self.assertEqual(full.corridor_ids, tuple(range(1, 45)))
        for attr in ("t0_min", "kappa_tt", "w_mov_m_s", "kappa_blk", "alpha0", "alpha1",
                     "points_xy_km", "arc_corridor_indices"):
            np.testing.assert_array_equal(getattr(full, attr)[:80], getattr(self.model, attr))
        np.testing.assert_array_equal(full.gamma[:, :80], self.model.gamma)
        full_t = full.draw_thresholds(EVENTS, "michael2018", 0, development=True)
        t = self.thresholds()
        np.testing.assert_array_equal(full_t.corridor_uniforms[:40], t.corridor_uniforms)
        np.testing.assert_array_equal(full_t.arc_uniforms[:80], t.arc_uniforms)
        self.assertEqual(full_t.stream_ids[:40], t.stream_ids)
        self.assertEqual(len(set(full_t.stream_ids)), 44)
        wind = np.zeros((104, 88, 2)); wind[:, :, 0] = 20.
        lines = self.lines()
        for regime in ("R1", "R3"):
            base_result = self.model.replay(self.times, wind[:, :80], t, lines, regime=regime)
            full_result = full.replay(self.times, wind, full_t, lines, regime=regime)
            for attr in ("wind_probability", "blockage_probability", "blocked", "movable",
                         "travel_time_min", "cumulative_blockage_probability"):
                np.testing.assert_array_equal(getattr(full_result, attr)[:, :80],
                                             getattr(base_result, attr))
            np.testing.assert_array_equal(full_result.blocked[:, ::2], full_result.blocked[:, 1::2])

    def test_formal_oos_lock_and_replay_input_validation(self):
        with self.assertRaises(ValueError):
            self.model.draw_thresholds(EVENTS, "michael2018", 0, split="oos")
        dev_oos = self.thresholds(split="oos")
        self.assertEqual(dev_oos.arc_uniforms.shape, (80,))
        wind = np.zeros((104, 80, 2))
        for threshold in (replace(dev_oos, development=False),
                          replace(self.thresholds(), model_sha256="0" * 64)):
            with self.assertRaises(ValueError):
                self.model.replay(self.times, wind, threshold, self.lines())
        with self.assertRaises(ValueError):
            self.model.replay(self.times, wind, self.thresholds(), self.lines(2.))
        with self.assertRaises(ValueError):
            self.model.replay(self.times, wind, self.thresholds(),
                               replace(self.lines(), line_ids=tuple(range(6, 38))))
        with self.assertRaises(ValueError):
            self.model.replay(self.times, np.full((104, 80, 2), math.nan),
                               self.thresholds(), self.lines())
        for q, u in (([[1.1]], [.5]), ([[math.nan]], [.5]), ([[.2]], [-.1])):
            with self.assertRaises(ValueError):
                persistent_blockages(q, u)


if __name__ == "__main__":
    unittest.main()
