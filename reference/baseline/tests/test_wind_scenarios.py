"""Independent numerical acceptance of the frozen S01 wind/residual contract.

All sampled weather uses the development namespace. Native-field references are
computed directly from the supplied arrays, rather than the wind implementation.
"""
from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths

from evcs_experiments.wind_field import latlon_to_xy
from evcs_experiments.wind_scenarios import (
    draw_parameters, generator_contract, load_wind_bank, resample_residual_blocks,
)

CASE_ROOT = get_paths().data_root


class WindScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = load_wind_bank(CASE_ROOT)
        cls.events = cls.bank.event_ids
        with np.load(CASE_ROOT / "07_weather/xiW_base__registered_wind_8events_25states.npz",
                     allow_pickle=False) as archive:
            cls.raw = {key: archive[key] for key in archive.files}

    def parameters(self, variant=0, **kwargs):
        return draw_parameters(self.events, self.events[0], self.events[1],
                               variant, development=True, **kwargs)

    def test_native_reference_no_error_and_true_residual(self):
        self.assertEqual(self.events, tuple(sorted(self.raw["event_ids"].tolist())))
        with (CASE_ROOT / "07_weather/base_event__spatial_registration.csv").open() as stream:
            caps = {r["event_id"]: float(r["source_peak_gust_m_s"])
                    for r in csv.DictReader(stream)}
        cells = [(3, 4), (10, 10), (17, 16)]
        points = latlon_to_xy([[self.raw["latitude"][y], self.raw["longitude"][x]]
                              for y, x in cells])
        times = [-2, -1, 0, 4.5, 23, 23.75]
        for event in self.events:
            i = self.raw["event_ids"].tolist().index(event)
            vectors = {}
            for prefix in ("forecast", "analysis"):
                u, v, g = [self.raw[f"{prefix}_{name}_grid"][i].astype(float)
                           for name in ("u10", "v10", "gust")]
                vectors[prefix] = np.stack((u, v), axis=-1) * (g / np.hypot(u, v))[..., None]
            np.testing.assert_allclose(self.bank.fields[event].vectors, vectors["forecast"],
                                       rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(self.bank.residuals[event],
                                       vectors["analysis"] - vectors["forecast"],
                                       rtol=1e-12, atol=1e-13)
            p = replace(draw_parameters(self.events, self.events[0], event, 0,
                                        development=True), amplitude=1., offset_xy_km=(0., 0.),
                        lag_h=0., error_scale=0.)
            expected = np.empty((len(times), len(cells), 2))
            for t, time in enumerate(times):
                hour = float(np.clip(time, -1, 23))
                left = min(int(math.floor(hour)) + 1, 24)
                right = min(left + 1, 24)
                fraction = hour - math.floor(hour)
                for a, (y, x) in enumerate(cells):
                    expected[t, a] = ((1 - fraction) * vectors["forecast"][left, y, x]
                                      + fraction * vectors["forecast"][right, y, x])
            sample = self.bank.sample(p, times, points)
            np.testing.assert_allclose(sample.vectors, expected, rtol=1e-11, atol=1e-11)
            self.assertEqual(sample.cap_m_s, caps[event])
            self.assertEqual(len(sample.parameters_sha256), 64)
        self.assertTrue(self.bank.input_hashes)
        self.assertTrue(all(len(value) == 64 for value in self.bank.input_hashes.values()))
        self.assertIsInstance(generator_contract(), dict)

    def test_blocks_preserve_whole_grid_and_linear_join(self):
        hour, y, x, component = np.indices((25, 21, 21, 2))
        residual = hour * 10000. + y * 100. + x + component / 10.
        starts = (0, 22, 1, 21, 2, 20, 3, 19, 4)
        indices = np.array([j for start in starts for j in range(start, start + 3)][:25])
        actual = resample_residual_blocks(residual, starts)
        np.testing.assert_array_equal(actual, residual[indices])
        # The same reindexing must apply to every grid point and both components.
        np.testing.assert_allclose(actual[:, 20, 19, 1] - actual[:, 0, 0, 0],
                                   np.full(25, 2019.1), rtol=0, atol=1e-10)
        p = replace(self.parameters(), block_start_indices=starts)
        field = self.bank.field(p)
        expected = (self.bank.fields[p.base_event].vectors
                    + p.error_scale * self.bank.residuals[p.donor_event][indices])
        np.testing.assert_allclose(field.vectors, expected, rtol=0, atol=1e-13)
        # The first seam connects knots at hour 1 and 2, with no reset jump.
        values = field.sample([1., 1.5, 2., 2. - 1e-8, 2. + 1e-8], [[0., 0.]])
        np.testing.assert_allclose(values[1], (values[0] + values[2]) / 2,
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(values[3], values[4], rtol=0, atol=1e-5)

    def test_frozen_loeo_crn_split_separation_and_lead_invariance(self):
        primitive_fields = ("amplitude", "offset_xy_km", "lag_h", "error_scale",
                            "block_start_indices")
        for variant in range(12):
            original = self.parameters(variant)
            for target in self.events:
                p = draw_parameters(tuple(reversed(self.events)), target, self.events[1],
                                    variant, development=True)
                self.assertNotEqual(p.donor_event, target)
                self.assertTrue(all(getattr(p, k) == getattr(original, k)
                                    for k in primitive_fields))
                repeated = draw_parameters(self.events, target, self.events[1], variant,
                                           development=True)
                self.assertEqual(p.as_dict(), repeated.as_dict())
            with self.assertRaises((FrozenInstanceError, AttributeError)):
                original.amplitude = .99
        p = self.parameters()
        calibration = self.parameters(split="calibration")
        self.assertNotEqual(tuple(getattr(p, k) for k in primitive_fields),
                            tuple(getattr(calibration, k) for k in primitive_fields))
        full_times = np.arange(-2., 24., .25)
        full = self.bank.sample(p, full_times, [[0., 0.], [1., -1.]])
        for lead in (0., .5, 1., 2.):
            mask = full_times >= -lead
            shorter = self.bank.sample(p, full_times[mask], [[0., 0.], [1., -1.]])
            np.testing.assert_array_equal(shorter.vectors, full.vectors[mask])
            self.assertEqual(shorter.parameters_sha256, full.parameters_sha256)

    def test_common_regimes_cap_and_exact_tail_factor(self):
        times = np.arange(-2., 24., .25)
        points = [[-3.9, 2.7], [0., 0.], [3.8, -2.7]]
        for base in self.events:
            p = draw_parameters(self.events, self.events[0], base, 3, development=True)
            r1 = self.bank.sample(p, times, points, regime="R1")
            r2 = self.bank.sample(p, times, points, regime="R2")
            r3 = self.bank.sample(p, times, points, regime="R3")
            np.testing.assert_array_equal(r1.vectors, r3.vectors)
            np.testing.assert_array_equal(r1.hazard, r3.hazard)
            np.testing.assert_allclose(r2.speed, np.minimum(1.15 * r1.speed, r1.cap_m_s),
                                       rtol=1e-12, atol=1e-12)
            for result in (r1, r2, r3):
                np.testing.assert_allclose(result.speed, np.linalg.norm(result.vectors, axis=-1),
                                           rtol=1e-13, atol=1e-13)
                self.assertTrue(np.all(result.speed <= result.cap_m_s + 1e-12))
                self.assertTrue(np.all((result.hazard >= 0) & (result.hazard <= 1)))
            self.assertTrue(np.all(r2.hazard >= r1.hazard - 1e-13))

    def test_development_distribution_bounds_and_frozen_moments(self):
        draws = [self.parameters(v) for v in range(2000)]
        amplitudes = np.array([p.amplitude for p in draws])
        locations = np.array([p.offset_xy_km for p in draws])
        lags = np.array([p.lag_h for p in draws])
        scales = np.array([p.error_scale for p in draws])
        starts = np.array([p.block_start_indices for p in draws])
        self.assertTrue(np.all((amplitudes > .85) & (amplitudes < 1.2)))
        self.assertTrue(np.all(np.linalg.norm(locations, axis=1) <= 10))
        self.assertTrue(np.all((scales >= .75) & (scales <= 1.25)))
        self.assertEqual(starts.shape, (2000, 9))
        self.assertTrue(np.issubdtype(starts.dtype, np.integer))
        self.assertTrue(np.all((starts >= 0) & (starts <= 22)))
        self.assertEqual(set(lags), {-2, -1, 0, 1, 2})
        # Fixed, broad finite-sample thresholds; no retrying seeds to obtain PASS.
        self.assertLess(abs(amplitudes.mean() - 1.0043), .01)
        self.assertLess(abs(scales.mean() - 1.), .025)
        self.assertTrue(np.all(abs(locations.mean(axis=0)) < .35))
        a = 100. / 32.
        expected_radius_squared = 32. * (1. - a / math.expm1(a))
        self.assertLess(abs(np.mean(np.sum(locations ** 2, axis=1))
                            - expected_radius_squared), 2.)
        for lag, probability in zip((-2, -1, 0, 1, 2), (.1, .2, .4, .2, .1)):
            self.assertLess(abs(np.mean(lags == lag) - probability), .035)

    def test_formal_oos_rejected_before_any_random_draw(self):
        with patch("evcs_experiments.random_streams.Stream.generator",
                   side_effect=AssertionError("Formal OOS must not materialize random draws")):
            with self.assertRaises(ValueError):
                draw_parameters(self.events, self.events[0], self.events[1], 0,
                                split="oos", development=False)
        development = self.parameters(split="oos")
        self.assertTrue(development.development)
        self.assertEqual(development.split, "oos")

    def test_invalid_addresses_residuals_and_queries_rejected(self):
        for changes in ({"target_event": "unknown"}, {"base_event": "unknown"},
                        {"variant_index": -1}, {"variant_index": True}, {"split": "bad"}):
            args = dict(event_ids=self.events, target_event=self.events[0],
                        base_event=self.events[1], variant_index=0, development=True)
            args.update(changes)
            with self.assertRaises(ValueError):
                draw_parameters(**args)
        residual = np.zeros((25, 21, 21, 2))
        for starts in ((0,) * 8, (23,) * 9, (-1,) * 9, (.5,) * 9):
            with self.assertRaises(ValueError):
                resample_residual_blocks(residual, starts)
        for invalid in (np.zeros((24, 21, 21, 2)), np.full(residual.shape, np.nan)):
            with self.assertRaises(ValueError):
                resample_residual_blocks(invalid, (0,) * 9)
        p = self.parameters()
        for times, points, kwargs in (([np.nan], [[0., 0.]], {}),
                                      ([0.], [0., 0.], {}),
                                      ([0.], [[0., np.inf]], {}),
                                      ([0.], [[0., 0.]], {"regime": "bad"}),
                                      ([0.], [[0., 0.]], {"kappa_W": -1.})):
            with self.assertRaises(ValueError):
                self.bank.sample(p, times, points, **kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
