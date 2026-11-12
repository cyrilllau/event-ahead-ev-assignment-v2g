"""Independent hand-calculated five-feature statistics and domain checks."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.scenario_features import prepare_feature_tasks, compute_features, native_weather_grid


def task_bank(empty=False):
    tasks = [] if empty else [
        {'task_id': 'a', 'vehicle_id': 'EV1', 'usable_energy_kwh': 10.},
        {'task_id': 'b', 'vehicle_id': 'EV1', 'usable_energy_kwh': 20.},
        {'task_id': 'c', 'vehicle_id': 'EV1', 'usable_energy_kwh': 30.},
        {'task_id': 'd', 'vehicle_id': 'EV2', 'usable_energy_kwh': 40.}]
    bank = dict(status='COMPLETE', certified=True, dependency_sha256='a'*64,
                config={'basic_soc': .3, 'tau_blk_h': 1}, tasks=tasks)
    bank['task_sha256'] = canonical_hash({k: bank[k] for k in
        ('dependency_sha256', 'config', 'certified', 'tasks')})
    return bank


class ScenarioFeatureTests(unittest.TestCase):
    def setUp(self):
        self.prepared = prepare_feature_tasks(task_bank())
        self.times = np.arange(-1., 24., .25)
        self.hazard = np.broadcast_to(np.arange(100)[:, None] / 100., (100, 441)).copy()
        self.lines = np.zeros((100, 32), dtype=bool)
        self.lines[:50] = True
        self.roads = np.zeros((100, 80), dtype=bool)
        self.roads[:, :20] = True
        self.base = np.ones((24, 33))

    def sample(self, **overrides):
        args = dict(grid_hazard=self.hazard, line_outages=self.lines, road_blocked=self.roads,
                    task_usable_dc=np.array([10., 20., 30., 0.]),
                    load_p_kw=1.06*self.base, base_p_kw=self.base,
                    prepared_tasks=self.prepared, times_h=self.times,
                    task_ids=self.prepared.task_ids, road_ids=tuple(range(1, 81)))
        args.update(overrides)
        return compute_features(**args)

    def test_five_values_with_unequal_vehicle_task_counts_and_load_above_one(self):
        result = self.sample()
        np.testing.assert_allclose(result.values, [.495, .5, .25, .5, 1.06], atol=1e-14, rtol=0)
        self.assertNotAlmostEqual(result.values[3], .25)  # Wrong equal-task average.
        np.testing.assert_allclose(self.prepared.task_weights, [1/6, 1/6, 1/6, 1/2], atol=1e-15)
        self.assertEqual(self.prepared.vehicle_count, 2)
        self.assertFalse(self.prepared.empty_H)
        self.assertFalse(result.values.flags.writeable)

    def test_native_grid_includes_all_boundary_points_and_requires_common_axes(self):
        latitude = np.linspace(30., 31., 21)
        longitude = np.linspace(-84., -83., 21)
        fields = {str(i): SimpleNamespace(latitude=latitude.copy(), longitude=longitude.copy())
                  for i in range(8)}
        grid = native_weather_grid(SimpleNamespace(fields=fields))
        expected = np.array([(lat, lon) for lat in latitude for lon in longitude])
        np.testing.assert_array_equal(grid.points_latlon, expected)
        self.assertEqual(grid.point_ids, tuple(range(441)))
        self.assertEqual(grid.points_xy_km.shape, (441, 2))
        self.assertTrue(np.isfinite(grid.points_xy_km).all())
        self.assertFalse(grid.points_latlon.flags.writeable)
        self.assertEqual(grid.grid_sha256,
            native_weather_grid(SimpleNamespace(fields=dict(reversed(list(fields.items()))))).grid_sha256)
        fields['7'].longitude[0] -= .001
        with self.assertRaises(ValueError):
            native_weather_grid(SimpleNamespace(fields=fields))

    def test_deliverability_caps_only_ratio_and_empty_H_flag(self):
        result = self.sample(task_usable_dc=np.array([20., 40., 60., 80.]))
        self.assertEqual(result.values[3], 0.)
        prepared = prepare_feature_tasks(task_bank(empty=True))
        self.assertTrue(prepared.empty_H)
        empty = self.sample(prepared_tasks=prepared, task_ids=(), task_usable_dc=np.empty(0))
        self.assertEqual(empty.values[3], 0.)
        self.assertTrue(empty.diagnostics['empty_H'])

    def test_connector_blockage_is_excluded_from_canonical_road_feature(self):
        roads = np.column_stack((self.roads, np.ones((100, 8), dtype=bool)))
        extended = self.sample(road_blocked=roads, road_ids=tuple(range(1, 89)))
        self.assertEqual(extended.values[2], .25)
        np.testing.assert_array_equal(extended.values, self.sample().values)

    def test_weighted_load_ratio_includes_whole_day_not_mean_hourly_ratios(self):
        base = np.ones((24, 33))
        base[:12] = 3.
        factors = np.r_[np.full(12, 1.08), np.full(12, .92)]
        result = self.sample(base_p_kw=base, load_p_kw=base*factors[:, None])
        self.assertAlmostEqual(result.values[4], 1.04)
        self.assertNotAlmostEqual(result.values[4], factors.mean())

    def test_feature_repeat_and_identity_bind_payload_not_loop_order(self):
        identity = {'target': 'test', 'split': 'planning', 'variant': 0}
        a = self.sample(scenario_identity=identity, grid_sha256='b'*64)
        b = self.sample(scenario_identity=dict(reversed(list(identity.items()))), grid_sha256='b'*64)
        self.assertEqual(a.feature_sha256, b.feature_sha256)
        self.assertNotEqual(a.feature_sha256,
            self.sample(scenario_identity={**identity, 'variant': 1}, grid_sha256='b'*64).feature_sha256)

    def test_bad_domains_time_axes_and_id_alignment_are_rejected(self):
        invalid = [dict(grid_hazard=np.zeros((100, 440))),
                   dict(grid_hazard=np.zeros((104, 441))),
                   dict(line_outages=np.zeros((100, 37), dtype=bool)),
                   dict(times_h=self.times + .01),
                   dict(task_ids=tuple(reversed(self.prepared.task_ids))),
                   dict(road_ids=tuple(range(2, 82))),
                   dict(line_ids=tuple(range(2, 34))),
                   dict(grid_ids=tuple(range(1, 442))),
                   dict(bus_ids=tuple(range(2, 35))),
                   dict(task_usable_dc=np.array([-1., 20., 30., 40.])),
                   dict(grid_hazard=np.full((100, 441), 1.01)),
                   dict(load_p_kw=np.full((24, 33), 1.10)),
                   dict(load_p_kw=np.full((24, 33), np.nan))]
        for overrides in invalid:
            with self.subTest(overrides=list(overrides)):
                with self.assertRaises(ValueError):
                    self.sample(**overrides)

    def test_forecast_energy_must_be_positive_and_H_hash_certified(self):
        bad = task_bank()
        bad['tasks'][0]['usable_energy_kwh'] = 0.
        bad['task_sha256'] = canonical_hash({k: bad[k] for k in
            ('dependency_sha256', 'config', 'certified', 'tasks')})
        with self.assertRaises(ValueError):
            prepare_feature_tasks(bad)
        for field, value in [('certified', False), ('task_sha256', '0'*64)]:
            bad = copy.deepcopy(task_bank())
            bad[field] = value
            with self.assertRaises(ValueError):
                prepare_feature_tasks(bad)


if __name__ == '__main__':
    unittest.main()
