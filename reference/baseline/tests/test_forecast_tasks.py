"""Forecast adapter, task-bank serialization and certification checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths


from evcs_experiments.case_inputs import load_case_inputs
from evcs_experiments.forecast_tasks import load_forecast_task_inputs, build_candidate_tasks
from evcs_experiments.road_disruptions import load_road_model


ROOT = get_paths().data_root


class ForecastTaskAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event = load_case_inputs(ROOT).event_ids[0]
        cls.inputs = load_forecast_task_inputs(ROOT, cls.event)

    def test_forecast_travel_mapping_has_88_arcs_and_single_minutes_conversion(self):
        model = load_road_model(ROOT, include_connectors=True)
        inputs = self.inputs
        hazard = np.minimum(1., np.maximum(0., (inputs.forecast_speed - 10) / 15))
        expected = model.t0_min * (1 + model.kappa_tt * hazard) / 60
        expected[inputs.forecast_speed > model.w_mov_m_s] = np.inf
        np.testing.assert_array_equal(inputs.forecast_travel_hours, expected)
        self.assertEqual(inputs.forecast_travel_hours.shape, (100, 88))
        self.assertFalse(inputs.forecast_travel_hours.flags.writeable)
        self.assertEqual(len(inputs.network.arcs), 88)

    def test_complete_bank_json_serializes_and_semantic_hash_ignores_runtime_budget(self):
        first = build_candidate_tasks(self.inputs, time_limit_seconds=2, overall_time_limit_seconds=20)
        second = build_candidate_tasks(self.inputs, time_limit_seconds=3, overall_time_limit_seconds=30)
        payload = json.loads(json.dumps(first.as_dict(), allow_nan=False))
        self.assertTrue(first.certified)
        self.assertEqual(len(first.queries), 640)
        self.assertEqual(len(first.vehicle_summaries), 80)
        self.assertEqual(first.task_sha256, second.task_sha256)
        self.assertEqual(payload["task_count"], len(first.tasks))
        self.assertTrue(any(row["cache_hit"] for row in first.queries))

    def test_incomplete_query_makes_entire_task_bank_provisional(self):
        from evcs_experiments.forecast_routes import search_route
        def incomplete(*args, **kwargs):
            kwargs["max_expansions"] = 0
            return search_route(*args, **kwargs)
        with patch("evcs_experiments.forecast_tasks.search_route", side_effect=incomplete):
            bank = build_candidate_tasks(self.inputs, overall_time_limit_seconds=20)
        self.assertFalse(bank.certified)
        self.assertEqual(bank.status, "SEARCH_INCOMPLETE")
        self.assertTrue(bank.as_dict()["tasks_provisional"])
        json.dumps(bank.as_dict(), allow_nan=False)

    def test_two_hour_tasks_enforce_all_compulsory_periods_and_no_realized_draws(self):
        with patch("evcs_experiments.wind_scenarios.load_wind_bank", side_effect=AssertionError("analysis forbidden")):
            inputs = load_forecast_task_inputs(ROOT, self.event)
            bank = build_candidate_tasks(inputs, tau_blk_h=2, overall_time_limit_seconds=20)
        self.assertTrue(bank.certified)
        for task in bank.tasks:
            self.assertLessEqual(task.start_period, 22)
            self.assertEqual(len(task.block_periods), 2)
            self.assertTrue(all(task.reserve[t - 1] for t in task.block_periods))
            self.assertGreaterEqual(task.arrival_soc, task.qualification_soc)


if __name__ == "__main__":
    unittest.main()
