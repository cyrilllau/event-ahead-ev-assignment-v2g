"""Executable with unittest; uses the bundled case without production/OOS access."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths

from evcs_experiments.case_inputs import (
    FORECAST_KEYS, INPUT_PATHS, InputValidationError, VehicleState,
    load_case_inputs, sample_clock_profile,
)

CASE_ROOT = get_paths().data_root


class CaseInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_case_inputs(CASE_ROOT)

    def test_baseline_all_eight_events_exact(self):
        with (CASE_ROOT / INPUT_PATHS[4]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        reference = {(row["event_id"], row["e"]): row for row in rows}
        self.assertEqual(len(reference), 640)
        for event in self.case.event_ids:
            target = self.case.target(event, 1)
            self.assertEqual(len(target.vehicle_states), 80)
            self.assertEqual(len(target.station_ids), 8)
            for state in target.vehicle_states:
                row = reference[event, state.vehicle_id]
                self.assertEqual(state.road_node, int(row["loc_ini_e_road_node"]))
                self.assertEqual(state.soc, float(row["SOC_ini_e"]))
                self.assertEqual(target.assignment_time_local.isoformat(), row["theta_asg_local_eastern"])

    def test_half_hour_uses_service_clock(self):
        target = self.case.target("michael_2018", .5)
        self.assertEqual((target.assignment_time_local.hour, target.assignment_time_local.minute), (13, 30))
        prior = self.case.target("michael_2018", 1)
        following = self.case.target("michael_2018", 0)
        for a, b, halfway in zip(prior.vehicle_states, following.vehicle_states, target.vehicle_states):
            self.assertEqual(halfway.road_node, a.road_node)
            self.assertAlmostEqual(halfway.soc, (a.soc + b.soc) / 2, places=14)

    def test_midnight_wrap_and_location_hold(self):
        profile = {hour: VehicleState("EV", "D", hour + 1, hour / 24,
                                      hour, hour, 0, 0) for hour in range(24)}
        state = sample_clock_profile(profile, 23.5)
        self.assertEqual(state.road_node, 24)
        self.assertEqual(state.x_km, 23)
        self.assertAlmostEqual(state.soc, (23 / 24) / 2)
        self.assertEqual(sample_clock_profile(profile, -.5), state)
        self.assertEqual(sample_clock_profile(profile, 24), profile[0])

    def test_axes_endpoint_hold_and_linear_interpolation(self):
        for lead in (0, .5, 1, 2):
            target = self.case.target("ian_2022", lead)
            self.assertEqual(target.operating_intervals_h, tuple((t, t + 1) for t in range(24)))
            self.assertEqual(target.replay_times_h[0], -lead)
            self.assertEqual(target.replay_times_h[-1], 23.75)
            self.assertTrue(np.all(np.diff(target.replay_times_h) == .25))
        event = self.case.forecast.for_event("ian_2022")
        samples = event.at_times([-2, -1, -.5, 23, 23.75, 24])
        for field in ("u10", "v10", "gust"):
            values = getattr(samples, field)
            raw = getattr(event, field)
            np.testing.assert_array_equal(values[0], raw[0])
            np.testing.assert_array_equal(values[1], raw[0])
            np.testing.assert_array_equal(values[2], (raw[0] + raw[1]) / 2)
            np.testing.assert_array_equal(values[3:], np.stack([raw[-1]] * 3))
        self.assertTrue(self.case.target("ian_2022", 2).forecast_issued_after_assignment)
        self.assertTrue(self.case.target("milton_2024", 2).forecast_issued_after_assignment)
        self.assertFalse(self.case.target("michael_2018", 2).forecast_issued_after_assignment)

    def test_analysis_arrays_never_loaded(self):
        original_load = np.load
        read_keys = []
        class TrackedArchive:
            def __init__(self, *args, **kwargs):
                self.inner = original_load(*args, **kwargs)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.inner.close()
            def __getitem__(self, key):
                if key not in FORECAST_KEYS:
                    raise AssertionError(f"Forbidden weather read: {key}")
                read_keys.append(key)
                return self.inner[key]
        with patch("evcs_experiments.case_inputs.np.load", side_effect=TrackedArchive):
            case = load_case_inputs(CASE_ROOT)
        self.assertEqual(read_keys, list(FORECAST_KEYS))
        self.assertFalse(hasattr(case.forecast.for_event(case.event_ids[0]), "analysis_u10"))
        with self.assertRaises((AttributeError, TypeError)):
            case.forecast.for_event(case.event_ids[0]).analysis = None

    def test_stable_identity_hash_and_rejected_inputs(self):
        a = self.case.target("michael_2018", 1)
        b = load_case_inputs(CASE_ROOT).target("michael_2018", 1)
        self.assertEqual((a.target_id, a.state_sha256), (b.target_id, b.state_sha256))
        self.assertNotEqual(a.state_sha256, self.case.target("michael_2018", 0).state_sha256)
        self.assertEqual(set(a.input_hashes), set(INPUT_PATHS))
        self.assertTrue(all(len(value) == 64 for value in a.input_hashes.values()))
        for invalid in (-1, .25, 1.5, 3, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(InputValidationError):
                self.case.target("michael_2018", invalid)
        with self.assertRaises(InputValidationError):
            self.case.target("unknown")
        with self.assertRaises(InputValidationError):
            a.forecast.at_times([float("nan")])
        with self.assertRaises(ValueError):
            a.forecast.u10[0, 0, 0] = 123

    def test_duplicate_profile_and_nonfinite_weather_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in INPUT_PATHS:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CASE_ROOT / name, root / name)
            profile_path = root / INPUT_PATHS[3]
            original = profile_path.read_text()
            profile_path.write_text(original + original.splitlines()[1] + "\n")
            with self.assertRaisesRegex(InputValidationError, "duplicate primary key"):
                load_case_inputs(root)
            profile_path.write_text(original)
            weather_path = root / INPUT_PATHS[0]
            with np.load(weather_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in FORECAST_KEYS}
            arrays["forecast_gust_grid"][0, 0, 0, 0] = np.nan
            np.savez(weather_path, **arrays)
            with self.assertRaisesRegex(InputValidationError, "nonfinite weather"):
                load_case_inputs(root)


if __name__ == "__main__":
    unittest.main()
