"""Analytic interpolation and bundled-coordinate checks; no solver required."""
from __future__ import annotations

import csv
import unittest

import numpy as np

from evcs_experiments.config import get_paths

from evcs_experiments.case_inputs import InputValidationError, load_case_inputs
from evcs_experiments.wind_field import (
    COMMON_CENTER_LATITUDE, COMMON_CENTER_LONGITUDE, GridVectorField,
    forecast_vector_field, gust_vectors, latlon_to_xy, radial_cap, xy_to_latlon,
)

CASE_ROOT = get_paths().data_root


def affine(t, x, y):
    return np.stack((3 + 2 * t + .7 * x - .2 * y,
                     -1 + .3 * t - .4 * x + .8 * y), axis=-1)


def affine_grid(descending=False):
    hours = np.array([-2., 0., 3.])
    lat = COMMON_CENTER_LATITUDE + np.array([-.08, -.01, .07])
    lon = COMMON_CENTER_LONGITUDE + np.array([-.09, .02, .08])
    if descending:
        hours, lat, lon = hours[::-1], lat[::-1], lon[::-1]
    t, la, lo = np.meshgrid(hours, lat, lon, indexing="ij")
    xy = latlon_to_xy(np.stack((la, lo), axis=-1))
    return GridVectorField(hours, lat, lon, affine(t, xy[..., 0], xy[..., 1]))


class WindFieldTests(unittest.TestCase):
    def test_analytical_affine_trilinear_and_descending_axes(self):
        times = np.array([-1.7, -.2, 1.6, 2.9])
        points = np.array([[-3, -5], [0, 0], [4, 6]])
        expected = affine(np.broadcast_to(times[:, None], (4, 3)),
                          np.broadcast_to(points[None, :, 0], (4, 3)),
                          np.broadcast_to(points[None, :, 1], (4, 3)))
        for descending in (False, True):
            np.testing.assert_allclose(affine_grid(descending).sample(times, points), expected,
                                       rtol=0, atol=8e-13)

    def test_time_and_spatial_endpoint_hold(self):
        field = affine_grid()
        points = np.array([[-100, 100], [100, -100]])
        ll = xy_to_latlon(points)
        ll[:, 0] = np.clip(ll[:, 0], field.latitude[0], field.latitude[-1])
        ll[:, 1] = np.clip(ll[:, 1], field.longitude[0], field.longitude[-1])
        clipped = latlon_to_xy(ll)
        np.testing.assert_allclose(field.sample([-50, 50], points),
            affine(np.array([[-2., -2.], [3., 3.]]),
                   np.broadcast_to(clipped[:, 0], (2, 2)),
                   np.broadcast_to(clipped[:, 1], (2, 2))), atol=1e-13)

    def test_translation_and_time_delay_sign_convention(self):
        field = affine_grid()
        time = np.array([1.])
        point = np.array([[3., 4.]])
        offset = np.array([2., -1.])
        lag = .5
        translated = field.sample(time - lag, point - offset)
        np.testing.assert_allclose(translated[0, 0], affine(.5, 1., 5.), atol=8e-13)
        self.assertGreater(np.linalg.norm(translated - field.sample(time + lag, point + offset)), 1.)

    def test_convert_native_vectors_before_temporal_interpolation(self):
        u = np.array([1., 0.]).reshape(2, 1, 1)
        v = np.array([0., 100.]).reshape(2, 1, 1)
        field = GridVectorField([0, 1], [COMMON_CENTER_LATITUDE], [COMMON_CENTER_LONGITUDE],
                                gust_vectors(u, v, np.full_like(u, 10.)))
        np.testing.assert_allclose(field.sample([.5], [[0, 0]]), [[[5., 5.]]], atol=1e-15)

    def test_angle_wrap_and_opposing_vector_cancellation(self):
        for angles, expected in (([359., 1.], [10 * np.cos(np.deg2rad(1)), 0.]),
                                 ([90., 270.], [0., 0.])):
            theta = np.deg2rad(angles).reshape(1, 1, 2)
            field = GridVectorField([0], [COMMON_CENTER_LATITUDE],
                                    [COMMON_CENTER_LONGITUDE - .01, COMMON_CENTER_LONGITUDE + .01],
                                    gust_vectors(np.cos(theta), np.sin(theta), np.full_like(theta, 10.)))
            np.testing.assert_allclose(field.sample([0], [[0, 0]])[0, 0], expected, atol=1e-13)

    def test_gust_direction_and_zero_handling(self):
        np.testing.assert_allclose(gust_vectors([3., 0., -3.], [4., 0., 4.], [10., 0., 5.]),
                                   [[6., 8.], [0., 0.], [-3., 4.]])
        with self.assertRaisesRegex(InputValidationError, "zero native wind direction"):
            gust_vectors([0.], [0.], [1.])
        for u, v, g in (([1.], [0.], [-1.]), ([1.], [float("nan")], [0.]),
                        ([1., 2.], [1.], [1.])):
            with self.assertRaises(InputValidationError):
                gust_vectors(u, v, g)

    def test_radial_cap_preserves_direction_and_zero(self):
        raw = np.array([[3., 4.], [0., 0.], [-6., 8.], [.6, -.8]])
        np.testing.assert_allclose(radial_cap(raw, 5.), [[3., 4.], [0., 0.], [-3., 4.], [.6, -.8]])
        np.testing.assert_array_equal(radial_cap(raw, 0.), np.zeros_like(raw))
        np.testing.assert_array_equal(raw[2], [-6., 8.])
        for cap in (-1., float("nan"), float("inf")):
            with self.assertRaises(InputValidationError):
                radial_cap(raw, cap)

    def test_registration_matches_stored_road_coordinates(self):
        with (CASE_ROOT / "04_transportation/NR__road_nodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        xy = np.array([[float(r["loc_rnode_x_km"]), float(r["loc_rnode_y_km"])] for r in rows])
        ll = np.array([[float(r["loc_rnode_latitude"]), float(r["loc_rnode_longitude"])] for r in rows])
        np.testing.assert_allclose(xy_to_latlon(xy), ll, rtol=0, atol=1e-8)
        np.testing.assert_allclose(latlon_to_xy(xy_to_latlon(xy)), xy, rtol=0, atol=1e-12)

    def test_real_forecast_native_grid_recovery_and_vector_bound(self):
        case = load_case_inputs(CASE_ROOT)
        for event_id in case.event_ids:
            event = case.forecast.for_event(event_id)
            field = forecast_vector_field(event)
            points = latlon_to_xy([[event.latitude[2], event.longitude[3]],
                                  [event.latitude[10], event.longitude[10]]])
            samples = field.sample([event.model_hours[0], event.model_hours[-1]], points)
            expected = field.vectors[[0, -1]][:, [2, 10], [3, 10]]
            np.testing.assert_allclose(samples, expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(np.hypot(field.vectors[..., 0], field.vectors[..., 1]),
                                       event.gust, atol=1e-13)
            sample = field.sample([-.5, 3.25, 23.75], [[0, 0], [1, 2]])
            self.assertLessEqual(np.hypot(sample[..., 0], sample[..., 1]).max(), event.gust.max())

    def test_grid_is_copied_readonly_and_invalid_queries_rejected(self):
        raw = np.ones((1, 1, 1, 2))
        field = GridVectorField([0], [COMMON_CENTER_LATITUDE], [COMMON_CENTER_LONGITUDE], raw)
        raw[:] = 10
        np.testing.assert_array_equal(field.sample([-5, 0, 100], [[0, 0]]), np.ones((3, 1, 2)))
        with self.assertRaises(ValueError):
            field.vectors[:] = 2
        for times, points in (([float("nan")], [[0, 0]]), ([[0, 1]], [[0, 0]]), ([0], [0, 1])):
            with self.assertRaises(InputValidationError):
                field.sample(times, points)
        with self.assertRaises(InputValidationError):
            GridVectorField([0, 0], [0], [0], np.ones((2, 1, 1, 2)))
        with self.assertRaises(InputValidationError):
            GridVectorField([0], [0], [0], np.ones((1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
