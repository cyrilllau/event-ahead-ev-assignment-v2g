"""Native-grid gust vectors and convex interpolation for the frozen wind model.

u10/v10 already point towards east/north. Convert native points to gust-scaled
Cartesian vectors *before* space/time interpolation (paper Appendix B); never
interpolate a direction angle. Geographic registration matches the stored case
coordinates, using 111.32 km/degree and the common-center latitude for longitude.
The current native forecast/analysis arrays have no positive-gust zero-direction
points. Such a future inconsistent point raises instead of inventing a direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .case_inputs import ForecastEvent, InputValidationError

COMMON_CENTER_LATITUDE = 30.841236
COMMON_CENTER_LONGITUDE = -83.289403
KM_PER_LATITUDE_DEGREE = 111.32
KM_PER_LONGITUDE_DEGREE = 111.32 * np.cos(np.deg2rad(COMMON_CENTER_LATITUDE))


def _finite_array(values: object, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise InputValidationError(f"Nonfinite {label}")
    return array


def _readonly(values: object) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def xy_to_latlon(points_xy_km: object) -> np.ndarray:
    """Convert (..., 2) local east/north coordinates to (latitude, longitude)."""
    xy = _finite_array(points_xy_km, "local coordinates")
    if xy.ndim < 1 or xy.shape[-1] != 2:
        raise InputValidationError("Local coordinates need a final east/north axis of length 2")
    return np.stack((COMMON_CENTER_LATITUDE + xy[..., 1] / KM_PER_LATITUDE_DEGREE,
                     COMMON_CENTER_LONGITUDE + xy[..., 0] / KM_PER_LONGITUDE_DEGREE), axis=-1)


def latlon_to_xy(points_latlon: object) -> np.ndarray:
    """Inverse of xy_to_latlon, with the same explicitly registered center."""
    ll = _finite_array(points_latlon, "geographic coordinates")
    if ll.ndim < 1 or ll.shape[-1] != 2:
        raise InputValidationError("Geographic coordinates need a final latitude/longitude axis of length 2")
    return np.stack(((ll[..., 1] - COMMON_CENTER_LONGITUDE) * KM_PER_LONGITUDE_DEGREE,
                     (ll[..., 0] - COMMON_CENTER_LATITUDE) * KM_PER_LATITUDE_DEGREE), axis=-1)


def gust_vectors(u10: object, v10: object, gust: object) -> np.ndarray:
    """Return gust * (u10, v10) / norm at native grid points, in m/s.

    A zero gust yields the zero vector. A positive gust with exactly zero 10 m
    speed is undefined and rejected deterministically. No angular conversion or
    meteorological 'from' reversal is needed for native u/v components.
    """
    u, v, g = (_finite_array(value, label) for value, label in
               ((u10, "u10"), (v10, "v10"), (gust, "gust")))
    if u.shape != v.shape or u.shape != g.shape:
        raise InputValidationError("u10, v10 and gust shapes must match")
    if np.any(g < 0):
        raise InputValidationError("Negative gust")
    speed = np.hypot(u, v)
    if np.any((speed == 0) & (g > 0)):
        raise InputValidationError("Positive gust has zero native wind direction")
    direction = np.divide(np.stack((u, v), axis=-1), speed[..., None],
                          out=np.zeros(u.shape + (2,), dtype=float),
                          where=speed[..., None] > 0)
    return direction * g[..., None]


def radial_cap(vectors: object, max_speed: float) -> np.ndarray:
    """Cap vector magnitude in m/s while preserving nonzero vector direction."""
    values = _finite_array(vectors, "wind vectors")
    if values.ndim < 1 or values.shape[-1] != 2:
        raise InputValidationError("Wind vectors need a final component axis of length 2")
    if not np.isscalar(max_speed) or not np.isfinite(max_speed) or max_speed < 0:
        raise InputValidationError("Wind cap must be finite and nonnegative")
    magnitude = np.hypot(values[..., 0], values[..., 1])
    scale = np.divide(max_speed, magnitude, out=np.ones_like(magnitude), where=magnitude > 0)
    return values * np.minimum(1.0, scale)[..., None]


def _bracket(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(axis) == 1:
        zero = np.zeros(values.shape, dtype=int)
        return zero, zero, np.zeros(values.shape)
    clipped = np.clip(values, axis[0], axis[-1])
    right = np.clip(np.searchsorted(axis, clipped, side="right"), 1, len(axis) - 1)
    left = right - 1
    return left, right, (clipped - axis[left]) / (axis[right] - axis[left])


@dataclass(frozen=True, slots=True)
class GridVectorField:
    """Tensor grid (hour, latitude, longitude, east/north), in m/s.

    All three axes may ascend or descend; arrays are copied, normalized to
    ascending order and made read-only. Sampling is trilinear and holds the
    nearest spatial/temporal endpoint outside the native grid. A scenario with
    east/north translation ``offset`` and positive delay ``lag`` samples this
    field at ``sample(times_h - lag, points_xy_km - offset)``.
    """

    hours: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        axes = [_finite_array(getattr(self, key), key)
                for key in ("hours", "latitude", "longitude")]
        values = _finite_array(self.vectors, "grid vectors")
        for axis in axes:
            if axis.ndim != 1 or len(axis) == 0:
                raise InputValidationError("Each wind grid axis must be a nonempty vector")
            if len(axis) > 1 and not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0)):
                raise InputValidationError("Wind grid axes must be strictly monotone")
        if values.shape != tuple(len(axis) for axis in axes) + (2,):
            raise InputValidationError("Wind grid vector shape must be (hour, latitude, longitude, 2)")
        for dimension, (name, axis) in enumerate(zip(("hours", "latitude", "longitude"), axes)):
            if len(axis) > 1 and axis[0] > axis[-1]:
                axis = axis[::-1]
                values = np.flip(values, axis=dimension)
            object.__setattr__(self, name, _readonly(axis))
        object.__setattr__(self, "vectors", _readonly(values))

    def sample(self, times_h: Sequence[float], points_xy_km: object) -> np.ndarray:
        """Sample a shared set of A locations at T times; return shape (T,A,2)."""
        times = np.atleast_1d(_finite_array(times_h, "query times"))
        points = _finite_array(points_xy_km, "query points")
        if times.ndim != 1 or points.ndim != 2 or points.shape[1] != 2:
            raise InputValidationError("Queries require time vector and (A,2) local points")
        ll = xy_to_latlon(points)
        t0, t1, wt = _bracket(self.hours, times)
        y0, y1, wy = _bracket(self.latitude, ll[:, 0])
        x0, x1, wx = _bracket(self.longitude, ll[:, 1])
        result = np.zeros((len(times), len(points), 2), dtype=float)
        for ti, tw in ((t0, 1 - wt), (t1, wt)):
            for yi, yw in ((y0, 1 - wy), (y1, wy)):
                for xi, xw in ((x0, 1 - wx), (x1, wx)):
                    weight = tw[:, None] * (yw * xw)[None, :]
                    result += weight[..., None] * self.vectors[ti[:, None], yi[None, :], xi[None, :]]
        return result


def forecast_vector_field(event: ForecastEvent) -> GridVectorField:
    """Build a baseline exclusively from the event's already-loaded forecast."""
    return GridVectorField(event.model_hours, event.latitude, event.longitude,
                           gust_vectors(event.u10, event.v10, event.gust))
