"""S01 wind and LOEO residual generator; no line/road/load/OOS evaluation.

The manuscript specifies error transfer and block length but omits their exact
composition. This module implements the predeclared S01-I02 decision: native
gust-vector errors, shared moving blocks of three hourly samples, linear joins,
shared field transform, and final base-event radial magnitude cap.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError
from .evaluation_access import require_oos_access
from .random_streams import canonical_hash, event_order, residual_donor, stream
from .wind_field import (GridVectorField, gust_vectors, radial_cap, xy_to_latlon,
                         COMMON_CENTER_LATITUDE, COMMON_CENTER_LONGITUDE)

WEATHER_PATH = "07_weather/xiW_base__registered_wind_8events_25states.npz"
CAP_PATH = "07_weather/base_event__spatial_registration.csv"
GEO_PATH = "07_weather/spatial_registration__IEEE33.json"


@dataclass(frozen=True)
class WindParameters:
    target_event: str
    base_event: str
    donor_event: str
    split: str
    development: bool
    variant_index: int
    amplitude: float
    offset_xy_km: tuple[float, float]
    lag_h: int
    error_scale: float
    block_start_indices: tuple[int, ...]

    def as_dict(self):
        return asdict(self)


def draw_parameters(event_ids, target_event, base_event, variant_index,
                    split="planning", development=False):
    order = event_order(event_ids)
    if target_event not in order or base_event not in order:
        raise InputValidationError("Unknown target/base event")
    if split == "oos" and not development:
        require_oos_access(target_event=target_event, base_event=base_event,
                           base_index=order.index(base_event), variant_index=variant_index)
    if split not in ("planning", "calibration", "oos"):
        raise InputValidationError("Unknown split")
    if type(variant_index) is not int or variant_index < 0:
        raise InputValidationError("variant_index must be a nonnegative integer")
    if not development and variant_index >= {"planning": 20, "calibration": 10, "oos": 50}[split]:
        raise InputValidationError("Variant exceeds the frozen formal bank")
    base_index = order.index(base_event)
    def rng(mechanism):
        return stream(split, base_index, variant_index, mechanism,
                      development=development).generator()
    u = rng("wind_amplitude").random()
    normal = NormalDist(mu=1., sigma=.08)
    low, high = normal.cdf(.85), normal.cdf(1.2)
    amplitude = float(normal.inv_cdf(low + u * (high - low)))
    loc_rng = rng("wind_location")
    while True:
        offset = loc_rng.normal(0, 4, size=2)
        if np.linalg.norm(offset) <= 10:
            break
    lag = int(rng("wind_lag").choice([-2, -1, 0, 1, 2], p=[.1, .2, .4, .2, .1]))
    donor = residual_donor(order, target_event, split, base_index, variant_index,
                           development=development)
    starts = tuple(int(v) for v in rng("residual_block").integers(0, 23, size=9))
    return WindParameters(target_event, base_event, donor, split, bool(development), variant_index,
                          amplitude, tuple(float(v) for v in offset), lag,
                          float(rng("residual_scale").uniform(.75, 1.25)), starts)


def resample_residual_blocks(residual, starts):
    values = np.asarray(residual, dtype=float)
    if values.ndim < 2 or values.shape[0] != 25 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise InputValidationError("Residual requires 25 finite hourly vector states")
    starts = tuple(starts)
    if len(starts) != 9 or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or not 0 <= s <= 22 for s in starts):
        raise InputValidationError("Nine source block starts in 0..22 required")
    result = np.concatenate([values[s:s+3] for s in starts], axis=0)[:25].copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WindSample:
    vectors: np.ndarray
    speed: np.ndarray
    hazard: np.ndarray
    cap_m_s: float
    parameters_sha256: str
    temporal_projection_count: int
    spatial_projection_count: int


@dataclass(frozen=True)
class WindBank:
    event_ids: tuple[str, ...]
    fields: Mapping[str, GridVectorField]
    residuals: Mapping[str, np.ndarray]
    caps: Mapping[str, float]
    input_hashes: Mapping[str, str]
    diagnostics: Mapping[str, object]

    def _check_parameters(self, p):
        if not isinstance(p, WindParameters):
            raise InputValidationError("Expected WindParameters")
        if any(e not in self.event_ids for e in (p.target_event, p.base_event, p.donor_event)) or p.donor_event == p.target_event:
            raise InputValidationError("Unknown event or target used as residual donor")
        if p.split not in ("planning", "calibration", "oos"):
            raise InputValidationError("Unknown split")
        if p.split == "oos" and not p.development:
            require_oos_access(target_event=p.target_event, base_event=p.base_event,
                               variant_index=p.variant_index)
        offset = np.asarray(p.offset_xy_km, dtype=float)
        if offset.shape != (2,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > 10 + 1e-12:
            raise InputValidationError("Spatial offset must lie in the 10 km disk")
        # Zero error and exact identity transforms are allowed for analytical checks.
        if not np.isfinite(p.amplitude) or p.amplitude <= 0 or not np.isfinite(p.error_scale) or p.error_scale < 0:
            raise InputValidationError("Amplitude must be positive and residual multiplier nonnegative")
        if p.lag_h not in (-2, -1, 0, 1, 2):
            raise InputValidationError("Unsupported temporal lag")

    def field(self, parameters):
        self._check_parameters(parameters)
        base = self.fields[parameters.base_event]
        errors = resample_residual_blocks(self.residuals[parameters.donor_event],
                                          parameters.block_start_indices)
        return GridVectorField(base.hours, base.latitude, base.longitude,
                               base.vectors + parameters.error_scale * errors)

    def sample(self, parameters, times_h, points_xy_km, kappa_W=1.0, regime="R1"):
        if regime not in ("R1", "R2", "R3") or not np.isfinite(kappa_W) or kappa_W <= 0:
            raise InputValidationError("Invalid regime or storm multiplier")
        if isinstance(parameters, WindParameters) and parameters.split == "oos" and not parameters.development:
            require_oos_access(regime=regime)
        field = self.field(parameters)
        times = np.atleast_1d(np.asarray(times_h, dtype=float)) - parameters.lag_h
        points = np.asarray(points_xy_km, dtype=float) - np.asarray(parameters.offset_xy_km)
        values = field.sample(times, points)
        multiplier = parameters.amplitude * float(kappa_W) * (1.15 if regime == "R2" else 1.0)
        cap = self.caps[parameters.base_event]
        vectors = radial_cap(multiplier * values, cap)
        speed = np.hypot(vectors[..., 0], vectors[..., 1])
        hazard = np.clip((speed - 10) / 15, 0, 1)
        locations = xy_to_latlon(points)
        spatial_count = int(np.count_nonzero((locations[:, 0] < field.latitude[0]) |
                             (locations[:, 0] > field.latitude[-1]) |
                             (locations[:, 1] < field.longitude[0]) |
                             (locations[:, 1] > field.longitude[-1])))
        temporal_count = int(np.count_nonzero((times < field.hours[0]) | (times > field.hours[-1])))
        for array in (vectors, speed, hazard):
            array.setflags(write=False)
        return WindSample(vectors, speed, hazard, cap, canonical_hash(parameters.as_dict()),
                          temporal_count, spatial_count)


def load_wind_bank(case_root):
    root = Path(case_root)
    with np.load(root / WEATHER_PATH, allow_pickle=False) as archive:
        ids = tuple(str(v) for v in archive["event_ids"])
        order = event_order(ids)
        hours, lat, lon = (np.asarray(archive[k], dtype=float) for k in ("model_hours", "latitude", "longitude"))
        if not np.array_equal(hours, np.arange(-1, 24)):
            raise InputValidationError("Expected native hourly weather states -1..23")
        data = {k: np.asarray(archive[k], dtype=float) for k in
                ("forecast_u10_grid", "forecast_v10_grid", "forecast_gust_grid",
                 "analysis_u10_grid", "analysis_v10_grid", "analysis_gust_grid")}
    expected = (8, 25, len(lat), len(lon))
    if any(v.shape != expected or not np.isfinite(v).all() for v in data.values()):
        raise InputValidationError("Invalid weather array shape or nonfinite value")
    with (root / CAP_PATH).open(newline="") as f:
        cap_rows = list(csv.DictReader(f))
    caps = {r["event_id"]: float(r["source_peak_gust_m_s"]) for r in cap_rows}
    if len(cap_rows) != 8 or set(caps) != set(order) or any(not np.isfinite(c) or c <= 0 for c in caps.values()):
        raise InputValidationError("Expected positive registered cap for each event")
    geo = json.loads((root / GEO_PATH).read_text())["common_center"]
    if abs(geo["latitude"] - COMMON_CENTER_LATITUDE) > 1e-9 or abs(geo["longitude"] - COMMON_CENTER_LONGITUDE) > 1e-9:
        raise InputValidationError("Geographic registration differs from the frozen kernel")
    fields, residuals = {}, {}
    diagnostics = {}
    for kind in ("forecast", "analysis"):
        speed = np.hypot(data[kind+"_u10_grid"], data[kind+"_v10_grid"])
        gust = data[kind+"_gust_grid"]
        diagnostics[kind] = {"gust_below_sustained_count": int(np.count_nonzero(gust < speed)),
                             "max_gust_deficit_m_s": float(np.maximum(speed-gust, 0).max()),
                             "positive_gust_zero_direction_count": int(np.count_nonzero((speed == 0) & (gust > 0)))}
    for i, event in enumerate(ids):
        fvec = gust_vectors(data["forecast_u10_grid"][i], data["forecast_v10_grid"][i], data["forecast_gust_grid"][i])
        avec = gust_vectors(data["analysis_u10_grid"][i], data["analysis_v10_grid"][i], data["analysis_gust_grid"][i])
        fields[event] = GridVectorField(hours, lat, lon, fvec)
        afield = GridVectorField(hours, lat, lon, avec)
        residuals[event] = afield.vectors - fields[event].vectors
        residuals[event].setflags(write=False)
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in (WEATHER_PATH, CAP_PATH, GEO_PATH)}
    return WindBank(order, MappingProxyType(fields), MappingProxyType(residuals),
                    MappingProxyType(caps), MappingProxyType(hashes), MappingProxyType(diagnostics))


def generator_contract():
    return {
        "revision": "S01-I02-v1", "status": "implementation_convention_before_formal_OOS",
        "gust_vector": "gust*(u10,v10)/hypot(u10,v10) at native grid, keep gust even slightly below sustained",
        "zero_direction": "zero gust gives zero vector; positive gust with zero direction is rejected (none in current data)",
        "error": "LOEO donor native analysis gust vector minus forecast gust vector",
        "blocks": {"hourly_samples_per_block": 3, "blocks_per_scenario": 9,
                   "source_start_indices_inclusive": [0, 22], "output_hourly_knots": 25,
                   "sampling": "replacement; same index for every grid point and both vector components",
                   "join": "concatenate, truncate to 25; linear interpolation across all adjacent knots including one-hour block joins"},
        "field_equation": "radial_cap_base[amp*kappa_W*regime_factor*Interp(F_base + kappa_err*Block(E_donor), xy-offset, clip(t-lag))]",
        "amplitude": {"distribution": "truncated_normal", "mean": 1., "std": .08, "bounds": [.85, 1.2]},
        "location": {"distribution": "N(0,16 I2) conditioned on Euclidean norm<=10km", "algorithm": "joint pair rejection"},
        "lag_h": {"support": [-2,-1,0,1,2], "probabilities": [.1,.2,.4,.2,.1]},
        "error_scale": {"distribution": "uniform", "bounds": [.75,1.25], "one_per_scenario": True},
        "transform_order": ["native vectors", "donor residual blocks", "add error", "joint translation and lag", "component interpolation", "common amplitude and stress scaling", "radial cap"],
        "cap_source": CAP_PATH+"#source_peak_gust_m_s", "cap_event": "base_weather_event",
        "boundaries": "nearest endpoint hold in time and space; record projection counts; current assets+10km offset remain spatially inside grid",
        "regime_factor": {"R1":1., "R2":1.15, "R3":1.},
        "future_manuscript_update": "Add explicit residual-composition equation and moving-block/join convention; existing main equation specifies only common storm transform",
        "formal_OOS": "physical sampling locked pending S04 plan-lock interface",
        "scope": "weather component only, not complete physical scenario or optimization",
    }
