"""Correlated, persistent line outages on the frozen curved IEEE33 geometry.

Only the 32 normally closed lines receive latent thresholds. The threshold
vector is drawn once per scenario, independently of the wind streams, and is
held across time. Replay starts at the actual assignment epoch, even when its
wind input was sampled on the longer common reference grid.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError, VALID_LEADS_H
from .evaluation_access import require_oos_access
from .random_streams import canonical_hash, event_order, stream

INPUT_PATHS = (
    "01_config/model_parameters__paper_symbols.csv",
    "02_power_network/L_z__branch_parameters.csv",
    "02_power_network/Lclosed__radial_line_set.csv",
    "02_power_network/Ltie__fixed_open_tie_set.csv",
    "02_power_network/loc_L__line_geography.csv",
    "02_power_network/alpha_frag__line_assignment.csv",
    "07_weather/loc_L_loc_A__weather_sampling_points.csv",
    "00_documentation/paper_v54_IEEE33_CCG_corrected.tex",
)
# Table ieee33_fragility_coefficients of the archived v54 manuscript. Baseline
# coefficients are read from the line-indexed CSV and checked against class 4.
CLASS_COEFFICIENTS = MappingProxyType({
    2: ((-10.289, .0529, .0677), (-6.843, .0345, .0388)),
    4: ((-9.313, .0494, .0776), (-6.037, .0219, .0471)),
    5: ((-8.884, .0468, .0821), (-6.688, .0260, .0476)),
})


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise InputValidationError(f"{name} must be finite")
    return result


def correlation_matrix(points_xy_km, d_corr_km=4.0, epsilon=.05):
    """Unit-diagonal Gaussian correlation, with the nugget applied once."""
    points = _finite(points_xy_km, "Line locations")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise InputValidationError("Line locations require a nonempty (L,2) array")
    if not np.isfinite(d_corr_km) or d_corr_km <= 0:
        raise InputValidationError("Correlation length must be positive")
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise InputValidationError("Nugget must lie in (0,1]")
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    sigma = (1 - epsilon) * np.exp(-distance / d_corr_km)
    np.fill_diagonal(sigma, 1.0)
    return _readonly(sigma)


def crosswind_components(vectors, azimuth_deg_from_north):
    """Magnitude and absolute perpendicular component of east/north vectors.

    Stored azimuth A runs clockwise from north: tangent=(sin A, cos A),
    perpendicular=(cos A, -sin A). No wind-direction or chord approximation.
    """
    wind = _finite(vectors, "Wind vectors")
    azimuth = _finite(azimuth_deg_from_north, "Line azimuth")
    if wind.ndim < 2 or wind.shape[-1] != 2 or azimuth.shape != (wind.shape[-2],):
        raise InputValidationError("Expected wind (...,L,2) and azimuth (L,)")
    angle = np.deg2rad(azimuth)
    speed = np.linalg.norm(wind, axis=-1)
    crosswind = np.abs(wind[..., 0] * np.cos(angle) - wind[..., 1] * np.sin(angle))
    return _readonly(speed), _readonly(crosswind)


def _logistic(values):
    # Equivalent stable logistic without overflow for large finite inputs.
    return np.exp(-np.logaddexp(0.0, -values))


@dataclass(frozen=True)
class FailureProbabilities:
    speed: np.ndarray
    crosswind: np.ndarray
    pole: np.ndarray
    wire: np.ndarray
    combined: np.ndarray


def failure_probabilities(vectors, azimuth_deg_from_north,
                          pole_coefficients, wire_coefficients):
    speed, crosswind = crosswind_components(vectors, azimuth_deg_from_north)
    coefficients = []
    for name, values in (("Pole", pole_coefficients), ("Wire", wire_coefficients)):
        a = _finite(values, f"{name} coefficients")
        if a.shape not in ((3,), (speed.shape[-1], 3)):
            raise InputValidationError(f"{name} coefficients require (3,) or (L,3)")
        coefficients.append(a)
    def component(a):
        return _logistic(a[..., 0] + a[..., 1] * speed + a[..., 2] * crosswind)
    pole, wire = (component(a) for a in coefficients)
    combined = 1 - (1 - pole) * (1 - wire)
    return FailureProbabilities(speed, crosswind, _readonly(pole),
                                _readonly(wire), _readonly(combined))


def persistent_outages(probabilities, uniforms):
    """Return running failure probability and outage state (time, line).

    Callers must slice to the actual assignment epoch first. An independent
    new failure draw at every slot would change the model's marginal law.
    """
    probability = _finite(probabilities, "Failure probabilities")
    threshold = _finite(uniforms, "Uniform thresholds")
    if probability.ndim != 2 or probability.shape[0] == 0 or threshold.shape != (probability.shape[1],):
        raise InputValidationError("Expected nonempty probabilities (T,L) and thresholds (L,)")
    if np.any((probability < 0) | (probability > 1)) or np.any((threshold < 0) | (threshold > 1)):
        raise InputValidationError("Probabilities and thresholds must lie in [0,1]")
    cumulative = np.maximum.accumulate(probability, axis=0)
    return _readonly(cumulative), _readonly(threshold[None, :] <= cumulative, bool)


def _check_grid(times_h):
    times = _finite(times_h, "Replay times")
    if times.ndim != 1 or not len(times):
        raise InputValidationError("Replay times require a nonempty vector")
    if (np.any(np.abs(times * 4 - np.round(times * 4)) > 1e-10)
            or np.any(np.abs(np.diff(times) - .25) > 1e-10)):
        raise InputValidationError("Replay grid must be contiguous quarter hours")
    return times


def operating_outages(times_h, outages):
    """Conservatively map each operating hour to max of its four replay slots."""
    times = _check_grid(times_h)
    values = np.asarray(outages)
    if values.ndim != 2 or values.shape[0] != len(times) or not np.isin(values, (0, 1)).all():
        raise InputValidationError("Outages require binary (T,L) states")
    positions = np.flatnonzero((times >= 0) & (times < 24))
    if len(positions) != 96 or not np.allclose(times[positions], np.arange(96) / 4, rtol=0, atol=1e-10):
        raise InputValidationError("Operating mapping requires all 96 quarter-hour slots in [0,24)")
    return _readonly(values[positions].reshape(24, 4, values.shape[1]).max(axis=1), bool)


@dataclass(frozen=True)
class LineThresholds:
    standard_normals: np.ndarray
    correlated_normals: np.ndarray
    uniforms: np.ndarray
    stream_id: str
    base_event: str
    variant_index: int
    split: str
    development: bool
    model_sha256: str


@dataclass(frozen=True)
class LineReplay:
    times_h: np.ndarray
    line_ids: tuple[int, ...]
    speed: np.ndarray
    crosswind: np.ndarray
    pole_probability: np.ndarray
    wire_probability: np.ndarray
    failure_probability: np.ndarray
    cumulative_failure_probability: np.ndarray
    outages: np.ndarray
    operating_outages: np.ndarray
    thresholds: LineThresholds
    lead_h: float
    pole_class: int


@dataclass(frozen=True)
class LineModel:
    line_ids: tuple[int, ...]
    tie_ids: tuple[int, ...]
    points_xy_km: np.ndarray
    azimuth_deg_from_north: np.ndarray
    pole_coefficients: np.ndarray
    wire_coefficients: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    input_hashes: Mapping[str, str]
    d_corr_km: float
    epsilon: float
    model_sha256: str

    def draw_thresholds(self, event_ids, base_event, variant_index,
                        split="planning", development=False):
        order = event_order(event_ids)
        if base_event not in order:
            raise InputValidationError("Unknown base event")
        if split == "oos" and not development:
            require_oos_access(base_event=base_event, base_index=order.index(base_event),
                               variant_index=variant_index)
        if split not in ("planning", "calibration", "oos"):
            raise InputValidationError("Unknown split")
        if type(variant_index) is not int or variant_index < 0:
            raise InputValidationError("variant_index must be a nonnegative integer")
        if not development and variant_index >= {"planning": 20, "calibration": 10, "oos": 50}[split]:
            raise InputValidationError("Variant exceeds the frozen formal bank")
        identity = stream(split, order.index(base_event), variant_index,
                          "line_gaussian", development=development)
        normal = identity.generator().standard_normal(len(self.line_ids))
        correlated = self.cholesky @ normal
        uniform = np.fromiter((.5 * math.erfc(-float(z) / math.sqrt(2)) for z in correlated),
                              dtype=float, count=len(correlated))
        return LineThresholds(_readonly(normal), _readonly(correlated), _readonly(uniform),
                              identity.stream_id, base_event, variant_index, split,
                              bool(development), self.model_sha256)

    def replay(self, times_h, vectors, thresholds, lead_h=1.0, pole_class=None):
        times = _check_grid(times_h)
        wind = _finite(vectors, "Wind vectors")
        if wind.shape != (len(times), len(self.line_ids), 2):
            raise InputValidationError("Wind requires one vector per time and canonical closed line")
        if isinstance(lead_h, bool) or lead_h not in VALID_LEADS_H:
            raise InputValidationError("Lead must be one of 0, 0.5, 1, 2 hours")
        if (times[0] < -2 - 1e-10 or times[0] > -lead_h + 1e-10
                or abs(times[-1] - 23.75) > 1e-10):
            raise InputValidationError("Replay input must cover assignment epoch through 23.75h within [-2,24)")
        if not isinstance(thresholds, LineThresholds) or thresholds.model_sha256 != self.model_sha256:
            raise InputValidationError("Thresholds must belong to this line model")
        if thresholds.split == "oos" and not thresholds.development:
            require_oos_access(base_event=thresholds.base_event,
                               variant_index=thresholds.variant_index)
        keep = times >= -float(lead_h) - 1e-10
        selected_times = times[keep]
        if abs(selected_times[0] + lead_h) > 1e-10:
            raise InputValidationError("Assignment epoch absent from replay grid")
        selected_class = 4 if pole_class is None else pole_class
        if isinstance(selected_class, bool) or selected_class not in CLASS_COEFFICIENTS:
            raise InputValidationError("Pole class must be 2, 4, or 5")
        coefficients = ((self.pole_coefficients, self.wire_coefficients) if selected_class == 4
                        else CLASS_COEFFICIENTS[selected_class])
        probabilities = failure_probabilities(wind[keep], self.azimuth_deg_from_north, *coefficients)
        cumulative, outages = persistent_outages(probabilities.combined, thresholds.uniforms)
        return LineReplay(_readonly(selected_times), self.line_ids, probabilities.speed,
                          probabilities.crosswind, probabilities.pole, probabilities.wire,
                          probabilities.combined, cumulative, outages,
                          operating_outages(selected_times, outages), thresholds,
                          float(lead_h), int(selected_class))


def _read_csv(root, relative):
    with (root / relative).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _indexed(rows, key="ell"):
    indexed = {int(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise InputValidationError("Duplicate asset identifier in source table")
    return indexed


def load_line_model(case_root):
    root = Path(case_root)
    hashes = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
              for relative in INPUT_PATHS}
    parameters = {r["code_key"]: r["value"] for r in _read_csv(root, INPUT_PATHS[0])}
    d_corr, epsilon = float(parameters["d_corr_km"]), float(parameters["epsilon_Sigma"])
    if (d_corr, epsilon) != (4.0, .05):
        raise InputValidationError("Frozen covariance parameters changed")
    branches = _indexed(_read_csv(root, INPUT_PATHS[1]))
    closed = _indexed(_read_csv(root, INPUT_PATHS[2]))
    ties = _indexed(_read_csv(root, INPUT_PATHS[3]))
    geography = _indexed(_read_csv(root, INPUT_PATHS[4]))
    fragility = _indexed(_read_csv(root, INPUT_PATHS[5]))
    sampling = _indexed([r for r in _read_csv(root, INPUT_PATHS[6]) if r["index_type"] == "ell"], "index_id")
    line_ids, tie_ids = tuple(sorted(closed)), tuple(sorted(ties))
    if line_ids != tuple(range(1, 33)) or tie_ids != tuple(range(33, 38)):
        raise InputValidationError("Expected canonical 32 closed lines and five fixed-open ties")
    expected_all = set(line_ids + tie_ids)
    if any(set(table) != expected_all for table in (branches, geography, fragility, sampling)):
        raise InputValidationError("Line-indexed tables disagree on canonical asset IDs")
    points, azimuths, poles, wires = [], [], [], []
    for line_id in line_ids + tie_ids:
        branch, geo, frag, sample = (table[line_id] for table in (branches, geography, fragility, sampling))
        active = int(line_id in closed)
        flags = (branch["chi_closed_ell"], branch["hazard_eligible_ell"], geo["normally_closed"],
                 geo["active_in_baseline"], geo["hazard_eligible"], frag["active_in_baseline"],
                 frag["hazard_eligible"], sample["hazard_eligible_ell"])
        if any(int(flag) != active for flag in flags):
            raise InputValidationError("Closed/open or hazard eligibility flags disagree")
        endpoints = (int(branch["i_bus"]), int(branch["j_bus"]))
        membership = closed.get(line_id, ties.get(line_id))
        for row, first, second in ((geo, "i_bus", "j_bus"), (frag, "i_bus", "j_bus"),
                                   (sample, "from_index", "to_index"), (membership, "i_bus", "j_bus")):
            if (int(row[first]), int(row[second])) != endpoints:
                raise InputValidationError("Line endpoints disagree across source tables")
        if not active:
            continue
        point = (float(sample["loc_x_km"]), float(sample["loc_y_km"]))
        azimuth = float(sample["phi_deg_from_north"])
        if (not np.allclose(point, (float(geo["loc_ell_x_km"]), float(geo["loc_ell_y_km"])), rtol=0, atol=1e-10)
                or abs(azimuth - float(geo["phi_ell_deg_from_north"])) > 1e-10):
            raise InputValidationError("Weather sampling differs from locked line geometry")
        if frag["fragility_phase_group"] != "phase_3" or int(frag["pole_class"]) != 4:
            raise InputValidationError("Baseline requires three-phase class-4 fragility")
        pole = tuple(float(frag[f"alpha_frag_pole_{j}"]) for j in range(3))
        wire = tuple(float(frag[f"alpha_frag_wire_{j}"]) for j in range(3))
        if (pole, wire) != CLASS_COEFFICIENTS[4]:
            raise InputValidationError("Baseline coefficients differ from frozen manuscript table")
        points.append(point)
        azimuths.append(azimuth)
        poles.append(pole)
        wires.append(wire)
    points = _readonly(points)
    sigma = correlation_matrix(points, d_corr, epsilon)
    cholesky = _readonly(np.linalg.cholesky(sigma))
    identity = canonical_hash({"input_hashes": hashes, "contract": generator_contract()})
    return LineModel(line_ids, tie_ids, points, _readonly(azimuths), _readonly(poles),
                     _readonly(wires), sigma, cholesky, MappingProxyType(hashes),
                     d_corr, epsilon, identity)


def generator_contract():
    return {
        "version": "S01-I03-v1",
        "line_order": "Canonical closed IDs 1..32; fixed-open IDs 33..37 excluded from hazard",
        "geometry": "Stored curved-line midpoint and tangent azimuth, clockwise from north",
        "wind_axes": "Toward east,north; perpendicular=abs(wx*cos(A)-wy*sin(A))",
        "fragility": "logistic(alpha0+alpha1*speed+alpha2*crosswind); union of independent pole/wire mechanisms",
        "class_coefficients": {str(k): {"pole": list(v[0]), "wire": list(v[1])}
                               for k, v in CLASS_COEFFICIENTS.items()},
        "covariance": "diag=1; offdiag=0.95*exp(-midpoint_distance_km/4)",
        "threshold": "One Cholesky-correlated standard-normal vector transformed by NormalCDF; held across replay",
        "stream": "line_gaussian mechanism 7; canonical base/variant split only, target/lead/method/regime absent",
        "persistence": "Slice wind at assignment -lead first; running max failure probability; outage=(uniform<=runningmax)",
        "time_grid": "0.25h slot starts; assignment through 23.75h; no repair",
        "operation": "Hourly outage=max of all four replay slots in that operating hour",
        "OOS": "Formal physical draws/replay closed until S04 plan-lock API",
        "ablation": "This kernel is the correlated physical law; noLineDependence transformation is a later support ablation",
    }
