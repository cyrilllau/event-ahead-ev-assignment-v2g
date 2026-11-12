"""Persistent corridor blockage and recoverable wind-safety restrictions.

The canonical 80 directed arcs expand 40 physical corridors. Reverse arcs
share a single resistance threshold, while different corridors retain distinct
streams even if their sampling coordinates coincide. Depot connectors are
outside this canonical hazard/feature domain and require an explicit adapter.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError, VALID_LEADS_H
from .evaluation_access import require_oos_access
from .line_outages import LineReplay, _check_grid
from .random_streams import canonical_hash, event_order, stream

INPUT_PATHS = (
    "01_config/model_parameters__paper_symbols.csv",
    "04_transportation/A__directed_road_arcs.csv",
    "04_transportation/Rcorr__road_corridors.csv",
    "08_grid_road_coupling/L_A_gamma_blk__line_road_pairs.csv",
    "02_power_network/Lclosed__radial_line_set.csv",
    "02_power_network/Ltie__fixed_open_tie_set.csv",
    "07_weather/loc_L_loc_A__weather_sampling_points.csv",
    "00_documentation/paper_v54_IEEE33_CCG_corrected.tex",
)


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise InputValidationError(f"{name} must be finite")
    return result


def _arc_parameter(value, arc_count, name):
    result = _finite(value, name)
    if result.shape not in ((), (arc_count,)):
        raise InputValidationError(f"{name} requires a scalar or one value per arc")
    return result


@dataclass(frozen=True)
class BlockageProbabilities:
    wind: np.ndarray
    combined: np.ndarray


def road_probabilities(speed, kappa_blk, alpha0, alpha1, gamma, line_outages):
    """Instantaneous qwind and qblk, using contemporaneous replay-slot lines."""
    wind = _finite(speed, "Road wind speed")
    if wind.ndim != 2 or not all(wind.shape) or np.any(wind < 0):
        raise InputValidationError("Wind speed requires nonnegative nonempty (T,A)")
    kappa, intercept, slope = (
        _arc_parameter(v, wind.shape[1], name) for v, name in
        ((kappa_blk, "Blockage scale"), (alpha0, "Logit intercept"),
         (alpha1, "Logit slope")))
    if np.any((kappa < 0) | (kappa > 1)) or np.any(slope < 0):
        raise InputValidationError("Blockage scale must be in [0,1] and slope nonnegative")
    coupling = _finite(gamma, "Line-road coupling")
    outage = np.asarray(line_outages)
    if (coupling.ndim != 2 or coupling.shape[1] != wind.shape[1]
            or outage.shape != (wind.shape[0], coupling.shape[0])
            or not np.isin(outage, (0, 1)).all()
            or np.any((coupling < 0) | (coupling > 1))):
        raise InputValidationError("Expected gamma (L,A) in [0,1] and binary outage (T,L)")
    direct = kappa * np.exp(-np.logaddexp(0.0, -(intercept + slope * wind)))
    survival = np.ones(wind.shape, dtype=float)
    # Avoid a T*L*A intermediate; the empty line product remains one.
    for index in range(coupling.shape[0]):
        survival *= 1 - outage[:, index, None] * coupling[index]
    combined = 1 - (1 - direct) * survival
    return BlockageProbabilities(_readonly(direct), _readonly(combined))


def persistent_blockages(probabilities, uniforms):
    """Running maxima and fixed-threshold states after slicing to assignment."""
    probability = _finite(probabilities, "Blockage probabilities")
    threshold = _finite(uniforms, "Road resistance thresholds")
    if (probability.ndim != 2 or not all(probability.shape)
            or threshold.shape != (probability.shape[1],)
            or np.any((probability < 0) | (probability > 1))
            or np.any((threshold < 0) | (threshold > 1))):
        raise InputValidationError("Expected probabilities (T,A) and thresholds (A,) in [0,1]")
    cumulative = np.maximum.accumulate(probability, axis=0)
    return _readonly(cumulative), _readonly(threshold[None, :] <= cumulative, bool)


def road_travel_times(speed, hazard, blocked, T0, kappa_tt, Wmov):
    """Return movement mask and minutes; unsafe wind may later recover."""
    wind, exposure = _finite(speed, "Road wind speed"), _finite(hazard, "Road hazard")
    blockage = np.asarray(blocked)
    if (wind.ndim != 2 or not all(wind.shape) or exposure.shape != wind.shape
            or blockage.shape != wind.shape or not np.isin(blockage, (0, 1)).all()
            or np.any(wind < 0) or np.any((exposure < 0) | (exposure > 1))):
        raise InputValidationError("Expected matching speed/hazard/binary blockage (T,A)")
    base, sensitivity, limit = (
        _arc_parameter(v, wind.shape[1], name) for v, name in
        ((T0, "Free-flow minutes"), (kappa_tt, "Travel sensitivity"), (Wmov, "Safety limit")))
    if np.any(base <= 0) or np.any(sensitivity < 0) or np.any(limit <= 0):
        raise InputValidationError("Free-flow time and safety limit must be positive; travel sensitivity nonnegative")
    movable = (~blockage.astype(bool)) & (wind <= limit)
    travel = np.where(movable, base * (1 + sensitivity * exposure), np.inf)
    return _readonly(movable, bool), _readonly(travel)


@dataclass(frozen=True)
class RoadThresholds:
    corridor_uniforms: np.ndarray
    arc_uniforms: np.ndarray
    stream_ids: tuple[str, ...]
    model_sha256: str
    base_event: str
    variant_index: int
    split: str
    development: bool


@dataclass(frozen=True)
class RoadReplay:
    times_h: np.ndarray
    arc_ids: tuple[int, ...]
    speed: np.ndarray
    hazard: np.ndarray
    wind_probability: np.ndarray
    blockage_probability: np.ndarray
    cumulative_blockage_probability: np.ndarray
    blocked: np.ndarray
    movable: np.ndarray
    travel_time_min: np.ndarray
    effective_gamma: np.ndarray
    effective_kappa_tt: np.ndarray
    max_passable_travel_time_min: np.ndarray
    thresholds: RoadThresholds
    lead_h: float
    regime: str
    no_coupling: bool


@dataclass(frozen=True)
class RoadModel:
    arc_ids: tuple[int, ...]
    corridor_ids: tuple[int, ...]
    arc_corridor_indices: np.ndarray
    points_xy_km: np.ndarray
    line_ids: tuple[int, ...]
    gamma: np.ndarray
    t0_min: np.ndarray
    kappa_tt: np.ndarray
    w_mov_m_s: np.ndarray
    kappa_blk: np.ndarray
    alpha0: np.ndarray
    alpha1: np.ndarray
    input_hashes: Mapping[str, str]
    model_sha256: str
    w_thr_m_s: float
    w_sev_m_s: float
    gamma_cap: float
    r3_multiplier: float

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
        identities = tuple(stream(split, order.index(base_event), variant_index,
                                  "road_uniform", development=development, asset_index=c)
                           for c in self.corridor_ids)
        uniform = np.array([identity.generator().random() for identity in identities])
        return RoadThresholds(_readonly(uniform), _readonly(uniform[self.arc_corridor_indices]),
                              tuple(identity.stream_id for identity in identities), self.model_sha256,
                              base_event, variant_index, split, bool(development))

    def replay(self, times_h, vectors, thresholds, line_replay,
               lead_h=1.0, regime="R1", no_coupling=False):
        times = _check_grid(times_h)
        wind = _finite(vectors, "Road wind vectors")
        if wind.shape != (len(times), len(self.arc_ids), 2):
            raise InputValidationError("Wind requires one vector per time and canonical directed arc")
        if isinstance(lead_h, bool) or lead_h not in VALID_LEADS_H:
            raise InputValidationError("Lead must be one of 0, 0.5, 1, 2 hours")
        if regime not in ("R1", "R2", "R3") or type(no_coupling) is not bool:
            raise InputValidationError("Expected R1/R2/R3 and boolean no_coupling")
        if (times[0] < -2 - 1e-10 or times[0] > -lead_h + 1e-10
                or abs(times[-1] - 23.75) > 1e-10):
            raise InputValidationError("Replay must cover assignment through 23.75h within [-2,24)")
        if not isinstance(thresholds, RoadThresholds) or thresholds.model_sha256 != self.model_sha256:
            raise InputValidationError("Thresholds must belong to this road model")
        if thresholds.split == "oos" and not thresholds.development:
            require_oos_access(base_event=thresholds.base_event,
                               variant_index=thresholds.variant_index, regime=regime)
        if (thresholds.corridor_uniforms.shape != (len(self.corridor_ids),)
                or not np.array_equal(thresholds.arc_uniforms,
                                      thresholds.corridor_uniforms[self.arc_corridor_indices])):
            raise InputValidationError("Road thresholds must preserve physical corridor sharing")
        keep = times >= -float(lead_h) - 1e-10
        selected_times, selected_wind = times[keep], wind[keep]
        if abs(selected_times[0] + lead_h) > 1e-10:
            raise InputValidationError("Assignment epoch absent from replay grid")
        if (not isinstance(line_replay, LineReplay) or line_replay.line_ids != self.line_ids
                or not np.array_equal(line_replay.times_h, selected_times)
                or line_replay.lead_h != float(lead_h)):
            raise InputValidationError("Line replay must match closed line IDs, lead and every road quarter hour")
        line_thresholds = line_replay.thresholds
        if any(getattr(line_thresholds, name) != getattr(thresholds, name)
               for name in ("base_event", "variant_index", "split", "development")):
            raise InputValidationError("Line and road must use the same scenario primitive identity")
        if not np.array_equal(selected_wind[:, ::2], selected_wind[:, 1::2]):
            raise InputValidationError("Reverse arcs of one physical corridor require identical local wind")
        speed = np.linalg.norm(selected_wind, axis=-1)
        hazard = np.clip((speed - self.w_thr_m_s) / (self.w_sev_m_s - self.w_thr_m_s), 0, 1)
        multiplier = self.r3_multiplier if regime == "R3" else 1.0
        gamma = (np.zeros_like(self.gamma) if no_coupling
                 else np.minimum(self.gamma_cap, multiplier * self.gamma))
        sensitivity = multiplier * self.kappa_tt
        probability = road_probabilities(speed, self.kappa_blk, self.alpha0, self.alpha1,
                                         gamma, line_replay.outages)
        cumulative, blocked = persistent_blockages(probability.combined, thresholds.arc_uniforms)
        movable, travel = road_travel_times(speed, hazard, blocked, self.t0_min,
                                           sensitivity, self.w_mov_m_s)
        upper = self.t0_min * (1 + sensitivity * np.clip(
            (self.w_mov_m_s - self.w_thr_m_s) / (self.w_sev_m_s - self.w_thr_m_s), 0, 1))
        return RoadReplay(_readonly(selected_times), self.arc_ids, _readonly(speed), _readonly(hazard),
                          probability.wind, probability.combined, cumulative, blocked, movable, travel,
                          _readonly(gamma), _readonly(sensitivity), _readonly(upper), thresholds,
                          float(lead_h), regime, no_coupling)


def _read_csv(root, relative):
    with (root / relative).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _indexed(rows, key):
    indexed = {int(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise InputValidationError("Duplicate source asset identifier")
    return indexed


def load_road_model(case_root, include_connectors=False):
    if type(include_connectors) is not bool:
        raise InputValidationError("include_connectors must be boolean")
    root = Path(case_root)
    hashes = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
              for relative in INPUT_PATHS}
    parameters = {r["code_key"]: r["value"] for r in _read_csv(root, INPUT_PATHS[0])}
    frozen = {"W_thr_m_s": 10., "W_sev_m_s": 25., "d_LR_km": .25,
              "kappa_vul": 1.2, "gamma_blk_bar": .45, "kappa_R3": 1.25}
    if any(float(parameters[key]) != value for key, value in frozen.items()):
        raise InputValidationError("Frozen road generation parameters changed")
    arcs = _indexed(_read_csv(root, INPUT_PATHS[1]), "a")
    corridors = _indexed(_read_csv(root, INPUT_PATHS[2]), "corridor_id")
    pairs = _read_csv(root, INPUT_PATHS[3])
    closed = _indexed(_read_csv(root, INPUT_PATHS[4]), "ell")
    ties = _indexed(_read_csv(root, INPUT_PATHS[5]), "ell")
    samples = _indexed([r for r in _read_csv(root, INPUT_PATHS[6]) if r["index_type"] == "a"], "index_id")
    arc_ids, corridor_ids, line_ids = tuple(sorted(arcs)), tuple(sorted(corridors)), tuple(sorted(closed))
    if (arc_ids != tuple(range(1, 81)) or corridor_ids != tuple(range(1, 41))
            or line_ids != tuple(range(1, 33)) or tuple(sorted(ties)) != tuple(range(33, 38))
            or set(samples) != set(arcs)):
        raise InputValidationError("Expected canonical 80 arcs, 40 corridors, 32 closed lines and five excluded ties")
    fields = {"T0_a_min": "T0_corridor_min", "D_a_km": "D_corridor_km",
              "v0_a_km_h": "v0_corridor_km_h", "kappa_tt_a": "kappa_tt_corridor",
              "W_mov_type_a_m_s": "W_mov_corridor_m_s", "kappa_blk_type_a": "kappa_blk_corridor",
              "alpha_R_type_a_0": "alpha_R0_corridor", "alpha_R_type_a_1": "alpha_R1_corridor",
              "gamma_blk_base_type_a": "gamma_blk_base_corridor", "storm_vulnerable_a": "storm_vulnerable",
              "loc_a_x_km": "loc_corridor_x_km", "loc_a_y_km": "loc_corridor_y_km"}
    class_values = {"main": (.20, -12., .25, 30., .15, .17, .23),
                    "secondary": (.35, -11.25, .25, 27., .25, .27, .33),
                    "local": (.50, -10.5, .25, 25., .35, .37, .43)}
    points, arc_corridor_indices = [], []
    for arc_id in arc_ids:
        row, sample = arcs[arc_id], samples[arc_id]
        corridor_id = int(row["corridor_id"])
        if corridor_id != (arc_id + 1) // 2:
            raise InputValidationError("Arc ordering must be forward/reverse pairs per canonical corridor")
        corridor = corridors[corridor_id]
        endpoints = (int(corridor["m_road"]), int(corridor["n_road"]))
        if arc_id % 2 == 0:
            endpoints = endpoints[::-1]
        if ((int(row["m_road"]), int(row["n_road"])) != endpoints
                or (int(sample["from_index"]), int(sample["to_index"])) != endpoints
                or row["direction"] != ("forward" if arc_id % 2 else "reverse")
                or row["road_class_type_a"] != corridor["road_class_group"]
                or sample["road_class_type_a"] != row["road_class_type_a"]):
            raise InputValidationError("Road direction/endpoints/class disagree across tables")
        for arc_key, corridor_key in fields.items():
            if not np.isclose(float(row[arc_key]), float(corridor[corridor_key]), rtol=0, atol=1e-10):
                raise InputValidationError(f"Arc/corridor property mismatch: {arc_key}")
        road_class = row["road_class_type_a"]
        if road_class not in class_values:
            raise InputValidationError("Unknown canonical road class")
        kblk, intercept, slope, wmov, gamma_base, low_tt, high_tt = class_values[road_class]
        expected = {"kappa_blk_type_a": kblk, "alpha_R_type_a_0": intercept,
                    "alpha_R_type_a_1": slope, "W_mov_type_a_m_s": wmov,
                    "gamma_blk_base_type_a": gamma_base}
        if (any(float(row[key]) != value for key, value in expected.items())
                or not low_tt <= float(row["kappa_tt_a"]) <= high_tt
                or int(row["storm_vulnerable_a"]) not in (0, 1)
                or float(row["T0_a_min"]) <= 0 or float(row["D_a_km"]) <= 0):
            raise InputValidationError("Road class properties differ from frozen contract")
        upper = float(row["T0_a_min"]) * (1 + float(row["kappa_tt_a"]))
        if abs(upper - float(corridor["max_passable_travel_time_min"])) > 2e-6:
            raise InputValidationError("Stored baseline travel upper bound is inconsistent")
        point = (float(row["loc_a_x_km"]), float(row["loc_a_y_km"]))
        if not np.allclose(point, (float(sample["loc_x_km"]), float(sample["loc_y_km"])), rtol=0, atol=1e-10):
            raise InputValidationError("Road sampling coordinates disagree")
        points.append(point)
        arc_corridor_indices.append(corridor_id - 1)
    gamma = np.zeros((len(line_ids), len(arc_ids)))
    seen = set()
    for pair in pairs:
        line_id, arc_id = int(pair["ell"]), int(pair["a"])
        if line_id not in closed or arc_id not in arcs or (line_id, arc_id) in seen:
            raise InputValidationError("Unknown/duplicate coupling pair or fixed-open tie in road hazard")
        seen.add((line_id, arc_id))
        row, line = arcs[arc_id], closed[line_id]
        if (int(pair["is_in_L_a"]) != 1 or float(pair["d_LR_km"]) != frozen["d_LR_km"]
                or not 0 <= float(pair["dist_ell_a_km"]) <= frozen["d_LR_km"]
                or (int(pair["i_bus"]), int(pair["j_bus"])) != (int(line["i_bus"]), int(line["j_bus"]))
                or any(pair[key] != row[key] for key in ("corridor_id", "m_road", "n_road", "road_class_type_a"))):
            raise InputValidationError("Line-road coupling geometry or membership disagrees")
        expected = min(frozen["gamma_blk_bar"], float(row["gamma_blk_base_type_a"]) *
                       (frozen["kappa_vul"] if int(row["storm_vulnerable_a"]) else 1))
        value = float(pair["gamma_blk_ell_a"])
        if abs(value - expected) > 1e-12:
            raise InputValidationError("Effective stored coupling must apply vulnerability multiplier exactly once")
        gamma[line_id - 1, arc_id - 1] = value
    if len(seen) != 112 or not np.array_equal(gamma[:, ::2], gamma[:, 1::2]):
        raise InputValidationError("Expected 56 closed-line/corridor pairs expanded to 112 symmetric arc pairs")
    arrays = [_readonly([float(arcs[a][key]) for a in arc_ids]) for key in
              ("T0_a_min", "kappa_tt_a", "W_mov_type_a_m_s", "kappa_blk_type_a",
               "alpha_R_type_a_0", "alpha_R_type_a_1")]
    identity = canonical_hash({"input_hashes": hashes, "contract": generator_contract()})
    model = RoadModel(arc_ids, corridor_ids, _readonly(arc_corridor_indices, int), _readonly(points),
                      line_ids, _readonly(gamma), *arrays, MappingProxyType(hashes), identity,
                      frozen["W_thr_m_s"], frozen["W_sev_m_s"], frozen["gamma_blk_bar"], frozen["kappa_R3"])
    return _include_connectors(root, model) if include_connectors else model


def _include_connectors(root, model):
    """Extend the canonical model with separately derived physical connectors."""
    from .road_connectors import derive_road_connectors

    connectors = derive_road_connectors(root)
    arcs = _indexed(connectors.arc_rows, "a")
    if tuple(sorted(arcs)) != tuple(range(81, 89)):
        raise InputValidationError("Expected eight directed depot connector arcs 81..88")
    points, indices = [], []
    for arc_id in sorted(arcs):
        row = arcs[arc_id]
        corridor_id = int(row["corridor_id"])
        if corridor_id != (arc_id + 1) // 2:
            raise InputValidationError("Connector threshold IDs must preserve physical corridor IDs 41..44")
        points.append((float(row["loc_a_x_km"]), float(row["loc_a_y_km"])))
        indices.append(corridor_id - 1)
        if arc_id % 2:
            reverse = arcs[arc_id + 1]
            if ((row["m_road"], row["n_road"]) != (reverse["n_road"], reverse["m_road"])
                    or any(row[key] != reverse[key] for key in (
                        "corridor_id", "T0_a_min", "kappa_tt_a", "W_mov_type_a_m_s",
                        "kappa_blk_type_a", "alpha_R_type_a_0", "alpha_R_type_a_1",
                        "loc_a_x_km", "loc_a_y_km"))):
                raise InputValidationError("Reverse connector arcs must represent the same physical corridor")
    gamma = np.zeros((len(model.line_ids), 8))
    seen = set()
    for row in connectors.coupling_rows:
        line_id, arc_id = int(row["ell"]), int(row["a"])
        value = float(row["gamma_blk_ell_a"])
        if (line_id not in model.line_ids or arc_id not in arcs or (line_id, arc_id) in seen
                or not 0 <= value <= model.gamma_cap):
            raise InputValidationError("Invalid derived connector coupling or fixed-open tie")
        seen.add((line_id, arc_id))
        gamma[line_id - 1, arc_id - 81] = value
    if not np.array_equal(gamma[:, ::2], gamma[:, 1::2]):
        raise InputValidationError("Connector coupling must be shared by reverse arcs")
    updated = {}
    for attribute, key in (("t0_min", "T0_a_min"), ("kappa_tt", "kappa_tt_a"),
                           ("w_mov_m_s", "W_mov_type_a_m_s"), ("kappa_blk", "kappa_blk_type_a"),
                           ("alpha0", "alpha_R_type_a_0"), ("alpha1", "alpha_R_type_a_1")):
        updated[attribute] = _readonly(np.concatenate([
            getattr(model, attribute), [float(arcs[a][key]) for a in sorted(arcs)]]))
    hashes = dict(model.input_hashes)
    for key, value in connectors.input_hashes.items():
        if key in hashes and hashes[key] != value:
            raise InputValidationError("Connector and canonical inputs disagree on source provenance")
        hashes[key] = value
    identity = canonical_hash({"input_hashes": hashes, "contract": generator_contract(),
                               "canonical_model_sha256": model.model_sha256,
                               "connector_provenance_sha256": connectors.provenance_sha256,
                               "include_connectors": True})
    return replace(model, arc_ids=tuple(range(1, 89)), corridor_ids=tuple(range(1, 45)),
                   arc_corridor_indices=_readonly(np.concatenate([model.arc_corridor_indices, indices]), int),
                   points_xy_km=_readonly(np.vstack([model.points_xy_km, points])),
                   gamma=_readonly(np.column_stack([model.gamma, gamma])),
                   input_hashes=MappingProxyType(hashes), model_sha256=identity, **updated)


def generator_contract():
    return {
        "version": "S01-I04-v1",
        "domain": "Default: 80 canonical directed arcs = 40 physical corridors. include_connectors=True appends supplied depot corridors 41..44 as arcs 81..88; canonical hazard/features remain first 80 arcs",
        "connectors": "Use supplied auxiliary road parameters with geometry-derived midpoint/coupling and own corridor thresholds; current four connectors have no line within 0.25km",
        "threshold": "One U[0,1) per physical corridor; reverse arcs share it; different corridors independent conditional on upstream wind/line state",
        "stream": "road_uniform mechanism 8, stable positive asset_index=corridor_id; no target/lead/method/regime seed",
        "probability": "qwind=kappa_blk*logistic(alpha0+alpha1*speed); qblk=1-(1-qwind)*product_line(1-gamma*quarter_hour_outage)",
        "coupling": "Read already vulnerability-adjusted stored gamma; 56 line/corridor pairs expand to 112 line/arc pairs; exclude five fixed-open ties",
        "persistence": "Slice to actual assignment -lead before running max; blocked=(uniform<=runningmax); no clearance",
        "movement": "movable=(not blocked) and speed<=Wmov; temporary unsafe wind may recover without road clearance",
        "travel": "T0_minutes*(1+kappa_tt*clip((speed-10)/15,0,1)) if movable, otherwise +inf",
        "R3": "Reuse R1 wind and line/road innovations; gamma=min(.45,1.25*stored_gamma); kappa_tt=1.25*stored_kappa_tt without clipping; recompute finite travel upper bound",
        "no_coupling": "Set all gamma to zero without changing threshold draws or wind/line states",
        "time_grid": "Quarter-hour slot starts from actual assignment through 23.75h; line state must match every slot",
        "OOS": "Formal physical draws/replay closed until S04 plan-lock API",
    }
