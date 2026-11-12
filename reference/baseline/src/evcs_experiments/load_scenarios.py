"""Hourly feeder-wide load errors, using exact whole-vector rejection.

The input baseline is already scaled by 0.8. Archived rounded Q is retained
for audit. Model Q uses the authorized Q1 exact rated-power-factor mapping,
which preserves complete recourse when all active load is shed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .case_inputs import InputValidationError
from .evaluation_access import require_oos_access
from .random_streams import canonical_hash, stream

INPUT_PATHS = (
    "01_config/model_parameters__paper_symbols.csv",
    "03_load/P_base_Q_base__N_T.csv",
    "03_load/Ptilde_base_Qtilde_base__N_T.csv",
    "03_load/Pbase_system_Qbase_system_lambda_load__T.csv",
    "03_load/load_scaling_metadata.json",
    "02_power_network/N_P_rated_Q_rated__bus_parameters.csv",
    "03_load/lambda_load__T.csv",
    "01_config/reactive_support__v54_q1.json",
    "tools/reactive_support_model.py",
)


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def covariance_matrix(sigma_load=.03, rho_load=.8, hours=24):
    """Untruncated Gaussian covariance, with lag measured in hours."""
    if (isinstance(hours, bool) or not isinstance(hours, Integral) or hours < 1
            or not np.isfinite(sigma_load) or sigma_load <= 0
            or not np.isfinite(rho_load) or not 0 <= rho_load < 1):
        raise InputValidationError("Invalid hourly load covariance parameters")
    time = np.arange(hours)
    return _readonly(sigma_load ** 2 * rho_load ** np.abs(time[:, None] - time))


def reject_multivariate_normal(rng, cholesky, lower, upper, max_proposals=100000):
    """Condition N(0, L L.T) on the entire box, without clipping or redrawing cells.

    The proposal counter includes the accepted draw. A fake RNG implementing
    standard_normal(n) can be used to verify rejection semantics independently.
    """
    factor = np.asarray(cholesky, dtype=float)
    if (factor.ndim != 2 or factor.shape[0] == 0 or factor.shape[0] != factor.shape[1]
            or not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0)
            or not np.array_equal(factor, np.tril(factor))):
        raise InputValidationError("Expected a finite nonsingular lower triangular factor")
    try:
        lo, hi = (np.broadcast_to(np.asarray(v, dtype=float), (len(factor),))
                  for v in (lower, upper))
    except ValueError as exc:
        raise InputValidationError("Bounds must be scalar or one per hour") from exc
    if (not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo >= hi)
            or isinstance(max_proposals, bool) or not isinstance(max_proposals, Integral)
            or max_proposals < 1):
        raise InputValidationError("Invalid rejection bounds or proposal limit")
    for count in range(1, max_proposals + 1):
        standard = np.asarray(rng.standard_normal(len(factor)), dtype=float)
        if standard.shape != (len(factor),) or not np.isfinite(standard).all():
            raise InputValidationError("RNG returned malformed standard innovation")
        epsilon = factor @ standard
        if np.all((epsilon >= lo) & (epsilon <= hi)):
            return _readonly(standard), _readonly(epsilon), count
    raise RuntimeError("Whole-vector load rejection exhausted the explicit proposal limit")


@dataclass(frozen=True)
class LoadModel:
    bus_ids: tuple[int, ...]
    hours_h: np.ndarray
    baseline_p_kw: np.ndarray
    baseline_q_kvar: np.ndarray
    archived_baseline_q_kvar: np.ndarray
    tan_phi_load: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    sigma_load: float
    rho_load: float
    error_bounds: tuple[float, float]
    input_hashes: Mapping[str, str]
    model_sha256: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class LoadDraw:
    accepted_standard_innovation: np.ndarray
    epsilon: np.ndarray
    proposal_count: int
    rejection_count: int
    stream_id: str
    model_sha256: str
    split: str
    development: bool
    base_index: int
    variant_index: int
    draw_sha256: str

    def as_dict(self):
        return {"accepted_standard_innovation": self.accepted_standard_innovation.tolist(),
                "epsilon": self.epsilon.tolist(), "proposal_count": self.proposal_count,
                "rejection_count": self.rejection_count, "stream_id": self.stream_id,
                "model_sha256": self.model_sha256, "split": self.split,
                "development": self.development, "base_index": self.base_index,
                "variant_index": self.variant_index, "draw_sha256": self.draw_sha256}


@dataclass(frozen=True)
class LoadSample:
    p_kw: np.ndarray
    q_kvar: np.ndarray
    archived_q_kvar: np.ndarray
    phi_load: float
    draw_sha256: str
    model_sha256: str


def _identity(split, development, base_index, variant_index):
    if type(development) is not bool:
        raise InputValidationError("development must be an explicit boolean")
    if split == "oos" and not development:
        require_oos_access(base_index=base_index, variant_index=variant_index)
    identity = stream(split, base_index, variant_index, "load_innovation", development=development)
    if not development and variant_index >= {"planning": 20, "calibration": 10, "oos": 50}[split]:
        raise InputValidationError("Variant exceeds the frozen formal bank")
    return identity


def _draw_hash(standard, epsilon, count, identity, model, split, development, base_index, variant_index):
    return canonical_hash({"standard_innovation": standard.tolist(), "epsilon": epsilon.tolist(),
                           "proposal_count": count, "stream_id": identity.stream_id,
                           "model_sha256": model.model_sha256, "split": split,
                           "development": development, "base_index": int(base_index),
                           "variant_index": int(variant_index)})


def draw_load_errors(model, base_index, variant_index, *, split="planning", development=False):
    if not isinstance(model, LoadModel):
        raise InputValidationError("Expected a LoadModel")
    identity = _identity(split, development, base_index, variant_index)
    standard, epsilon, count = reject_multivariate_normal(
        identity.generator(), model.cholesky, *model.error_bounds)
    digest = _draw_hash(standard, epsilon, count, identity, model, split,
                        development, base_index, variant_index)
    return LoadDraw(standard, epsilon, count, count - 1, identity.stream_id,
                    model.model_sha256, split, development, int(base_index),
                    int(variant_index), digest)


def realize_load(model, draw):
    if not isinstance(model, LoadModel) or not isinstance(draw, LoadDraw):
        raise InputValidationError("Expected LoadModel and LoadDraw")
    identity = _identity(draw.split, draw.development, draw.base_index, draw.variant_index)
    if draw.model_sha256 != model.model_sha256 or draw.stream_id != identity.stream_id:
        raise InputValidationError("Load draw belongs to a different model or stream")
    epsilon, standard = np.asarray(draw.epsilon), np.asarray(draw.accepted_standard_innovation)
    if (epsilon.shape != (24,) or standard.shape != (24,)
            or not np.isfinite(epsilon).all() or not np.isfinite(standard).all()
            or np.any(epsilon < model.error_bounds[0]) or np.any(epsilon > model.error_bounds[1])
            or not np.allclose(epsilon, model.cholesky @ standard, rtol=0, atol=1e-14)
            or type(draw.proposal_count) is not int or draw.proposal_count < 1
            or type(draw.rejection_count) is not int or draw.rejection_count != draw.proposal_count - 1):
        raise InputValidationError("Malformed load draw")
    if draw.draw_sha256 != _draw_hash(standard, epsilon, draw.proposal_count, identity,
                                     model, draw.split, draw.development,
                                     draw.base_index, draw.variant_index):
        raise InputValidationError("Load draw hash mismatch")
    p = model.baseline_p_kw * (1 + epsilon[:, None])
    q = model.baseline_q_kvar * (1 + epsilon[:, None])
    archived_q = model.archived_baseline_q_kvar * (1 + epsilon[:, None])
    return LoadSample(_readonly(p), _readonly(q), _readonly(archived_q), float(p.sum() / model.baseline_p_kw.sum()),
                      draw.draw_sha256, model.model_sha256)


def _csv(root, relative):
    with (root / relative).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _indexed(rows, columns, expected):
    indexed = {}
    try:
        for row in rows:
            key = tuple(int(row[c]) for c in columns)
            if key in indexed:
                raise InputValidationError("Duplicate load table primary key")
            indexed[key] = row
    except (KeyError, ValueError, TypeError) as exc:
        raise InputValidationError("Invalid load table primary key") from exc
    if set(indexed) != expected:
        raise InputValidationError("Load table does not cover the complete canonical keys")
    return indexed


def _numbers(values):
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputValidationError("Non-numeric load table value") from exc
    if not np.isfinite(array).all() or np.any(array < 0):
        raise InputValidationError("Load table values must be finite and nonnegative")
    return array


def load_load_model(case_root):
    root = Path(case_root)
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in INPUT_PATHS}
    parameter_rows = _csv(root, INPUT_PATHS[0])
    parameters = {r["code_key"]: r["value"] for r in parameter_rows}
    if len(parameters) != len(parameter_rows):
        raise InputValidationError("Duplicate parameter code key")
    sigma, rho, lower, upper, scale = (float(parameters[k]) for k in
        ("sigma_load", "rho_load", "epsilon_load_under", "epsilon_load_bar", "kappa_load"))
    if (sigma, rho, lower, upper, scale) != (.03, .8, -.09, .09, .8):
        raise InputValidationError("Frozen load parameters changed")
    keys = {(t, n) for t in range(1, 25) for n in range(1, 34)}
    base = _indexed(_csv(root, INPUT_PATHS[1]), ("t", "n"), keys)
    unscaled = _indexed(_csv(root, INPUT_PATHS[2]), ("t", "n"), keys)
    hour_keys = {(t,) for t in range(1, 25)}
    totals = _indexed(_csv(root, INPUT_PATHS[3]), ("t",), hour_keys)
    metadata = json.loads((root / INPUT_PATHS[4]).read_text())
    buses = _indexed(_csv(root, INPUT_PATHS[5]), ("n",), {(n,) for n in range(1, 34)})
    lambdas = _indexed(_csv(root, INPUT_PATHS[6]), ("t",), hour_keys)
    ordered = [(t, n) for t in range(1, 25) for n in range(1, 34)]
    p, q = (_numbers([base[k][column] for k in ordered]).reshape(24, 33)
            for column in ("P_base_n_t_kW", "Q_base_n_t_kvar"))
    original_p, original_q = (_numbers([unscaled[k][column] for k in ordered]).reshape(24, 33)
            for column in ("P_tilde_base_n_t_kW", "Q_tilde_base_n_t_kvar"))
    scales = _numbers([base[k]["kappa_load"] for k in ordered])
    if not np.all(scales == scale) or metadata["active_case_load_scale"] != scale:
        raise InputValidationError("Load scale metadata and cell table disagree")
    p_difference = float(np.max(np.abs(p - scale * original_p)))
    q_difference = float(np.max(np.abs(q - scale * original_q)))
    if max(p_difference, q_difference) > 1e-10:
        raise InputValidationError("Baseline is not exactly the archived 0.8-scaled load")
    system_p, system_q = (_numbers([totals[(t,)][column] for t in range(1, 25)])
            for column in ("P_base_system_t_kW", "Q_base_system_t_kvar"))
    if (not np.allclose(p.sum(axis=1), system_p, rtol=0, atol=1e-9)
            or not np.allclose(q.sum(axis=1), system_q, rtol=0, atol=1e-9)
            or np.any(system_p <= 0)):
        raise InputValidationError("Bus load sums disagree with positive hourly system totals")
    temporal_shape = _numbers([totals[(t,)]["lambda_load_t"] for t in range(1, 25)])
    source_shape = _numbers([lambdas[(t,)]["lambda_load_t"] for t in range(1, 25)])
    if (not np.array_equal(temporal_shape, source_shape)
            or not np.allclose(temporal_shape, system_p / system_p.max(), rtol=0, atol=1e-9)):
        raise InputValidationError("Hourly normalized shape disagrees with system totals")
    rated_p, rated_q = (_numbers([buses[(n,)][column] for n in range(1, 34)])
            for column in ("P_rated_n_kW", "Q_rated_n_kvar"))
    zero_bus = rated_p == 0
    if (np.any(rated_q[zero_bus] != 0) or np.any(p[:, zero_bus] != 0)
            or np.any(q[:, zero_bus] != 0) or np.any(p[:, ~zero_bus] <= 0)):
        raise InputValidationError("Rated and time-dependent zero-load buses disagree")
    rated_ratio = np.divide(rated_q, rated_p, out=np.zeros(33), where=rated_p > 0)
    q1_settings = json.loads((root / INPUT_PATHS[7]).read_text())
    if (q1_settings["revision"] != "v54-q1"
            or q1_settings["load_Q_mapping"] != "P_base * Q_rated / P_rated; zero at zero-load bus"):
        raise InputValidationError("Expected the authorized Q1 exact rated-power-factor mapping")
    q_model = p * rated_ratio
    diagnostics = MappingProxyType({"source_cells": len(base), "hourly_points": 24,
        "bus_count": 33, "baseline_already_scaled": True, "load_scale": scale,
        "max_p_scaling_residual_kw": p_difference, "max_q_scaling_residual_kvar": q_difference,
        "max_hourly_p_sum_residual_kw": float(np.max(np.abs(p.sum(axis=1) - system_p))),
        "max_hourly_q_sum_residual_kvar": float(np.max(np.abs(q.sum(axis=1) - system_q))),
        "max_archived_q_minus_exact_rated_ratio_kvar": float(np.max(np.abs(q - p * rated_ratio))),
        "zero_load_bus_ids": tuple(n + 1 for n in np.flatnonzero(zero_bus).tolist()),
        "reactive_mapping": "Q1 model Q=P_base*(Q_rated/P_rated); archived Q retained separately.",
        "q1_total_q_adjustment_kvar_hours": float(np.sum(q_model - q))})
    covariance = covariance_matrix(sigma, rho)
    digest = canonical_hash({"input_hashes": hashes, "contract": generator_contract()})
    return LoadModel(tuple(range(1, 34)), _readonly(np.arange(24)), _readonly(p), _readonly(q_model),
        _readonly(q), _readonly(rated_ratio),
        covariance, _readonly(np.linalg.cholesky(covariance)), sigma, rho, (lower, upper),
        MappingProxyType(hashes), digest, diagnostics)


def generator_contract():
    return {"version": "S01-I05-v1", "native_timestep_h": 1,
        "hour_start_h": list(range(24)), "array_axes": ["hour", "bus"],
        "distribution": "N(0, Sigma) conditioned jointly on [-0.09,0.09]^24",
        "covariance": "Sigma[t,t'] = 0.03^2 * 0.8^abs(t-t') for hourly indices",
        "sampler": "Whole 24-dimensional standard-normal proposals; epsilon = cholesky(Sigma) @ z; reject entire proposal if any coordinate lies outside closed box.",
        "mechanism": "load_innovation", "mechanism_id": 9,
        "scenario_identity": ["split", "development", "base_index", "variant_index"],
        "shared_across": ["target", "regime", "lead", "method", "fleet_size", "support_size"],
        "independent_of": ["wind", "line_gaussian", "road_uniform"],
        "scale_policy": "Stored baseline already includes kappa_load=0.8; apply only 1+epsilon per hour to every P/Q cell.",
        "reactive_policy": "Authorized Q1 exact rated ratio Q=P*(Q_rated/P_rated); retain archived Q and common-hourly-scaling trace separately.",
        "phi_load": "sum(realized P) / sum(baseline P), range [0.91,1.09]; may exceed one",
        "formal_oos": "Draw and realization locked until S04; development namespace permitted",
        "proposal_limit": 100000}
