"""Independent raw-witness audit of the finite-support probability LP.

No probability solver, LP matrices, or solver certificate helpers are imported.
The paper's alpha is the LOWER-moment multiplier and beta is the UPPER one.
An infeasible solver status alone is never an infeasibility certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    _require(result.shape == shape and np.isfinite(result).all(),
             f"{name} must be finite with shape {shape}")
    return result


def _ids(value, name, allow_empty=False):
    _require(value is not None and not isinstance(value, (str, bytes)), f"Invalid {name}")
    ids = tuple(value)
    _require((allow_empty or len(ids) > 0) and
             all(isinstance(v, str) and bool(v) for v in ids) and len(set(ids)) == len(ids),
             f"{name} must contain unique nonempty strings")
    return ids


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json(value):
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ProbabilityCheckReport:
    passed: bool
    primal_feasible: bool
    dual_feasible: bool
    gap_passed: bool
    status: str
    residuals: Mapping
    objective_components: Mapping
    bounds: Mapping
    separation: Mapping
    reasons: tuple[str, ...]
    identities: Mapping
    tolerances: Mapping
    infeasibility_certified: bool = False

    def as_dict(self):
        return {name: _json(getattr(self, name)) for name in self.__dataclass_fields__}


def check_probability_solution(probability_set, q, result, *, q_support_ids,
                               feasibility_tolerance=1e-8,
                               objective_abs_tolerance=1e-4,
                               objective_rel_tolerance=1e-7,
                               separation_abs_tolerance=1e-4,
                               separation_rel_tolerance=1e-8):
    """Check raw p/theta0/alpha/beta, identities, and any reported bounds.

    Invalid input identity/shape/nonfinite values raise ValueError. Finite but
    infeasible witnesses return passed=False without clipping or reordering.
    Missing witnesses return NEED_WITNESSES, even for an INFEASIBLE status.
    Feasibility requires BOTH original-unit and row-normalized 1e-8 checks.
    Separation uses its distinct paper tolerance and cannot excuse a failed
    dual-feasibility check. No constraint restricts the sign of free theta0.
    """
    tolerances = {"feasibility": float(feasibility_tolerance),
                  "objective_absolute_usd": float(objective_abs_tolerance),
                  "objective_relative": float(objective_rel_tolerance),
                  "separation_absolute_usd": float(separation_abs_tolerance),
                  "separation_relative": float(separation_rel_tolerance)}
    _require(all(np.isfinite(v) and v > 0 for v in tolerances.values()), "Invalid audit tolerance")
    ids = _ids(_get(probability_set, "support_ids"), "support_ids")
    feature_ids = _ids(_get(probability_set, "feature_ids"), "feature_ids", True)
    s, f = len(ids), len(feature_ids)
    phi = _array(_get(probability_set, "features"), (s, f), "features")
    lower = _array(_get(probability_set, "lower"), (f,), "lower")
    upper = _array(_get(probability_set, "upper"), (f,), "upper")
    _require(np.all(lower <= upper), "Inverted moment interval")
    qids = _ids(q_support_ids, "q_support_ids")
    _require(qids == ids, "q support axis mismatch; never reorder silently")
    q = _array(q, (s,), "q")
    _require(tuple(_get(result, "support_ids", ())) == ids, "Result support axis mismatch")
    _require(tuple(_get(result, "feature_ids", ())) == feature_ids, "Result feature axis mismatch")
    ambiguity_hash = _hash({"support_ids": list(ids), "feature_ids": list(feature_ids),
                            "features": phi.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
                            "feature_orientation": "scenario_by_feature",
                            "dual_convention": "alpha_lower_beta_upper"})
    q_hash = _hash({"support_ids": list(ids), "q": q.tolist()})
    supplied_set_hash = _get(probability_set, "ambiguity_sha256")
    _require(supplied_set_hash is None or supplied_set_hash == ambiguity_hash,
             "Probability-set hash disagrees with raw inputs")
    _require(_get(result, "ambiguity_sha256") == ambiguity_hash, "Result ambiguity hash mismatch")
    _require(_get(result, "q_sha256") == q_hash, "Result q hash mismatch")
    identities = {"support_ids": ids, "feature_ids": feature_ids,
                  "ambiguity_sha256": ambiguity_hash, "q_sha256": q_hash,
                  "reported_status": str(_get(result, "status", "UNSPECIFIED")),
                  "reported_certified_optimal": bool(_get(result, "certified_optimal", False)),
                  "dual_convention": "alpha_lower_beta_upper"}
    residuals, reasons = {}, []

    def record(name, excess, scale, unit, equation):
        raw = np.asarray(excess, dtype=float)
        _require(np.isfinite(raw).all(), f"Nonfinite arithmetic in {name}")
        denominator = np.broadcast_to(np.maximum(np.asarray(scale, dtype=float), 1.), raw.shape)
        _require(np.isfinite(denominator).all(), f"Nonfinite scale in {name}")
        violation = np.maximum(raw, 0.)
        normalized = violation / denominator
        passed = bool(np.all(violation <= feasibility_tolerance) and
                      np.all(normalized <= feasibility_tolerance))
        residuals[name] = {"passed": passed, "unit": unit, "paper_equation": equation,
                           "constraint_count": int(raw.size),
                           "max_violation": float(np.max(violation, initial=0.)),
                           "max_normalized_violation": float(np.max(normalized, initial=0.)),
                           "original_tolerance": float(feasibility_tolerance),
                           "normalized_tolerance": float(feasibility_tolerance)}
        if not passed:
            reasons.append(name)
        return passed

    primal_eq, dual_eq = "eq:joint_worst_case_primal", "eq:joint_distribution_dual"
    raw_p = _get(result, "probabilities")
    p = None if raw_p is None else _array(raw_p, (s,), "probabilities")
    dual_fields = [_get(result, name) for name in ("theta0", "alpha", "beta")]
    dual_present = all(value is not None for value in dual_fields)
    if any(value is not None for value in dual_fields) and not dual_present:
        reasons.append("incomplete_dual_witness")
    pobj = dobj = moments = lhs = alpha = beta = None
    primal_feasible = dual_feasible = False
    if p is not None:
        moments = phi.T @ p
        checks = [record("probability_nonnegative", -p, 1., "probability", primal_eq),
                  record("probability_normalization", abs(float(np.sum(p)) - 1.),
                         np.sum(np.abs(p)), "probability", primal_eq)]
        moment_scale = np.maximum(np.maximum(np.abs(lower), np.abs(upper)), np.abs(phi).T @ np.abs(p))
        checks.extend([record("moment_lower", lower-moments, moment_scale, "feature", primal_eq),
                       record("moment_upper", moments-upper, moment_scale, "feature", primal_eq)])
        primal_feasible = all(checks)
        pobj = float(q @ p)
        _require(np.isfinite(pobj), "Nonfinite primal objective")
    else:
        reasons.append("missing_primal_witness")
    qscale = max(1., float(np.max(np.abs(q))))
    sep_tol = float(separation_abs_tolerance + separation_rel_tolerance*qscale)
    separation = {"scale_usd": qscale, "tolerance_usd": sep_tol, "passed": False,
                  "max_positive_residual_usd": None, "residuals_usd": None,
                  "support_ids": ids, "scope": "all support rows, including previously included rows"}
    if dual_present:
        theta = float(_array(dual_fields[0], (), "theta0"))
        alpha = _array(dual_fields[1], (f,), "alpha")
        beta = _array(dual_fields[2], (f,), "beta")
        lhs = theta + phi @ (beta-alpha)
        row_scale = np.maximum(np.abs(q), abs(theta)+np.abs(phi) @ (np.abs(alpha)+np.abs(beta)))
        checks = [record("alpha_nonnegative", -alpha, np.abs(alpha), "USD/feature", dual_eq),
                  record("beta_nonnegative", -beta, np.abs(beta), "USD/feature", dual_eq),
                  record("dual_full_support", q-lhs, row_scale, "USD", dual_eq)]
        dual_feasible = all(checks)
        dobj = float(theta - lower @ alpha + upper @ beta)
        _require(np.isfinite(dobj), "Nonfinite dual objective")
        positive = float(np.max(np.maximum(q-lhs, 0.), initial=0.))
        separation.update(passed=positive <= sep_tol, max_positive_residual_usd=positive,
                          residuals_usd=(q-lhs).tolist())
    else:
        reasons.append("missing_dual_witness")

    objscale = max(1., abs(pobj) if pobj is not None else 0., abs(dobj) if dobj is not None else 0.)
    objtol = float(objective_abs_tolerance + objective_rel_tolerance*objscale)
    gap = None if pobj is None or dobj is None else float(dobj-pobj)
    gap_passed = gap is not None and abs(gap) <= objtol
    if not gap_passed:
        reasons.append("primal_dual_gap")
    complementarity = None
    complementarity_passed = False
    if p is not None and dual_present:
        values = np.concatenate((np.abs(alpha*(moments-lower)), np.abs(beta*(upper-moments)),
                                 np.abs(p*(lhs-q))))
        _require(np.isfinite(values).all(), "Nonfinite complementarity")
        comp_max = float(np.max(values, initial=0.))
        complementarity_passed = comp_max <= objtol
        complementarity = {"maximum_absolute_usd": comp_max, "sum_absolute_usd": float(values.sum()),
                           "tolerance_usd": objtol, "passed": complementarity_passed}
        if not complementarity_passed:
            reasons.append("complementarity")

    claims = {}
    claims_passed = True
    for key, rebuilt, feasible, comparison in (
            ("primal_objective_usd", pobj, p is not None, "equal"),
            ("dual_objective_usd", dobj, dual_present, "equal"),
            ("lower_bound_usd", pobj, primal_feasible, "lower"),
            ("upper_bound_usd", dobj, dual_feasible, "upper")):
        value = _get(result, key)
        if value is None:
            claims[key] = {"reported": None, "passed": True}
            continue
        value = float(_array(value, (), key))
        tol = objective_abs_tolerance + objective_rel_tolerance*max(1., abs(value), abs(rebuilt or 0.))
        error = None if rebuilt is None else (abs(value-rebuilt) if comparison == "equal" else
                    max(0., value-rebuilt) if comparison == "lower" else max(0., rebuilt-value))
        ok = bool(feasible and error is not None and error <= tol)
        claims[key] = {"reported": value, "recomputed": rebuilt, "violation_usd": error,
                       "tolerance_usd": float(tol), "passed": ok}
        claims_passed = claims_passed and ok
        if not ok:
            reasons.append("reported_"+key)

    bounds = {"certified_lower_bound_usd": pobj if primal_feasible else None,
              "certified_upper_bound_usd": dobj if dual_feasible else None,
              "minimum_scenario_cost_usd": float(q.min()), "maximum_scenario_cost_usd": float(q.max()),
              "nonemptiness_witness_available": primal_feasible,
              "reported_claims": claims}
    objectives = {"primal_usd": pobj, "dual_usd": dobj, "signed_gap_usd": gap,
                  "absolute_gap_usd": None if gap is None else abs(gap),
                  "objective_scale_usd": objscale, "objective_tolerance_usd": objtol,
                  "feature_means": None if moments is None else moments.tolist(),
                  "complementarity": complementarity}
    passed = bool(primal_feasible and dual_feasible and gap_passed and
                  complementarity_passed and claims_passed)
    status = "PASS_RAW_PROBABILITY_CERTIFICATE" if passed else (
        "NEED_WITNESSES" if p is None or not dual_present else "FAIL_RAW_PROBABILITY_CERTIFICATE")
    return ProbabilityCheckReport(passed, primal_feasible, dual_feasible, bool(gap_passed), status,
                                  _freeze(residuals), _freeze(objectives), _freeze(bounds),
                                  _freeze(separation), tuple(reasons), _freeze(identities), _freeze(tolerances))
