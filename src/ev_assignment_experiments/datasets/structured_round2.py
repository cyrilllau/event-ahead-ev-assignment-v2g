"""Round 2 fixed-load candidate-pool expansion; no solves and no input writes.

``make_fixed_load_case`` generates the frozen C11 revision1 N6/G6 physics for
the requested support, then changes available counts only. For a saved R0
pickle, the caller loads/checks that exact artifact and passes its dictionary
to ``expand_available_fleet``; this module does not regenerate R0 or load pickle.

The healthy-grid check tests a sufficient witness against the ORIGINAL matrix
at zero EV injection AND zero shedding. Witness failure means
NEED_ORIGINAL_FEASIBILITY_LP, never model infeasibility. The caller owns that LP,
all charged computation, and its evidence. No old bounds/cuts are transferred.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from ev_assignment.l08 import certificates, matrix
from ev_assignment_experiments.datasets import structured


FAMILY = "structured_fixed_load_round2_v1"
BASE_CONFIG = dict(gap_h=2., early_duration_h=.5, joint_nominal=.08, revision=1)
TOLERANCE = 1e-9


def _pool(fleet_size, groups):
    n = structured._integer(fleet_size, "fleet_size", 6)
    g = structured._integer(groups, "groups")
    if g != 6:
        raise ValueError("Fixed C11 preserves six physical types. G3 would merge distinct "
                         "SOC/trajectories; G15/G30 label splitting is not genuine "
                         "heterogeneity. A separate declared stress protocol is required.")
    return [n // 6 + int(j < n % 6) for j in range(6)]


def _source_hash(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _validate_base(case):
    if (case.get("config") != BASE_CONFIG or case.get("type_counts") != [1] * 6
            or case.get("fleet_scale") != 1. or len(case.get("packages", [])) != 72
            or len(case.get("types", [])) != 6):
        raise ValueError("Expected the exact C11 revision1 N6/G6 base case, not a scaled case")
    if not isinstance(case.get("identity"), dict) or not case["identity"].get("sha256"):
        raise ValueError("Base case needs its original scientific identity")
    for g, typ in enumerate(case["types"]):
        if typ["g"] != g or typ["count"] != 1:
            raise ValueError("Base type order/count changed")
        if sum(p["g"] == g for p in case["packages"]) != 12:
            raise ValueError("Base catalogue must retain twelve packages per type")
    expected = np.zeros((96, 32), bool)
    expected[68:70, 19] = True
    forecast = case["forecast"]["physical"]
    if (not np.array_equal(forecast["line_operating_outages"], expected)
            or forecast["road_blocked"].any()):
        raise ValueError("Base forecast is not the frozen core-only late outage")
    if not isinstance(case.get("physical_audit"), dict):
        raise ValueError("Base generator physical audit is required")


def make_fixed_load_case(source_root, split, seed, scenario_count, fleet_size=30, groups=6):
    """Return the usual case dictionary with a NEW pool/science identity.

    Actual generation/auditing belongs inside the caller's charged wrapper.
    N and G never enter the frozen physics generator: it always receives N6/G6.
    Support size still addresses a new TRAIN bank; CAL retains its frozen rule.
    Only six physical types are supported; unsupported G fails BEFORE data IO.
    """
    _pool(fleet_size, groups)
    base = structured.make_case(dict(BASE_CONFIG), source_root, split, seed,
                                scenario_count=scenario_count, fleet_size=6, groups=6)
    return expand_available_fleet(base, fleet_size=fleet_size, groups=groups)


def expand_available_fleet(base_case, fleet_size=30, groups=6):
    """Expand a supplied frozen N6 case without drawing anything or mutating it.

    At N30 counts become [5]*6. Nonmultiples allocate the remainder in original
    type order, so successive pools are nested. Coefficients and scenario IDs
    remain exact; changed first-stage domains require a new Run/certificate.
    Healthy-grid ORIGINAL-matrix audits are new work, not reused solve evidence.
    """
    counts = _pool(fleet_size, groups)
    _validate_base(base_case)
    case = deepcopy(base_case)
    parent_identity = deepcopy(base_case["identity"])
    case["type_counts"] = counts
    for typ, count in zip(case["types"], counts):
        typ["count"] = count
    case["fixed_load_round2"] = True
    case["fleet_scale"] = 1.
    case["source_generator_audit"] = deepcopy(base_case["physical_audit"])
    case["source_revision_plan"] = case.pop("revision_plan", None)
    scope = case["feature_scope"]
    case["source_feature_scope"] = deepcopy(scope)
    # Preserve historical wording in source_feature_scope, not as the active rule.
    scope["differences"] = [v for v in scope["differences"] if not v.startswith((
        "Revision1 scales only", "Revision1 references use", "Original three-key RNG addresses"))]
    scope.update(model_name="P3-DRO", marginal_model_name="P2-DRO", round=2,
                 available_fleet_size=sum(counts), physical_load_fleet_size=6,
                 fleet_scale=1., reference_units_per_target=1,
                 parent_revision_plan_sha256=scope.pop("revision_plan_sha256", None))
    scope["differences"] += [
        "Round2 fixes C11 revision1 N6 bus/time demand and all physical generation rules; N only changes available type counts.",
        "Reference deployments remain the same N6 one/two-car vectors, not ceil(N/6). Optional vehicles may remain unused.",
        "This is P3/P2 synthetic moment-domain computation, not original 322311-row Full ST-DRO; no physical R3 revision is implemented.",
    ]
    scope["group_stress"] = "NOT_IMPLEMENTED: do not recompute SOC offsets using a different G"
    case["pool_expansion"] = dict(
        base_case_identity=parent_identity["sha256"], source_type_counts=[1] * 6,
        type_counts=counts, physical_groups=6, physical_load_fleet_size=6,
        packages_unchanged=True, type_coefficients_unchanged=True,
        physical_arrays_unchanged=True, route_states_unchanged=True,
        weather_and_calibration_unchanged=True, scenario_order_and_weights_unchanged=True,
        random_draws_during_expansion=0, reference_policy_counts=[1, 1, 2],
        interpretation="fixed demand, expanded optional candidate pool; not a larger service workload",
        monotonicity="V_expanded <= V_base only for identical support/risk/fees and certified optima or compatible bounds",
        inherited_bounds_or_cuts=False,
    )
    identity = deepcopy(parent_identity)
    for k in ("sha256", "revision_plan_sha256"):
        identity.pop(k, None)
    identity.update(family=FAMILY, round=2, fleet_size=sum(counts), groups=6,
                    type_counts=counts, fleet_scale=1., physical_load_fleet_size=6,
                    parent_case_identity=parent_identity["sha256"],
                    parent_generator_code_sha256=parent_identity.get("code_sha256"),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    original_matrix_code_sha256=_source_hash(matrix),
                    row_audit_code_sha256=_source_hash(certificates),
                    features=deepcopy(scope), pool_expansion=deepcopy(case["pool_expansion"]),
                    moment_scope="P3-DRO / P2-DRO", prior_bound_reuse_allowed=False)
    identity["sha256"] = structured._digest(identity)
    case["identity"] = identity
    case["reference_policies"] = make_reference_policies(case)
    healthy = audit_healthy_grid_zero_ev(case)
    case["healthy_grid_audit"] = healthy
    original_errors = list(case["source_generator_audit"].get("errors", []))
    other_errors = [e for e in original_errors if e != "ZERO_EV_NETWORK_WITNESS_FAILED"]
    pending = not healthy["passed"] or "ZERO_EV_NETWORK_WITNESS_FAILED" in original_errors
    status = ("PHYSICAL_GATE_FAILED" if other_errors else
              "NEED_ORIGINAL_FEASIBILITY_LP" if pending else "PASS")
    audit = deepcopy(case["source_generator_audit"])
    audit.update(passed=status == "PASS", status=status, type_counts=counts,
                 inherited_generator_errors=original_errors, errors=other_errors,
                 pending_checks=["ORIGINAL_ZERO_SHED_FEASIBILITY_LP"] if pending else [],
                 healthy_grid_audit=healthy, model_infeasible=None,
                 scope="Round2 generator checks plus healthy original-matrix witness; no optimization",
                 solver_calls=0)
    case["physical_audit"] = audit
    return case


def make_reference_policies(case):
    """Preserve the canonical N6 vectors, including which type supplies each car.

    Using a virtual one-per-type availability view avoids both automatic count
    scaling and changing z2's marginal vehicle to the newly plentiful core type.
    All resulting policies bind the expanded case's new identity.
    """
    if not case.get("fixed_load_round2") or len(case["type_counts"]) != 6:
        raise ValueError("Expected the Round2 fixed six-type pool")
    view = dict(case, type_counts=[1] * 6)
    policies = structured.make_reference_policies(view)
    for policy in policies:
        policy["reference_rule"] = "exact original N6 vector; no N-dependent scaling"
    return policies


def healthy_zero_ev_system(net, physical):
    """Build the ORIGINAL no-EV, zero-shed healthy system and a candidate witness.

    Returns {matrix, z, witness, physical, fast_witness}. ``matrix`` comes from
    matrix.assemble with an empty EV cohort; only shed bounds are additionally
    fixed to zero. This is the exact no-EV slice of the original equations,
    including P/Q balance, line/voltage/root bounds and station/reference rows.
    The caller may solve its feasibility LP elsewhere; this function never does.
    """
    load = np.asarray(physical["load_p_kw"], float)
    reactive = np.asarray(physical["load_q_kvar"], float)
    if (load.shape != (96, 33) or reactive.shape != load.shape
            or not np.isfinite(load).all() or not np.isfinite(reactive).all()
            or (load < 0).any() or (reactive < 0).any()):
        raise ValueError("Healthy counterfactual requires finite nonnegative original P/Q loads")
    healthy = dict(physical, line_operating_outages=np.zeros((96, len(net["from"])), bool))
    fast = structured._zero_ev(net, healthy)
    mat = matrix.assemble(net, [], healthy, [], 0, global_M=np.empty(0))
    mat.hi[mat.index["shed"]] = 0.
    y = np.zeros(len(mat.cost))
    fr = np.array(net["from"], int) - 1
    to = np.array(net["to"], int) - 1
    downstream = load.copy()
    flow = np.zeros((96, len(fr)))
    for j in reversed(range(len(fr))):
        flow[:, j] = downstream[:, to[j]]
        downstream[:, fr[j]] += flow[:, j]
    voltage = np.full(load.shape, float(net["u_ref"]))
    for j, (u, v) in enumerate(zip(fr, to)):
        voltage[:, v] = voltage[:, u] - flow[:, j] * (net["mp"][j] + .33 * net["mq"][j])
    ix = mat.index
    y[ix["line_p"]], y[ix["line_q"]] = flow, .33 * flow
    y[ix["grid_p"]], y[ix["grid_q"]] = downstream[:, 0], .33 * downstream[:, 0]
    y[ix["u"]] = voltage
    return dict(matrix=mat, z=np.empty(0), witness=y, physical=healthy, fast_witness=fast)


def _column_label(mat, net, column):
    for name, indices in mat.index.items():
        where = np.argwhere(indices == column)
        if where.size:
            position = [int(v) for v in where[0]]
            result = dict(variable=name, index=position)
            if name in {"line_p", "line_q"}:
                result.update(time_h=position[0] / 4, line_id=net["line_ids"][position[1]])
            elif name in {"u", "shed"}:
                result.update(time_h=position[0] / 4, bus=net["buses"][position[1]])
            elif name in {"grid_p", "grid_q"}:
                result["time_h"] = position[0] / 4
            return result
    raise ValueError("Original column has no declared index")


def _check_system(net, system):
    mat, y = system["matrix"], system["witness"]
    ub = certificates.row_audit(mat.A, mat.b, np.full(len(mat.b), "<"), mat.lo, mat.hi, y)
    eq = certificates.row_audit(mat.H, mat.d, np.full(len(mat.d), "="), mat.lo, mat.hi, y)
    residual = max(*ub.values(), *eq.values())
    violations = []
    for block, A, rhs in (("ub", mat.A, mat.b), ("eq", mat.H, mat.d)):
        raw = A.astype(np.longdouble) @ np.asarray(y, np.longdouble) - np.asarray(rhs, np.longdouble)
        violation = abs(raw) if block == "eq" else np.maximum(0., raw)
        bad = np.flatnonzero(violation > TOLERANCE)
        for i in sorted(bad, key=lambda i: -violation[i])[:5]:
            violations.append(dict(kind="original_row", block=block, row=int(i),
                                   name=mat.row_names[block][i], violation=float(violation[i])))
    for label, raw in (("lower", np.asarray(mat.lo, np.longdouble) - y),
                       ("upper", np.asarray(y, np.longdouble) - mat.hi)):
        bad = np.flatnonzero(raw > TOLERANCE)
        for i in sorted(bad, key=lambda i: -raw[i])[:5]:
            violations.append(dict(kind="original_bound", side=label, column=int(i),
                                   **_column_label(mat, net, int(i)), value=float(y[i]),
                                   bound=float(mat.lo[i] if label == "lower" else mat.hi[i]),
                                   violation=float(raw[i])))
    matrix_hash = structured._array_digest(
        mat.A.data, mat.A.indices, mat.A.indptr, mat.b,
        mat.H.data, mat.H.indices, mat.H.indptr, mat.d, mat.lo, mat.hi)
    passed = bool(system["fast_witness"]["passed"] and residual <= TOLERANCE)
    return dict(passed=passed, status="PASS" if passed else "NEED_ORIGINAL_FEASIBILITY_LP",
                fast_witness=system["fast_witness"], original_matrix_checked=True,
                original_rows_checked=len(mat.b) + len(mat.d), original_columns_checked=len(y),
                matrix_nnz=mat.A.nnz + mat.H.nnz, matrix_sha256=matrix_hash,
                witness_sha256=structured._array_digest(y),
                residual=dict(inequality=ub["row"], equality=eq["row"], bound=max(ub["bound"], eq["bound"])),
                max_residual=residual, tolerance=TOLERANCE,
                largest_violations=sorted(violations, key=lambda v: -v["violation"]),
                zero_ev_injection=True, shedding_fixed_zero=True, solver_calls=0,
                model_infeasible=None, optimal_healthy_ens_kwh=None,
                evidence="numerical original-row/bound audit, not exact-rational LP certification")


def audit_healthy_grid_zero_ev(case):
    """Check forecast AND every support demand with every grid line healthy.

    Neither disaster scenes nor demand are overwritten. Failure preserves the
    offending scene and rows/bounds for the charged original feasibility LP.
    CAL features remain unchanged; this audit covers the actual optimization
    support, not unseen samples or a claim about all possible generator draws.
    """
    reports = []
    for scene in [case["forecast"], *case["scenarios"]]:
        system = healthy_zero_ev_system(case["network"], scene["physical"])
        result = _check_system(case["network"], system)
        result.update(scenario_id=scene.get("id", scene["identity"]),
                      source_scenario_identity=scene["identity"],
                      load_sha256=structured._array_digest(scene["physical"]["load_p_kw"],
                                                          scene["physical"]["load_q_kvar"]))
        reports.append(result)
    passed = all(r["passed"] for r in reports)
    return dict(passed=passed, status="PASS" if passed else "NEED_ORIGINAL_FEASIBILITY_LP",
                scenario_count=len(reports), scenarios=reports,
                failed_scenario_ids=[r["scenario_id"] for r in reports if not r["passed"]],
                max_residual=max(r["max_residual"] for r in reports), tolerance=TOLERANCE,
                original_matrix_code_sha256=_source_hash(matrix),
                row_audit_code_sha256=_source_hash(certificates),
                unchanged_demand=True, all_grid_lines_healthy=True,
                zero_ev_injection=True, shedding_fixed_zero=True,
                solver_calls=0, model_infeasible=None,
                next_action=None if passed else "Run the original zero-shed feasibility LP on the named healthy counterfactuals inside the charged wrapper")
