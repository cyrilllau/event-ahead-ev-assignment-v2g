"""Preregistered G stress at fixed C11 revision1 N6 demand and N30 availability.

Only an explicit call generates data; real-source calls belong in the charged
remote wrapper. No solver is imported or called. G changes physical business
types, SOC offsets and the catalogue, not just representation or vehicle labels.
Consequently neither equal total SOC nor nested feasible domains is asserted.
"""

from copy import deepcopy
import hashlib
import math
from pathlib import Path

from ev_assignment_experiments.datasets import structured as source
from ev_assignment_experiments.datasets import structured_round2 as fixed


FAMILY = "structured_fixed_load_group_stress_v1"
GROUPS = (3, 15, 30)


def preregistered_protocol():
    """Return the rule frozen before validation, without reading any inputs."""
    return dict(
        family=FAMILY, groups=list(GROUPS), fleet_size=30, seed=2101,
        scenario_count=400, split="stress", config=dict(fixed.BASE_CONFIG),
        physical_load_fleet_size=6, fleet_scale=1., base_groups=6,
        type_rule="First G source EV IDs in the original sorted business roster",
        initial_offset_dc_rule="60 + 50*g/(G-1) kWh DC; original continuous business trajectory",
        counts_rule="30/G vehicles of each physical type; G is 3, 15 or 30",
        catalogue_rule="Original _packages(source, config, G, 30); twelve packages per type",
        replay_rule="Original actual-entry fixed-route replay against unchanged G6 physical arrays",
        pairing="Identical seed/split/S implies identical G6 weather, demand, weights and CAL across G",
        rng_draws_after_base=0, calibration_rule="Unchanged independent CAL2000 pointwise Wilson95%",
        audits="Rerun all original generator checks and healthy zero-EV zero-shed original-matrix rows",
        strict_margin_h=.25, strict_margin_operator=">", healthy_original_matrix_tolerance=1e-9,
        failure_rule="Keep every audit error; construction failure returns an explicitly incomplete failed case",
        r3_physical_revision=False, equal_total_soc_claim=False,
        nested_domain_claim=False, prior_bound_or_cut_reuse=False,
        references="G6 reference vectors are not inherited into the changed catalogue",
        overrides="Explicit seed/S overrides have new identities and are not the preregistered G stress",
    )


def _soc_summary(packages, types, counts):
    """Count each vehicle once at each departure, not once per offered package."""
    n = sum(counts)
    initial = math.fsum(c * t["initial_dc"] for t, c in zip(types, counts))
    by_departure = []
    for d in sorted({p["d"] for p in packages}):
        values = []
        for g in range(len(counts)):
            energies = {p["departure_dc"] for p in packages if p["g"] == g and p["d"] == d}
            if len(energies) != 1:
                raise ValueError("Each type must have one actual business SOC at each departure")
            values.append(energies.pop())
        total = math.fsum(c * e for c, e in zip(counts, values))
        by_departure.append(dict(d=d, time_h=d / 4, per_type_dc_kwh=values,
                                 weighted_total_dc_kwh=total, weighted_mean_dc_kwh=total / n))
    return dict(fleet_size=n, weights=list(counts),
                per_type_initial_dc_kwh=[t["initial_dc"] for t in types],
                weighted_initial_dc_kwh=initial, weighted_mean_initial_dc_kwh=initial / n,
                departures=by_departure,
                interpretation="Available fleet energy, not energy of a selected deployment; departures are alternatives")


def _replay(scene, packages, arcs, groups):
    physical = scene["physical"]
    states, cache = [], {}
    for p in packages:
        key = (p["g"], p["c"], p["d"])
        if key not in cache:
            cache[key] = source.replay_route(
                p["arc_ids"], p["d"] / 4, arcs,
                physical["road_travel_hours"], physical["road_blocked"],
                fractions=p["arc_fractions"], origin=p["route"]["origin"],
                destination=p["route"]["destination"])
        states.append(deepcopy(cache[key]))
    scene["parent_scene_identity"] = scene["identity"]
    scene["parent_scene_id"] = scene["id"]
    scene["id"] = f"G{groups}/{scene['id']}"
    scene["route_states"] = states
    scene["identity"] = source._digest(dict(
        family=FAMILY, groups=groups, parent=scene["parent_scene_identity"],
        primitive=scene["primitive_identity"]["sha256"],
        physical=scene["physical_sha256"], packages=source._digest(packages),
        route_states=states))


def make_stress_case(inputs, groups, seed=2101, scenario_count=400):
    """Build a new G3/G15/G30 N30 case from the explicit source-bundle root.

    The G6 base is generated once by make_fixed_load_case(inputs, "stress", ...).
    No RNG is called after that base; all scenario and forecast physical arrays,
    CAL, feature values and sampling weights are retained exactly. Support IDs
    change because routes and package coefficients change. A failed case must
    not be optimized. Original feasibility-LP fallback is the remote caller's
    responsibility; a failed sufficient witness is never labelled infeasibility.
    """
    groups = source._integer(groups, "groups")
    if groups not in GROUPS:
        raise ValueError("Preregistered stress groups are 3, 15 and 30")
    seed = source._integer(seed, "seed", 0)
    count = source._integer(scenario_count, "scenario_count", 4)
    protocol = preregistered_protocol()
    base = fixed.make_fixed_load_case(inputs, "stress", seed, count, fleet_size=30, groups=6)
    raw = source.read_source(inputs)
    if (raw["source_refs"] != base["source_refs"]
            or source._digest(raw["network"]) != source._digest(base["network"])
            or raw["arcs"] != base["roads"]):
        raise ValueError("Source bundle changed after the fixed-load base was built")
    case = deepcopy(base)
    case["base_case_identity"] = deepcopy(base["identity"])
    case["base_physical_audit"] = deepcopy(base["physical_audit"])
    case["base_pool_expansion"] = case.pop("pool_expansion")
    case["base_feature_scope"] = deepcopy(case["feature_scope"])
    case.pop("reference_policies", None)
    construction_error = None
    try:
        packages, types, counts = source._packages(raw, dict(fixed.BASE_CONFIG), groups, 30)
    except ValueError as exc:
        # No clipped SOC, replacement source type, dropped package or route repair.
        packages, types, counts = [], [], [30 // groups] * groups
        construction_error = f"{type(exc).__name__}: {exc}"
    case.update(packages=packages, types=types, type_counts=counts,
                structured_stress=True, catalogue_complete=construction_error is None)
    for scene in [case["forecast"], *case["scenarios"]]:
        _replay(scene, packages, case["roads"], groups)
    base_soc = _soc_summary(base["packages"], base["types"], base["type_counts"])
    soc = _soc_summary(packages, types, counts) if construction_error is None else None
    case["soc_report"] = dict(base_g6=base_soc, stress=soc, equal_total_soc_claim=False)
    if soc is not None:
        case["soc_report"]["initial_dc_delta_kwh"] = soc["weighted_initial_dc_kwh"] - base_soc["weighted_initial_dc_kwh"]
        case["soc_report"]["departure_dc_deltas_kwh"] = [dict(
            d=a["d"], delta_kwh=a["weighted_total_dc_kwh"] - b["weighted_total_dc_kwh"])
            for a, b in zip(soc["departures"], base_soc["departures"])]
    stress = dict(protocol=protocol, protocol_sha256=source._digest(protocol),
                  preregistered_configuration=seed == 2101 and count == 400,
                  groups=groups, fleet_size=30, seed=seed, scenario_count=count,
                  physical_groups_changed=True, catalogue_changed=True,
                  source_type_ids=[t["source_ev"] for t in types],
                  type_counts=counts, source_type_count=len(types),
                  physical_arrays_unchanged=True, network_unchanged=True,
                  calibration_unchanged=True, features_unchanged=True, weights_unchanged=True,
                  route_states_replayed=True, rng_draws_after_base=0,
                  package_count=len(packages), expected_package_count=12 * groups,
                  first_stage_variable_count=2 * len(packages),
                  individual_first_stage_variable_count=2 * 30 * 12 if packages else None,
                  reference_vectors_inherited=False, construction_error=construction_error,
                  r3_physical_revision=False, nested_domain_claim=False,
                  prior_bound_or_cut_reuse=False)
    case["stress_protocol"] = stress
    scope = case["feature_scope"]
    scope.update(group_stress=FAMILY, groups=groups, available_fleet_size=30,
                 reference_units_per_target=None)
    scope["differences"] = [v for v in scope["differences"] if not v.startswith((
        "Round2 fixes C11", "Reference deployments remain"))] + [
        "Protocol G stress retains C11 revision1 N6 demand and all G6 physical trajectories at N30; this is not R3.",
        "First G original source types with 60+50*g/(G-1) initial DC offsets replace the G6 catalogue; routes are replayed without weather redraw.",
        "Physical type/SOC coefficients and package domain change; neither total SOC equality, domain nesting nor pure representation-only comparison is claimed.",
        "CAL2000, Wilson bounds, feature values and stratified weights stay unchanged; support identities change with the new routes. G6 reference vectors are not inherited.",
    ]
    identity = dict(
        family=FAMILY, config=dict(fixed.BASE_CONFIG), split="stress", seed=seed,
        groups=groups, fleet_size=30, type_counts=counts, scenario_count=count,
        calibration_count=base["identity"]["calibration_count"], revision=1,
        candidate_id="C11", physical_load_fleet_size=6, fleet_scale=1.,
        parent_case_identity=base["identity"]["sha256"],
        source=case["source_refs"]["snapshot_sha256"],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_generator_code_sha256=base["identity"]["code_sha256"],
        kernel_sha256=deepcopy(base["identity"].get("kernel_sha256", {})),
        original_matrix_code_sha256=base["identity"]["original_matrix_code_sha256"],
        row_audit_code_sha256=base["identity"]["row_audit_code_sha256"],
        catalogue=source._digest(packages), types=source._digest(types),
        calibration=case["calibration"]["identity"],
        scenarios=[s["identity"] for s in case["scenarios"]],
        forecast=case["forecast"]["identity"], weights=case["weights"],
        physical_scenarios=[s["physical_sha256"] for s in case["scenarios"]],
        features=deepcopy(scope), stress_protocol=stress, soc_report=case["soc_report"],
        catalogue_complete=case["catalogue_complete"], prior_bound_reuse_allowed=False,
        old_frozen_case=False, semantics=base["identity"]["semantics"])
    case["identity"] = dict(identity, sha256=source._digest(identity))
    healthy = fixed.audit_healthy_grid_zero_ev(case)
    case["healthy_grid_audit"] = healthy
    generated = (source.audit_physical_case(case) if construction_error is None else dict(
        passed=False, errors=["PACKAGE_CONSTRUCTION_FAILED"], reason=construction_error,
        solver_calls=0, scope="Catalogue construction failed; route/energy audits could not run"))
    case["source_generator_audit"] = deepcopy(generated)
    errors = list(generated["errors"])
    pending = not healthy["passed"] or "ZERO_EV_NETWORK_WITNESS_FAILED" in errors
    hard_fail = any(e != "ZERO_EV_NETWORK_WITNESS_FAILED" for e in errors)
    status = "PHYSICAL_GATE_FAILED" if hard_fail else "NEED_ORIGINAL_FEASIBILITY_LP" if pending else "PASS"
    case["physical_audit"] = dict(
        generated, passed=status == "PASS", status=status, errors=errors,
        healthy_grid_audit=healthy, model_infeasible=None, solver_calls=0,
        strict_audits_completed=construction_error is None,
        pending_checks=["ORIGINAL_FEASIBILITY_LP"] if pending else [],
        scope="Fresh G-stress original generator checks plus healthy original-matrix audit; no optimization")
    return case
