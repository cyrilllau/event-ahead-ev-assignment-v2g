"""Round2 sidecar regression on synthetic portable fixtures only; zero solves."""

from copy import deepcopy
import runpy

import numpy as np
import pytest

from ev_assignment.l08 import catalogue, matrix
from ev_assignment_experiments.datasets import structured, structured_round2 as r2


@pytest.fixture(scope="module", autouse=True)
def prohibit_solves():
    import scipy.optimize

    def forbidden(*args, **kwargs):
        pytest.fail("Round2 sidecar unit tests must not call a solver")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(matrix.Matrix, "solve", forbidden)
        patch.setattr(matrix, "linprog", forbidden)
        patch.setattr(scipy.optimize, "linprog", forbidden)
        patch.setattr(scipy.optimize, "milp", forbidden)
        yield


@pytest.fixture(scope="module")
def source_root(pytestconfig, tmp_path_factory):
    # Reuse the existing source-SCHEMA fixture, not any external research input.
    definitions = runpy.run_path(str(pytestconfig.rootpath / "tests/unit/test_structured_generator.py"))
    return definitions["source_root"].__wrapped__(tmp_path_factory)


@pytest.fixture(scope="module")
def base_case(source_root, prohibit_solves):
    return structured.make_case(dict(r2.BASE_CONFIG), source_root, "development", 101,
                                scenario_count=4, fleet_size=6, groups=6)


@pytest.fixture(scope="module")
def expanded_case(base_case):
    return r2.expand_available_fleet(base_case, fleet_size=30)


def test_make_case_always_calls_frozen_n6_physics(source_root, monkeypatch):
    original = structured.make_case
    invocations = []

    def observe(*args, **kwargs):
        invocations.append((args, kwargs.copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(structured, "make_case", observe)
    case = r2.make_fixed_load_case(source_root, "development", 101, 8, fleet_size=30)
    assert len(invocations) == 1
    args, kw = invocations[0]
    assert args == (r2.BASE_CONFIG, source_root, "development", 101)
    assert kw == dict(scenario_count=8, fleet_size=6, groups=6)
    assert case["type_counts"] == [5] * 6 and case["fleet_scale"] == 1.
    assert case["identity"]["fleet_size"] == 30
    assert case["identity"]["physical_load_fleet_size"] == 6
    assert case["healthy_grid_audit"]["scenario_count"] == 9
    assert case["physical_audit"]["status"] == "PASS"


def test_expansion_preserves_all_frozen_physics_and_does_not_draw(base_case, monkeypatch):
    before = structured._digest(base_case)

    def forbidden(*args, **kwargs):
        pytest.fail("Expanding a loaded R0 artifact must not regenerate or draw")

    monkeypatch.setattr(structured, "make_case", forbidden)
    monkeypatch.setattr(structured, "_rng", forbidden)
    case = r2.expand_available_fleet(base_case, fleet_size=30)
    assert structured._digest(base_case) == before
    assert case["network"] == base_case["network"]
    assert case["packages"] == base_case["packages"]
    assert case["roads"] == base_case["roads"]
    assert case["config"] == base_case["config"] == r2.BASE_CONFIG
    assert case["source_refs"] == base_case["source_refs"]
    for a, b in zip(base_case["types"], case["types"]):
        assert a == dict(b, count=1) and b["count"] == 5
    for a, b in zip([base_case["forecast"], *base_case["scenarios"]],
                    [case["forecast"], *case["scenarios"]]):
        assert a["identity"] == b["identity"]
        assert a["route_states"] == b["route_states"]
        assert a["primitive_identity"] == b["primitive_identity"]
        assert a["physical_sha256"] == b["physical_sha256"]
        for key in a["physical"]:
            np.testing.assert_array_equal(a["physical"][key], b["physical"][key])
    assert structured._digest(case["calibration"]) == structured._digest(base_case["calibration"])
    for key in ("features", "weights", "calibration_features", "calibration_mean",
                "calibration_lower", "calibration_upper"):
        np.testing.assert_array_equal(case[key], base_case[key])
    assert case["identity"]["sha256"] != base_case["identity"]["sha256"]
    assert case["identity"]["parent_case_identity"] == base_case["identity"]["sha256"]
    assert not case["identity"]["prior_bound_reuse_allowed"]
    assert case["feature_scope"]["model_name"] == "P3-DRO"
    assert case["feature_scope"]["marginal_model_name"] == "P2-DRO"


def test_reference_vectors_are_exact_n6_vectors_and_old_domain_embeds(base_case, expanded_case):
    old_policies = structured.make_reference_policies(base_case)
    new_policies = r2.make_reference_policies(expanded_case)
    old = catalogue.build_first(base_case["packages"], base_case["type_counts"], 8)
    new = catalogue.build_first(expanded_case["packages"], expanded_case["type_counts"], 8)
    assert (old["A"] != new["A"]).nnz == 0
    np.testing.assert_array_equal(old["cost"], new["cost"])
    assert np.all(new["rhs"] >= old["rhs"])
    assert np.all(new["hi"] >= old["hi"])
    np.testing.assert_array_equal(new["lo"], old["lo"])
    for a, b, selected in zip(old_policies, new_policies, (1, 1, 2)):
        np.testing.assert_array_equal(a["x"], b["x"])
        np.testing.assert_array_equal(a["cost_parts"], b["cost_parts"])
        assert b["n"].sum() == selected and b["units_per_protection_target"] == 1
        assert b["case_identity"] == expanded_case["identity"]["sha256"]
        assert np.max(new["A"] @ b["x"] - new["rhs"]) <= 1e-9
    # z2 must retain the original marginal vehicle type, not use a second g0.
    broad = new_policies[2]
    assert {expanded_case["packages"][h]["g"] for h in broad["package_ids"]} == {0, 1}


@pytest.mark.parametrize("groups", [3, 15, 30])
def test_no_accidental_group_or_soc_change(groups, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported G must fail before source IO/generation")

    monkeypatch.setattr(structured, "make_case", forbidden)
    with pytest.raises(ValueError, match="six physical types"):
        r2.make_fixed_load_case("not-an-input-root", "development", 101, 64, groups=groups)


def test_scaled_old_cases_rejected_before_auditing(base_case, monkeypatch):
    old_scaled = deepcopy(base_case)
    old_scaled["type_counts"] = [2] * 6
    old_scaled["fleet_scale"] = 2.

    def forbidden(*args, **kwargs):
        pytest.fail("A scaled base must not reach matrix construction")

    monkeypatch.setattr(matrix, "assemble", forbidden)
    with pytest.raises(ValueError, match="exact C11"):
        r2.expand_available_fleet(old_scaled)


def test_healthy_rows_are_original_and_zero_shed_is_fixed(base_case):
    net, scene = base_case["network"], base_case["forecast"]["physical"]
    before = structured._digest(scene)
    system = r2.healthy_zero_ev_system(net, scene)
    mat, y = system["matrix"], system["witness"]
    expected = matrix.assemble(net, [], system["physical"], [], 0, global_M=np.empty(0))
    assert not system["physical"]["line_operating_outages"].any()
    assert scene["line_operating_outages"].any()
    assert structured._digest(scene) == before
    for key in ("load_p_kw", "load_q_kvar"):
        np.testing.assert_array_equal(system["physical"][key], scene[key])
    for key in ("A", "B", "H", "G"):
        assert (getattr(mat, key) != getattr(expected, key)).nnz == 0
    for key in ("cost", "parts", "b", "d", "lo"):
        np.testing.assert_array_equal(getattr(mat, key), getattr(expected, key))
    assert mat.row_names == expected.row_names
    assert mat.index["p"].size == 0
    assert not y[mat.index["shed"]].any()
    assert not mat.hi[mat.index["shed"]].any()
    assert mat.residual(y, system["z"]) <= 1e-9
    result = r2._check_system(net, system)
    assert result["status"] == "PASS" and result["original_matrix_checked"]
    assert result["original_rows_checked"] == mat.A.shape[0] + mat.H.shape[0]
    assert result["original_columns_checked"] == len(y)
    assert result["solver_calls"] == 0


def test_capacity_failure_requests_lp_not_infeasible(base_case):
    case = deepcopy(base_case)
    case["network"]["p_limit"][19] = 20.
    before = structured._digest(case)
    result = r2.audit_healthy_grid_zero_ev(case)
    assert structured._digest(case) == before
    assert result["status"] == "NEED_ORIGINAL_FEASIBILITY_LP"
    assert result["model_infeasible"] is None and result["solver_calls"] == 0
    assert result["failed_scenario_ids"]
    violations = [v for scene in result["scenarios"] for v in scene["largest_violations"]]
    assert any(v.get("line_id") == 20 and v.get("variable") == "line_p" for v in violations)
    expanded = r2.expand_available_fleet(case)
    assert expanded["physical_audit"]["status"] == "NEED_ORIGINAL_FEASIBILITY_LP"
    assert expanded["physical_audit"]["model_infeasible"] is None


def test_fast_witness_cannot_replace_original_q_balance_rows(base_case):
    case = deepcopy(base_case)
    case["forecast"]["physical"]["load_q_kvar"][0, 20] += 1.
    # _zero_ev does not itself consume the supplied Q array. The actual matrix
    # must catch this discrepancy even though the hand witness reports success.
    result = r2.audit_healthy_grid_zero_ev(case)
    forecast = result["scenarios"][0]
    assert forecast["fast_witness"]["passed"]
    assert forecast["status"] == "NEED_ORIGINAL_FEASIBILITY_LP"
    assert any(v.get("name") == "Q_balance/21/0" for v in forecast["largest_violations"])


def test_every_support_demand_is_checked_not_just_forecast(base_case):
    case = deepcopy(base_case)
    affected = case["scenarios"][-1]
    affected["physical"]["load_p_kw"][7, 20] = 20000.
    affected["physical"]["load_q_kvar"][7, 20] = .33 * 20000.
    result = r2.audit_healthy_grid_zero_ev(case)
    assert result["scenarios"][0]["passed"]
    assert result["failed_scenario_ids"] == [affected["id"]]
    assert result["scenario_count"] == len(case["scenarios"]) + 1


def test_larger_support_keeps_new_identity_and_independent_calibration_rule(source_root, base_case):
    more = r2.make_fixed_load_case(source_root, "development", 101, 8, fleet_size=6)
    assert more["identity"]["scenario_count"] == 8
    assert more["identity"]["sha256"] != base_case["identity"]["sha256"]
    assert more["calibration"]["identity"] == base_case["calibration"]["identity"]
    assert not {s["identity"] for s in more["scenarios"]} & {s["identity"] for s in base_case["scenarios"]}
