"""Catalogue coefficients and bidirectional ORIGINAL-matrix audits: zero solves."""

from copy import deepcopy
from fractions import Fraction
import json
import runpy

import numpy as np
import pytest
from scipy import sparse as sp

from ev_assignment.l08 import catalogue as cat, matrix, physics, recourse
from ev_assignment.l08.certificates import row_audit


@pytest.fixture(autouse=True)
def prohibit_solves(monkeypatch):
    import scipy.optimize

    def forbidden(*args, **kwargs):
        pytest.fail("Catalogue unit tests must not call a solver")

    monkeypatch.setattr(scipy.optimize, "linprog", forbidden)
    monkeypatch.setattr(scipy.optimize, "milp", forbidden)
    monkeypatch.setattr(matrix, "linprog", forbidden)
    monkeypatch.setattr(matrix.Matrix, "solve", forbidden)


@pytest.fixture
def cases(pytestconfig):
    worker = runpy.run_path(str(pytestconfig.rootpath / "scripts/structured/a1_worker.py"))
    return worker["build_cases"]()


def build(case, sid=0, **kwargs):
    fs = cat.build_first(case["packages"], case["type_counts"],
                         len(case["net"]["station_buses"]), ports=case["ports"], **kwargs)
    rc = cat.build_recourse(case["net"], case["packages"], case["scenes"][sid],
                            case["route_states"][sid], fs)
    return fs, rc


def witness(case, fs, rc, sid=0, n=None):
    """Analytical feasible witnesses, never optimized and never labelled optima."""
    n = (1 if case["name"] == "port_binding" else 2) if n is None else n
    x = np.zeros(len(fs["cost"]))
    D = len(fs["package_map"])
    x[0], x[D] = n, n * case["packages"][0]["Ebar"]
    y = np.zeros(len(rc.names))
    ix = rc.data["index"]
    scene = case["scenes"][sid]
    y[ix["u"]] = 1
    y[ix["shed"]] = scene["load_p_kw"]
    y[ix["shed"][:, 0]] = 0
    y[ix["grid_p"]] = scene["load_p_kw"][:, 0]
    y[ix["grid_q"]] = scene["load_q_kvar"][:, 0]
    for j, p in enumerate(fs["packages"]):
        if x[j] and rc.data["mu"][j].any():
            slots = rc.data["mu"][j]
            delivered = min(x[D + j], .95 * rc.data["Euse"][j] * x[j],
                            rc.data["Eav"][j] * x[j])
            y[ix["p"][j, slots]] = delivered / (.25 * slots.sum())
            bus = case["net"]["buses"].index(case["net"]["station_buses"][p["c"]])
            y[ix["shed"][:, bus]] -= y[ix["p"][j]]
        y[ix["short"][j]] = max(0, rc.data["chi"][j] * x[D + j] - rc.data["Eav"][j] * x[j])
    return x, y


def test_first_cost_maps_capacity_ports_and_contracts(cases):
    case = cases[0]
    fs, _ = build(case, weights=(2, 3, 4, 5))
    inf, _ = build(case, weights=(2, 3, 4, 5), individual=True)
    assert sp.isspmatrix_csr(fs["A"])
    assert fs["parts"].shape == (4, 4)
    np.testing.assert_array_equal(fs["package_map"], [0, 1])
    np.testing.assert_array_equal(inf["package_map"], [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(inf["owner_map"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(fs["vtype"], ["I", "I", "C", "C"])
    assert (inf["vtype"][:6] == "B").all()
    tasks = [dict(p, e=int(e)) for p, e in zip(inf["packages"], inf["owner_map"])]
    old_A, old_rhs, old_parts, _, _ = matrix.first_stage(tasks, 3, 1, [90.] * 3, ports=1)
    # Original per-vehicle contracts are sums of the perspective contracts.
    projection = np.zeros((9, 12))
    projection[:6, :6] = np.eye(6)
    for j, e in enumerate(inf["owner_map"]):
        projection[6 + e, 6 + j] = 1
    np.testing.assert_array_equal(old_parts @ projection, inf["parts"])
    np.testing.assert_array_equal(inf["cost"], np.array([2, 3, 4, 5]) @ inf["parts"])
    x = np.array([1., 0., 90., 0.])
    xi, _ = cat.expand_count(fs, inf, x)
    assert (old_A @ (projection @ xi) <= old_rhs + 1e-9).all()
    for label, expected in (("type_capacity/0", [1, 1, 0, 0]),
                             ("contract_min/0", [15, 0, -1, 0]),
                             ("contract_max/0", [-90, 0, 1, 0]),
                             ("ports/0/4", [1, 1, 0, 0]),
                             ("ports/0/8", [0, 0, 0, 0])):
        np.testing.assert_array_equal(fs["A"].getrow(fs["row_names"].index(label)).toarray()[0], expected)


@pytest.mark.parametrize("which,sid", [(0, 0), (1, 0), (1, 1)])
def test_canonical_roundtrip_original_all_big_m_rows_and_component_costs(cases, which, sid):
    case = cases[which]
    cf, cr = build(case, sid)
    inf, ir = build(case, sid, individual=True)
    x, y = witness(case, cf, cr, sid)
    xi, yi = cat.expand_count(cf, inf, x, y, count_recourse=cr, individual_recourse=ir)
    adapter = cat.build_original_adapter(case["net"], case["packages"], case["scenes"][sid],
                                         case["route_states"][sid], inf)
    z, oy = cat.to_original(adapter, inf, ir, xi, yi)
    old = adapter["matrix"]
    assert old.residual(oy, z) <= 1e-9
    assert len(adapter["tasks"]) == sum(case["type_counts"]) * len(case["packages"])
    assert sum(s.startswith("incident_capability/") for s in old.row_names["ub"]) == len(adapter["tasks"])
    assert (z[:len(adapter["tasks"])] == 0).any()
    # Compare every original coefficient, not only a mapped objective or feasibility.
    independent = matrix.assemble(case["net"], adapter["tasks"], case["scenes"][sid],
                                   adapter["route_states"], sum(case["type_counts"]),
                                   adapter["global_M"], ports=case["ports"])
    for key in ("A", "B", "H", "G"):
        assert (getattr(old, key) != getattr(independent, key)).nnz == 0
    for key in ("b", "d", "cost", "lo", "hi"):
        np.testing.assert_array_equal(getattr(old, key), getattr(independent, key))
    xr, yr = cat.from_original(adapter, inf, ir, z, oy)
    xc, yc = cat.sum_individual(inf, cf, xr, yr, individual_recourse=ir, count_recourse=cr)
    np.testing.assert_allclose(xc, x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(yc, y, rtol=0, atol=1e-9)
    np.testing.assert_allclose(cf["parts"] @ x, adapter["first"]["parts"] @ z, rtol=0, atol=1e-9)
    np.testing.assert_allclose(cr.data["parts"] @ y, old.parts @ oy, rtol=0, atol=1e-9)
    # Original certificate API can consume the template without a solve/dual fit.
    audited = recourse.audit(cr, x, y, np.zeros(cr.W.shape[0]))
    assert max(audited["residual"].values()) <= 1e-9
    assert audited["qL"] <= cr.data["c"] @ y <= audited["qU_audited"]


def test_grid_equations_powerfactor_and_exact_scenario_q_preserved(cases):
    case = deepcopy(cases[0])
    case["scenes"][0]["load_q_kvar"] *= 1.7  # no manual .33 load-Q replacement
    fs, rc = build(case)
    old = matrix.assemble(case["net"], [], case["scenes"][0], [], 0, [])
    base = sp.vstack([old.A, old.H], format="csr")
    assert (rc.W[:base.shape[0], :len(old.cost)] != base).nnz == 0
    np.testing.assert_array_equal(rc.data["h"][:base.shape[0]], np.r_[old.b, old.d])
    assert rc.T[:base.shape[0]].nnz == 0
    for label, coef in (("P_balance/2/4", 1), ("Q_balance/2/4", .33), ("station_power/0/4", 1)):
        row = rc.data["row_names"].index(label)
        assert rc.W[row, rc.data["index"]["p"][0, 4]] == coef
    np.testing.assert_array_equal(rc.data["parts"][2, rc.data["index"]["p"][0]], .25 * physics.tou())


def test_global_power_bound_allows_multiple_vehicles_and_fractional_points(cases):
    case = deepcopy(cases[0])
    case["ports"] = 3
    cf, cr = build(case)
    x, y = witness(case, cf, cr, n=2)
    p = cr.data["index"]["p"]
    assert y[p[0, 4]] == 180 > 90
    assert cr.data["hi"][p[0, 4]] == 270
    # Fractional count is a valid recourse input; integer expansion must reject it.
    fractional_x, fractional_y = x / 4, y.copy()
    fractional_y[p] /= 4
    fractional_y[cr.data["index"]["shed"][4:8, 1]] = 180 - 45
    check = row_audit(cr.W, cr.data["h"] - cr.T @ fractional_x, cr.data["sense"],
                      cr.data["lo"], cr.data["hi"], fractional_y)
    assert max(check.values()) <= 1e-9
    inf, _ = build(case, individual=True)
    with pytest.raises(ValueError, match="fractional"):
        cat.expand_count(cf, inf, fractional_x)


def test_physics_shortage_coefficients_and_finite_bound_proofs(cases):
    case = cases[1]
    fs, rc = build(case, 1)
    D = len(fs["package_map"])
    for j in range(D):
        row = rc.data["row_names"].index(f"incident_capability/{j}")
        assert rc.T[row, D + j] == 1
        assert rc.T[row, j] == 0  # failed route: available energy is zero
        assert rc.W[row, rc.data["index"]["short"][j]] == -1
    assert np.isfinite(rc.data["lo"]).all() and np.isfinite(rc.data["hi"]).all()
    assert np.isfinite(rc.qbar)
    assert len(rc.bound_proof) == 192
    assert {p["variable"].split("/")[0] for p in rc.bound_proof} == {"grid_p", "grid_q"}
    joint = sp.vstack([sp.hstack([fs["A"], sp.csr_matrix((len(fs["rhs"]), len(rc.names)))]),
                       sp.hstack([rc.T, rc.W])], format="csr")
    rhs = np.r_[fs["rhs"], rc.data["h"]]
    lo, hi = np.r_[fs["lo"], rc.data["lo"]], np.r_[fs["hi"], rc.data["hi"]]
    names = fs["names"] + rc.names
    for proof in rc.bound_proof:
        j = names.index(proof["variable"])
        row = joint.getrow(proof["row"]) * proof["direction"]
        terms = sum((Fraction(float(v)) * Fraction(float(lo[k] if v > 0 else hi[k]))
                     for k, v in zip(row.indices, row.data) if k != j), Fraction(0))
        exact = (Fraction(float(proof["direction"] * rhs[proof["row"]])) - terms) / Fraction(float(row[0, j]))
        assert exact == Fraction(*proof["exact"])
        assert Fraction(proof["outward"]) >= exact
    # Redundant Ebar*N box bounds must also be rounded outward for certificates.
    fs5 = cat.build_first(case["packages"], [5], 1)
    assert Fraction(float(fs5["hi"][2])) >= 5 * Fraction(case["packages"][0]["Ebar"])


def test_unequal_contracts_sum_without_claiming_fixed_policy_q_equivalence(cases):
    case = cases[1]
    cf, cr = build(case, 1)
    inf, ir = build(case, 1, individual=True)
    x, y = witness(case, cf, cr, 1)
    xi, yi = cat.expand_count(cf, inf, x, y, count_recourse=cr, individual_recourse=ir)
    # Use a smaller aggregate contract to permit unequal shares under Ebar.
    J = len(inf["package_map"])
    xi[J], xi[J + 2] = 30, 70
    yi[ir.data["index"]["short"][[0, 2]]] = [30, 70]
    xc, yc = cat.sum_individual(inf, cf, xi, yi, individual_recourse=ir, count_recourse=cr)
    xcanonical, ycanonical = cat.expand_count(cf, inf, xc, yc, count_recourse=cr, individual_recourse=ir)
    np.testing.assert_array_equal(xcanonical[[J, J + 2]], [50, 50])
    assert not np.array_equal(xi, xcanonical)
    assert inf["cost"] @ xi == pytest.approx(inf["cost"] @ xcanonical, abs=1e-9)
    assert ir.data["c"] @ yi == pytest.approx(ir.data["c"] @ ycanonical, abs=1e-9)


def test_mapping_identity_rejection_and_departure_choice_energy_allowed(cases):
    case = cases[0]
    assert case["packages"][0]["departure_dc"] != case["packages"][1]["departure_dc"]
    cf, cr = build(case)
    inf, ir = build(case, individual=True)
    x, y = witness(case, cf, cr)
    altered = deepcopy(case)
    altered["scenes"][0]["load_p_kw"][0, 0] += 1
    _, other = build(altered, individual=True)
    with pytest.raises(ValueError, match="identity"):
        cat.expand_count(cf, inf, x, y, count_recourse=cr, individual_recourse=other)
    altered["packages"][0]["Ebar"] -= 1
    with pytest.raises(ValueError, match="Packages differ"):
        cat.build_recourse(case["net"], altered["packages"], case["scenes"][0], case["route_states"][0], cf)
    with pytest.raises(ValueError, match="First-stage"):
        cat.expand_count(cf, inf, [1, 1, 90, 90])  # shared-port violation


def test_empty_types_and_shared_station_ports(cases):
    case = deepcopy(cases[0])
    p = case["packages"][0]
    packages = [p, dict(p, g=1)]
    fs = cat.build_first(packages, [0, 2, 1], 1, ports=1)
    inf = cat.build_first(packages, [0, 2, 1], 1, ports=1, individual=True)
    np.testing.assert_array_equal(inf["owner_map"], [0, 1])  # idle type 2 still has owner 2
    assert "one_task/2" in inf["row_names"]
    assert fs["hi"][0] == 0 and fs["hi"][2] == 0
    row = fs["row_names"].index("ports/0/4")
    np.testing.assert_array_equal(fs["A"].getrow(row).toarray()[0], [1, 1, 0, 0])
    empty = cat.build_first([], [1], 1)
    rc = cat.build_recourse(case["net"], [], case["scenes"][0], [], empty)
    assert rc.T.shape[1] == 0 and rc.data["index"]["p"].shape == (0, 96)


@pytest.mark.parametrize("change", [dict(g=4), dict(c=1), dict(k=1.5), dict(Ebar=-1), dict(r=96)])
def test_invalid_package_rejected(cases, change):
    packages = deepcopy(cases[0]["packages"])
    packages[0].update(change)
    with pytest.raises(ValueError):
        cat.build_first(packages, [3], 1)


@pytest.mark.parametrize("arrival,success", [(1.01, True), (1.75, True), (2., True), (24., False)])
def test_availability_uses_original_slot_boundary_and_failure_rule(cases, arrival, success):
    case = deepcopy(cases[0])
    case["route_states"][0][0] = dict(arrival=arrival, success=success)
    fs, rc = build(case)
    p = case["packages"][0]
    _, mu, use, available = physics.qualify(p["departure_dc"], p["distance"], arrival,
                                           p["k"], p["r"], success)
    np.testing.assert_array_equal(rc.data["mu"][0], mu)
    assert rc.data["Euse"][0] == use and rc.data["Eav"][0] == available
    for t in range(96):
        row = rc.data["row_names"].index(f"p_availability/0/{t}")
        assert rc.T[row, 0] == -90 * mu[t]
    row = rc.data["row_names"].index("incident_capability/0")
    assert rc.T[row, 0] == -available


def test_shared_station_power_cap_and_nonincident_shortage(cases):
    case = deepcopy(cases[0])
    case["type_counts"], case["ports"] = [30], 30
    case["scenes"][0]["line_operating_outages"][:] = False
    fs, rc = build(case)
    p = rc.data["index"]["p"]
    row = rc.data["row_names"].index("station_power/0/4")
    assert rc.data["hi"][p[0, 4]] == 2700
    assert rc.data["h"][row] == 1800
    assert rc.W[row, p[0, 4]] == rc.W[row, p[1, 4]] == 1
    assert (rc.data["chi"] == 0).all()
    row = rc.data["row_names"].index("incident_capability/0")
    assert rc.T[row, len(fs["package_map"]) ] == 0


def test_multiple_stations_and_types_match_original_grid_mapping(cases):
    case = deepcopy(cases[0])
    case["net"]["station_buses"] = [1, 2]
    case["packages"] = [dict(case["packages"][0], c=0),
                         dict(case["packages"][0], c=1, g=1)]
    case["type_counts"] = [1, 1]
    case["route_states"][0] = [dict(success=True, arrival=1.)] * 2
    fs, rc = build(case)
    inf, ir = build(case, individual=True)
    x = np.array([1., 1., 15., 15.])
    _, y = witness(case, fs, rc)  # reset to an all-shedding feasible grid
    ix = rc.data["index"]
    y[ix["p"]] = 0
    y[ix["short"]] = 0
    y[ix["shed"][:, 0]] = 0
    y[ix["shed"][:, 1]] = case["scenes"][0]["load_p_kw"][:, 1]
    y[ix["p"][0, 4:8]] = 5
    y[ix["p"][1, 4:8]] = 10
    y[ix["grid_p"][4:8]] -= 5
    y[ix["grid_q"][4:8]] -= .33 * 5
    y[ix["shed"][4:8, 1]] -= 10
    xi, yi = cat.expand_count(fs, inf, x, y, count_recourse=rc, individual_recourse=ir)
    adapter = cat.build_original_adapter(case["net"], case["packages"], case["scenes"][0],
                                         case["route_states"][0], inf)
    zo, yo = cat.to_original(adapter, inf, ir, xi, yi)
    old = adapter["matrix"]
    assert old.residual(yo, zo) <= 1e-9
    assert (yo[old.index["p"][0, 1]] == 0).all()
    assert (yo[old.index["p"][1, 0]] == 0).all()


@pytest.mark.parametrize("which", [0, 1])
def test_remote_worker_build_and_audit_path_without_native_calls(cases, pytestconfig, tmp_path, which):
    # Exercise the worker's full post-solve path with analytical primals. These
    # stand-ins test plumbing only; their supplied status/bounds prove NO optimum.
    worker = runpy.run_path(str(pytestconfig.rootpath / "scripts/structured/a1_worker.py"))
    case = cases[which]
    cf, _ = build(case)
    inf, _ = build(case, individual=True)
    expected = {label: dict(x=None, ys=[]) for label in ("count", "individual", "original")}
    for sid in range(len(case["scenes"])):
        _, cr = build(case, sid)
        _, ir = build(case, sid, individual=True)
        x, y = witness(case, cf, cr, sid)
        xi, yi = cat.expand_count(cf, inf, x, y, count_recourse=cr, individual_recourse=ir)
        adapter = cat.build_original_adapter(case["net"], case["packages"], case["scenes"][sid],
                                             case["route_states"][sid], inf)
        z, oy = cat.to_original(adapter, inf, ir, xi, yi)
        for label, xx, yy in (("count", x, y), ("individual", xi, yi), ("original", z, oy)):
            expected[label]["x"] = xx
            expected[label]["ys"].append(yy)

    def analytical_standin(data, env, remaining, events, label):
        value = expected[label.rsplit("/", 1)[1]]
        joined = np.concatenate([value["x"], *value["ys"]])
        audit = row_audit(data["A"], data["rhs"], data["sense"], data["lo"], data["hi"], joined)
        assert max(audit.values()) <= 1e-9
        return dict(value, raw=joined, accepted=joined, objective=float(data["cost"] @ joined),
                    native=dict(status="ANALYTICAL_TEST_STANDIN_NO_SOLVE"), row_audit=audit)

    worker["_case"].__globals__["_optimize"] = analytical_standin
    events = []
    result = worker["_case"](case, None, lambda: 100., events, tmp_path)
    assert result["passed"] and events == []
    assert result["mappings"] and set(result["formulations"]) == {"count", "individual", "original"}
    for mapping in result["mappings"]:
        assert mapping["original_residual"] <= 1e-9
        assert mapping["original_tasks"] == mapping["original_big_m_rows"]

    if which == 0:
        # Synthetic metadata exercises the resume parser, not a native-optimality
        # claim: all primals above were hand constructed and no solver was called.
        for entry in result["formulations"].values():
            entry["native"]["native_status"] = 2
            entry["native_objective"] = entry["native_bound"] = entry["objective"]
        worker["_write_new"](tmp_path / "INPUTS.json", cases)
        worker["_write_new"](tmp_path / "port_binding_CHECKPOINT.json", result)
        saved, refs = worker["_previous_port"](tmp_path, cases)
        assert len(refs) == 3 and all(len(ref["sha256"]) == 64 for ref in refs)

        def forbidden(*args, **kwargs):
            pytest.fail("A resumed port case must never call _optimize")

        worker["_case"].__globals__["_optimize"] = forbidden
        resumed = tmp_path / "resumed"
        resumed.mkdir()
        reread = worker["_case"](case, None, lambda: 100., events, resumed, replay=saved)
        assert reread["passed"] and events == []
        assert all(v["reused_without_new_solve"] for v in reread["formulations"].values())
        # Even a changed ENERGY case invalidates port reuse: compare ALL inputs.
        changed = deepcopy(cases)
        changed[1]["packages"][0]["departure_dc"] += 1
        with pytest.raises(ValueError, match="all-case"):
            worker["_previous_port"](tmp_path, changed)
        # Keep input/checkpoint fixed but corrupt the primal: the original rows
        # must detect it before the checkpoint is accepted in the new attempt.
        saved[1]["count/accepted"][0] = 30
        with pytest.raises(ValueError, match="witness failed"):
            worker["_case"](case, None, lambda: 100., events, resumed, replay=saved)
        parsed = json.loads((tmp_path / "port_binding_CHECKPOINT.json").read_text())
        assert parsed["passed"]  # historical evidence was read only
