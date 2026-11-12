"""Pure synthetic coefficients and mocked execution; NO native/science solves."""

from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sp

from ev_assignment.domain.planning import AmbiguitySet, Control
from ev_assignment.l08 import physics, recourse
from ev_assignment.methods import catalogue_direct as direct


@pytest.fixture(autouse=True)
def no_real_optimizers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Direct unit tests must not invoke a real optimizer")

    monkeypatch.setattr("scipy.optimize.linprog", forbidden)
    monkeypatch.setattr("scipy.optimize.milp", forbidden)
    monkeypatch.setattr("ev_assignment.l08.matrix.linprog", forbidden)
    monkeypatch.setattr("ev_assignment.l08.risk.linprog", forbidden)
    monkeypatch.setitem(sys.modules, "gurobipy", None)


def domain(ids=("healthy_route", "failed_route")):
    phi = sp.csr_matrix([[0., 1.], [0., 1.], [0., 1.]])
    lower, upper = np.full(3, .1), np.full(3, .9)

    def check(p):
        p = np.asarray(p)
        passed = (p.shape == (2,) and np.isfinite(p).all() and (p >= 0).all() and
                  p.sum() == 1 and (phi @ p >= lower).all() and (phi @ p <= upper).all())
        return dict(passed=bool(passed))

    return AmbiguitySet(tuple(ids), phi, lower, upper, "synthetic-P3", check, np.array([.5, .5]))


def case_fixture():
    net = dict(buses=[1, 2], station_buses=[2], **{"from": [1], "to": [2]},
               p_limit=[1000.], q_limit=[1000.], u_min=.9, u_max=1.1, u_ref=1.,
               mp=[.00001], mq=[.00001], big_m=1., shed_cost=[10., 50.])
    p = dict(g=0, c=0, d=0, k=4, r=11, departure_dc=200., distance=10., arrival=1.,
             Ebar=physics.ebar(200., 10., 1., 4, 11))
    # Keep an unchosen alternative, including original Big-M rows.
    alternative = dict(p, d=1, departure_dc=220., Ebar=physics.ebar(220., 10., 1., 4, 11))
    load = np.zeros((96, 2))
    load[:, 0] = 10
    load[4:12, 1] = 250
    off = np.zeros((96, 1), bool)
    off[4:12] = True
    physical = dict(load_p_kw=load, load_q_kvar=.33 * load, line_operating_outages=off)
    scenes = [dict(identity=label, physical=deepcopy(physical),
                   route_states=[dict(success=success, arrival=1.), dict(success=False, arrival=24.)])
              for label, success in (("healthy_route", True), ("failed_route", False))]
    return dict(identity="pure-two-bus-fixture", network=net, type_counts=[2],
                packages=[p, alternative], ports=20, scenarios=scenes)


def analytical_y(case, template, x, sid):
    ix = template.data["index"]
    y = np.zeros(len(template.names))
    scene = case["scenarios"][sid]["physical"]
    y[ix["u"]] = 1
    y[ix["grid_p"]] = scene["load_p_kw"][:, 0]
    y[ix["grid_q"]] = scene["load_q_kvar"][:, 0]
    y[ix["shed"][:, 1]] = scene["load_p_kw"][:, 1]
    if sid == 0:
        y[ix["p"][0, 4:12]] = x[2] / 2
        y[ix["shed"][4:12, 1]] -= x[2] / 2
    else:
        y[ix["short"][0]] = x[2]
    return y


def mock_audit_oracles(monkeypatch, case, fail_sid=None):
    calls = []

    def oracle(template, x, options, control, *, env):
        sid = 0 if template.data["mu"][0].any() else 1
        calls.append(sid)
        assert options.seconds <= 60 and options.threads == 1 and control.remaining <= 60
        if sid == fail_sid:
            raise TimeoutError("synthetic interruption")
        y = analytical_y(case, template, x, sid)
        result = recourse.audit(template, x, y, np.zeros(template.W.shape[0]))
        return dict(result, y=y, raw_pi=np.zeros(template.W.shape[0]), native_status=2,
                    runtime=0., test_only_no_native_solve=True)

    def probability(ambiguity, ql, qu, qbars, control):
        # beta0=max(Q_U)+guard, plus=minus=0 is a valid conservative dual.
        upper = float(max(qu) + 1e-6)
        return dict(pstar=ambiguity.feasible_anchor, risk_L=float(np.mean(ql)), risk_U=upper,
                    beta0=upper, plus=np.zeros(3), minus=np.zeros(3), plane=np.full(2, upper),
                    all_support_ids=list(ambiguity.support_ids), test_only_no_native_solve=True)

    monkeypatch.setattr(direct.recourse, "solve", oracle)
    monkeypatch.setattr(direct.risk, "solve", probability)
    return calls


def test_direct_matrix_retains_all_first_and_physical_blocks():
    case, ambiguity = case_fixture(), domain()
    built = direct.build_direct(case, ambiguity)
    a, f = built["arrays"], built["first"]
    nx = len(f["cost"])
    assert sp.isspmatrix_csr(a["A"])
    assert (a["A"][:len(f["rhs"]), :nx] != f["A"]).nnz == 0
    assert np.isneginf(a["lo"][a["index"]["beta0"]])
    assert np.isposinf(a["hi"][a["index"]["beta0"]])
    np.testing.assert_array_equal(a["cost"][slice(*a["index"]["plus"])], ambiguity.upper)
    np.testing.assert_array_equal(a["cost"][slice(*a["index"]["minus"])], -ambiguity.lower)
    begin = len(f["rhs"])
    for sid, template in enumerate(built["templates"]):
        end = begin + template.W.shape[0]
        cols = slice(*a["index"]["y"][sid])
        assert (a["A"][begin:end, :nx] != template.T).nnz == 0
        assert (a["A"][begin:end, cols] != template.W).nnz == 0
        np.testing.assert_array_equal(a["rhs"][begin:end], template.data["h"])
        np.testing.assert_array_equal(a["sense"][begin:end], template.data["sense"])
        np.testing.assert_array_equal(a["lo"][cols], template.data["lo"])
        np.testing.assert_array_equal(a["hi"][cols], template.data["hi"])
        assert not a["cost"][cols].any()  # recourse costs enter epigraph, not objective twice
        np.testing.assert_array_equal(a["A"][-2 + sid, cols].toarray()[0], template.data["c"])
        begin = end
    np.testing.assert_array_equal(a["A"][-2:, slice(*a["index"]["plus"])].toarray(), -ambiguity.phi.T.toarray())
    np.testing.assert_array_equal(a["A"][-2:, slice(*a["index"]["minus"])].toarray(), ambiguity.phi.T.toarray())
    assert len(a["names"]) == len(a["cost"]) and len(a["row_names"]) == a["A"].shape[0]


@pytest.mark.parametrize("q,beta,plus,minus", [([100., 300.], 100., 200., 0.),
                                              ([300., 100.], 300., 0., 200.)])
def test_moment_dual_signs_give_analytic_risk_280(q, beta, plus, minus):
    first = dict(A=sp.csr_matrix((0, 0)), rhs=np.zeros(0), sense=np.array([], dtype=str),
                 cost=np.zeros(0), lo=np.zeros(0), hi=np.zeros(0), vtype=np.array([], dtype=str),
                 names=(), row_names=())
    template = recourse.RecourseTemplate(sp.csr_matrix((0, 1)), sp.csr_matrix((0, 0)),
        dict(c=np.ones(1), h=np.zeros(0), sense=np.array([], dtype=str), lo=np.zeros(1),
             hi=np.full(1, 1000.), xlo=np.zeros(0), xhi=np.zeros(0), row_names=()), ("q",), (), 1000.)
    a = direct.assemble_direct(first, [template, template], domain())
    z = np.zeros(len(a["cost"]))
    z[a["index"]["beta0"]] = beta
    z[a["index"]["plus"][0]] = plus
    z[a["index"]["minus"][0]] = minus
    for (start, stop), value in zip(a["index"]["y"], q):
        z[start:stop] = value
    np.testing.assert_array_equal(a["A"] @ z, [0., 0.])
    assert a["cost"] @ z == 280.


def test_audit_has_fixed_contracts_original_big_m_rows_and_certificates(monkeypatch):
    case, ambiguity = case_fixture(), domain()
    built = direct.build_direct(case, ambiguity)
    calls = mock_audit_oracles(monkeypatch, case)
    x = np.array([2., 0., 160., 0.])
    evidence = []
    result = direct.certify_incumbent(case, ambiguity, built, x, Control(10), env=None,
                                      witness=lambda *args: evidence.append(args))
    assert result["UB"] is not None and result["original_expansion_audited"]
    assert calls == [0, 1] and result["coverage"] == dict(expected=2, recourse=2, original=2, risk=True)
    originals = [e for e in evidence if e[0] == "original_audit"]
    np.testing.assert_array_equal(originals[0][4], originals[1][4])
    for entry in result["original_audit"]:
        assert entry["residual"] <= 1e-9
        assert entry["original_tasks"] == entry["original_shortfall_big_m_rows"] == 4
        assert max(map(abs, entry["operating_component_delta"])) <= 1e-5
    assert evidence[-1][0] == "risk" and evidence[-1][5]["certificate"] is result["best_risk_certificate"]


def test_fractional_or_incomplete_audit_never_produces_ub(monkeypatch):
    case, ambiguity = case_fixture(), domain()
    built = direct.build_direct(case, ambiguity)
    calls = mock_audit_oracles(monkeypatch, case, fail_sid=1)
    incomplete = direct.certify_incumbent(case, ambiguity, built, [2, 0, 160, 0], Control(10), env=None)
    assert incomplete["UB"] is None and incomplete["best"] is None
    assert incomplete["coverage"]["original"] == 1 and not incomplete["original_expansion_audited"]
    calls.clear()
    fractional = direct.certify_incumbent(case, ambiguity, built, [.5, 0, 20, 0], Control(10), env=None)
    assert fractional["status"] == "AUDIT_FAILED" and fractional["UB"] is None and calls == []
    with pytest.raises(ValueError, match="Invalid"):
        direct.canonical_integer(built["first"], [np.nan, 0, 20, 0])


def test_support_mismatch_and_non_p3_rejected():
    case = case_fixture()
    with pytest.raises(ValueError, match="ordered support"):
        direct.build_direct(case, domain(("failed_route", "healthy_route")))
    a = domain()
    p2 = AmbiguitySet(a.support_ids, a.phi[:2], a.lower[:2], a.upper[:2], "P2", a.check_original, a.feasible_anchor)
    with pytest.raises(ValueError, match="three moments"):
        direct.build_direct(case, p2)


@pytest.mark.parametrize("fail_audit", [False, True])
def test_native_pipeline_preserves_lb_and_never_promotes_unaudited_incumbent(monkeypatch, fail_audit):
    case, ambiguity = case_fixture(), domain()
    built = direct.build_direct(case, ambiguity)
    a = built["arrays"]
    x = np.array([2., 0., 160., 0.])
    native = np.zeros(len(a["cost"]))
    native[:4] = x
    costs = []
    for sid, (start, stop) in enumerate(a["index"]["y"]):
        y = analytical_y(case, built["templates"][sid], x, sid)
        native[start:stop] = y
        costs.append(built["templates"][sid].data["c"] @ y)
    native[a["index"]["beta0"]] = max(costs)
    times = []

    class Variables(np.ndarray):
        @property
        def X(self):
            return native.copy()

    class Context:
        def __init__(self, *args, **kwargs):
            self.Params = SimpleNamespace()
            self.Status, self.Runtime, self.NodeCount, self.SolCount = 2, 0., 1., 1
            self.ObjVal, self.ObjBoundC = float(a["cost"] @ native), 1.

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def addMVar(self, size, **kwargs):
            return np.zeros(size).view(Variables)

        def addMConstr(self, *args):
            pass

        def setObjective(self, *args):
            pass

        def update(self):
            pass

        def optimize(self, callback):
            times.append(self.Params.TimeLimit)  # STUB: no optimization occurs

    fake_gp = SimpleNamespace(Env=Context, Model=Context, GRB=SimpleNamespace(TIME_LIMIT=9))
    monkeypatch.setitem(sys.modules, "gurobipy", fake_gp)
    mock_audit_oracles(monkeypatch, case, fail_sid=1 if fail_audit else None)
    witnesses, events = [], []
    result = direct.solve_direct(case, ambiguity, Control(120),
                                 witness=lambda *args: witnesses.append(args), event=events.append)
    assert len(times) == 1 and 0 < times[0] <= 60
    assert result["LB"] < 1 and result["native_incumbent"]["objective"] == a["cost"] @ native
    assert result["native_incumbent"]["audited_UB"] is False
    assert result["history"][0]["native_bound"] == 1
    assert any(w[0] == "master_bound" for w in witnesses) and events[0]["status"] == "BUILDING"
    if fail_audit:
        assert result["UB"] is None and not result["passed"] and result["status"] == "AUDIT_INCOMPLETE"
    else:
        assert result["UB"] > result["native_incumbent"]["objective"]
        assert result["original_expansion_audited"] and result["status"] == "DIRECT_COMPARISON_PENDING"
