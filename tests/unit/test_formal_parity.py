import hashlib
import json
import runpy

import numpy as np
import pytest

from ev_assignment.domain.planning import Control
from ev_assignment.l08.matrix import assemble
from ev_assignment.l08.physics import task
from ev_assignment.l08.reduction import pf_reduce
from ev_assignment.validation.l08 import audit_plan
from ev_assignment.validation.parity import decode_first, interval_parity, saved_dispatch
from ev_assignment_experiments.datasets.synthetic import problem


@pytest.mark.parametrize("values", [(1, 2, 1.5, 3), (1, 1, 1, 1)])
def test_overlapping_certificates(values):
    assert interval_parity(*values)["passed"]


@pytest.mark.parametrize("values", [(1, 2, 2.000001, 3), (3, 4, 1, 2)])
def test_separated_intervals_are_not_rounded_into_parity(values):
    assert not interval_parity(*values)["passed"]


@pytest.mark.parametrize("values", [(2, 1, 0, 1), (0, 1, 2, 1), (0, np.nan, 0, 1)])
def test_invalid_certificates(values):
    with pytest.raises(ValueError, match="Invalid comparison interval"):
        interval_parity(*values)


def test_saved_dispatch_lift_physics_and_corruption():
    p = problem("DET")
    entries = [dict(task(p.routes[0], 4, 11), contract=60.0)]
    a = audit_plan(p, entries, Control(10), retain_witness=True).checks
    scene = p.load_scenario("forecast44")
    mat = assemble(p.network, entries, scene.physical, scene.route_states, 1, p.energy_bounds)
    _, mapping = pf_reduce(mat, p.network, scene.physical)
    y = np.asarray(a["scenarios"][0]["y"])[mapping["keep"]]
    checked = saved_dispatch(p, entries, "forecast44", compact_y=y)
    assert checked["actual_V2G_ac"] == pytest.approx(60, abs=1e-7)
    assert checked["primal"] == pytest.approx(a["scenarios"][0]["primal"], abs=1e-8)
    assert checked["first_cost"] == pytest.approx(a["first_cost"], abs=1e-8)
    # A grossly altered dispatch must fail physical validation, even with an old PASS label.
    with pytest.raises(ValueError, match="physics"):
        saved_dispatch(p, entries, "forecast44", compact_y=np.full_like(y, 1e6))
    with pytest.raises(ValueError, match="Invalid historical dispatch"):
        saved_dispatch(p, entries, "forecast44", compact_y=np.full_like(y, np.nan))


def test_empty_plan_and_hidden_auxiliary_change():
    p = problem("DET")
    # One route/start/end, four vehicle scalars, window/station/reserved, three 97 clocks.
    x = np.zeros(1 + 1 + 1 + 4 + 96 + 1 + 96 + 3 * 97)
    assert decode_first(p, x) == []
    x[-1] = 1
    with pytest.raises(ValueError, match="does not match"):
        decode_first(p, x)


def test_manifest_and_full_support_fail_closed(pytestconfig, tmp_path):
    api = runpy.run_path(str(pytestconfig.rootpath / "scripts/validation/check_formal_parity.py"))
    body = b"original evidence"
    (tmp_path / "point").write_bytes(body)
    refs = {"point": dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())}
    manifest = tmp_path / "manifest.json"
    raw = json.dumps(refs).encode()
    manifest.write_bytes(raw)
    evidence = api["Evidence"](tmp_path, manifest, hashlib.sha256(raw).hexdigest())
    assert evidence.path("point").read_bytes() == body
    (tmp_path / "point").write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        evidence.path("point")
    with pytest.raises(ValueError, match="escapes"):
        evidence.path("../point")
    rows = [dict(sid=i, point_sha="locked", qL=0, qU_audited=1) for i in range(400)]
    assert api["locked_records"](rows, "locked") == rows
    for bad in (rows[:-1], list(reversed(rows)), rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match="ordered complete"):
            api["locked_records"](bad, "locked")
    rows[3] = dict(rows[3], point_sha="another_plan")
    with pytest.raises(ValueError, match="Mixed historical plan"):
        api["locked_records"](rows, "locked")
