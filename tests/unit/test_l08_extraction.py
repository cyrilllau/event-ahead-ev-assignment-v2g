"""Compare extracted kernels against frozen, unmodified mathematical definitions.

Only named definitions are evaluated; side-effectful script imports/setup never run.
The reference is the baseline source, not a second handwritten implementation.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as sp
from scipy.optimize import linprog
from dataclasses import dataclass

from ev_assignment.l08 import energy, physics, matrix, reduction


def baseline(root):
    folder = root / "powerai_l08/L08_K4L4_DRO_READY_PARALLEL_R36/tools"
    ns = {
        "np": np,
        "sp": sp,
        "linprog": linprog,
        "dataclass": dataclass,
        "CFG": {"ports": 20, "epsilon_ls": 1e-6},
        "digest": reduction.digest,
    }
    selections = {
        "energy_ledger.py": {"forecast", "actual", "battery_trace"},
        "r3_common.py": {"tou", "qualify", "ebar", "first_cost", "components", "incident", "task"},
        "contract_matrix.py": {
            "Rows",
            "Matrix",
            "first_stage",
            "assemble",
            "fallback",
            "compress",
            "restore",
        },
        "exact_transform.py": {"identity_map", "pf_reduce", "lift"},
    }
    for name, names in selections.items():
        parsed = ast.parse((folder / name).read_text())
        nodes = [
            n
            for n in parsed.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(folder / name), "exec"), ns)
    return ns


def inputs(outage=True, success=True, arrival=1.0):
    net = dict(
        buses=[1, 2],
        line_ids=[0],
        **{"from": [1], "to": [2]},
        line_from=[1],
        line_to=[2],
        station_buses=[2],
        shed_cost=[10, 50],
        p_limit=[100.0],
        q_limit=[100.0],
        u_min=0.9,
        u_max=1.1,
        u_ref=1.0,
        mp=[0.0001],
        mq=[0.0001],
        big_m=1.0,
    )
    off = np.zeros((96, 1), dtype=bool)
    if outage:
        off[4:12] = True
    load = np.full((96, 2), 10.0)
    scene = dict(line_operating_outages=off, load_p_kw=load, load_q_kvar=0.33 * load)
    route = dict(
        route_id=0,
        e=0,
        c=0,
        d=0,
        departure_dc=300.0,
        distance=10.0,
        arrival=1.0,
        B=188.0,
        ev="synthetic-ev",
    )
    tasks = [physics.task(route, 4, 11)]
    states = [dict(success=success, arrival=arrival)]
    return net, tasks, scene, states


def same_matrix(a, b):
    for name in ("cost", "parts", "b", "d", "lo", "hi"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ("A", "B", "H", "G"):
        aa, bb = getattr(a, name), getattr(b, name)
        assert aa.shape == bb.shape
        np.testing.assert_array_equal(aa.indptr, bb.indptr)
        np.testing.assert_array_equal(aa.indices, bb.indices)
        np.testing.assert_array_equal(aa.data, bb.data)
    assert a.row_names == b.row_names and a.metadata == b.metadata
    for key in a.index:
        np.testing.assert_array_equal(a.index[key], b.index[key])


@pytest.mark.parametrize(
    "outage,success,arrival",
    [
        (False, True, 1.0),
        (True, True, 1.0),
        (True, True, 1.75),
        (True, False, float("inf")),
        (True, True, 8.0),
    ],
)
def test_original_ordered_matrices_physics_and_solution(pytestconfig, outage, success, arrival):
    old = baseline(Path(pytestconfig.rootpath))
    net, tasks, scene, states = inputs(outage, success, arrival)
    args = net, tasks, scene, states, 1, np.array([180.0])
    a, b = old["assemble"](*args), matrix.assemble(*args)
    same_matrix(a, b)
    old_first = old["first_stage"](tasks, 1, 1, [180.0])
    new_first = matrix.first_stage(tasks, 1, 1, [180.0])
    assert (old_first[0] != new_first[0]).nnz == 0
    for i in (1, 2, 3):
        np.testing.assert_array_equal(old_first[i], new_first[i])
    assert old_first[4] == new_first[4]
    z = np.array([1.0, 90.0])
    solved_a, solved_b = a.solve(z), b.solve(z)
    assert solved_a["objective"] == pytest.approx(solved_b["objective"], abs=1e-7)
    assert b.residual(solved_b["y"], z) <= 1e-7
    if not outage:
        assert solved_b["objective"] == pytest.approx(0.0, abs=1e-7)
    if outage and (not success or arrival >= 3.0):
        assert solved_b["objective"] == pytest.approx(1090.0, abs=1e-7)
    ra, ma = old["pf_reduce"](a, net, scene)
    rb, mb = reduction.pf_reduce(b, net, scene)
    same_matrix(ra, rb)
    assert ma["applicable"] == mb["applicable"]
    yy = rb.solve(z)["y"]
    lifted = reduction.lift(yy, mb)
    assert b.residual(lifted, z) <= 1e-7
    assert b.cost @ lifted == pytest.approx(solved_b["objective"], abs=1e-6)


def test_nonradial_reduction_fails_closed(pytestconfig):
    net, tasks, scene, states = inputs()
    m = matrix.assemble(net, tasks, scene, states, 1, [180.0])
    net["line_from"] = [2]
    red, mapping = reduction.pf_reduce(m, net, scene)
    assert red is m and not mapping["applicable"]
    assert mapping["reason"] == "NOT_APPLICABLE_ENDPOINT_METADATA_MISMATCH"


def test_full_dwell_energy_and_departure_floor():
    forecast = energy.forecast(288.0, 1.0, 4, 11)
    assert forecast["aux_dc"] == 2.0
    assert forecast["cap_ac"] == pytest.approx(176.7)
    actual = energy.actual(288.0, 1.75, 4, 11)
    assert actual["mu"].sum() == 5
    assert actual["available_ac"] == 112.5
    trace = energy.battery_trace(288.0, 1.0, 3.0, np.zeros(96))
    assert trace["terminal_dc"] == 286.0
    assert energy.actual(288.0, 1.0, 4, 11, success=False)["available_ac"] == 0.0
