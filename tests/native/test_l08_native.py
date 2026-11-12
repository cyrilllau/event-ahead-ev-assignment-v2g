"""Licensed, local Gurobi validation; deliberately excluded from unlicensed CI."""

import ast
from pathlib import Path

import gurobipy as gp
import numpy as np
import pytest

from ev_assignment.domain.planning import PlanningProblem, Scenario, SolverOptions
from ev_assignment.l08 import native, builders, physics, reduction, matrix
from ev_assignment.l08.energy import actual, forecast


def fixture():
    net = dict(
        buses=[1, 2],
        line_ids=[0],
        line_from=[1],
        line_to=[2],
        **{"from": [1], "to": [2]},
        station_buses=[2],
        shed_cost=[10.0, 50.0],
        p_limit=[200.0],
        q_limit=[200.0],
        u_min=0.9,
        u_max=1.1,
        u_ref=1.0,
        mp=[0.0001],
        mq=[0.0001],
        big_m=1.0,
    )
    load = np.zeros((96, 2))
    load[4:12, 1] = 30.0
    a = dict(
        line_operating_outages=np.ones((96, 1), dtype=bool), load_p_kw=load, load_q_kvar=0.33 * load
    )
    routes = (
        dict(
            e=0,
            c=0,
            d=0,
            route_id=0,
            arrival=1.0,
            distance=10.0,
            departure_dc=300.0,
            B=188.0,
            ev="synthetic-ev",
        ),
    )

    def scene(sid):
        return Scenario(sid, a, (dict(success=True, arrival=1.0),))

    return PlanningProblem(
        "synthetic-native-l08",
        "L08_FULL_DWELL_FORECAST44_V1",
        routes,
        1,
        net,
        np.array([180.0]),
        ("forecast44",),
        scene,
        starts=(4,),
        ends=(12,),
    )


def baseline(root):
    folder = root / "powerai_l08/L08_K4L4_DRO_READY_PARALLEL_R36/tools"
    ns = dict(
        np=np,
        gp=gp,
        GRB=gp.GRB,
        CFG={"aux_dc_kw": 1.0, "ports": 20, "epsilon_ls": 1e-6},
        task=physics.task,
        assemble=matrix.assemble,
        pf_reduce=reduction.pf_reduce,
        actual=actual,
        forecast=forecast,
    )
    for name, keep in [
        ("compact_first_stage.py", {"add_first", "decode", "fix_first"}),
        ("exact_transform.py", {"add_strengthening", "compact_base", "original_expressions"}),
        ("compact_recourse_pf.py", {"add_recourse"}),
    ]:
        nodes = [
            n
            for n in ast.parse((folder / name).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in keep
        ]
        # compact_base's only internal import resolves to the already baseline-
        # identical explicit matrix kernel, without importing campaign setup.
        for n in nodes:
            if n.name == "compact_base":
                n.body = [x for x in n.body if not isinstance(x, ast.ImportFrom)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), ns)
    return ns


def compare(a, b):
    assert (a["A"] != b["A"]).nnz == 0 and a["A"].shape == b["A"].shape
    for key in ("cost", "lo", "hi", "rhs", "sense", "vtype"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["names"] == b["names"] and a["row_names"] == b["row_names"]
    assert a["objcon"] == b["objcon"]


def test_real_native_compact_coefficients_and_mip(pytestconfig):
    problem = fixture()
    old = baseline(Path(pytestconfig.rootpath))
    with gp.Env(params={"OutputFlag": 0}) as env:
        with gp.Model(env=env) as ref:
            fs = old["add_first"](
                ref, list(problem.routes), 1, 1, problem.energy_bounds, problem.starts, problem.ends
            )
            old["add_strengthening"](ref, fs)
            sc = problem.load_scenario("forecast44")
            rc = old["add_recourse"](ref, fs, problem.network, sc.physical, sc.route_states, 0)
            ref.setObjective(fs["objective"] + rc["objective"])
            ref.update()
            new, nfs, _, _ = builders.extensive(problem, "DET", SolverOptions(), env=env)
            try:
                compare(builders.arrays(ref), builders.arrays(new))
                builders.parameters(ref, SolverOptions(), 60)
                ref.optimize()
                new.optimize()
                assert ref.Status == new.Status == gp.GRB.OPTIMAL
                assert ref.ObjVal == pytest.approx(new.ObjVal, abs=1e-7)
                assert native.decode(nfs) == old["decode"](fs)
                assert new.ObjVal < 3000  # zero-plan 2 h * 30 kW * $50/kWh
                assert len(native.decode(nfs)) == 1
            finally:
                new.dispose()


def test_no_solver_on_native_module_import(tmp_path):
    import subprocess, sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ev_assignment.l08.native; import ev_assignment.l08.builders; assert 'gurobipy' not in sys.modules",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert not list(tmp_path.iterdir())
