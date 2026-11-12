"""Native compact formulation shared by DET, DRO and ARO, with explicit inputs."""

from __future__ import annotations

import numpy as np

from ev_assignment.domain.planning import PlanningProblem, SolverOptions
from .native import add_first, add_recourse, add_strengthening


def parameters(model, options: SolverOptions, seconds):
    values = dict(
        Threads=options.threads,
        Seed=options.seed,
        Method=1,
        NodeMethod=1,
        PreSparsify=1,
        MIPFocus=0,
        MIPGap=options.mip_gap,
        MIPGapAbs=options.absolute_gap,
        FeasibilityTol=1e-9,
        IntFeasTol=1e-9,
        OptimalityTol=1e-9,
        SoftMemLimit=options.soft_memory_gb,
        TimeLimit=max(0.01, min(options.seconds, seconds)),
        OutputFlag=0,
    )
    for key, value in values.items():
        setattr(model.Params, key, value)


def first_stage(problem: PlanningProblem, *, env):
    import gurobipy as gp

    model = gp.Model("L08_SHARED_INTEGER_FIRST_STAGE", env=env)
    model.Params.OutputFlag = 0
    first = add_first(
        model,
        list(problem.routes),
        problem.fleet_size,
        len(problem.network["station_buses"]),
        np.asarray(problem.energy_bounds),
        problem.starts,
        problem.ends,
    )
    add_strengthening(model, first)
    model.setObjective(first["objective"])
    model.update()
    return model, first


def add_scene(model, first, problem, sid):
    scene = problem.load_scenario(sid)
    return add_recourse(
        model,
        first,
        problem.network,
        scene.physical,
        scene.route_states,
        problem.scenario_ids.index(sid),
    )


def extensive(problem, method, options, *, env):
    """The original compact deterministic/finite-support epigraph formulations."""
    import gurobipy as gp

    if method not in {"DET", "DRO", "ARO"}:
        raise ValueError("Unknown L08 method")
    if method == "DET" and len(problem.scenario_ids) != 1:
        raise ValueError("DET requires exactly one frozen forecast")
    if method == "DRO" and problem.ambiguity is None:
        raise ValueError("DRO requires the original ambiguity set")
    model, first = first_stage(problem, env=env)
    try:
        objective = first["objective"]
        risk = {}
        copies = {}
        if method == "ARO":
            risk["theta"] = model.addVar(lb=-gp.GRB.INFINITY, name="ARO_ETA")
            objective += risk["theta"]
        elif method == "DRO":
            a = problem.ambiguity
            n = len(a.lower)
            risk = dict(
                beta0=model.addVar(lb=-gp.GRB.INFINITY, name="r36_beta0"),
                plus=model.addMVar(n, name="r36_betaPlus"),
                minus=model.addMVar(n, name="r36_betaMinus"),
            )
            risk["plane_expr"] = risk["beta0"] + a.phi.T @ (risk["plus"] - risk["minus"])
            model.addConstr(
                risk["plane_expr"] >= np.zeros(len(problem.scenario_ids)),
                name="R36_ALL_SUPPORT_ZERO_LOWER_PLANE",
            )
            objective += risk["beta0"] + a.upper @ risk["plus"] - a.lower @ risk["minus"]
        for i, sid in enumerate(problem.scenario_ids):
            rc = add_scene(model, first, problem, sid)
            copies[sid] = rc
            if method == "DET":
                objective += rc["objective"]
            elif method == "ARO":
                model.addConstr(risk["theta"] >= rc["objective"])
            else:
                model.addConstr(
                    risk["plane_expr"][i] >= rc["objective"], name=f"R36_RUNNING_PLANE/{i}"
                )
        model.setObjective(objective)
        parameters(model, options, options.seconds)
        model.update()
        if method == "DET" and problem.det_proof is not None:
            from .det_strengthening import install
            install(model, problem, copies)
        return model, first, copies, risk
    except BaseException:
        model.dispose()
        raise


def arrays(model):
    model.update()
    return dict(
        A=model.getA().tocsr(),
        cost=np.asarray(model.getAttr("Obj")),
        lo=np.asarray(model.getAttr("LB")),
        hi=np.asarray(model.getAttr("UB")),
        rhs=np.asarray(model.getAttr("RHS")),
        sense=np.asarray(model.getAttr("Sense")),
        vtype=np.asarray(model.getAttr("VType")),
        names=model.getAttr("VarName"),
        row_names=model.getAttr("ConstrName"),
        objcon=float(model.ObjCon),
    )
