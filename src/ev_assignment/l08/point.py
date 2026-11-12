"""Bounded native point projection, preserving the chosen discrete first stage."""

import hashlib
import numpy as np
from .certificates import row_audit


def canonical(raw, first, *, control=None, env=None):
    raw = np.asarray(raw, float)
    binary = first["vtype"] == "B"
    if (
        raw.shape != first["lo"].shape
        or not np.isfinite(raw).all()
        or np.max(abs(raw[binary] - np.rint(raw[binary])), initial=0) > 1e-9
    ):
        raise ValueError("Noninteger or invalid first-stage candidate")
    before = row_audit(first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], raw)
    if max(before.values()) > 1e-9:
        raise ValueError(("Original first-stage residual", before))
    accepted = np.clip(raw, first["lo"], first["hi"])
    accepted[binary] = np.rint(raw[binary])
    anchors = (first["lo"] == 0) & (abs(raw) <= 1e-9)
    accepted[anchors] = 0
    after = row_audit(first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], accepted)
    projection = None
    if max(after.values()) > 1e-9:
        if control is None or env is None:
            raise ValueError("Point projection requires explicit solver budget")
        import gurobipy as gp

        control.check()
        scale = 1e-9
        ld = np.longdouble
        rhs = np.asarray(
            (np.asarray(first["rhs"], ld) - first["A"].astype(ld) @ np.asarray(raw, ld))
            / ld(scale),
            float,
        )
        low = np.maximum(np.asarray((np.asarray(first["lo"], ld) - raw) / ld(scale), float), -100)
        high = np.minimum(np.asarray((np.asarray(first["hi"], ld) - raw) / ld(scale), float), 100)
        fixed = anchors | binary
        low[fixed] = (accepted[fixed] - raw[fixed]) / scale
        high[fixed] = low[fixed]
        with gp.Model("L08_POINT_REPAIR", env=env) as model:
            model.Params.OutputFlag = 0
            dv = model.addMVar(len(raw), lb=low, ub=high)
            radius = model.addVar(lb=0, ub=100)
            model.addMConstr(first["A"], dv, first["sense"], rhs)
            model.addConstr(dv <= radius)
            model.addConstr(-dv <= radius)
            model.setObjective(radius)
            for k, v in dict(
                Threads=1,
                Method=1,
                FeasibilityTol=1e-9,
                OptimalityTol=1e-9,
                SoftMemLimit=3,
                TimeLimit=max(0.01, control.remaining),
            ).items():
                setattr(model.Params, k, v)
            control.event(dict(kind="native_begin", stage="point_repair"))
            model.optimize(lambda m, w: m.terminate() if control.stop() else None)
            control.event(
                dict(
                    kind="native_end",
                    stage="point_repair",
                    native_status=model.Status,
                    runtime=model.Runtime,
                )
            )
            if model.Status != gp.GRB.OPTIMAL:
                raise ValueError("Point projection not optimal")
            accepted = np.asarray(np.asarray(raw, ld) + ld(scale) * np.asarray(dv.X, ld), float)
            accepted = np.clip(accepted, first["lo"], first["hi"])
            accepted[anchors] = 0
            accepted[binary] = np.rint(raw[binary])
            projection = dict(status=model.Status, radius_scaled=radius.X)
        after = row_audit(
            first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], accepted
        )
    displacement = float(np.max(abs(accepted - raw), initial=0))
    if max(after.values()) > 1e-9 or displacement > 1e-7:
        raise ValueError("Point projection exceeded original tolerance")
    if control is not None:
        control.event(
            dict(
                kind="point_audit",
                before=before,
                after=after,
                displacement=displacement,
                projection=projection,
                raw_sha=hashlib.sha256(raw.tobytes()).hexdigest(),
                accepted_sha=hashlib.sha256(accepted.tobytes()).hexdigest(),
            )
        )
    return accepted
