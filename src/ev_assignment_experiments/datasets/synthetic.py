"""Small L08 physics example; explicitly developmental, never a paper result."""

from fractions import Fraction
import hashlib
import json
import numpy as np
from scipy import sparse
from ev_assignment.domain.planning import PlanningProblem, Scenario, AmbiguitySet


def definition():
    return dict(
        schema_version=1,
        kind="synthetic_l08",
        load_kw=30.0,
        outage_start=4,
        outage_end=12,
        arrival_h=1.0,
        probability_lower=0.25,
        probability_upper=0.75,
    )


def problem(method, config=None):
    config = definition() if config is None else dict(config)
    if (
        set(config) != set(definition())
        or config["schema_version"] != 1
        or config["kind"] != "synthetic_l08"
    ):
        raise ValueError("Unknown synthetic input schema")
    if method not in {"DET", "DRO", "ARO"}:
        raise ValueError("Unknown method")
    if not np.isfinite(config["load_kw"]) or not 0 < config["load_kw"] <= 90:
        raise ValueError("Invalid synthetic load")
    if (config["outage_start"], config["outage_end"], config["arrival_h"]) != (4, 12, 1.0):
        raise ValueError("This fixture locks its clock and routes")
    lo, hi = config["probability_lower"], config["probability_upper"]
    if not 0 <= lo <= 0.5 <= hi <= 1:
        raise ValueError("Invalid original probability interval")
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
    ids = ("forecast44",) if method == "DET" else ("train/0000", "train/0001")

    def scene(sid):
        if sid not in ids:
            raise ValueError("Outside locked synthetic domain")
        load = np.zeros((96, 2))
        load[4:12, 1] = config["load_kw"]
        outage = np.zeros((96, 1), dtype=bool)
        if sid in {"forecast44", "train/0000"}:
            outage[4:12] = True
        a = dict(line_operating_outages=outage, load_p_kw=load, load_q_kvar=0.33 * load)
        return Scenario(sid, a, (dict(success=True, arrival=1.0),))

    identity = hashlib.sha256(
        json.dumps(dict(config, method=method), sort_keys=True).encode()
    ).hexdigest()
    ambiguity = None
    if method == "DRO":

        def check(p):
            p = np.asarray(p, float)
            ok = (
                p.shape == (2,)
                and np.isfinite(p).all()
                and np.all(p >= 0)
                and sum(map(Fraction.from_float, p), Fraction(0)) == 1
                and lo <= p[0] <= hi
            )
            return dict(passed=bool(ok), original_rows=1)

        ambiguity = AmbiguitySet(
            ids,
            sparse.csr_matrix([[1.0, 0.0]]),
            np.array([lo]),
            np.array([hi]),
            identity,
            check,
            np.array([0.5, 0.5]),
        )
    return PlanningProblem(
        identity,
        "L08_FULL_DWELL_FORECAST44_V1",
        routes,
        1,
        net,
        np.array([180.0]),
        ids,
        scene,
        ambiguity,
        starts=(4,),
        ends=(12,),
        scope="development",
    )
