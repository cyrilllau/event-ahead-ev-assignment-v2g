"""Explicit paired samples. Sample bytes are opened only after a sealed plan lock."""

import io
import json
from pathlib import Path
import numpy as np
from ev_assignment.domain.planning import Scenario
from ev_assignment_experiments.tracking.store import Ref, encode, sha256
from ev_assignment_experiments.tracking.transfer import file_digest
from ev_assignment_experiments.workflows.l08 import plain


def domain_identity(problem):
    return sha256(
        encode(
            plain(
                dict(
                    protocol=problem.protocol,
                    routes=problem.routes,
                    fleet_size=problem.fleet_size,
                    network=problem.network,
                    energy_bounds=problem.energy_bounds,
                    starts=problem.starts,
                    ends=problem.ends,
                )
            )
        )
    )


def validate_manifest(value):
    if (
        set(value)
        != {"schema_version", "scope", "domain_sha256", "protocol", "random_stream", "samples"}
        or value["schema_version"] != 1
    ):
        raise ValueError("Invalid evaluation manifest")
    if (
        value["scope"] not in {"development", "oos"}
        or not value["samples"]
        or not value["random_stream"]
    ):
        raise ValueError("Explicit evaluation scope and stream required")
    Ref("objects", value["domain_sha256"])
    if value["protocol"] != "L08_PAIRED_FIXED_PLAN_V1":
        raise ValueError("Unsupported common evaluation protocol")
    ids = []
    for row in value["samples"]:
        if set(row) != {"sample_id", "physical", "route_states", "origin"} or not row["origin"]:
            raise ValueError("Missing evaluation sample provenance")
        if not isinstance(row["sample_id"], str) or not row["sample_id"]:
            raise ValueError("Invalid sample identity")
        ids.append(row["sample_id"])
        for key in ("physical", "route_states"):
            item = row[key]
            if set(item) != {"path", "sha256"}:
                raise ValueError("Invalid sample object")
            p = Path(item["path"])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Sample path escape")
            Ref("objects", item["sha256"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate paired sample identity")
    return ids


def seal_sample(store, root, descriptor):
    root = Path(root).resolve()
    refs = {}
    for key in ("physical", "route_states"):
        row = descriptor[key]
        ref = Ref("objects", row["sha256"])
        if not store.path(ref).exists():
            path = root / row["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root)
                or file_digest(path) != ref.id
            ):
                raise ValueError("Evaluation sample digest/path mismatch")
            if store.put_file(path) != ref:
                raise ValueError("Evaluation sample changed during sealing")
        store.closure([ref])
        refs[key] = ref
    return refs


def scene(store, descriptor, problem):
    with np.load(
        io.BytesIO(store.object(Ref("objects", descriptor["physical"]["sha256"]))),
        allow_pickle=False,
    ) as z:
        physical = {k: z[k] for k in z.files}
    expected = {"line_operating_outages", "load_p_kw", "load_q_kvar"}
    if not expected <= set(physical):
        raise ValueError("Missing quarterly physical evaluation arrays")
    shape = (96, len(problem.network["buses"]))
    load = physical["load_p_kw"]
    outage = physical["line_operating_outages"]
    if (
        load.shape != shape
        or not np.isfinite(load).all()
        or np.any(load < 0)
        or physical["load_q_kvar"].shape != shape
        or not np.array_equal(physical["load_q_kvar"], 0.33 * load)
        or outage.shape != (96, len(problem.network["line_ids"]))
        or not np.isin(outage, [0, 1]).all()
    ):
        raise ValueError("Evaluation physical dimensions/units/clock changed")
    replay = json.loads(store.object(Ref("objects", descriptor["route_states"]["sha256"])))
    if set(replay) != {"route_ids", "states"} or replay["route_ids"] != [
        r["route_id"] for r in problem.routes
    ]:
        raise ValueError("Evaluation changed fixed route order")
    states = replay["states"]
    if len(states) != len(problem.routes):
        raise ValueError("Evaluation route state length changed")
    cleaned = []
    for st in states:
        if set(st) != {"success", "arrival"} or type(st["success"]) is not bool:
            raise ValueError("Invalid replay state")
        # Failed trips use null arrival; never smuggle NaN into sealed records.
        if st["success"] and (
            not isinstance(st["arrival"], (int, float))
            or not np.isfinite(st["arrival"])
            or st["arrival"] < 0
        ):
            raise ValueError("Invalid successful trip arrival")
        if not st["success"] and st["arrival"] is not None:
            raise ValueError("Failed trip must have null arrival")
        cleaned.append(
            dict(success=st["success"], arrival=st["arrival"] if st["success"] else float("inf"))
        )
    return Scenario(descriptor["sample_id"], physical, tuple(cleaned))
