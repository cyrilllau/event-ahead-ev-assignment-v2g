"""Normalize supplied depot connectors without changing canonical road inputs.

Connector static parameters are already present in the auxiliary source. Only
canonical column names, direction expansion, midpoint sampling, and geometric
line coupling are derived here. The auxiliary endpoint XY chord defines
coupling (cross-checked against supplied WGS84 geometry); supplied curved
length and free-flow time remain the travel inputs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .wind_field import (COMMON_CENTER_LATITUDE, COMMON_CENTER_LONGITUDE,
                         latlon_to_xy, xy_to_latlon)


@dataclass(frozen=True)
class ConnectorInputs:
    arc_rows: tuple[dict[str, str], ...]
    coupling_rows: tuple[dict[str, str], ...]
    input_hashes: dict[str, str]
    diagnostics: dict
    provenance_sha256: str


def parse_linestring_wkt(value: str) -> np.ndarray:
    """Read a finite two-dimensional LINESTRING in its stored coordinate units."""
    match = re.fullmatch(r"\s*LINESTRING\s*\(([^()]*)\)\s*", value, re.IGNORECASE)
    if match is None:
        raise ValueError("Expected two-dimensional LINESTRING WKT")
    try:
        points = np.asarray([[float(x) for x in p.split()]
                             for p in match.group(1).split(",")], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed LINESTRING coordinates") from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("LINESTRING must contain at least two finite XY points")
    return points


def _point_segment_distance(p, a, b) -> float:
    delta = b - a
    norm2 = float(delta @ delta)
    weight = min(1.0, max(0.0, float((p - a) @ delta) / norm2)) if norm2 else 0.0
    return float(np.linalg.norm(p - (a + weight * delta)))


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_distance(a, b, c, d) -> float:
    ab, cd = b - a, d - c
    det = _cross(ab, cd)
    if det != 0.0:
        u, v = _cross(c - a, cd) / det, _cross(c - a, ab) / det
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def polyline_distance_km(first, second) -> float:
    """Minimum Euclidean distance between any two segments, including crossings."""
    arrays = [np.asarray(points, dtype=float) for points in (first, second)]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all() for p in arrays):
        raise ValueError("Polylines must have at least two finite XY points")
    return min(_segment_distance(a, b, c, d)
               for a, b in zip(arrays[0][:-1], arrays[0][1:])
               for c, d in zip(arrays[1][:-1], arrays[1][1:]))


def _fmt(value: float) -> str:
    return format(float(value), ".15g")


def _wkt(points) -> str:
    return "LINESTRING (" + ", ".join(" ".join(_fmt(x) for x in p) for p in points) + ")"


def derive_road_connectors(case_root: str | Path) -> ConnectorInputs:
    """Return eight normalized arcs (81–88) and their closed-line coupling rows."""
    root = Path(case_root)
    paths = {
        "edges": "04_transportation/transport_edges_with_depots__auxiliary.csv",
        "nodes": "04_transportation/transport_nodes_with_depots__auxiliary.csv",
        "lines": "02_power_network/loc_L__line_geography.csv",
        "arc_schema": "04_transportation/A__directed_road_arcs.csv",
        "coupling_schema": "08_grid_road_coupling/L_A_gamma_blk__line_road_pairs.csv",
        "registration": "07_weather/spatial_registration__IEEE33.json",
        "parameters": "01_config/model_parameters__paper_symbols.csv",
    }
    input_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                    for path in paths.values()}

    def read(name):
        with (root / paths[name]).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    registration = json.loads((root / paths["registration"]).read_text())
    center = registration["common_center"]
    if (center["latitude"], center["longitude"]) != (COMMON_CENTER_LATITUDE, COMMON_CENTER_LONGITUDE):
        raise ValueError("Connector registration differs from wind field coordinates")
    threshold = float(registration["line_road_coupling"]["minimum_segment_distance_threshold_km"])
    parameters = {r["code_key"]: float(r["value"]) for r in read("parameters")
                  if r["code_key"] in ("kappa_vul", "gamma_blk_bar")}
    edges = sorted((r for r in read("edges") if r["edge_type"] == "depot_connector"),
                   key=lambda r: int(r["edge_id"]))
    if [int(r["edge_id"]) for r in edges] != [41, 42, 43, 44]:
        raise ValueError("Expected supplied depot connector corridor IDs 41–44")
    nodes = {r["road_node"]: r for r in read("nodes")}
    lines = [r for r in read("lines") if r["hazard_eligible"] == "1" and r["normally_closed"] == "1"]
    if len(lines) != 32:
        raise ValueError("Expected 32 closed hazard-eligible lines")
    line_points = {r["ell"]: parse_linestring_wkt(r["geometry_ell_local_wkt"]) for r in lines}
    arc_columns, coupling_columns = tuple(read("arc_schema")[0]), tuple(read("coupling_schema")[0])
    fields = {
        "road_class_type_a": "road_class_group", "D_a_km": "length_km",
        "T0_a_min": "freeflow_time_min", "v0_a_km_h": "freeflow_speed_km_h",
        "kappa_tt_a": "kappa_tt", "W_mov_type_a_m_s": "movement_limit_m_s",
        "kappa_blk_type_a": "kappa_block", "alpha_R_type_a_0": "alpha0_R",
        "alpha_R_type_a_1": "alpha1_R", "gamma_blk_base_type_a": "gamma_block_base",
        "storm_vulnerable_a": "storm_vulnerable", "W_mid_type_a_m_s": "gust_midpoint_m_s",
        "source_type": "source_type",
    }
    arcs, pairs, nearest, max_endpoint_error = [], [], {}, 0.0
    for edge in edges:
        corridor = int(edge["edge_id"])
        node_points = np.array([[float(nodes[n]["x_km"]), float(nodes[n]["y_km"])]
                                for n in (edge["from_node"], edge["to_node"])])
        wgs84 = parse_linestring_wkt(edge["geometry_wkt"])
        local = latlon_to_xy(wgs84[:, ::-1])
        error = float(np.max(np.abs(local[[0, -1]] - node_points)))
        max_endpoint_error = max(max_endpoint_error, error)
        if error > 1e-6:
            raise ValueError(f"Connector {corridor} WGS84 geometry disagrees with node XY")
        midpoint = node_points.mean(axis=0)
        lat, lon = xy_to_latlon(midpoint)
        distances = {r["ell"]: polyline_distance_km(line_points[r["ell"]], node_points) for r in lines}
        nearest[str(corridor)] = min(distances.values())
        for reverse in (False, True):
            m, n = (edge["to_node"], edge["from_node"]) if reverse else (edge["from_node"], edge["to_node"])
            delta = node_points[0] - node_points[1] if reverse else node_points[1] - node_points[0]
            row = {key: "" for key in arc_columns}
            row.update({key: edge[source] for key, source in fields.items()})
            row.update(a=str(2 * corridor - 1 + int(reverse)), corridor_id=str(corridor),
                       direction="reverse" if reverse else "forward", m_road=m, n_road=n,
                       loc_a_x_km=_fmt(midpoint[0]), loc_a_y_km=_fmt(midpoint[1]),
                       loc_a_latitude=_fmt(lat), loc_a_longitude=_fmt(lon),
                       phi_a_deg_from_north=_fmt(math.degrees(math.atan2(delta[0], delta[1])) % 360),
                       geometry_a_wgs84_wkt=_wkt(wgs84[::-1] if reverse else wgs84),
                       source_status="derived directed expansion of supplied auxiliary depot connector")
            arcs.append(row)
            for line in lines:
                distance = distances[line["ell"]]
                if distance <= threshold:
                    pair = {key: "" for key in coupling_columns}
                    pair.update(ell=line["ell"], i_bus=line["i_bus"], j_bus=line["j_bus"],
                                a=row["a"], corridor_id=str(corridor), m_road=m, n_road=n,
                                road_class_type_a=row["road_class_type_a"], dist_ell_a_km=_fmt(distance),
                                d_LR_km=_fmt(threshold), gamma_blk_ell_a=_fmt(min(parameters["gamma_blk_bar"], float(row["gamma_blk_base_type_a"]) *
                                  (parameters["kappa_vul"] if int(row["storm_vulnerable_a"]) else 1.0))),
                                is_in_L_a="1", line_geometry_version="topology_preserving_v2")
                    pairs.append(pair)
    pairs.sort(key=lambda r: (int(r["ell"]), int(r["a"])))
    diagnostics = {"corridor_count": 4, "directed_arc_count": len(arcs),
                   "hazard_eligible_line_count": len(lines), "coupling_pair_count": len(pairs),
                   "physical_line_corridor_pair_count": len(pairs) // 2,
                   "distance_threshold_km": threshold, "nearest_line_distance_km": nearest,
                   "max_geometry_endpoint_error_km": max_endpoint_error,
                   "travel_length_rule": "preserve supplied length_km, including curvature factor",
                   "coupling_geometry_rule": "minimum polyline distance; supplied auxiliary endpoint XY straight chord (WGS84 cross-checked)",
                   "sampling_rule": "midpoint of supplied auxiliary endpoint XY; shared by reverse arcs"}
    payload = {"version": "S01-I04-connector-derivation-v1", "input_hashes": input_hashes,
               "arc_rows": arcs, "coupling_rows": pairs, "diagnostics": diagnostics}
    provenance = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    return ConnectorInputs(tuple(arcs), tuple(pairs), input_hashes, diagnostics, provenance)
