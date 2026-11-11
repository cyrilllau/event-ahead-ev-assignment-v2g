import csv
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from evcs_experiments.config import get_paths


from evcs_experiments.road_connectors import (
    derive_road_connectors, parse_linestring_wkt, polyline_distance_km,
)

CASE_ROOT = get_paths().data_root


class GeometryTests(unittest.TestCase):
    def test_crossing_endpoint_parallel_and_collinear_segments(self):
        horizontal = [[0, 0], [2, 0]]
        for points, expected in [([[1, -1], [1, 1]], 0), ([[2, 0], [3, 1]], 0),
                                 ([[0, 3], [2, 3]], 3), ([[1, 0], [3, 0]], 0),
                                 ([[3, 0], [4, 0]], 1), ([[1, 2], [1, 2]], 2)]:
            with self.subTest(points=points):
                self.assertAlmostEqual(polyline_distance_km(horizontal, points), expected)
                self.assertAlmostEqual(polyline_distance_km(points, horizontal), expected)

    def test_minimum_over_polyline_not_endpoints(self):
        self.assertEqual(polyline_distance_km([[0, 0], [1, 1], [2, 0]], [[.5, .5], [.5, 2]]), 0)
        np.testing.assert_array_equal(parse_linestring_wkt("linestring (0 0, 1 2)"), [[0, 0], [1, 2]])
        for value in ["POINT (0 0)", "LINESTRING (0 0)", "LINESTRING (0 0 1, 1 2 3)",
                      "LINESTRING (0 0, NaN 1)"]:
            with self.assertRaises(ValueError):
                parse_linestring_wkt(value)


class ConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = derive_road_connectors(CASE_ROOT)

    def test_all_supplied_static_parameters_and_ids_preserved(self):
        with (CASE_ROOT / "04_transportation/transport_edges_with_depots__auxiliary.csv").open() as f:
            source = {int(r["edge_id"]): r for r in csv.DictReader(f)}
        with (CASE_ROOT / "04_transportation/A__directed_road_arcs.csv").open() as f:
            columns = tuple(next(csv.DictReader(f)))
        self.assertEqual([int(r["a"]) for r in self.data.arc_rows], list(range(81, 89)))
        mapping = {"D_a_km": "length_km", "T0_a_min": "freeflow_time_min",
                   "v0_a_km_h": "freeflow_speed_km_h", "kappa_tt_a": "kappa_tt",
                   "W_mov_type_a_m_s": "movement_limit_m_s", "kappa_blk_type_a": "kappa_block",
                   "alpha_R_type_a_0": "alpha0_R", "alpha_R_type_a_1": "alpha1_R",
                   "gamma_blk_base_type_a": "gamma_block_base", "storm_vulnerable_a": "storm_vulnerable",
                   "W_mid_type_a_m_s": "gust_midpoint_m_s"}
        for row in self.data.arc_rows:
            self.assertEqual(tuple(row), columns)
            original = source[int(row["corridor_id"])]
            for target, key in mapping.items():
                self.assertEqual(row[target], original[key])
            self.assertGreater(float(row["D_a_km"]), 1.8)

    def test_reverse_arcs_share_midpoint_but_reverse_geometry_and_endpoints(self):
        expected_midpoints = [(-2.7, 2.7), (4.5, 2.7), (-2.7, -2.7), (2.7, -2.7)]
        for j in range(4):
            forward, reverse = self.data.arc_rows[2*j:2*j+2]
            self.assertEqual((forward["m_road"], forward["n_road"]), (reverse["n_road"], reverse["m_road"]))
            for axis in ("x", "y"):
                self.assertEqual(forward[f"loc_a_{axis}_km"], reverse[f"loc_a_{axis}_km"])
            np.testing.assert_allclose([float(forward["loc_a_x_km"]), float(forward["loc_a_y_km"])], expected_midpoints[j])
            np.testing.assert_array_equal(parse_linestring_wkt(forward["geometry_a_wgs84_wkt"]),
                                          parse_linestring_wkt(reverse["geometry_a_wgs84_wkt"])[::-1])
            self.assertAlmostEqual((float(reverse["phi_a_deg_from_north"])-float(forward["phi_a_deg_from_north"])) % 360, 180)

    def test_no_forced_nearest_line_coupling(self):
        self.assertEqual(self.data.coupling_rows, ())
        self.assertEqual(self.data.diagnostics["hazard_eligible_line_count"], 32)
        np.testing.assert_allclose(list(self.data.diagnostics["nearest_line_distance_km"].values()),
                                   [.96, 2.1773458616502976, .8, 1.18], atol=1e-12)
        self.assertLess(self.data.diagnostics["max_geometry_endpoint_error_km"], 1e-6)

    def test_effective_gamma_vulnerability_and_cap_on_new_geometric_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in self.data.input_hashes:
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CASE_ROOT / relative, root / relative)
            geom = root / "02_power_network/loc_L__line_geography.csv"
            with geom.open() as f:
                rows = list(csv.DictReader(f))
            rows[0]["geometry_ell_local_wkt"] = "LINESTRING (-3.5 2.8, -2.5 2.8)"
            rows[1]["geometry_ell_local_wkt"] = "LINESTRING (4.3 2.8, 5.2 2.8)"
            with geom.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            result = derive_road_connectors(root)
            relevant = {(int(r["ell"]), int(r["a"])): r for r in result.coupling_rows}
            self.assertAlmostEqual(float(relevant[1, 81]["gamma_blk_ell_a"]), .35)
            self.assertAlmostEqual(float(relevant[2, 83]["gamma_blk_ell_a"]), .42)
            self.assertAlmostEqual(float(relevant[2, 84]["gamma_blk_ell_a"]), .42)
            self.assertAlmostEqual(float(relevant[1, 81]["dist_ell_a_km"]), .1)
            params = root / "01_config/model_parameters__paper_symbols.csv"
            with params.open() as f:
                rows = list(csv.DictReader(f))
            next(r for r in rows if r["code_key"] == "gamma_blk_bar")["value"] = ".4"
            with params.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            capped = derive_road_connectors(root)
            self.assertAlmostEqual(float(next(r for r in capped.coupling_rows if r["a"] == "83")["gamma_blk_ell_a"]), .4)
            self.assertNotEqual(capped.provenance_sha256, result.provenance_sha256)

    def test_deterministic_provenance_and_source_immutability(self):
        repeated = derive_road_connectors(CASE_ROOT)
        self.assertEqual(self.data, repeated)
        self.assertEqual(len(self.data.input_hashes), 7)
        self.assertEqual(len(self.data.provenance_sha256), 64)
        self.assertEqual(len(self.data.arc_rows), 8)


if __name__ == "__main__":
    unittest.main()
