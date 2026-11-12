"""First-stage regression fixtures: procurement, occupancy and DC energy only."""
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from evcs_experiments.first_stage import FirstStagePlan, assemble_first_stage


def fixture(**task_changes):
    case = SimpleNamespace(periods=(1, 2, 3, 4), station_ids=("CS1", "CS2"),
        station_bus_ids=(2, 3), station_plugs=np.array([2, 2]),
        station_interface_kw=np.array([30., 20.]), dt_h=.5)
    attributes = dict(task_ids=("E1-CS1-k2", "E1-CS2-k2", "E2-CS1-k2"),
        vehicle_ids=("E1", "E1", "E2"), station_ids=("CS1", "CS2", "CS1"),
        chi_occ=np.array([[0, 1, 1, 1]]*3), chi_res=np.array([[0, 1, 1, 0]]*3),
        block_mask=np.array([[0, 1, 1, 0]]*3), p_dis_kw=np.array([15., 12., 10.]),
        p_block_kw=np.array([8., 6., 4.]), eta_dis=np.array([.8, .75, 1.]),
        e_forecast_dc_kwh=np.array([12.5, 12., 10.]), c_act_usd=np.array([25., 30., 20.]),
        c_cap_usd_per_kwh=np.array([[.02, .03, .04, .05], [.03, .03, .03, .03], [.01, .02, .03, .04]]),
        task_sha256="frozen-tasks", config_hash="frozen-config")
    attributes.update(task_changes)
    return case, SimpleNamespace(**attributes)


def plan(tasks, x, r):
    return FirstStagePlan(x, r, tasks.task_sha256, tasks.config_hash)


class FirstStageTests(unittest.TestCase):
    def test_positive_procurement_and_cost_units(self):
        case, tasks = fixture()
        matrix = assemble_first_stage(case, tasks)
        fixed = plan(tasks, [1, 0, 1], [[0, 8, 12, 0], [0, 0, 0, 0], [0, 4, 5, 0]])
        evaluation = matrix.evaluate(fixed)
        self.assertTrue(evaluation.feasible)
        self.assertEqual(evaluation.c_act_usd, 45.)
        self.assertAlmostEqual(evaluation.c_cap_usd, .5*(8*.03 + 12*.04 + 4*.02 + 5*.03))
        self.assertAlmostEqual(evaluation.objective_usd, 45.475)
        self.assertEqual(matrix.A.shape[1], 3 + 3*4)
        np.testing.assert_array_equal(matrix.vector(fixed), np.r_[fixed.x, fixed.r.ravel()])
        self.assertEqual(matrix.variable_map[3+1*4+2]["task_id"], tasks.task_ids[1])
        self.assertEqual(matrix.variable_map[3+1*4+2]["period"], 3)
        self.assertEqual(matrix.variable_map[3+1*4+2]["unit"], "kW_AC")
        json.dumps(matrix.as_dict(), allow_nan=False)
        json.dumps(evaluation.as_dict(), allow_nan=False)

    def test_one_vehicle_cannot_select_two_station_tasks(self):
        case, tasks = fixture()
        matrix = assemble_first_stage(case, tasks)
        result = matrix.evaluate(plan(tasks, [1, 1, 0], [[0, 8, 8, 0], [0, 6, 6, 0], [0, 0, 0, 0]]))
        bad = [matrix.constraint_map[i] for i in np.flatnonzero(result.row_violation > 1e-8)]
        self.assertFalse(result.feasible)
        self.assertEqual([row["name"] for row in bad], ["one_task_per_ev"])
        self.assertEqual(bad[0]["vehicle_id"], "E1")

    def test_two_period_block_uses_task_start_and_keeps_final_occupancy(self):
        case, tasks = fixture()
        case.station_plugs = np.array([1, 2])
        matrix = assemble_first_stage(case, tasks)
        result = matrix.evaluate(plan(tasks, [1, 0, 1], [[0, 8, 8, 0], [0, 0, 0, 0], [0, 4, 4, 0]]))
        bad = [matrix.constraint_map[i] for i in np.flatnonzero(result.row_violation > 1e-8)]
        self.assertEqual([(row["name"], row["period"]) for row in bad],
                         [("plug_occupancy", 2), ("plug_occupancy", 3), ("plug_occupancy", 4)])
        block_periods = {row["period"] for row in matrix.constraint_map if row["name"] == "minimum_block"}
        self.assertEqual(block_periods, {2, 3})

    def test_procurement_gate_minimum_block_interface_and_energy(self):
        case, tasks = fixture()
        matrix = assemble_first_stage(case, tasks)
        def families(x, r):
            result = matrix.evaluate(plan(tasks, x, r))
            return {matrix.constraint_map[i]["name"] for i in np.flatnonzero(result.row_violation > 1e-8)}
        self.assertIn("reserve_upper", families([0, 0, 0], [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        self.assertIn("reserve_upper", families([1, 0, 0], [[0, 8, 8, 1], [0, 0, 0, 0], [0, 0, 0, 0]]))
        self.assertIn("minimum_block", families([1, 0, 0], [[0, 7, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        # Both periods are below Pdis; the battery limit alone cuts this reserve.
        self.assertEqual(families([1, 0, 0], [[0, 11, 10, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), {"forecast_energy"})
        case.station_interface_kw = np.array([15., 20.])
        matrix = assemble_first_stage(case, tasks)
        result = matrix.evaluate(plan(tasks, [1, 0, 1], [[0, 8, 12, 0], [0, 0, 0, 0], [0, 4, 5, 0]]))
        bad = [matrix.constraint_map[i] for i in np.flatnonzero(result.row_violation > 1e-8)]
        self.assertEqual([(row["name"], row["period"]) for row in bad], [("station_interface", 3)])

    def test_fixed_plan_reports_integrality_and_bounds_without_clipping(self):
        case, tasks = fixture()
        matrix = assemble_first_stage(case, tasks)
        fractional = plan(tasks, [.5, 0, 0], [[0, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        result = matrix.evaluate(fractional)
        self.assertFalse(result.feasible)
        self.assertEqual(result.max_row_violation, 0.)
        self.assertEqual(result.max_bound_violation, 0.)
        self.assertEqual(result.max_integrality_violation, .5)
        negative = plan(tasks, [0, 0, 0], [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(matrix.evaluate(negative).max_bound_violation, 1.)
        self.assertEqual(negative.r[0, 0], -1.)
        with self.assertRaises(ValueError):
            matrix.evaluate(fractional, tolerance=-1)

    def test_copy_immutability_hashes_and_identity(self):
        case, tasks = fixture()
        matrix = assemble_first_stage(case, tasks)
        x, r = np.zeros(3), np.zeros((3, 4))
        fixed = plan(tasks, x, r)
        x[0], r[0, 0] = 1, 7
        self.assertEqual(fixed.x[0], 0.)
        self.assertEqual(fixed.r[0, 0], 0.)
        for array in (fixed.x, fixed.r):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        for array in (fixed.x, fixed.r, matrix.A.data, matrix.A.indices, matrix.A.indptr,
                      matrix.row_lower, matrix.row_upper, matrix.lower_bounds,
                      matrix.upper_bounds, matrix.integrality, matrix.c):
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            fixed.config_hash = "changed"
        with self.assertRaises(TypeError):
            matrix.variable_map[0]["index"] = 8
        self.assertEqual(assemble_first_stage(case, tasks).matrix_sha256, matrix.matrix_sha256)
        tasks.c_act_usd[0] += 1
        self.assertEqual(matrix.c[0], 25.)
        self.assertNotEqual(assemble_first_stage(case, tasks).matrix_sha256, matrix.matrix_sha256)
        with self.assertRaises(ValueError):
            matrix.evaluate(FirstStagePlan(fixed.x, fixed.r, "foreign-task", fixed.config_hash))
        with self.assertRaises(ValueError):
            matrix.evaluate(FirstStagePlan(fixed.x, fixed.r, fixed.task_sha256, "foreign-config"))
        with self.assertRaises(ValueError):
            matrix.evaluate(plan(tasks, [0], [[0, 0, 0, 0]]))

    def test_invalid_input_rejected_before_assembly(self):
        for change in (
            {"chi_res": np.array([[0, 1, 1, 2]]*3)},
            {"block_mask": np.array([[1, 1, 0, 0]]*3)},
            {"block_mask": np.zeros((3, 4))},
            {"eta_dis": np.array([0., .75, 1.])},
            {"eta_dis": np.array([1.1, .75, 1.])},
            {"p_block_kw": np.array([16., 6., 4.])},
            {"e_forecast_dc_kwh": np.array([np.nan, 12., 10.])},
            {"station_ids": ("foreign", "CS2", "CS1")},
            {"task_ids": ("same", "same", "different")},
        ):
            with self.subTest(change=tuple(change)):
                with self.assertRaises(ValueError):
                    assemble_first_stage(*fixture(**change))
        with self.assertRaises(ValueError):
            FirstStagePlan([float('nan')], [[0]], "tasks", "config")

    def test_empty_h_is_feasible_without_fake_selected_tasks(self):
        case, tasks = fixture(task_ids=(), vehicle_ids=(), station_ids=(), chi_occ=np.zeros((0, 4)),
            chi_res=np.zeros((0, 4)), block_mask=np.zeros((0, 4)), p_dis_kw=np.empty(0),
            p_block_kw=np.empty(0), eta_dis=np.empty(0), e_forecast_dc_kwh=np.empty(0),
            c_act_usd=np.empty(0), c_cap_usd_per_kwh=np.empty((0, 4)))
        matrix = assemble_first_stage(case, tasks)
        self.assertEqual(matrix.A.shape, (16, 0))
        result = matrix.evaluate(plan(tasks, [], np.empty((0, 4))))
        self.assertTrue(result.feasible)
        self.assertEqual(result.objective_usd, 0.)

    def test_sparse_template_runs_in_highs_without_silent_first_stage_reoptimization(self):
        case, tasks = fixture()
        matrix = assemble_first_stage(case, tasks)
        lower, upper = matrix.lower_bounds.copy(), matrix.upper_bounds.copy()
        lower[0], upper[0] = 1., 1.
        # HiGHS shares a process-global scheduler with the existing one-thread
        # moment LPs. SciPy forwards this HiGHS option while warning about it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                message=r"Unrecognized options detected: \{'threads'\}.*")
            result = milp(matrix.c, integrality=matrix.integrality, bounds=Bounds(lower, upper),
                          constraints=LinearConstraint(matrix.A, matrix.row_lower, matrix.row_upper),
                          options={"time_limit": 5., "threads": 1})
        self.assertEqual(result.status, 0)
        fixed = plan(tasks, result.x[:3], result.x[3:].reshape(3, 4))
        self.assertTrue(matrix.evaluate(fixed).feasible)
        np.testing.assert_allclose(fixed.r[0], [0, 8, 8, 0], rtol=0, atol=1e-10)
        self.assertAlmostEqual(result.fun, 25.28)
        np.testing.assert_allclose(fixed.x, [1, 0, 0], rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
