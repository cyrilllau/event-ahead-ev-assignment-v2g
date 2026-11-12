"""Hand-computed physical checks; no first-stage assembler is imported."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from evcs_experiments.first_stage_check import check_first_stage


POLICY = {"LP_scaled_feasibility_tolerance": 1e-8,
          "physical_power_residual_tolerance_kW_kvar": 1e-5}


def handwritten_case():
    """Four periods, tau=2, including k=2 and two alternatives for EV1.

    Selected H0: r=(0,4,4,0), eta=.8, so DC=10 kWh and capacity payment=2 USD.
    Selected H2: r=(2,2,0,0), eta=.5, so DC=8 kWh and payment=1.8 USD.
    Activation=7+11=18; Fpre=21.8. Station A occupies (1,2,2,2) plugs,
    although the last period has no reserve eligibility for either task.
    """
    case = SimpleNamespace(periods=(1, 2, 3, 4), station_ids=("A", "B"),
                           station_plugs=np.array([2, 1]), station_interface_kw=np.array([10., 10.]), dt_h=1.)
    tasks = SimpleNamespace(
        task_ids=("H0", "H1", "H2"), vehicle_ids=("EV1", "EV1", "EV2"), station_ids=("A", "B", "A"),
        chi_occ=np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool),
        chi_res=np.array([[0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]], dtype=bool),
        block_mask=np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool),
        start_periods=np.array([2, 1, 1]), tau_blk_h=2.,
        p_dis_kw=np.array([10., 6., 6.]), p_block_kw=np.array([4., 3., 2.]),
        eta_dis=np.array([.8, .5, .5]), e_forecast_dc_kwh=np.array([10., 12., 12.]),
        c_act_usd=np.array([7., 9., 11.]),
        c_cap_usd_per_kwh=np.array([[.1, .2, .3, .4], [.5, .5, .5, .5], [.4, .5, .6, .7]]),
        task_sha256="handwritten-H", config_hash="handwritten-config")
    plan = SimpleNamespace(x=np.array([1., 0., 1.]), r=np.array([[0., 4., 4., 0.],
                          [0., 0., 0., 0.], [2., 2., 0., 0.]]),
                          task_sha256=tasks.task_sha256, config_hash=tasks.config_hash)
    return case, tasks, plan


class FirstStageIndependentChecks(unittest.TestCase):
    def check(self, case, tasks, plan):
        return check_first_stage(case, tasks, plan, runtime_policy=POLICY)

    def test_hand_computed_multi_period_physics_and_cost(self):
        case, tasks, plan = handwritten_case()
        before = deepcopy((case, tasks, plan))
        report = self.check(case, tasks, plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["costs"]["activation_usd"], 18.)
        self.assertAlmostEqual(report["costs"]["reservation_usd"], 3.8)
        self.assertAlmostEqual(report["costs"]["Fpre_usd"], 21.8)
        self.assertEqual(report["aggregates"]["task_reserved_energy_dc_kwh"], [10., 0., 8.])
        self.assertEqual(report["aggregates"]["station_reserved_kw"], [[2., 6., 4., 0.], [0., 0., 0., 0.]])
        self.assertEqual(report["aggregates"]["station_plug_occupancy"], [[1., 2., 2., 2.], [0., 0., 0., 0.]])
        self.assertEqual(report["max_normalized_violation"], 0.)
        json.dumps(report, allow_nan=False)
        for current, original in zip((case, tasks, plan), before):
            for name, value in vars(current).items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, getattr(original, name))

    def test_k_two_minimum_block_has_no_global_period_one_requirement(self):
        case, tasks, plan = handwritten_case()
        self.assertEqual(plan.r[0, 0], 0.)
        self.assertTrue(self.check(case, tasks, plan)["residuals"]["minimum_block"]["passed"])
        plan.r[0, 2] = 3.
        row = self.check(case, tasks, plan)["residuals"]["minimum_block"]
        self.assertFalse(row["passed"])
        self.assertEqual(row["max_violation"], 1.)
        self.assertEqual(row["worst_location"], {"task_id": "H0", "period": 3})

    def test_tau_two_terminal_occupancy_is_not_reserve_eligibility(self):
        case, tasks, plan = handwritten_case()
        report = self.check(case, tasks, plan)
        self.assertEqual(report["aggregates"]["station_plug_occupancy"][0][-1], 2.)
        self.assertFalse(tasks.chi_res[0, -1])
        plan.r[0, -1] = .1
        row = self.check(case, tasks, plan)["residuals"]["reserve_eligibility"]
        self.assertFalse(row["passed"])
        self.assertEqual(row["worst_location"], {"task_id": "H0", "period": 4})

    def test_one_vehicle_two_tasks_even_at_different_stations(self):
        case, tasks, plan = handwritten_case()
        plan.x[1] = 1.
        plan.r[1] = [3., 3., 0., 0.]
        report = self.check(case, tasks, plan)
        self.assertFalse(report["passed"])
        self.assertEqual(report["residuals"]["one_task_per_vehicle"]["max_violation"], 1.)
        self.assertEqual([key for key, row in report["residuals"].items() if not row["passed"]],
                         ["one_task_per_vehicle"])

    def test_forecast_energy_divides_by_efficiency_once(self):
        case, tasks, plan = handwritten_case()
        tasks.e_forecast_dc_kwh[0] = 9.
        row = self.check(case, tasks, plan)["residuals"]["forecast_energy"]
        self.assertFalse(row["passed"])
        self.assertEqual(row["unit"], "kWh_DC")
        self.assertAlmostEqual(row["max_violation"], 1.)
        self.assertAlmostEqual(row["max_normalized_violation"], 1/9)

    def test_period_duration_scales_energy_and_payment(self):
        case, tasks, plan = handwritten_case()
        case.dt_h = .5
        tasks.tau_blk_h = 1.
        report = self.check(case, tasks, plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["aggregates"]["task_reserved_energy_dc_kwh"], [5., 0., 4.])
        self.assertAlmostEqual(report["costs"]["Fpre_usd"], 19.9)

    def test_plugs_and_interface_are_separate_constraints(self):
        case, tasks, plan = handwritten_case()
        case.station_plugs[0] = 1
        report = self.check(case, tasks, plan)
        self.assertFalse(report["residuals"]["plug_capacity"]["passed"])
        self.assertEqual(report["residuals"]["plug_capacity"]["violation_count"], 3)
        self.assertTrue(report["residuals"]["interface_capacity"]["passed"])
        case.station_plugs[0] = 2
        case.station_interface_kw[0] = 5.
        report = self.check(case, tasks, plan)
        self.assertTrue(report["residuals"]["plug_capacity"]["passed"])
        self.assertEqual(report["residuals"]["interface_capacity"]["max_violation"], 1.)
        self.assertEqual(report["residuals"]["interface_capacity"]["worst_location"], {"station_id": "A", "period": 2})

    def test_extra_reserve_outside_minimum_block_can_be_feasible(self):
        case, tasks, plan = handwritten_case()
        plan.r[2, 2] = 1.
        report = self.check(case, tasks, plan)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["costs"]["Fpre_usd"], 22.4)

    def test_binary_nonnegative_and_unselected_reserve_gates(self):
        for field, value, expected in (("x", .5, "binary"), ("r", -.2, "r_nonnegative"),
                                        ("r", .2, "reserve_eligibility")):
            with self.subTest(field=field, value=value):
                case, tasks, plan = handwritten_case()
                if field == "x": plan.x[0] = value
                else: plan.r[1, 0] = value
                self.assertFalse(self.check(case, tasks, plan)["residuals"][expected]["passed"])

    def test_nonfinite_wrong_shape_and_hashes_are_rejected(self):
        for field, value in (("x", np.array([np.nan, 0., 1.])), ("r", np.full((3, 4), np.inf)),
                             ("x", np.ones(4)), ("task_sha256", "other-H"), ("config_hash", "other-config")):
            with self.subTest(field=field):
                case, tasks, plan = handwritten_case()
                setattr(plan, field, value)
                with self.assertRaises(ValueError): self.check(case, tasks, plan)

    def test_invalid_input_metadata_rejected(self):
        for corrupt in (lambda c, t: setattr(t, "eta_dis", np.zeros(3)),
                        lambda c, t: setattr(t, "station_ids", ("unknown", "B", "A")),
                        lambda c, t: t.chi_occ.__setitem__((0, 3), False),
                        lambda c, t: setattr(t, "p_block_kw", np.full(3, 100.))):
            case, tasks, plan = handwritten_case()
            corrupt(case, tasks)
            with self.assertRaises(ValueError): self.check(case, tasks, plan)

    def test_empty_H_is_feasible_with_zero_cost_and_occupancy(self):
        case, tasks, plan = handwritten_case()
        for name in ("task_ids", "vehicle_ids", "station_ids"): setattr(tasks, name, ())
        for name in ("chi_occ", "chi_res", "block_mask", "c_cap_usd_per_kwh"):
            setattr(tasks, name, np.empty((0, 4)))
        for name in ("start_periods", "p_dis_kw", "p_block_kw", "eta_dis", "e_forecast_dc_kwh", "c_act_usd"):
            setattr(tasks, name, np.empty(0))
        plan.x, plan.r = np.empty(0), np.empty((0, 4))
        report = self.check(case, tasks, plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["costs"]["Fpre_usd"], 0.)
        self.assertEqual(report["aggregates"]["station_plug_occupancy"], [[0.]*4, [0.]*4])

    def test_normalized_and_physical_power_tolerances_both_apply(self):
        case = SimpleNamespace(periods=(1,), station_ids=("A",), station_plugs=np.array([1]),
                               station_interface_kw=np.array([1e9]), dt_h=1.)
        tasks = SimpleNamespace(task_ids=("H",), vehicle_ids=("EV",), station_ids=("A",),
                                chi_occ=np.ones((1, 1)), chi_res=np.ones((1, 1)), block_mask=np.ones((1, 1)),
                                p_dis_kw=np.array([1e6]), p_block_kw=np.array([1.]), eta_dis=np.array([1.]),
                                e_forecast_dc_kwh=np.array([1e9]), c_act_usd=np.array([1.]),
                                c_cap_usd_per_kwh=np.ones((1, 1)), task_sha256="H", config_hash="C")
        plan = SimpleNamespace(x=np.ones(1), r=np.array([[1e6+5e-6]]), task_sha256="H", config_hash="C")
        self.assertTrue(self.check(case, tasks, plan)["passed"])
        plan.r[0, 0] = 1e6+5e-5
        row = self.check(case, tasks, plan)["residuals"]["reserve_eligibility"]
        self.assertLess(row["max_normalized_violation"], 1e-8)
        self.assertFalse(row["passed"])
        tasks.p_dis_kw[0], plan.r[0, 0] = 10., 10.+2e-7
        row = self.check(case, tasks, plan)["residuals"]["reserve_eligibility"]
        self.assertLess(row["max_violation"], 1e-5)
        self.assertFalse(row["passed"])

    def test_default_runtime_policy_is_available_through_project_config(self):
        case, tasks, plan = handwritten_case()
        report = check_first_stage(case, tasks, plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["tolerances"]["LP_scaled_feasibility_tolerance"], 1e-8)
        self.assertEqual(report["tolerances"]["physical_power_residual_tolerance_kW_kvar"], 1e-5)


if __name__ == "__main__":
    unittest.main()
