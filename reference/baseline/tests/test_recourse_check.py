"""Checker-only handwritten candidates: no solver or recourse matrix imports."""
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from evcs_experiments.recourse_check import check_recourse
from recourse_fixtures import make_fixture, manual_solution, golden_single_bus


class RecourseCheckerTests(unittest.TestCase):
    def report(self, fixture, solution, **kwargs):
        return check_recourse(fixture.case, fixture.tasks, fixture.scenario,
                              fixture.plan, solution, **kwargs)

    def test_handwritten_costs_and_original_solution_unchanged(self):
        f = golden_single_bus()
        s = manual_solution(f, rav=10., p=6., q=3., soc_dep=.725)
        before = {k:v.copy() for k,v in vars(s).items() if isinstance(v,np.ndarray)}
        report = self.report(f,s,check_objective_bound=True)
        self.assertTrue(report['passed'])
        for key,value in dict(Fshed=0.,Fshort=0.,FV2G=.72,Fdeg=.3,Qop=1.02).items():
            self.assertAlmostEqual(report['costs'][key],value,places=12)
        self.assertEqual(report['cmax_usd'],318.)
        self.assertEqual(report['fallback_cost_usd'],28.)
        self.assertFalse(report['optimality_certified'])
        json.dumps(report,allow_nan=False)
        for name, array in before.items(): np.testing.assert_array_equal(getattr(s,name),array)

    def test_high_cost_feasible_candidate_is_not_an_optimal_value(self):
        f = golden_single_bus()
        s = manual_solution(f,rav=10.,p=6.,q=3.,short=400.,soc_dep=.725)
        report = self.report(f,s)
        self.assertTrue(report['passed'])
        self.assertTrue(report['physical_feasible'])
        self.assertFalse(report['objective_bound']['passed'])
        self.assertGreater(report['costs']['Qop'],report['cmax_usd'])
        self.assertFalse(self.report(f,s,check_objective_bound=True)['passed'])

    def test_first_stage_is_checked_without_reoptimizing_plan(self):
        f = golden_single_bus()
        f.plan = SimpleNamespace(x=np.array([.5]),r=np.array([[10.]]),
                                 task_sha256=f.tasks.task_sha256,config_hash=f.tasks.config_hash)
        s = manual_solution(f,rav=0.,p=0.,q=0.,short=10.,shed=6.,soc_dep=.4)
        report = self.report(f,s)
        self.assertFalse(report['passed'])
        self.assertFalse(report['first_stage']['residuals']['binary']['passed'])
        np.testing.assert_array_equal(f.plan.x,[.5])

    def test_station_capability_is_independent_of_task_facets(self):
        f = golden_single_bus(); f.case.station_rating_kva[:] = 10.
        s = manual_solution(f,rav=10.,p=6.,q=3.)
        report = self.report(f,s)
        self.assertTrue(report['residuals']['task_capability']['passed'])
        self.assertFalse(report['residuals']['station_capability']['passed'])

    def test_negative_reactive_task_dispatch_is_allowed(self):
        f = make_fixture(task_station_indices=(0,0),load_p=[[0.]],tan_phi=0.,reserve=1.)
        s = manual_solution(f,rav=1.,p=0.,q=[[-3.],[3.]],soc_dep=[.8,.8])
        self.assertTrue(self.report(f,s)['passed'])

    def test_voltage_reference_uses_one_minimum_evcs_bus_per_island(self):
        f = make_fixture(bus_ids=(3,1,2),station_bus_ids=(3,2),line_edges=((1,2),(2,3)),
                         task_station_indices=(0,),load_p=[[0.,0.,0.]],tan_phi=0.,reserve=1.)
        s = manual_solution(f,rav=1.,p=0.,q=0.)
        report = self.report(f,s)
        self.assertTrue(report['passed'])
        self.assertEqual(report['reference_bus_ids_by_period'],[[2]])
        f.scenario.line_operating_outages[0,1] = True
        report = self.report(f,s)
        self.assertEqual(report['reference_bus_ids_by_period'],[[2,3]])
        s.u[0,0] = 1.01
        report = self.report(f,s)
        self.assertFalse(report['residuals']['voltage_reference']['passed'])
        self.assertTrue(report['residuals']['voltage_drop']['passed'])

    def test_voltage_drop_and_squared_voltage_bounds_are_checked(self):
        f = make_fixture(bus_ids=(1,2),station_bus_ids=(1,),line_edges=((1,2),),
                         load_p=[[0.,6.]],tan_phi=.5,reserve=10.)
        good = manual_solution(f,rav=10.,p=6.,q=3.,line_p=[[6.]],line_q=[[3.]],u=[[1.,.991]])
        self.assertTrue(self.report(f,good)['passed'])
        bad = copy.deepcopy(good); bad.u[0,1] = .992
        self.assertFalse(self.report(f,bad)['residuals']['voltage_drop']['passed'])
        bad.u[0,1] = 1.2
        self.assertFalse(self.report(f,bad)['residuals']['voltage_bounds']['passed'])

    def test_eligibility_and_shortfall_are_separate_gates(self):
        f = golden_single_bus()
        f.scenario.task_mu_full[:] = False  # Qualified EV arrives too late for this period.
        s = manual_solution(f,rav=10.,p=0.,q=0.,shed=6.)
        report = self.report(f,s)
        self.assertFalse(report['residuals']['rav_available']['passed'])
        s.rav[:] = 0.
        report = self.report(f,s)
        self.assertTrue(report['residuals']['rav_available']['passed'])
        self.assertFalse(report['residuals']['shortfall']['passed'])

    def test_power_tolerance_and_normalization_are_both_required(self):
        f = golden_single_bus()
        s = manual_solution(f,rav=10.,p=6.,q=3.)
        s.p[0,0] += 1e-7
        self.assertTrue(self.report(f,s)['residuals']['p_balance']['passed'])
        s.p[0,0] += 1e-6
        row = self.report(f,s)['residuals']['p_balance']
        self.assertLess(row['max_violation'],1e-5)
        self.assertGreater(row['max_normalized_violation'],1e-8)
        self.assertFalse(row['passed'])
        f.tasks.p_dis_kw[:] = 1e9  # Large fixed scale cannot excuse a physical kW residual.
        s.p[0,0] = 6.+2e-5
        row = self.report(f,s)['residuals']['p_balance']
        self.assertLess(row['max_normalized_violation'],1e-8)
        self.assertGreater(row['max_violation'],1e-5)
        self.assertFalse(row['passed'])

    def test_energy_and_soc_tolerances_are_reported_in_physical_units(self):
        f = make_fixture(periods=3,tau=2.,eta=.8,arrival_soc=.7,
                         reserve=[[15.,15.,0.]],load_p=[[8.],[0.],[0.]],tan_phi=0.)
        s = manual_solution(f,rav=[[15.,12.,0.]],p=[[8.,0.,0.]],q=0.,short=[[0.,3.,0.]],soc_dep=.575)
        report = self.report(f,s)
        self.assertTrue(report['passed'])
        row = report['residuals']['sustainability']
        self.assertEqual(row['unit'],'kWh_DC')
        self.assertAlmostEqual(row['original_tolerance_min'],2/.8*1e-5)
        self.assertAlmostEqual(row['original_tolerance_max'],4/.8*1e-5)
        s.soc_dep[0] += 5e-8
        row = self.report(f,s)['residuals']['soc_definition']
        self.assertLess(row['max_violation'],1e-7)
        self.assertFalse(row['passed'])

    def test_identity_shape_nonfinite_and_physical_input_mismatch_rejected(self):
        f = golden_single_bus(); s = manual_solution(f,rav=10.,p=6.,q=3.)
        changes = [('scenario_sha256','different'),('task_sha256','different'),
                   ('config_hash','different'),('q',np.array([[np.nan]])),('p',np.array([6.]))]
        for key,value in changes:
            with self.subTest(key=key):
                bad = copy.deepcopy(s); setattr(bad,key,value)
                with self.assertRaises(ValueError): self.report(f,bad)
        for mutate in (lambda g: setattr(g.scenario,'task_ids',('wrong',)),
                       lambda g: setattr(g.scenario,'physical_case_sha256','wrong'),
                       lambda g: g.scenario.load_q_kvar.__setitem__((0,0),3.001),
                       lambda g: g.scenario.task_usable_energy_dc_kwh.__setitem__(0,49.),
                       lambda g: g.tasks.bus_indices.__setitem__(0,1),
                       lambda g: setattr(g.case,'capability_active_coordinate','actual_dispatch')):
            bad_fixture = copy.deepcopy(f); mutate(bad_fixture)
            with self.assertRaises(ValueError): self.report(bad_fixture,s)

    def test_empty_h_all_shedding_has_finite_report(self):
        f = make_fixture(task_station_indices=(),load_p=[[6.]],tan_phi=.5)
        s = manual_solution(f,rav=np.empty((0,1)),p=np.empty((0,1)),q=np.empty((0,1)),shed=6.)
        report = self.report(f,s,check_objective_bound=True)
        self.assertTrue(report['passed'])
        self.assertEqual(report['costs']['Qop'],18.)
        self.assertEqual(report['fallback_cost_usd'],18.)
        self.assertEqual(report['residuals']['task_capability']['constraint_count'],0)
        json.dumps(report,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
