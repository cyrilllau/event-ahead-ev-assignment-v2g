"""Independent golden recourse cases and deliberately infeasible candidates."""
import unittest

import numpy as np

from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan, full_shedding_fallback
from evcs_experiments.recourse_check import check_recourse
from recourse_fixtures import make_fixture, manual_solution, golden_single_bus, golden_reverse_flow, golden_islands


class RecoursePhysicsTests(unittest.TestCase):
    def report(self,f,solution):
        return check_recourse(f.case,f.tasks,f.scenario,f.plan,solution)

    def solve(self,f):
        template=assemble_recourse(f.case,f.tasks,f.scenario)
        result=solve_fixed_plan(template,f.plan,time_limit_seconds=10)
        self.assertTrue(result.optimal,result.message)
        self.assertIsNotNone(result.solution)
        report=self.report(f,result.solution)
        self.assertTrue(report['passed'],report)
        self.assertAlmostEqual(result.objective_usd,report['costs']['Qop'],places=7)
        return template,result,report

    def test_single_bus_positive_service_and_cost_separation(self):
        f=golden_single_bus();_,r,report=self.solve(f)
        np.testing.assert_allclose(r.solution.p,[[6.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.q,[[3.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.rav,[[10.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.short,[[0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.shed,[[0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.soc_dep,[.725],atol=1e-8)
        self.assertAlmostEqual(report['costs']['FV2G'],.72,places=7)
        self.assertAlmostEqual(report['costs']['Fdeg'],.30,places=7)
        self.assertAlmostEqual(r.objective_usd,1.02,places=7)
        self.assertLess(r.objective_usd,25.)  # No duplicate first-stage activation charge.

    def test_p_below_rav_has_no_certification_shortfall(self):
        f=golden_single_bus()
        s=manual_solution(f,rav=10.,p=6.,q=3.,short=0.,soc_dep=.725)
        report=self.report(f,s)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['costs']['Fshort'],0.)
        self.assertEqual(report['costs']['Qop'],1.02)

    def test_selected_available_q_only_task_is_allowed(self):
        f=make_fixture(task_station_indices=(0,0),load_p=[[6.]],tan_phi=.5,reserve=[[6.],[1.]])
        s=manual_solution(f,rav=[[6.],[1.]],p=[[6.],[0.]],q=[[0.],[3.]],soc_dep=[.725,.8])
        report=self.report(f,s)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['costs']['Qop'],1.02)
        self.assertEqual(s.p[1,0],0.)
        self.assertEqual(s.q[1,0],3.)
        # Reactive power is signed; internal absorption/supply remains legal.
        signed=manual_solution(f,rav=[[6.],[1.]],p=[[6.],[0.]],q=[[-2.],[5.]])
        self.assertTrue(self.report(f,signed)['passed'])

    def test_capability_uses_rav_instead_of_actual_p(self):
        f=make_fixture(load_p=[[6.]],tan_phi=2.5,reserve=15.)
        # (p,q)=(6,15) lies well inside 18.75 kVA, while (rav,q)=(15,15) does not.
        s=manual_solution(f,rav=15.,p=6.,q=15.)
        report=self.report(f,s)
        self.assertFalse(report['passed'])
        self.assertFalse(report['residuals']['task_capability']['passed'])
        self.assertTrue(report['residuals']['p_balance']['passed'])
        self.assertTrue(report['residuals']['q_balance']['passed'])
        # A separate station rating must bind even when the task rating passes.
        f=golden_single_bus();f.case.station_rating_kva=np.array([10.])
        s=manual_solution(f,rav=10.,p=6.,q=3.)
        report=self.report(f,s)
        self.assertTrue(report['residuals']['task_capability']['passed'])
        self.assertFalse(report['residuals']['station_capability']['passed'])

    def test_unactivated_reserve_does_not_accumulate_as_energy(self):
        f=make_fixture(periods=2,load_p=[[0.],[0.]],tan_phi=0.,reserve=15.,arrival_soc=.3875)
        _,r,_=self.solve(f)
        np.testing.assert_allclose(r.solution.rav,[[15.,15.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.p,0.,atol=1e-7)
        np.testing.assert_allclose(r.solution.short,0.,atol=1e-7)
        self.assertAlmostEqual(r.objective_usd,0.,places=7)

    def test_actual_earlier_discharge_tightens_later_reserve(self):
        f=make_fixture(periods=2,load_p=[[10.],[0.]],tan_phi=0.,reserve=15.,arrival_soc=.45)
        bad=manual_solution(f,rav=[[15.,15.]],p=[[10.,0.]],q=0.,soc_dep=.325)
        bad_report=self.report(f,bad)
        self.assertFalse(bad_report['residuals']['sustainability']['passed'])
        _,r,_=self.solve(f)
        np.testing.assert_allclose(r.solution.p,[[10.,0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.rav,[[15.,10.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.short,[[0.,5.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.soc_dep,[.325],atol=1e-8)
        self.assertAlmostEqual(r.objective_usd,6.7,places=7)

    def test_tau_two_eta_and_departure_soc_are_each_applied_once(self):
        f=make_fixture(periods=3,load_p=[[8.],[0.],[0.]],tan_phi=0.,reserve=[[15.,15.,0.]],
                       arrival_soc=.7,eta=.8,tau=2.)
        _,r,report=self.solve(f)
        np.testing.assert_allclose(r.solution.p,[[8.,0.,0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.rav,[[15.,12.,0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.short,[[0.,3.,0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.soc_dep,[.575],atol=1e-8)
        self.assertAlmostEqual(report['costs']['FV2G'],.96,places=7)
        self.assertAlmostEqual(report['costs']['Fdeg'],.5,places=7)
        self.assertAlmostEqual(r.objective_usd,4.46,places=7)

    def test_unselected_and_unavailable_tasks_cannot_supply_power(self):
        for selected,available,soc,expected in [(0.,True,.8,18.),(1.,False,.1,28.)]:
            with self.subTest(selected=selected,available=available):
                f=make_fixture(load_p=[[6.]],tan_phi=.5,reserve=10.,selected=selected,
                               available=available,arrival_soc=soc)
                _,r,_=self.solve(f)
                for value in (r.solution.rav,r.solution.p,r.solution.q):
                    np.testing.assert_allclose(value,0.,atol=1e-7)
                np.testing.assert_allclose(r.solution.shed,[[6.]],atol=1e-7)
                self.assertAlmostEqual(r.objective_usd,expected,places=7)
                self.assertAlmostEqual(r.solution.soc_dep[0],soc*selected,places=8)

    def test_reverse_signed_flow_and_one_voltage_anchor(self):
        f=golden_reverse_flow();_,r,_=self.solve(f)
        np.testing.assert_allclose(r.solution.line_p,[[-6.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.line_q,[[-3.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.u,[[.991,1.]],atol=1e-7)
        self.assertAlmostEqual(r.objective_usd,1.02,places=7)

    def test_disconnected_component_has_no_free_reference_source(self):
        f=golden_islands();_,r,_=self.solve(f)
        np.testing.assert_allclose(r.solution.shed,[[0.,0.,5.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.line_p,[[4.,0.]],atol=1e-7)
        np.testing.assert_allclose(r.solution.u[0,[0,2]],[1.,1.],atol=1e-7)
        self.assertAlmostEqual(r.objective_usd,15.68,places=7)
        f=golden_islands(selected=0.);_,r,_=self.solve(f)
        np.testing.assert_allclose(r.solution.shed,[[0.,4.,5.]],atol=1e-7)
        self.assertAlmostEqual(r.objective_usd,27.,places=7)

    def test_full_shedding_fallback_is_independently_feasible(self):
        fixtures=[golden_single_bus(),golden_reverse_flow(),golden_islands(),
                  make_fixture(periods=3,tau=2.,eta=.8,load_p=[[8.],[0.],[0.]],tan_phi=0.,
                               arrival_soc=.7,reserve=[[15.,15.,0.]])]
        for f in fixtures:
            with self.subTest(buses=f.case.bus_ids,periods=f.case.periods):
                template=assemble_recourse(f.case,f.tasks,f.scenario)
                fallback=full_shedding_fallback(template,f.plan)
                report=self.report(f,fallback)
                self.assertTrue(report['passed'],report)
                np.testing.assert_allclose(fallback.shed,f.scenario.load_p_kw,atol=0)
                np.testing.assert_allclose(fallback.p,0.,atol=0)
                np.testing.assert_allclose(fallback.q,0.,atol=0)
                np.testing.assert_allclose(fallback.rav,0.,atol=0)
                self.assertAlmostEqual(report['costs']['Qop'],report['fallback_cost_usd'],places=7)
                self.assertLessEqual(report['costs']['Qop'],report['cmax_usd']+1e-7)

    def test_incorrect_soc_and_nonzero_outage_flow_are_rejected(self):
        f=golden_single_bus()
        bad=manual_solution(f,rav=10.,p=6.,q=3.,soc_dep=.8)
        report=self.report(f,bad)
        self.assertFalse(report['residuals']['soc_definition']['passed'])
        f=golden_islands()
        bad=manual_solution(f,rav=10.,p=9.,q=0.,line_p=[[9.,5.]],u=[[1.,.991,.986]])
        report=self.report(f,bad)
        self.assertFalse(report['residuals']['line_p_limit']['passed'])


if __name__=='__main__':
    unittest.main()
