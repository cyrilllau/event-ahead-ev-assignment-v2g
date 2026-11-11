"""S02 closure cases independent of assembler/checker implementation details."""
import unittest

import numpy as np

from evcs_experiments.recourse import assemble_recourse, full_shedding_fallback, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from recourse_fixtures import make_fixture


class S02ClosureTests(unittest.TestCase):
    def solve(self, fixture):
        f = fixture
        model = assemble_recourse(f.case, f.tasks, f.scenario)
        result = solve_fixed_plan(model, f.plan, time_limit_seconds=5)
        self.assertTrue(result.optimal, result.message)
        self.assertTrue(result.certificate['optimal_certificate_valid'])
        report = check_recourse(f.case, f.tasks, f.scenario, f.plan, result.solution,
                                check_objective_bound=True)
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['costs']['Qop'], result.objective_usd, places=10)
        return model, result

    def test_empty_h_actual_lp_and_fallback_keep_source_free_balance(self):
        f = make_fixture(task_station_indices=(), load_p=[[6.]], tan_phi=.5)
        model, result = self.solve(f)
        self.assertEqual(result.solution.p.shape, (0, 1))
        np.testing.assert_allclose(result.solution.shed, [[6.]], atol=1e-10, rtol=0)
        self.assertAlmostEqual(result.objective_usd, 18.)
        fallback = full_shedding_fallback(model, f.plan)
        self.assertTrue(check_recourse(f.case, f.tasks, f.scenario, f.plan, fallback)['passed'])
        self.assertEqual(model.objective_components(fallback)['Qop'], 18.)

    def test_unqualified_arrival_below_departure_soc_is_not_restored(self):
        f = make_fixture(load_p=[[6.]], tan_phi=.5, arrival_soc=.1, available=False)
        # The route completed, but travel left the selected EV unqualified.
        f.scenario.task_gamma_route[:] = True
        model, result = self.solve(f)
        np.testing.assert_allclose(result.solution.soc_dep, [.1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.solution.p, [[0.]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.solution.q, [[0.]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.solution.rav, [[0.]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.solution.short, [[10.]], atol=1e-10, rtol=0)
        self.assertAlmostEqual(result.objective_usd, 28.)
        self.assertTrue(check_recourse(f.case, f.tasks, f.scenario, f.plan,
                                      full_shedding_fallback(model, f.plan))['passed'])

    def test_closed_line_active_and_reactive_limits_separately_bind_service(self):
        for p_limit, q_limit, served in ((2., 100., 2.), (100., .5, 1.)):
            with self.subTest(p_limit=p_limit, q_limit=q_limit):
                f = make_fixture(bus_ids=(1, 2), station_bus_ids=(2,), line_edges=((1, 2),),
                                 load_p=[[6., 0.]], tan_phi=.5)
                f.case.line_p_limit_kw[:] = p_limit
                f.case.line_q_limit_kvar[:] = q_limit
                _, result = self.solve(f)
                np.testing.assert_allclose(result.solution.p, [[served]], atol=1e-9, rtol=0)
                np.testing.assert_allclose(result.solution.line_p, [[-served]], atol=1e-9, rtol=0)
                np.testing.assert_allclose(result.solution.line_q, [[-.5*served]], atol=1e-9, rtol=0)
                self.assertAlmostEqual(result.objective_usd, (6.-served)*3.+served*(.12+.05), places=9)

    def test_permuted_bus_axis_preserves_physical_identity_and_complete_maps(self):
        f = make_fixture(bus_ids=(2, 1), station_bus_ids=(2,), line_edges=((1, 2),),
                         load_p=[[0., 6.]], tan_phi=.5)
        model, result = self.solve(f)
        np.testing.assert_allclose(result.solution.line_p, [[-6.]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.solution.line_q, [[-3.]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.solution.u, [[1., .991]], atol=1e-10, rtol=0)
        self.assertEqual(model.reference_bus_ids_by_period, ((2,),))
        self.assertAlmostEqual(result.objective_usd, 1.02)
        self.assertEqual([row['index'] for row in model.variable_map], list(range(len(model.c))))
        self.assertEqual([row['index'] for row in model.constraint_map_ub], list(range(model.A_ub.shape[0])))
        self.assertEqual([row['index'] for row in model.constraint_map_eq], list(range(model.A_eq.shape[0])))
        for row in (*model.variable_map, *model.constraint_map_ub, *model.constraint_map_eq):
            self.assertTrue(row['unit'])
            self.assertTrue(row['eq_label'].startswith('eq:'))


if __name__ == '__main__':
    unittest.main()
