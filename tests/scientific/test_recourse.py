"""Parameterized LP interface and certificates; physical golden cases are separate."""
from dataclasses import replace
import json
import unittest

import numpy as np

from ev_assignment.models.first_stage import FirstStagePlan
from ev_assignment.models.recourse import (
    assemble_recourse, full_shedding_fallback, solve_fixed_plan,
)
from recourse_fixtures import golden_single_bus, golden_islands, make_fixture


class RecourseTemplateTests(unittest.TestCase):
    def test_rhs_retains_distinct_selection_and_procurement_columns(self):
        f = make_fixture(periods=2, reserve=[[10., 12.]], load_p=[[4.], [2.]])
        model = assemble_recourse(f.case,f.tasks,f.scenario)
        bu,be = model.fixed_rhs(f.plan)
        zero = FirstStagePlan([0.],[[0.,0.]],f.tasks.task_sha256,f.tasks.config_hash)
        zu,ze = model.fixed_rhs(zero)
        np.testing.assert_allclose(bu-zu,model.Bx_ub@f.plan.x+model.Br_ub@f.plan.r.ravel(),atol=0,rtol=0)
        np.testing.assert_allclose(be-ze,model.Bx_eq@f.plan.x+model.Br_eq@f.plan.r.ravel(),atol=0,rtol=0)
        self.assertEqual(model.Bx_ub.shape[1],1)
        self.assertEqual(model.Br_ub.shape[1],2)
        procurement = [r['index'] for r in model.constraint_map_ub if r['name']=='certification_procurement']
        np.testing.assert_array_equal(bu[procurement],[10.,12.])
        shortfall = [r['index'] for r in model.constraint_map_ub if r['name']=='station_shortfall']
        np.testing.assert_array_equal(bu[shortfall],[-10.,-12.])
        self.assertFalse(any('substation' in row['name'] for row in model.variable_map))

    def test_lp_primal_dual_certificate_is_numerical_evidence(self):
        f = golden_single_bus()
        model = assemble_recourse(f.case,f.tasks,f.scenario)
        result = solve_fixed_plan(model,f.plan,time_limit_seconds=5)
        self.assertTrue(result.optimal)
        self.assertEqual(result.solver_status,0)
        self.assertTrue(result.certificate['optimal_certificate_valid'])
        self.assertEqual(result.certificate['certificate_status'],'PASS_NUMERIC_LP_CERTIFICATE')
        self.assertAlmostEqual(result.objective_usd,1.02)
        self.assertAlmostEqual(result.certificate['dual_bound_usd'],1.02)
        self.assertLess(result.certificate['duality_gap_usd'],1e-8)
        self.assertLess(result.certificate['max_dual_stationarity_violation'],1e-8)
        self.assertEqual(result.certificate['max_invalid_infinite_bound_marginal'],0.)
        self.assertAlmostEqual(result.certificate['duality_gap_tolerance_usd'],1e-4+1e-7*1.02)
        self.assertEqual(result.certificate['options']['threads'],1)
        self.assertGreater(result.wall_seconds,0)
        self.assertEqual(set(result.objective_components),{'Fshed','Fshort','FV2G','Fdeg','Qop'})
        self.assertNotIn('Fpre',result.objective_components)
        json.dumps(result.as_dict(),allow_nan=False)
        json.dumps(model.as_dict(),allow_nan=False)

    def test_fallback_and_cost_bound_have_different_scopes(self):
        f = golden_islands()
        model = assemble_recourse(f.case,f.tasks,f.scenario)
        fallback = full_shedding_fallback(model,f.plan)
        vector = model.pack(fallback)
        bu,be = model.fixed_rhs(f.plan)
        self.assertLessEqual(np.max(model.A_ub@vector-bu),1e-10)
        self.assertLessEqual(np.max(np.abs(model.A_eq@vector-be)),1e-10)
        self.assertLessEqual(model.objective_components(fallback)['Qop'],model.cmax_usd)
        expensive = replace(fallback,short=fallback.short+model.cmax_usd+10)
        vector = model.pack(expensive)
        self.assertLessEqual(np.max(model.A_ub@vector-bu),1e-10)
        self.assertLessEqual(np.max(np.abs(model.A_eq@vector-be)),1e-10)
        self.assertGreater(model.objective_components(expensive)['Qop'],model.cmax_usd)
        # Feasible surplus shortfall is costly, not a violation of LP physics.
        self.assertEqual(model.reference_bus_ids_by_period,((1,3),))

    def test_inputs_and_output_identity_and_immutable_solution(self):
        f = golden_single_bus()
        model = assemble_recourse(f.case,f.tasks,f.scenario)
        fallback = full_shedding_fallback(model,f.plan)
        for name in model.variable_shapes:
            array = getattr(fallback,name)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        reconstructed = model.unpack(model.pack(fallback))
        np.testing.assert_array_equal(reconstructed.shed,fallback.shed)
        with self.assertRaises(ValueError):
            model.pack(replace(fallback,scenario_sha256='foreign'))
        with self.assertRaises(ValueError):
            model.fixed_rhs(FirstStagePlan(f.plan.x,f.plan.r,'foreign',f.plan.config_hash))
        with self.assertRaises(ValueError):
            solve_fixed_plan(model,FirstStagePlan([.5],[[5.]],f.tasks.task_sha256,f.tasks.config_hash))
        original_hash = model.template_sha256
        self.assertEqual(assemble_recourse(f.case,f.tasks,f.scenario).template_sha256,original_hash)
        f.scenario.load_q_kvar = np.array([[3.01]])
        with self.assertRaises(ValueError):
            assemble_recourse(f.case,f.tasks,f.scenario)

    def test_own_solver_timeout_parameter_is_validated(self):
        f=golden_single_bus(); model=assemble_recourse(f.case,f.tasks,f.scenario)
        for limit in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                solve_fixed_plan(model,f.plan,time_limit_seconds=limit)


if __name__=='__main__':
    unittest.main()
