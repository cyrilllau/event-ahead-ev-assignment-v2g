"""Full physical extensive-form hand goldens and honest MIP status boundaries."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.extensive import assemble_extensive, solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases


class ExtensiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=build_golden_cases(); cls.inputs=[]; cls.models={}; cls.results={}
        for i,case in enumerate(cls.cases):
            first=assemble_first_stage(case.case,case.tasks)
            templates=tuple(assemble_recourse(case.case,case.tasks,s) for s in case.scenarios)
            cls.inputs.append((first,templates))
            for method in ("SAA","DRO","ARO"):
                model=cls.build(i,method)
                cls.models[i,method]=model
                cls.results[i,method]=solve_extensive(model,time_limit_seconds=5)

    @classmethod
    def build(cls,i,method,**overrides):
        case=cls.cases[i]; first,templates=cls.inputs[i]
        args={"risk_model":method,"scenario_ids":case.scenario_ids,
              "probability_set":case.probability_set,"empirical_probabilities":case.empirical_probabilities,
              "scenario_cost_constants_usd":case.scenario_cost_constants_usd}
        args.update(overrides)
        return assemble_extensive(first,templates,**args)

    def test_explicit_gurobi_solver_refinement_forwarding_and_default_unchanged(self):
        model=self.models[0,'SAA']
        for tol in (None,1e-9,1e-8):
            with patch('evcs_experiments.native_gurobi.solve_milp',side_effect=RuntimeError('captured')) as call:
                with self.assertRaisesRegex(RuntimeError,'captured'):
                    solve_extensive(model,master_backend='gurobi',solver_log_path='unused.log',
                                    solver_feasibility_tolerance=tol)
                opts=call.call_args.kwargs['options']
                if tol is None:self.assertNotIn('solver_feasibility_tolerance',opts)
                else:self.assertEqual(opts['solver_feasibility_tolerance'],tol)
                self.assertEqual(opts['primal_feasibility_tolerance'],1e-8)
                self.assertEqual(opts['dual_feasibility_tolerance'],1e-8)
        for backend,tol in [('highs',1e-9),('gurobi',1e-7),('gurobi',True)]:
            with self.assertRaises(ValueError):
                solve_extensive(model,master_backend=backend,solver_log_path='unused.log',solver_feasibility_tolerance=tol)

    def test_nine_full_model_hand_goldens_and_independent_physics(self):
        # Constants (1,2,3) are deliberately nonuniform in the first case.
        expected=((30.266666666666666,32.12,34.2),(39.,39.,43.2),(35.,41.6,41.6))
        counts=((2,3,3),(1,1,2),(2,4,4))
        for i,case in enumerate(self.cases):
            for j,method in enumerate(("SAA","DRO","ARO")):
                with self.subTest(case=case.name,method=method):
                    result=self.results[i,method]
                    self.assertTrue(result.optimal,result.as_dict())
                    self.assertAlmostEqual(result.objective_usd,expected[i][j],places=7)
                    self.assertAlmostEqual(result.lower_bound_usd,expected[i][j],places=7)
                    self.assertAlmostEqual(result.solution.plan.x.sum(),counts[i][j],places=7)
                    self.assertTrue(check_first_stage(case.case,case.tasks,result.solution.plan)["passed"])
                    for scene,solution in zip(case.scenarios,result.solution.recourse_solutions):
                        self.assertTrue(check_recourse(case.case,case.tasks,scene,result.solution.plan,
                                                       solution,check_objective_bound=False)["passed"])

    def test_binary_columns_and_risk_intercept_remain_unfixed_and_free(self):
        for (i,method),model in self.models.items():
            h=len(self.cases[i].tasks.task_ids)
            np.testing.assert_array_equal(model.integrality[:h],np.ones(h))
            np.testing.assert_array_equal(model.integrality[h:],np.zeros(len(model.c)-h))
            np.testing.assert_array_equal(model.lower_bounds[:h],np.zeros(h))
            np.testing.assert_array_equal(model.upper_bounds[:h],np.ones(h))
            if method!="SAA":
                key="theta0" if method=="DRO" else "risk_threshold"
                index=model.variable_slices[key][0]
                self.assertEqual(model.lower_bounds[index],-np.inf)
                self.assertEqual(model.upper_bounds[index],np.inf)

    def test_uniform_negative_cost_shift_preserves_plan_and_shifts_both_bounds(self):
        for method in ("SAA","DRO","ARO"):
            model=self.build(0,method,scenario_cost_constants_usd=self.cases[0].scenario_cost_constants_usd-100.)
            shifted=solve_extensive(model,time_limit_seconds=5); original=self.results[0,method]
            self.assertTrue(shifted.optimal,shifted.as_dict())
            self.assertAlmostEqual(shifted.objective_usd,original.objective_usd-100.,places=7)
            self.assertAlmostEqual(shifted.lower_bound_usd,original.lower_bound_usd-100.,places=7)
            self.assertAlmostEqual(shifted.solution.plan.x.sum(),original.solution.plan.x.sum(),places=7)
            if method=="ARO": self.assertLess(shifted.solution.risk_threshold,0.)
            if method=="DRO": self.assertLess(shifted.solution.theta0,0.)

    def test_nonbinding_scenario_witness_is_not_a_recourse_optimum(self):
        case=self.cases[2];model=self.models[2,"ARO"];original=self.results[2,"ARO"].solution
        vector=original.vector.copy();template=model.recourses[0]
        start,stop=model.variable_slices["recourses"][case.scenario_ids[0]]
        local=np.zeros(stop-start)
        for name,value in (("rav",original.plan.r),("u",np.ones((2,1))),
                           ("soc_dep",.8*original.plan.x)):
            local[slice(*template.variable_slices[name])]=np.asarray(value).ravel()
        local[template.variable_slices["short"][0]]=.25
        vector[start:stop]=local
        candidate=model.unpack(vector)
        check=check_recourse(case.case,case.tasks,case.scenarios[0],candidate.plan,
                            candidate.recourse_solutions[0],check_objective_bound=False)
        self.assertTrue(check["passed"])
        activity=model.A@vector
        self.assertLessEqual(float(np.max(activity-model.row_upper)),1e-8)
        self.assertAlmostEqual(candidate.scenario_witness_costs_usd[0],.25)
        fresh=solve_fixed_plan(template,candidate.plan)
        self.assertTrue(fresh.optimal)
        self.assertAlmostEqual(fresh.objective_usd,0.,places=8)
        self.assertTrue(model.as_dict()["scenario_witness_costs_need_fixed_plan_reoptimization"])

    def test_explicit_identity_and_probability_order_checks(self):
        with self.assertRaisesRegex(ValueError,"Probability support"):
            self.build(0,"DRO",scenario_ids=self.cases[0].scenario_ids[::-1])
        first,recourses=self.inputs[0]
        bad=replace(recourses[0],task_sha256="wrong")
        with self.assertRaisesRegex(ValueError,"identity"):
            assemble_extensive(first,(bad,*recourses[1:]),risk_model="ARO",scenario_ids=self.cases[0].scenario_ids)
        with self.assertRaisesRegex(ValueError,"explicit scenario"):
            self.build(0,"ARO",scenario_ids=("a","a","b"),probability_set=None)

    def test_invalid_empirical_law_and_missing_probability_model_rejected(self):
        with self.assertRaisesRegex(ValueError,"empirical probability"):
            self.build(0,"SAA",empirical_probabilities=[-.1,.6,.5])
        with self.assertRaisesRegex(ValueError,"empirical probability"):
            self.build(0,"SAA",empirical_probabilities=[.2,.2,.2])
        with self.assertRaisesRegex(ValueError,"requires ProbabilitySet"):
            self.build(0,"DRO",probability_set=None)
        with self.assertRaisesRegex(ValueError,"requires explicit"):
            self.build(0,"SAA",empirical_probabilities=None)

    def test_single_scenario_saa_dro_aro_agree_without_removing_tasks(self):
        case=self.cases[0];first,templates=self.inputs[0]
        ids=(case.scenario_ids[-1],)
        probability=ProbabilitySet(ids,(),np.empty((1,0)),[],[])
        for method in ("SAA","DRO","ARO"):
            model=assemble_extensive(first,(templates[-1],),risk_model=method,scenario_ids=ids,
                probability_set=probability,empirical_probabilities=[1.],scenario_cost_constants_usd=[3.])
            result=solve_extensive(model,time_limit_seconds=5)
            self.assertTrue(result.optimal,result.as_dict())
            self.assertAlmostEqual(result.objective_usd,34.2,places=7)
            self.assertEqual(len(result.solution.plan.x),3)

    def _fake(self,status,*,vector=None,bound=None,gap=None,fun=None):
        return SimpleNamespace(status=status,x=vector,mip_dual_bound=bound,mip_gap=gap,fun=fun,
                               message="injected boundary",success=status==0,mip_node_count=0)

    def test_timeout_without_incumbent_keeps_only_reported_bound(self):
        model=self.models[0,"SAA"]
        with patch("evcs_experiments.extensive.milp",return_value=self._fake(1,bound=10.)):
            result=solve_extensive(model)
        self.assertEqual(result.status,"TIME_LIMIT");self.assertFalse(result.optimal)
        self.assertIsNone(result.solution)
        self.assertEqual(result.lower_bound_usd,12.)  # SAA raw bound + constant.

    def test_timeout_valid_incumbent_is_not_optimal_even_with_zero_gap(self):
        model=self.models[0,"SAA"];good=self.results[0,"SAA"]
        rawobjective=good.objective_usd-model.objective_constant_usd
        fake=self._fake(1,vector=good.solution.vector,bound=rawobjective,gap=0.,fun=rawobjective)
        with patch("evcs_experiments.extensive.milp",return_value=fake): result=solve_extensive(model)
        self.assertEqual(result.status,"TIME_LIMIT");self.assertFalse(result.optimal)
        self.assertTrue(result.certificate["primal_feasible"])
        self.assertAlmostEqual(result.lower_bound_usd,good.objective_usd)

    def test_solver_status_zero_cannot_hide_unclosed_mip_gap(self):
        model=self.models[0,"SAA"];good=self.results[0,"SAA"]
        rawobjective=good.objective_usd-model.objective_constant_usd
        for gap in (.1,1.00001e-9):
            with self.subTest(gap=gap):
                fake=self._fake(0,vector=good.solution.vector,bound=rawobjective,gap=gap,fun=rawobjective)
                with patch("evcs_experiments.extensive.milp",return_value=fake): result=solve_extensive(model)
                self.assertFalse(result.optimal);self.assertFalse(result.certificate["raw_gap_passed"])

    def test_missing_bound_and_failure_state_bound_are_not_invented(self):
        model=self.models[0,"SAA"];good=self.results[0,"SAA"]
        fake=self._fake(0,vector=good.solution.vector,gap=0.,fun=good.objective_usd-2.)
        with patch("evcs_experiments.extensive.milp",return_value=fake): result=solve_extensive(model)
        self.assertIsNone(result.lower_bound_usd);self.assertFalse(result.optimal)
        for status in (2,3,4):
            with self.subTest(status=status):
                with patch("evcs_experiments.extensive.milp",return_value=self._fake(status,bound=999.)):
                    result=solve_extensive(model)
                self.assertIsNone(result.lower_bound_usd)
                self.assertFalse(result.certificate["solver_bound_usable"])
                self.assertEqual(result.raw_solver["raw_lower_bound_without_constant_usd"],999.)

    def test_immutable_arrays_json_metadata_and_complete_row_blocks(self):
        model=self.models[0,"DRO"];result=self.results[0,"DRO"]
        for array in (model.c,model.A.data,model.row_lower,result.solution.vector):
            with self.assertRaises(ValueError):array.setflags(write=True)
        cursor=0
        for block in model.block_map:
            self.assertEqual(block["row_start"],cursor);cursor=block["row_stop"]
        self.assertEqual(cursor,model.A.shape[0])
        self.assertEqual(len(model.variable_slices["recourses"]),3)
        json.dumps(model.as_dict(),allow_nan=False)
        json.dumps(result.as_dict(),allow_nan=False)


if __name__=="__main__":unittest.main()
