"""Hand optima, original-dual separation, and interruption boundary tests."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.ccg import CCGProblem, run_ccg
from evcs_experiments.extensive import solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases


class CCGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = build_golden_cases()
        cls.problems = {}
        cls.results = {}
        for i, f in enumerate(cls.fixtures):
            for risk in ('DRO', 'ARO'):
                p = CCGProblem(assemble_first_stage(f.case, f.tasks),
                    tuple(assemble_recourse(f.case, f.tasks, s) for s in f.scenarios),
                    f.probability_set, f.scenario_cost_constants_usd, risk)
                cls.problems[i, risk] = p
                cls.results[i, risk] = run_ccg(p, wall_seconds=10,
                    mip_relative_gap=1e-9, optimality_relative_gap=1e-8)

    def test_six_hand_optima_and_independent_best_plan_reoptimization(self):
        expected = ((32.12, 34.2), (39., 43.2), (41.6, 41.6))
        for (i, risk), result in self.results.items():
            with self.subTest(i=i, risk=risk):
                self.assertEqual(result.status, 'CONVERGED')
                optimum = expected[i][risk == 'ARO']
                self.assertAlmostEqual(result.state['upper_bound_usd'], optimum, places=7)
                self.assertAlmostEqual(result.state['lower_bound_usd'], optimum, places=7)
                f, p = self.fixtures[i], self.problems[i, risk]
                self.assertTrue(check_first_stage(f.case, f.tasks, result.best_plan)['passed'])
                costs = []
                for j, template in enumerate(p.recourses):
                    fresh = solve_fixed_plan(template, result.best_plan, time_limit_seconds=5)
                    self.assertTrue(fresh.certificate['optimal_certificate_valid'])
                    self.assertTrue(check_recourse(f.case, f.tasks, f.scenarios[j], result.best_plan,
                                                  fresh.solution, check_objective_bound=True)['passed'])
                    costs.append(fresh.objective_usd+p.scenario_cost_constants_usd[j])
                np.testing.assert_allclose(costs, result.state['best']['q_usd'], atol=1e-8, rtol=0)

    def test_history_bounds_contain_known_optimum_and_are_monotone(self):
        for (i, risk), result in self.results.items():
            optimum = self.fixtures[i].expected[risk]['objective_usd']
            previous_lb, previous_ub = -np.inf, np.inf
            for row in result.state['trace']:
                lb, ub = row['lower_bound_usd'], row['upper_bound_usd']
                self.assertGreaterEqual(lb, previous_lb)
                self.assertLessEqual(ub, previous_ub)
                self.assertLessEqual(lb, optimum+1e-7)
                self.assertGreaterEqual(ub, optimum-1e-7)
                self.assertLessEqual(row['rmp']['lower_bound_usd'], row['rmp']['objective_usd']+1e-7)
                previous_lb, previous_ub = lb, ub

    def test_original_rmp_dual_is_used_on_every_support_atom(self):
        saw_violation = False
        for (i, risk), result in self.results.items():
            phi = self.problems[i, risk].ambiguity.features
            for row in result.state['trace']:
                q = np.array(row['q_usd']); dual = row['raw_rmp_dual']
                v = q-dual['threshold'] if risk == 'ARO' else q-dual['theta0']+phi@(np.array(dual['alpha'])-dual['beta'])
                np.testing.assert_array_equal(v, row['residual_usd'])
                self.assertEqual(row['separation_scale_usd'], max(1., max(abs(q))))
                if row['max_full_residual_usd'] > row['separation_tolerance_usd']:
                    saw_violation = True
                    self.assertIsNotNone(row['added_scenario_index'])
                self.assertLessEqual(row['max_included_residual_usd'], row['separation_tolerance_usd'])
        self.assertTrue(saw_violation)

    def test_resume_after_each_completed_iteration_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.problems[2, 'DRO']; path = Path(tmp)/'checkpoint.json'
            result = run_ccg(p, checkpoint_path=path, wall_seconds=10, max_iterations=1,
                             mip_relative_gap=1e-9, optimality_relative_gap=1e-8)
            previous_time = result.state['elapsed_seconds']
            for _ in range(15):
                if result.status == 'CONVERGED':
                    break
                result = run_ccg(p, checkpoint_path=path, resume=True, wall_seconds=10, max_iterations=1,
                                 mip_relative_gap=1e-9, optimality_relative_gap=1e-8)
                self.assertGreaterEqual(result.state['elapsed_seconds'], previous_time)
                previous_time = result.state['elapsed_seconds']
            self.assertEqual(result.status, 'CONVERGED')
            self.assertAlmostEqual(result.state['upper_bound_usd'], self.results[2, 'DRO'].state['upper_bound_usd'])
            self.assertGreater(len(result.state['sessions']), 1)
            self.assertAlmostEqual(sum(s['elapsed_seconds'] for s in result.state['sessions']), previous_time)
            again = run_ccg(p, checkpoint_path=path, resume=True, mip_relative_gap=1e-9,
                            optimality_relative_gap=1e-8)
            self.assertEqual(again.state, result.state)

    def test_timeout_without_incumbent_retains_real_bound_only(self):
        def timed(model, **kw):
            solved = solve_extensive(model, **kw)
            return replace(solved, status='TIME_LIMIT', solver_status=1, optimal=False,
                           solution=None, objective_usd=None)
        with patch('evcs_experiments.ccg.solve_extensive', side_effect=timed):
            result = run_ccg(self.problems[0, 'DRO'], wall_seconds=10)
        self.assertEqual(result.status, 'TIME_LIMIT')
        self.assertIsNotNone(result.state['lower_bound_usd'])
        self.assertIsNone(result.state['upper_bound_usd'])
        self.assertIsNone(result.best_plan)

    def test_timeout_incumbent_is_not_renamed_converged(self):
        def timed(model, **kw):
            return replace(solve_extensive(model, **kw), status='TIME_LIMIT', solver_status=1, optimal=False)
        with patch('evcs_experiments.ccg.solve_extensive', side_effect=timed):
            result = run_ccg(self.problems[1, 'DRO'], wall_seconds=10)
        self.assertEqual(result.status, 'TIME_LIMIT')
        self.assertIsNotNone(result.best_plan)
        self.assertEqual(len(result.state['trace']), 1)
        self.assertIsNone(result.state['trace'][0]['max_new_residual_usd'])

    def test_native_limit_reason_survives_with_and_without_incumbent(self):
        for reason in ('MEM_LIMIT', 'INTERRUPTED'):
            for has_incumbent in (True, False):
                def stopped(model, **kw):
                    result = solve_extensive(model, **kw)
                    extra = {} if has_incumbent else dict(solution=None, objective_usd=None)
                    return replace(result, status=reason, solver_status=1, optimal=False, **extra)
                with self.subTest(reason=reason, incumbent=has_incumbent):
                    with patch('evcs_experiments.ccg.solve_extensive', side_effect=stopped):
                        result = run_ccg(self.problems[1, 'DRO'], wall_seconds=10)
                    self.assertEqual(result.status, reason)
                    self.assertIsNotNone(result.state['lower_bound_usd'])

    def test_resume_rejects_switching_to_gurobi_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.json'
            problem = self.problems[2, 'DRO']
            run_ccg(problem, checkpoint_path=path, wall_seconds=10, max_iterations=1)
            with self.assertRaisesRegex(ValueError, 'problem/source/config mismatch'):
                run_ccg(problem, checkpoint_path=path, resume=True, master_backend='gurobi',
                        optimizer_threads=4, solver_log_directory=Path(tmp)/'logs')

    def test_all_support_present_unclosed_gap_is_not_convergence(self):
        def loose(model, **kw):
            result = solve_extensive(model, **kw)
            return replace(result, lower_bound_usd=result.lower_bound_usd-10., optimal=False)
        with patch('evcs_experiments.ccg.solve_extensive', side_effect=loose):
            result = run_ccg(self.problems[1, 'DRO'], wall_seconds=10, mip_relative_gap=1e-9)
        self.assertEqual(result.status, 'RMP_NEEDS_MORE_WORK')
        self.assertTrue(result.state['trace'][0]['no_omitted_scenarios'])
        self.assertIsNone(result.state['trace'][0]['added_scenario_index'])

    def test_included_positive_residual_fails_instead_of_adding_duplicate(self):
        def inconsistent(model, **kw):
            result = solve_extensive(model, **kw)
            return replace(result, solution=replace(result.solution, theta0=result.solution.theta0-100.))
        with patch('evcs_experiments.ccg.solve_extensive', side_effect=inconsistent):
            result = run_ccg(self.problems[0, 'DRO'], wall_seconds=10)
        self.assertEqual(result.status, 'RMP_RESIDUAL_FAILED')
        self.assertIsNone(result.state['trace'][0]['added_scenario_index'])

    def test_recourse_timeout_preserves_completed_costs_for_resume(self):
        calls = []
        def interrupted(template, plan, **kw):
            calls.append(template.scenario_sha256)
            result = solve_fixed_plan(template, plan, **kw)
            return replace(result, status='LIMIT_REACHED', solver_status=1, optimal=False) if len(calls)==2 else result
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.json'; p = self.problems[0, 'DRO']
            with patch('evcs_experiments.ccg.solve_fixed_plan', side_effect=interrupted):
                first = run_ccg(p, checkpoint_path=path, wall_seconds=10)
            self.assertEqual(first.status, 'TIME_LIMIT')
            self.assertEqual(len(first.state['pending']['q_usd']), 1)
            with patch('evcs_experiments.ccg.solve_fixed_plan', wraps=solve_fixed_plan) as spy:
                resumed = run_ccg(p, checkpoint_path=path, resume=True, wall_seconds=10, max_iterations=1)
            self.assertEqual(spy.call_count, len(p.recourses)-1)
            self.assertIsNone(resumed.state['pending'])
            self.assertEqual(resumed.state['trace'][0]['iteration'], 1)

    def test_failed_recourse_never_supplies_zero_or_partial_q_bound(self):
        def failed(template, plan, **kw):
            result = solve_fixed_plan(template, plan, **kw)
            return replace(result, status='SOLVER_ERROR', solver_status=4, optimal=False, objective_usd=None)
        with patch('evcs_experiments.ccg.solve_fixed_plan', side_effect=failed):
            result = run_ccg(self.problems[0, 'DRO'], wall_seconds=10)
        self.assertEqual(result.status, 'RECOURSE_FAILED')
        self.assertIsNone(result.state['upper_bound_usd'])
        self.assertEqual(result.state['pending']['q_usd'], [])

    def test_resume_rejects_changed_inputs_options_or_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.problems[2, 'DRO']; path = Path(tmp)/'checkpoint.json'
            run_ccg(p, checkpoint_path=path, wall_seconds=10, max_iterations=1)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                run_ccg(p, checkpoint_path=path)
            changed = replace(p, scenario_cost_constants_usd=p.scenario_cost_constants_usd+1.)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                run_ccg(changed, checkpoint_path=path, resume=True)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                run_ccg(p, checkpoint_path=path, resume=True, optimality_relative_gap=.1)
            saved = json.loads(path.read_text()); saved['state']['upper_bound_usd'] = -99.
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                run_ccg(p, checkpoint_path=path, resume=True)

    def test_single_scenario_dro_and_aro_agree(self):
        p = self.problems[0, 'DRO']
        a = ProbabilitySet((p.ambiguity.support_ids[-1],), (), np.empty((1, 0)), [], [])
        for risk in ('DRO', 'ARO'):
            result = run_ccg(CCGProblem(p.first_stage, (p.recourses[-1],), a, np.array([3.]), risk), wall_seconds=10)
            self.assertEqual(result.status, 'CONVERGED')
            self.assertAlmostEqual(result.state['upper_bound_usd'], 34.2)

    def test_diagnostic_constant_shift_restores_both_bounds(self):
        p = self.problems[0, 'DRO']
        result = run_ccg(replace(p, scenario_cost_constants_usd=p.scenario_cost_constants_usd-100.), wall_seconds=10,
                         mip_relative_gap=1e-9, optimality_relative_gap=1e-8)
        self.assertEqual(result.status, 'CONVERGED')
        self.assertAlmostEqual(result.state['lower_bound_usd'], -67.88)
        self.assertAlmostEqual(result.state['upper_bound_usd'], -67.88)

    def test_invalid_budgets_and_unsupported_method_rejected(self):
        p = self.problems[0, 'DRO']
        for kwargs in ({'wall_seconds':0}, {'mip_relative_gap':-1}, {'max_iterations':0}):
            with self.assertRaises(ValueError):
                run_ccg(p, **kwargs)
        with self.assertRaises(ValueError):
            replace(p, risk_model='SAA')

    def test_resume_in_fresh_process_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.problems[2, 'DRO']; path = Path(tmp)/'checkpoint.json'
            initial = run_ccg(p, checkpoint_path=path, max_iterations=1, wall_seconds=10)
            code = '''import sys,json
from evcs_experiments.ccg import CCGProblem,run_ccg
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
f=build_golden_cases()[2]
p=CCGProblem(assemble_first_stage(f.case,f.tasks),tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios),f.probability_set,f.scenario_cost_constants_usd)
r=run_ccg(p,checkpoint_path=sys.argv[1],resume=True,wall_seconds=10)
print(json.dumps(r.as_dict()))
'''
            child = subprocess.run([sys.executable, '-c', code, str(path)], cwd=tmp,
                                   text=True, capture_output=True, timeout=20, check=True)
            result = json.loads(child.stdout)
            self.assertEqual(result['status'], 'CONVERGED')
            self.assertAlmostEqual(result['upper_bound_usd'], 41.6)
            self.assertGreaterEqual(result['elapsed_seconds'], initial.state['elapsed_seconds'])
            self.assertEqual(len(result['sessions']), 2)

    def test_tiny_positive_phase1_mass_is_kept_in_initial_master(self):
        p = self.problems[0, 'DRO']
        # A direct 1e-12 equality can legally be rounded to zero by the LP's
        # 1e-8 feasibility tolerance. This scaling forces a REAL raw 1e-12 p,
        # as in the independently accepted Phase-I regression.
        a = ProbabilitySet(p.ambiguity.support_ids, ('tiny','half'),
                           [[1e6,0.],[0.,1.],[0.,0.]], [1e-6,.5], [1e-6,.5])
        result = run_ccg(replace(p, probability_set=a), wall_seconds=10)
        self.assertEqual(result.status, 'CONVERGED')
        self.assertEqual(result.state['trace'][0]['active_indices'], [0,1,2])
        self.assertEqual(result.state['phase1']['probabilities'][0], 1e-12)


if __name__ == '__main__':
    unittest.main()
