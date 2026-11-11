import dataclasses
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from evcs_experiments.case0_planning import run_case0_planning, audit_case0_recourse, audit_case0_master
from evcs_experiments.extensive import assemble_extensive, solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.validation.s03_toys import build_golden_cases


class Case0PlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=build_golden_cases()[0]
        # Explicit synthetic physics placed in the Case0 development planning domain.
        for scene in self.f.scenarios: scene.split='planning'
        self.first=assemble_first_stage(self.f.case,self.f.tasks)
        self.rec=tuple(assemble_recourse(self.f.case,self.f.tasks,s) for s in self.f.scenarios)

    def run_case(self,method,**updates):
        now=time.monotonic()
        kw=dict(case=self.f.case,tasks=self.f.tasks,scenes=self.f.scenarios,method_id=method,
            output_dir=Path(self.tmp.name)/method,backend='highs',deadline=now+1800.,
            configuration_started_at=now,empirical_probabilities=self.f.empirical_probabilities,
            scenario_cost_constants_usd=self.f.scenario_cost_constants_usd,master_seconds=5.)
        kw.update(updates)
        return run_case0_planning(self.first,self.rec,self.f.scenario_ids,self.f.probability_set,**kw)

    def test_all_four_direct_forms_match_hand_optimum_and_keep_raw_q(self):
        for method,risk in [('EA_DRO','DRO'),('OR_DRO','DRO'),('SAA','SAA'),('ARO','ARO')]:
            _,r=self.run_case(method)
            expected=self.f.expected[risk]['objective_usd']
            self.assertAlmostEqual(r['upper_bound_usd'],expected)
            self.assertAlmostEqual(r['lower_bound_usd'],expected)
            self.assertTrue(r['numerical_valid'] and r['gap_passed'])
            self.assertFalse(r['formal_acceptance'] or r['extensive_witness_costs_are_optimal_q'])
            self.assertEqual(len(r['q_usd']),3)
            self.assertTrue((Path(self.tmp.name)/method/'scenario_000/fixed_solver.json').exists())

    def test_explicit_refinement_is_forwarded_and_wrong_backend_rejected(self):
        with patch('evcs_experiments.case0_planning.solve_extensive',side_effect=RuntimeError('captured')) as call:
            with self.assertRaisesRegex(RuntimeError,'captured'):
                self.run_case('OR_DRO',backend='gurobi',solver_feasibility_tolerance=1e-9)
            self.assertEqual(call.call_args.kwargs['solver_feasibility_tolerance'],1e-9)
        with self.assertRaisesRegex(ValueError,'refinement'):
            self.run_case('SAA',solver_feasibility_tolerance=1e-9)

    def test_budget_and_nonuniform_saa_rejected(self):
        now=time.monotonic()
        with self.assertRaises(ValueError): self.run_case('SAA',deadline=now-1.)
        with self.assertRaises(ValueError): self.run_case('SAA',empirical_probabilities=[.2,.3,.5])
        with self.assertRaises(ValueError): self.run_case('SAA',master_seconds=1201.)
        with self.assertRaisesRegex(ValueError,'600-second'):
            self.run_case('SAA',deadline=time.monotonic()+500.)

    def solved(self):
        m=assemble_extensive(self.first,self.rec,risk_model='SAA',scenario_ids=self.f.scenario_ids,
                            empirical_probabilities=self.f.empirical_probabilities,
                            scenario_cost_constants_usd=self.f.scenario_cost_constants_usd)
        return m,solve_extensive(m,time_limit_seconds=5.)

    def test_raw_dual_tamper_not_trusted_even_when_flags_pass(self):
        _,s=self.solved();r=solve_fixed_plan(self.rec[0],s.solution.plan,time_limit_seconds=5.)
        cert=dict(r.certificate);cert['dual_lower_marginals']=cert['dual_lower_marginals'].copy()
        cert['dual_lower_marginals'][0]+=1.
        bad=dataclasses.replace(r,certificate=cert)
        with self.assertRaisesRegex(ValueError,'KKT'):
            audit_case0_recourse(self.rec[0],s.solution.plan,bad)

    def test_missing_native_bound_and_changed_constant_rejected(self):
        m,s=self.solved();raw=dict(s.raw_solver)
        raw['raw_lower_bound_without_constant_usd']=None
        with self.assertRaisesRegex(ValueError,'finite native'):
            audit_case0_master(m,dataclasses.replace(s,raw_solver=raw))
        with self.assertRaisesRegex(ValueError,'constant mismatch'):
            audit_case0_master(dataclasses.replace(m,objective_constant_usd=m.objective_constant_usd+100.),s)

    def test_formal_or_oos_input_rejected(self):
        self.f.scenarios[0].development=False
        with self.assertRaisesRegex(ValueError,'development planning'):
            self.run_case('SAA')

    def test_negative_constants_counted_once(self):
        _,r=self.run_case('SAA',scenario_cost_constants_usd=self.f.scenario_cost_constants_usd-100.)
        self.assertAlmostEqual(r['upper_bound_usd'],self.f.expected['SAA']['objective_usd']-100.)
        self.assertAlmostEqual(r['lower_bound_usd'],r['upper_bound_usd'])

    def test_original_coefficient_tamper_with_old_hash_rejected(self):
        badc=self.rec[0].c.copy();badc[0]+=1.
        self.rec=(dataclasses.replace(self.rec[0],c=badc),)+self.rec[1:]
        with self.assertRaisesRegex(ValueError,'coefficient mismatch'):
            self.run_case('SAA')

    def test_native_limit_is_preserved_when_numerical_gap_passes(self):
        import evcs_experiments.case0_planning as module
        real=module.solve_extensive
        def synthetic_limit(*args,**kwargs):
            solved=real(*args,**kwargs);raw=dict(solved.raw_solver)
            raw.update(status=1,success=False)
            return dataclasses.replace(solved,status='TIME_LIMIT',solver_status=1,optimal=False,raw_solver=raw)
        with patch.object(module,'solve_extensive',side_effect=synthetic_limit):
            _,r=self.run_case('SAA')
        self.assertEqual(r['native_status'],'TIME_LIMIT')
        self.assertTrue(r['gap_passed'])
        self.assertFalse(r['formal_acceptance'])

    def test_failure_of_fixed_q_never_emits_valid_result(self):
        _,s=self.solved();r=solve_fixed_plan(self.rec[0],s.solution.plan,time_limit_seconds=5.)
        with patch('evcs_experiments.case0_planning.solve_fixed_plan',return_value=dataclasses.replace(r,optimal=False)):
            with self.assertRaisesRegex(ValueError,'not optimal'):self.run_case('SAA')
        self.assertFalse((Path(self.tmp.name)/'SAA/result.json').exists())


if __name__=='__main__':unittest.main()
