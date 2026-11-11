"""Retained native SAA status/bounds and sequential original physical witnesses."""
from dataclasses import replace
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.saa_streaming import assemble_streaming_saa
from evcs_experiments.saa_streaming_solve import solve_streaming_saa
from evcs_experiments.validation.s03_toys import build_golden_cases


class StreamingSAASolveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.toy=build_golden_cases()[0]; toy=self.toy
        first=assemble_first_stage(toy.case,toy.tasks); hashes=tuple(s.scenario_sha256 for s in toy.scenarios)
        self.store=LazyRecourses(toy.scenario_ids,hashes,first.matrix_sha256,{'fixture':'physical'},
            lambda i:assemble_recourse(toy.case,toy.tasks,toy.scenarios[i]),capacity=1)
        self.model=assemble_streaming_saa(first,self.store,scenario_ids=toy.scenario_ids,scenario_sha256s=hashes,
            empirical_probabilities=toy.empirical_probabilities,output_directory=self.root/'model',
            scenario_cost_constants_usd=toy.scenario_cost_constants_usd-100,max_coefficient_bytes=50_000_000)

    def tearDown(self): self.temp.cleanup()

    def run_model(self,**overrides):
        args=dict(case=self.toy.case,tasks=self.toy.tasks,scene_loader=lambda i:self.toy.scenarios[i],
            output_directory=self.root/'solve',backend='highs',time_limit_seconds=5,
            deadline=time.monotonic()+15,evidence_reserve_seconds=5)
        args.update(overrides)
        return solve_streaming_saa(self.model,self.store,**args)

    def test_native_result_original_witnesses_and_constant_shift(self):
        result=self.run_model()
        self.assertEqual(result['status'],'OPTIMAL')
        self.assertAlmostEqual(result['objective_usd'],-69.73333333333333,places=7)
        self.assertAlmostEqual(result['lower_bound_usd'],result['objective_usd'],places=7)
        self.assertTrue(result['incumbent_physical_checks_passed'])
        self.assertEqual(len(result['original_witness_checks']),3)
        self.assertFalse(result['full_support_verified'])
        self.assertTrue(result['full_support_fixed_plan_reoptimization_required'])

    def test_time_limit_without_incumbent_keeps_native_bound(self):
        raw=OptimizeResult(status=1,message='limit',x=None,fun=None,mip_dual_bound=-120.,mip_gap=None,limit_status=None)
        with patch('evcs_experiments.saa_streaming_solve.milp',return_value=raw):
            result=self.run_model()
        self.assertEqual(result['status'],'TIME_LIMIT')
        self.assertIsNone(result['objective_usd']); self.assertIsNone(result['gap'])
        self.assertAlmostEqual(result['lower_bound_usd'],-218.)
        self.assertFalse(result['normal_convergence'])

    def test_changed_source_and_budget_limits_refuse_solver(self):
        with patch('evcs_experiments.saa_streaming_solve.milp') as solve:
            with self.assertRaisesRegex(ValueError,'600'):
                self.run_model(time_limit_seconds=601)
            with self.assertRaisesRegex(ValueError,'reserve'):
                self.run_model(deadline=time.monotonic()+1,evidence_reserve_seconds=2)
            solve.assert_not_called()
        self.model.c[0]+=1; self.model.c.flush()
        with self.assertRaisesRegex(ValueError,'numeric source changed'):
            self.run_model(output_directory=self.root/'changed')

    def test_formal_master_gap_is_separate_from_normal_convergence_gap(self):
        self.model=replace(self.model,manifest={**self.model.manifest,'domain':'formal'})
        with self.assertRaisesRegex(ValueError,'MIPGap'):
            self.run_model(backend='gurobi',optimizer_threads=4,mip_relative_gap=.005)

    def test_missing_explicit_backend_does_not_fallback(self):
        with self.assertRaisesRegex(ValueError,'backend'):
            self.run_model(backend='automatic')


if __name__=='__main__': unittest.main()
