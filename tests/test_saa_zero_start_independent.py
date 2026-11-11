"""Real development physics check for a complete zero-plan SAA MIP hint.

This validates the projection mathematics, not a formal recovery adapter,
Gurobi execution, native lower bound, or new candidate acceptance.
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.native_gurobi import _check_vector, _model_arrays
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.saa_candidate_warm_start import map_primal
from evcs_experiments.saa_streaming import assemble_streaming_saa
from evcs_experiments.validation.s03_toys import build_golden_cases


class ZeroStartProjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.toy=build_golden_cases()[0]
        self.first=assemble_first_stage(self.toy.case,self.toy.tasks)
        self.recourse=tuple(assemble_recourse(self.toy.case,self.toy.tasks,s) for s in self.toy.scenarios)
        self.model=assemble_streaming_saa(self.first,self.recourse,scenario_ids=self.toy.scenario_ids,
            scenario_sha256s=[s.scenario_sha256 for s in self.toy.scenarios],
            empirical_probabilities=self.toy.empirical_probabilities,
            scenario_cost_constants_usd=np.array([-100.,2.,-3.]),
            output_directory=Path(self.tmp.name)/'model',max_coefficient_bytes=50_000_000)
        self.zero_plan=self.model.plan(np.zeros_like(self.model.c))
        self.arrays=_model_arrays(self.model.c,self.model.integrality,self.model.lower_bounds,
            self.model.upper_bounds,self.model.A,self.model.row_lower,self.model.row_upper)

    def complete_hint(self):
        self.assertTrue(check_first_stage(self.toy.case,self.toy.tasks,self.zero_plan)['passed'])
        vector=np.zeros_like(self.model.c); costs=[]
        for i,(scene,template,row) in enumerate(zip(self.toy.scenarios,self.recourse,self.model.manifest['scenarios'])):
            result=solve_fixed_plan(template,self.zero_plan,time_limit_seconds=3.)
            self.assertTrue(result.optimal); self.assertIsNotNone(result.solution)
            self.assertTrue(check_recourse(self.toy.case,self.toy.tasks,scene,self.zero_plan,result.solution)['passed'])
            raw=template.pack(result.solution)
            mapping=np.load(row['mapping_reference']['path'],allow_pickle=False)
            lo,hi=row['column_slice']
            vector[lo:hi]=map_primal(raw,mapping,row['original_recourse_columns'],hi-lo)
            costs.append(result.objective_usd)
            self.assertTrue(check_recourse(self.toy.case,self.toy.tasks,scene,self.zero_plan,
                template.unpack(self.model.original_recourse_vector(i,vector)))['passed'])
        return vector,np.array(costs)

    def test_zero_first_stage_plus_real_all_scenario_recourse_is_complete_feasible_hint(self):
        vector,costs=self.complete_hint()
        check=_check_vector(*self.arrays,vector)
        self.assertTrue(check['passed'],check)
        self.assertEqual(np.count_nonzero(vector[:len(self.first.c)]),0)
        self.assertGreater(np.count_nonzero(vector[len(self.first.c):]),0)
        expected=self.first.evaluate(self.zero_plan).objective_usd+float(self.toy.empirical_probabilities@(costs+np.array([-100.,2.,-3.])))
        self.assertAlmostEqual(check['objective_without_constant_usd']+self.model.objective_constant_usd,expected,places=7)
        self.assertEqual(self.model.manifest['solver_calls'],0)

    def test_zero_full_vector_or_incomplete_recourse_cannot_be_called_a_feasible_start(self):
        self.assertFalse(_check_vector(*self.arrays,np.zeros_like(self.model.c))['passed'])
        vector,_=self.complete_hint()
        lo,hi=self.model.manifest['scenarios'][0]['column_slice']; vector[lo:hi]=0
        self.assertFalse(_check_vector(*self.arrays,vector)['passed'])


if __name__=='__main__':unittest.main()
