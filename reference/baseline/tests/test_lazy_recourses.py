import unittest,tempfile,subprocess,sys,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.ccg import CCGProblem,run_ccg
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases

class LazyRecourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=build_golden_cases()[0];cls.first=assemble_first_stage(cls.f.case,cls.f.tasks)
        cls.templates=tuple(assemble_recourse(cls.f.case,cls.f.tasks,s) for s in cls.f.scenarios)

    def bank(self,loader=None):
        return LazyRecourses(self.f.scenario_ids,[s.scenario_sha256 for s in self.f.scenarios],
            self.first.matrix_sha256,{'test_source':'immutable-fixture-v1'},loader or self.templates.__getitem__,capacity=1)

    def problem(self,b):return CCGProblem(self.first,b,self.f.probability_set,self.f.scenario_cost_constants_usd)

    def test_identity_does_not_assemble_and_cache_is_bounded(self):
        b=self.bank();p=self.problem(b);identity=p.identity
        self.assertEqual(b.loads,0)
        for i in range(len(b)):self.assertEqual(b[i].template_sha256,self.templates[i].template_sha256)
        self.assertEqual(len(b._cache),1)
        self.assertEqual(p.identity,identity)
        self.assertEqual(b[0].template_sha256,self.templates[0].template_sha256)
        self.assertEqual(b.loads,len(b)+1)

    def test_wrong_scenario_and_changed_reassembly_rejected(self):
        b=self.bank(lambda i:self.templates[(i+1)%len(self.templates)])
        with self.assertRaisesRegex(ValueError,'identity'):b[0]
        b=self.bank();b[0];b[1]
        b._loader=lambda i:replace(self.templates[i],template_sha256='bad')
        with self.assertRaisesRegex(ValueError,'changed'):b[0]

    def test_reordered_axis_rejected_before_any_assembly(self):
        b=self.bank();b.scenario_ids=tuple(reversed(b.scenario_ids))
        with self.assertRaisesRegex(ValueError,'identity'):self.problem(b)
        self.assertEqual(b.loads,0)

    def test_full_ccg_lazy_zero_reduced_and_resume_match_eager(self):
        eager=run_ccg(self.problem(self.templates),wall_seconds=10,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
        b=self.bank();p=self.problem(b)
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json'
            r=run_ccg(p,wall_seconds=10,checkpoint_path=cp,max_iterations=1,zero_presolve=True,
                      mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            elapsed=r.state['elapsed_seconds']
            fresh=self.problem(self.bank())
            r=run_ccg(fresh,wall_seconds=10,checkpoint_path=cp,resume=True,zero_presolve=True,
                      mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            self.assertEqual(r.status,'CONVERGED');self.assertGreater(r.state['elapsed_seconds'],elapsed)
            self.assertAlmostEqual(r.state['upper_bound_usd'],eager.state['upper_bound_usd'],places=7)
            self.assertAlmostEqual(r.state['lower_bound_usd'],eager.state['lower_bound_usd'],places=7)
            np.testing.assert_allclose(r.state['best']['q_usd'],eager.state['best']['q_usd'],atol=1e-8,rtol=0)
            for row in r.state['trace']:
                self.assertLessEqual(row['max_included_residual_usd'],row['separation_tolerance_usd'])
            self.assertLessEqual(len(fresh.recourses._cache),1)

    def test_lazy_reduced_checkpoint_resumes_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json'
            first=run_ccg(self.problem(self.bank()),wall_seconds=10,checkpoint_path=cp,max_iterations=1,
                zero_presolve=True,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            code='''import json,sys
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.ccg import CCGProblem,run_ccg
f=build_golden_cases()[0];first=assemble_first_stage(f.case,f.tasks)
b=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],first.matrix_sha256,
 {'test_source':'immutable-fixture-v1'},lambda i:assemble_recourse(f.case,f.tasks,f.scenarios[i]),capacity=1)
p=CCGProblem(first,b,f.probability_set,f.scenario_cost_constants_usd)
r=run_ccg(p,checkpoint_path=sys.argv[1],resume=True,wall_seconds=10,zero_presolve=True,
 mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
print(json.dumps(r.as_dict()))
'''
            child=subprocess.run([sys.executable,'-c',code,str(cp)],capture_output=True,text=True,check=True,timeout=20)
            value=json.loads(child.stdout)
            self.assertEqual(value['status'],'CONVERGED')
            self.assertAlmostEqual(value['upper_bound_usd'],32.12,places=7)
            self.assertGreater(value['elapsed_seconds'],first.state['elapsed_seconds'])

if __name__=='__main__':unittest.main()
