"""Independent checks of retained physical evidence and resume accounting."""
from copy import deepcopy
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases

class IndependentSAAResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f=build_golden_cases()[0]
        cls.problem=kernel.DevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd)
        cls.options=dict(mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
        cls.result=kernel.run_saa_l_shaped(cls.problem,**cls.options)
        assert cls.result['status']=='CONVERGED'

    def reject(self,mutate,regex=None):
        value=deepcopy(self.result);mutate(value)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resealed.json'
            path.write_text(json.dumps(dict(state=value,sha256=canonical_hash(value))))
            # A reused CONVERGED state must be validated before trusting its tag.
            with patch.object(kernel,'_master',side_effect=AssertionError('Invalid resume must not solve')):
                with self.assertRaises(ValueError):
                    kernel.run_saa_l_shaped(self.problem,checkpoint_path=path,resume=True,expected_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**self.options)

    def test_resealed_best_summary_cannot_override_full_physical_sweep(self):
        self.reject(lambda s:s.update(upper_bound_usd=s['upper_bound_usd']-10))
        self.reject(lambda s:s['best'].update(plan_sha256='0'*64))
        self.reject(lambda s:s['best']['atoms'].pop())
        # Every swapped atom is itself valid; its scenario/anchor must still match.
        self.reject(lambda s:s['best']['atoms'].__setitem__(0,deepcopy(s['cuts'][1][0])))
        self.reject(lambda s:s['best'].update(anchor=[0.0]*len(s['best']['anchor'])))

    def test_resealed_bound_and_session_summaries_cannot_reset_evidence(self):
        self.reject(lambda s:s.update(lower_bound_usd=s['lower_bound_usd']+1))
        self.reject(lambda s:s.update(elapsed_seconds=-1.0))
        self.reject(lambda s:s.update(elapsed_seconds=0.0))
        self.reject(lambda s:s['sessions'][0].update(elapsed_seconds=-1.0))
        self.reject(lambda s:s['cuts'].pop())
        self.reject(lambda s:s.update(iterations=s['iterations']+1))

    def test_all_retained_affine_planes_bound_independent_physical_primal_points(self):
        p=self.problem;s=self.result
        # Audit every cut at every fully evaluated plan, not just its own anchor.
        points=[]
        for row in s['trace']:
            vec=np.asarray(row['master']['primal'][:len(p.first_stage.c)])
            points.append((vec,row['q_usd']))
        for i,atoms in enumerate(s['cuts']):
            for atom in atoms:
                g=np.asarray(atom['gradient']);intercept=atom['cut']['intercept_usd']
                for vec,q in points:
                    self.assertLessEqual(intercept+float(g@vec),q[i]+1e-7)
        # Uniform expectation includes each constant and first-stage cost exactly once.
        best=s['best'];vec=np.asarray(best['anchor']);expected=float(p.first_stage.c@vec)+sum(a['q_usd'] for a in best['atoms'])/len(p.support_ids)
        self.assertAlmostEqual(s['upper_bound_usd'],expected,places=9)
        self.assertFalse(s['formal_admission']);self.assertFalse(s['formal_OOS'])

    def test_joint_native_history_rewrite_requires_external_checkpoint_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            s=kernel.run_saa_l_shaped(self.problem,checkpoint_path=path,max_iterations=1,**self.options)
            self.assertEqual(s['status'],'ITERATION_LIMIT')
            trusted=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                kernel.run_saa_l_shaped(self.problem,checkpoint_path=path,resume=True,**self.options)
            s['status']='CONVERGED';s['lower_bound_usd']=s['upper_bound_usd']
            for master in s['master_history']:master['lower_bound_usd']=s['upper_bound_usd']
            for row in s['trace']:row['master']['lower_bound_usd']=s['upper_bound_usd']
            path.write_text(json.dumps(dict(state=s,sha256=canonical_hash(s))))
            with patch.object(kernel,'_master',side_effect=AssertionError('Unpinned input must not solve')):
                with self.assertRaises(ValueError):
                    kernel.run_saa_l_shaped(self.problem,checkpoint_path=path,resume=True,expected_checkpoint_sha256=trusted,**self.options)
