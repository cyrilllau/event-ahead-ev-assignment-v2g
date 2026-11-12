"""Real physical SAA decomposition/extensive goldens and stop/resume boundaries."""
from copy import deepcopy
from dataclasses import replace
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evcs_experiments.saa_l_shaped import DevelopmentSAAProblem,run_saa_l_shaped,build_master,ALGORITHM_ID
from evcs_experiments.extensive import assemble_extensive,solve_extensive
from evcs_experiments.recourse import solve_fixed_plan
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases

def checkpoint_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()

OPTIONS=dict(mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
class SAALShapedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=build_golden_cases();cls.problems=[];cls.results=[];cls.extensive=[];cls.physical=[]
        for f in cls.fixtures:
            p=DevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd);cls.problems.append(p)
            r=run_saa_l_shaped(p,**OPTIONS);cls.results.append(r)
            model=assemble_extensive(p.first_stage,p.recourses,risk_model='SAA',scenario_ids=p.support_ids,
                empirical_probabilities=f.empirical_probabilities,scenario_cost_constants_usd=f.scenario_cost_constants_usd)
            e=solve_extensive(model,time_limit_seconds=5);cls.extensive.append(e)
        dest=os.environ.get('EVCS_SAA_L_SHAPED_TEST_ARTIFACTS_ROOT')
        if dest:
            root=Path(dest);root.mkdir(parents=True,exist_ok=True)
            for f,r,e in zip(cls.fixtures,cls.results,cls.extensive):
                (root/(f.name+'.json')).write_text(json.dumps(dict(decomposition=r,extensive=e.as_dict()),indent=2)+'\n')

    def test_three_real_goldens_match_extensive_and_independent_physics(self):
        for f,p,r,e in zip(self.fixtures,self.problems,self.results,self.extensive):
            with self.subTest(fixture=f.name):
                self.assertEqual(r['status'],'CONVERGED');self.assertTrue(e.optimal)
                self.assertAlmostEqual(r['lower_bound_usd'],e.objective_usd,places=7)
                self.assertAlmostEqual(r['upper_bound_usd'],e.objective_usd,places=7)
                self.assertAlmostEqual(e.objective_usd,f.expected['SAA']['objective_usd'],places=7)
                plan=p.plan(r['best']['anchor']);self.assertTrue(check_first_stage(f.case,f.tasks,plan)['passed'])
                self.assertAlmostEqual(plan.x.sum(),f.expected['SAA']['selected_count'])
                q=[]
                for i,(scene,t) in enumerate(zip(f.scenarios,p.recourses)):
                    solved=solve_fixed_plan(t,plan,time_limit_seconds=5)
                    self.assertTrue(solved.certificate['optimal_certificate_valid'])
                    self.assertTrue(check_recourse(f.case,f.tasks,scene,plan,solved.solution,check_objective_bound=True)['passed'])
                    q.append(solved.objective_usd+p.constants_usd[i])
                np.testing.assert_allclose(q,r['best']['q_usd'],atol=1e-8,rtol=0)
                previous_lb=-np.inf;previous_ub=np.inf
                for row in r['trace']:
                    self.assertGreaterEqual(row['lower_bound_usd'],previous_lb)
                    self.assertLessEqual(row['upper_bound_usd'],previous_ub)
                    self.assertLessEqual(row['lower_bound_usd'],e.objective_usd+1e-7)
                    self.assertGreaterEqual(row['upper_bound_usd'],e.objective_usd-1e-7)
                    m=row['master'];self.assertEqual(m['recourse_columns'],0)
                    self.assertEqual(m['matrix_shape'][1],len(p.first_stage.c)+len(p.support_ids))
                    previous_lb=row['lower_bound_usd'];previous_ub=row['upper_bound_usd']
                self.assertEqual(r['algorithm_id'],ALGORITHM_ID);self.assertFalse(r['formal_admission'])
                self.assertGreater(r['iterations'],2)

    def test_resume_multiple_cuts_repeated_plan_and_no_budget_reset(self):
        p=self.problems[0]
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json';r=run_saa_l_shaped(p,checkpoint_path=cp,max_iterations=1,**OPTIONS)
            previous=r['elapsed_seconds']
            for _ in range(20):
                if r['status']=='CONVERGED':break
                r=run_saa_l_shaped(p,checkpoint_path=cp,resume=True,expected_checkpoint_sha256=checkpoint_hash(cp),max_iterations=1,**OPTIONS)
                self.assertGreaterEqual(r['elapsed_seconds'],previous);previous=r['elapsed_seconds']
            self.assertEqual(r['status'],'CONVERGED');self.assertAlmostEqual(r['upper_bound_usd'],self.results[0]['upper_bound_usd'])
            self.assertAlmostEqual(sum(x['elapsed_seconds'] for x in r['sessions']),r['elapsed_seconds'])
            again=run_saa_l_shaped(p,checkpoint_path=cp,resume=True,expected_checkpoint_sha256=checkpoint_hash(cp),**OPTIONS);self.assertEqual(again,r)
            with self.assertRaisesRegex(ValueError,'budget/config'):run_saa_l_shaped(p,checkpoint_path=cp,resume=True,expected_checkpoint_sha256=checkpoint_hash(cp),budget_seconds=31,**OPTIONS)

    def test_exact_repeated_plan_converges_without_readding_identical_cuts(self):
        from evcs_experiments.validation.s03_toys import _golden
        f=_golden('one_ev_no_use',1,1,(0.,4.),100.,np.empty((2,0)),[],[],(0.,0.),{}, {})
        p=DevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd)
        r=run_saa_l_shaped(p,**OPTIONS)
        self.assertEqual(r['status'],'CONVERGED');self.assertEqual(r['upper_bound_usd'],6.)
        self.assertTrue(r['trace'][-1]['repeated_plan']);self.assertEqual(r['trace'][-1]['added_cuts'],0)

    def test_partial_full_q_resume_preserves_completed_original_lp(self):
        calls=[]
        def stop(t,plan,**kw):
            calls.append(t.scenario_sha256);r=solve_fixed_plan(t,plan,**kw)
            return replace(r,optimal=False,solver_status=1,status='LIMIT_REACHED') if len(calls)==2 else r
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json'
            with patch('evcs_experiments.saa_l_shaped.solve_fixed_plan',side_effect=stop):
                r=run_saa_l_shaped(self.problems[0],checkpoint_path=cp,**OPTIONS)
            self.assertEqual(r['status'],'TIME_LIMIT');self.assertIsNone(r['upper_bound_usd'])
            self.assertEqual(len(r['pending']['atoms']),1)
            with patch('evcs_experiments.saa_l_shaped.solve_fixed_plan',wraps=solve_fixed_plan) as spy:
                resumed=run_saa_l_shaped(self.problems[0],checkpoint_path=cp,resume=True,expected_checkpoint_sha256=checkpoint_hash(cp),max_iterations=1,**OPTIONS)
            self.assertEqual(spy.call_count,2);self.assertEqual(len(resumed['trace']),1)
            self.assertIsNotNone(resumed['upper_bound_usd'])

    def test_exhausted_budget_never_calls_solver_or_renews_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json'
            with patch('evcs_experiments.saa_l_shaped.milp',side_effect=AssertionError('No solver after deadline')):
                r=run_saa_l_shaped(self.problems[0],checkpoint_path=cp,budget_seconds=1e-12)
                self.assertEqual(r['status'],'TIME_LIMIT');self.assertIsNone(r['upper_bound_usd'])
                again=run_saa_l_shaped(self.problems[0],checkpoint_path=cp,resume=True,expected_checkpoint_sha256=checkpoint_hash(cp),budget_seconds=1e-12)
                self.assertEqual(again['status'],'TIME_LIMIT')

    def test_formal_oos_nonuniform_foreign_and_tampered_cut_rejected(self):
        f=self.fixtures[0]
        for domain in ('formal','oos'):
            with self.assertRaises(ValueError):DevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd,domain)
        scene=deepcopy(f.scenarios[0]);scene.development=False
        with self.assertRaises(ValueError):DevelopmentSAAProblem(f.case,f.tasks,(scene,*f.scenarios[1:]),f.scenario_cost_constants_usd)
        with self.assertRaises(TypeError):run_saa_l_shaped(self.problems[0],empirical_probabilities=[.1,.2,.7])
        cuts=deepcopy(self.results[0]['cuts']);cuts[0][0]['gradient'][0]+=1
        with self.assertRaises(ValueError):build_master(self.problems[0],cuts)
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json';run_saa_l_shaped(self.problems[0],checkpoint_path=cp,max_iterations=1,**OPTIONS)
            envelope=json.loads(cp.read_text());envelope['state']['cuts'][0][0]['primal'][0]+=1;envelope['sha256']=canonical_hash(envelope['state']);cp.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):run_saa_l_shaped(self.problems[0],checkpoint_path=cp,resume=True,expected_checkpoint_sha256=checkpoint_hash(cp),**OPTIONS)

    def test_negative_constants_shift_both_bounds_once(self):
        f=self.fixtures[0];p=DevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd-100)
        r=run_saa_l_shaped(p,**OPTIONS)
        self.assertEqual(r['status'],'CONVERGED')
        self.assertAlmostEqual(r['upper_bound_usd'],self.results[0]['upper_bound_usd']-100,places=7)
        self.assertAlmostEqual(r['lower_bound_usd'],self.results[0]['lower_bound_usd']-100,places=7)

    def test_failed_original_lp_does_not_supply_zero_or_partial_upper_bound(self):
        def fail(t,plan,**kw):return replace(solve_fixed_plan(t,plan,**kw),optimal=False,solver_status=4,status='SOLVER_ERROR',objective_usd=None)
        with patch('evcs_experiments.saa_l_shaped.solve_fixed_plan',side_effect=fail):r=run_saa_l_shaped(self.problems[0],**OPTIONS)
        self.assertEqual(r['status'],'RECOURSE_FAILED');self.assertIsNone(r['upper_bound_usd']);self.assertEqual(r['trace'],[])
