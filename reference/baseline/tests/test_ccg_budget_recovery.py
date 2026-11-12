import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments.ccg import CCGProblem,run_ccg
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.checkpoint_recovery import migrate_budget_guard,OLD_GUARD,NEW_GUARD
from evcs_experiments.random_streams import canonical_hash

class BudgetGuardTests(unittest.TestCase):
    def test_lazy_loading_exhausts_budget_without_zero_time_lp_and_resumes(self):
        f=build_golden_cases()[0];first=assemble_first_stage(f.case,f.tasks)
        templates=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
        clock=[0.];phase={'rmp':False}
        def loader(i):
            if phase['rmp']:clock[0]=6.
            return templates[i]
        bank=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],first.matrix_sha256,{'fixture':f.name},loader,capacity=1)
        p=CCGProblem(first,bank,f.probability_set,f.scenario_cost_constants_usd)
        def sink(event,iteration,**data):
            if event=='rmp':phase['rmp']=True
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            with patch('evcs_experiments.ccg.time.monotonic',side_effect=lambda:clock[0]),patch('evcs_experiments.ccg.solve_fixed_plan',side_effect=AssertionError('LP must not receive zero time')):
                r=run_ccg(p,wall_seconds=5,checkpoint_path=path,evidence_sink=sink,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            self.assertEqual(r.status,'TIME_LIMIT');self.assertEqual(r.state['pending']['q_usd'],[])
            self.assertEqual(r.state['elapsed_seconds'],6.)
            phase['rmp']=False
            resumed=run_ccg(p,wall_seconds=5,checkpoint_path=path,resume=True,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            self.assertEqual(resumed.status,'CONVERGED');self.assertAlmostEqual(resumed.state['upper_bound_usd'],32.12,places=7)
            self.assertGreater(resumed.state['elapsed_seconds'],6.)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        identity={'lazy_input_binding':{'binding':{'source_manifest_sha256':'old'}}}
        plan={'x':[0.],'r':[[0.]]};q=list(map(float,range(26)))
        s={'identity':identity,'problem_sha256':canonical_hash(identity),'engine_sha256':'engine-old','status':'RUNNING','pending':{'plan':plan,'q_usd':q,'recourse_certificates':['immutable']*26},'elapsed_seconds':570.,'sessions':[{'elapsed_seconds':570.}], 'trace':[],'best':None}
        self.envelope={'state':s,'sha256':canonical_hash(s)}
        self.audit={'status':'PASS_RECOVERABLE_PARTIAL_26_LP_NOT_S03','full_S03_pass':False,'q_count':26,'pending_plan_sha256':canonical_hash(plan),'verified_LPs':[{'objective_usd':v} for v in q],'checkpoint_elapsed_seconds':570.,'actual_job_seconds':586.,'engine_sha256':'engine-old','source_manifest_sha256':'old'}
        self.kw={'audit':self.audit,'old_source':OLD_GUARD,'new_source':NEW_GUARD,'new_engine':'engine-new','new_source_manifest_sha256':'new','audit_sha256':'verified-audit'}
    def test_preserves_every_pending_evidence_and_time_without_modifying_old(self):
        before=copy.deepcopy(self.envelope);out=migrate_budget_guard(self.envelope,**self.kw)
        self.assertEqual(self.envelope,before);self.assertEqual(out['state']['pending'],before['state']['pending'])
        self.assertEqual(out['state']['sessions'],before['state']['sessions']);self.assertEqual(out['state']['elapsed_seconds'],570.)
        self.assertEqual(out['state']['checkpoint_migration']['additional_wall_seconds'],16.)
        self.assertEqual(out['sha256'],canonical_hash(out['state']))
    def test_unrelated_code_change_is_rejected(self):
        kw={**self.kw,'new_source':NEW_GUARD+'\n# unrelated edit'}
        with self.assertRaisesRegex(ValueError,'only the exact'):migrate_budget_guard(self.envelope,**kw)
    def test_relabelled_q_or_time_is_rejected(self):
        for field,value in [('checkpoint_elapsed_seconds',0.),('verified_LPs',[{'objective_usd':999.}]*26)]:
            bad={**self.audit,field:value}
            with self.assertRaises(ValueError):migrate_budget_guard(self.envelope,**{**self.kw,'audit':bad})
if __name__=='__main__':unittest.main()
