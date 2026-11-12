"""Genuine tiny saved recourse proves the auditor checks numbers, not hashes."""
import copy,importlib.util,json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.ccg import CCGProblem
from evcs_experiments.first_stage import assemble_first_stage,FirstStagePlan
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases

ROOT=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module('candidate_recourse_audit',ROOT/'scripts/audits/audit_s05_candidate_recourse.py')
h=module('candidate_prior_numerical_helper',ROOT/'scripts/audits/audit_s05_pending_recovery.py')

class CandidateRecourseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name);f=build_golden_cases()[0];cls.f=f
        cls.first=assemble_first_stage(f.case,f.tasks);t=assemble_recourse(f.case,f.tasks,f.scenarios[0]);cls.t=t
        cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],cls.first.matrix_sha256,{'fixture':'candidate_audit'},lambda i:t)
        problem=CCGProblem(cls.first,cache,f.probability_set,f.scenario_cost_constants_usd,'DRO')
        e=RawPlanningEvidence(root/'raw',f.case,f.tasks,problem,f.scenarios.__getitem__,domain='development',input_binding={'fixture':'candidate_audit'})
        plan=FirstStagePlan(np.zeros(len(f.tasks.task_ids)),np.zeros((len(f.tasks.task_ids),len(f.case.periods))),f.tasks.task_sha256,f.tasks.config_hash)
        result=solve_fixed_plan(t,plan,time_limit_seconds=5);assert result.optimal
        e.sink_factory(root/'call')('recourse',1,index=0,plan=plan,result=result)
        er=e.events[0];ed=Path(er['path']).parent
        cls.saved=dict(event=h.read(er['path']),plan=h.read(ed/'plan.json'),solver=h.read(ed/'solver.json'),solution=h.npz(ed/'solution.npz'),vector=h.npz(ed/'primal_vector.npz')['vector'],dual=h.npz(ed/'dual_kkt.npz'),fixed=h.npz(ed/'fixed_rhs.npz'),first_stage_sha256=cls.first.matrix_sha256)
        cls.kw=dict(index=0,scenario_id=f.scenario_ids[0],plan_dict=cls.saved['plan'],constant=float(f.scenario_cost_constants_usd[0]),expected_template_sha256=t.template_sha256,meta=t.as_dict(),
            matrices={k:getattr(t,k) for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')},arrays={k:getattr(t,k) for k in ('b_ub','b_eq','c','lower_bounds','upper_bounds','component_objectives')},case=f.case,tasks=f.tasks,scene=f.scenarios[0])
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_saved(self,saved=None,**kw):return d.audit_scenario(h,**dict(self.kw,**kw),saved=copy.deepcopy(self.saved) if saved is None else saved)
    def test_genuine_raw_physical_KKT_and_scenario_constant(self):
        r=self.run_saved();self.assertTrue(r['physical_passed']);self.assertLessEqual(r['stationarity'],1e-8)
        self.assertAlmostEqual(r['q_usd'],25.);self.assertAlmostEqual(r['primal_objective_usd'],24.)
    def test_new_fixed_rhs_change_rejected(self):
        s=copy.deepcopy(self.saved);s['fixed']['b_ub'][0]+=1.
        with self.assertRaisesRegex(ValueError,'RHS'):self.run_saved(s)
    def test_mutated_dual_even_with_matching_solver_copy_rejected(self):
        s=copy.deepcopy(self.saved);s['dual']['lower'][0]+=.01;s['solver']['certificate']['dual_lower_marginals']=s['dual']['lower'].tolist()
        with self.assertRaisesRegex(ValueError,'KKT'):self.run_saved(s)
    def test_physical_witness_and_vector_tamper_rejected(self):
        s=copy.deepcopy(self.saved);s['solution']['shed'][:]=0.;s['vector']=np.concatenate([np.asarray(s['solution'][k]).ravel() for k in h.ORDER])
        with self.assertRaisesRegex(ValueError,'physical'):self.run_saved(s)
    def test_scalar_q_constant_or_scenario_relabel_rejected(self):
        s=copy.deepcopy(self.saved);s['event']['q_usd']+=1.
        with self.assertRaisesRegex(ValueError,'constant or q'):self.run_saved(s)
        with self.assertRaisesRegex(ValueError,'scenario/plan'):self.run_saved(scenario_id='other')
    def test_incomplete_solver_cannot_pass_via_valid_raw_label(self):
        s=copy.deepcopy(self.saved);s['solver']['optimal']=False;s['solver']['status']='TIME_LIMIT'
        with self.assertRaisesRegex(ValueError,'OPTIMAL'):self.run_saved(s)
    def test_cost_aggregation_checked_independently_from_template_digest(self):
        arrays=dict(self.kw['arrays']);arrays['c']=arrays['c'].copy();arrays['c'][0]+=1.
        with self.assertRaises(ValueError):self.run_saved(arrays=arrays)
    def test_first_stage_digest_binds_cost_and_integrality(self):
        root=Path(self.temp.name)/'raw/first_stage';meta=h.read(root/'identity.json');v=h.npz(root/'arrays.npz');A=self.first.A
        self.assertEqual(d.first_digest(meta,A,v),self.first.matrix_sha256)
        v['c'][0]+=1.;self.assertNotEqual(d.first_digest(meta,A,v),self.first.matrix_sha256)
    def test_full160_completion_axis_cannot_hide_missing_or_duplicate_events(self):
        identity={'scenario_ids':[str(i) for i in range(160)]};ph='plan';events=[{'path':str(i)} for i in range(160)]
        result=dict(status='FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY',completed_count=160,full_q_available=True,probability_evaluated=False,formal_OOS=False,plan_sha256=ph,problem_sha256=canonical_hash(identity))
        cp=dict(status=result['status'],completed_count=160,next_index=160,support_ids=identity['scenario_ids'],q_usd=[1.]*160,plan_sha256=ph,formal_OOS=False,recourse_event_references=events)
        snapshot=dict(events=events,templates=[]);d.validate_completion(result,cp,snapshot,identity)
        cp['q_usd'][159]=None
        with self.assertRaisesRegex(ValueError,'finite q'):d.validate_completion(result,cp,snapshot,identity)
        cp['q_usd']=[1.]*160;cp['support_ids']=cp['support_ids'][::-1]
        with self.assertRaisesRegex(ValueError,'ordered'):d.validate_completion(result,cp,snapshot,identity)
        cp['support_ids']=identity['scenario_ids'];events[1]=events[0]
        with self.assertRaisesRegex(ValueError,'coverage'):d.validate_completion(result,cp,snapshot,identity)

if __name__=='__main__':unittest.main()
