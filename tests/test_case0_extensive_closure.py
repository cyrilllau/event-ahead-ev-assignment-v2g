import copy,importlib.util,unittest
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.extensive import assemble_extensive,solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.probability import solve_worst_distribution
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.validation.s03_toys import build_golden_cases
sp=importlib.util.spec_from_file_location('test_case0_full_closure',get_paths().project_root/'scripts/audits/audit_case0_extensive_closure.py')
closure=importlib.util.module_from_spec(sp);sp.loader.exec_module(closure)

class ExtensiveClosureTests(unittest.TestCase):
    def setUp(self):
        f=build_golden_cases()[0];self.f=f;first=assemble_first_stage(f.case,f.tasks)
        ts=[assemble_recourse(f.case,f.tasks,s) for s in f.scenarios]
        model=assemble_extensive(first,ts,risk_model='DRO',scenario_ids=f.scenario_ids,probability_set=f.probability_set)
        result=solve_extensive(model,time_limit_seconds=5.)
        solution=result.as_dict()['solution'];p={k:solution[k] for k in ('x','r','task_sha256','config_hash')}
        plan=FirstStagePlan(**p);qs=[solve_fixed_plan(t,plan,time_limit_seconds=5.) for t in ts]
        self.primals={sid:t.pack(q.solution) for sid,t,q in zip(f.scenario_ids,ts,qs)}
        self.P=solve_worst_distribution(f.probability_set,np.asarray([q.objective_usd for q in qs]),q_support_ids=f.scenario_ids).as_dict()
        pre=float(first.c@np.r_[p['x'],np.asarray(p['r']).ravel()])
        self.ub=dict(passed=True,plan=p,plan_sha256=canonical_hash(p),model_sha256=model.model_sha256,scenario_ids=list(f.scenario_ids),Fpre_usd=pre,upper_bound_usd=pre+self.P['dual_objective_usd'])
        self.header=model.as_dict();self.A=model.A
        self.arrays={k:getattr(model,k) for k in ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')}
    def test_complete_original_vector_matches_independent_fullQ_upper_bound(self):
        z,r=closure.check_witness(self.A,self.arrays,self.header,self.ub,self.P,self.primals)
        self.assertTrue(r['passed']);self.assertFalse(r['original_native_primal_recovered'] or r['planning_acceptance'])
        self.assertEqual(z.shape,(self.A.shape[1],));self.assertAlmostEqual(r['objective_usd'],self.ub['upper_bound_usd'])
    def test_wrong_scenario_axis_rejected(self):
        p=copy.deepcopy(self.P);p['support_ids']=list(reversed(p['support_ids']))
        with self.assertRaisesRegex(ValueError,'support'):closure.assemble_witness(self.header,self.ub,p,self.primals)
    def test_invalid_probability_dual_fails_original_risk_rows(self):
        p=copy.deepcopy(self.P);p['theta0']-=100
        with self.assertRaisesRegex(ValueError,'original extensive'):closure.check_witness(self.A,self.arrays,self.header,self.ub,p,self.primals)
    def test_nonintegral_first_stage_is_not_rescued_by_saved_passed(self):
        u=copy.deepcopy(self.ub);u['plan']['x'][0]=.5;u['plan_sha256']=canonical_hash(u['plan'])
        with self.assertRaisesRegex(ValueError,'original extensive'):closure.check_witness(self.A,self.arrays,self.header,u,self.P,self.primals)
    def test_mismatched_original_cost_cannot_relabel_an_upper_bound(self):
        u=copy.deepcopy(self.ub);u['upper_bound_usd']+=100
        with self.assertRaisesRegex(ValueError,'objective'):closure.check_witness(self.A,self.arrays,self.header,u,self.P,self.primals)
