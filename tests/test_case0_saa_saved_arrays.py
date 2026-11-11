import copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.case0_saa_saved_arrays import derive_saved_saa_cuts,check_model_hash
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.relaxation_bound import solve_relaxation_bound

def rehash(A,d,h):
    m=dict(risk_model=h['risk_model'],scenario_ids=h['scenario_ids'],scenario_sha256s=h['scenario_sha256s'],template_sha256s=h['recourse_template_sha256s'],first_stage_sha256=h['first_stage_sha256'],ambiguity_sha256=h['ambiguity_sha256'],empirical_probabilities=h['empirical_probabilities'],slices=h['variable_slices'],objective_constant_usd=float(d['objective_constant_usd']))
    x=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode())
    for v in (A.data,A.indices,A.indptr,d['row_lower'],d['row_upper'],d['lower_bounds'],d['upper_bounds'],d['c'],d['integrality'],np.asarray(h['scenario_cost_constants_usd'],float)):
        a=np.asarray(v,dtype='<i8' if np.asarray(v).dtype.kind in 'iu' else '<f8');x.update(str(a.shape).encode());x.update(a.tobytes())
    h['model_sha256']=x.hexdigest()

class SavedArraysTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);p=Path(self.tmp.name)
        A=csr_matrix([[-1.,-1.,0.],[1.,-1.,0.],[-1.,0.,-1.],[1.,0.,-1.]])
        d=dict(c=np.array([.1,.5,.5]),row_lower=np.full(4,-np.inf),row_upper=np.array([-2.,0.,-2.,0.]),lower_bounds=np.zeros(3),upper_bounds=np.array([2.,np.inf,np.inf]),integrality=np.zeros(3,dtype=int),objective_constant_usd=np.array(0.))
        blocks=[dict(scope='first_stage',kind='first_stage',row_start=0,row_stop=0,source_sha256='f')]
        entries=[]
        for i in range(2):
            sid='s'+str(i); th='t'+str(i)
            blocks.extend([dict(scope=sid,kind='recourse_ub',row_start=2*i,row_stop=2*i+2,source_sha256=th),dict(scope=sid,kind='recourse_eq',row_start=2*i+2,row_stop=2*i+2,source_sha256=th)])
            q=p/sid;q.mkdir();(q/'solver.json').write_text(json.dumps(dict(status='OPTIMAL',optimal=True,solver_status=0,objective_usd=2.,certificate=dict(template_sha256=th,dual_inequality_marginals=[-1.,0.],dual_equality_marginals=[],dual_lower_marginals=[0.],dual_upper_marginals=[0.]))));np.savez(q/'primal.npz',fixed=[2.],witness=[2.]);(q/'checks.json').write_text(json.dumps(dict(kkt=dict(passed=True),physical=dict(passed=True))))
            entries.append(dict(index=i,scenario_id=sid,template_sha256=th,files={k:file_reference(q/f) for k,f in [('fixed_solver','solver.json'),('primal','primal.npz'),('checks','checks.json')]}))
        h=dict(risk_model='SAA',shape=list(A.shape),nnz=A.nnz,first_stage_fixed=False,all_original_tasks_retained=True,objective_constant_usd=0.,scenario_cost_constants_usd=[0.,0.],scenario_ids=['s0','s1'],scenario_sha256s=['a','b'],recourse_template_sha256s=['t0','t1'],first_stage_sha256='f',ambiguity_sha256=None,empirical_probabilities=[.5,.5],task_sha256='tasks',config_hash='config',variable_slices=dict(x=[0,1],r=[1,1],first_stage=[0,1],recourses=dict(s0=[1,2],s1=[2,3])),block_map=blocks)
        rehash(A,d,h);plan=dict(x=[0.],r=[],task_sha256='tasks',config_hash='config')
        u=dict(method_id='SAA',model_sha256=h['model_sha256'],scenario_ids=['s0','s1'],plan=plan,plan_sha256=canonical_hash(plan),numerical_valid=True,full_support_independently_evaluated=True,q_usd=[2.,2.],risk_value_usd=2.,Fpre_usd=0.,upper_bound_usd=2.)
        self.A,self.d,self.h,self.u,self.e=A,d,h,u,entries
    def run_model(self):return derive_saved_saa_cuts(self.A,self.d,self.h,self.u,self.e,deadline=time.monotonic()+5)
    def test_cost_unweighting_signed_coupling_and_known_bound(self):
        m,r,c=self.run_model();np.testing.assert_allclose(m['c'],[-.9]);self.assertEqual(m['objective_constant_usd'],2.);self.assertEqual(r['all_saved_Q_rechecked'],2)
        s,_=solve_relaxation_bound(m,time_limit_seconds=2);self.assertAlmostEqual(s['lower_bound_usd'],.2)
        for x in np.linspace(0,2,21):self.assertLessEqual(m['c'][0]*x+2.,.1*x+max(2-x,x)+1e-12)
    def test_nonuniform_law_even_rehashed_is_rejected(self):
        self.h['empirical_probabilities']=[.25,.75];rehash(self.A,self.d,self.h);self.u['model_sha256']=self.h['model_sha256']
        with self.assertRaisesRegex(ValueError,'uniform'):self.run_model()
    def test_wrong_Q_pairing_rejected(self):
        self.e.reverse()
        with self.assertRaisesRegex(ValueError,'pairing'):self.run_model()
    def test_changed_original_array_rejected(self):
        self.d['c'][1]=.6
        with self.assertRaisesRegex(ValueError,'hash'):self.run_model()
    def test_infeasible_saved_primal_rejected_even_with_new_file_reference(self):
        p=Path(self.e[0]['files']['primal']['path']);np.savez(p,fixed=[0.],witness=[2.]);self.e[0]['files']['primal']=file_reference(p)
        with self.assertRaisesRegex(ValueError,'primal feasibility'):self.run_model()
    def test_rehashed_wrong_coupling_rejected_by_primal_or_dual_anchor(self):
        self.A.data[0]=1.;rehash(self.A,self.d,self.h);self.u['model_sha256']=self.h['model_sha256']
        # At x=0 this altered model still has a valid different affine minorant;
        # the signed gradient must change, never retain the old direction.
        m,r,c=self.run_model();self.assertEqual(c[0][1][0],1.);self.assertEqual(c[1][1][0],-1.)
