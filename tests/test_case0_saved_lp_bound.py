"""Independent dual examples and real saved extensive-array relaxation checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,vstack
from evcs_experiments.config import get_paths
from evcs_experiments.case0_planning import run_case0_planning
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.random_streams import canonical_hash

path=get_paths().project_root/'scripts/audits/audit_case0_saved_lp_bound.py'
spec=importlib.util.spec_from_file_location('case0_saved_lp_bound_test_module',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def scipy_witness(matrix,arrays,output,deadline):
    m=audit._row_map(arrays['row_lower'],arrays['row_upper'])
    a=vstack([matrix[m['le']],-matrix[m['ge']]],format='csr')
    rhs=np.r_[arrays['row_upper'][m['le']],-arrays['row_lower'][m['ge']]]
    r=linprog(arrays['c'],A_ub=a,b_ub=rhs,A_eq=matrix[m['eq']],b_eq=arrays['row_lower'][m['eq']],
        bounds=np.c_[arrays['lower_bounds'],arrays['upper_bounds']],method='highs-ds',
        options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    if not r.success:raise AssertionError(r.message)
    k=len(m['le']);w=dict(primal=r.x,reduced_cost=r.lower.marginals+r.upper.marginals,
        eq=r.eqlin.marginals,le=r.ineqlin.marginals[:k],ge=-r.ineqlin.marginals[k:])
    np.savez_compressed(Path(output)/'lp_witness.npz',**w)
    return dict(status_code=2),w


class DualCertificateTests(unittest.TestCase):
    def setUp(self):
        # Free x fixed by equality, bounded y lower-active, z upper-active.
        self.a=csr_matrix([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
        self.c=np.array([2.,3.,-4.]);self.lo=np.array([-np.inf,1.,0.]);self.hi=np.array([np.inf,5.,2.])
        self.rl=np.array([1.,0.,-np.inf,-np.inf]);self.ru=np.array([1.,4.,3.,np.inf])
        self.w=dict(primal=np.array([1.,1.,2.]),reduced_cost=np.array([0.,3.,-4.]),eq=np.array([2.]),le=np.array([0.,0.]),ge=np.array([0.]))
    def check(self,w=None):return audit.check_dual(self.a,self.c,self.lo,self.hi,self.rl,self.ru,self.w if w is None else w,7.)
    def test_ranged_free_columns_and_nonzero_constant(self):
        r=self.check();self.assertTrue(r['passed']);self.assertEqual(r['dual_objective_usd'],4.)
        self.assertEqual(r['primal_objective_usd'],4.)
    def test_wrong_row_sign_or_free_variable_reduced_cost_rejected(self):
        for key,index,value in [('le',0,.1),('reduced_cost',0,.1)]:
            w=copy.deepcopy(self.w);w[key][index]=value
            with self.assertRaisesRegex(ValueError,'dual signs/stationarity'):self.check(w)
    def test_stationarity_and_fake_objective_cannot_replace_dual(self):
        w=copy.deepcopy(self.w);w['eq'][0]=2.01
        with self.assertRaisesRegex(ValueError,'stationarity'):self.check(w)
        w=copy.deepcopy(self.w);w['primal'][1]=2.
        r=self.check(w);self.assertTrue(r['passed']);self.assertFalse(r['strong_duality_passed'])
    def test_primal_axes_and_original_unit_threshold_enforced(self):
        w=copy.deepcopy(self.w);w['primal'][0]+=1e-6
        r=self.check(w);self.assertTrue(r['passed']);self.assertFalse(r['primal_feasible'])
        w=copy.deepcopy(self.w);w['le']=np.zeros(1)
        with self.assertRaisesRegex(ValueError,'row axes'):self.check(w)


class SavedLPBoundTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        f=build_golden_cases()[1]
        for scene in f.scenarios:scene.split='planning'
        first=assemble_first_stage(f.case,f.tasks);rec=[assemble_recourse(f.case,f.tasks,s) for s in f.scenarios]
        self.solve=self.root/'producer';now=time.monotonic()
        _,r=run_case0_planning(first,rec,f.scenario_ids,f.probability_set,case=f.case,tasks=f.tasks,scenes=f.scenarios,
            method_id='EA_DRO',output_dir=self.solve,backend='highs',deadline=now+1800,configuration_started_at=now,
            master_seconds=5.,zero_presolve=True)
        self.ub=r['upper_bound_usd'];p=self.solve/'master.json';m=json.loads(p.read_text())
        m.update(status='TIME_LIMIT',optimal=False,lower_bound_usd=None,mip_gap=None);p.write_text(json.dumps(m))
        self.refs={k:audit.file_reference(self.solve/v) for k,v in [('master','master.json'),('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]}
    def run_numeric(self,native=scipy_witness):
        out=self.root/'audit';out.mkdir()
        with patch.object(audit,'native_lp',side_effect=native),\
             patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No MIP or Q rebuild')):
            return audit.numeric_diagnostic(references=self.refs,output=out,master_deadline=time.monotonic()+10.,upper_bound=self.ub)
    def test_real_saved_model_zero_proof_lp_dual_and_original_lift(self):
        r=self.run_numeric();self.assertTrue(r['bound_available']);self.assertLessEqual(r['lower_bound_usd'],self.ub+1e-8)
        self.assertFalse(r['planning_acceptance'] or r['formal_OOS'] or r['gap_passed']);self.assertTrue(r['numerical_auxiliary_only']);self.assertGreater(r['exact_zero_proof']['zero_columns'],0)
        d=audit.read(r['dual_check_reference']);self.assertTrue(d['original_relaxation_check']['passed'])
        self.assertTrue(d['zero_lift_exact'])
    def test_no_pi_is_no_bound_not_native_reported_objective(self):
        r=self.run_numeric(lambda *a:({'status_code':9,'objective':self.ub},None))
        self.assertEqual(r['status'],'NO_VALID_BOUND');self.assertIsNone(r['lower_bound_usd']);self.assertIsNone(r['gap'])
    def test_feasible_dual_without_primal_still_supplies_auxiliary_bound(self):
        def dual_only(*args):
            _,w=scipy_witness(*args);w.pop('primal');return dict(status_code=9),w
        r=self.run_numeric(dual_only);self.assertTrue(r['bound_available']);self.assertFalse(r['gap_passed'])
        d=audit.read(r['dual_check_reference']);self.assertFalse(d['primal_available']);self.assertFalse(d['strong_duality_passed'])
        self.assertFalse(d['original_relaxation_check']['passed'])
    def test_dual_feasible_infeasible_primal_retains_bound(self):
        def infeasible(*args):
            _,w=scipy_witness(*args);w['primal']+=100.;return dict(status_code=9),w
        r=self.run_numeric(infeasible);self.assertTrue(r['bound_available']);d=audit.read(r['dual_check_reference'])
        self.assertFalse(d['primal_feasible']);self.assertFalse(d['original_relaxation_check']['passed'])
        self.assertFalse(r['gap_passed'] or r['planning_acceptance'])
    def test_resealed_wrong_original_matrix_fails_before_solver(self):
        from scipy.sparse import load_npz,save_npz
        p=self.solve/'matrix.npz';m=load_npz(p);m.data[0]*=2.;save_npz(p,m);self.refs['matrix']=audit.file_reference(p)
        with self.assertRaisesRegex(ValueError,'semantic/array hash'):self.run_numeric(lambda *a:(_ for _ in ()).throw(AssertionError('No LP')))
    def test_resealed_zero_proof_nonredundant_row_deletion_fails(self):
        p=self.solve/'master.json';m=json.loads(p.read_text());proof=m['raw_solver']['zero_reduction']
        proof['retained_rows']=proof['retained_rows'][1:];proof.pop('sha256');proof['sha256']=canonical_hash(proof)
        p.write_text(json.dumps(m));self.refs['master']=audit.file_reference(p)
        with self.assertRaisesRegex(ValueError,'nonredundant'):self.run_numeric()
    def test_stopped_lp_bad_dual_persists_diagnosis_without_bound(self):
        def wrong(*args):
            n,w=scipy_witness(*args);w['reduced_cost'][0]+=1.;return dict(status_code=9),w
        r=self.run_numeric(wrong);self.assertFalse(r['bound_available']);self.assertIn('stationarity',r['certificate_rejection'])


if __name__=='__main__':unittest.main()

class NativeLPTranslationTests(unittest.TestCase):
    def test_actual_builder_relaxes_integrality_and_preserves_ranged_rows(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        class GPError(Exception):pass
        gp=SimpleNamespace(GurobiError=GPError,gurobi=SimpleNamespace(version=lambda:(12,0,3)),
            GRB=SimpleNamespace(MINIMIZE=1,CONTINUOUS='C',EQUAL='=',LESS_EQUAL='<',GREATER_EQUAL='>'))
        model=MagicMock();model.IsMIP=0;model.Status=2;model.Runtime=.01;model.SolCount=1
        model.addMVar.return_value=SimpleNamespace(RC=np.array([0.]))
        model.addMConstr.side_effect=lambda a,v,s,rhs,name:SimpleNamespace(Pi=np.zeros(len(rhs)))
        gp.Env=MagicMock(return_value=MagicMock());gp.Model=MagicMock(return_value=model)
        arrays=dict(c=np.ones(1),integrality=np.ones(1,dtype=int),lower_bounds=np.zeros(1),upper_bounds=np.ones(1)*2,
            row_lower=np.array([1.,-np.inf,0.,-np.inf]),row_upper=np.array([1.,3.,2.,np.inf]))
        matrix=csr_matrix([[1.],[2.],[3.],[4.]])
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'gurobipy':gp}):
            r,w=audit.native_lp(matrix,arrays,tmp,time.monotonic()+40.)
            saved=json.loads((Path(tmp)/'native_started.json').read_text())
        self.assertFalse(r['is_mip']);self.assertNotIn('primal',w);self.assertIn('primal_unavailable',r);self.assertEqual(saved['original_integer_columns'],1)
        self.assertEqual(saved['native_integer_columns'],0)
        self.assertEqual(model.addMVar.call_args.kwargs['vtype'],'C')
        calls=model.addMConstr.call_args_list
        self.assertEqual([x.kwargs['name'] for x in calls],['eq','le','ge'])
        np.testing.assert_array_equal(calls[1].args[0].toarray(),[[2.],[3.]])
        np.testing.assert_array_equal(calls[2].args[0].toarray(),[[3.]])
        params=dict(x.args for x in model.setParam.call_args_list)
        self.assertEqual((params['PreDual'],params['Method'],params['Threads']),(0,1,1))
        self.assertEqual(params['FeasibilityTol'],1e-9);self.assertEqual(params['OptimalityTol'],1e-8)
        self.assertLess(params['TimeLimit'],20.);model.dispose.assert_called_once()
