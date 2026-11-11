import copy,importlib.util,tempfile,time,unittest,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from evcs_experiments.config import get_paths
from evcs_experiments.extensive import assemble_extensive,solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases
from unittest.mock import patch

def module(name,file):
 s=importlib.util.spec_from_file_location(name,get_paths().project_root/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
api=module('case0_partial_candidate_test','scripts/audits/audit_case0_partial_candidate.py')
checker=module('case0_partial_hash_test','scripts/audits/audit_case0_saved_lp_bound.py')
class PartialCandidateTests(unittest.TestCase):
 def setUp(self):
  f=build_golden_cases()[0];self.f=f;first=assemble_first_stage(f.case,f.tasks);self.ts=[assemble_recourse(f.case,f.tasks,s) for s in f.scenarios]
  self.m=assemble_extensive(first,self.ts,risk_model='DRO',scenario_ids=f.scenario_ids,probability_set=f.probability_set);self.h=self.m.as_dict();self.d={k:getattr(self.m,k) for k in ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')}
  sol=solve_extensive(self.m,time_limit_seconds=5).as_dict()['solution'];self.z=np.r_[sol['x'],np.asarray(sol['r']).ravel()]
 def selected(self):return api.extract_scenario(self.m.A,self.d,self.h,0,self.f.probability_set)
 def solved(self,t):
  H=t['Bx_ub'].shape[1];bub=t['b_ub']+t['Bx_ub']@self.z[:H]+t['Br_ub']@self.z[H:];beq=t['b_eq']+t['Bx_eq']@self.z[:H]+t['Br_eq']@self.z[H:]
  r=linprog(t['c'],A_ub=t['A_ub'],b_ub=bub,A_eq=t['A_eq'],b_eq=beq,bounds=np.c_[t['lower_bounds'],t['upper_bounds']],method='highs')
  d={k:getattr(r,n).marginals for k,n in zip(('inequality','equality','lower','upper'),('ineqlin','eqlin','lower','upper'))};raw=dict(solver_status=int(r.status),success=bool(r.success),objective_without_constant_usd=float(r.fun));return r,d,raw
 def test_selected_original_coefficients_and_global_minorant(self):
  t=self.selected();original=self.ts[0]
  for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'):self.assertEqual((t[k]!=getattr(original,k)).nnz,0)
  for k in ('b_ub','b_eq','c','lower_bounds','upper_bounds'):np.testing.assert_array_equal(t[k],getattr(original,k))
  r,d,raw=self.solved(t);cut,g,check=api.audit_selected_lp(t,self.z,r.x,d,raw);self.assertTrue(check['passed']);self.assertFalse(check['full_candidate_verified'] or check['new_upper_bound_verified'])
  self.assertAlmostEqual(cut['intercept_usd']+g@self.z,r.fun)
 def test_actual_original_hash_accepts_model_loader_arrays_only(self):
  from scipy.sparse import save_npz
  from evcs_experiments.campaign_batch import file_reference
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);save_npz(p/'A.npz',self.m.A);np.savez_compressed(p/'v.npz',**self.d);m=api.model_files(file_reference(p/'A.npz'),file_reference(p/'v.npz'))
   checker.check_model_hash(m['matrix'],{k:v for k,v in m.items() if k!='matrix'},self.h)
   bad={k:v for k,v in m.items() if k!='matrix'};bad['c']=bad['c'].copy();bad['c'][0]+=1
   with self.assertRaises(ValueError):checker.check_model_hash(m['matrix'],bad,self.h)
 def test_original_probability_from_risk_rows(self):
  with patch.object(api,'FEATURE_NAMES',self.f.probability_set.feature_ids):P=api.original_probability(self.m.A,self.d,self.h)
  self.assertEqual(P.ambiguity_sha256,self.f.probability_set.ambiguity_sha256)
 def test_false_dual_and_primal_rejected(self):
  t=self.selected();r,d,raw=self.solved(t);d=copy.deepcopy(d);d['lower'][0]+=1
  with self.assertRaisesRegex(ValueError,'dual cut'):api.audit_selected_lp(t,self.z,r.x,d,raw)
  r,d,raw=self.solved(t);v=r.x.copy();v[0]=-100
  with self.assertRaisesRegex(ValueError,'violates'):api.audit_selected_lp(t,self.z,v,d,raw)
 def test_wrong_original_scenario_provenance_rejected(self):
  h=copy.deepcopy(self.h);h['block_map'][1]['source_sha256']='wrong'
  with self.assertRaisesRegex(ValueError,'provenance'):api.extract_scenario(self.m.A,self.d,h,0,self.f.probability_set)
 def test_limit_is_not_fabricated_cut(self):
  t=self.selected();r,d,raw=self.solved(t);raw['solver_status']=1
  with self.assertRaisesRegex(ValueError,'lacks optimal'):api.audit_selected_lp(t,self.z,r.x,d,raw)
if __name__=='__main__':unittest.main()
