import ast,hashlib,importlib.util,json,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from scipy.optimize import milp,Bounds,LinearConstraint
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments import case0_partial_chain as helper
from tests import test_case0_partial_chain as fixtures

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
root=get_paths().project_root
d=module('partial_master_test_driver',root/'scripts/audits/audit_case0_partial_chain_master.py')
a=module('partial_native_test_checker',root/'scripts/audits/audit_case0_partial_chain_native.py')

class PartialChainMasterTests(unittest.TestCase):
 def setUp(self):
  f=fixtures.ChainTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f;self.root=f.root
  def write(n,v):p=f.root/n;p.write_text(json.dumps(v));return file_reference(p)
  def sm(prefix,m):
   p=f.root/(prefix+'matrix.npz');save_npz(p,m['matrix']);q=f.root/(prefix+'arrays.npz');np.savez_compressed(q,**{k:v for k,v in m.items() if k!='matrix'});return file_reference(p),file_reference(q)
  sel=f.select();cut,g=f.cut(0,f.v[:2],[0.,-1.],.5);m,check,ex=f.append();self.model=m
  refs={};refs['base_matrix'],refs['base_arrays']=sm('base_',f.original);refs['chain_matrix'],refs['chain_arrays']=sm('chain_',f.model);refs['matrix'],refs['arrays']=sm('augmented_',m)
  refs['base_audit']=write('base_audit.json',f.oa);refs['chain_audit']=write('chain_audit.json',f.audit)
  refs['retained_cuts']=write('retained_cuts.json',f.retained);refs['cut_sets']=write('base_cuts.json',[{i:[c,g.tolist()] for i,(c,g) in grp.items()} for grp in f.cuts]);refs['selection']=write('selection.json',sel);refs['cut']=write('new_cut.json',cut)
  refs['partial_audit']=write('partial_model_audit.json',check);refs['exclusion']=write('incumbent_exclusion.json',ex)
  refs['probability_set']=write('probability_set.json',f.P.as_dict())
  for k,n,v in [('gradient','new_gradient.npy',g),('chain_vector','chain_vector.npy',f.v)]:p=f.root/n;np.save(p,v);refs[k]=file_reference(p)
  refs['old_ub']=write('old_ub.json',dict(upper_bound_usd=1.,plan_sha256='old'))
  q=dict(kind='CASE0_PARTIAL_CHAIN_CUT_RESULT_V1',passed=True,new_Q_solve_count=1,master_solve_count=0,planning_acceptance=False,full_candidate_verified=False,
    incumbent_exclusion_reference=refs['exclusion'],selection_reference=refs['selection'],cut_reference=refs['cut'],partial_model_audit_reference=refs['partial_audit'],
    source_binding={k:refs[k] for k in ('base_matrix','base_arrays','base_audit','chain_matrix','chain_arrays','chain_audit','chain_vector','old_ub')},upper_bound_usd=1.,old_plan_sha256='old',lower_bound_usd=0.,model_sha256='original',selected_index=0,scenario_id='s0')
  refs['q_result']=write('result.json',q);self.refs=refs;self.out=f.root/'second';self.out.mkdir()
 def fake(self,*args,**kw):
  m=self.model;r=milp(m['c'],integrality=m['integrality'],bounds=Bounds(m['lower_bounds'],m['upper_bounds']),constraints=LinearConstraint(m['matrix'],m['row_lower'],m['row_upper']))
  rl,ru=m['row_lower'],m['row_upper'];eq=np.isfinite(rl)&np.isfinite(ru)&(rl==ru)
  maps=dict(eq=np.flatnonzero(eq),le=np.flatnonzero(np.isfinite(ru)&~eq),ge=np.flatnonzero(np.isfinite(rl)&~eq));rows={k:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for k,v in maps.items()};rows.update(original_rows=len(rl),translated_rows=sum(map(len,maps.values())))
  nr=dict(backend='gurobipy',gurobi_version=[12,0,3],original_matrix_shape=list(m['matrix'].shape),original_matrix_nnz=m['matrix'].nnz,effective_options=dict(Threads=1,MIPGap=.005,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-8,IntFeasTol=1e-8),requested_time_limit_seconds=kw['options']['time_limit'],raw_model_status=2,status='OPTIMAL',certificate_failures=[],is_mip=True,bound_source='MIP_ObjBound',solver_bound_consistent=True,raw_solver_bound_without_constant_usd=float(r.fun),solver_bound_without_constant_usd=float(r.fun),row_map_summary=rows,model_rows=rows['translated_rows'],model_columns=m['matrix'].shape[1],valid_incumbent=True,objective_matches=True,solution_count=1,incumbent_without_constant_usd=float(r.fun))
  return SimpleNamespace(x=r.x,status=0,mip_dual_bound=float(r.fun),native_record=nr)
 def run_it(self):return d.run_numeric(self.refs,self.out,time.monotonic()+12,6,helper,a)
 def test_real_partial_model_and_native_audit_no_q(self):
  with patch.object(d,'solve_milp',side_effect=self.fake) as call:r=self.run_it()
  self.assertEqual(call.call_count,1);self.assertAlmostEqual(r['lower_bound_usd'],.1);self.assertEqual(r['upper_bound_usd'],1.);self.assertFalse(r['planning_acceptance']);self.assertEqual(r['new_Q_solve_count'],0)
 def test_saturated_or_wrong_child_never_calls_master(self):
  p=Path(self.refs['exclusion']['path']);old=json.loads(p.read_text());old['master_call_required']=False;p.write_text(json.dumps(old));self.refs['exclusion']=file_reference(p)
  with patch.object(d,'solve_milp',side_effect=AssertionError('No redundant solve')),self.assertRaises(ValueError):self.run_it()
 def test_original_prefix_tamper_rejected(self):
  p=Path(self.refs['arrays']['path']);m=self.model.copy();m['c']=m['c'].copy();m['c'][0]+=1.;np.savez_compressed(p,**{k:v for k,v in m.items() if k!='matrix'});self.refs['arrays']=file_reference(p)
  with patch.object(d,'solve_milp',side_effect=AssertionError('No bad model')),self.assertRaises(ValueError):self.run_it()
 def test_native_bound_tamper_rejected(self):
  def bad(*args,**kw):r=self.fake(*args,**kw);r.mip_dual_bound+=1;return r
  with patch.object(d,'solve_milp',side_effect=bad),self.assertRaises(ValueError):self.run_it()

class NativeCopyTests(unittest.TestCase):
 def test_exact_three_token_native_adapter_delta(self):
  old=(root/'scripts/audits/audit_case0_partial_native.py').read_text()
  expected=old.replace('audit_partial_native_bound','audit_partial_chain_native_bound').replace('INDEPENDENT_CASE0_PARTIAL_SCENARIO_MODEL_AUDIT_V1','INDEPENDENT_CASE0_PARTIAL_CHAIN_MODEL_AUDIT_V1').replace('INDEPENDENT_CASE0_PARTIAL_NATIVE_AUXILIARY_BOUND_AUDIT_V1','INDEPENDENT_CASE0_PARTIAL_CHAIN_NATIVE_AUXILIARY_BOUND_AUDIT_V1')
  self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((root/'scripts/audits/audit_case0_partial_chain_native.py').read_text())))
