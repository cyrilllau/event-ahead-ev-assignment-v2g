import hashlib,importlib.util,json,time,unittest,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from scipy.optimize import milp,Bounds,LinearConstraint
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments import saa_saved_dual_adapter as adapter,saa_dual_bound as combiner
from tests import test_saa_saved_dual_adapter as fixture

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
R=get_paths().project_root
d=load('test_saved_saa_controller',R/'scripts/diagnostics/s05_saa_saved_dual_bound.py')
a=load('test_saved_saa_native',R/'scripts/audits/audit_saa_affine_native.py')

class SAABoundControllerTests(unittest.TestCase):
 def setUp(self):
  self.f=fixture.SavedSAADualTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.f.root
  fs=self.f.f.ts[0].first_stage;folder=self.root/'first';folder.mkdir();save_npz(folder/'matrix.npz',fs.A)
  np.savez_compressed(folder/'arrays.npz',**{k:getattr(fs,k) for k in ('c','integrality','row_lower','row_upper','lower_bounds','upper_bounds')})
  (folder/'identity.json').write_text(json.dumps(fs.as_dict()));self.write(folder/'record.json',dict(kind='first_stage',artifacts=[file_reference(p) for p in sorted(folder.iterdir())]))
  k=self.f.kw;refs=dict(candidate_audit=k['candidate_audit_reference'],fixed_q_request=k['request_reference'],fixed_q_checkpoint=k['checkpoint_reference'],template_envelope=k['template_envelope_reference'],first_stage=file_reference(folder/'record.json'))
  self.req=dict(references=refs,solver_seconds=2.,post_master_reserve_seconds=.1);self.out=self.root/'bound';self.out.mkdir()
 def write(self,p,v):p.write_text(json.dumps(v));return file_reference(p)
 def fake(self,c,integ,lo,hi,A,rl,ru,**kw):
  r=milp(c,integrality=integ,bounds=Bounds(lo,hi),constraints=LinearConstraint(A,rl,ru));eq=np.isfinite(rl)&np.isfinite(ru)&(rl==ru)
  maps=dict(eq=np.flatnonzero(eq),le=np.flatnonzero(np.isfinite(ru)&~eq),ge=np.flatnonzero(np.isfinite(rl)&~eq));rows={k:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for k,v in maps.items()};rows.update(original_rows=len(rl),translated_rows=sum(map(len,maps.values())))
  n=dict(backend='gurobipy',gurobi_version=[12,0,3],original_matrix_shape=list(A.shape),original_matrix_nnz=A.nnz,effective_options=dict(Threads=1,MIPGap=.005,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-8,IntFeasTol=1e-8),requested_time_limit_seconds=kw['options']['time_limit'],raw_model_status=2,status='OPTIMAL',certificate_failures=[],is_mip=True,bound_source='MIP_ObjBound',solver_bound_consistent=True,raw_solver_bound_without_constant_usd=float(r.fun),solver_bound_without_constant_usd=float(r.fun),row_map_summary=rows,model_rows=rows['translated_rows'],model_columns=A.shape[1],valid_incumbent=True,objective_matches=True,solution_count=1,incumbent_without_constant_usd=float(r.fun))
  return SimpleNamespace(x=r.x,status=0,mip_dual_bound=float(r.fun),native_record=n)
 def run_it(self):return d.run_numeric(self.req,self.out,time.monotonic()+20,adapter,combiner,a)
 def test_real_saved_saa_evidence_integer_bound_and_original_ub_no_q(self):
  with patch.object(d,'solve_milp',side_effect=self.fake) as call,patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q')):r=self.run_it()
  old=json.loads(Path(self.req['references']['candidate_audit']['path']).read_text())
  self.assertEqual(call.call_count,1);self.assertEqual(r['upper_bound_usd'],old['upper_bound_usd']);self.assertGreaterEqual(r['lower_bound_usd'],old['lower_bound_usd'])
  self.assertEqual(r['new_Q_solve_count'],0);self.assertFalse(r['planning_acceptance']);self.assertTrue(r['original_saa_status_not_relabelled'])
 def test_native_bound_inconsistent_is_never_accepted(self):
  def bad(*args,**kw):r=self.fake(*args,**kw);r.mip_dual_bound+=1;return r
  with patch.object(d,'solve_milp',side_effect=bad),self.assertRaises(ValueError):self.run_it()
  self.assertFalse((self.out/'result.json').exists())
 def test_first_stage_semantic_tamper_rejected_before_solve(self):
  p=self.root/'first/arrays.npz'
  with np.load(p) as z:v={k:z[k].copy() for k in z.files}
  v['c'][0]+=1;np.savez_compressed(p,**v)
  with patch.object(d,'solve_milp',side_effect=AssertionError('No bad model')),self.assertRaises(ValueError):self.run_it()
 def test_auditor_retains_original_integer_domain_and_constant(self):
  with patch.object(d,'solve_milp',side_effect=self.fake):self.run_it()
  refs=self.req['references'];q=json.loads(Path(refs['fixed_q_request']['path']).read_text());aa=json.loads(Path(refs['candidate_audit']['path']).read_text());env=json.loads(Path(refs['template_envelope']['path']).read_text())
  first=d.load_first(refs['first_stage'],env['first_stage_sha256'],q['plan'],aa['fixed_q_audit']['first_stage_cost_usd']);ctx=adapter.SAASavedDualContext(**self.f.kw)
  cuts={i:ctx.derive(i)[:2] for i in range(len(ctx.ids))};m,_=combiner.build_saa_dual_bound(first,ctx.ids,cuts,constants_usd=q['constants_usd']);bad=copy.deepcopy(m);bad['integrality'][:]=0
  with self.assertRaises(ValueError):d.audit_affine_model(bad,first,ctx.ids,cuts,q['constants_usd'])
 def test_source_review_gate_rejects_resealed_wrong_driver_before_loading(self):
  ref=file_reference(d.__file__);rr=self.req['references']['candidate_audit'];adm=dict(kind='SAA_SAVED_DUAL_BOUND_ADMISSION_V1',driver_reference=ref,request_reference=rr)
  review=dict(kind='SAA_SAVED_DUAL_BOUND_PRODUCTION_REVIEW_V1',passed=True,admission_payload_sha256=d.canonical_hash(adm),driver_reference=ref,request_reference=rr)
  adm['independent_review_reference']=self.write(self.root/'source-review.json',review)
  self.assertTrue(d.reviewed_admission(adm,rr)['passed'])
  review['driver_reference']=rr;adm['independent_review_reference']=self.write(self.root/'source-review.json',review)
  with self.assertRaisesRegex(ValueError,'independent source review'):d.reviewed_admission(adm,rr)
 def test_native_auditor_only_three_explicit_saa_names_change(self):
  import ast
  original=(R/'scripts/audits/audit_case0_partial_native.py').read_text()
  expected=original.replace('audit_partial_native_bound','audit_saa_affine_native_bound').replace('INDEPENDENT_CASE0_PARTIAL_SCENARIO_MODEL_AUDIT_V1','INDEPENDENT_SAA_AFFINE_MODEL_AUDIT_V1').replace('INDEPENDENT_CASE0_PARTIAL_NATIVE_AUXILIARY_BOUND_AUDIT_V1','INDEPENDENT_SAA_AFFINE_NATIVE_BOUND_AUDIT_V1')
  self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse(Path(a.__file__).read_text())))
