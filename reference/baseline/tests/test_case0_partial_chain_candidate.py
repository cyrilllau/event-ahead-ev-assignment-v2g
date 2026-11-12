"""Full original saved-file numeric entry; exactly one real tiny LP after selection."""
import json,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.case0_planning import _json
from tests import test_case0_two_anchor_bound as ancestry
from evcs_experiments.config import get_paths
from tests.test_case0_partial_candidate import api
from evcs_experiments import case0_partial_cut as helper
from evcs_experiments import case0_partial_chain as chain
import importlib.util
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
driver=load("chain_entry",get_paths().project_root/"scripts/audits/audit_case0_partial_chain_candidate.py")
nativecheck=load("chain_entry_native",get_paths().project_root/"scripts/audits/audit_case0_partial_native.py")

class PartialChainEntryTests(unittest.TestCase):
 def setUp(self):ancestry.TwoAnchorTests.setUp(self)
 def test_saved_chain_entry_reuses_exact_q_and_rejects_nonoptimal_prior(self):
  model,check,_=ancestry.TwoAnchorTests.prepare(self);native=ancestry.test_native_result(model);base=self.root/'base';base.mkdir()
  def save(name,value):
   p=base/name;p.write_text(json.dumps(_json(value)));return file_reference(p)
  save_npz(base/'matrix.npz',model['matrix']);np.savez_compressed(base/'arrays.npz',**{k:model[k] for k in ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')});np.save(base/'vector.npy',native.x)
  vector=file_reference(base/'vector.npy');oldpath=self.saved/'result.json';old=json.loads(oldpath.read_text());old['source_binding']['model']=file_reference(self.producer/'model.json');oldpath.write_text(json.dumps(old));oldref=file_reference(oldpath)
  na=ancestry.fixture.audit.audit_case0_native_bound(model,check,native.native_record,native_status=native.status,native_bound=native.mip_dual_bound,incumbent=native.x,expected_coefficient_sha256=check['coefficient_sha256'],fixed_upper_bound=self.ub['upper_bound_usd'],deadline=time.monotonic()+5)
  naref=save('native_bound_audit.json',na)
  prior=dict(new_native_lower_bound_usd=na['lower_bound_usd'],native_status=native.status,auxiliary_mip_gap=native.mip_gap,native_bound_audit_reference=naref,auxiliary_objective_usd=float(model['c']@native.x+model['objective_constant_usd']),status='AUDITED_TWO_ANCHOR_AUXILIARY_BOUND',model_sha256=self.header['model_sha256'],plan_sha256=self.ub['plan_sha256'],source_binding={'ub_result':oldref},auxiliary_model_sha256=check['coefficient_sha256'],auxiliary_vector_reference=vector,lower_bound_usd=self.weighted_result['lower_bound_usd'],gap=.1)
  groups=[]
  for folder in (self.root/'two/anchor_0/revalidated_cuts',self.root/'two/anchor_1'):
   groups.append([dict(cut=file_reference(folder/f'cut_{i:03d}.json'),gradient=file_reference(folder/f'gradient_{i:03d}.npy')) for i in range(len(self.P.support_ids))])
  refs=dict(model=file_reference(self.producer/'model.json'),matrix=file_reference(self.producer/'matrix.npz'),model_arrays=file_reference(self.producer/'model_arrays.npz'),model_hash_checker=file_reference(get_paths().project_root/'scripts/audits/audit_case0_saved_lp_bound.py'),base_matrix=file_reference(base/'matrix.npz'),base_arrays=file_reference(base/'arrays.npz'),base_vector=vector,base_result=save('result.json',prior),base_audit=save('audit.json',check),base_construction=save('construction.json',{'master_coefficient_sha256':check['coefficient_sha256']}),base_cuts=save('cuts.json',{'groups':groups}),old_ub=oldref,old_probability=file_reference(self.saved/'probability.json'))
  out=self.root/'partial';out.mkdir();original=api.linprog;calls=[]
  def solve(*a,**kw):
   self.assertTrue((out/'selection.json').exists());self.assertTrue((out/'lp_started.json').exists());self.assertLessEqual(kw['options']['time_limit'],3);calls.append(1);return original(*a,**kw)
  with patch.object(api,'FEATURE_NAMES',self.P.feature_ids),patch.object(api,'linprog',side_effect=solve):
   r=api.run_numeric(refs,out,time.monotonic()+15,3,helper)
  self.assertEqual(len(calls),1);self.assertEqual(r['master_solve_count'],0);self.assertFalse(r['new_upper_bound_verified']);self.assertTrue(r['passed'])
  cm=api.model_files(file_reference(out/'augmented_matrix.npz'),file_reference(out/'augmented_arrays.npz'));ca=json.loads((out/'partial_model_audit.json').read_text())
  cn=ancestry.test_native_result(cm);nf=self.header['variable_slices']['first_stage'][1]
  na=nativecheck.audit_partial_native_bound(cm,ca,cn.native_record,native_status=cn.status,native_bound=cn.mip_dual_bound,incumbent=cn.x,expected_coefficient_sha256=ca['coefficient_sha256'],fixed_upper_bound=old['upper_bound_usd'],deadline=time.monotonic()+5)
  chainroot=self.root/'chain';chainroot.mkdir();np.save(chainroot/'auxiliary_vector.npy',cn.x)
  nap=chainroot/'native_audit.json';nap.write_text(json.dumps(na))
  cr=dict(status='AUDITED_PARTIAL_SCENARIO_AUXILIARY_BOUND',native_status='OPTIMAL',model_sha256=self.header['model_sha256'],old_plan_sha256=old['plan_sha256'],source_binding=dict(old_ub=oldref,matrix=file_reference(out/'augmented_matrix.npz'),arrays=file_reference(out/'augmented_arrays.npz'),partial_audit=file_reference(out/'partial_model_audit.json')),native_audit_reference=file_reference(nap),new_native_lower_bound_usd=na['lower_bound_usd'],lower_bound_usd=na['lower_bound_usd'],gap=.1)
  crp=chainroot/'result.json';crp.write_text(json.dumps(cr))
  tr=save('retained_terminal.json',{'kind':'CASE0_PARTIAL_SCENARIO_CUT_TERMINAL_REVIEW_V1','passed':True,'numerical_recheck_passed':True,'references':{'result':file_reference(out/'result.json')}})
  sett=save('retained_settlement.json',{'kind':'CASE0_PARTIAL_SCENARIO_CUT_SETTLEMENT_V1','passed':True,'result_reference':file_reference(out/'result.json'),'terminal_review_reference':tr})
  refs.update(chain_matrix=file_reference(out/'augmented_matrix.npz'),chain_arrays=file_reference(out/'augmented_arrays.npz'),chain_audit=file_reference(out/'partial_model_audit.json'),chain_result=file_reference(crp),chain_vector=file_reference(chainroot/'auxiliary_vector.npy'),retained_selection=file_reference(out/'selection.json'),retained_cut=file_reference(out/'new_cut.json'),retained_gradient=file_reference(out/'new_gradient.npy'),retained_lp_audit=file_reference(out/'selected_lp_audit.json'),retained_q_result=file_reference(out/'result.json'),retained_settlement=sett)
  dest=self.root/'nextchain';dest.mkdir();calls=[]
  def only_one(*a,**kw):
   self.assertTrue((dest/'selection.json').exists());calls.append(1);self.assertEqual(len(calls),1);return original(*a,**kw)
  with patch.object(driver,'FEATURE_NAMES',self.P.feature_ids),patch.object(driver,'linprog',side_effect=only_one):
   result=driver.run_numeric(refs,dest,time.monotonic()+20,3,chain)
  self.assertTrue(result['passed']);self.assertEqual(result['new_Q_solve_count'],len(calls));self.assertEqual(len(calls),0)
  self.assertEqual(result['status'],'REUSED_EXACT_ANCHOR_SCENARIO')
  self.assertEqual(result['upper_bound_usd'],old['upper_bound_usd']);self.assertFalse(result['planning_acceptance'])
  if calls:
   added=driver.model_files(file_reference(dest/'augmented_matrix.npz'),file_reference(dest/'augmented_arrays.npz'))
   self.assertEqual(added['matrix'].shape[0],cm['matrix'].shape[0]+1)
   np.testing.assert_array_equal(added['matrix'][:-1].toarray(),cm['matrix'].toarray())
  # A valid hash of a changed native status cannot authorize another Q.
  cr['native_status']='TIME_LIMIT';crp.write_text(json.dumps(cr));refs['chain_result']=file_reference(crp)
  rejected=self.root/'badnext';rejected.mkdir()
  with patch.object(driver,'FEATURE_NAMES',self.P.feature_ids),patch.object(driver,'linprog',side_effect=AssertionError('No invalid Q')),self.assertRaisesRegex(ValueError,'optimal prior chain'):
   driver.run_numeric(refs,rejected,time.monotonic()+20,3,chain)
if __name__=='__main__':unittest.main()
