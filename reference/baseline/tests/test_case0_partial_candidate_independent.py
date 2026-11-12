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

class PartialEntryTests(unittest.TestCase):
 def setUp(self):ancestry.TwoAnchorTests.setUp(self)
 def test_full_saved_entry_persists_selection_before_one_lp(self):
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
  prior['native_status']=1;refs['base_result']=save('result.json',prior);rejected=self.root/'rejected';rejected.mkdir()
  with patch.object(api,'FEATURE_NAMES',self.P.feature_ids),patch.object(api,'linprog',side_effect=AssertionError('No second Q')):
   with self.assertRaisesRegex(ValueError,'optimal original auxiliary'):api.run_numeric(refs,rejected,time.monotonic()+15,3,helper)
if __name__=='__main__':unittest.main()
