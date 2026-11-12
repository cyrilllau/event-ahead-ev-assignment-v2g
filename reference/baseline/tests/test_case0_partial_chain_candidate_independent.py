"""Independent driver checks: real saved physical entry and exact new-Q routing."""
import ast,json,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.config import get_paths
from tests import test_case0_partial_chain_candidate as fixture

class ChainDriverIndependentTests(unittest.TestCase):
 def test_original_saved_lp_and_probability_kernels_are_literal_ast_copies(self):
  root=get_paths().project_root
  old=ast.parse((root/'scripts/audits/audit_case0_partial_candidate.py').read_text())
  new=ast.parse((root/'scripts/audits/audit_case0_partial_chain_candidate.py').read_text())
  for name in ('original_probability','extract_scenario','audit_selected_lp'):
   a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)
   b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
   self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
 def test_canonical_new_continuous_point_recomputes_one_original_q(self):
  # The fixture uses genuine physical original A/B/Q and an explicitly synthetic
  # native receipt. A 1e-12 continuous change still inside the unchanged numerical
  # feasibility/optimality checks is a NEW anchor, never a reused old Q label.
  f=fixture.PartialChainEntryTests('test_saved_chain_entry_reuses_exact_q_and_rejects_nonoptimal_prior');f.setUp();self.addCleanup(f.doCleanups)
  original=fixture.driver.run_numeric;seen=[]
  def intercept(refs,out,deadline,lp,helper):
   result=original(refs,out,deadline,lp,helper)
   if not seen:
    seen.append(1);path=Path(refs['chain_vector']['path']);raw=path.read_bytes();v=np.load(path,allow_pickle=False)
    arrays=fixture.driver.model_files(refs['chain_matrix'],refs['chain_arrays']);header=fixture.driver.read(refs['model']);nf=header['variable_slices']['first_stage'][1]
    indices=np.flatnonzero((np.asarray(arrays['integrality'][:nf])==0)&(v[:nf]>1e-6))
    self.assertGreater(len(indices),0);v[int(indices[0])]-=1e-12;np.save(path,v,allow_pickle=False);updated=dict(refs,chain_vector=file_reference(path))
    newout=f.root/'independent_new_canonical_anchor';newout.mkdir();calls=[];lp_original=fixture.api.linprog
    def solve(*a,**kw):
     sel=json.loads((newout/'selection.json').read_text());self.assertEqual(sel['status'],'NEW_Q_REQUIRED');self.assertTrue(sel['new_Q_required']);calls.append(1)
     return lp_original(*a,**kw)
    try:
     with patch.object(fixture.driver,'linprog',side_effect=solve):
      fresh=original(updated,newout,time.monotonic()+20,3,helper)
     self.assertEqual(len(calls),1);self.assertEqual(fresh['new_Q_solve_count'],1);self.assertEqual(fresh['master_solve_count'],0)
     self.assertFalse(fresh['new_upper_bound_verified'] or fresh['planning_acceptance']);self.assertTrue(fresh['passed'])
     m=fixture.driver.model_files(file_reference(newout/'augmented_matrix.npz'),file_reference(newout/'augmented_arrays.npz'))
     self.assertEqual((m['matrix'][:-1]!=arrays['matrix']).nnz,0)
     self.assertEqual(m['matrix'].shape[0],arrays['matrix'].shape[0]+1)
     self.assertTrue((newout/'auxiliary_preparation_started.json').exists());self.assertGreaterEqual(json.loads((newout/'auxiliary_preparation_accounting.json').read_text())['actual_seconds'],0.)
    finally:path.write_bytes(raw)
   return result
  with patch.object(fixture.driver,'run_numeric',side_effect=intercept):f.test_saved_chain_entry_reuses_exact_q_and_rejects_nonoptimal_prior()
  self.assertEqual(len(seen),1)

if __name__=='__main__':unittest.main()
