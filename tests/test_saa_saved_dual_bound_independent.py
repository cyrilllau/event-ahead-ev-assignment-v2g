"""Independent risk/domain/native-policy negatives for bound-only controller."""
import ast,copy,unittest
import numpy as np
from unittest.mock import patch
from tests.test_saa_saved_dual_bound_controller import SAABoundControllerTests,d,a,R

class IndependentSAAControllerTests(SAABoundControllerTests):
 def test_nonuniform_combiner_objective_rejected_before_native(self):
  original=d.audit_affine_model
  def audit(model,*args,**kw):
   changed=copy.deepcopy(model);changed['c'][0]+=.25
   return original(changed,*args,**kw)
  with patch.object(d,'audit_affine_model',side_effect=audit),patch.object(d,'solve_milp',side_effect=AssertionError('No nonuniform solve')),self.assertRaises(ValueError):self.run_it()
 def test_double_counted_scenario_constant_rejected(self):
  original=d.audit_affine_model
  def audit(model,*args,**kw):
   changed=copy.deepcopy(model);changed['objective_constant_usd']+=1
   return original(changed,*args,**kw)
  with patch.object(d,'audit_affine_model',side_effect=audit),patch.object(d,'solve_milp',side_effect=AssertionError('No doublecount solve')),self.assertRaises(ValueError):self.run_it()
 def test_wrong_native_policy_retains_no_result(self):
  def bad(*args,**kw):
   z=self.fake(*args,**kw);z.native_record['effective_options']['Threads']=4;return z
  with patch.object(d,'solve_milp',side_effect=bad),self.assertRaises(ValueError):self.run_it()
  self.assertFalse((self.out/'result.json').exists())

class NativeDeltaTest(unittest.TestCase):
 def test_original_three_token_native_copy(self):
  old=(R/'scripts/audits/audit_case0_partial_native.py').read_text()
  expected=old.replace('audit_partial_native_bound','audit_saa_affine_native_bound').replace('INDEPENDENT_CASE0_PARTIAL_SCENARIO_MODEL_AUDIT_V1','INDEPENDENT_SAA_AFFINE_MODEL_AUDIT_V1').replace('INDEPENDENT_CASE0_PARTIAL_NATIVE_AUXILIARY_BOUND_AUDIT_V1','INDEPENDENT_SAA_AFFINE_NATIVE_BOUND_AUDIT_V1')
  self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((R/'scripts/audits/audit_saa_affine_native.py').read_text())))

if __name__=='__main__':unittest.main()
