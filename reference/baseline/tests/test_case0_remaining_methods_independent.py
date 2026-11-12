"""Independent orchestration receipt and failed-parent budget guards."""
import json,unittest
from tests.test_case0_remaining_methods import RemainingMethodsTests

class IndependentRemainingTests(RemainingMethodsTests):
 def test_completed_methods_with_intact_receipts_skip_without_solver(self):
  self.run_it();state=self.api.read(self.root/'state.json');state['remaining_methods_parent_pending_settlement']=False
  self.api.save_state(self.root,state);self.op=self.root/'operations/intact_resume';self.op.mkdir();self.calls=[]
  self.api.configuration_attempt=lambda root,state,key:(root/'planning'/key,state['configuration_spent_seconds'][key])
  result=self.run_it();self.assertEqual(self.calls,[]);self.assertEqual(result['status'],'FOUR_METHOD_RESULTS_PENDING_ADAPTED_INDEPENDENT_AUDIT')
 def test_unsettled_previous_parent_blocks_all_children(self):
  self.s['remaining_methods_parent_pending_settlement']=True;self.api.save_state(self.root,self.s)
  with self.assertRaises(ValueError):self.run_it()
  self.assertEqual(self.calls,[])
 def test_start_keeps_original_producer_config_hash_field(self):
  self.run_it()
  for method in ('OR_DRO','SAA','ARO'):
   row=json.loads((self.root/'planning/size0'/method/'start.json').read_text())
   self.assertIn('config_sha256',row)
 def test_coherently_kept_result_cannot_reset_completed_budget(self):
  self.run_it();state=self.api.read(self.root/'state.json');state['remaining_methods_parent_pending_settlement']=False
  key='size0/OR_DRO';state['configuration_spent_seconds'][key]=0.;self.api.save_state(self.root,state)
  self.op=self.root/'operations/retry';self.op.mkdir();self.calls=[]
  with self.assertRaises(ValueError):self.run_it()
  self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()
