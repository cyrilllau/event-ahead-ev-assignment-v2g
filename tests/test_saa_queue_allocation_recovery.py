"""Actual original queue and SAA numerical child survive an immutable orphan."""
import json
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import campaign_batch as batch,queue_attempt_allocation as allocation,saa_validator_recovery as adapter
from evcs_experiments.random_streams import canonical_hash
from test_saa_validator_recovery import RecoveryTests

class QueueAllocationRecoveryTests(RecoveryTests):
 def test_orphan_preserved_real_worker_receipt_and_budget(self):
  orphan=self.queue/'saa__attempt_0002';orphan.mkdir();(orphan/'failure.txt').write_text('unchanged failed attempt')
  self.admit()
  self.a.pop('independent_review_reference');self.a['queue_allocator_reference']=batch.file_reference(allocation.__file__)
  review=dict(kind='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1',passed=True,admission_payload_sha256=canonical_hash(self.a),adapter_reference=self.a['adapter_reference'],validator_reference=self.a['validator_reference'],queue_allocator_reference=self.a['queue_allocator_reference'])
  self.a['independent_review_reference']=self.f.write(self.root/'allocator-review.json',review)
  self.ref=self.f.write(self.root/'allocator-admission.json',self.a)
  manifest=json.loads((self.parent_directory/'manifest.json').read_text());manifest['argv'][-3:]=[self.ref['path'],'--expected-sha256',self.ref['sha256']];(self.parent_directory/'manifest.json').write_text(json.dumps(manifest))
  original=batch.run_batch;answer=self.run_parent()
  self.assertEqual(answer['queue_result']['status'],'COMPLETE_BATCH',answer)
  self.assertIs(batch.run_batch,original)
  self.assertEqual((orphan/'failure.txt').read_text(),'unchanged failed attempt')
  state=json.loads((self.queue/'state.json').read_text());self.assertEqual(len(state['attempt_receipts']),1)
  invocation=json.loads((self.queue/'saa__attempt_0003/invocation.json').read_text())
  self.assertEqual(invocation['cumulative_plan_seconds'],0.)
  self.assertGreater(answer['queue_result']['cumulative_plan_seconds']['tiny.saa'],0.)
  parent=json.loads(Path(self.a['parent_execution_path']).read_text())
  self.assertEqual(parent['queue_allocation_execution']['allocator_reference'],self.a['queue_allocator_reference'])
  result_path=self.queue/'saa__attempt_0003/worker_result.json';result=json.loads(result_path.read_text())
  old_receipt=result['validation_recovery_execution_reference'];receipt=json.loads(Path(old_receipt['path']).read_text())
  for key,bad in [('derived_function_sha256','0'*64),('original_function_sha256','0'*64),('receipts_and_budget_unchanged',False)]:
   changed=dict(parent,queue_allocation_execution=dict(parent['queue_allocation_execution'],**{key:bad}))
   changed_parent=self.f.write(self.root/('tampered-'+key+'.json'),changed)
   changed_receipt=self.f.write(self.root/('receipt-'+key+'.json'),dict(receipt,parent_execution_reference=changed_parent,actual_argv=adapter.child_argv(self.a,self.ref,changed_parent)))
   changed_result=dict(result,validation_recovery_execution_reference=changed_receipt,artifacts=[changed_receipt if r==old_receipt else r for r in result['artifacts']])
   changed_ref=self.f.write(self.root/('result-'+key+'.json'),changed_result)
   with self.assertRaisesRegex(ValueError,'allocator execution provenance'):
    adapter.validate_execution_result(changed_ref,batch.file_reference(result_path.parent/'invocation.json'))
 def test_source_or_allocation_schema_tamper_rejected(self):
  ref=batch.file_reference(batch.__file__);bad=dict(ref,sha256='0'*64)
  with self.assertRaises(ValueError):allocation.bind_original_queue(batch,bad)
  with patch('inspect.getsource',return_value='def run_batch(a,b):\n return None\n'):
   with self.assertRaisesRegex(ValueError,'schema'):allocation.bind_original_queue(batch,ref)
