"""SAA carryover dispatch/accounting through real queue, subprocess and numerics.

Only the formal predecessor certificate validator is mocked at its explicit
boundary (its own tests verify original formal evidence). No production bypass,
worker, assembly, numerical audit, receipt or final seal is mocked.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from evcs_experiments import campaign_batch as batch,saa_failure_carryover as carry
from evcs_experiments.campaign_batch import file_reference
import test_saa_queue_prelude as queue_fixture
import test_saa_planning_worker as workers
import test_campaign_batch as queues

class SAACarryoverWiringTests(unittest.TestCase):
 def setUp(self):
  self.f=workers.SAAWorkerTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.parent=queues.CampaignBatchTests(methodName='runTest');self.parent.setUp();self.addCleanup(self.parent.tearDown)
  f=self.f;self.queue=f.root/'queue'
  self.ref=f.write(f.root/'carryover.json',dict(kind=carry.KIND,fixture_certificate_boundary=True))
  self.verified=dict(reference=self.ref,accounted_seconds=2.113231,plan_id='tiny.saa',successor_queue_root=str(self.queue))
  f.req.pop('model_manifest_reference');f.req.update(model_input_mode='assemble',max_coefficient_bytes=50_000_000,failure_carryover_reference=self.ref)
  f.reqref=f.write(f.root/'request.json',f.req)
  harness=queue_fixture.HARNESS.replace("recourses=r,input_binding={'fixture':'pipeline'})","recourses=r,input_binding={'fixture':'pipeline'},empirical_probabilities=f.empirical_probabilities,scenario_cost_constants_usd=f.scenario_cost_constants_usd-100)")
  line="run_attempt(ref,_development_fixture=_DevelopmentFixture(prepare=lambda **kw:p,budget=req['budget']))"
  harness=harness.replace(line,"from unittest.mock import patch\nfrom evcs_experiments import saa_failure_carryover as helper\nwith patch.object(helper,'verify_saa_failure_carryover',return_value="+repr(self.verified)+"):\n "+line)
  h=f.root/'worker.py';h.write_text(harness)
  self.req=dict(schema_version=1,batch_id='saa-carry',spec_id='S05',model_id='IEEE33-v54-Q1',domain='development',queue_root=str(self.queue),
   job_wall_seconds=30.,shutdown_reserve_seconds=1.,parent_wall_policy='verified_parent_remaining_v1',
   source_references=[file_reference(batch.__file__),file_reference(h),file_reference(carry.__file__)],budget_reference=f.write(f.root/'budget.json',f.budget),
   items=[dict(item_id='saa',plan_id='tiny.saa',plan_budget_seconds=30.,argv=[sys.executable,str(h)],cwd=str(f.root),request_reference=f.reqref,failure_carryover_reference=self.ref)])
  self.reqref=f.write(f.root/'queue_request.json',self.req)
 def test_failed_parent_cost_once_actual_worker_and_final_settlement(self):
  with patch.object(carry,'verify_saa_failure_carryover',return_value=self.verified),self.parent.parent(29.):
   result=batch.run_batch(self.reqref['path'],self.queue)
  self.assertEqual(result['status'],'COMPLETE_BATCH',result)
  attempt=self.queue/'saa__attempt_0001';inv=json.loads((attempt/'invocation.json').read_text())
  self.assertEqual(inv['cumulative_plan_seconds'],2.113231)
  self.assertEqual(inv['opening_carryover_seconds'],2.113231)
  self.assertEqual(inv['plan_remaining_seconds'],30.-2.113231)
  self.assertIsNone(inv['previous_checkpoint'])
  worker=json.loads((attempt/'worker_result.json').read_text());self.assertTrue(worker['full_support_verified'])
  self.assertEqual(worker['failure_carryover_reference'],self.ref)
  state=json.loads((self.queue/'state.json').read_text());self.assertEqual(len(state['finalization_receipts']),1)
  with patch.object(carry,'verify_saa_failure_carryover',return_value=self.verified),self.parent.parent(29.):
   repeated=batch.run_batch(self.reqref['path'],self.queue)
  self.assertEqual(repeated['new_attempt_item_ids'],[])
  self.assertEqual(repeated['cumulative_plan_seconds'],result['cumulative_plan_seconds'])
  self.assertGreater(result['cumulative_plan_seconds']['tiny.saa'],2.113231)
  self.assertLessEqual(result['cumulative_plan_seconds']['tiny.saa'],30.)
 def test_queue_source_or_method_mismatch_refused_before_certificate_call(self):
  self.req['source_references']=[r for r in self.req['source_references'] if r['path']!=carry.__file__]
  with patch.object(carry,'verify_saa_failure_carryover') as fn:
   with self.assertRaisesRegex(ValueError,'source-bound'):batch._carryovers(self.req)
   fn.assert_not_called()
  self.req['source_references'].append(file_reference(carry.__file__))
  self.f.req['method_id']='EA_DRO';self.req['items'][0]['request_reference']=self.f.write(self.f.root/'request.json',self.f.req)
  with self.assertRaisesRegex(ValueError,'SAA worker'):batch._carryovers(self.req)
 def invocation(self):
  ref=self.f.invocation();inv=json.loads(Path(ref['path']).read_text())
  inv.update(request_reference=self.reqref,failure_carryover_reference=self.ref,opening_carryover_seconds=2.113231,
   cumulative_plan_seconds=2.113231,plan_remaining_seconds=30.-2.113231)
  inv['remaining_seconds']=25.;inv['worker_deadline_monotonic']=inv['attempt_started_monotonic']+25.
  return inv,ref
 def test_worker_rejects_reset_opening_and_ccg_receipt(self):
  inv,ref=self.invocation();inv['opening_carryover_seconds']=0.
  with patch.object(carry,'verify_saa_failure_carryover',return_value=self.verified):
   with self.assertRaisesRegex(ValueError,'opening'):self.f.run_worker(self.f.write(Path(ref['path']),inv))
  self.f.write(Path(self.ref['path']),dict(kind='FAILED_PRE_RMP_CARRYOVER_V1'))
  self.ref=file_reference(self.ref['path']);self.f.req['failure_carryover_reference']=self.ref
  self.f.reqref=self.f.write(self.f.root/'request.json',self.f.req)
  inv['item_request_reference']=self.f.reqref;inv['failure_carryover_reference']=self.ref
  with self.assertRaisesRegex(ValueError,'CCG failure'):self.f.run_worker(self.f.write(Path(ref['path']),inv))
 def test_worker_rejects_free_preassembled_input(self):
  self.f.req.update(model_input_mode='preassembled');self.f.reqref=self.f.write(self.f.root/'request.json',self.f.req)
  inv,ref=self.invocation()
  with patch.object(carry,'verify_saa_failure_carryover',return_value=self.verified):
   with self.assertRaisesRegex(ValueError,'free preassembled'):self.f.run_worker(self.f.write(Path(ref['path']),inv))
