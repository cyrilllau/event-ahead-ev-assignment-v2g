"""Completed assembly survives short grants; incomplete work never claims resume."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from evcs_experiments import saa_streaming,saa_streaming_store
from evcs_experiments.campaign_batch import file_reference,_result
import test_saa_planning_worker as workers

class AssemblyCheckpointTests(unittest.TestCase):
 def setUp(self):
  self.f=workers.SAAWorkerTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.f.req.pop('model_manifest_reference');self.f.req.update(model_input_mode='assemble',max_coefficient_bytes=50_000_000)
  self.f.reqref=self.f.write(self.f.root/'request.json',self.f.req)
 def short_invocation(self):
  ref=self.f.invocation();inv=json.loads(Path(ref['path']).read_text())
  inv['worker_deadline_monotonic']=inv['attempt_started_monotonic']+10.;inv['remaining_seconds']=10.
  return self.f.write(Path(ref['path']),inv)
 def test_short_grant_commits_model_before_reload_then_longer_grant_reuses(self):
  real=saa_streaming.assemble_streaming_saa
  with patch.object(saa_streaming,'assemble_streaming_saa',wraps=real) as assemble,patch.object(saa_streaming_store,'load_streaming_saa',side_effect=AssertionError('Short grant must not reload')):
   inv=self.short_invocation();old=self.f.run_worker(inv)
  self.assertEqual(assemble.call_count,1);self.assertEqual(old['status'],'TIME_LIMIT')
  self.assertEqual(_result(file_reference(self.f.root/'attempt0/worker_result.json'),inv)['status'],'TIME_LIMIT')
  cp=json.loads(Path(old['checkpoint']['path']).read_text())
  self.assertIsNone(cp['phase_checkpoint_reference']);self.assertEqual(cp['phase_status'],'TIME_LIMIT_ASSEMBLY_COMPLETE_BEFORE_MODEL_RELOAD')
  self.assertFalse(old['full_support_verified']);self.assertFalse((self.f.root/'output/phases').exists())
  completion=json.loads(Path(cp['model_completion_reference']['path']).read_text())
  self.assertEqual(completion['model_manifest_reference'],cp['model_manifest_reference'])
  self.assertEqual(completion['planning_source_references'],self.f.req['planning_source_references'])
  original_model_ref=dict(cp['model_manifest_reference']);prior=old['cumulative_accounted_seconds']+.1
  with patch.object(saa_streaming,'assemble_streaming_saa',side_effect=AssertionError('No repeated assembly')):
   result=self.f.run_worker(self.f.invocation(1,prior,old))
  self.assertEqual(result['status'],'COMPLETE');self.assertGreater(result['cumulative_accounted_seconds'],prior)
  latest=json.loads(Path(result['checkpoint']['path']).read_text())
  self.assertEqual(latest['model_manifest_reference'],original_model_ref)
  self.assertEqual(latest['model_completion_reference'],cp['model_completion_reference'])
  self.assertEqual(file_reference(original_model_ref['path']),original_model_ref)
 def test_changed_complete_model_bytes_cannot_resume(self):
  old=self.f.run_worker(self.short_invocation());cp=json.loads(Path(old['checkpoint']['path']).read_text())
  manifest=json.loads(Path(cp['model_manifest_reference']['path']).read_text())
  Path(manifest['numeric_artifacts'][0]['path']).write_bytes(b'altered original model array')
  with self.assertRaises(ValueError):self.f.run_worker(self.f.invocation(1,old['cumulative_accounted_seconds']+.1,old))
 def test_incomplete_assembly_is_failure_without_resumable_checkpoint(self):
  with patch.object(saa_streaming,'assemble_streaming_saa',side_effect=TimeoutError('incomplete assembly')):
   with self.assertRaisesRegex(TimeoutError,'incomplete'):self.f.run_worker(self.short_invocation())
  attempt=self.f.root/'attempt0'
  self.assertTrue((attempt/'worker_failure.json').exists())
  self.assertFalse((attempt/'worker_result.json').exists());self.assertFalse((attempt/'worker_checkpoint.json').exists())
  self.assertFalse((self.f.root/'output/model_completion.json').exists())
 def test_checkpoint_reserve_is_given_to_constructor_and_invalid_reserve_rejected(self):
  real=saa_streaming.assemble_streaming_saa;invref=self.short_invocation();inv=json.loads(Path(invref['path']).read_text())
  with patch.object(saa_streaming,'assemble_streaming_saa',wraps=real) as assemble:self.f.run_worker(invref)
  self.assertEqual(assemble.call_args.kwargs['deadline'],inv['worker_deadline_monotonic']-1.)
  self.f.req['model_checkpoint_reserve_seconds']=0.
  self.f.reqref=self.f.write(self.f.root/'request.json',self.f.req)
  with self.assertRaisesRegex(ValueError,'reserve'):self.f.run_worker(self.f.invocation(1))
