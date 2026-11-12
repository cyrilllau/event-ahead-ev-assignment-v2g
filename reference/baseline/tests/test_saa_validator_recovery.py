"""Explicit recovery dispatch; real installed SAA child and immutable receipts."""
import json,os,sys,tempfile,time,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import saa_validator_recovery as adapter,campaign_batch as batch
from evcs_experiments import formal_planning_worker as fw,saa_planning_worker as sw,saa_record_bridge as bridge,checkpoint_closure
from evcs_experiments.random_streams import canonical_hash
import test_saa_queue_prelude as prior
import test_saa_planning_worker as workers
import test_campaign_batch as queues

HARNESS=prior.HARNESS[:prior.HARNESS.index("ref=file_reference(os.environ")]+'''
from evcs_experiments.saa_validator_recovery import main
from evcs_experiments import formal_planning_worker,saa_planning_worker
# Explicit test-only reproduction of the old generic relative-ref defect.
def old_closure(ref,seen=None):
 import json
 from evcs_experiments.campaign_batch import verify_reference
 verify_reference(ref)
 def walk(v):
  if isinstance(v,dict):
   if set(v)=={'path','sha256','size_bytes'}:old_closure(v)
   else:
    for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 walk(json.loads(Path(ref['path']).read_text()))
formal_planning_worker._closure=saa_planning_worker._closure=old_closure
ref=file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
inv=json.loads(Path(ref['path']).read_text());req=json.loads(Path(inv['item_request_reference']['path']).read_text())
p.empirical_probabilities=[1./len(r)]*len(r)
probe=req['development_inventory_probe_reference']
try:old_closure(probe)
except ValueError:pass
else:raise AssertionError('Original generic closure unexpectedly accepted relative inventory')
p.input_binding['historical_release']=probe
main(_development_fixture=_DevelopmentFixture(prepare=lambda **kw:p,budget=req['budget']))
'''

class RecoveryTests(unittest.TestCase):
 def setUp(self):
  self.f=workers.SAAWorkerTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.parent=queues.CampaignBatchTests(methodName='runTest');self.parent.setUp();self.addCleanup(self.parent.tearDown)
  release=self.f.root/'historical-release';release.mkdir();(release/'data.txt').write_text('immutable')
  member=batch.file_reference(release/'data.txt');member['path']='data.txt'
  probe=self.f.write(release/'RELEASE_MANIFEST.json',dict(schema_version=1,release_name=release.name,created_at_utc='2026-09-09T00:00:00+00:00',git_head='0'*40,working_tree_snapshot=True,files=[member]))
  self.f.req['development_inventory_probe_reference']=probe
  self.f.req.pop('model_manifest_reference');self.f.req.update(model_input_mode='assemble',max_coefficient_bytes=50_000_000);self.f.reqref=self.f.write(self.f.root/'request.json',self.f.req)
  self.root=self.f.root;self.queue=self.root/'queue';self.queue.mkdir();self.h=self.root/'child.py';self.h.write_text(HARNESS)
  self.package_root=Path(batch.__file__).resolve().parents[2]
  self.manifest=self.package_root/('.test-recovery-'+uuid.uuid4().hex+'.json');self.addCleanup(lambda:self.manifest.unlink(missing_ok=True))
  rows=[dict(path=Path(m.__file__).resolve().relative_to(self.package_root).as_posix(),sha256=batch.file_reference(m.__file__)['sha256']) for m in (batch,fw,sw,bridge)]
  self.manifest.write_text(json.dumps(dict(files=rows)))
  self.q=dict(schema_version=1,batch_id='saa-recovery',spec_id='S05',model_id='IEEE33-v54-Q1',domain='development',queue_root=str(self.queue),job_wall_seconds=30.,shutdown_reserve_seconds=1.,parent_wall_policy='verified_parent_remaining_v1',source_references=[batch.file_reference(batch.__file__),batch.file_reference(self.h)],budget_reference=self.f.write(self.root/'budget.json',self.f.budget),items=[dict(item_id='saa',plan_id='tiny.saa',plan_budget_seconds=30.,argv=[sys.executable,str(self.h)],cwd=str(self.root),request_reference=self.f.reqref)])
  self.qref=self.f.write(self.root/'queue.json',self.q)
  self.f.write(self.queue/'state.json',dict(schema_version=1,request_reference=self.qref,attempt_receipts=[],active=None,halted=None,runs=[]))
 def admit(self,index=1,timeout=29.):
  self.parent_context=self.parent.parent(timeout);self.parent_directory=self.parent_context.__enter__();self.addCleanup(lambda:self.parent_context.__exit__(None,None,None))
  parent=batch._parent(self.q)
  state=json.loads((self.queue/'state.json').read_text());_,spent=batch._history(state,self.qref,self.queue)
  self.a=dict(kind=adapter.KIND,domain='development',adapter_reference=batch.file_reference(adapter.__file__),validator_reference=batch.file_reference(checkpoint_closure.__file__),queue_reference=self.qref,worker_request_reference=self.f.reqref,original_release_reference=batch.file_reference(self.manifest),baseline_ledger_reference=self.f.write(self.root/f'baseline{index}.json',state),queue_root=str(self.queue),plan_id='tiny.saa',original_module_references={name:batch.file_reference(m.__file__) for name,m in [('campaign_batch',batch),('formal_planning_worker',fw),('saa_planning_worker',sw),('saa_record_bridge',bridge)]},original_python=sys.executable,parent_job_id=parent['job_id'],parent_job_directory=parent['directory'],baseline_receipt_references=state['attempt_receipts'],baseline_result_references=[json.loads(Path(r['path']).read_text())['result_reference'] for r in state['attempt_receipts']],legacy_result_references=[],opening_cumulative_seconds=spent.get('tiny.saa',0.),parent_execution_path=str(self.root/f'parent{index}.json'),parent_coverage_path=str(self.root/f'coverage{index}.json'),development_entry_reference=batch.file_reference(self.h))
  review=dict(kind='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1',passed=True,admission_payload_sha256=canonical_hash(self.a),adapter_reference=self.a['adapter_reference'],validator_reference=self.a['validator_reference'])
  self.a['independent_review_reference']=self.f.write(self.root/f'review{index}.json',review)
  self.ref=self.f.write(self.root/f'admission{index}.json',self.a)
  manifest=json.loads((self.parent_directory/'manifest.json').read_text());manifest['argv']=[sys.executable,adapter.__file__,'parent','--admission',self.ref['path'],'--expected-sha256',self.ref['sha256']];(self.parent_directory/'manifest.json').write_text(json.dumps(manifest))
  return self.ref
 def run_parent(self):
  argv=[sys.executable,adapter.__file__,'parent','--admission',self.ref['path'],'--expected-sha256',self.ref['sha256']]
  clock=batch._parent(self.q)['wall_clock']
  with patch.object(adapter,'actual_argv',return_value=argv),patch.object(batch,'_supervisor_remaining',return_value=clock),patch.object(batch,'process_identity',side_effect=queues.real_identity):return adapter.run_parent(self.ref)
 def test_real_child_complete_and_first_written_provenance(self):
  self.admit();out=self.run_parent();self.assertEqual(out['queue_result']['status'],'COMPLETE_BATCH',str(out)+''.join(p.read_text() for p in self.queue.rglob('stderr.log')))
  inv=batch.file_reference(self.queue/'saa__attempt_0001/invocation.json');result=batch.file_reference(self.queue/'saa__attempt_0001/worker_result.json')
  value=adapter.validate_execution_result(result,inv);self.assertIn(value['validation_recovery_execution_reference'],value['artifacts'])
  candidate=json.loads((self.queue/'saa__attempt_0001/candidate_solver_record.json').read_text())
  self.assertEqual(candidate['validation_recovery_execution_reference'],value['validation_recovery_execution_reference'])
 def test_changed_admission_or_validator_refused(self):
  self.admit();self.a['opening_cumulative_seconds']=0.1;ref=self.f.write(self.root/'wrong.json',self.a)
  with self.assertRaisesRegex(ValueError,'reviewed'):adapter.admission(ref)
  self.a['validator_reference']['sha256']='0'*64
  with self.assertRaises(ValueError):adapter.admission(self.f.write(self.root/'wrong2.json',self.a))
 def test_missing_worker_receipt_is_rejected(self):
  self.admit();self.run_parent();p=self.queue/'saa__attempt_0001/worker_result.json';value=json.loads(p.read_text());value.pop('validation_recovery_execution_reference');p.write_text(json.dumps(value))
  with self.assertRaisesRegex(ValueError,'omitted'):adapter.validate_execution_result(batch.file_reference(p),batch.file_reference(p.parent/'invocation.json'))

 def test_real_time_limit_then_same_queue_resume_preserves_model_and_charge(self):
  self.admit(timeout=9.);first=self.run_parent();self.assertEqual(first['queue_result']['status'],'TIME_LIMIT',first)
  first_attempt=self.queue/'saa__attempt_0001';old=json.loads((first_attempt/'worker_result.json').read_text());self.assertEqual(old['status'],'TIME_LIMIT')
  cp=json.loads(Path(old['checkpoint']['path']).read_text());model=dict(cp['model_manifest_reference']);completion=dict(cp['model_completion_reference'])
  prior_charge=first['queue_result']['cumulative_plan_seconds']['tiny.saa'];self.assertGreater(prior_charge,0)
  self.parent_context.__exit__(None,None,None)
  self.admit(2);second=self.run_parent();self.assertEqual(second['queue_result']['status'],'COMPLETE_BATCH',second)
  inv=json.loads((self.queue/'saa__attempt_0002/invocation.json').read_text());self.assertEqual(inv['cumulative_plan_seconds'],prior_charge)
  final=json.loads((self.queue/'saa__attempt_0002/worker_result.json').read_text());latest=json.loads(Path(final['checkpoint']['path']).read_text())
  self.assertEqual(latest['model_manifest_reference'],model);self.assertEqual(latest['model_completion_reference'],completion)
  self.assertEqual(batch.file_reference(model['path']),model);self.assertGreater(second['queue_result']['cumulative_plan_seconds']['tiny.saa'],prior_charge)
 def test_formal_admission_is_closed_until_external_fee_ledger_exists(self):
  self.admit();self.a['domain']='formal';ref=self.f.write(self.root/'formal.json',self.a)
  with self.assertRaisesRegex(ValueError,'fee ledger'):adapter.admission(ref)
