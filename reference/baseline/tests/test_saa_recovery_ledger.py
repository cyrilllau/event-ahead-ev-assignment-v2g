"""Real tiny SAA queue/child accounting with external immutable parent settlements.

Linux supervisor observations are simulated; physical worker, native small solve,
queue receipts, completion model and numerical settlement remain real.
"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from evcs_experiments import saa_recovery_ledger as ledger, campaign_batch as batch
from evcs_experiments.random_streams import canonical_hash
import test_saa_validator_recovery as recovery

class LedgerTests(unittest.TestCase):
 def setUp(self):
  self.f=recovery.RecoveryTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.root=self.f.root.resolve();self.f.root=self.root;self.f.queue=self.f.queue.resolve()
  self.f.q['queue_root']=str(self.f.queue);self.f.qref=self.f.f.write(self.root/'queue.json',self.f.q)
  state=json.loads((self.f.queue/'state.json').read_text());state['request_reference']=self.f.qref
  self.f.f.write(self.f.queue/'state.json',state)
  self.jobs=self.root/'jobs';self.jobs.mkdir();self.f.parent.root=self.jobs
  self.folder=self.root/'external-ledger';self.folder.mkdir()
 def write(self,name,value):return self.f.f.write(self.root/name,value)
 def admit(self,index=1,timeout=9.):
  self.f.admit(index,timeout=timeout)
  p=self.f.parent_directory/'manifest.json';m=json.loads(p.read_text())
  m.update(cwd=str(self.root),outroot=str(self.jobs),threads=1,grace_seconds=.01)
  p.write_text(json.dumps(m));self.mref=batch.file_reference(p)
  self.jref=self.write(f'job-request-{index}.json',m)
  self.admission=self.f.ref
 def anchor(self):
  state=json.loads((self.f.queue/'state.json').read_text())
  self.opening=self.write('initial-snapshot.json',state)
  _,q=ledger._snapshot(self.opening,self.f.qref)
  a=dict(kind=ledger.ANCHOR,ledger_root=str(self.folder),campaign_root=str(self.root),queue_reference=self.f.qref,
   plan_id='tiny.saa',budget_seconds=30.,initial_queue_snapshot_reference=self.opening,initial_queue_seconds=q,
   ledger_source_reference=batch.file_reference(ledger.__file__))
  review=dict(kind='SAA_RECOVERY_LEDGER_ANCHOR_REVIEW_V1',passed=True,anchor_payload_sha256=canonical_hash(a))
  a['independent_review_reference']=self.write('anchor-review.json',review)
  self.aref=self.write('external-ledger/anchor.json',a)
 def reserve(self,opening=None):
  return ledger.reserve_recovery_parent(self.aref,parent_job_request_reference=self.jref,parent_manifest_reference=self.mref,
    admission_reference=self.admission,opening_queue_snapshot_reference=opening or self.opening)
 def terminal(self,seconds,index=1,state='SUCCEEDED'):
  job=self.f.parent_directory
  (job/'status.json').write_text(json.dumps(dict(job_id=job.name,state=state)))
  self.tref=self.write(str(job.relative_to(self.root)/'result.json'),dict(job_id=job.name,state=state,complete=True,elapsed_seconds=seconds))
  self.close=self.write(f'closing-{index}.json',json.loads((self.f.queue/'state.json').read_text()))
  return self.tref,self.close
 def test_prequeue_failure_charges_all_and_repeated_settlement_is_idempotent(self):
  self.admit();self.anchor();r=self.reserve();t,c=self.terminal(2.113231,state='FAILED')
  s=ledger.settle_recovery_parent(r,t,c);v=ledger.read(s)
  self.assertEqual(v['new_queue_charge_seconds'],0);self.assertEqual(v['extra_parent_seconds'],2.113231)
  self.assertEqual(ledger.settle_recovery_parent(r,t,c),s)
  self.assertEqual(len(list(self.folder.glob('settlement-*.json'))),1)
 def test_unsettled_parent_blocks_restart_and_missing_terminal_refused(self):
  self.admit();self.anchor();r=self.reserve()
  with self.assertRaisesRegex(ValueError,'Unsettled'):self.reserve()
  t,c=self.terminal(1,state='RUNNING')
  with self.assertRaisesRegex(ValueError,'not terminal'):ledger.settle_recovery_parent(r,t,c)
 def test_live_terminal_wrong_parent_and_wrong_argv_rejected(self):
  self.admit();self.anchor();r=self.reserve();t,c=self.terminal(1)
  with patch.object(ledger,'observation',return_value=dict(blocks_new_launch=True,supervisor_alive=True,command_alive=False)):
   with self.assertRaisesRegex(ValueError,'still live'):ledger.settle_recovery_parent(r,t,c)
  terminal=ledger.read(t);terminal['job_id']='wrong';Path(t['path']).write_text(json.dumps(terminal))
  with self.assertRaisesRegex(ValueError,'Wrong terminal'):ledger.settle_recovery_parent(r,batch.file_reference(t['path']),c)
 def test_over_budget_parent_and_resealed_anchor_reset_rejected(self):
  self.admit(timeout=30.);self.anchor()
  with self.assertRaisesRegex(ValueError,'timeout plus grace'):self.reserve()
  a=ledger.read(self.aref);a['initial_queue_seconds']=0.1
  self.aref=self.write('external-ledger/anchor.json',a)
  with self.assertRaisesRegex(ValueError,'cannot reset'):ledger._anchor(self.aref)
 def test_wrong_request_argv_cross_campaign_and_duplicate_receipt_rejected(self):
  self.admit();self.anchor();request=ledger.read(self.jref);request['argv']=['wrong'];self.jref=self.write('wrong-job.json',request)
  with self.assertRaisesRegex(ValueError,'argv differs'):self.reserve()
  a=ledger.read(self.aref);a['campaign_root']=str(self.root/'other');self.aref=self.write('external-ledger/anchor.json',a)
  with self.assertRaisesRegex(ValueError,'Cross-campaign'):ledger._anchor(self.aref)
 def test_real_short_parent_resume_complete_keeps_model_and_full_budget(self):
  self.admit();self.anchor();r1=self.reserve();first=self.f.run_parent()
  self.assertEqual(first['queue_result']['status'],'TIME_LIMIT',first)
  q1=first['queue_result']['cumulative_plan_seconds']['tiny.saa'];t,c=self.terminal(q1+0.75)
  s1=ledger.settle_recovery_parent(r1,t,c);self.assertAlmostEqual(ledger.read(s1)['extra_parent_seconds'],.75)
  old=ledger.read(batch.file_reference(self.f.queue/'saa__attempt_0001/worker_checkpoint.json'))
  self.f.parent_context.__exit__(None,None,None)
  self.admit(2,timeout=18.);r2=self.reserve(opening=c);second=self.f.run_parent()
  self.assertEqual(second['queue_result']['status'],'COMPLETE_BATCH',second)
  q2=second['queue_result']['cumulative_plan_seconds']['tiny.saa'];t2,c2=self.terminal(q2-q1+.4,index=2)
  s2=ledger.settle_recovery_parent(r2,t2,c2);v=ledger.read(s2)
  self.assertAlmostEqual(v['cumulative_extra_seconds'],1.15);self.assertAlmostEqual(v['effective_spent_seconds'],q2+1.15)
  current=ledger.read(batch.file_reference(self.f.queue/'saa__attempt_0002/worker_checkpoint.json'))
  self.assertEqual(current['model_manifest_reference'],old['model_manifest_reference'])
  result=ledger.read(batch.file_reference(self.f.queue/'saa__attempt_0002/worker_result.json'))
  gate=ledger.verify_settled_recovery_evidence(self.aref,s2,result['validation_recovery_execution_reference'])
  self.assertTrue(gate['passed']);self.assertFalse(gate['scientific_acceptance'])
  candidate=batch.file_reference(self.f.queue/'saa__attempt_0002/candidate_solver_record.json')
  seal=ledger.read(batch.file_reference(self.f.queue/'saa__attempt_0002/finalization.json'))
  joined=ledger.verify_recovery_candidate_budget(self.aref,s2,candidate,final_solver_record_reference=seal['references']['solver_record'])
  self.assertEqual(joined['candidate_reference'],candidate)
  explicit=ledger.verify_recovery_candidate_budget(self.aref,s2,candidate,execution_result_validator=recovery.adapter.validate_execution_result)
  self.assertEqual(explicit['candidate_reference'],candidate)
  with self.assertRaisesRegex(ValueError,'validator source'):ledger.verify_recovery_candidate_budget(self.aref,s2,candidate,execution_result_validator=lambda *args: {})
  wrong=ledger.read(candidate);wrong['plan_id']='foreign';wrongref=self.write('wrong-candidate.json',wrong)
  with self.assertRaisesRegex(ValueError,'Foreign recovery candidate'):ledger.verify_recovery_candidate_budget(self.aref,s2,wrongref)
  with self.assertRaisesRegex(ValueError,'latest settled'):ledger.verify_settled_recovery_evidence(self.aref,s1,result['validation_recovery_execution_reference'])
  # Old settlements remain idempotent after later queue work.
  self.assertEqual(ledger.settle_recovery_parent(r1,t,c),s1)

 def test_nonzero_original_queue_charge_is_retained_not_recharged(self):
  self.admit();first=self.f.run_parent()
  self.assertEqual(first['queue_result']['status'],'TIME_LIMIT')
  opening=first['queue_result']['cumulative_plan_seconds']['tiny.saa'];self.assertGreater(opening,0.)
  self.f.parent_context.__exit__(None,None,None)
  self.admit(2,timeout=18.);self.anchor();r=self.reserve();t,c=self.terminal(1.25,state='FAILED')
  settled=ledger.read(ledger.settle_recovery_parent(r,t,c))
  self.assertEqual(settled['new_queue_charge_seconds'],0.)
  self.assertEqual(settled['effective_spent_seconds'],opening+1.25)
  self.assertEqual(settled['cumulative_extra_seconds'],1.25)
 def test_exclusive_ledger_blocks_new_anchor_reset(self):
  self.admit();self.anchor();self.reserve()
  other=self.root/'second-ledger';other.mkdir();a=ledger.read(self.aref)
  a['ledger_root']=str(other);a.pop('independent_review_reference')
  a['independent_review_reference']=self.write('second-review.json',dict(kind='SAA_RECOVERY_LEDGER_ANCHOR_REVIEW_V1',passed=True,anchor_payload_sha256=canonical_hash(a)))
  second=self.write('second-ledger/anchor.json',a)
  with self.assertRaisesRegex(ValueError,'already claimed'):ledger.claim_recovery_ledger(second)
 def test_duplicate_actual_receipt_and_tampered_settlement_refused(self):
  self.admit();self.anchor();r=self.reserve();out=self.f.run_parent()
  q=out['queue_result']['cumulative_plan_seconds']['tiny.saa'];t,c=self.terminal(q+.5)
  snapshot=ledger.read(c);snapshot['attempt_receipts'].append(snapshot['attempt_receipts'][0])
  duplicate=self.write('duplicate-closing.json',snapshot)
  with self.assertRaisesRegex(ValueError,'Duplicate queue'):ledger.settle_recovery_parent(r,t,duplicate)
  sr=ledger.settle_recovery_parent(r,t,c);v=ledger.read(sr);v['extra_parent_seconds']=0.
  Path(sr['path']).write_text(json.dumps(v))
  with self.assertRaisesRegex(ValueError,'arithmetic'):ledger._chain(self.aref)
 def test_overrun_is_preserved_as_charge_and_blocks_acceptance(self):
  self.admit();self.anchor();r=self.reserve();t,c=self.terminal(31.,state='TIMED_OUT')
  s=ledger.settle_recovery_parent(r,t,c);v=ledger.read(s)
  self.assertEqual(v['effective_spent_seconds'],31.);self.assertFalse(v['budget_within_limit'])
  with self.assertRaisesRegex(ValueError,'budget exceeded'):ledger.verify_settled_recovery_evidence(self.aref,s,{})

 def enable_hook(self,index=1):
  prior=ledger._chain(self.aref)[2] if (self.folder/'reservation-0001.json').exists() else None
  a=self.f.a;a['external_ledger']=dict(anchor_reference=self.aref,module_reference=batch.file_reference(ledger.__file__),parent_job_request_path=self.jref['path'],expected_last_settlement_reference=prior)
  a.pop('independent_review_reference')
  review=dict(kind='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1',passed=True,admission_payload_sha256=canonical_hash(a),adapter_reference=a['adapter_reference'],validator_reference=a['validator_reference'],external_ledger_anchor_reference=self.aref,external_ledger_module_reference=batch.file_reference(ledger.__file__))
  a['independent_review_reference']=self.write(f'hook-review-{index}.json',review)
  self.f.ref=self.write(f'hook-admission-{index}.json',a);self.admission=self.f.ref
  manifest=ledger.read(self.mref);manifest['argv']=[a['original_python'],a['adapter_reference']['path'],'parent','--admission',self.f.ref['path'],'--expected-sha256',self.f.ref['sha256']]
  Path(self.mref['path']).write_text(json.dumps(manifest));self.mref=batch.file_reference(self.mref['path']);self.jref=self.write(f'job-request-{index}.json',manifest)
 def test_hook_actual_child_resume_and_terminal_candidate_join(self):
  self.admit();self.anchor();self.enable_hook();first=self.f.run_parent()
  self.assertEqual(first['queue_result']['status'],'TIME_LIMIT')
  r1=batch.file_reference(self.folder/'reservation-0001.json');q1=first['queue_result']['cumulative_plan_seconds']['tiny.saa']
  t,c=self.terminal(q1+.3)
  firstcard=recovery.adapter.settle_and_validate_terminal(self.f.ref,r1,t,c,output_path=self.root/'terminal-binding-1.json')
  self.assertEqual(ledger.read(firstcard)['status'],'SETTLED_NO_COMPLETE_CANDIDATE')
  self.f.parent_context.__exit__(None,None,None)
  self.admit(2,timeout=18.);self.enable_hook(2);second=self.f.run_parent()
  self.assertEqual(second['queue_result']['status'],'COMPLETE_BATCH')
  q2=second['queue_result']['cumulative_plan_seconds']['tiny.saa'];t2,c2=self.terminal(q2-q1+.2,index=2)
  r2=batch.file_reference(self.folder/'reservation-0002.json')
  final=recovery.adapter.settle_and_validate_terminal(self.f.ref,r2,t2,c2,output_path=self.root/'terminal-binding-2.json')
  card=ledger.read(final);self.assertEqual(len(card['candidate_final_budget_bindings']),1)
  self.assertFalse(card['scientific_acceptance']);self.assertAlmostEqual(ledger.read(card['settlement_reference'])['effective_spent_seconds'],q2+.5)
  result=ledger.read(batch.file_reference(self.f.queue/'saa__attempt_0002/worker_result.json'))
  receipt=ledger.read(result['validation_recovery_execution_reference']);self.assertEqual(receipt['external_parent_reservation_reference'],r2)
 def test_hook_prequeue_failure_reservation_survives_and_blocks_retry(self):
  self.admit();self.anchor();self.enable_hook()
  with patch.object(recovery.adapter,'admission',side_effect=ValueError('injected prequeue admission failure')):
   with self.assertRaisesRegex(ValueError,'injected'):self.f.run_parent()
  self.assertTrue((self.folder/'reservation-0001.json').exists())
  with self.assertRaisesRegex(ValueError,'Unsettled'):self.f.run_parent()
  t,c=self.terminal(2.,state='FAILED')
  r=batch.file_reference(self.folder/'reservation-0001.json');out=recovery.adapter.settle_and_validate_terminal(self.f.ref,r,t,c,output_path=self.root/'failed-terminal-binding.json')
  self.assertEqual(ledger.read(ledger.read(out)['settlement_reference'])['extra_parent_seconds'],2.)
 def test_formal_boolean_without_typed_source_review_remains_closed(self):
  self.admit();self.anchor();self.f.a['domain']='formal';self.f.a['formal_enabled']=True
  ref=self.write('formal-boolean.json',self.f.a)
  with self.assertRaisesRegex(ValueError,'fee ledger'):recovery.adapter.admission(ref)

 def test_unreviewed_module_is_rejected_before_any_loader_execution(self):
  self.admit();self.anchor();self.enable_hook()
  value=dict(self.f.a);review=ledger.read(value['independent_review_reference']);review['admission_payload_sha256']='0'*64
  value['independent_review_reference']=self.write('unreviewed-hook.json',review)
  with patch.object(recovery.adapter.importlib.util,'spec_from_file_location') as loader:
   with self.assertRaisesRegex(ValueError,'before loading'):recovery.adapter.load_external_ledger(value)
   loader.assert_not_called()

 def test_deleted_entire_settled_tail_cannot_reset_admitted_head(self):
  self.admit();self.anchor();self.enable_hook()
  with patch.object(recovery.adapter,'admission',side_effect=ValueError('prequeue failure')):
   with self.assertRaisesRegex(ValueError,'prequeue'):self.f.run_parent()
  r=batch.file_reference(self.folder/'reservation-0001.json');t,c=self.terminal(2.,state='FAILED')
  ledger.settle_recovery_parent(r,t,c)
  self.f.parent_context.__exit__(None,None,None);self.admit(2,timeout=18.);self.enable_hook(2)
  (self.folder/'reservation-0001.json').unlink();(self.folder/'settlement-0001.json').unlink()
  with self.assertRaisesRegex(ValueError,'head differs'):self.f.run_parent()

if __name__=='__main__':unittest.main()
