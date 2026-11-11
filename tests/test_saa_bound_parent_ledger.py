"""Bound-only parent admission uses the existing immutable debt ledger."""
import json
import unittest
from pathlib import Path
import test_saa_recovery_ledger as fixtures
from evcs_experiments import saa_recovery_ledger as ledger,campaign_batch as batch
from evcs_experiments.random_streams import canonical_hash

class BoundLedgerTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.LedgerTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.f.admit();self.f.anchor();self.make_admission()
 def make_admission(self,**changes):
  f=self.f;old=ledger.read(f.admission);self.old=old
  req=dict(kind='S05_SAA_SAVED_DUAL_BOUND_ONLY_V1',domain=f.f.q['domain'],opening_effective_seconds=0.,solver_seconds=1.,post_master_reserve_seconds=100.,wall_seconds=5.)
  # This is admission-only: its short grant deliberately cannot enter a numerical master.
  rr=f.write('bound-request.json',req)
  a=dict(kind='SAA_SAVED_DUAL_BOUND_ADMISSION_V1',domain=req['domain'],queue_reference=f.f.qref,
   queue_root=str(f.f.queue),parent_job_id=f.f.parent_directory.name,parent_job_directory=str(f.f.parent_directory),
   opening_cumulative_seconds=0.,opening_effective_seconds=0.,expected_last_settlement_reference=None,
   baseline_ledger_reference=f.opening,baseline_receipt_references=[],driver_reference=batch.file_reference(__file__),
   request_reference=rr,original_python=old['original_python'])
  a.update(changes)
  review=dict(kind='SAA_SAVED_DUAL_BOUND_PRODUCTION_REVIEW_V1',passed=True,admission_payload_sha256=canonical_hash(a),driver_reference=a['driver_reference'],request_reference=a['request_reference'])
  a['independent_review_reference']=f.write('bound-review.json',review);f.admission=f.write('bound-admission.json',a)
  m=ledger.read(f.mref);m['argv']=[a['original_python'],a['driver_reference']['path'],'--request',rr['path'],'--expected-sha256',rr['sha256'],'--admission',f.admission['path'],'--admission-sha256',f.admission['sha256']]
  Path(f.mref['path']).write_text(json.dumps(m));f.mref=batch.file_reference(f.mref['path']);f.jref=f.write('bound-job-request.json',m)
 def test_whole_parent_extra_once_without_queue_receipt(self):
  r=self.f.reserve();t,c=self.f.terminal(2.25);s=ledger.settle_recovery_parent(r,t,c);value=ledger.read(s)
  self.assertEqual(value['new_queue_charge_seconds'],0.)
  self.assertEqual(value['effective_spent_seconds'],2.25)
  self.assertEqual(ledger.settle_recovery_parent(r,t,c),s)
 def test_unsettled_bound_parent_blocks_restart(self):
  self.f.reserve()
  with self.assertRaisesRegex(ValueError,'Unsettled'):self.f.reserve()
 def test_resealed_wrong_opening_or_head_rejected(self):
  for change in ({'opening_effective_seconds':.5},{'expected_last_settlement_reference':self.f.opening}):
   with self.subTest(change=change):
    self.make_admission(**change)
    with self.assertRaisesRegex(ValueError,'reset original queue'):self.f.reserve()
 def test_actual_argv_and_expanded_timeout_rejected(self):
  for changes,pattern in (({'argv':['wrong']},'actual argv'),({'timeout_seconds':30.},'remaining allowance')):
   self.make_admission();m=ledger.read(self.f.mref);m.update(changes)
   Path(self.f.mref['path']).write_text(json.dumps(m));self.f.mref=batch.file_reference(self.f.mref['path']);self.f.jref=self.f.write('bound-job-request.json',m)
   with self.assertRaisesRegex(ValueError,pattern):self.f.reserve()
