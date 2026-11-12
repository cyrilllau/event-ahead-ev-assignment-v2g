"""Reviewed terminal-marker rearm preserves the append-only numerical ledger."""
import json,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import saa_recovery_ledger as l,campaign_batch as b
from evcs_experiments.random_streams import canonical_hash
import test_saa_recovery_ledger as f

class RearmTests(f.LedgerTests):
 def setup_rearm(self):
  self.admit();self.anchor();r=self.reserve()
  inv=self.write('failed-invocation.json',dict(parent_job=dict(manifest_reference=self.mref,job_id=l.read(self.mref)['job_id'])))
  s=b.read_json(self.f.queue/'state.json');s.update(active=dict(invocation_reference=inv,process={'pid':999999999}),halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
  self.write(str((self.f.queue/'state.json').relative_to(self.root)),s)
  t,c=self.terminal(2.,state='FAILED');settled=l.settle_recovery_parent(r,t,c)
  ready=self.write('ready.json',dict(s,active=None,halted=None))
  payload=dict(kind='SAA_SETTLED_QUEUE_REARM_V1',anchor_reference=self.aref,settlement_reference=settled,old_snapshot_reference=c,ready_snapshot_reference=ready,terminal_result_reference=t,terminal_manifest_reference=self.mref)
  review=self.write('rearm-review.json',dict(kind='SAA_SETTLED_QUEUE_REARM_REVIEW_V1',passed=True,rearm_payload_sha256=canonical_hash(payload)))
  rr=self.write('rearm.json',dict(payload,independent_review_reference=review));return rr,settled,c,s
 def test_rearm_keeps_history_extra_and_snapshot_bytes(self):
  rr,settled,old,s=self.setup_rearm();before=Path(old['path']).read_bytes()
  with patch.object(b,'identity_alive',return_value=False):l.apply_queue_rearm(rr,self.aref)
  self.assertEqual(b.read_json(self.f.queue/'state.json'),dict(s,active=None,halted=None))
  self.assertEqual(Path(old['path']).read_bytes(),before);self.assertEqual(l._chain(self.aref)[1],2.)
  with patch.object(b,'identity_alive',return_value=False),self.assertRaisesRegex(ValueError,'no repeat'):l.apply_queue_rearm(rr,self.aref)
 def test_changed_receipts_or_live_process_or_head_rejected(self):
  rr,settled,old,s=self.setup_rearm()
  with patch.object(b,'identity_alive',return_value=True),self.assertRaisesRegex(ValueError,'still alive'):l.verify_queue_rearm(rr,self.aref,settled)
  with self.assertRaisesRegex(ValueError,'scope/head'):l.verify_queue_rearm(rr,self.aref,None)
  v=l.read(rr);ready=l.read(v['ready_snapshot_reference']);ready['runs']=[] if s['runs'] else [dict(foreign=True)]
  v['ready_snapshot_reference']=self.write('bad-ready.json',ready);payload={k:w for k,w in v.items() if k!='independent_review_reference'}
  v['independent_review_reference']=self.write('bad-review.json',dict(kind='SAA_SETTLED_QUEUE_REARM_REVIEW_V1',passed=True,rearm_payload_sha256=canonical_hash(payload)))
  with self.assertRaisesRegex(ValueError,'beyond terminal'):l.verify_queue_rearm(self.write('bad-rearm.json',v),self.aref,settled)
 def test_rearmed_next_reservation_keeps_old_opening_and_all_extra(self):
  rr,settled,old,s=self.setup_rearm()
  with patch.object(b,'identity_alive',return_value=False):l.apply_queue_rearm(rr,self.aref)
  self.f.parent_context.__exit__(None,None,None);self.admit(2,timeout=9.)
  a=l.read(self.admission);a.update(queue_rearm_reference=rr,baseline_ledger_reference=old)
  rv=l.read(a['independent_review_reference']);rv['admission_payload_sha256']=canonical_hash({k:v for k,v in a.items() if k!='independent_review_reference'})
  a['independent_review_reference']=self.write('new-review.json',rv);self.admission=self.write('new-admission.json',a)
  # The actual manifest must bind exactly the new immutable admission.
  manifest=l.read(self.mref)
  manifest['argv']=[a['original_python'],a['adapter_reference']['path'],'parent','--admission',self.admission['path'],'--expected-sha256',self.admission['sha256']]
  self.mref=self.write(str(Path(self.mref['path']).relative_to(self.root)),manifest);self.jref=self.write('new-job.json',manifest)
  with patch.object(b,'identity_alive',return_value=False):reservation=self.reserve(opening=old)
  record=l.read(reservation);self.assertEqual(record['opening_queue_snapshot_reference'],old);self.assertEqual(record['opening_extra_seconds'],2.)

 def test_source_upgrade_requires_exact_original_and_reviewed_new_bytes(self):
  self.admit();self.anchor();a=l.read(self.aref)
  prior=self.root/'prior_ledger.py';prior.write_text('# immutable prior source')
  a['ledger_source_reference']=b.file_reference(prior)
  a['independent_review_reference']=self.write('prior-anchor-review.json',dict(kind='SAA_RECOVERY_LEDGER_ANCHOR_REVIEW_V1',passed=True,anchor_payload_sha256=canonical_hash({k:v for k,v in a.items() if k!='independent_review_reference'})))
  self.aref=self.write('external-ledger/anchor.json',a)
  payload=dict(kind='SAA_LEDGER_SOURCE_UPGRADE_V1',anchor_reference=self.aref,original_source_reference=a['ledger_source_reference'],upgraded_source_reference=b.file_reference(l.__file__))
  reviewed=self.write('upgrade-review.json',dict(kind='SAA_LEDGER_SOURCE_UPGRADE_REVIEW_V1',passed=True,upgrade_payload_sha256=canonical_hash(payload)))
  upgrade=self.write('upgrade.json',dict(payload,independent_review_reference=reviewed))
  with patch.object(l,'SOURCE_UPGRADE_REFERENCE',upgrade):
   self.assertEqual(l._anchor(self.aref)['ledger_source_reference'],b.file_reference(prior))
   prior.write_text('tampered original')
   with self.assertRaises(ValueError):l._anchor(self.aref)
