"""Independent counterexamples for supplemental fee arithmetic and next grant."""
import unittest
from evcs_experiments import saa_recovery_ledger as ledger
import test_saa_recovery_ledger as author

class RecoveryLedgerIndependentTests(unittest.TestCase):
 def setUp(self):
  self.f=author.LedgerTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
 def test_reserved_queue_closing_allocation_is_not_charged_again(self):
  f=self.f;f.admit();f.anchor();reservation=f.reserve();output=f.f.run_parent()
  q=output['queue_result']['cumulative_plan_seconds']['tiny.saa']
  self.assertEqual(output['queue_result']['status'],'TIME_LIMIT');self.assertGreater(q,1.)
  terminal,closing=f.terminal(q-.25)
  settled=ledger.read(ledger.settle_recovery_parent(reservation,terminal,closing))
  self.assertEqual(settled['extra_parent_seconds'],0.);self.assertEqual(settled['effective_spent_seconds'],q)
  self.assertEqual(settled['new_queue_charge_seconds'],q)
 def test_previous_external_fee_tightens_next_actual_parent_grant(self):
  f=self.f;f.admit();f.anchor();reservation=f.reserve();terminal,closing=f.terminal(2.113231,state='FAILED')
  ledger.settle_recovery_parent(reservation,terminal,closing)
  f.f.parent_context.__exit__(None,None,None)
  # Within raw queue30s but exceeds effective27.886769s after the failed parent.
  f.admit(2,timeout=28.)
  with self.assertRaisesRegex(ValueError,'effective remaining'):f.reserve(opening=closing)
  self.assertEqual(len(list(f.folder.glob('reservation-*.json'))),1)
 def test_receipt_free_failure_cannot_settle_again_with_shorter_parent_wall(self):
  f=self.f;f.admit();f.anchor();reservation=f.reserve();terminal,closing=f.terminal(1.75,state='FAILED')
  first=ledger.settle_recovery_parent(reservation,terminal,closing)
  # A different strong-ref representation of a mutable terminal file cannot erase debt.
  value=ledger.read(terminal);value['elapsed_seconds']=0.;new=f.write(str(f.f.parent_directory.relative_to(f.root)/'result.json'),value)
  with self.assertRaises(ValueError):ledger.settle_recovery_parent(reservation,new,closing)
  self.assertEqual(ledger.read(first)['extra_parent_seconds'],1.75)
