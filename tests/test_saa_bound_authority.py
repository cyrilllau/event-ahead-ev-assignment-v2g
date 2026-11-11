"""Exercise real ledger failure recovery and crash replay, without new solves."""
import unittest
from pathlib import Path
from unittest.mock import patch
import test_saa_bound_parent_ledger as fixtures
from evcs_experiments import saa_bound_authority as authority, saa_recovery_ledger as ledger


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.BoundLedgerTests(methodName='runTest')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.f = self.base.f
        ledger.claim_recovery_ledger(self.f.aref)
        manifest = ledger.read(self.f.mref)
        allowed = ('job_id', 'argv', 'cwd', 'outroot', 'threads', 'timeout_seconds', 'grace_seconds')
        self.f.jref = self.f.write('sealed-job.json', {k: manifest[k] for k in allowed})
        self.state = dict(active_jobs=[], saa_recovery_external_ledger=dict(
            anchor_reference=self.f.aref, latest_settlement_reference=None,
            effective_spent_seconds=0., pending_job_id=None))

    def register(self):
        return authority.registration(self.state, self.f.aref, self.f.admission,
                                      self.f.jref, ledger=ledger)

    def test_failure_before_reservation_charged_once_and_crash_replay(self):
        state = self.register()
        _, close = self.f.terminal(2.25, state='FAILED')
        updated, settled = authority.terminal(state, close, ledger=ledger)
        self.assertEqual(updated['saa_recovery_external_ledger']['effective_spent_seconds'], 2.25)
        self.assertFalse(updated['saa_bound_only_terminal']['scientific_acceptance'])
        self.assertEqual(authority.terminal(state, close, ledger=ledger), (updated, settled))
        self.assertEqual(len(list(self.f.folder.glob('settlement-*.json'))), 1)

    def test_normal_reservation_settles_without_new_queue_work(self):
        state = self.register()
        self.f.reserve()
        _, close = self.f.terminal(1.)
        updated, _ = authority.terminal(state, close, ledger=ledger)
        self.assertEqual(updated['active_jobs'], [])
        self.assertEqual(updated['saa_recovery_external_ledger']['queue_cumulative_seconds'], 0.)

    def test_pending_or_wrong_budget_blocks_registration(self):
        self.state['active_jobs'] = ['another']
        with self.assertRaisesRegex(ValueError, 'blocks dispatch'):
            self.register()
        self.state['active_jobs'] = []
        self.state['saa_recovery_external_ledger']['effective_spent_seconds'] = 1.
        with self.assertRaisesRegex(ValueError, 'budget differs'):
            self.register()

    def test_live_parent_cannot_be_settled_or_clear_debt(self):
        state = self.register()
        _, close = self.f.terminal(1.)
        with patch.object(authority.jobctl, 'observation', return_value=dict(blocks_new_launch=True)):
            with self.assertRaisesRegex(ValueError, 'confirmed terminal'):
                authority.terminal(state, close, ledger=ledger)
        self.assertFalse(list(self.f.folder.glob('reservation-*.json')))

    def test_modified_queue_cannot_be_accepted(self):
        state = self.register()
        self.f.reserve()
        _, close = self.f.terminal(1.)
        changed = ledger.read(close)
        changed['unexpected'] = True
        close = self.f.write('changed-closing.json', changed)
        with self.assertRaisesRegex(ValueError, 'changed original queue'):
            authority.terminal(state, close, ledger=ledger)
        self.assertFalse(list(self.f.folder.glob('settlement-*.json')))
