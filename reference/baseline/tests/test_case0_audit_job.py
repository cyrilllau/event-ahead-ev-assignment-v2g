import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.case0_audit_job import settle_audit_job, _admit_phase
from evcs_experiments.runtime.jobctl import atomic_json, read_json


class AuditTerminalAccountingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.op = self.root / 'operations' / 'audit1'
        self.job = self.root / 'jobs' / 'audit1'
        self.op.mkdir(parents=True); self.job.mkdir(parents=True)
        atomic_json(self.job / 'manifest.json', {'job_id': 'audit1'})
        atomic_json(self.job / 'result.json', {'complete': True, 'state': 'TIMED_OUT', 'elapsed_seconds': 601.125})
        atomic_json(self.job / 'status.json', {'state': 'TIMED_OUT'})
        atomic_json(self.op / 'reservation.json', {
            'kind': 'CASE0_AUDIT_RESERVATION_V1', 'operation_root': str(self.op),
            'job_id': 'audit1', 'opening_compute_seconds': 2570.898,
            'parent_manifest_reference': file_reference(self.job / 'manifest.json')})
        self.ref = file_reference(self.op / 'reservation.json')
        atomic_json(self.root / 'state.json', {'compute_seconds': 2570.898,
            'active_job_id': 'audit1', 'audit_reservation_reference': self.ref})
        self.live = patch('evcs_experiments.case0_audit_job.observation', return_value={'blocks_new_launch': False})
        self.live.start(); self.addCleanup(self.live.stop)

    def test_hard_kill_without_child_finally_is_charged_once(self):
        ref = settle_audit_job(self.root, self.ref)
        state = read_json(self.root / 'state.json')
        self.assertAlmostEqual(state['compute_seconds'], 3172.023)
        self.assertIsNone(state['audit_reservation_reference'])
        self.assertEqual(settle_audit_job(self.root, self.ref), ref)
        self.assertEqual(read_json(self.root / 'state.json')['compute_seconds'], state['compute_seconds'])

    def test_crash_after_receipt_before_state_reconciles_once(self):
        before = read_json(self.root / 'state.json')
        settle_audit_job(self.root, self.ref)
        atomic_json(self.root / 'state.json', before)
        settle_audit_job(self.root, self.ref)
        self.assertAlmostEqual(read_json(self.root / 'state.json')['compute_seconds'], 3172.023)

    def test_live_or_unsettled_parent_cannot_release_reservation(self):
        with patch('evcs_experiments.case0_audit_job.observation', return_value={'blocks_new_launch': True}):
            with self.assertRaisesRegex(ValueError, 'live'):
                settle_audit_job(self.root, self.ref)
        self.assertEqual(read_json(self.root / 'state.json')['compute_seconds'], 2570.898)

    def test_changed_parent_or_concurrent_budget_fails_closed(self):
        atomic_json(self.job / 'manifest.json', {'job_id': 'different'})
        with self.assertRaisesRegex(ValueError, 'Frozen reference'):
            settle_audit_job(self.root, self.ref)
        self.assertFalse((self.op / 'terminal_accounting.json').exists())

    def test_incomplete_parent_cannot_charge_or_clear(self):
        atomic_json(self.job / 'result.json', {'complete': False})
        with self.assertRaisesRegex(ValueError, 'not terminal'):
            settle_audit_job(self.root, self.ref)
        self.assertEqual(read_json(self.root / 'state.json')['active_job_id'], 'audit1')

    def test_completed_oos_must_have_parent_settlement_before_audit(self):
        state = dict(active_configuration=None, evaluation_opened=True,
            status='CASE0_EVALUATION_COMPLETE_PENDING_REVIEW')
        with self.assertRaisesRegex(ValueError, 'settled'):
            _admit_phase(state, 'evaluation')
        state['last_evaluation_settlement'] = {'path': 'reviewed-terminal'}
        _admit_phase(state, 'evaluation')
        for key in ('pending_evaluation_attempt', 'audit_reservation_reference'):
            state[key] = {'path': 'still-unsettled'}
            with self.assertRaisesRegex(ValueError, 'Unsettled'):
                _admit_phase(state, 'evaluation')
            state[key] = None
        with self.assertRaisesRegex(ValueError, 'Planning state'):
            _admit_phase(state, 'planning')


if __name__ == '__main__':
    unittest.main()
