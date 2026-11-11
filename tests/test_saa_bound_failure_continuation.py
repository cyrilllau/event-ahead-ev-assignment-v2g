import copy
import json
import tempfile
import unittest
from pathlib import Path
from tests.test_saa_saved_dual_bound_controller import d
from evcs_experiments.campaign_batch import file_reference


class FailureContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        refs = {k: self.write(k, {}) for k in ('candidate_audit', 'fixed_q_request',
            'fixed_q_checkpoint', 'template_envelope', 'first_stage', 'final_budget_join',
            'final_acceptance_review', 'final_record', 'settlement', 'ledger_anchor')}
        self.old = dict(kind='S05_SAA_SAVED_DUAL_BOUND_ONLY_V1', domain='formal',
                        references=refs, output_directory=str(self.root))
        previous = self.write('old-request', self.old)
        admission = self.write('admission', dict(request_reference=previous))
        reservation = self.write('reservation', dict(admission_reference=admission,
            previous_settlement_reference=refs['settlement'], anchor_reference=refs['ledger_anchor'],
            effective_spent_seconds=5541.))
        terminal = self.write('terminal', dict(complete=True, state='FAILED'))
        self.settlement = dict(reservation_reference=reservation,
            terminal_job_result_reference=terminal, new_queue_charge_seconds=0,
            effective_spent_seconds=5581., actual_parent_wall_seconds=40.)
        self.last = self.write('latest-settlement', self.settlement)
        failure = self.write('failure', dict(type='ValueError', error='Original full Q checkpoint differs'))
        self.card = dict(kind='SAA_BOUND_PRE_NATIVE_FAILURE_CONTINUATION_REVIEW_V1', passed=True,
            pre_native_failure=True, settlement_reference=self.last,
            previous_request_reference=previous, failure_reference=failure)
        self.req = dict(self.old, opening_effective_seconds=5581.,
                        failed_parent_continuation=self.write('review', self.card))
        self.join = dict(effective_spent_seconds=5541.)

    def write(self, name, value):
        path = self.root / (name + '.json')
        path.write_text(json.dumps(value))
        return file_reference(path)

    def test_pre_native_failed_parent_retains_full_cost(self):
        d.continuation_budget(self.req, self.join, self.last)

    def test_reset_or_wrong_original_evidence_rejected(self):
        req = copy.deepcopy(self.req)
        req['opening_effective_seconds'] = 5541.
        with self.assertRaisesRegex(ValueError, 'cost reset'):
            d.continuation_budget(req, self.join, self.last)
        req = copy.deepcopy(self.req)
        req['references']['candidate_audit'] = self.write('foreign', {})
        with self.assertRaisesRegex(ValueError, 'original evidence changed'):
            d.continuation_budget(req, self.join, self.last)

    def test_native_started_or_cut_output_blocks_narrow_retry(self):
        for name in ('native_started.json', 'cut_000.json'):
            path = self.root / name
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'scope changed'):
                d.continuation_budget(self.req, self.join, self.last)
            path.unlink()

    def test_missing_independent_review_or_wrong_head_rejected(self):
        card = dict(self.card, passed=False)
        self.req['failed_parent_continuation'] = self.write('bad-review', card)
        with self.assertRaisesRegex(ValueError, 'review required'):
            d.continuation_budget(self.req, self.join, self.last)
        self.req['failed_parent_continuation'] = self.write('good-review', self.card)
        with self.assertRaisesRegex(ValueError, 'history differs'):
            d.continuation_budget(self.req, self.join, self.req['references']['settlement'])
