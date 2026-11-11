"""Original real L-shaped child/queue with typed parent/ledger integration.

The Mac development fixture explicitly mocks Linux supervision metadata. Actual
queue workers are subprocesses and use the original physical/numerical fixture;
there is no actual Linux jobctl or licensed Gurobi admission in these tests.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from evcs_experiments import campaign_batch as batch, saa_recovery_ledger as ledger
from evcs_experiments import saa_l_shaped_parent as parent, saa_l_shaped_failure_debt as debt
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import THREAD_VARS


def fixture_type():
    path = get_paths().project_root/'tests/test_saa_l_shaped_queue.py'
    spec = importlib.util.spec_from_file_location('original_l_shaped_queue_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LShapedQueueIntegrationTests


class ParentTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get('EVCS_SAA_PARENT_TEST_ARTIFACTS_ROOT')
        if retained:
            Path(retained).mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix='parent_', dir=retained)).resolve()
        else:
            temporary = tempfile.TemporaryDirectory(prefix='saa-l-shaped-parent-')
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name).resolve()
        fixture = fixture_type()(methodName='runTest')
        fixture.root = self.root
        fixture.project = get_paths().project_root
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.workerref, original, self.plan_id = fixture.prepare()
        queue = parent.read(original)
        worker = parent.read(self.workerref)
        for key in ('model_id','spec_id','config_reference','bank_acceptance_reference','summary_reference','audit_reference'):
            worker.setdefault(key, 'IEEE33-v54-Q1' if key=='model_id' else 'S05' if key=='spec_id' else {})
        debtref = self.failure_debt(worker, queue['queue_root'])
        worker['failure_carryover_reference'] = debtref
        debt.claim_failure_debt(debtref, worker, queue['queue_root'])
        self.workerref = self.put('indebted_worker_request.json', worker)
        queue['items'][0]['request_reference'] = self.workerref
        queue['items'][0]['failure_carryover_reference'] = debtref
        queue['parent_wall_policy'] = 'verified_parent_remaining_v1'
        self.queueref = self.put('parent_queue_request.json', queue)
        self.queue = queue
        self.queue_root = Path(queue['queue_root'])
        self.queue_root.mkdir()
        initial = dict(schema_version=1, request_reference=self.queueref, attempt_receipts=[],
                       active=None, halted=None, runs=[])
        self.put('queue/state.json', initial)
        self.opening = self.put('initial_queue_snapshot.json', initial)
        folder = self.root/'ledger'
        folder.mkdir()
        anchor = dict(kind=ledger.ANCHOR, ledger_root=str(folder), campaign_root=str(self.root),
            queue_reference=self.queueref, plan_id=self.plan_id, budget_seconds=30.,
            initial_queue_snapshot_reference=self.opening, initial_queue_seconds=6.,
            ledger_source_reference=batch.file_reference(ledger.__file__))
        anchor['independent_review_reference'] = self.put('anchor_review.json', dict(
            kind='SAA_RECOVERY_LEDGER_ANCHOR_REVIEW_V1', passed=True,
            anchor_payload_sha256=canonical_hash(anchor), synthetic_development_test_review=True))
        self.anchorref = self.put('ledger/anchor.json', anchor)
        ledger.claim_recovery_ledger(self.anchorref)
        self.last = None
        self.extra = 0.
        self.total = 6.

    def put(self, name, data):
        return self.fixture.put(name, data)

    def failure_debt(self, worker, queue_root):
        """Original accounting kinds, deliberately synthetic 2.5 + 3.5 seconds."""
        oldworker = self.put('debt/old_worker.json', dict(worker, model_input_mode='assemble'))
        firstq = self.put('debt/first_queue.json', dict(queue_root=str(self.root/'debt/old_queue'),
            items=[dict(request_reference=oldworker, plan_id=self.plan_id, plan_budget_seconds=30.),
                   dict(plan_id='unused.ARO')]))
        firststate = dict(request_reference=firstq, attempt_receipts=[], halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
        self.put('debt/old_queue/state.json', firststate)
        firstsnapshot = self.put('debt/first_snapshot.json', firststate)
        firstterminal = self.put('debt/first_terminal.json', dict(job_id='historical-first', complete=True,
            state='FAILED', returncode=2, elapsed_seconds=2.5))
        fee = self.put('debt/first_fee.json', dict(kind='SAA_ASSEMBLED_MASTER_FAILED_PARENT_EXPENSE_V1',
            passed=True, accounting_PASS=True, status='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION',
            references={'worker_request':oldworker}, original_queue_reference=firstq,
            closing_snapshot_reference=firstsnapshot, terminal_job_result_reference=firstterminal,
            plan_id=self.plan_id, SAA_charged_seconds=2.5, SAA_remaining_seconds=27.5,
            ARO_charged_seconds=0, ARO_no_invocation=True, ARO_no_worker_outputs=True,
            parent_charged_once=True, nested_seconds_added=False, cumulative_accounted_seconds_decimal='2.5',
            opening_cumulative_seconds_decimal='0'))
        isolation = self.put('debt/isolation.json', dict(kind='SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1',
            old_queue_permanently_quarantined=True, original_queue_rearm_prohibited=True,
            terminal_fee_settlement_reference=fee, original_queue_reference=firstq,
            original_queue_snapshot_reference=firstsnapshot))
        secondq = self.put('debt/second_queue.json', dict(queue_root=str(self.root/'debt/recovery_queue'),
            items=[dict(plan_id=self.plan_id, request_reference=oldworker)]))
        secondstate = dict(request_reference=secondq, attempt_receipts=[], halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
        self.put('debt/recovery_queue/state.json', secondstate)
        secondsnapshot = self.put('debt/second_snapshot.json', secondstate)
        terminal = self.put('debt/second_terminal.json', dict(job_id='historical-second', complete=True,
            state='FAILED', returncode=2, elapsed_seconds=3.5))
        old_admission = self.put('debt/old_admission.json', dict(campaign_root=str(self.root),
            queue_reference=secondq, worker_request_reference=oldworker))
        reservation = self.put('debt/old_reservation.json', dict(admission_reference=old_admission, opening=dict(reference=fee, plan_id=self.plan_id,
            accounted_seconds=2.5, budget_seconds=30., remaining_seconds=27.5, domain='development'),
            queue_reference=secondq, worker_request_reference=oldworker,
            parent_job={'job_id':'historical-second','directory':str(self.root/'jobs/historical-second')}))
        expense = self.put('debt/second_fee.json', dict(kind='SAA_SAVED_MODEL_RECOVERY_TERMINAL_EXPENSE_V1',
            passed=False, status='HOLD_FAILED_OR_UNCOVERED_PARENT_EXPENSE_RETAINED', automatic_resume_authorized=False,
            reservation_reference=reservation, terminal_job_result_reference=terminal,
            closing_snapshot_reference=secondsnapshot, original_failed_parent_expense_reference=fee,
            original_spent_seconds=2.5, new_queue_charge_seconds=0., actual_parent_seconds=3.5,
            new_accounted_seconds=3.5, uncovered_parent_seconds=3.5, cumulative_spent_seconds=6.,
            remaining_seconds=24., budget_within_limit=True))
        native = self.put('debt/native_failure.json', dict(native_interface=dict(raw_model_status=17,
            status='MEM_LIMIT', solution_count=0, valid_incumbent=False, solver_bound_without_constant_usd=None)))
        failure = self.put('debt/failure_review.json', dict(kind='SAA_SAVED_MODEL_FAILED_PARENT_FEE_AND_CAUSE_REVIEW_V1',
            passed=True, status='PASS_FAILED_PARENT_EXPENSE_SETTLED_NO_NUMERICAL_RESULT', required_references=[],
            original_worker_reference=oldworker, original_failed_fee_reference=fee,
            original_isolation_decision_reference=isolation, terminal_expense_reference=expense,
            terminal_job_result_reference=terminal, reservation_reference=reservation, recovery_queue_reference=secondq,
            closing_queue_snapshot_reference=secondsnapshot, recovery_worker_reference=oldworker,
            original_accounted_seconds=2.5, new_actual_parent_seconds=3.5, cumulative_accounted_seconds=6.,
            remaining_original_seconds=24., parent_cost_fully_retained=True, receipt_count=0, numerical_acceptance=False,
            raw_native_status=17, native_status='MEM_LIMIT', has_valid_incumbent=False, has_valid_bound=False,
            native_result_reference=native))
        adoption = self.put('debt/adoption.json', dict(kind='SAA_SECOND_MEMORY_FAILURE_TERMINAL_ADOPTION_V1',
            failure_accounting_review_reference=failure, recovery_queue_rearm_prohibited=True,
            repeat_same_request_prohibited=True, formal_SAA_algorithm_admitted=False, plan_id=self.plan_id,
            terminal_expense_reference=expense, cumulative_accounted_seconds=6., remaining_original_seconds=24.))
        envelope = dict(kind=debt.KIND, domain='development', plan_id=self.plan_id, campaign_root=str(self.root),
            module_reference=batch.file_reference(debt.__file__), failure_review_reference=failure,
            terminal_adoption_reference=adoption, successor_queue_root=queue_root,
            successor_worker_identity_sha256=debt.worker_identity(worker))
        envelope['independent_review_reference'] = self.put('debt/successor_review.json', dict(
            kind=debt.REVIEW_KIND, passed=True, payload_sha256=canonical_hash(envelope)))
        return self.put('debt/carryover.json', envelope)

    def admit(self, index=1):
        job_id = f'parent-{index}'
        job = self.root/'jobs'/job_id
        job.mkdir(parents=True)
        self.job = job
        # Path only avoids admission/manifest hash cycle; exact request is still verified.
        request_path = self.root/f'job_request_{index}.json'
        admission = dict(kind=parent.KIND, domain='development', campaign_root=str(self.root),
            plan_id=self.plan_id, parent_job_id=job_id, original_python=os.sys.executable,
            parent_execution_path=str(job/'l_shaped_parent_execution.json'),
            parent_source_reference=batch.file_reference(parent.__file__),
            ledger_source_reference=batch.file_reference(ledger.__file__),
            queue_source_reference=batch.file_reference(batch.__file__),
            queue_reference=self.queueref, worker_request_reference=self.workerref,
            anchor_reference=self.anchorref, opening_queue_snapshot_reference=self.opening,
            expected_last_settlement_reference=self.last, opening_queue_seconds=self.total,
            opening_extra_seconds=self.extra, opening_effective_seconds=self.total+self.extra,
            parent_job_request_path=str(request_path))
        admission['independent_review_reference'] = self.put(f'admission_review_{index}.json', dict(
            kind=parent.REVIEW, passed=True, admission_payload_sha256=canonical_hash(admission),
            synthetic_development_test_review=True))
        self.admission = self.put(f'admission_{index}.json', admission)
        self.argv = parent.parent_argv(self.admission)
        self.timeout = min(20., 30.-self.total-self.extra-.1)
        manifest = dict(job_id=job_id, spec_id='S05', outroot=str(job.parent),
            cwd=str(self.fixture.project), argv=self.argv, timeout_seconds=self.timeout,
            grace_seconds=.01, threads=1)
        self.manifest = self.put(str(job.relative_to(self.root)/'manifest.json'), manifest)
        self.jobref = self.put(f'job_request_{index}.json', manifest)
        self.started = time.monotonic()
        self.original_parent = dict(job_id=job_id, directory=str(job), manifest_reference=self.manifest,
            wall_clock=dict(clock_kind='linux_supervisor_birth_boottime_v1',
                observed_monotonic=self.started, elapsed_seconds=0., timeout_seconds=self.timeout,
                conservative_deadline_monotonic=self.started+self.timeout,
                synthetic_supervisor_metadata=True))

    def run_parent(self):
        with patch.object(parent, 'actual_argv', return_value=self.argv), \
             patch.object(batch, '_parent', return_value=self.original_parent), \
             patch.dict(os.environ, {key: '1' for key in THREAD_VARS}):
            return parent.run_parent(self.admission)

    def settle(self, index, *, state='SUCCEEDED', extra_tail=.25):
        now = time.monotonic()
        terminal = self.put(str(self.job.relative_to(self.root)/'result.json'), dict(
            complete=True, job_id=self.job.name, state=state, returncode=0 if state=='SUCCEEDED' else 2,
            elapsed_seconds=now-self.started+extra_tail, synthetic_jobctl_terminal=True))
        self.put(str(self.job.relative_to(self.root)/'status.json'), dict(job_id=self.job.name, state=state))
        closing = self.put(f'closing_snapshot_{index}.json', batch.read_json(self.queue_root/'state.json'))
        reservation = batch.file_reference(self.root/'ledger'/f'reservation-{index:04d}.json')
        cardref = parent.settle_parent(self.admission, reservation, terminal, closing,
                                     output_path=self.root/f'terminal_budget_{index}.json')
        card = parent.read(cardref)
        settled = parent.read(card['settlement_reference'])
        self.last = card['settlement_reference']
        self.opening = closing
        self.extra = settled['cumulative_extra_seconds']
        self.total = settled['closing_queue_seconds']
        return cardref, card, settled

    def test_real_queue_two_attempts_original_seal_canonical_and_parent_debt(self):
        self.admit()
        first = self.run_parent()
        self.assertEqual(first['status'], 'TIME_LIMIT', first)
        first_execution = parent.read(first['execution_reference'])
        self.assertTrue(first_execution['parent_job']['wall_clock']['synthetic_supervisor_metadata'])
        old_checkpoint = first['queue_result']['checkpoint']
        old_bytes = Path(old_checkpoint['path']).read_bytes()
        _, one, settled_one = self.settle(1, extra_tail=1.25)
        self.assertTrue(one['passed'])
        self.assertEqual(one['status'], 'SETTLED_NO_COMPLETE_CANDIDATE')
        self.assertGreater(self.total, 0.)
        self.assertGreater(self.extra, 0.)
        old_total, old_extra = self.total, self.extra
        self.admit(2)
        second = self.run_parent()
        self.assertEqual(second['status'], 'COMPLETE_BATCH', second)
        _, two, settled_two = self.settle(2)
        self.assertTrue(two['passed'])
        self.assertEqual(two['status'], 'SETTLED_COMPLETE_CANDIDATE')
        self.assertEqual(len(two['complete_candidate_bindings']), 1)
        self.assertEqual(Path(old_checkpoint['path']).read_bytes(), old_bytes)
        self.assertEqual(settled_two['effective_spent_seconds'], self.total+self.extra)
        self.assertGreaterEqual(self.extra, old_extra)
        self.assertEqual(settled_two['new_queue_charge_seconds'], self.total-old_total)
        new_receipt = parent.read(settled_two['new_queue_receipt_references'][0])
        invocation = parent.read(new_receipt['invocation_reference'])
        self.assertEqual(invocation['cumulative_plan_seconds'], old_total)
        self.assertFalse(two['scientific_acceptance'])
        self.put('integration_summary.json', dict(first_parent=one, second_parent=two,
            first_settlement=settled_one, second_settlement=settled_two,
            real_queue_child_processes=True, actual_linux_jobctl_parent=False, actual_gurobi_calls=0,
            formal_integration_passed=False, historical_helene_expenses_admitted=False,
            synthetic_two_failed_parent_seconds=6., failure_debt_reference=parent.read(self.workerref)['failure_carryover_reference'],
            source_references=self.queue['source_references']))

    def test_prequeue_failure_keeps_reservation_and_charges_whole_parent(self):
        self.admit()
        with patch.object(batch, 'run_batch', side_effect=ValueError('injected prequeue failure')):
            with self.assertRaisesRegex(ValueError, 'injected prequeue'):
                self.run_parent()
        self.assertTrue((self.root/'ledger/reservation-0001.json').exists())
        with self.assertRaisesRegex(ValueError, 'unsettled'):
            self.run_parent()
        _, card, settled = self.settle(1, state='FAILED')
        self.assertEqual(settled['new_queue_charge_seconds'], 0.)
        self.assertEqual(settled['extra_parent_seconds'], settled['actual_parent_wall_seconds'])
        self.assertFalse(card['passed'])

    def test_renamed_parent_cannot_hide_unreserved_failed_admission(self):
        self.admit()
        old = parent.read(self.manifest)
        old['job_id'] = 'unreserved-old'
        old['outroot'] = str(self.root/'jobs')
        self.put('jobs/unreserved-old/manifest.json', old)
        with self.assertRaisesRegex(ValueError, 'unreserved parent expenses'):
            self.run_parent()
        self.assertFalse((self.root/'ledger/reservation-0001.json').exists())

    def test_missing_exclusive_failure_claim_rejected_before_dispatch(self):
        self.admit()
        claims = list((self.root/'operations/saa-l-shaped-debt-claims').glob('*.json'))
        self.assertEqual(len(claims), 1)
        claims[0].unlink()
        with self.assertRaisesRegex(ValueError, 'claim missing'):
            self.run_parent()
        self.assertFalse((self.root/'ledger/reservation-0001.json').exists())

    def test_remaining_budget_includes_old_failure_debt_and_exit_grace(self):
        self.admit()
        manifest = parent.read(self.manifest)
        manifest['timeout_seconds'] = 24.
        manifest['grace_seconds'] = .01
        Path(self.manifest['path']).write_text(json.dumps(manifest))
        self.manifest = batch.file_reference(self.manifest['path'])
        Path(self.jobref['path']).write_text(json.dumps(manifest))
        self.original_parent['manifest_reference'] = self.manifest
        with self.assertRaisesRegex(ValueError, 'original remaining budget'):
            self.run_parent()
        self.assertFalse((self.root/'ledger/reservation-0001.json').exists())

    def test_formal_flag_does_not_admit_development_sources(self):
        self.admit()
        value = parent.read(self.admission)
        value['domain'] = 'formal'
        value['formal_enabled'] = True
        value['independent_review_reference'] = self.put('formal_flag_review.json', dict(
            kind=parent.REVIEW, passed=True, admission_payload_sha256=canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'})))
        bad = self.put('formal_flag_admission.json', value)
        with self.assertRaisesRegex(ValueError, 'same single SAA'):
            parent.verify_admission(bad)

    def test_wrong_running_argv_and_missing_linux_clock_rejected(self):
        self.admit()
        with patch.object(parent, 'actual_argv', return_value=['wrong']):
            with self.assertRaisesRegex(ValueError, 'argv differs'):
                parent.run_parent(self.admission)
        self.original_parent['wall_clock']['clock_kind'] = 'pretend_clock'
        with self.assertRaisesRegex(ValueError, 'Linux supervisor'):
            self.run_parent()
        self.assertFalse((self.root/'ledger/reservation-0001.json').exists())

    def test_resealed_prior_and_source_change_cannot_reset_debt(self):
        self.admit()
        value = parent.read(self.admission)
        value['opening_effective_seconds'] = 1.
        value['independent_review_reference'] = self.put('changed_review.json', dict(
            kind=parent.REVIEW, passed=True, admission_payload_sha256=canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'})))
        bad = self.put('changed_admission.json', value)
        with self.assertRaisesRegex(ValueError, 'arithmetic'):
            parent.verify_admission(bad)
        value = parent.read(self.admission)
        value['parent_source_reference'] = batch.file_reference(ledger.__file__)
        bad = self.put('changed_source.json', value)
        with self.assertRaisesRegex(ValueError, 'source differs'):
            parent.verify_admission(bad)


if __name__ == '__main__':
    unittest.main()
