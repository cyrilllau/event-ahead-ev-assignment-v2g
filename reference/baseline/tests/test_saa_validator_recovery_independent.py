"""Independent recovery-source negative cases using an actual tiny worker."""
import json
import unittest

from evcs_experiments import saa_validator_recovery as adapter, campaign_batch as batch
from evcs_experiments.random_streams import canonical_hash
import test_saa_validator_recovery as fixtures


class RecoveryIndependentEvidenceTests(unittest.TestCase):
    setUp = fixtures.RecoveryTests.setUp
    admit = fixtures.RecoveryTests.admit
    run_parent = fixtures.RecoveryTests.run_parent

    def completed(self):
        self.admit()
        answer = self.run_parent()
        self.assertEqual(answer['queue_result']['status'], 'COMPLETE_BATCH')
        self.invref = batch.file_reference(self.queue / 'saa__attempt_0001/invocation.json')
        self.result_path = self.queue / 'saa__attempt_0001/worker_result.json'
        self.result = json.loads(self.result_path.read_text())
        self.execution = self.result['validation_recovery_execution_reference']
        self.receipt = adapter.read(self.execution)

    def reseal_receipt(self):
        newref = self.f.write(self.root / 'altered-execution.json', self.receipt)
        self.result['artifacts'] = [newref if x == self.execution else x
                                    for x in self.result['artifacts']]
        self.result['validation_recovery_execution_reference'] = newref
        return self.f.write(self.root / 'altered-worker-result.json', self.result)

    def test_resealed_parent_argv_cannot_claim_ordinary_worker_launch(self):
        self.completed()
        parent = adapter.read(self.receipt['parent_execution_reference'])
        parent['actual_argv'] = ['python', '-m', 'evcs_experiments.saa_planning_worker']
        self.receipt['parent_execution_reference'] = self.f.write(self.root / 'wrong-parent.json', parent)
        ref = self.reseal_receipt()
        with self.assertRaisesRegex(ValueError, 'manifest argv|actual child argv'):
            adapter.validate_execution_result(ref, self.invref)

    def test_resealed_receipt_cannot_name_another_invocation_path(self):
        self.completed()
        other = self.f.write(self.root / 'other-invocation.json', adapter.read(self.invref))
        self.receipt['invocation_reference'] = other
        ref = self.reseal_receipt()
        with self.assertRaisesRegex(ValueError, 'invocation/parent'):
            adapter.validate_execution_result(ref, self.invref)

    def test_resealed_receipt_cannot_replace_parent_birth_identity(self):
        self.completed()
        self.receipt['observed_parent_identity'] = dict(
            self.receipt['observed_parent_identity'], token='foreign-parent-birth')
        ref = self.reseal_receipt()
        with self.assertRaisesRegex(ValueError, 'process identity'):
            adapter.validate_execution_result(ref, self.invref)

    def test_resealed_admission_cannot_reset_actual_queue_spending(self):
        self.admit(timeout=9.)
        first = self.run_parent()
        self.assertEqual(first['queue_result']['status'], 'TIME_LIMIT')
        self.assertGreater(first['queue_result']['cumulative_plan_seconds']['tiny.saa'], 0.)
        self.parent_context.__exit__(None, None, None)
        self.admit(2)
        self.a['opening_cumulative_seconds'] = 0.
        review = adapter.read(self.a['independent_review_reference'])
        review['admission_payload_sha256'] = canonical_hash({
            k: v for k, v in self.a.items() if k != 'independent_review_reference'})
        self.a['independent_review_reference'] = self.f.write(self.root / 'false-opening-review.json', review)
        self.ref = self.f.write(self.root / 'false-opening-admission.json', self.a)
        manifest_path = self.parent_directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['argv'] = [self.a['original_python'], adapter.__file__, 'parent',
            '--admission', self.ref['path'], '--expected-sha256', self.ref['sha256']]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'budget opening'):
            self.run_parent()
        self.assertFalse((self.queue / 'saa__attempt_0002').exists())


if __name__ == '__main__':
    unittest.main()
