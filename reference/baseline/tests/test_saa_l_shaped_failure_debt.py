"""Two genuine accounting kinds in an explicitly synthetic development fixture."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from evcs_experiments import campaign_batch as batch
from evcs_experiments import saa_l_shaped_failure_debt as debt
from evcs_experiments.random_streams import canonical_hash


class DecompositionFailureDebtTests(unittest.TestCase):
    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return batch.file_reference(path)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.plan = 'development.SAA'
        self.old = dict(domain='development', method_id='SAA', plan_id=self.plan,
                        budget={'cumulative_wall_seconds': 60}, model_id='IEEE33-v54-Q1',
                        spec_id='S05', target_id='synthetic', canonical_identity={'plan_id':self.plan},
                        canonical_batch_id='synthetic', plan_input_references={}, protocol_reference={},
                        config_reference={}, bank_acceptance_reference={}, summary_reference={},
                        audit_reference={}, template_references=[])
        oldworker = self.put('old-worker.json', self.old)
        q1 = self.put('q1.json', dict(queue_root=str(self.root/'old'), items=[
            dict(request_reference=oldworker, plan_id=self.plan, plan_budget_seconds=60),
            dict(plan_id='development.ARO')]))
        s1 = dict(request_reference=q1, attempt_receipts=[], halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
        self.put('old/state.json', s1)
        s1ref = self.put('s1.json', s1)
        t1 = self.put('t1.json', dict(job_id='first', complete=True, state='FAILED', returncode=2, elapsed_seconds=2.5))
        self.fee1 = self.put('fee1.json', dict(kind='SAA_ASSEMBLED_MASTER_FAILED_PARENT_EXPENSE_V1',
            passed=True, accounting_PASS=True, status='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION',
            references={'worker_request':oldworker}, original_queue_reference=q1,
            closing_snapshot_reference=s1ref, terminal_job_result_reference=t1, plan_id=self.plan,
            SAA_charged_seconds=2.5, SAA_remaining_seconds=57.5, ARO_charged_seconds=0,
            ARO_no_invocation=True, ARO_no_worker_outputs=True, parent_charged_once=True,
            nested_seconds_added=False, cumulative_accounted_seconds_decimal='2.5', opening_cumulative_seconds_decimal='0'))
        isolate = self.put('isolate.json', dict(kind='SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1',
            old_queue_permanently_quarantined=True, original_queue_rearm_prohibited=True,
            terminal_fee_settlement_reference=self.fee1, original_queue_reference=q1,
            original_queue_snapshot_reference=s1ref))
        rw = self.put('recovery-worker.json', self.old)
        q2 = self.put('q2.json', dict(queue_root=str(self.root/'recovery'), items=[dict(plan_id=self.plan, request_reference=rw)]))
        s2 = dict(request_reference=q2, attempt_receipts=[], halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
        self.put('recovery/state.json', s2)
        s2ref = self.put('s2.json', s2)
        t2 = self.put('t2.json', dict(job_id='second', complete=True, state='FAILED', returncode=2, elapsed_seconds=3.5))
        opening = dict(reference=self.fee1, plan_id=self.plan, accounted_seconds=2.5,
                       budget_seconds=60, remaining_seconds=57.5, domain='development')
        admission = self.put('old-admission.json', dict(campaign_root=str(self.root),
            queue_reference=q2, worker_request_reference=rw))
        reservation = self.put('reservation.json', dict(opening=opening, queue_reference=q2,
            admission_reference=admission, worker_request_reference=rw,
            parent_job={'job_id':'second','directory':str(self.root/'jobs/second')}))
        self.expense_data = dict(kind='SAA_SAVED_MODEL_RECOVERY_TERMINAL_EXPENSE_V1', passed=False,
            status='HOLD_FAILED_OR_UNCOVERED_PARENT_EXPENSE_RETAINED', automatic_resume_authorized=False,
            reservation_reference=reservation, terminal_job_result_reference=t2,
            closing_snapshot_reference=s2ref, original_failed_parent_expense_reference=self.fee1,
            original_spent_seconds=2.5, new_queue_charge_seconds=0, actual_parent_seconds=3.5,
            new_accounted_seconds=3.5, uncovered_parent_seconds=3.5, cumulative_spent_seconds=6.,
            remaining_seconds=54., budget_within_limit=True)
        e = self.put('fee2.json', self.expense_data)
        native = self.put('native.json', {'native_interface':dict(raw_model_status=17, status='MEM_LIMIT',
            solution_count=0, valid_incumbent=False, solver_bound_without_constant_usd=None)})
        self.failure_data = dict(kind='SAA_SAVED_MODEL_FAILED_PARENT_FEE_AND_CAUSE_REVIEW_V1', passed=True,
            status='PASS_FAILED_PARENT_EXPENSE_SETTLED_NO_NUMERICAL_RESULT', required_references=[],
            original_worker_reference=oldworker, original_failed_fee_reference=self.fee1,
            original_isolation_decision_reference=isolate, terminal_expense_reference=e,
            terminal_job_result_reference=t2, reservation_reference=reservation,
            recovery_queue_reference=q2, closing_queue_snapshot_reference=s2ref,
            recovery_worker_reference=rw, original_accounted_seconds=2.5, new_actual_parent_seconds=3.5,
            cumulative_accounted_seconds=6., remaining_original_seconds=54., parent_cost_fully_retained=True,
            receipt_count=0, numerical_acceptance=False, raw_native_status=17, native_status='MEM_LIMIT',
            has_valid_incumbent=False, has_valid_bound=False, native_result_reference=native)
        self.failure = self.put('failure.json', self.failure_data)
        self.adoption_data = dict(kind='SAA_SECOND_MEMORY_FAILURE_TERMINAL_ADOPTION_V1',
            failure_accounting_review_reference=self.failure, recovery_queue_rearm_prohibited=True,
            repeat_same_request_prohibited=True, formal_SAA_algorithm_admitted=False, plan_id=self.plan,
            terminal_expense_reference=e, cumulative_accounted_seconds=6., remaining_original_seconds=54.)
        self.adoption = self.put('adoption.json', self.adoption_data)
        self.worker = dict(self.old, model_input_mode='l_shaped_development')
        self.queue_root = str(self.root/'successor')
        self.ref = self.envelope('debt', self.queue_root)
        self.worker['failure_carryover_reference'] = self.ref

    def envelope(self, name, queue_root):
        v = dict(kind=debt.KIND, domain='development', plan_id=self.plan, campaign_root=str(self.root),
                 module_reference=batch.file_reference(debt.__file__), failure_review_reference=self.failure,
                 terminal_adoption_reference=self.adoption, successor_queue_root=queue_root,
                 successor_worker_identity_sha256=debt.worker_identity(self.worker))
        v['independent_review_reference'] = self.put(name+'-review.json', dict(kind=debt.REVIEW_KIND,
            passed=True, payload_sha256=canonical_hash(v)))
        return self.put(name+'.json', v)

    def test_two_failed_parents_preserved_and_original_queue_history_receives_six(self):
        before = [(self.root/p).read_bytes() for p in ('old/state.json','recovery/state.json','fee2.json')]
        with self.assertRaisesRegex(ValueError, 'claim missing'):
            debt.verify_failure_debt(self.ref,self.worker,self.queue_root)
        claim = debt.claim_failure_debt(self.ref,self.worker,self.queue_root)
        self.assertEqual(claim,debt.claim_failure_debt(self.ref,self.worker,self.queue_root))
        w = self.put('new-worker.json',self.worker)
        q = dict(queue_root=self.queue_root, items=[dict(plan_id=self.plan, request_reference=w,
                 failure_carryover_reference=self.ref)], source_references=[batch.file_reference(debt.__file__)])
        opening = batch._carryovers(q)
        self.assertEqual(opening[self.plan]['accounted_seconds'],6.)
        self.assertEqual(opening[self.plan]['remaining_seconds'],54.)
        qr = self.put('new-queue.json',q)
        _, spent = batch._history(dict(request_reference=qr,attempt_receipts=[]),qr,opening=opening)
        self.assertEqual(spent,{self.plan:6.})
        self.assertEqual(before,[(self.root/p).read_bytes() for p in ('old/state.json','recovery/state.json','fee2.json')])

    def test_second_successor_cannot_claim_same_debt(self):
        debt.claim_failure_debt(self.ref,self.worker,self.queue_root)
        other_root=str(self.root/'renamed');ref=self.envelope('other',other_root)
        worker=dict(self.worker,failure_carryover_reference=ref)
        with self.assertRaisesRegex(ValueError,'already claimed'):
            debt.claim_failure_debt(ref,worker,other_root)

    def test_native_failure_cannot_be_relabelled_success(self):
        (self.root/'fee2.json').write_text(json.dumps(dict(self.expense_data,passed=True)))
        with self.assertRaises(ValueError):debt.claim_failure_debt(self.ref,self.worker,self.queue_root)

    def test_old_queue_rearm_and_debt_reset_rejected(self):
        p=self.root/'recovery/state.json';v=json.loads(p.read_text());v['halted']=None;p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'queue changed'):
            debt.claim_failure_debt(self.ref,self.worker,self.queue_root)

    def test_worker_budget_physical_or_domain_change_rejected(self):
        for key,value in [('budget',{'cumulative_wall_seconds':61}),('target_id','other'),('domain','formal')]:
            w=copy.deepcopy(self.worker);w[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                debt.claim_failure_debt(self.ref,w,self.queue_root)

    def test_queue_requires_source_and_matching_worker_debt(self):
        debt.claim_failure_debt(self.ref,self.worker,self.queue_root)
        w=self.put('new-worker.json',self.worker)
        q=dict(queue_root=self.queue_root,source_references=[],items=[dict(plan_id=self.plan,
               request_reference=w,failure_carryover_reference=self.ref)])
        with self.assertRaisesRegex(ValueError,'source-bound'):batch._carryovers(q)

    def test_new_campaign_cannot_reset_claim_namespace(self):
        debt.claim_failure_debt(self.ref,self.worker,self.queue_root)
        original_root=self.root
        self.root=self.root/'foreign-campaign';self.root.mkdir()
        other=str(self.root/'queue');ref=self.envelope('other',other)
        worker=dict(self.worker,failure_carryover_reference=ref)
        with self.assertRaisesRegex(ValueError,'campaign differs'):
            debt.claim_failure_debt(ref,worker,other)
        self.assertFalse((self.root/'operations/saa-l-shaped-debt-claims').exists())
        self.root=original_root

    def test_renamed_failure_review_cannot_create_second_claim(self):
        debt.claim_failure_debt(self.ref,self.worker,self.queue_root)
        self.failure=self.put('failure-renamed.json',self.failure_data)
        self.adoption=self.put('adoption-renamed.json',dict(self.adoption_data,
            failure_accounting_review_reference=self.failure))
        other=str(self.root/'renamed-queue');ref=self.envelope('renamed',other)
        worker=dict(self.worker,failure_carryover_reference=ref)
        with self.assertRaisesRegex(ValueError,'already claimed'):
            debt.claim_failure_debt(ref,worker,other)


if __name__ == '__main__':
    unittest.main()
