import json
from pathlib import Path
import tempfile
import unittest

from evcs_experiments.evaluation_lock import file_reference, verify_formal_numerical_records
from evcs_experiments.formal_record_settlement import settle_plan_record, finalize_settled_record
from evcs_experiments.random_streams import canonical_hash


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.budget = dict(cumulative_wall_seconds=100, single_master_seconds=10,
            final_certificate_reserve_seconds=20, shutdown_reserve_seconds=1)

    def write(self, name, data):
        path = self.root/name
        path.write_text(json.dumps(data))
        return file_reference(path)

    def fixture(self, *, elapsed=6, prior=3, native='TIME_LIMIT', driver='TIME_LIMIT',
                gap=.2, lb=80, ub=100, raw_status='TIME_LIMIT', part_plan=None, part_status='PASS_TEST',
                audit_pass=True, candidate_bound_offset=0, mutate_result=None, mutate_invocation=None, method_id='EA_DRO'):
        plan = dict(x=[1], r=[[0]], task_sha256='tasks', config_hash='config')
        digest = canonical_hash(plan)
        request = self.write('request.json', dict(plan_id='p', target_id='target',
            method_id=method_id, budget=self.budget))
        queue = self.write('queue.json', dict(domain='development', items=[dict(item_id='i',
            plan_id='p', request_reference=request, plan_budget_seconds=100)]))
        candidate = self.write('candidate.json', dict(plan_id='p', raw_engine_status=raw_status,
            driver_status=driver, candidate=dict(plan=plan, plan_sha256=digest,
            lower_bound_usd=lb+candidate_bound_offset, upper_bound_usd=ub+candidate_bound_offset, gap_relative=gap,
            native_master_status=native, native_master_solver_status=9,
            original_rmp=dict(status=native, lower_bound_usd=lb))))
        parts = {name+'_audit_reference': self.write(name+'.json', dict(status=part_status,
            plan_sha256=digest if part_plan is None else part_plan)) for name in ('recourse','probability','master')}
        audit = self.write('audit.json', dict(status='PASS_INDEPENDENT_PLANNING_NUMERICAL_AUDIT',
            passed=audit_pass, full_support_verified=True, plan_id='p', plan_sha256=digest,
            request_reference=request, candidate_record_reference=candidate,
            lower_bound_usd=lb, upper_bound_usd=ub, gap_relative=gap,
            native_master_status=native, raw_engine_status=raw_status, **parts))
        invocation = dict(domain='development', item_id='i', plan_id='p', request_reference=queue,
            item_request_reference=request, cumulative_plan_seconds=prior,
            plan_remaining_seconds=100-prior, plan_budget_seconds=100,
            result_path=str(self.root/'result.json'))
        if mutate_invocation: mutate_invocation(invocation)
        invref = self.write('invocation.json', invocation)
        result = dict(status='COMPLETE', invocation_sha256=invref['sha256'],
            numerical_audit_reference=audit, artifacts=[candidate,audit],
            attempt_accounted_seconds=5, queue_prior_attempt_seconds=prior,
            cumulative_accounted_seconds=prior+5, raw_engine_status=raw_status)
        if mutate_result: mutate_result(result)
        resultref = self.write('result.json', result)
        receipt = self.write('receipt.json', dict(item_id='i', plan_id='p', request_reference=queue,
            invocation_reference=invref, result_reference=resultref,
            elapsed_seconds=elapsed, status='COMPLETE', returncode=0))
        return dict(receipt_reference=receipt, numerical_audit_reference=audit,
            prior_cumulative_seconds=prior, planning_budget=self.budget)

    def test_existing_methods_keep_probability_and_raw_convergence_gates(self):
        for method in ('EA_DRO','ARO','OR_DRO'):
            with self.subTest(method=method):
                self.assertEqual(settle_plan_record(**self.fixture(method_id=method))['solver_record']['method_id'],method)
                with self.assertRaises(ValueError):
                    settle_plan_record(**self.fixture(method_id=method,part_status='FAIL'))
                with self.assertRaises(ValueError):
                    settle_plan_record(**self.fixture(method_id=method,driver='CONVERGED',lb=99.9,gap=.001,raw_status='TIME_LIMIT'))

    def test_actual_receipt_retains_timelimit_and_passes_numeric_gate(self):
        args = self.fixture()
        result = settle_plan_record(**args, output_directory=self.root/'settled')
        record = result['solver_record']
        self.assertEqual(record['cumulative_wall_seconds'],9)
        self.assertEqual(record['queue_verification_overhead_seconds'],1)
        self.assertEqual(record['status'],'TIME_LIMIT')
        self.assertEqual(record['native_master_status'],'TIME_LIMIT')
        self.assertFalse(record['formal_OOS'])
        self.assertFalse(record['scientific_spec_pass'])
        # The adapter preserves its actual domain, so a development result
        # cannot pass the formal OOS record gate.
        with self.assertRaises(ValueError):
            verify_formal_numerical_records(dict(configuration=dict(planning_budget=self.budget)),
                dict(plans=[dict(references=result['references'],plan_sha256=record['plan_sha256'])]))
        self.assertEqual(file_reference(args['receipt_reference']['path']),args['receipt_reference'])
        with self.assertRaises(FileExistsError):
            settle_plan_record(**args, output_directory=self.root/'settled')

    def test_budget_boundary_is_inclusive(self):
        record = settle_plan_record(**self.fixture(prior=94))['solver_record']
        self.assertEqual(record['cumulative_wall_seconds'],100)
        with self.assertRaises(ValueError): settle_plan_record(**self.fixture(prior=94.001))

    def test_prior_cannot_reset(self):
        args=self.fixture(); args['prior_cumulative_seconds']=0
        with self.assertRaises(ValueError): settle_plan_record(**args)

    def test_bad_wall_times(self):
        for elapsed in (True,-1,float('nan'),float('inf'),4.999):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                settle_plan_record(**self.fixture(elapsed=elapsed))

    def test_foreign_or_failed_component(self):
        for options in (dict(part_plan='foreign'),dict(part_status='FAIL'),dict(audit_pass=False)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                settle_plan_record(**self.fixture(**options))

    def test_invalid_bounds_or_false_convergence(self):
        for options in (dict(lb=101),dict(gap=.1),dict(driver='CONVERGED'),dict(driver='CONVERGED',lb=99.9,gap=.001),dict(lb=float('nan'))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                settle_plan_record(**self.fixture(**options))
        self.assertEqual(settle_plan_record(**self.fixture(driver='CONVERGED',lb=99.9,gap=.001,raw_status='CONVERGED'))['solver_record']['status'],'CONVERGED')

    def test_independent_money_roundoff_uses_paper_tolerance(self):
        settle_plan_record(**self.fixture(candidate_bound_offset=1e-5))
        with self.assertRaises(ValueError):
            settle_plan_record(**self.fixture(candidate_bound_offset=.01))

    def test_reference_tamper_detected(self):
        args=self.fixture()
        (self.root/'recourse.json').write_text('{}')
        with self.assertRaises(ValueError): settle_plan_record(**args)

    def test_resealed_identity_and_elapsed_tamper_detected(self):
        for mutate in (lambda r:r.update(invocation_sha256='wrong'),
                       lambda r:r.update(cumulative_accounted_seconds=0),
                       lambda r:r.update(artifacts=[])):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                settle_plan_record(**self.fixture(mutate_result=mutate))
        with self.assertRaises(ValueError):
            settle_plan_record(**self.fixture(mutate_invocation=lambda i:i.update(plan_id='foreign')))

    def test_finalization_charges_actual_and_allocated_seal_separately(self):
        answer=settle_plan_record(**self.fixture())
        final=finalize_settled_record(answer, actual_attempt_seconds_at_cutoff=7,
            final_seal_reserve_seconds=.1, output_directory=self.root/'final')
        record=final['solver_record']
        self.assertAlmostEqual(record['cumulative_wall_seconds'],10.1)
        self.assertEqual(record['cumulative_charged_seconds_before_final_seal'],10)
        self.assertEqual(record['post_receipt_administration_seconds_at_cutoff'],1)
        self.assertTrue(record['final_budget_acceptance'])
        self.assertTrue(record['requires_seal_confirmation'])
        self.assertFalse(record['seal_confirmation_verified'])
        self.assertFalse(record['formal_OOS'])
        self.assertEqual(answer['solver_record']['cumulative_wall_seconds'],9)
        with self.assertRaises(FileExistsError):
            finalize_settled_record(answer, actual_attempt_seconds_at_cutoff=7,
                final_seal_reserve_seconds=.1, output_directory=self.root/'final')

    def test_finalization_budget_boundary_and_invalid_allocation(self):
        answer=settle_plan_record(**self.fixture(prior=93))
        final=finalize_settled_record(answer, actual_attempt_seconds_at_cutoff=6,
            final_seal_reserve_seconds=1, output_directory=self.root/'exact')
        self.assertEqual(final['solver_record']['cumulative_wall_seconds'],100)
        for actual,reserve in ((6.001,1),(5,.1),(6,0),(6,-1),(6,True),(6,2),(float('nan'),.1)):
            with self.subTest(actual=actual,reserve=reserve),self.assertRaises(ValueError):
                finalize_settled_record(answer, actual_attempt_seconds_at_cutoff=actual,
                    final_seal_reserve_seconds=reserve, output_directory=self.root/'bad')
        with self.assertRaises(ValueError):
            finalize_settled_record(final, actual_attempt_seconds_at_cutoff=6,
                final_seal_reserve_seconds=.1, output_directory=self.root/'twice')


if __name__ == '__main__': unittest.main()
