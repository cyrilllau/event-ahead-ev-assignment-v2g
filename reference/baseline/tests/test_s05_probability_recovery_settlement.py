"""Synthetic job files with the real predecessor carryover verifier; no solver.

Only the release loader is injected: the actual bridge opening-budget function
and complete pre-RMP receipt verifier run against a compact frozen fixture.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evcs_experiments.config import get_paths
from evcs_experiments import failure_carryover as fc
from evcs_experiments.runtime import jobctl

ROOT = get_paths().project_root


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


m = module('settlement_under_test', 'scripts/recovery/settle_s05_probability_recovery.py')
bridge = module('real_settlement_opening_bridge', 'scripts/recovery/prepare_s05_probability_recovery.py')
fixture_module = module('settlement_prior_receipt_fixture', 'tests/test_failure_carryover.py')


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.old = fixture_module.CarryoverTests(); self.old.setUp(); self.addCleanup(self.old.doCleanups)
        self.c = self.old.c; self.root = self.old.root
        self.put = self.old.put
        fc.claim_failure_carryover(self.old.pin, self.old.new, self.old.queue)
        worker = dict(self.old.new, failure_carryover_reference=self.old.pin)
        worker_ref = self.put(self.root/'worker-r2.json', worker)
        queue_ref = self.put(self.root/'queue-r2.json', dict(queue_root=str(self.old.queue),
            items=[dict(plan_id='p', request_reference=worker_ref)]))
        oldmanifest = self.put(self.c/'jobs/formal-fail/manifest.json', dict(job_id='formal-fail'))
        oldresult = self.put(self.c/'jobs/formal-fail/result.json', dict(job_id='formal-fail', complete=True,
            state='FAILED', returncode=2, elapsed_seconds=3351.921871))
        inv_ref = self.put(self.root/'invocation-r2.json', dict(request_reference=queue_ref,
            parent_job=dict(job_id='formal-fail', manifest_reference=oldmanifest), item_request_reference=worker_ref,
            plan_id='p', previous_checkpoint=None, previous_result_reference=None,
            failure_carryover_reference=self.old.pin, cumulative_plan_seconds=16.85686,
            opening_carryover_seconds=16.85686, plan_budget_seconds=7200.))
        self.parentdir = self.c/'jobs/probability-recovery'
        self.output = self.c/'recoveries/probability-recovery'
        self.request = dict(schema_version=1, kind=m.BRIDGE_KIND, domain='formal', spec_id='S05', plan_id='p',
            wall_seconds=120., output_root=str(self.output), parent_job=dict(job_id=self.parentdir.name,
                directory=str(self.parentdir), timeout_seconds=130., grace_seconds=5, threads=1,
                controller_reference=m.reference(jobctl.__file__)),
            references=dict(job_result=oldresult, job_manifest=oldmanifest, worker_request=worker_ref,
                invocation=inv_ref, recovery_script=m.reference(bridge.__file__), settlement_script=m.reference(m.__file__),
                new_source_manifest=self.old.newrelease))
        self.rr = self.put(self.root/'bridge-request.json', self.request)
        manifest = dict(job_id=self.parentdir.name, timeout_seconds=130., grace_seconds=5, threads=1,
            cwd=str(ROOT), outroot=str(self.c/'jobs'), spec_id='S05', controller_path=jobctl.__file__,
            argv=[str(ROOT/'.venv/bin/python'), '-B', bridge.__file__, '--request', self.rr['path'],
                  '--expected-request-sha256', self.rr['sha256'], '--action', 'run'])
        self.mr = self.put(self.parentdir/'manifest.json', manifest)
        self.pr = self.put(self.parentdir/'result.json', dict(job_id=self.parentdir.name, state='SUCCEEDED',
            complete=True, returncode=0, elapsed_seconds=12.5))
        self.binding = dict(job_id=self.parentdir.name, directory=str(self.parentdir), manifest_reference=self.mr,
                            available_seconds=120., deadline_monotonic=1000., manifest_age_seconds=0.)
        self.attempt = self.c/'recovery_attempts'/m.canonical_hash(oldresult)
        self.reservation = dict(kind=m.BRIDGE_KIND, request_reference=self.rr, predecessor_job_result_reference=oldresult,
            output_root=str(self.output), automatic_retry_allowed=False, parent_job=self.binding)
        self.resref = self.put(self.attempt/'reservation.json', self.reservation)
        claim = dict(kind=m.BRIDGE_KIND, request_reference=self.rr, predecessor_job_result_reference=oldresult,
                     successor_output_root=str(self.output), plan_id='p')
        self.claimref = self.put(self.c/'recovery_claims'/(m.canonical_hash(oldresult)+'.json'), claim)
        opening = bridge.opening_budget(self.request['references'], worker, json.loads(Path(inv_ref['path']).read_text()))
        self.derived = self.put(self.output/'derived_state.json', dict(status='VALID_INCUMBENT_BOUND_UNAVAILABLE',
            lower_bound_usd=None, gap_relative=None, upper_bound_usd=232648.24342916225,
            lockable=False, formal_OOS=False, scientific_acceptance=False))
        self.result = dict(kind=m.BRIDGE_KIND, status='VALID_INCUMBENT_BOUND_UNAVAILABLE', request_reference=self.rr,
            claim_reference=self.claimref, original_job_result_reference=oldresult, opening_budget=opening,
            parent_job=self.binding, attempt_elapsed_seconds=10., cumulative_accounted_seconds=3368.778731+10.,
            cumulative_seconds_provisional=True, scientific_acceptance=False, lockable=False, formal_OOS=False,
            lower_bound_usd=None, gap_relative=None, artifacts=[self.derived])
        self.resultref = self.put(self.output/'result.json', self.result)
        paths = SimpleNamespace(project_root=ROOT, campaign_root=self.c)
        for item in (patch.object(m, 'get_paths', return_value=paths), patch.object(m, '_bridge', return_value=bridge)):
            item.start(); self.addCleanup(item.stop)

    def change_parent(self, **values):
        old = json.loads(Path(self.pr['path']).read_text())
        self.pr = self.put(Path(self.pr['path']), dict(old, **values))

    def test_parent_replaces_nested_time_and_is_written_exactly_once(self):
        settled = m.settle(self.rr, self.pr)
        record = settled['record']
        self.assertEqual(record['opening_budget']['opening_cumulative_seconds'], 3368.778731)
        self.assertEqual(record['terminal_parent_seconds'], 12.5)
        self.assertEqual(record['cumulative_accounted_seconds'], 3381.278731)
        self.assertFalse(record['nested_seconds_added']); self.assertFalse(record['cumulative_seconds_provisional'])
        self.assertFalse(record['lockable']); self.assertFalse(record['scientific_acceptance'])
        self.assertEqual(m.verify_settlement(settled['settlement_reference']), record)
        original = Path(settled['settlement_reference']['path']).read_bytes()
        with self.assertRaises(FileExistsError): m.settle(self.rr, self.pr)
        self.assertEqual(Path(settled['settlement_reference']['path']).read_bytes(), original)

    def test_killed_before_result_still_charges_full_terminal_parent(self):
        Path(self.resultref['path']).unlink(); Path(self.claimref['path']).unlink()
        self.change_parent(state='TIMED_OUT', returncode=-9, elapsed_seconds=135.1)
        record = m.settle(self.rr, self.pr)['record']
        self.assertEqual(record['status'], 'SETTLED_FAILED_ATTEMPT_REQUIRES_REVIEW')
        self.assertAlmostEqual(record['cumulative_accounted_seconds'], 3503.878731)
        self.assertIsNone(record['nested_provisional_seconds']); self.assertFalse(record['automatic_retry_allowed'])

    def test_budget_overrun_is_charged_truthfully_and_never_promoted(self):
        self.change_parent(state='FAILED', returncode=2, elapsed_seconds=4000.)
        record = m.settle(self.rr, self.pr)['record']
        self.assertEqual(record['status'], 'SETTLED_BUDGET_EXCEEDED')
        self.assertEqual(record['remaining_seconds'], 0.); self.assertGreater(record['cumulative_accounted_seconds'], 7200.)

    def test_timeout_that_exits_zero_remains_failed_and_is_settled(self):
        self.change_parent(state='TIMED_OUT', returncode=0, elapsed_seconds=130.5)
        record = m.settle(self.rr, self.pr)['record']
        self.assertEqual(record['status'], 'SETTLED_FAILED_ATTEMPT_REQUIRES_REVIEW')
        self.assertAlmostEqual(record['cumulative_accounted_seconds'], 3499.278731)

    def test_unfinished_wrong_id_and_live_parent_are_rejected(self):
        for changes in (dict(complete=False), dict(job_id='unrelated')):
            original = self.pr; before = Path(original['path']).read_bytes()
            self.change_parent(**changes)
            with self.assertRaises(ValueError): m.checked_settlement(self.rr, self.pr)
            Path(original['path']).write_bytes(before); self.pr = original
        with patch.object(m, 'observation', return_value={'blocks_new_launch': True}):
            with self.assertRaisesRegex(ValueError, 'live'): m.checked_settlement(self.rr, self.pr)

    def test_changed_parent_command_and_manifest_binding_rejected(self):
        manifest = json.loads(Path(self.mr['path']).read_text()); manifest['argv'][-1] = 'verify'
        self.put(Path(self.mr['path']), manifest)
        with self.assertRaisesRegex(ValueError, 'exact atomic'): m.checked_settlement(self.rr, self.pr)

    def test_changed_successor_claim_or_derived_bytes_rejected(self):
        Path(self.derived['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'): m.checked_settlement(self.rr, self.pr)

    def test_nested_clock_cannot_exceed_parent_or_reset_opening(self):
        for mutation in (dict(attempt_elapsed_seconds=13.), dict(cumulative_accounted_seconds=10.)):
            self.put(Path(self.resultref['path']), dict(self.result, **mutation))
            with self.assertRaises(ValueError): m.checked_settlement(self.rr, self.pr)

    def test_terminal_parent_cannot_hide_missing_success_result(self):
        Path(self.resultref['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'lacks'): m.checked_settlement(self.rr, self.pr)

    def test_false_lower_bound_and_wrong_output_location_rejected(self):
        self.put(Path(self.resultref['path']), dict(self.result, lower_bound_usd=0.))
        with self.assertRaisesRegex(ValueError, 'scientific'): m.checked_settlement(self.rr, self.pr)
        self.put(Path(self.resultref['path']), self.result)
        with self.assertRaisesRegex(ValueError, 'unique attempt'): m.settle(self.rr, self.pr, self.root/'extra.json')

    def test_same_size_terminal_tamper_invalidates_existing_settlement(self):
        settled = m.settle(self.rr, self.pr)
        path = Path(self.pr['path']); before = path.read_bytes(); after = before.replace(b'12.5', b'13.5')
        self.assertEqual(len(before), len(after)); path.write_bytes(after)
        with self.assertRaisesRegex(ValueError, 'bytes changed'): m.verify_settlement(settled['settlement_reference'])


if __name__ == '__main__': unittest.main()
