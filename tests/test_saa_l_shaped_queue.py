"""Real two-parent queue/child, native solve, resume, audit and canonical join.

This is a small synthetic development integration. It exercises the original
campaign_batch receipt scope, not Linux jobctl or historical Helene failure debt.
Input/template fixture construction is separately measured outside queue receipts.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evcs_experiments import campaign_batch as batch
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments.canonical_in_attempt import POLICY, verify_publication
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import PROTOCOL_REFERENCE_KINDS, create_protocol_lock
from evcs_experiments.runtime.jobctl import THREAD_VARS


def load_template_fixture():
    # Test-only fixture loading by explicit project configuration, no sys.path.
    path = get_paths().project_root/'tests/test_recourse_template_index.py'
    spec = importlib.util.spec_from_file_location('queue_template_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LShapedQueueIntegrationTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get('EVCS_SAA_QUEUE_TEST_ARTIFACTS_ROOT')
        if retained:
            Path(retained).mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix='queue_', dir=retained)).resolve()
        else:
            temporary = tempfile.TemporaryDirectory(prefix='saa-l-shaped-queue-')
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name).resolve()
        self.project = get_paths().project_root

    def put(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return batch.file_reference(path)

    @staticmethod
    def read(ref):
        batch.verify_reference(ref)
        return batch.read_json(ref['path'])

    def prepare(self):
        module = load_template_fixture()
        fixture_root = self.root/'physical'
        fixture_root.mkdir()
        fixture = module.TemplateIndexTests()
        started = time.monotonic()
        with patch.object(module.tempfile, 'TemporaryDirectory',
                return_value=SimpleNamespace(name=str(fixture_root), cleanup=lambda: None)):
            fixture.setUp()
        elapsed = time.monotonic() - started
        self.addCleanup(fixture.doCleanups)
        toy = fixture.fixture
        worker = self.project/'tests/fixtures/saa_l_shaped_queue_worker.py'
        helper = self.project/'scripts/audits/audit_s05_pending_recovery.py'
        sources = [batch.file_reference(p) for p in sorted((self.project/'src/evcs_experiments').rglob('*.py'))]
        sources.extend([batch.file_reference(worker), batch.file_reference(helper)])
        self.sources = sources
        budget = dict(cumulative_wall_seconds=30., single_master_seconds=5.,
            final_certificate_reserve_seconds=5., shutdown_reserve_seconds=1.)
        budgetref = self.put('budget.json', budget)
        physical = self.put('physical_inputs.json', dict(task_sha256=toy.tasks.task_sha256,
            config_hash=toy.tasks.config_hash, scenario_ids=list(toy.scenario_ids),
            fixture_setup_seconds=elapsed, setup_outside_campaign_receipt_scope=True,
            template_index_reference=fixture.index, association_review_reference=fixture.review))
        identity = dict(plan_id='tiny.SAA.lshaped', target_id='tiny', method_id='SAA', setting_id='baseline')
        protocol = dict(domain='development', model_version='IEEE33-v54-Q1', target_ids=['tiny'], method_ids=['SAA'],
            configuration=dict(planning_budget=budget), references={k:[budgetref] for k in PROTOCOL_REFERENCE_KINDS},
            batches=[dict(batch_id='saa_lshaped', expected_plans=[identity], sample_ids_by_target={'tiny':['never_opened']})])
        protocol['references']['source'] = sources
        protocolref = create_protocol_lock(self.root/'locks', protocol)
        binding = dict(fixture='tiny original numerical templates and real L-shaped queue',
            task_sha256=toy.tasks.task_sha256, physical_input_reference=physical)
        request = dict(domain='development', plan_id=identity['plan_id'], target_id='tiny', method_id='SAA',
            model_input_mode='l_shaped_development', algorithm_id=kernel.ALGORITHM_ID, budget=budget,
            input_binding=binding, planning_source_references=sources,
            template_index_reference=fixture.index, template_association_review_reference=fixture.review,
            template_references=[row['original_template_reference'] for row in fixture.provider().index['rows']],
            helper_reference=batch.file_reference(helper), output_root=str(self.root/'work'),
            protocol_reference=protocolref, canonical_identity=identity, canonical_batch_id='saa_lshaped',
            canonical_export_policy=POLICY,
            plan_input_references={k:physical for k in ('tasks','routes','initial_state','planning_support')})
        workerref = self.put('worker_request.json', request)
        queue = dict(schema_version=1, batch_id='saa_lshaped', spec_id='S05', model_id='IEEE33-v54-Q1',
            domain='development', queue_root=str(self.root/'queue'), source_references=sources,
            budget_reference=budgetref, job_wall_seconds=30., shutdown_reserve_seconds=1.,
            items=[dict(item_id='tiny.SAA', plan_id=identity['plan_id'], request_reference=workerref,
                plan_budget_seconds=30., argv=[sys.executable, str(worker)], cwd=str(self.project))])
        queueref = self.put('queue_request.json', queue)
        return workerref, queueref, identity['plan_id']

    def parent(self, reference, name):
        env = dict(os.environ)
        env.pop('EVCS_JOB_DIRECTORY', None)
        env.pop('EVCS_JOB_ID', None)
        for key in THREAD_VARS:
            env[key] = '1'
        argv = [sys.executable, '-m', 'evcs_experiments.campaign_batch',
            '--request', reference['path'], '--outputroot', str(self.root/'queue')]
        started = time.monotonic()
        with (self.root/(name+'.stdout.log')).open('x') as out, (self.root/(name+'.stderr.log')).open('x') as err:
            process = subprocess.Popen(argv, cwd=self.project, env=env, stdout=out, stderr=err, start_new_session=True)
            code = process.wait(timeout=35)
        elapsed = time.monotonic() - started
        self.put(name+'_parent_execution.json', dict(pid=process.pid, ppid=os.getpid(), argv=argv,
            returncode=code, elapsed_seconds=elapsed, domain='development', actual_subprocess=True,
            jobctl_supervision=False, whole_parent_expense_settled=False))
        self.assertEqual(code, 0, (self.root/(name+'.stderr.log')).read_text())
        result = json.loads((self.root/(name+'.stdout.log')).read_text())
        return result, process.pid

    def test_real_two_attempt_resume_original_receipt_seal_and_canonical(self):
        workerref, queueref, plan_id = self.prepare()
        first, parent1 = self.parent(queueref, 'first')
        self.assertEqual(first['status'], 'TIME_LIMIT', first)
        self.assertTrue(first['resumable'])
        checkpoint = first['checkpoint']
        original = Path(checkpoint['path']).read_bytes()
        opening = batch.read_json(self.root/'queue/state.json')
        first_receipt_ref = opening['attempt_receipts'][0]
        first_receipt = self.read(first_receipt_ref)
        first_result = self.read(first_receipt['result_reference'])
        first_trace = batch.read_json(self.root/'queue/tiny.SAA__attempt_0001/actual_execution.json')
        self.assertEqual(first_trace['ppid'], parent1)
        self.assertNotEqual(first_trace['pid'], parent1)
        self.assertEqual(first_result['raw_engine_status'], 'ITERATION_LIMIT')
        self.assertEqual(first_trace['algorithm_status'], 'ITERATION_LIMIT')
        self.assertGreater(first['cumulative_plan_seconds'][plan_id], 0.)
        old_master = first_trace['master_execution_references']
        old_events = self.read(first_trace['event_inventory_reference'])['files']
        second, parent2 = self.parent(queueref, 'second')
        self.assertEqual(second['status'], 'COMPLETE_BATCH', second)
        self.assertEqual(Path(checkpoint['path']).read_bytes(), original)
        closing = batch.read_json(self.root/'queue/state.json')
        self.assertEqual(len(closing['attempt_receipts']), 2)
        self.assertEqual(len(closing['canonical_publication_receipts']), 1)
        self.assertEqual(len(closing['finalization_receipts']), 1)
        second_receipt_ref = closing['attempt_receipts'][1]
        receipt = self.read(second_receipt_ref)
        invocation = self.read(receipt['invocation_reference'])
        result = self.read(receipt['result_reference'])
        trace = batch.read_json(self.root/'queue/tiny.SAA__attempt_0002/actual_execution.json')
        self.assertEqual(trace['ppid'], parent2)
        self.assertNotEqual(trace['pid'], first_trace['pid'])
        self.assertEqual(invocation['previous_checkpoint'], checkpoint)
        self.assertEqual(invocation['cumulative_plan_seconds'], first['cumulative_plan_seconds'][plan_id])
        self.assertGreater(invocation['cumulative_plan_seconds'], 0.)
        self.assertLess(invocation['plan_remaining_seconds'], 30.)
        self.assertEqual(trace['kernel_sessions'][0], first_trace['kernel_sessions'][0])
        self.assertEqual(trace['kernel_trace'][0], first_trace['kernel_trace'][0])
        self.assertEqual(trace['master_execution_references'][:len(old_master)], old_master)
        for reference in old_master + old_events:
            batch.verify_reference(reference)
        inventory = self.read(trace['event_inventory_reference'])
        actual_files = [batch.file_reference(p) for p in (self.root/'work/events').rglob('*') if p.is_file()]
        self.assertEqual(sorted(actual_files, key=lambda r:r['path']),
            sorted(inventory['files'] + [trace['event_inventory_reference']], key=lambda r:r['path']))
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['raw_engine_status'], 'CONVERGED')
        completionref = closing['canonical_publication_receipts'][0]
        sealref = closing['finalization_receipts'][0]
        publication = verify_publication(completionref, receipt_reference=second_receipt_ref,
            finalization_reference=sealref, worker_request_reference=workerref)
        canonical = self.read(publication['canonical_record_reference'])
        numeric = self.read(canonical['references']['solver_record'])
        self.assertTrue(numeric['full_support_verified'])
        self.assertEqual(numeric['driver_status'], 'CONVERGED')
        self.assertAlmostEqual(numeric['lower_bound_usd'], 30.266666666666666 - 100., places=7)
        self.assertAlmostEqual(numeric['upper_bound_usd'], numeric['lower_bound_usd'], places=7)
        self.assertEqual(numeric['prior_cumulative_wall_seconds'], first['cumulative_plan_seconds'][plan_id])
        self.assertEqual(numeric['cumulative_charged_seconds'], second['cumulative_plan_seconds'][plan_id])
        self.assertAlmostEqual(first_receipt['elapsed_seconds'] + numeric['charged_attempt_seconds'],
            numeric['cumulative_charged_seconds'])
        self.assertFalse(numeric['scientific_acceptance'])
        self.assertFalse(numeric['formal_OOS'])
        third, _ = self.parent(queueref, 'readback')
        self.assertEqual(third['new_attempt_item_ids'], [])
        self.assertEqual(third['cumulative_plan_seconds'], second['cumulative_plan_seconds'])
        for source in self.sources:
            batch.verify_reference(source)
        self.put('integration_summary.json', dict(status='PASS_DEVELOPMENT_REAL_L_SHAPED_QUEUE_CANONICAL',
            worker_request_reference=workerref, queue_reference=queueref, first_checkpoint=checkpoint,
            opening_receipt=first_receipt_ref, closing_receipt=second_receipt_ref,
            completion_reference=completionref, finalization_reference=sealref,
            canonical_reference=publication['canonical_record_reference'], source_references=self.sources,
            inherited_seconds=first['cumulative_plan_seconds'][plan_id],
            cumulative_seconds=second['cumulative_plan_seconds'][plan_id],
            native_solver_and_original_full_q_audit_real=True, first_algorithm_status='ITERATION_LIMIT',
            second_algorithm_status='CONVERGED', whole_jobctl_parent_settled=False,
            historical_helene_expenses_admitted=False, formal_admission=False, formal_OOS=False))


if __name__ == '__main__':
    unittest.main()
