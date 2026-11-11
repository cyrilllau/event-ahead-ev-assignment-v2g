"""Adversarial metadata publication tests; physical wiring has separate goldens."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from evcs_experiments import formal_record_settlement, sealed_plan_record
from evcs_experiments.evaluation_lock import (PROTOCOL_REFERENCE_KINDS, create_protocol_lock,
                                            file_reference, lock_plan_batch, verify_plan_batch)
from evcs_experiments.formal_record_settlement import settle_plan_record, finalize_settled_record
from evcs_experiments.sealed_plan_record import prepare_sealed_plan_record
from tests import test_formal_record_settlement as settlement_tests


class SealedPlanRecordTests(unittest.TestCase):
    def setUp(self):
        self.base = settlement_tests.SettlementTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.identity = dict(plan_id='p', target_id='target', method_id='EA_DRO', setting_id='baseline')
        self.sources = [file_reference(p) for p in
                        (sealed_plan_record.__file__, formal_record_settlement.__file__)]
        inp = self.base.write('input.json', dict(task_sha256='tasks', scenario_ids=['s1']))
        self.inputs = dict.fromkeys(('tasks', 'routes', 'initial_state', 'planning_support'), inp)
        self.locks = self.root/'locks'
        protocol = dict(domain='development', model_version='IEEE33-v54-Q1', target_ids=['target'],
                        method_ids=['EA_DRO'], references={k:self.sources for k in PROTOCOL_REFERENCE_KINDS},
                        configuration=dict(planning_budget=self.base.budget),
                        batches=[dict(batch_id='test', expected_plans=[self.identity],
                                      sample_ids_by_target={'target':['test/oos/1']})])
        self.protocol = create_protocol_lock(self.locks, protocol)
        original_write = self.base.write
        def write(name, value):
            if name == 'request.json':
                value.update(domain='development', canonical_identity=self.identity, canonical_batch_id='test',
                             protocol_reference=self.protocol, plan_input_references=self.inputs,
                             planning_source_references=self.sources)
            if name == 'candidate.json':
                value['candidate'].update(support_ids=['s1'], input_binding={'fixture':'unit'})
            return original_write(name, value)
        self.base.write = write
        args = self.base.fixture(method_id='EA')
        answer = settle_plan_record(**args)
        final = finalize_settled_record(answer, actual_attempt_seconds_at_cutoff=7,
                                        final_seal_reserve_seconds=.1, output_directory=self.root/'final')
        seal = dict(status='PASS_FINAL_RECORD_SEAL', receipt_reference=args['receipt_reference'],
                    references=final['references'], actual_attempt_seconds_before_confirmation_write=7.01,
                    charged_attempt_seconds=7.1, seal_allocation_seconds=.1,
                    allocation_is_measured_solver_time=False)
        self.seal_ref = self.base.write('finalization.json', seal)
        self.request = dict(kind=sealed_plan_record.KIND, identity=self.identity, batch_id='test',
                            references=dict(protocol=self.protocol, worker_request=file_reference(self.root/'request.json'),
                                            budget_seal=self.seal_ref),
                            plan_input_references=self.inputs, planning_source=self.sources)

    def test_original_native_record_remains_unchanged_and_lock_reuses(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*.json')}
        output = self.root/'published'
        answer = prepare_sealed_plan_record(self.request, output=output)
        record = json.loads(Path(answer['plan_record_reference']['path']).read_text())
        self.assertFalse(answer['production_admission'])
        self.assertEqual(answer['solver_calls'], 0)
        self.assertEqual(record['references']['solver_record'], file_reference(self.root/'final/solver_record.json'))
        numeric = json.loads(Path(record['references']['solver_record']['path']).read_text())
        self.assertEqual(numeric['native_master_status'], 'TIME_LIMIT')
        self.assertEqual(numeric['gap_relative'], .2)
        lock_plan_batch(self.locks, 'test', [record])
        self.assertEqual(verify_plan_batch(self.locks, 'test')['batch']['plans'], [record])
        again = prepare_sealed_plan_record(self.request, output=output)
        self.assertTrue(again['verified_reuse'])
        self.assertEqual(again['plan_record_reference'], answer['plan_record_reference'])
        self.assertTrue(all(p.read_bytes() == b for p,b in before.items()))

    def test_resealed_native_and_accounting_edits_rejected(self):
        seal_path = Path(self.seal_ref['path'])
        seal = json.loads(seal_path.read_text())
        solver_path = Path(seal['references']['solver_record']['path'])
        original = solver_path.read_bytes()
        for field, value in [('native_master_status','OPTIMAL'), ('cumulative_wall_seconds',0),
                             ('full_support_verified',False), ('target_id','another')]:
            with self.subTest(field=field):
                numeric = json.loads(original); numeric[field] = value
                solver_path.write_text(json.dumps(numeric))
                modified = deepcopy(seal); modified['references']['solver_record'] = file_reference(solver_path)
                seal_path.write_text(json.dumps(modified))
                request = deepcopy(self.request); request['references']['budget_seal'] = file_reference(seal_path)
                with self.assertRaises(ValueError):
                    prepare_sealed_plan_record(request, output=self.root/'rejected')
                self.assertFalse((self.root/'rejected').exists())
        solver_path.write_bytes(original)

    def test_identity_inputs_sources_and_seal_tamper_rejected(self):
        bad = []
        r = deepcopy(self.request); r['identity']['method_id'] = 'ARO'; bad.append(r)
        r = deepcopy(self.request); r['planning_source'] = self.sources[:1]; bad.append(r)
        r = deepcopy(self.request); r['plan_input_references']['tasks'] = self.base.write('foreign.json', {'task_sha256':'x'}); bad.append(r)
        r = deepcopy(self.request); r['references']['budget_seal']['sha256'] = '0'*64; bad.append(r)
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ValueError):
                prepare_sealed_plan_record(request, output=self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())

    def test_published_tamper_never_repaired_or_overwritten(self):
        output = self.root/'published'
        prepare_sealed_plan_record(self.request, output=output)
        path = output/'plan.json'
        bad = path.read_bytes()+b' '
        path.write_bytes(bad)
        with self.assertRaises(ValueError):
            prepare_sealed_plan_record(self.request, output=output)
        self.assertEqual(path.read_bytes(), bad)

    def test_truncated_or_reordered_manifest_is_rejected(self):
        output = self.root/'published'
        prepare_sealed_plan_record(self.request, output=output)
        path = output/'manifest.json'
        original = json.loads(path.read_text())
        for files in ([], original['files'][:-1], list(reversed(original['files']))):
            with self.subTest(files=files):
                changed = json.dumps({'files':files})
                path.write_text(changed)
                with self.assertRaisesRegex(ValueError, 'manifest differs'):
                    prepare_sealed_plan_record(self.request, output=output)
                self.assertEqual(path.read_text(), changed)

    def test_formal_publication_has_no_development_bypass(self):
        # Even a successfully validated formal payload is not sufficient until
        # the post-seal publication work has a reviewed accounting integration.
        with patch.object(sealed_plan_record, 'validated_sealed_payload', return_value={'domain':'formal'}):
            with self.assertRaisesRegex(ValueError, 'publication accounting'):
                prepare_sealed_plan_record(self.request, output=self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
