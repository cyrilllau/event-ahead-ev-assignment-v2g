"""Real lock writer/readback accepts only declared evidence-reference shapes.

These are synthetic structural fixtures, not numerical evidence or OOS access.
The composite shape matches prepare_saa_composite_record's published record;
its numerical validation remains the responsibility of its dedicated gateway.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evcs_experiments.evaluation_lock import (
    PLAN_REFERENCE_KINDS, PROTOCOL_REFERENCE_KINDS,
    create_protocol_lock, file_reference, lock_plan_batch, verify_plan_batch,
    verify_formal_numerical_records,
)


class PlanReferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.serial = 0

    def write(self, value):
        self.serial += 1
        path = self.root / f'evidence_{self.serial}.json'
        path.write_text(json.dumps(value))
        return file_reference(path)

    def record(self, kind='base', method='EA_DRO'):
        identity = dict(plan_id='toy.'+method, target_id='toy',
                        method_id=method, setting_id='baseline')
        refs = {key: self.write({'fixture': key}) for key in PLAN_REFERENCE_KINDS}
        refs['planning_source'] = [self.write({'fixture': 'source'})]
        if kind == 'sealed':
            refs['solver_record'] = self.write(dict(
                domain='formal', plan_sha256='a'*64, full_support_verified=True,
                status='TIME_LIMIT', lower_bound_usd=1, upper_bound_usd=2,
                cumulative_wall_seconds=20, final_budget_acceptance=True,
                requires_seal_confirmation=True, charged_attempt_seconds=10,
                actual_attempt_seconds_at_cutoff=8))
            refs['budget'] = self.write(dict(
                planning_budget={'cumulative_wall_seconds': 7200},
                final_budget_acceptance=True, charged_attempt_seconds=10))
            refs['budget_seal'] = self.write(dict(
                status='PASS_FINAL_RECORD_SEAL',
                references={key: refs[key] for key in ('solver_record', 'budget')},
                actual_attempt_seconds_before_confirmation_write=9,
                charged_attempt_seconds=10))
        elif kind == 'composite':
            refs.update(composite_request=self.write({'fixture': 'request'}),
                        composite_accounting=self.write({'fixture': 'accounting'}),
                        adapter_sources=[self.write({'fixture': 'adapter'})])
        return dict(**identity, plan_sha256='a'*64, references=refs)

    def lock(self, record):
        self.serial += 1
        directory = self.root / f'lock_{self.serial}'
        identity = {key: record[key] for key in
                    ('plan_id', 'target_id', 'method_id', 'setting_id')}
        protocol = dict(domain='development', model_version='v54-q1',
            target_ids=['toy'], method_ids=[record['method_id']],
            references={key: [self.write({'fixture': key})]
                        for key in PROTOCOL_REFERENCE_KINDS},
            batches=[dict(batch_id='test', expected_plans=[identity],
                          sample_ids_by_target={'toy': ['development-only']})])
        create_protocol_lock(directory, protocol)
        lock_plan_batch(directory, 'test', [record])
        checked = verify_plan_batch(directory, 'test')
        self.assertEqual(checked['batch']['plans'], [record])
        self.assertFalse((directory / 'access-test.json').exists())
        return checked

    def test_legacy_base_roundtrips(self):
        self.lock(self.record())

    def test_matching_seal_passes_numerical_gate_and_real_lock(self):
        record = self.record('sealed')
        verify_formal_numerical_records(
            {'configuration': {'planning_budget': {'cumulative_wall_seconds': 7200}}},
            {'plans': [record]})
        self.lock(record)

    def test_saa_composite_reference_shape_roundtrips(self):
        self.lock(self.record('composite', 'SAA'))

    def test_unknown_mixed_partial_and_foreign_extras_rejected(self):
        base = self.record()
        unknown = deepcopy(base)
        unknown['references']['unreviewed_extra'] = self.write({})
        mixed = self.record('composite', 'SAA')
        mixed['references']['budget_seal'] = self.write({})
        foreign = self.record('composite', 'EA_DRO')
        candidates = [unknown, mixed, foreign]
        for missing in ('composite_request', 'composite_accounting', 'adapter_sources'):
            partial = self.record('composite', 'SAA')
            del partial['references'][missing]
            candidates.append(partial)
        for record in candidates:
            with self.subTest(keys=sorted(record['references']), method=record['method_id']):
                with self.assertRaises(ValueError):
                    self.lock(record)

    def test_extra_evidence_bytes_and_adapter_list_are_verified(self):
        for kind, key in (('sealed', 'budget_seal'),
                          ('composite', 'composite_request'),
                          ('composite', 'composite_accounting'),
                          ('composite', 'adapter_sources')):
            record = self.record(kind, 'SAA' if kind == 'composite' else 'EA_DRO')
            ref = record['references'][key]
            if isinstance(ref, list):
                ref = ref[0]
            Path(ref['path']).write_text('{"changed":true}')
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.lock(record)
        for empty in (True, False):
            record = self.record('composite', 'SAA')
            refs = record['references']['adapter_sources']
            record['references']['adapter_sources'] = [] if empty else refs+refs
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                self.lock(record)


if __name__ == '__main__':
    unittest.main()
