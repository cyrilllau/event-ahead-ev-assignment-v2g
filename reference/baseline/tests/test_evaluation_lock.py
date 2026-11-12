"""Append-only paired-plan gates with real byte tampering; no OOS data read."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evcs_experiments.evaluation_lock import (
    PLAN_REFERENCE_KINDS, PROTOCOL_REFERENCE_KINDS, begin_evaluation,
    create_protocol_lock, file_reference, lock_plan_batch, verify_plan_batch,
)
from evcs_experiments.random_streams import canonical_hash


class EvaluationLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.locks = self.root / 'locks'
        self.files = self.root / 'files'; self.files.mkdir()

    def write(self, name, value):
        path = self.files / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return file_reference(path)

    def fixture(self, *, domain='development', batches=1):
        targets = [f't{i}' for i in range(8 if domain == 'formal' else 2)]
        refs = {key: [self.write(key + '.json', {'frozen': key})] for key in PROTOCOL_REFERENCE_KINDS}
        protocol = dict(domain=domain, model_version='IEEE33-v54-Q1', target_ids=targets,
            method_ids=['EA', 'DRO'], references=refs, batches=[])
        plans_by_batch = {}
        for b in range(batches):
            batch_id = f'b{b}'
            expected, plans = [], []
            for target in targets:
                for method in protocol['method_ids']:
                    identity = dict(plan_id=f'{batch_id}-{target}-{method}', target_id=target,
                        method_id=method, setting_id='baseline')
                    expected.append(identity)
                    plan = dict(identity, plan_sha256=canonical_hash({'x': [1, 0], 'r': [2, 0]}),
                        references={key: self.write(identity['plan_id'] + '-' + key + '.json',
                            {'kind': key, 'x': [1, 0], 'r': [2, 0]}) for key in PLAN_REFERENCE_KINDS})
                    plan['references']['planning_source'] = refs['source']
                    plans.append(plan)
            protocol['batches'].append(dict(batch_id=batch_id, expected_plans=expected,
                sample_ids_by_target={target: [f'{target}/{domain}/oos/base/v001',
                    f'{target}/{domain}/oos/base/v002'] for target in targets}))
            plans_by_batch[batch_id] = plans
        return protocol, plans_by_batch

    def lock_fixture(self, **kwargs):
        protocol, plans = self.fixture(**kwargs)
        reference = create_protocol_lock(self.locks, protocol)
        lock_plan_batch(self.locks, 'b0', plans['b0'])
        return protocol, plans, reference

    def begin(self, protocol, **kwargs):
        return begin_evaluation(self.locks, 'b0', domain=protocol['domain'],
            sample_ids_by_target=protocol['batches'][0]['sample_ids_by_target'], **kwargs)

    def test_complete_development_scope_returns_verified_payloads_and_stable_access(self):
        protocol, plans, ref = self.lock_fixture()
        verified = verify_plan_batch(self.locks, 'b0')
        self.assertEqual(verified['protocol'], protocol)
        self.assertEqual(verified['batch']['plans'], plans['b0'])
        self.assertEqual(verified['protocol_reference'], ref)
        first = self.begin(protocol)
        second = self.begin(protocol)
        self.assertEqual(first['access_reference'], second['access_reference'])
        self.assertEqual(first['access']['domain'], 'development')
        json.dumps(first, allow_nan=False)

    def test_protocol_and_plan_ids_cannot_be_overwritten_or_extended_after_access(self):
        protocol, plans, _ = self.lock_fixture()
        self.begin(protocol)
        with self.assertRaisesRegex(ValueError, 'cannot be replaced'):
            create_protocol_lock(self.locks, protocol)
        with self.assertRaisesRegex(ValueError, 'late plans'):
            lock_plan_batch(self.locks, 'b0', plans['b0'])
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            lock_plan_batch(self.locks, 'new-setting-after-OOS', plans['b0'])

    def test_missing_duplicate_extra_or_mismatched_method_never_creates_a_lock(self):
        protocol, plans = self.fixture()
        create_protocol_lock(self.locks, protocol)
        bad = deepcopy(plans['b0']); bad[0]['method_id'] = 'SAA'
        for candidate in (plans['b0'][:-1], plans['b0'] + [plans['b0'][0]], bad):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                lock_plan_batch(self.locks, 'b0', candidate)
            self.assertFalse((self.locks / 'batch-b0.json').exists())
        with self.assertRaises(FileNotFoundError):
            self.begin(protocol)
        self.assertFalse((self.locks / 'access-b0.json').exists())

    def test_every_use_rechecks_raw_plan_source_config_and_input_bytes(self):
        protocol, plans, _ = self.lock_fixture()
        self.begin(protocol)
        references = [protocol['references']['source'][0], protocol['references']['evaluator'][0],
            protocol['references']['config'][0], *[plans['b0'][0]['references'][key]
                for key in PLAN_REFERENCE_KINDS]]
        for reference in references:
            path = Path(reference['path']); original = path.read_bytes()
            path.write_bytes(original.replace(b'0', b'9') if b'0' in original else original + b' ')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Evidence bytes'):
                self.begin(protocol)
            path.write_bytes(original)
        self.begin(protocol)

    def test_domain_and_exact_sample_target_order_are_required(self):
        protocol, _, _ = self.lock_fixture()
        samples = protocol['batches'][0]['sample_ids_by_target']
        bad = deepcopy(samples); bad['t0'].reverse()
        for sample_ids in (bad, {'t0': samples['t0']}, {**samples, 'extra': ['extra/s']}):
            with self.subTest(samples=sample_ids), self.assertRaisesRegex(ValueError, 'sample IDs/order/target'):
                begin_evaluation(self.locks, 'b0', domain='development', sample_ids_by_target=sample_ids)
        with self.assertRaisesRegex(ValueError, 'domain differs'):
            begin_evaluation(self.locks, 'b0', domain='formal', sample_ids_by_target=samples)
        self.assertFalse((self.locks / 'access-b0.json').exists())

    def test_formal_requires_full_s04_pass_card_pinning_this_protocol(self):
        protocol, _, protocol_ref = self.lock_fixture(domain='formal')
        with self.assertRaisesRegex(ValueError, 'full S04 PASS'):
            self.begin(protocol)
        base = dict(spec_id='S04', status='PASS', formal_protocol_lock=protocol_ref)
        bad_cards = [dict(base, status='PASS_INCREMENT'), dict(base, increment_id='I02'),
            dict(base, spec_id='S03'), dict(base, formal_protocol_lock={**protocol_ref, 'sha256': '0' * 64})]
        for i, card in enumerate(bad_cards):
            with self.subTest(card=card), self.assertRaises(ValueError):
                self.begin(protocol, s04_acceptance_ref=self.write(f'bad-card-{i}.json', card))
        self.assertFalse((self.locks / 'access-b0.json').exists())
        card = self.write('s04-pass.json', base)
        access = self.begin(protocol, s04_acceptance_ref=card)
        self.assertEqual(access['access']['s04_acceptance_reference'], card)
        Path(card['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Evidence bytes'):
            self.begin(protocol, s04_acceptance_ref=card)

    def test_future_predeclared_batch_plans_can_lock_under_unchanged_protocol(self):
        protocol, plans, ref = self.lock_fixture(batches=2)
        self.begin(protocol)
        lock_plan_batch(self.locks, 'b1', plans['b1'])
        checked = verify_plan_batch(self.locks, 'b1')
        self.assertEqual(checked['protocol_reference'], ref)
        self.assertFalse((self.locks / 'access-b1.json').exists())

    def test_empty_source_refs_duplicate_scope_and_non_eight_formal_targets_rejected(self):
        original, _ = self.fixture()
        bad_source = deepcopy(original); bad_source['references']['source'] = []
        bad_evaluator = deepcopy(original); bad_evaluator['references']['evaluator'] = []
        bad_batch = deepcopy(original); bad_batch['batches'].append(bad_batch['batches'][0])
        bad_targets = deepcopy(original); bad_targets['domain'] = 'formal'
        bad_samples = deepcopy(original); bad_samples['batches'][0]['sample_ids_by_target']['extra'] = ['x']
        for candidate in (bad_source, bad_evaluator, bad_batch, bad_targets, bad_samples):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                create_protocol_lock(self.locks, candidate)
            self.assertFalse((self.locks / 'protocol.json').exists())

    def test_lock_and_access_envelopes_cannot_change_without_detection(self):
        protocol, _, _ = self.lock_fixture()
        self.begin(protocol)
        for name in ('protocol.json', 'batch-b0.json', 'access-b0.json'):
            path = self.locks / name; original = path.read_text()
            envelope = json.loads(original); envelope['payload_sha256'] = '0' * 64
            path.write_text(json.dumps(envelope))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Corrupt'):
                self.begin(protocol)
            path.write_text(original)

    def test_concurrent_first_access_has_one_immutable_marker(self):
        protocol, _, _ = self.lock_fixture()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.begin, protocol) for _ in range(2)]
            records = [future.result() for future in futures]
        self.assertEqual(records[0]['access_reference'], records[1]['access_reference'])
        self.assertEqual(len(list(self.locks.glob('access-*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
