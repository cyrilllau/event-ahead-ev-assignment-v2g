"""Small checkpoint-bank orchestration fixtures, never the 1920-scene bank."""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments import formal_bank as module
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.joint_scenarios import array_hash
from evcs_experiments.random_streams import canonical_hash


class FakeContext:
    calls = []
    interrupt_next = False
    wrong_domain = False
    bad_float = None

    def __init__(self, *args):
        self.events = ('t0', 't1')
        self.input_hashes = {}

    def prepare_target(self, target, bank, *, lead_h):
        assert lead_h == 1.
        return SimpleNamespace(tasks=SimpleNamespace(task_sha256=bank['task_sha256'], task_ids=('H',)),
            target=SimpleNamespace(state_sha256=canonical_hash(target), replay_times_h=np.arange(-1., 24., .25)),
            feature_tasks=SimpleNamespace(empty_H=False))

    def generate(self, prepared, base, variant, split, *, domain, regime):
        assert domain == 'formal' and regime == 'R1' and split in ('planning', 'calibration')
        target = 't0' if prepared.target.state_sha256 == canonical_hash('t0') else 't1'
        sid = f'{target}/formal/{split}/{base}/v{variant + 1:03d}/R1'
        self.calls.append(sid)
        if self.interrupt_next and len(self.calls) == 2:
            type(self).interrupt_next = False
            raise RuntimeError('synthetic interruption after first saved atom')
        arrays = dict(features=np.array([.2, .3, .4, .5, 1.]),
            task_arrival_h=np.array([np.inf]), road_travel_time_min=np.array([[np.inf, 1.]]))
        if self.bad_float is not None:
            name, value = self.bad_float
            arrays[name].flat[0] = value
        ledger = dict(scenario_id=sid, development=self.wrong_domain, residual_donor_event='t1' if target == 't0' else 't0',
            task_sha256=prepared.tasks.task_sha256, assignment_state_sha256=prepared.target.state_sha256,
            array_hashes={name: array_hash(value) for name, value in arrays.items()})
        digest = canonical_hash(ledger); ledger['scenario_sha256'] = digest
        return SimpleNamespace(arrays=arrays, ledger=ledger, diagnostics={}, scenario_sha256=digest)


class FormalBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.out = self.root / 'bank'
        FakeContext.calls = []; FakeContext.interrupt_next = False; FakeContext.wrong_domain = False
        FakeContext.bad_float = None
        refs = {}
        for target in ('t0', 't1'):
            path = self.root / (target + '.json'); path.write_text(json.dumps(dict(task_sha256=canonical_hash(target + 'tasks'))))
            refs[target] = file_reference(path)
        self.taskrefs = refs
        self.request = dict(target_ids=['t0', 't1'], domain='formal', lead_h=1, regime='R1',
            split_sizes={'planning': 1, 'calibration': 1}, source_manifest={}, formal_protocol_lock={})
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(module, 'accepted_task_sources',
            return_value=({'references': {'physical_inputs': []}}, refs)))
        self.stack.enter_context(patch.object(module, 'PlanningContext', FakeContext))
        self.stack.enter_context(patch.object(module, 'network_arcs', return_value=()))
        self.stack.enter_context(patch.object(module, 'get_paths', return_value=SimpleNamespace(data_root=self.root)))

    def run_bank(self, resume=False, request=None):
        return module.generate_bank(self.request if request is None else request,
            wall_seconds=60, outputroot=self.out, resume=resume)

    def test_complete_is_pending_independent_audit_and_all_rhos_are_calibrated(self):
        result = self.run_bank()
        self.assertEqual(result['atom_count'], 8)
        self.assertEqual(result['status'], 'COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT')
        self.assertTrue(result['independent_scientific_audit_required'])
        self.assertFalse(result['formal_OOS']); self.assertFalse(result['formal_plans_solved'])
        for target in self.request['target_ids']:
            moment = json.loads((self.out / target / 'moments.json').read_text())
            self.assertEqual(moment['status'], 'PASS_MOMENT_CALIBRATION')
            self.assertEqual(set(moment['bounds_by_rho']), {'0.5', '1.0', '1.5', '2.0'})
            self.assertFalse(moment['bootstrap']['stream_identity']['development'])
            self.assertEqual(moment['bootstrap']['stream_identity']['root_seed'], 20260902)

    def test_resume_preserves_completed_atom_and_never_regenerates_it(self):
        FakeContext.interrupt_next = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            self.run_bank()
        before = json.loads((self.out / 'checkpoint.json').read_text())
        self.assertEqual(len(before['completed_atoms']), 1)
        retained_sid = next(iter(before['completed_atoms']))
        retained_ref = before['completed_atoms'][retained_sid]
        result = self.run_bank(resume=True)
        self.assertEqual(result['atom_count'], 8)
        self.assertEqual(Counter(FakeContext.calls)[retained_sid], 1)
        after = json.loads((self.out / 'checkpoint.json').read_text())
        self.assertEqual(after['completed_atoms'][retained_sid], retained_ref)
        self.assertGreaterEqual(after['cumulative_seconds'], before['cumulative_seconds'])

    def test_complete_resume_does_not_generate_any_atoms(self):
        self.run_bank(); calls = list(FakeContext.calls)
        result = self.run_bank(resume=True)
        self.assertEqual(result['atom_count'], 8)
        self.assertEqual(FakeContext.calls, calls)

    def test_changed_request_task_bytes_and_saved_atom_are_rejected(self):
        self.run_bank()
        with self.assertRaisesRegex(ValueError, 'request changed'):
            self.run_bank(resume=True, request={**self.request, 'domain': 'development'})
        task = Path(self.taskrefs['t0']['path']); saved = task.read_bytes(); task.write_bytes(saved + b' ')
        with self.assertRaisesRegex(ValueError, 'Bound source changed'):
            self.run_bank(resume=True)
        task.write_bytes(saved)
        ledger = next((self.out / 't0').glob('planning/*/ledger.json')); ledger.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'completed atom changed'):
            self.run_bank(resume=True)

    def test_completed_moment_and_bootstrap_bytes_are_verified_on_resume(self):
        self.run_bank()
        for name in ('moments.json', 'bootstrap.npz'):
            path = self.out / 't0' / name; saved = path.read_bytes(); path.write_bytes(saved + b' ')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_bank(resume=True)
            path.write_bytes(saved)

    def test_complete_atom_written_before_checkpoint_is_recovered_without_regeneration(self):
        self.run_bank()
        path = self.out / 'checkpoint.json'; record = json.loads(path.read_text())
        sid = next(iter(record['completed_atoms'])); del record['completed_atoms'][sid]
        path.write_text(json.dumps(record)); calls = list(FakeContext.calls)
        result = self.run_bank(resume=True)
        self.assertEqual(result['atom_count'], 8)
        self.assertEqual(FakeContext.calls, calls)

    def test_development_atom_is_rejected_without_saving_fake_formal_result(self):
        FakeContext.wrong_domain = True
        with self.assertRaisesRegex(ValueError, 'Formal stream/target/LOEO'):
            self.run_bank()
        self.assertFalse((self.out / 'summary.json').exists())

    def test_infinity_fields_still_reject_nan_and_negative_infinity(self):
        for i, (name, value) in enumerate((('road_travel_time_min', np.nan),
                ('road_travel_time_min', -np.inf), ('task_arrival_h', np.nan),
                ('task_arrival_h', -np.inf))):
            self.out = self.root / f'bad-{i}'
            FakeContext.bad_float = (name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.run_bank()

    def test_crash_between_moment_and_bootstrap_recovers_without_repeating_atoms(self):
        save = module.np.savez_compressed
        for partial_write in (False, True):
            with self.subTest(truncated_npz=partial_write):
                self.out = self.root / f'interrupted-pair-{partial_write}'
                FakeContext.calls = []
                def fail_bootstrap(path, **arrays):
                    if Path(path).name == 'bootstrap.npz':
                        if partial_write:
                            Path(path).write_bytes(b'PK truncated synthetic NPZ')
                        raise RuntimeError('synthetic bootstrap write interruption')
                    return save(path, **arrays)
                with patch.object(module.np, 'savez_compressed', side_effect=fail_bootstrap):
                    with self.assertRaisesRegex(RuntimeError, 'bootstrap write interruption'):
                        self.run_bank()
                moment_bytes = (self.out / 't0/moments.json').read_bytes()
                completed = list(FakeContext.calls)
                self.assertEqual(len(completed), 4)
                result = self.run_bank(resume=True)
                self.assertEqual(result['atom_count'], 8)
                counts = Counter(FakeContext.calls)
                for sid in completed:
                    self.assertEqual(counts[sid], 1)
                recovery = next((self.out / 'incomplete_writes').glob('moments_t0_*'))
                self.assertEqual((recovery / 'moments.json').read_bytes(), moment_bytes)
                record = json.loads((recovery / 'record.json').read_text())
                self.assertFalse(record['scenario_atoms_regenerated'])
                if partial_write:
                    self.assertEqual((recovery / 'bootstrap.npz').read_bytes(), b'PK truncated synthetic NPZ')


if __name__ == '__main__':
    unittest.main()
