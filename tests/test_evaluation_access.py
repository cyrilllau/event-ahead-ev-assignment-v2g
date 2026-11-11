"""Formal-scope gates using synthetic locks/events; no held-out weather is read."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evcs_experiments.case_inputs import InputValidationError
from evcs_experiments.evaluation_access import locked_oos_access, require_oos_access
from evcs_experiments.evaluation_lock import (
    PLAN_REFERENCE_KINDS, PROTOCOL_REFERENCE_KINDS, create_protocol_lock,
    file_reference, lock_plan_batch,
)
from evcs_experiments.load_scenarios import _identity
from evcs_experiments.random_streams import stream
from evcs_experiments.wind_scenarios import WindBank, WindParameters, draw_parameters
from evcs_experiments.forecast_routes import ForecastNetwork
from evcs_experiments.task_replay import PreparedTaskBank, replay_prepared


class EvaluationAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.locks = self.root / 'locks'
        self.targets = [f't{i}' for i in range(8)]
        self.sample = 't0/formal/oos/t1/v003/R1'
        self.other = 't0/formal/oos/t1/v004/R1'
        self.source = self.write('source.json', {'synthetic': True})
        self.identity = dict(plan_id='p0', target_id='t0', method_id='EA', setting_id='baseline')
        protocol = dict(domain='formal', model_version='synthetic-gate-test',
            target_ids=self.targets, method_ids=['EA'],
            references={key: [self.source] for key in PROTOCOL_REFERENCE_KINDS},
            batches=[dict(batch_id='b0', expected_plans=[self.identity],
                sample_ids_by_target={'t0': [self.sample, self.other]})])
        self.protocol_ref = create_protocol_lock(self.locks, protocol)
        plan = dict(self.identity, plan_sha256='a' * 64,
            references={key: self.source for key in PLAN_REFERENCE_KINDS})
        plan['references']['planning_source'] = [self.source]
        lock_plan_batch(self.locks, 'b0', [plan])
        self.acceptance = self.write('acceptance.json', dict(spec_id='S04', status='PASS',
            formal_protocol_lock=self.protocol_ref, synthetic_unit_test=True))

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return file_reference(path)

    def scope(self, **kwargs):
        return locked_oos_access(self.locks, 'b0', kwargs.pop('target_id', 't0'),
            kwargs.pop('sample_id', self.sample), kwargs.pop('acceptance', self.acceptance), **kwargs)

    def test_outside_scope_rejects_raw_draw_and_load_identity(self):
        with self.assertRaises(InputValidationError):
            require_oos_access()
        with self.assertRaises(InputValidationError):
            draw_parameters(self.targets, 't0', 't1', 2, split='oos')
        with self.assertRaises(InputValidationError):
            _identity('oos', False, 1, 2)

    def test_full_s04_card_and_complete_scope_required_before_access_marker(self):
        bad = self.write('increment.json', dict(spec_id='S04', status='PASS',
            increment_id='I02', formal_protocol_lock=self.protocol_ref))
        for card in (None, bad):
            with self.subTest(card=card), self.assertRaises(ValueError), self.scope(acceptance=card):
                self.fail('Invalid card opened scope')
        for target, sample in (('t1', self.sample), ('t0', 't0/formal/oos/t1/v050/R1')):
            with self.subTest(target=target, sample=sample), self.assertRaises(InputValidationError):
                with self.scope(target_id=target, sample_id=sample):
                    self.fail('Undeclared sample opened scope')
        self.assertFalse((self.locks / 'access-b0.json').exists())

    def test_exact_identity_and_original_random_addresses(self):
        with self.scope() as scope:
            self.assertEqual((scope.base_index, scope.variant_index), (1, 2))
            self.assertIs(require_oos_access(target_event='t0', base_event='t1',
                base_index=1, variant_index=2, sample_id=self.sample, regime='R1'), scope)
            # Stop before drawing: this increment must not reveal formal OOS.
            with patch('evcs_experiments.wind_scenarios.stream', side_effect=RuntimeError('draw sentinel')) as draw:
                with self.assertRaisesRegex(RuntimeError, 'draw sentinel'):
                    draw_parameters(self.targets, 't0', 't1', 2, split='oos')
                draw.assert_called_once_with('oos', 1, 2, 'wind_amplitude', development=False)
            identity = _identity('oos', False, 1, 2)
            self.assertEqual(identity, stream('oos', 1, 2, 'load_innovation', development=False))
            self.assertNotEqual(identity.stream_id,
                stream('oos', 1, 2, 'load_innovation', development=True).stream_id)
        with self.assertRaises(InputValidationError):
            require_oos_access()

    def test_every_available_coordinate_must_match(self):
        bad_values = dict(target_event='t1', base_event='t2', base_index=2,
            variant_index=3, sample_id=self.other, regime='R2')
        with self.scope():
            for key, value in bad_values.items():
                with self.subTest(key=key), self.assertRaises(InputValidationError):
                    require_oos_access(**{key: value})
            with self.assertRaises(InputValidationError):
                draw_parameters(self.targets, 't1', 't1', 2, split='oos')
            with self.assertRaises(InputValidationError):
                _identity('oos', False, 2, 2)
            for key, value in (('base_index', True), ('variant_index', 2.0)):
                with self.subTest(key=key), self.assertRaises(InputValidationError):
                    require_oos_access(**{key: value})

    def test_exception_and_nested_scope_restore_context(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            with self.scope() as outer:
                with self.scope() as inner:
                    self.assertEqual(inner, outer)
                self.assertIs(require_oos_access(), outer)
                with self.assertRaises(InputValidationError), self.scope(sample_id=self.other):
                    self.fail('Nested scope changed sample')
                raise RuntimeError('synthetic interruption')
        with self.assertRaises(InputValidationError):
            require_oos_access()

    def test_context_is_not_implicitly_granted_to_other_threads(self):
        with self.scope(), ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(InputValidationError):
                pool.submit(require_oos_access).result()

    def test_source_tampering_blocks_next_scope_even_after_prior_access(self):
        with self.scope():
            pass
        Path(self.source['path']).write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Evidence bytes'), self.scope():
            self.fail('Changed source opened scope')

    def test_wind_sampling_cannot_change_the_locked_regime(self):
        parameters = WindParameters('t0', 't1', 't2', 'oos', False, 2,
            1., (0., 0.), 0, 1., (0,))
        bank = object.__new__(WindBank)
        with self.scope(), patch.object(WindBank, 'field', side_effect=RuntimeError('field sentinel')) as field:
            with self.assertRaises(InputValidationError):
                bank.sample(parameters, [0.], [[0., 0.]], regime='R2')
            field.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'field sentinel'):
                bank.sample(parameters, [0.], [[0., 0.]], regime='R1')

    def test_task_replay_requires_complete_formal_identity(self):
        network = object.__new__(ForecastNetwork)
        prepared = object.__new__(PreparedTaskBank)
        full = dict(target_event='t0', base_weather_event='t1', variant_index=2,
            scenario_id=self.sample, regime='R1')
        malformed = [None, {}, {**full, 'target_event': None}, {**full, 'variant_index': 2.0},
            {key: value for key, value in full.items() if key != 'scenario_id'},
            {**full, 'scenario_id': self.other}]
        with self.scope():
            for identity in malformed:
                with self.subTest(identity=identity), self.assertRaises(InputValidationError):
                    replay_prepared(network, prepared, split='oos', scenario_identity=identity)


if __name__ == '__main__':
    unittest.main()
