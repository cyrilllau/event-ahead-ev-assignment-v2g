"""Protocol inventory construction only; no physical OOS draw or plan solve."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evcs_experiments.config import get_paths
from evcs_experiments.formal_protocol import protocol_spec
from evcs_experiments.evaluation_batch import evaluate_locked_batch
from evcs_experiments.evaluation_lock import (
    PLAN_REFERENCE_KINDS, begin_evaluation, create_protocol_lock,
    file_reference, lock_plan_batch,
)


class FormalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = get_paths().project_root
        for name in ('configs/formal_experiments.json', 'docs/experiments/experiment_matrix.csv'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((original / name).read_bytes())
        for name in ('docs/experiments/PROTOCOL.md', 'docs/experiments/runtime_policy.json',
                'configs/powerai.toml', 'src/evcs_experiments/evaluation_lock.py',
                'src/evcs_experiments/evaluation_access.py', 'src/evcs_experiments/evidence.py',
                'src/evcs_experiments/formal_protocol.py', 'src/evcs_experiments/random_streams.py'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('synthetic inventory fixture\n')
        data = self.root / 'data/cases/ieee33-v54-q1'
        data.mkdir(parents=True)
        (data / 'synthetic_input.csv').write_text('value\n1\n')
        (data / 'MANIFEST_SHA256.csv').write_text('relative_path\nsynthetic_input.csv\n')

    def test_complete_matrix_and_independent_one_dimensional_grids(self):
        spec = protocol_spec(self.root)
        self.assertEqual(len(spec['target_ids']), 8)
        self.assertEqual(len(spec['complete_matrix']), 15)
        batches = {row['batch_id']: row for row in spec['batches']}
        expected_counts = dict(mechanism=24, physical_chain=40, risk_model=24,
            lead_time=32, storm_intensity=64, ambiguity_width=32, arrival_SOC=24,
            block_duration=16, pole_strength=24, reactive_rating=24, certificate=8,
            fleet_scaling=3, support_scaling=4, anytime=1, AC_fidelity=8)
        self.assertEqual({key: len(value['expected_plans']) for key, value in batches.items()}, expected_counts)
        all_ids = [plan['plan_id'] for batch in batches.values() for plan in batch['expected_plans']]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual({p['setting_id'] for p in batches['lead_time']['expected_plans']},
            {'lead_h=0', 'lead_h=0.5', 'lead_h=1', 'lead_h=2'})
        self.assertEqual({p['target_id'] for p in batches['fleet_scaling']['expected_plans']}, {'michael_2018'})

    def test_shared_exogenous_sample_ids_and_disabled_algorithm_scopes(self):
        spec = protocol_spec(self.root)
        disabled = {'certificate', 'fleet_scaling', 'support_scaling', 'anytime'}
        for batch in spec['batches']:
            self.assertEqual(batch['oos_enabled'], batch['batch_id'] not in disabled)
            count = 1200 if batch['batch_id'] in ('risk_model', 'ambiguity_width') else 400
            for target, sample_ids in batch['sample_ids_by_target'].items():
                self.assertEqual(len(sample_ids), count)
                self.assertEqual(len(set(sample_ids)), count)
                self.assertTrue(all(value.startswith(target + '/formal/oos/') for value in sample_ids))
                self.assertTrue(all(len(value.split('/')) == 6 for value in sample_ids))

    def test_fair_budgets_frozen_in_machine_configuration(self):
        config = protocol_spec(self.root)['configuration']
        planning = config['planning_budget']
        self.assertEqual(planning['cumulative_wall_seconds'], 7200)
        self.assertEqual(planning['single_master_seconds'], 600)
        self.assertEqual(planning['final_certificate_reserve_seconds'], 1790)
        self.assertEqual(planning['shutdown_reserve_seconds'], 10)
        self.assertEqual(planning['recourse_workers'], 1)
        self.assertEqual(config['evaluation_budget']['recourse_workers'], 1)
        self.assertEqual(config['statistics']['bootstrap_replications'], 2000)

    def test_physical_manifest_and_each_input_are_bound(self):
        refs = protocol_spec(self.root)['references']['physical_inputs']
        self.assertEqual({Path(ref['path']).name for ref in refs},
            {'MANIFEST_SHA256.csv', 'synthetic_input.csv'})
        self.assertTrue(all(len(ref['sha256']) == 64 and ref['size_bytes'] > 0 for ref in refs))

    def test_formal_budget_and_numerical_record_rejected_before_oos(self):
        config = protocol_spec(self.root)['configuration']
        budget = self.root / 'plan_budget.json'
        numeric = self.root / 'solver.json'
        original = dict(domain='formal', plan_sha256='a' * 64, full_support_verified=True,
            status='TIME_LIMIT', cumulative_wall_seconds=1720., lower_bound_usd=10., upper_bound_usd=11.)
        record = dict(plan_sha256='a' * 64,
            references={'budget': {'path': str(budget)}, 'solver_record': {'path': str(numeric)}})
        verified = dict(batch={'plans': [record]}, batch_spec={'oos_enabled': True},
            protocol={'configuration': config})
        factory = Mock()
        cases = [('eval_budget', {}, {}, 30), ('plan_budget', {}, {'recourse_workers': 4}, 120),
            ('excess_time', {'cumulative_wall_seconds': 7201.}, {}, 120),
            ('missing_support', {'full_support_verified': False}, {}, 120),
            ('development_record', {'domain': 'development'}, {}, 120),
            ('other_plan', {'plan_sha256': 'b' * 64}, {}, 120),
            ('reversed_bounds', {'lower_bound_usd': 12.}, {}, 120),
            ('nonfinite_bound', {'upper_bound_usd': float('nan')}, {}, 120)]
        with patch('evcs_experiments.evaluation_batch.verify_plan_batch', return_value=verified), \
                patch('evcs_experiments.evaluation_batch.begin_evaluation') as begin:
            for name, numeric_change, budget_change, lp_seconds in cases:
                budget.write_text(json.dumps({'planning_budget': {**config['planning_budget'], **budget_change}}))
                numeric.write_text(json.dumps({**original, **numeric_change}))
                with self.subTest(case=name), self.assertRaises(ValueError):
                    evaluate_locked_batch('unused', 'b0', factory, self.root / 'output', domain='formal',
                        sample_ids_by_target={}, lp_time_limit_seconds=lp_seconds, wall_seconds=7190)
            begin.assert_not_called()
            factory.assert_not_called()
            budget.write_text(json.dumps({'planning_budget': config['planning_budget']}))
            numeric.write_text(json.dumps(original))
            with patch('evcs_experiments.evaluation_batch.inspect.getsourcefile', side_effect=RuntimeError('before OOS sentinel')):
                with self.assertRaisesRegex(RuntimeError, 'before OOS sentinel'):
                    evaluate_locked_batch('unused', 'b0', factory, self.root / 'output', domain='formal',
                        sample_ids_by_target={}, lp_time_limit_seconds=120, wall_seconds=7190)
            begin.assert_not_called()

    def test_central_gateway_refuses_algorithm_only_batch(self):
        spec = protocol_spec(self.root)
        locks = self.root / 'locks'
        create_protocol_lock(locks, spec)
        batch = next(b for b in spec['batches'] if b['batch_id'] == 'fleet_scaling')
        ref = file_reference(self.root / 'data/cases/ieee33-v54-q1/synthetic_input.csv')
        records = []
        for identity in batch['expected_plans']:
            references = {key: ref for key in PLAN_REFERENCE_KINDS}
            references['planning_source'] = [ref]
            records.append(dict(identity, plan_sha256='a' * 64, references=references))
        lock_plan_batch(locks, batch['batch_id'], records)
        with self.assertRaisesRegex(ValueError, 'Algorithm-only'):
            begin_evaluation(locks, batch['batch_id'], domain='formal',
                sample_ids_by_target=batch['sample_ids_by_target'])
        self.assertFalse((locks / 'access-fleet_scaling.json').exists())


if __name__ == '__main__':
    unittest.main()
