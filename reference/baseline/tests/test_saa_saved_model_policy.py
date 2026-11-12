"""Narrow resource-delta guards, independent of commercial solver availability."""
import unittest
from unittest.mock import patch

from evcs_experiments import native_gurobi
from evcs_experiments import saa_saved_model_recovery as recovery


class SavedModelMemoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.options = dict(time_limit=600., threads=4, mip_rel_gap=.001,
            primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8)

    def test_only_native_memory_limit_changes_and_restores(self):
        original = native_gurobi._parameters
        expected_seconds, expected = original(self.options)
        with recovery._memory_policy():
            seconds, actual = native_gurobi._parameters(self.options)
            self.assertEqual(seconds, expected_seconds)
            self.assertEqual(actual.pop('SoftMemLimit'), 64.)
            expected.pop('SoftMemLimit')
            self.assertEqual(actual, expected)
            self.assertEqual(self.options['time_limit'], 600.)
        self.assertIs(native_gurobi._parameters, original)

    def test_rejects_budget_gap_threads_or_tolerance_drift(self):
        for change in (dict(time_limit=601), dict(threads=1), dict(mip_rel_gap=.005),
                       dict(solver_feasibility_tolerance=1e-9)):
            with self.subTest(change=change), recovery._memory_policy():
                with self.assertRaises(ValueError):
                    native_gurobi._parameters({**self.options, **change})

    def test_nested_hook_rejected_and_exception_restores(self):
        original = native_gurobi._parameters
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with recovery._memory_policy():
                with self.assertRaisesRegex(ValueError, 'already installed'):
                    with recovery._memory_policy():
                        pass
                raise RuntimeError('deliberate')
        self.assertIs(native_gurobi._parameters, original)

    def test_public_caller_cannot_open_development_scope(self):
        with self.assertRaisesRegex(ValueError, 'formal inputs only'):
            recovery._formal(None, 'development')
        with self.assertRaisesRegex(ValueError, 'private fixture'):
            recovery._formal(object(), 'formal')

    def test_current_host_headroom_is_required_not_stale_policy_value(self):
        with patch.object(recovery.Path, 'read_text', return_value='MemAvailable: 134217728 kB\n'):
            result = recovery._live_resource_observation()
        self.assertEqual(result['available_memory_kib'], 128*1024*1024)
        self.assertEqual(result['native_soft_limit_gb'], 64.)
        for content in ('MemAvailable: 134217727 kB\n', 'MemTotal: 999999999 kB\n', 'MemAvailable: 999999999 bytes\n'):
            with self.subTest(content=content), patch.object(recovery.Path, 'read_text', return_value=content):
                with self.assertRaises(ValueError):
                    recovery._live_resource_observation()


if __name__ == '__main__':
    unittest.main()
