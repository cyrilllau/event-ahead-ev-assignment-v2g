"""Factory binding tests; physical LP goldens belong to test_saa_streaming."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from evcs_experiments.formal_saa import prepare_formal_saa,assemble_prepared_saa


class FormalSAAFactoryTests(unittest.TestCase):
    def physical(self,n=160):
        ids=tuple('s'+str(i) for i in range(n))
        def never(_):raise AssertionError('Factory eagerly loaded a recourse')
        return SimpleNamespace(bank=SimpleNamespace(scenario_ids=ids,scenario_sha256s=tuple(str(i)*64 for i in range(n))),
            first_stage=SimpleNamespace(matrix_sha256='f'*64),
            input_binding=dict(method_id='SAA',physical_case_sha256='a'*64),template_loader=never)

    def test_fixed_uniform_law_distinct_optimization_binding_without_eager_copies(self):
        physical=self.physical()
        with patch('evcs_experiments.formal_saa.prepare_baseline_physics',return_value=physical) as factory:
            prepared=prepare_formal_saa(summary_reference={'source':'summary'},audit_reference={'source':'audit'},target_id='target')
        self.assertEqual(factory.call_args.kwargs['method_id'],'SAA')
        self.assertEqual(prepared.recourses.loads,0)
        self.assertEqual(prepared.recourses.capacity,2)
        np.testing.assert_array_equal(prepared.empirical_probabilities,np.full(160,1/160))
        self.assertFalse(prepared.empirical_probabilities.flags.writeable)
        self.assertEqual(prepared.identity['risk_model'],'SAA')
        self.assertFalse(prepared.input_binding['objective_spec']['moment_ambiguity_used_for_optimization'])
        self.assertNotIn('objective_spec',physical.input_binding)

    def test_partial_support_and_expired_assembly_rejected(self):
        with patch('evcs_experiments.formal_saa.prepare_baseline_physics',return_value=self.physical(159)):
            with self.assertRaisesRegex(ValueError,'complete ordered160'):
                prepare_formal_saa(summary_reference={},audit_reference={},target_id='target')
        with patch('evcs_experiments.formal_saa.prepare_baseline_physics',return_value=self.physical()):
            prepared=prepare_formal_saa(summary_reference={},audit_reference={},target_id='target')
        with self.assertRaisesRegex(ValueError,'deadline'):
            assemble_prepared_saa(prepared,output_directory='/unused',max_coefficient_bytes=10,deadline=0)


if __name__=='__main__':unittest.main()
