"""Hand-tail masses and paired fixed-template resampling; no formal OOS read."""
import json
import unittest
from dataclasses import replace

import numpy as np

from evcs_experiments.evaluation_statistics import (
    PairedSamples, empirical_cvar95, fractional_tail, paired_bootstrap,
    paired_bootstrap_mean, worst_decile_clsr,
)


def fixture(values, reasons=None):
    values = np.asarray(values, dtype=float)
    targets, bases, variants, methods = values.shape
    ids = np.empty(values.shape, dtype=object)
    for t in range(targets):
        for b in range(bases):
            for v in range(variants):
                ids[t, b, v, :] = f't{t}/development/planning/b{b}/v{v}'
    return PairedSamples(values, tuple(f't{t}' for t in range(targets)),
        tuple(f'b{b}' for b in range(bases)),
        [[f'v{v}' for v in range(variants)] for _ in range(bases)],
        tuple(f'm{m}' for m in range(methods)), ids, reasons)


class FractionalTailTests(unittest.TestCase):
    def test_cvar_fractional_mass_and_integral_boundary(self):
        result = empirical_cvar95(np.arange(1., 31.))
        self.assertAlmostEqual(result['value'], (30 + .5 * 29) / 1.5)
        self.assertEqual(result['tail_mass_in_samples'], 1.5)
        self.assertEqual(result['full_tail_observations'], 1)
        self.assertEqual(result['boundary_observation_weight'], .5)
        self.assertEqual(result['tail_observation_count'], 2)
        exact = empirical_cvar95(np.arange(20.))
        self.assertEqual(exact['value'], 19)
        self.assertEqual(exact['tail_observation_count'], 1)
        self.assertEqual(exact['boundary_observation_weight'], 0)

    def test_ties_never_expand_tail_probability(self):
        result = empirical_cvar95([0.] * 20 + [10.] * 10)
        self.assertEqual(result['value'], 10)
        self.assertEqual(result['tail_mass_in_samples'], 1.5)
        self.assertEqual(result['tail_observation_count'], 2)

    def test_worst_clsr_is_lower_fractional_tail(self):
        result = worst_decile_clsr(np.arange(15.) / 100.)
        self.assertAlmostEqual(result['value'], .005 / 1.5)
        self.assertEqual(result['side'], 'lower')
        self.assertEqual(result['tail_mass_in_samples'], 1.5)
        self.assertEqual(worst_decile_clsr([.8] * 15)['value'], .8)

    def test_undefined_tail_propagates_and_serializes_without_nan(self):
        result = empirical_cvar95([1., np.nan, 2.])
        self.assertIsNone(result['value'])
        self.assertEqual(result['na_count'], 1)
        self.assertEqual(result['status'], 'NA')
        json.dumps(result, allow_nan=False)

    def test_invalid_tail_inputs(self):
        for values in ([], [[1]], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                empirical_cvar95(values)
        for fraction in (0, -1, 1.1, float('nan'), float('inf'), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                fractional_tail([1, 2], tail_fraction=fraction, side='upper')
        with self.assertRaises(ValueError):
            worst_decile_clsr([1.001])


class PairedBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values = np.empty((8, 2, 3, 2))
        base = np.array([[0, 2, 4], [10, 12, 14]])
        for t in range(8):
            values[t, :, :, 0] = (base if t % 2 == 0 else -base) + 100 * t
            values[t, :, :, 1] = values[t, :, :, 0] + 7
        cls.samples = fixture(values)
        cls.result = paired_bootstrap_mean(cls.samples, seed=20260911, reference_method='m0')

    def test_constant_paired_difference_has_point_interval(self):
        for result in self.result['per_target']:
            difference = result['methods']['m1']['difference_from_reference']
            self.assertAlmostEqual(difference['estimate'], 7.)
            np.testing.assert_allclose(difference['ci95'], [7, 7], atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.result['bootstrap_differences'],
            np.broadcast_to([0., 7.], (2000, 8, 2)), atol=1e-12, rtol=0)

    def test_matched_targets_and_fixed_base_strata_share_indices(self):
        indices = np.array(self.result['bootstrap_indices'])
        self.assertEqual(indices.shape, (2000, 2, 3))
        self.assertTrue(np.all((0 <= indices) & (indices < 3)))
        # Opposite variant effects cancel across target pairs on every draw.
        aggregate = self.result['aggregate']['m0']
        self.assertEqual(aggregate['estimate'], 350.)
        np.testing.assert_allclose(aggregate['ci95'], [350, 350], atol=1e-12, rtol=0)
        self.assertEqual(self.result['target_weights'], [.125] * 8)
        raw = self.samples.values[0, :, :, 0]
        expected = np.mean([raw[b, indices[0, b]] for b in range(2)])
        self.assertAlmostEqual(self.result['bootstrap_statistics'][0][0][0], expected)

    def test_seed_reproduces_and_methods_are_not_seed_inputs(self):
        again = paired_bootstrap_mean(self.samples, seed=20260911, reference_method='m1')
        self.assertEqual(again['bootstrap_indices'], self.result['bootstrap_indices'])
        self.assertEqual(again['bootstrap_statistics'], self.result['bootstrap_statistics'])
        changed = paired_bootstrap_mean(self.samples, seed=20260912, reference_method='m0')
        self.assertNotEqual(changed['bootstrap_indices_sha256'], self.result['bootstrap_indices_sha256'])

    def test_tail_bootstrap_reuses_paired_draws_and_difference_of_tail_means(self):
        result = paired_bootstrap(self.samples, seed=20260911, reference_method='m0', statistic='cvar95')
        self.assertEqual(result['bootstrap_indices'], self.result['bootstrap_indices'])
        self.assertEqual(result['tail_definition']['side'], 'upper')
        self.assertEqual(result['tail_definition']['tail_mass_in_samples'], .3)
        self.assertEqual(result['tail_definition']['tail_observation_count'], 1)
        self.assertEqual(result['per_target'][0]['methods']['m0']['estimate'], 14)
        np.testing.assert_allclose(result['aggregate']['m1']['difference_from_reference']['ci95'], [7, 7])
        service = fixture(np.array([[[[.1, .3], [.5, .7], [.7, .9]]]]))
        lower = paired_bootstrap(service, seed=1, reference_method='m0', statistic='worst_decile_clsr')
        self.assertAlmostEqual(lower['aggregate']['m0']['estimate'], .1)
        self.assertAlmostEqual(lower['aggregate']['m1']['difference_from_reference']['estimate'], .2)

    def test_na_is_explicit_never_zero_filled_or_target_dropped(self):
        values = self.samples.values.copy()
        values[0, 0, 0, 1] = np.nan
        reasons = np.full(values.shape, '', dtype=object)
        reasons[0, 0, 0, 1] = 'no_procured_reserve'
        result = paired_bootstrap_mean(fixture(values, reasons), seed=1, reference_method='m0')
        first = result['per_target'][0]['methods']['m1']
        self.assertIsNone(first['estimate'])
        self.assertEqual(first['na_count'], 1)
        self.assertAlmostEqual(first['na_rate'], 1 / 6)
        self.assertEqual(first['na_reasons'], {'no_procured_reserve': 1})
        self.assertIsNone(result['aggregate']['m1']['estimate'])
        self.assertEqual(result['aggregate']['m1']['undefined_target_count'], 1)
        self.assertAlmostEqual(result['aggregate']['m1']['na_rate'], 1 / 48)
        self.assertIsNone(result['aggregate']['m1']['difference_from_reference']['estimate'])
        self.assertIsNotNone(result['aggregate']['m0']['estimate'])
        json.dumps(result, allow_nan=False)

    def test_strict_layout_matching_and_na_validation(self):
        for kwargs in (dict(target_ids=('same',) * 8), dict(variant_ids=[['v', 'v', 'w']] * 2),
                       dict(values=np.zeros((8, 2, 3, 1)))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.samples, **kwargs)
        ids = self.samples.sample_ids.copy()
        ids[0, 0, 0, 1], ids[0, 0, 1, 1] = ids[0, 0, 1, 1], ids[0, 0, 0, 1]
        with self.assertRaisesRegex(ValueError, 'matched sample IDs'):
            replace(self.samples, sample_ids=ids)
        values = self.samples.values.copy(); values[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'explicit NA reason'):
            replace(self.samples, values=values)
        with self.assertRaises(ValueError):
            paired_bootstrap_mean(self.samples, seed=True, reference_method='m0')
        with self.assertRaises(ValueError):
            paired_bootstrap_mean(self.samples, seed=1, reference_method='m0', replications=100)
        with self.assertRaises(ValueError):
            paired_bootstrap_mean(self.samples, seed=1, reference_method='absent')


if __name__ == '__main__':
    unittest.main()
