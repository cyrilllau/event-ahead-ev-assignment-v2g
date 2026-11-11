"""Independent negative cases against saved numerical evidence, never server solves."""
import json
import unittest

from evcs_experiments.random_streams import canonical_hash
from tests import test_case0_planning_audit as fixtures


class Case0IndependentAdversarialTests(unittest.TestCase):
    setUp = fixtures.Case0IndependentPlanningAuditTests.setUp
    produce = fixtures.Case0IndependentPlanningAuditTests.produce
    audit = fixtures.Case0IndependentPlanningAuditTests.audit
    edit = fixtures.Case0IndependentPlanningAuditTests.edit

    def test_consistently_resealed_false_native_bound_fails_bound_order(self):
        self.produce('SAA')
        saved = json.loads((self.out / 'result.json').read_text())
        false_bound = saved['upper_bound_usd'] + 1000.
        self.edit('master.json', lambda row: (
            row.__setitem__('lower_bound_usd', false_bound),
            row['raw_solver'].__setitem__('raw_lower_bound_without_constant_usd', false_bound)))
        self.edit('result.json', lambda row: (
            row.__setitem__('lower_bound_usd', false_bound), row.__setitem__('gap', 0.)))
        with self.assertRaisesRegex(ValueError, 'bound/gap order'):
            self.audit()

    def test_false_producer_flags_do_not_substitute_for_recomputation(self):
        self.produce('ARO')
        self.edit('result.json', lambda row: (
            row.__setitem__('numerical_valid', False),
            row.__setitem__('full_support_independently_evaluated', False),
            row.__setitem__('gap_passed', False)))
        # The method auditor derives proof from arrays even without those flags;
        # the separate group admission still requires the accepted producer status.
        self.assertTrue(self.audit()['passed'])

    def test_resealed_plan_hash_cannot_change_original_vector(self):
        self.produce()
        def alter(row):
            row['plan']['x'][0] += 1.
            row['plan_sha256'] = canonical_hash(row['plan'])
        self.edit('result.json', alter)
        with self.assertRaisesRegex(ValueError, 'plan x'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
