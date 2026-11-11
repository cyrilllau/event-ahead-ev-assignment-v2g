"""Independent adversarial tests of the central SAA accounting branch."""
import json
from pathlib import Path
import tempfile
import unittest
from evcs_experiments.evaluation_lock import file_reference, verify_formal_numerical_records


class CompositeGatewayIndependentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pb = {'cumulative_wall_seconds':7200}
        self.numeric = dict(domain='formal', method_id='SAA', plan_sha256='plan',
            full_support_verified=True, status='TIME_LIMIT', cumulative_wall_seconds=5683.899147049283,
            lower_bound_usd=205364.35583154118, upper_bound_usd=225329.06402995915)
        self.budget = {'planning_budget':self.pb}

    def gate(self, **extra_refs):
        refs = {}
        for key, value in (('solver_record',self.numeric),('budget',self.budget)):
            path = self.root/(key+'.json')
            path.write_text(json.dumps(value))
            refs[key] = file_reference(path)
        refs.update(extra_refs)
        verify_formal_numerical_records({'configuration':{'planning_budget':self.pb}},
            {'plans':[dict(method_id='SAA',plan_sha256='plan',references=refs)]})

    def test_remove_all_composite_markers_cannot_use_generic_gate(self):
        with self.assertRaisesRegex(ValueError,'explicit original or composite'):
            self.gate()

    def test_each_surviving_composite_marker_requires_complete_kind(self):
        for key in ('composite_request','composite_accounting'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'Incomplete SAA composite'):
                self.gate(**{key:{'path':'unused'}})
        self.numeric['accounting_kind']='SAA_ORIGINAL_SEAL_PLUS_EXTERNAL_LEDGER_V1'
        with self.assertRaisesRegex(ValueError,'Incomplete SAA composite'):
            self.gate()

    def test_claiming_original_accounting_without_seal_is_rejected(self):
        self.numeric.update(final_budget_acceptance=True,requires_seal_confirmation=True)
        self.budget['final_budget_acceptance']=True
        with self.assertRaisesRegex(ValueError,'seal confirmation'):
            self.gate()

    def test_false_original_acceptance_does_not_bypass_gate(self):
        self.numeric['final_budget_acceptance']=False
        with self.assertRaisesRegex(ValueError,'accounting is incomplete'):
            self.gate()
