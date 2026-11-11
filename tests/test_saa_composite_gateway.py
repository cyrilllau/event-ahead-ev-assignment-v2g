"""The central gate cannot downgrade a composite to unchecked SAA accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from evcs_experiments.evaluation_lock import file_reference, verify_formal_numerical_records


class CompositeGatewayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.policy = {'cumulative_wall_seconds': 7200}
        self.numeric = dict(domain='formal', plan_sha256='a'*64, full_support_verified=True,
            status='TIME_LIMIT', cumulative_wall_seconds=5683.9,
            lower_bound_usd=205364., upper_bound_usd=225329.)
        self.budget = dict(planning_budget=self.policy)

    def check(self):
        refs = {}
        for name, value in [('solver_record', self.numeric), ('budget', self.budget)]:
            path = self.root/(name+'.json'); path.write_text(json.dumps(value))
            refs[name] = file_reference(path)
        verify_formal_numerical_records(dict(configuration=dict(planning_budget=self.policy)),
            dict(plans=[dict(method_id='SAA', plan_sha256='a'*64, references=refs)]))

    def test_each_partial_composite_marker_is_rejected(self):
        for target, key, value in [(self.numeric, 'kind', 'FORMAL_SAA_SAME_PLAN_AUXILIARY_NUMERICAL_RECORD_V1'),
                (self.numeric, 'accounting_kind', 'SAA_ORIGINAL_SEAL_PLUS_EXTERNAL_LEDGER_V1'),
                (self.budget, 'accounting_kind', 'SAA_ORIGINAL_SEAL_PLUS_EXTERNAL_LEDGER_V1')]:
            target[key] = value
            with self.assertRaisesRegex(ValueError, 'Incomplete SAA composite'):
                self.check()
            del target[key]
