"""Independent witness tampering tests for the Case0 OOS checker."""
import json
import unittest

import numpy as np
from tests import test_case0_evaluation_audit as fixtures
from evcs_experiments.campaign_batch import file_reference


class IndependentCase0OOSWitnessTests(unittest.TestCase):
    setUp = fixtures.Case0OOSAuditTests.setUp
    run_audit = fixtures.Case0OOSAuditTests.run_audit
    reseal = fixtures.Case0OOSAuditTests.reseal

    def test_resealed_infeasible_primal_cannot_reuse_old_positive_certificate(self):
        def alter(path):
            value = np.load(path, allow_pickle=False)
            value[:] = 1e9
            np.save(path, value, allow_pickle=False)
        self.reseal('primal.npy', alter)
        with self.assertRaisesRegex(ValueError, 'KKT'):
            self.run_audit()

    def test_resealed_wrong_scenario_cannot_reuse_correct_solution(self):
        from pathlib import Path
        path = Path(self.ref['path'])
        row = json.loads(path.read_text())
        row['scenario_sha256'] = '0' * 64
        path.write_text(json.dumps(row))
        self.ref = file_reference(path)
        with self.assertRaisesRegex(ValueError, 'scenario identity'):
            self.run_audit()

    def test_changed_plan_cannot_borrow_metrics_of_the_accepted_plan(self):
        def alter(path):
            row = json.loads(path.read_text())
            row['x'][0] = 0
            path.write_text(json.dumps(row))
        self.reseal('plan.json', alter)
        with self.assertRaisesRegex(ValueError, 'fixed plan changed'):
            self.run_audit()


if __name__ == '__main__':
    unittest.main()
