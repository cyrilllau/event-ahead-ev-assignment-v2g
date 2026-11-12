"""Synthetic gateway wiring checks; numerical integration is tested separately."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_plan_reference_contract as fixtures
from evcs_experiments.evaluation_lock import verify_formal_numerical_records


class CanonicalPublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.PlanReferenceContractTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.protocol={'configuration':{'planning_budget':{'cumulative_wall_seconds':7200}}}

    def record(self, policy='IN_ATTEMPT_FINAL_SEAL_V1'):
        r=self.f.record('sealed')
        worker=self.f.write({'canonical_export_policy':policy})
        numeric=json.loads(Path(r['references']['solver_record']['path']).read_text())
        numeric.update(request_reference=worker,receipt_reference=self.f.write({'fixture':'receipt'}))
        r['references']['solver_record']=self.f.write(numeric)
        seal=json.loads(Path(r['references']['budget_seal']['path']).read_text())
        seal['references']['solver_record']=r['references']['solver_record']
        r['references']['budget_seal']=self.f.write(seal)
        return r,numeric

    def test_new_policy_cannot_borrow_only_old_seal(self):
        r,_=self.record()
        with self.assertRaisesRegex(ValueError,'publication completion'):
            verify_formal_numerical_records(self.protocol,{'plans':[r]})

    def test_full_envelope_calls_completion_validator_with_original_bindings(self):
        r,n=self.record();original=self.f.write(r)
        completion={'canonical_record_reference':original}
        cr=self.f.write(completion)
        wrapper=dict(r,canonical_record_reference=original,publication_completion_reference=cr)
        with patch('evcs_experiments.canonical_in_attempt.verify_publication',return_value=completion) as check:
            verify_formal_numerical_records(self.protocol,{'plans':[wrapper]})
        check.assert_called_once_with(cr,receipt_reference=n['receipt_reference'],
            finalization_reference=r['references']['budget_seal'],worker_request_reference=n['request_reference'])
        for key in ('canonical_record_reference','publication_completion_reference'):
            broken=deepcopy(wrapper);broken.pop(key)
            with self.assertRaisesRegex(ValueError,'both original record and completion'):
                verify_formal_numerical_records(self.protocol,{'plans':[broken]})
        changed=deepcopy(wrapper);changed['method_id']='ARO'
        with self.assertRaisesRegex(ValueError,'differs from original canonical record'):
            verify_formal_numerical_records(self.protocol,{'plans':[changed]})

    def test_unknown_policy_rejected(self):
        r,_=self.record('UNREVIEWED')
        with self.assertRaisesRegex(ValueError,'Unknown canonical publication'):
            verify_formal_numerical_records(self.protocol,{'plans':[r]})
