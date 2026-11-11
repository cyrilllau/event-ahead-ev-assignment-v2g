"""Independent negative review of SAA failed-parent certificate semantics."""
import json
from pathlib import Path
import unittest
import test_saa_failure_carryover as certificate_fixture


class SAACarryoverIndependentReviewTests(unittest.TestCase):
    def fixture(self):
        fixture=certificate_fixture.SAAFailureCarryoverTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_settlement_plan_prior_and_nested_charge_cannot_be_reapproved(self):
        for key,value in (('plan_id','wrong.plan'),('opening_cumulative_seconds_decimal','1'),
                          ('nested_seconds_added',True),('remaining_seconds_decimal','7200')):
            with self.subTest(key=key):
                f=self.fixture();p=Path(f.refs['settlement']['path']);v=json.loads(p.read_text())
                v[key]=value;f.refs['settlement']=f.write(p,v);f.seal()
                with self.assertRaises(ValueError):f.verify(require_claim=False)

    def test_reapproved_association_axes_or_proofs_cannot_change(self):
        for key,value in (('scenario_ids',['different.atom']),('proof_references',{})):
            with self.subTest(key=key):
                f=self.fixture();p=Path(f.new['template_association_review_reference']['path'])
                v=json.loads(p.read_text());v[key]=value
                f.new['template_association_review_reference']=f.write(p,v);f.seal()
                with self.assertRaisesRegex(ValueError,'Association'):f.verify(require_claim=False)

    def test_current_reader_pair_cannot_be_omitted_even_with_review_payload(self):
        f=self.fixture();p=Path(f.new['template_association_review_reference']['path'])
        v=json.loads(p.read_text());v['implementation_sources'].pop()
        f.new['template_association_review_reference']=f.write(p,v);f.seal()
        with self.assertRaisesRegex(ValueError,'inventory'):f.verify(require_claim=False)

    def test_request_cannot_label_old_release_while_using_new_source_closure(self):
        f=self.fixture();f.new['release_reference']=f.manifests[0];f.seal()
        with self.assertRaisesRegex(ValueError,'release|source'):f.verify(require_claim=False)


if __name__=='__main__':unittest.main()
