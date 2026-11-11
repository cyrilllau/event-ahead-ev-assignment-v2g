import json
from tests.test_case0_remaining_settlement import RemainingSettlementTests
from evcs_experiments import case0_remaining_settlement as settle
from evcs_experiments.campaign_batch import file_reference

class IndependentSettlementTests(RemainingSettlementTests):
 def test_admitted_external_fee_cannot_be_omitted(self):
  fee=self.save(self.op/'fee.json',{'actual_seconds':2.});self.bind_extras([fee])
  with self.assertRaisesRegex(ValueError,'omitted or changed'):self.prep()
 def test_other_settler_source_cannot_settle(self):
  ad=json.loads(open(self.ad['path']).read());ad['settler_reference']=self.source
  self.ad=self.save(self.op/'launch_admission.json',ad)
  with self.assertRaisesRegex(ValueError,'settlement source'):self.prep()
 def test_state_changed_after_proposal_blocks_commit(self):
  p=self.prep();review=self.save(self.op/'review.json',dict(kind='CASE0_REMAINING_SETTLEMENT_REVIEW_V1',passed=True,proposal_sha256=p['proposal_sha256'],settlement_source_reference=file_reference(settle.__file__)))
  self.state['compute_seconds']+=1;self.save(self.root/'state.json',self.state)
  with self.assertRaises(ValueError):settle.commit_remaining_settlement(self.root,p,review)
  self.assertFalse((self.op/'terminal_settlement.json').exists())
