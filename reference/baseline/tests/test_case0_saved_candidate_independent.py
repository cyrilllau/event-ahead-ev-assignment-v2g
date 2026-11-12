"""Independent negative checks of saved-candidate evidence reuse."""
import copy
from unittest.mock import patch
from tests.test_case0_saved_candidate_audit import SavedCandidateTests, audit

class SavedCandidateIndependentTests(SavedCandidateTests):
    def test_ambiguity_change_rejected_before_any_fixed_q(self):
        self.header['ambiguity_sha256']='0'*64
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('Must fail before Q')):
            with self.assertRaisesRegex(ValueError,'ambiguity/support'):
                self.run_audit()

    def test_first_stage_array_identity_change_rejected_before_q(self):
        self.header['first_stage_sha256']='0'*64
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('Must fail before Q')):
            with self.assertRaisesRegex(ValueError,'identity'):
                self.run_audit()

    def test_resealed_checkpoint_q_value_does_not_bypass_raw_kkt(self):
        import json
        self.run_audit()
        checkpoint=json.loads((self.root/'audit/checkpoint.json').read_text())
        checkpoint['completed'][0]['q_usd']+=100.
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('Must recheck before new Q')):
            with self.assertRaisesRegex(ValueError,'Checkpoint Q cost'):
                self.run_audit('tampered_value',resume=checkpoint)
