import copy,time
from unittest.mock import patch
from tests.test_case0_saved_candidate_audit import SavedCandidateTests,audit
from evcs_experiments.random_streams import canonical_hash

class ProposedCandidateTests(SavedCandidateTests):
    def proposal(self):
        p={k:copy.deepcopy(self.master['solution'][k]) for k in ('x','r','task_sha256','config_hash')}
        return dict(plan=p,plan_sha256=canonical_hash(p),model_sha256=self.header['model_sha256'],changed_plan=True,
                    first_stage_numeric_passed=True,planning_acceptance=False)
    def proposed_run(self,p):
        out=self.root/'proposed';out.mkdir()
        return audit.audit_candidate(master=None,proposal=p,header=self.header,context=self.context,output_dir=out,
            deadline=time.monotonic()+20,binding={'proposal':'independent compact evidence'})
    def test_proposal_gets_new_physical_q_and_unambiguous_origin(self):
        with patch.object(audit,'solve_fixed_plan',wraps=audit.solve_fixed_plan) as solve:
            r=self.proposed_run(self.proposal())
        self.assertEqual(solve.call_count,len(self.f.scenario_ids))
        self.assertAlmostEqual(r['upper_bound_usd'],self.expected)
        self.assertEqual(r['kind'],'CASE0_PROPOSED_CANDIDATE_UB_AUDIT_V1')
        self.assertEqual(r['candidate_origin'],'compact_DRO_epigraph_proposal')
        self.assertIsNone(r['original_master_reported_objective_usd'])
        self.assertFalse(r['native_full_vector_missing_from_saved_master'])
        self.assertFalse(r['planning_acceptance'] or r['formal_OOS'])
    def test_compact_flags_cannot_rescue_nonintegral_plan(self):
        p=self.proposal();p['plan']['x'][0]=.5;p['plan_sha256']=canonical_hash(p['plan'])
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('Do not solve invalid plan')):
            with self.assertRaisesRegex(ValueError,'first-stage'):self.proposed_run(p)
    def test_same_plan_marker_rejected_before_q(self):
        p=self.proposal();p['changed_plan']=False
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('Do not redo unchanged Q')):
            with self.assertRaisesRegex(ValueError,'proposed candidate'):self.proposed_run(p)
