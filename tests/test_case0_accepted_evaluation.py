import copy
import unittest
from unittest.mock import patch
from evcs_experiments import case0_accepted_evaluation as m
from evcs_experiments import case0_evaluation_runner as runner
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.evaluation_metrics import plan_identity

class AcceptedComparisonTests(unittest.TestCase):
    def setUp(self):
        plan=dict(x=[1.],r=[[1.]],task_sha256='a'*64,config_hash='b'*64)
        sha=plan_identity(FirstStagePlan(**plan))
        self.row=dict(input_identity_reference={'fixture':'input'},comparison_result=dict(
            method_id='SAA',accepted_for_comparison=True,gap_passed=False,
            normal_gap_target=.005,gap=.00543,numerical_valid=True,plan=plan,plan_sha256=sha))
        self.physical=dict(first_stage_check={'passed':True},plan_sha256=sha,
                           input_identity_reference=self.row['input_identity_reference'])
        self.identity=dict(task_sha256='a'*64,lead_h=1,profile='expanded48-v1',target='zeta_2020',vehicle_ids=list(range(48)))
    def test_acceptance_does_not_relabel_original_gap(self):
        before=copy.deepcopy(self.row)
        result=m.validate_view('SAA',self.row,self.physical,self.identity)
        self.assertFalse(result['gap_passed']);self.assertEqual(before,self.row)
    def test_gap_acceptance_cannot_override_physical_failure(self):
        self.physical['first_stage_check']['passed']=False
        with self.assertRaisesRegex(ValueError,'first-stage'):m.validate_view('SAA',self.row,self.physical,self.identity)
    def test_foreign_plan_is_rejected(self):
        self.row['comparison_result']['plan']['x']=[0.]
        with self.assertRaisesRegex(ValueError,'plan differs'):m.validate_view('SAA',self.row,self.physical,self.identity)
    def test_information_timing_is_not_relaxed(self):
        self.identity['lead_h']=0
        with self.assertRaisesRegex(ValueError,'information'):m.validate_view('SAA',self.row,self.physical,self.identity)
    def test_general_one_percent_override_not_allowed(self):
        self.row['comparison_result']['gap_passed']=True
        with self.assertRaisesRegex(ValueError,'relabel'):m.validate_view('SAA',self.row,self.physical,self.identity)
    def test_distinct_card_dispatch_skips_old_normal_result_assumption(self):
        with patch.object(runner,'_read',return_value={'kind':m.KIND}),patch.object(m,'validate_accepted_comparison',return_value='checked') as check,patch.object(runner,'planning_evidence',side_effect=AssertionError('No normal result relabeling')):
            self.assertEqual(runner.validate_case0_planning_acceptance('fixture','card',_attempt_reference='attempt'),'checked')
            check.assert_called_once_with('fixture','card',_attempt_reference='attempt')

if __name__=='__main__':unittest.main()
