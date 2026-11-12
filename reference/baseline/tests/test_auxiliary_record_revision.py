import copy
import unittest
from evcs_experiments.auxiliary_record_revision import build_revision
from evcs_experiments.random_streams import canonical_hash


class AuxiliaryRecordRevisionTests(unittest.TestCase):
    def fixture(self):
        names=('old_plan_record','old_numeric','old_budget','old_plan','old_acceptance',
               'bound_result','bound_review','settlement','failed_settlement','protocol')
        refs={k:{'fixture':k} for k in names}
        plan=dict(x=[1],r=[[0]],task_sha256='t',config_hash='c')
        ph=canonical_hash(plan);plan.update(execution_mode='EA',departure_h=1)
        identity=dict(plan_id='p',target_id='michael_2018',method_id='EA_DRO',setting_id='baseline')
        pb=dict(cumulative_wall_seconds=7200)
        protocol=dict(configuration=dict(planning_budget=pb))
        old=dict(identity,domain='formal',status='TIME_LIMIT',full_support_verified=True,support_count=160,
            plan_sha256=ph,cumulative_wall_seconds_decimal='5438.82363936330092',lower_bound_usd=24135.73087520793,
            upper_bound_usd=227762.10627548306,complete_iteration_reserve_seconds_decimal='2410',
            original_native_master_status='TIME_LIMIT',original_native_master_lower_bound_usd=None,
            original_engine_status='PROBABILITY_FAILED',full_CCG_converged=False,
            normal_ccg_checkpoint_compatible=False,normal_gap_relative=.005,references={'old_Q':'unchanged'})
        oldbudget=dict(planning_budget=pb,cumulative_accounted_seconds_decimal=old['cumulative_wall_seconds_decimal'],
                       accounting_reference={'fixture':'original_completion'})
        record=dict(identity,plan_sha256=ph,references=dict(solver_record=refs['old_numeric'],budget=refs['old_budget']))
        acceptance=dict(status='PASS_COMPONENT',component_id='INDIVIDUAL_FORMAL_NUMERICAL_RECORD',plan_record_reference=refs['old_plan_record'])
        common=dict(plan_id='p',accounting_PASS=True,parent_charged_once=True,nested_seconds_added=False,
            cumulative_seconds_provisional=False,cumulative_cap_seconds=7200)
        failed=dict(common,kind='S05_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1',terminal_state='FAILED',
            previous_settlement_reference=oldbudget['accounting_reference'],opening_cumulative_seconds_decimal='5438.82363936330092',
            terminal_parent_seconds_decimal='195.136908',cumulative_accounted_seconds_decimal='5633.96054736330092',
            remaining_seconds_decimal='1566.03945263669908')
        settled=dict(common,kind='S05_CACHED_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1',terminal_state='SUCCEEDED',
            result_reference=refs['bound_result'],previous_settlement_reference=refs['failed_settlement'],
            opening_cumulative_seconds_decimal='5633.96054736330092',terminal_parent_seconds_decimal='64.811043',
            cumulative_accounted_seconds_decimal='5698.77159036330092',remaining_seconds_decimal='1501.22840963669908',
            request_reference={'fixture':'actual_request'},parent_result_reference={'fixture':'parent'})
        result=dict(kind='S05_CACHED_MULTI_ANCHOR_BOUND_ONLY_V1',plan_id='p',plan_sha256=ph,
            lower_bound_usd=210317.49823035582,upper_bound_usd=old['upper_bound_usd'],relative_gap=.0765913537172317,
            request_reference=settled['request_reference'])
        review=dict(kind='R23_INDEPENDENT_TERMINAL_AUXILIARY_BOUND_REVIEW_V1',passed=True,
            status='PASS_INDEPENDENT_RELAXED_AUXILIARY_BOUND_NUMERICAL_REVIEW',plan_id='p',plan_sha256=ph,
            lower_bound_usd=result['lower_bound_usd'],upper_bound_usd=result['upper_bound_usd'],
            relative_gap=result['relative_gap'],native_status='OPTIMAL',references=dict(result=refs['bound_result'],
            request=settled['request_reference'],parent_result=settled['parent_result_reference']))
        d=dict(old_plan_record=record,old_numeric=old,old_budget=oldbudget,old_plan=plan,old_acceptance=acceptance,
            bound_result=result,bound_review=review,settlement=settled,failed_settlement=failed,
            protocol=dict(payload=protocol,payload_sha256=canonical_hash(protocol)))
        return dict(references=refs,revision_sources=[]),d

    def test_retains_plan_native_and_original_evidence_with_exact_new_ledger(self):
        req,d=self.fixture();before=copy.deepcopy(d);n,b=build_revision(req,d)
        self.assertEqual(d,before)
        self.assertEqual(n['status'],'TIME_LIMIT');self.assertIsNone(n['original_native_master_lower_bound_usd'])
        self.assertEqual(n['original_engine_status'],'PROBABILITY_FAILED')
        self.assertEqual(n['references'],d['old_numeric']['references'])
        self.assertEqual(n['gap_relative'],.0765913537172317);self.assertFalse(n['normal_gap_met'])
        self.assertEqual(b['cumulative_accounted_seconds_decimal'],'5698.77159036330092')
        self.assertEqual(b['remaining_seconds_decimal'],'1501.22840963669908')

    def test_different_plan_ub_or_old_acceptance_rejected(self):
        for key,field,value in [('bound_result','plan_sha256','another'),('bound_result','upper_bound_usd',200000),
                              ('old_acceptance','plan_record_reference',{}),('old_numeric','support_count',159)]:
            req,d=self.fixture();d[key][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):build_revision(req,d)

    def test_unsettled_or_changed_numerical_review_rejected(self):
        for key,field,value in [('bound_review','passed',False),('settlement','terminal_state','FAILED'),
                              ('settlement','result_reference',{}),('bound_result','relative_gap',0.)]:
            req,d=self.fixture();d[key][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):build_revision(req,d)

    def test_failure_omission_doublecharge_and_budget_reset_rejected(self):
        for field,value in [('opening_cumulative_seconds_decimal','5438.82363936330092'),
                            ('terminal_parent_seconds_decimal','259.947951'),('nested_seconds_added',True),
                            ('cumulative_accounted_seconds_decimal','0')]:
            req,d=self.fixture();d['settlement'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):build_revision(req,d)


if __name__=='__main__':unittest.main()
