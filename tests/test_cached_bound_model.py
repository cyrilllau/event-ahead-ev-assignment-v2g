"""Bounded admission failures and exact failed-parent carryover; no full cache load."""
from decimal import Decimal
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from evcs_experiments import cached_bound_model as m
from evcs_experiments.campaign_batch import file_reference


class CachedBoundBudgetTests(unittest.TestCase):
    def fixture(self):
        prior_ref=dict(path='/original/settlement.json',sha256='a'*64,size_bytes=1)
        settled=dict(kind='S05_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1',
            status='SETTLED_BOUND_EVIDENCE_PENDING_FINAL_RESULT_REVIEW',terminal_state='FAILED',
            parent_job_id='s05-multi-anchor-r22-20260909T053500Z',plan_id=m.PLAN_ID,
            accounting_PASS=True,parent_charged_once=True,nested_seconds_added=False,
            cumulative_seconds_provisional=False,opening_cumulative_seconds_decimal=str(m.OLD_OPENING),
            terminal_parent_seconds_decimal=str(m.FAILED_SECONDS),cumulative_accounted_seconds_decimal=str(m.OPENING),
            remaining_seconds_decimal=str(m.REMAINING),cumulative_cap_seconds=7200,previous_settlement_reference=prior_ref)
        review=dict(passed=True,status='PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW',
            plan_id=m.PLAN_ID,plan_sha256=m.PLAN_SHA,upper_bound_usd=227762.10627548306,
            lower_bound_usd=24135.73087520793,references={'settlement':prior_ref})
        request=dict(kind=m.KIND,plan_id=m.PLAN_ID,parent_job=dict(timeout_seconds=1000,grace_seconds=5,threads=1),
            wall_seconds=990,solver_seconds=600,post_master_reserve_seconds=100,preprocessing_charge_seconds=0)
        return request,settled,review

    def test_exact_failed_parent_cost_preserved_not_nested_time(self):
        request,settled,review=self.fixture();settled['nested_phase_seconds']=9999.
        self.assertEqual(m.budget_guard(request,settled,review),Decimal('5633.96054736330092'))
        self.assertEqual(m.OLD_OPENING+m.FAILED_SECONDS,m.OPENING)
        self.assertEqual(Decimal(7200)-m.OPENING,m.REMAINING)

    def test_no_reset_omitted_failure_double_charge_or_false_terminal(self):
        for key,value in [('terminal_parent_seconds_decimal','0'),('terminal_parent_seconds_decimal','390.273816'),
                          ('cumulative_accounted_seconds_decimal',str(m.OLD_OPENING)),
                          ('remaining_seconds_decimal','1761.17636063669908'),
                          ('nested_seconds_added',True),('terminal_state','SUCCEEDED'),
                          ('parent_job_id','another-job'),('cumulative_seconds_provisional',True)]:
            request,settled,review=self.fixture();settled[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.budget_guard(request,settled,review)

    def test_candidate_review_and_predecessor_are_not_relabelled(self):
        for key,value in [('upper_bound_usd',1.),('lower_bound_usd',1.),('plan_sha256','other'),
                          ('status','PASS'),('references',{'settlement':{}})]:
            request,settled,review=self.fixture();review[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.budget_guard(request,settled,review)

    def test_limits_and_unaccounted_external_preprocessing_rejected(self):
        for key,value in [('wall_seconds',1200),('solver_seconds',601),('post_master_reserve_seconds',0),
                          ('preprocessing_charge_seconds',1)]:
            request,settled,review=self.fixture();request[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.budget_guard(request,settled,review)


class CachedBoundEarlyLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def write(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return file_reference(p)
    def test_expired_deadline_precedes_io(self):
        with self.assertRaisesRegex(ValueError,'deadline'):m.load_cached_model({'references':{'cached_model_review':{}}},deadline=time.monotonic()-1)
    def test_changed_review_bytes_rejected(self):
        ref=self.write('review.json',{'kind':'R22_PARTIAL_CUT_MODEL_REUSE_REVIEW_V1','passed':True})
        Path(ref['path']).write_text('{}')
        with self.assertRaises(ValueError):m.load_cached_model({'references':{'cached_model_review':ref}},deadline=time.monotonic()+2)
    def test_review_status_not_inferred_from_file_existence(self):
        ref=self.write('review.json',{'kind':'R22_PARTIAL_CUT_MODEL_REUSE_REVIEW_V1','passed':False})
        with self.assertRaisesRegex(ValueError,'review'):m.load_cached_model({'references':{'cached_model_review':ref}},deadline=time.monotonic()+2)
    def test_rehashed_original_request_wrong_kind_is_rejected(self):
        old=self.write('request.json',{'kind':'OTHER','plan_id':m.PLAN_ID})
        card=self.write('review.json',{'kind':'R22_PARTIAL_CUT_MODEL_REUSE_REVIEW_V1','passed':True,
                                    'references':{'original_request':old}})
        with self.assertRaisesRegex(ValueError,'original problem'):
            m.load_cached_model({'references':{'cached_model_review':card}},deadline=time.monotonic()+2)


if __name__=='__main__':unittest.main()
