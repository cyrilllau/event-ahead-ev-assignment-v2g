import copy
import importlib.util
import unittest
from evcs_experiments.config import get_paths

spec = importlib.util.spec_from_file_location('multi_anchor_driver', get_paths().project_root / 'scripts/diagnostics/s05_multi_anchor_bound.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class MultiAnchorDriverTests(unittest.TestCase):
    def inputs(self):
        s = dict(kind='S05_CANDIDATE_COMPLETION_PARENT_ACCOUNTING_V1', status='SETTLED_NUMERICAL_CANDIDATE_PENDING_FORMAL_PLAN_LOCK',
            accounting_PASS=True, parent_charged_once=True, nested_seconds_added=False, cumulative_seconds_provisional=False,
            opening_cumulative_seconds_decimal='5337.02842836330092', terminal_parent_seconds_decimal='101.795211',
            cumulative_accounted_seconds_decimal=str(d.OPENING), remaining_seconds_decimal=str(d.REMAINING),
            cumulative_cap_seconds=7200, original_cumulative_cap_seconds=7200,
            cumulative_accounted_seconds=float(d.OPENING), remaining_seconds=float(d.REMAINING),
            plan_id=d.PLAN_ID, plan_sha256=d.PLAN_SHA256, original_problem_sha256='identity')
        r = dict(kind=d.KIND, plan_id=d.PLAN_ID, parent_job=dict(timeout_seconds=1000, grace_seconds=5, threads=1),
            wall_seconds=990, solver_seconds=600, post_master_reserve_seconds=100, preprocessing_charge_seconds=0,
            references={'settlement': {'path':'/settlement','sha256':'hash','size_bytes':1}})
        a = dict(status='PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW', passed=True,
            plan_id=d.PLAN_ID, plan_sha256=d.PLAN_SHA256, upper_bound_usd=d.FIXED_UB,
            lower_bound_usd=d.PREVIOUS_LB, original_problem_sha256='identity', normal_ccg_checkpoint_compatible=False,
            references={'settlement': r['references']['settlement']})
        return r,s,a

    def test_exact_opening_and_limits(self):
        self.assertEqual(d.budget_guard(*self.inputs()), d.OPENING)
        for field,value in [('solver_seconds',601),('post_master_reserve_seconds',99),('wall_seconds',1000),('preprocessing_charge_seconds',1)]:
            r,s,a=self.inputs();r[field]=value
            with self.assertRaises(ValueError):d.budget_guard(r,s,a)

    def test_relabel_balance_candidate_or_review_rejected(self):
        for field,value in [('cumulative_accounted_seconds_decimal','0'),('plan_sha256','other'),('nested_seconds_added',True)]:
            r,s,a=self.inputs();s[field]=value
            with self.assertRaises(ValueError):d.budget_guard(r,s,a)
        r,s,a=self.inputs();a['references']['settlement']={}
        with self.assertRaises(ValueError):d.budget_guard(r,s,a)

    def test_actual_prep_subtracted_and_cutoff(self):
        r,_,_=self.inputs()
        self.assertEqual(d.solver_grant(r,990,200),600)
        self.assertEqual(d.solver_grant(r,990,500),390)
        with self.assertRaises(ValueError):d.solver_grant(r,990,890)

    def test_bound_only_retains_original_plan_ub_and_best_lb(self):
        row=d.bound_result(20000, audited=True)
        self.assertEqual(row['lower_bound_usd'],d.PREVIOUS_LB)
        self.assertEqual(row['upper_bound_usd'],d.FIXED_UB)
        self.assertFalse(row['normal_ccg_checkpoint_compatible'])
        self.assertEqual(d.bound_result(30000,audited=True)['lower_bound_usd'],30000)
        for value in [float('nan'),float('inf'),d.FIXED_UB+1]:
            with self.assertRaises(ValueError):d.bound_result(value,audited=True)
        with self.assertRaises(ValueError):d.bound_result(30000,audited=False)

    def test_terminal_actual_once_decimal_not_nested(self):
        _,s,_=self.inputs();r=d.settle_parent_seconds(s,'1000')
        self.assertEqual(r['remaining_seconds_decimal'],'761.17636063669908')
        with self.assertRaises(ValueError):d.settle_parent_seconds(s,'1762')

if __name__=='__main__':unittest.main()
