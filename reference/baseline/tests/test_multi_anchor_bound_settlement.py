from decimal import Decimal
import importlib.util
import unittest
from evcs_experiments.config import get_paths

spec = importlib.util.spec_from_file_location('bound_settlement', get_paths().project_root/'scripts/audits/settle_s05_multi_anchor_bound.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BoundSettlementTests(unittest.TestCase):
    def values(self):
        p = dict(kind='S05_CANDIDATE_COMPLETION_PARENT_ACCOUNTING_V1',
            status='SETTLED_NUMERICAL_CANDIDATE_PENDING_FORMAL_PLAN_LOCK',
            cumulative_cap_seconds=7200, original_cumulative_cap_seconds=7200, plan_sha256=m.PLAN_SHA256,
            accounting_PASS=True, parent_charged_once=True, nested_seconds_added=False,
            cumulative_seconds_provisional=False, plan_id=m.PLAN_ID,
            cumulative_accounted_seconds_decimal=str(m.OPENING), remaining_seconds_decimal='1761.17636063669908')
        t = dict(state='SUCCEEDED', complete=True, elapsed_seconds=Decimal('999.123456'))
        l = dict(blocks_new_launch=False, command_alive=False, supervisor_alive=False, live_group_members=[])
        return p,t,l

    def test_exact_decimal_and_no_nested_charge(self):
        p,t,l = self.values(); t['phase_seconds'] = 9000
        r = m.account_parent(p,t,l)
        self.assertEqual(r['cumulative_accounted_seconds_decimal'], '6437.94709536330092')
        self.assertEqual(r['remaining_seconds_decimal'], '762.05290463669908')
        self.assertFalse(r['numerical_acceptance'])

    def test_failed_and_timeout_parent_still_charged(self):
        for status in ('FAILED','TIMED_OUT','CANCELLED'):
            p,t,l = self.values();t['state'] = status
            self.assertEqual(m.account_parent(p,t,l)['terminal_parent_seconds_decimal'], '999.123456')

    def test_live_or_unsettled_parent_refused(self):
        p,t,l = self.values();l['command_alive'] = True
        with self.assertRaises(ValueError):m.account_parent(p,t,l)
        p,t,l = self.values();p['cumulative_seconds_provisional'] = True
        with self.assertRaises(ValueError):m.account_parent(p,t,l)

    def test_no_budget_reset_or_cap_escape(self):
        p,t,l = self.values();p['cumulative_accounted_seconds_decimal'] = '0'
        with self.assertRaises(ValueError):m.account_parent(p,t,l)
        p,t,l = self.values();t['elapsed_seconds'] = Decimal('1762')
        with self.assertRaises(ValueError):m.account_parent(p,t,l)


if __name__ == '__main__':unittest.main()
