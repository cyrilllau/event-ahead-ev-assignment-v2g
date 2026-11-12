"""New receipt-derived accounting cannot reach OOS without its pinned seal."""
import json
from pathlib import Path
import tempfile
import unittest
from evcs_experiments.evaluation_lock import file_reference, verify_formal_numerical_records


class GenericAccountingGateTests(unittest.TestCase):
    def test_cutoff_unconfirmed_and_disabled_seal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);policy={'cumulative_wall_seconds':7200}
            def write(name,value):
                path=root/name;path.write_text(json.dumps(value));return file_reference(path)
            for final,requires in ((False,None),(True,True),(True,False),(True,None)):
                with self.subTest(final=final,requires=requires):
                    numeric=dict(domain='formal',plan_sha256='p',full_support_verified=True,
                        status='TIME_LIMIT',lower_bound_usd=1,upper_bound_usd=2,
                        cumulative_wall_seconds=20,final_budget_acceptance=final)
                    if requires is not None:numeric['requires_seal_confirmation']=requires
                    refs=dict(solver_record=write('solver.json',numeric),
                        budget=write('budget.json',dict(planning_budget=policy,final_budget_acceptance=final)))
                    with self.assertRaises(ValueError):
                        verify_formal_numerical_records(dict(configuration=dict(planning_budget=policy)),
                            dict(plans=[dict(plan_sha256='p',references=refs)]))

    def test_strong_matching_seal_passes_but_tamper_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);policy={'cumulative_wall_seconds':7200}
            def write(name,value):
                path=root/name;path.write_text(json.dumps(value));return file_reference(path)
            refs=dict(solver_record=write('solver.json',dict(domain='formal',plan_sha256='p',
                full_support_verified=True,status='TIME_LIMIT',lower_bound_usd=1,upper_bound_usd=2,
                cumulative_wall_seconds=20,final_budget_acceptance=True,requires_seal_confirmation=True,
                charged_attempt_seconds=10,actual_attempt_seconds_at_cutoff=8)),
                budget=write('budget.json',dict(planning_budget=policy,final_budget_acceptance=True,charged_attempt_seconds=10)))
            seal=dict(status='PASS_FINAL_RECORD_SEAL',references=dict(refs),
                actual_attempt_seconds_before_confirmation_write=9,charged_attempt_seconds=10)
            refs['budget_seal']=write('seal.json',seal)
            protocol=dict(configuration=dict(planning_budget=policy))
            batch=dict(plans=[dict(plan_sha256='p',references=refs)])
            verify_formal_numerical_records(protocol,batch)
            for actual,charged in ((float('nan'),10),(9,float('nan')),(9,11),(7,10),(-1,10)):
                with self.subTest(actual=actual,charged=charged):
                    changed=dict(seal,actual_attempt_seconds_before_confirmation_write=actual,
                        charged_attempt_seconds=charged)
                    refs['budget_seal']=write('seal.json',changed)
                    with self.assertRaises(ValueError):verify_formal_numerical_records(protocol,batch)
            refs['budget_seal']=write('seal.json',seal)
            seal['charged_attempt_seconds']=8;write('seal.json',seal)
            with self.assertRaises(ValueError):verify_formal_numerical_records(protocol,batch)
