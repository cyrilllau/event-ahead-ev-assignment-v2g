"""Real physical fixed-plan LPs and immutable partial-Q resume boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments import saa_fixed_plan as m
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.validation.s03_toys import build_golden_cases


class FixedSAAQTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'q';self.toy=build_golden_cases()[0];t=self.toy
        self.templates=[assemble_recourse(t.case,t.tasks,s) for s in t.scenarios]
        self.plan=FirstStagePlan(x=np.zeros(len(t.tasks.task_ids)),r=np.zeros((len(t.tasks.task_ids),len(t.case.periods))),
            task_sha256=t.tasks.task_sha256,config_hash=t.tasks.config_hash)
        self.kw=dict(plan=self.plan,case=t.case,tasks=t.tasks,scenario_ids=t.scenario_ids,
            scenario_sha256s=[s.scenario_sha256 for s in t.scenarios],
            template_sha256s=[r.template_sha256 for r in self.templates],
            scenario_cost_constants_usd=t.scenario_cost_constants_usd-100,
            template_loader=self.templates.__getitem__,scene_loader=t.scenarios.__getitem__,
            source_references=[file_reference(p) for p in (m.__file__,m.recourse.__file__,m.recourse_check.__file__,m.first_stage_check.__file__)],input_binding={'fixture':'golden source'},
            output_directory=self.root,cumulative_budget_seconds=30,deadline=time.monotonic()+30)

    def run_q(self,**kw):return m.complete_saa_fixed_plan_q(**{**self.kw,**kw})

    def test_real_full_q_raw_witnesses_and_negative_constants(self):
        result=self.run_q();self.assertEqual(result['status'],'COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT')
        expected=[solve_fixed_plan(t,self.plan).objective_usd+c for t,c in zip(self.templates,self.kw['scenario_cost_constants_usd'])]
        np.testing.assert_allclose(result['q_usd'],expected,rtol=0,atol=1e-9)
        for ref in result['attempts']:
            row=json.loads(Path(ref['path']).read_text());self.assertTrue(row['completed'])
            self.assertTrue((Path(ref['path']).parent/'dual_kkt.npz').is_file())
        self.assertFalse(result['full_support_verified']);self.assertFalse(result['formal_OOS'])

    def partial(self):
        calls=[]
        def solve(template,plan,**kw):
            calls.append(template.scenario_sha256)
            raw=solve_fixed_plan(template,plan,**kw)
            if len(calls)==2:
                return replace(raw,status='LIMIT_REACHED',solver_status=1,optimal=False,solution=None,objective_usd=None,
                               certificate=MappingProxyType({'optimal_certificate_valid':False}))
            return raw
        with patch.object(m,'solve_fixed_plan',side_effect=solve):result=self.run_q()
        self.assertEqual(result['status'],'TIME_LIMIT');self.assertEqual(result['completed_indices'],[0])
        return result

    def test_partial_resume_skips_completed_and_preserves_prior_time(self):
        old=self.partial();calls=[]
        def solve(template,plan,**kw):calls.append(template.scenario_sha256);return solve_fixed_plan(template,plan,**kw)
        prior=old['cumulative_provisional_seconds']+.2
        with patch.object(m,'solve_fixed_plan',side_effect=solve):
            result=self.run_q(resume_checkpoint_reference=old['checkpoint_reference'],prior_accounted_seconds=prior)
        self.assertEqual(len(calls),2);self.assertNotIn(self.templates[0].scenario_sha256,calls)
        self.assertGreaterEqual(result['cumulative_provisional_seconds'],prior)
        self.assertEqual(len(result['attempts']),4)
        self.assertEqual(file_reference(old['checkpoint_reference']['path']),old['checkpoint_reference'])
        with self.assertRaisesRegex(ValueError,'Stale|changed'):
            self.run_q(resume_checkpoint_reference=old['checkpoint_reference'],prior_accounted_seconds=prior)

    def test_changed_plan_or_reordered_support_refused(self):
        old=self.partial();prior=old['cumulative_provisional_seconds']
        changed=FirstStagePlan(x=np.ones(len(self.plan.x)),r=self.plan.r,
            task_sha256=self.plan.task_sha256,config_hash=self.plan.config_hash)
        for delta in ({'plan':changed}, {'scenario_ids':list(reversed(self.kw['scenario_ids'])),
            'scenario_sha256s':list(reversed(self.kw['scenario_sha256s'])),
            'template_sha256s':list(reversed(self.kw['template_sha256s']))}):
            with self.subTest(delta=delta),self.assertRaisesRegex(ValueError,'changed|first-stage'):
                self.run_q(resume_checkpoint_reference=old['checkpoint_reference'],prior_accounted_seconds=prior,**delta)

    def test_tampered_saved_primal_rejected(self):
        old=self.partial();p=Path(old['attempts'][0]['path']).parent/'primal.npy';b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
        with self.assertRaisesRegex(ValueError,'changed'):
            self.run_q(resume_checkpoint_reference=old['checkpoint_reference'],prior_accounted_seconds=old['cumulative_provisional_seconds'])

    def test_prior_reset_or_changed_input_refused(self):
        old=self.partial()
        with self.assertRaisesRegex(ValueError,'reset'):
            self.run_q(resume_checkpoint_reference=old['checkpoint_reference'],prior_accounted_seconds=0)
        with self.assertRaisesRegex(ValueError,'changed'):
            self.run_q(resume_checkpoint_reference=old['checkpoint_reference'],prior_accounted_seconds=old['cumulative_provisional_seconds'],input_binding={'fixture':'other'})

    def test_invalid_optimal_certificate_retained_but_not_completed(self):
        raw=solve_fixed_plan(self.templates[0],self.plan)
        bad=replace(raw,certificate=MappingProxyType({**raw.certificate,'optimal_certificate_valid':False}))
        with patch.object(m,'solve_fixed_plan',return_value=bad),self.assertRaisesRegex(ValueError,'certificate'):
            self.run_q()
        saved=json.loads((self.root/'checkpoint.json').read_text())
        self.assertEqual(saved['status'],'FAILED');self.assertEqual(saved['completed_indices'],[])
        self.assertEqual(len(saved['attempts']),1)

    def test_invalid_limit_witness_fails_instead_of_resuming(self):
        raw=replace(solve_fixed_plan(self.templates[0],self.plan),status='LIMIT_REACHED',solver_status=1,optimal=False)
        with patch.object(m,'solve_fixed_plan',return_value=raw),patch.object(m,'check_recourse',return_value={'passed':False}),self.assertRaisesRegex(ValueError,'physical'):
            self.run_q()
        saved=json.loads((self.root/'checkpoint.json').read_text())
        self.assertEqual(saved['status'],'FAILED');self.assertEqual(len(saved['attempts']),1)

    def test_loader_consumes_absolute_deadline_without_new_lp(self):
        clock=[0.]
        def load(i):
            clock[0]=9.
            return self.templates[i]
        with patch.object(m.time,'monotonic',side_effect=lambda:clock[0]),patch.object(m,'solve_fixed_plan') as solve:
            result=self.run_q(template_loader=load,deadline=10.)
        solve.assert_not_called();self.assertEqual(result['status'],'TIME_LIMIT')
        self.assertEqual(result['cumulative_provisional_seconds'],9.)

    def test_export_deadline_overrun_preserves_failed_raw_attempt(self):
        clock=[0.];raw=solve_fixed_plan(self.templates[0],self.plan)
        def solve(*a,**k):
            clock[0]=11.
            return raw
        with patch.object(m.time,'monotonic',side_effect=lambda:clock[0]),patch.object(m,'solve_fixed_plan',side_effect=solve):
            with self.assertRaisesRegex(TimeoutError,'deadline'):self.run_q(deadline=10.)
        saved=json.loads((self.root/'checkpoint.json').read_text())
        self.assertEqual(saved['status'],'FAILED');self.assertEqual(len(saved['attempts']),1)
        self.assertEqual(saved['cumulative_provisional_seconds'],11.)

    def test_first_q_call_retains_earlier_master_charge(self):
        result=self.run_q(prior_accounted_seconds=3.)
        self.assertGreaterEqual(result['cumulative_provisional_seconds'],3.)
        self.assertEqual(result['prior_accounted_seconds'],3.)

    def test_expired_deadline_no_solver_and_formal_full_support_gate(self):
        with patch.object(m,'solve_fixed_plan') as solve:
            with self.assertRaisesRegex(ValueError,'remaining'):self.run_q(deadline=time.monotonic()-1)
            with self.assertRaisesRegex(ValueError,'160'):self.run_q(domain='formal')
            solve.assert_not_called()

if __name__=='__main__':unittest.main()
