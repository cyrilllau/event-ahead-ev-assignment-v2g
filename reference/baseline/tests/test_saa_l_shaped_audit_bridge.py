"""Genuine L-shaped LP events, unchanged original SAA full-Q auditor, no new LP in audit."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_audit_bridge as bridge
from evcs_experiments import saa_fixed_plan_audit as audit
from evcs_experiments import recourse, recourse_check, first_stage_check, saa_fixed_plan
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.config import get_paths
import test_saa_fixed_plan_audit as fixtures


class LShapedAuditBridgeTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SAAQAuditTests()
        retained=os.environ.get('EVCS_SAA_BRIDGE_TEST_ARTIFACTS_ROOT')
        if retained:
            Path(retained).mkdir(parents=True,exist_ok=True)
            original=tempfile.TemporaryDirectory
            def keep(*args,**kwargs):
                value=original(dir=retained,prefix=self._testMethodName+'-')
                value._finalizer.detach();value.cleanup=lambda:None
                return value
            with patch.object(tempfile,'TemporaryDirectory',side_effect=keep):self.f.setUp()
        else:self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.toy = self.f.f.toy; t = self.toy
        self.problem = kernel.DevelopmentSAAProblem(t.case, t.tasks, t.scenarios, t.scenario_cost_constants_usd - 100.)
        ident = self.problem.identity
        self.args = dict(case=t.case, tasks=t.tasks, scenario_ids=ident['support_ids'],
            scenario_sha256s=ident['scenario_sha256s'], template_sha256s=ident['template_sha256s'],
            constants_usd=ident['constants_usd'], scene_loader=t.scenarios.__getitem__,
            input_binding={'fixture': 'actual golden0 original matrices', 'task_sha256':t.tasks.task_sha256},
            source_references=[file_reference(p) for p in (bridge.__file__, kernel.__file__, recourse.__file__,
                recourse_check.__file__, first_stage_check.__file__, saa_fixed_plan.__file__)], output_root=(self.f.root/'l_shaped_events').resolve())
        self.sink = bridge.DevelopmentFixedQEvidence(**self.args)

    def test_real_kernel_events_original_audit_same_plan_negative_constants_once(self):
        state = kernel.run_saa_l_shaped(self.problem, evidence_sink=self.sink)
        self.assertEqual(state['status'], 'CONVERGED')
        best = state['best']; pd = self.problem.plan(best['anchor'])
        digest = canonical_hash(dict(x=pd.x.tolist(), r=pd.r.tolist(), task_sha256=pd.task_sha256, config_hash=pd.config_hash))
        candidates = [i for i,s in self.sink.sweeps.items() if s['plan_sha256']==digest and len(s['q'])==3]
        self.assertTrue(candidates)
        cp = self.sink.checkpoint(candidates[0]); row = json.loads(Path(cp['path']).read_text())
        env = audit.bind_saa_template_envelope(request_reference=row['request_reference'],
            template_references=self.f.templates, first_stage_sha256=self.problem.first_stage.matrix_sha256,
            source_references=self.args['source_references'], output_path=self.f.root/'l_shaped_envelope.json')
        with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q solve')):
            result = audit.audit_saa_fixed_plan_q(**{**self.f.args, 'checkpoint_reference':cp,
                'expected_request_reference':row['request_reference'], 'template_envelope_reference':env,
                'deadline':time.monotonic()+30})
        np.testing.assert_allclose(result['q_usd'],best['q_usd'],rtol=0,atol=1e-9)
        self.assertAlmostEqual(result['upper_bound_usd'], state['upper_bound_usd'])
        self.assertAlmostEqual(result['upper_bound_usd'],30.266666666666666-100.)
        self.assertEqual(result['probability_optimization_calls'],0)
        self.assertEqual(result['solver_calls'],0)
        self.assertTrue(result['full_support_recourse_verified'])
        self.assertFalse(result['master_bound_audited']); self.assertFalse(result['budget_settled'])
        self.assertFalse(result['formal_OOS']); self.assertFalse(result['scientific_spec_pass'])
        audit._write(self.f.root/'l_shaped_state.json',state)
        audit._write(self.f.root/'original_full_q_audit.json',result)
        audit._write(self.f.root/'bridge_test_summary.json',dict(
            status='PASS_DEVELOPMENT_REAL_L_SHAPED_EVENTS_ORIGINAL_FULL_Q_AUDIT',
            source_references=self.args['source_references'],
            checkpoint_reference=cp,template_envelope_reference=env,
            state_reference=file_reference(self.f.root/'l_shaped_state.json'),
            audit_reference=file_reference(self.f.root/'original_full_q_audit.json'),
            master_bound_audited=False,budget_settled=False,production_ready=False,formal_OOS=False))
        self.assertTrue(row['external_timekeeper_required'])
        self.assertNotIn('cumulative_provisional_seconds',row)
        with self.assertRaisesRegex(ValueError,'uniform'):
            audit.audit_saa_fixed_plan_q(**{**self.f.args, 'checkpoint_reference':cp,
                'expected_request_reference':row['request_reference'], 'template_envelope_reference':env,
                'empirical_probabilities':[.2,.3,.5], 'deadline':time.monotonic()+30})

    def event(self, i=0, iteration=1, result=None, template=None, plan=None):
        template = self.problem.recourses[i] if template is None else template
        plan = self.f.f.plan if plan is None else plan
        if result is None: result=recourse.solve_fixed_plan(template,plan)
        self.sink(iteration=iteration,scenario_index=i,template=template,plan=plan,result=result)

    def test_partial_duplicate_and_foreign_template_rejected(self):
        self.event()
        with self.assertRaisesRegex(ValueError,'Incomplete'): self.sink.checkpoint(1)
        with self.assertRaisesRegex(ValueError,'duplicate'): self.event()
        with self.assertRaisesRegex(ValueError,'template'): self.event(i=1,template=self.problem.recourses[0])

    def test_failed_native_certificate_retained_and_never_completed(self):
        raw=recourse.solve_fixed_plan(self.problem.recourses[0],self.f.f.plan)
        bad=replace(raw,optimal=False,solver_status=1,status='LIMIT_REACHED')
        self.event(result=bad)
        self.assertEqual(self.sink.sweeps[1]['q'],{})
        record=json.loads(Path(self.sink.sweeps[1]['rows'][0]['path']).read_text())
        self.assertFalse(record['completed']);self.assertEqual(record['native_status'],'LIMIT_REACHED')
        with self.assertRaisesRegex(ValueError,'Incomplete'): self.sink.checkpoint(1)

    def test_kernel_native_limit_keeps_partial_checkpoint_and_raw_event(self):
        actual=kernel.solve_fixed_plan;calls=[]
        def limited(template,plan,**kwargs):
            calls.append(template.scenario_sha256)
            raw=actual(template,plan,**kwargs)
            if len(calls)==2:
                return replace(raw,optimal=False,solver_status=1,status='LIMIT_REACHED',
                               solution=None,objective_usd=None,
                               certificate={'optimal_certificate_valid':False})
            return raw
        checkpoint=self.f.root/'partial_kernel.json'
        with patch.object(kernel,'solve_fixed_plan',side_effect=limited):
            state=kernel.run_saa_l_shaped(self.problem,evidence_sink=self.sink,checkpoint_path=checkpoint)
        self.assertEqual(state['status'],'TIME_LIMIT')
        self.assertEqual(len(state['pending']['atoms']),1)
        self.assertEqual(state['last_incomplete_recourse']['solver_status'],1)
        self.assertEqual(sorted(self.sink.sweeps[1]['q']),[0])
        self.assertEqual(len(self.sink.sweeps[1]['rows']),2)
        self.assertTrue(checkpoint.is_file())
        with self.assertRaisesRegex(ValueError,'Incomplete'): self.sink.checkpoint(1)

    def test_formal_source_and_support_identity_rejected(self):
        for delta,pattern in (({'domain':'formal'},'development'),({'source_references':[]},'source'),
            ({'scenario_ids':list(reversed(self.args['scenario_ids']))},'order'),
            ({'scenario_sha256s':['0'*64]*3},'source'),
            ({'output_root':'relative-events'},'absolute'),
            ({'output_root':get_paths().data_root/'forbidden-audit-events'},'frozen'),
            ({'output_root':get_paths().project_root/'src'/'forbidden-audit-events'},'source')):
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError,pattern):
                bridge.DevelopmentFixedQEvidence(**{**self.args,**delta})

if __name__=='__main__': unittest.main()
