"""Real small planning -> independent audits within one worker deadline."""
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.formal_planning_worker import run_attempt
import test_formal_planning_worker as fixtures


class PlanningAuditPipelineTests(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.FormalPlanningWorkerTests(methodName='runTest')
        self.h.setUpClass();self.h.setUp();self.addCleanup(self.h.doCleanups)
        root=get_paths().project_root
        self.h.request['numerical_audit']=dict(schema_version=1,postprocessing_reserve_seconds=.8,
            sources=dict(helper=file_reference(root/'scripts/audits/audit_s05_pending_recovery.py'),
                         numerical_kernel=file_reference(root/'scripts/audits/audit_s05_candidate_recourse.py')))
        self.h.write(self.h.request_path,self.h.request)

    def test_real_worker_finishes_full_audit_with_true_bounds(self):
        result=run_attempt(self.h.invocation(),_development_fixture=self.h.fixture())
        self.assertEqual(result['status'],'COMPLETE')
        self.assertTrue(result['full_support_verified'])
        audit=json.loads(Path(result['numerical_audit_reference']['path']).read_text())
        self.assertTrue(audit['passed'])
        self.assertAlmostEqual(audit['upper_bound_usd'],self.h.f.expected['DRO']['objective_usd'],places=7)
        self.assertLessEqual(audit['lower_bound_usd'],audit['upper_bound_usd']+1e-7)
        self.assertFalse(result['scientific_acceptance']);self.assertFalse(result['formal_OOS'])

    def test_actual_audit_failure_halts_worker_without_success(self):
        with patch('evcs_experiments.planning_audit_pipeline.audit_fixed_plan_recourses',side_effect=ValueError('actual numerical failure')):
            with self.assertRaisesRegex(ValueError,'actual numerical failure'):
                run_attempt(self.h.invocation(),_development_fixture=self.h.fixture())
        self.assertFalse((self.h.root/'attempt_1/worker_result.json').exists())
        self.assertTrue((self.h.root/'attempt_1/worker_failure.json').exists())

    def test_audit_reserve_cannot_increase_frozen_evidence_budget(self):
        self.h.request['numerical_audit']['postprocessing_reserve_seconds']=2.
        self.h.write(self.h.request_path,self.h.request)
        with self.assertRaisesRegex(ValueError,'Audit reserve exceeds'):
            run_attempt(self.h.invocation(),_development_fixture=self.h.fixture())

    def test_candidate_native_status_tamper_is_independently_rejected(self):
        from evcs_experiments.formal_planning_worker import _candidate
        def changed(*args,**kwargs):
            result=_candidate(*args,**kwargs)
            result['candidate']['native_master_status']='FAKE_OPTIMAL'
            return result
        with patch('evcs_experiments.formal_planning_worker._candidate',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'original master summary'):
                run_attempt(self.h.invocation(),_development_fixture=self.h.fixture())

    def test_partial_q_resume_finishes_audit_in_same_followup_attempt(self):
        from dataclasses import replace
        from evcs_experiments.recourse import solve_fixed_plan
        calls=[0]
        def bounded(template,plan,**kw):
            calls[0]+=1;value=solve_fixed_plan(template,plan,**kw)
            return value if calls[0]==1 else replace(value,optimal=False,solver_status=1,status='LIMIT_REACHED',
                certificate={**value.certificate,'optimal_certificate_valid':False})
        with patch('evcs_experiments.ccg.solve_fixed_plan',side_effect=bounded):
            first=run_attempt(self.h.invocation(),_development_fixture=self.h.fixture())
        self.assertEqual(first['status'],'TIME_LIMIT')
        checkpoint=json.loads(Path(first['checkpoint']['path']).read_text())
        previous=json.loads(Path(checkpoint['engine_checkpoint_snapshot']['path']).read_text())['state']
        self.assertEqual(len(previous['pending']['q_usd']),1)
        old_snapshot=json.loads(Path(checkpoint['raw_evidence_snapshot']['path']).read_text())
        second=run_attempt(self.h.invocation(2,prior=first['cumulative_accounted_seconds']+.01,previous=first),_development_fixture=self.h.fixture())
        self.assertEqual(second['status'],'COMPLETE')
        self.assertTrue(second['full_support_verified'])
        # Resume kept every original callback byte; no relabelled evidence.
        for ref in old_snapshot['events']:
            self.assertEqual(file_reference(ref['path']),ref)
