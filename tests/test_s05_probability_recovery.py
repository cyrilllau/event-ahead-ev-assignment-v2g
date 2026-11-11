"""Bounded local bridge tests; no formal bank or full-scale optimizer access."""
from __future__ import annotations
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.probability import ProbabilitySet, solve_worst_distribution
from evcs_experiments.probability_check import check_probability_solution

ROOT = get_paths().project_root
spec = importlib.util.spec_from_file_location('s05_probability_recovery', ROOT/'scripts/recovery/prepare_s05_probability_recovery.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    f = json.loads((ROOT/'tests/fixtures/probability_michael_160q_numerical.json').read_text())
    a = f['probability_set']; ids = a['support_ids']; q = f['q_usd']
    plan = dict(x=[0.], r=[[0.]], task_sha256='frozen-task', config_hash='frozen-config')
    pending = dict(iteration=1, plan=plan, q_usd=q, active_indices=[67,87,98,137],
        recourse_certificates=[dict(index=i, scenario_id=s, optimal_certificate_valid=True, template_sha256=f'template-{i}') for i,s in enumerate(ids)],
        rmp=dict(status='TIME_LIMIT', optimal=False, lower_bound_usd=None, certificate=dict(solver_bound_usable=False)),
        last_incomplete_probability=f['original_failed_result'],
        raw_rmp_dual=dict(theta0=-0.006948140522745174,
            alpha=[0.007307694022698023,2.041984701440713e-11,0.,0.,0.],
            beta=[0.,0.0084369505195964,0.,0.,231461.27546292375], threshold=None))
    state = dict(status='PROBABILITY_FAILED', trace=[], next_iteration=1, best=None, lower_bound_usd=None, upper_bound_usd=None,
                 identity=dict(scenario_ids=ids,effective_ambiguity_sha256=a['ambiguity_sha256']),
                 active_indices=pending['active_indices'], pending=pending,
                 config=dict(separation_absolute_usd=1e-4,separation_relative=1e-8))
    state['problem_sha256']=m.canonical_hash(state['identity'])
    return state, dict(support_ids=ids,q_usd=q,plan=plan), a


class PendingIdentityAndNumericalTests(unittest.TestCase):
    def test_only_exact_reviewed_probability_source_change_is_allowed(self):
        old={'probability.py':'original','ccg.py':'unchanged'}
        new=dict(old,**{'probability.py':m.PROBABILITY_SOURCE_SHA256})
        m.verify_source_transition(old,new)
        for changed in (dict(new,**{'ccg.py':'retagged'}),dict(new,**{'probability.py':'other-fix'}),dict(new,**{'extra.py':'new'})):
            with self.assertRaises(ValueError):m.verify_source_transition(old,changed)

    def test_real160_probability_repair_then_original_master_separation(self):
        state,qi,ai=fixture(); a,q=m.verify_pending(state,qi,ai)
        result=solve_worst_distribution(a,q,q_support_ids=a.support_ids,time_limit_seconds=5)
        self.assertTrue(result.certified_optimal)
        self.assertTrue(check_probability_solution(a,q,result,q_support_ids=a.support_ids).passed)
        derived=m.separate(state,a,q,0.,result.upper_bound_usd)
        self.assertEqual(derived['next_added_index'],60)
        self.assertLessEqual(derived['included_maximum_usd'],derived['separation_tolerance_usd'])
        self.assertGreater(derived['full_support_maximum_usd'],derived['separation_tolerance_usd'])
        self.assertAlmostEqual(derived['upper_bound_usd'],232648.24342916225,places=6)
        self.assertIsNone(derived['lower_bound_usd']);self.assertIsNone(derived['gap_relative'])
        self.assertFalse(derived['lockable']);self.assertEqual(derived['original_master_status'],'TIME_LIMIT')
        self.assertEqual(derived['original_rmp_dual'],state['pending']['raw_rmp_dual'])
        self.assertEqual(state['status'],'PROBABILITY_FAILED');self.assertIsNone(state['best'])

    def test_same_values_wrong_support_order_rejected(self):
        s,q,a=fixture();q['support_ids']=list(reversed(q['support_ids']))
        with self.assertRaisesRegex(ValueError,'q/plan/support'):m.verify_pending(s,q,a)

    def test_changed_q_fixed_plan_or_feature_digest_rejected(self):
        for field in ('q','plan','feature'):
            s,q,a=fixture()
            if field=='q':q['q_usd']=q['q_usd'].copy();q['q_usd'][0]+=1.
            elif field=='plan':q['plan']=dict(q['plan'],x=[1.])
            else:a['features'][0][0]+=1e-9
            with self.subTest(field=field),self.assertRaises(ValueError):m.verify_pending(s,q,a)

    def test_missing_support_false_certificate_and_fake_lower_bound_rejected(self):
        for change in ('missing','false','LB','threshold'):
            s,q,a=fixture()
            if change=='missing':s['pending']['recourse_certificates'].pop()
            elif change=='false':s['pending']['recourse_certificates'][3]['optimal_certificate_valid']=False
            elif change=='LB':s['pending']['rmp']['lower_bound_usd']=0.
            else:s['config']['separation_absolute_usd']=1.
            with self.subTest(change=change),self.assertRaises(ValueError):m.verify_pending(s,q,a)

    def test_bad_included_master_dual_is_not_promoted(self):
        s,qi,ai=fixture();a,q=m.verify_pending(s,qi,ai)
        s['pending']['raw_rmp_dual']['theta0']-=1.
        result=m.separate(s,a,q,0.,250000.)
        self.assertEqual(result['status'],'RMP_RESIDUAL_FAILED');self.assertIsNone(result['next_added_index'])
        self.assertFalse(result['lockable']);self.assertIsNone(result['lower_bound_usd'])


class AuditedProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.s,self.qi,self.ai=fixture()
        (self.root/'raw').mkdir()
        self.orig={k:m.write(self.root/(k+'.json'),dict(key=k)) for k in ('job_result','checkpoint','history','source_manifest')}
        self.orig['raw_request']=m.write(self.root/'raw/request.json',dict(key='raw_request'))
        self.raw=self.root/'same_size_raw.bin';self.raw.write_bytes(b'unchanged')
        self.rawref=m.reference(self.raw)
        self.out=self.root/'audit';self.out.mkdir()
        self.rmp=dict(status='TIME_LIMIT',lower_bound_usd=None,solver_bound_usable=False,physical_first_stage_cost_usd=0.)
        audit=dict(status=m.AUDIT_STATUS,completed_lp_count=160,expected_lp_count=160,solver_calls=0,formal_OOS=False,
                   probability_PASS=False,original_LB=None,original_engine_status='PROBABILITY_FAILED',rmp=self.rmp)
        self.refs=dict(full160_audit=m.write(self.out/'audit.json',audit),
                       audit_input_references=m.write(self.out/'input_references.json',list(self.orig.values())+[self.rawref]))
        m.write(self.out/'rmp.json',self.rmp);m.write(self.out/'raw_rmp_dual_residuals.json',{})
        for i,c in enumerate(self.s['pending']['recourse_certificates']):
            m.write(self.out/f'lp_{i:03d}.json',dict(index=i,scenario_id=c['scenario_id'],template_sha256=c['template_sha256'],
                q_usd=self.s['pending']['q_usd'][i],physical_passed=True,primal_residual=0.,stationarity=0.,dual_sign_violation=0.,
                infinite_bound_dual_violation=0.,duality_gap_usd=0.,primal_objective_usd=1.,dual_objective_usd=1.))
        self.refs['audit_artifact_manifest']=m.write(self.out/'artifact_manifest.json',[m.reference(p) for p in sorted(self.out.iterdir())])
        self.review=dict(status=m.REVIEW_STATUS,references={**self.orig,**self.refs},problem_sha256=self.s['problem_sha256'],
                         complete_lp_count=160,probability_PASS=False,formal_OOS=False)
        self.refs['independent_acceptance']=m.write(self.root/'acceptance.json',self.review)

    def test_complete_proof_refreshed_without_solver_calls(self):
        with patch.object(m,'solve_worst_distribution',side_effect=AssertionError('must not solve')):
            result=m.verify_audit(self.refs,self.s,[],self.orig)
        self.assertEqual(result[0]['completed_lp_count'],160)

    def test_same_size_original_raw_tamper_rejected(self):
        self.raw.write_bytes(b'tampered!')
        self.assertEqual(self.raw.stat().st_size,self.rawref['size_bytes'])
        with self.assertRaisesRegex(ValueError,'Evidence bytes changed'):m.verify_audit(self.refs,self.s,[],self.orig)

    def test_diagnostic_prefix_never_accepted_even_with_resigned_output(self):
        p=Path(self.refs['full160_audit']['path']);d=json.loads(p.read_text());d['status']='DIAGNOSTIC_PREFIX_ONLY_NOT_ACCEPTED'
        p.write_text(json.dumps(d));self.refs['full160_audit']=m.reference(p)
        with self.assertRaisesRegex(ValueError,'complete full160'):m.verify_audit(self.refs,self.s,[],self.orig)

    def test_changed_certificate_or_missing_proof_fails(self):
        p=self.out/'lp_159.json';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Evidence bytes changed'):m.verify_audit(self.refs,self.s,[],self.orig)


class BudgetClaimAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();s,qi,ai=fixture();a,q=m.verify_pending(s,qi,ai)
        self.paths_patch=patch.object(m,'get_paths',return_value=replace(get_paths(),campaign_root=self.root))
        self.paths_patch.start();self.addCleanup(self.paths_patch.stop)
        self.parent_patch=patch.object(m,'parent_binding',return_value=dict(deadline_monotonic=time.monotonic()+100))
        self.parent_patch.start();self.addCleanup(self.parent_patch.stop)
        (self.root/'jobs/origin').mkdir(parents=True)
        job=m.write(self.root/'jobs/origin/result.json',dict(elapsed_seconds=3351.921871))
        self.req=dict(kind=m.KIND,domain='formal',spec_id='S05',wall_seconds=10.,plan_id='michael_EA',output_root=str(self.root/'recoveries/out'),references=dict(job_result=job,checkpoint=job,
                    old_source_manifest=job,new_source_manifest=job,recovery_script=job))
        rr=m.write(self.root/'request.json',self.req)
        self.v=dict(request=self.req,request_reference=rr,campaign=self.root,output=self.root/'recoveries/out',state=s,ambiguity=a,q=q,
            first_stage_cost_usd=0.,origin_engine_sha256='original',recovery_engine_sha256='new',
            budget=dict(opening_cumulative_seconds=3368.778731,remaining_seconds=3831.221269,cap_seconds=7200.))
        self.claim=m.reserve_successor(self.v)

    def test_duplicate_tail_claim_and_foreign_claim_rejected(self):
        with self.assertRaises(FileExistsError):m.reserve_successor(self.v)
        wrong=copy.deepcopy(self.v);wrong['request_reference']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'another successor'):m.verify_claim(wrong,self.claim)

    def test_real_small_lp_execution_keeps_origin_and_provisional_budget(self):
        with patch.object(m,'prepare',return_value=self.v):
            result=m.execute(self.v['request_reference'],self.claim)
        self.assertEqual(result['status'],'VALID_INCUMBENT_BOUND_UNAVAILABLE')
        self.assertAlmostEqual(result['cumulative_accounted_seconds'],3368.778731+result['attempt_elapsed_seconds'])
        self.assertTrue(result['requires_final_parent_job_settlement']);self.assertTrue(result['cumulative_seconds_provisional'])
        self.assertEqual(result['master_solver_calls'],0);self.assertEqual(result['recourse_solver_calls'],0)
        self.assertFalse(result['automatic_resume_supported']);self.assertFalse(result['lockable'])
        state=json.loads((self.v['output']/'derived_state.json').read_text())
        self.assertEqual(state['origin_engine_sha256'],'original');self.assertEqual(state['recovery_engine_sha256'],'new')
        with patch.object(m,'prepare',return_value=self.v),patch.object(m,'solve_worst_distribution',side_effect=AssertionError('relaunch')):
            with self.assertRaisesRegex(ValueError,'Fresh recovery attempt'):m.execute(self.v['request_reference'],self.claim)

    def test_uncertified_original_probability_retained_as_failure(self):
        raw=self.v['state']['pending']['last_incomplete_probability']
        class Failed:
            status=raw['status'];certified_optimal=False
            def as_dict(self):return raw
        with patch.object(m,'prepare',return_value=self.v),patch.object(m,'solve_worst_distribution',return_value=Failed()):
            result=m.execute(self.v['request_reference'],self.claim)
        self.assertEqual(result['status'],'PROBABILITY_CERTIFICATE_FAILED')
        self.assertFalse(json.loads((self.v['output']/'probability_independent_check.json').read_text())['passed'])
        self.assertIsNone(json.loads((self.v['output']/'derived_state.json').read_text())['upper_bound_usd'])

    def test_hash_wall_timeout_is_not_success(self):
        token=m._HASH_DEADLINE.set(time.monotonic()-1)
        try:
            with self.assertRaises(TimeoutError):m.reference(self.v['request_reference']['path'])
        finally:m._HASH_DEADLINE.reset(token)

    def test_elapsed_budget_before_solver_is_not_reset(self):
        self.v['request']['wall_seconds']=.1
        with patch.object(m,'prepare',return_value=self.v),patch.object(m,'solve_worst_distribution',side_effect=AssertionError('must not launch')):
            result=m._execute(self.v['request_reference'],self.claim,time.monotonic()-.05)
        self.assertEqual(result['status'],'TIME_LIMIT')
        self.assertGreaterEqual(result['attempt_elapsed_seconds'],.05)
        self.assertTrue((self.v['output']/'failure.json').exists())

    def test_prepare_timeout_seals_failure_and_consumes_tail_attempt(self):
        with patch.object(m,'prepare',side_effect=TimeoutError('input hash wall expired')):
            with self.assertRaises(TimeoutError):m.execute(self.v['request_reference'],self.claim)
        directory=self.root/'recovery_attempts'/m.canonical_hash(self.req['references']['job_result'])
        result=json.loads((directory/'failure.json').read_text())
        self.assertEqual(result['status'],'TIME_LIMIT');self.assertFalse(result['automatic_retry_allowed'])
        with self.assertRaises(FileExistsError):m.execute(self.v['request_reference'],self.claim)

    def test_wrong_campaign_rejected_before_reservation(self):
        with patch.object(m,'get_paths',return_value=get_paths()):
            with self.assertRaisesRegex(ValueError,'configured authority'):m.execute(self.v['request_reference'],self.claim)
        self.assertFalse((self.root/'recovery_attempts').exists())

    def test_probability_deadline_still_seals_non_success(self):
        raw=self.v['state']['pending']['last_incomplete_probability']
        class Failed:
            status=raw['status'];certified_optimal=False
            def as_dict(self):return raw
        def expiry(*args,**kwargs):
            m._HASH_DEADLINE.set(time.monotonic()-1)
            raise TimeoutError('probability deadline')
        with patch.object(m,'prepare',return_value=self.v),patch.object(m,'solve_worst_distribution',side_effect=expiry):
            result=m.execute(self.v['request_reference'],self.claim)
        self.assertEqual(result['status'],'TIME_LIMIT')
        self.assertTrue((self.v['output']/'failure.json').exists());self.assertTrue((self.v['output']/'result.json').exists())


if __name__=='__main__':unittest.main()
