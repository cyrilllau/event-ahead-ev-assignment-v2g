"""Real small Q1 worker evidence and queue resume, never formal optimization."""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from unittest.mock import patch

from evcs_experiments.campaign_batch import file_reference,_result
from evcs_experiments.ccg import CCGProblem
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.formal_planning_worker import run_attempt,_DevelopmentFixture,worker_main,_candidate,_formal_prerequisites
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.validation.s03_toys import build_golden_cases

BUDGET=dict(cumulative_wall_seconds=30.,single_master_seconds=3.,final_certificate_reserve_seconds=2.,shutdown_reserve_seconds=.1)
OPTIONS=dict(master_backend='highs',optimizer_threads=1,oracle_time_limit_seconds=2.,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)


class FormalPlanningWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=build_golden_cases()[0];cls.first=assemble_first_stage(cls.f.case,cls.f.tasks)
        cls.templates=tuple(assemble_recourse(cls.f.case,cls.f.tasks,s) for s in cls.f.scenarios)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.request=dict(schema_version=1,domain='development',spec_id='S05',model_id='IEEE33-v54-Q1',plan_id='fixture.EA',
            target_id=self.f.name,method_id='EA',budget=BUDGET,output_root=str(self.root/'plan'))
        self.request_path=self.root/'request.json';self.write(self.request_path,self.request)

    def write(self,path,value):path.write_text(json.dumps(value));return file_reference(path)

    def prepare(self,**kwargs):
        f=self.f
        cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],self.first.matrix_sha256,
            {'fixture':f.name},self.templates.__getitem__,capacity=2)
        p=CCGProblem(self.first,cache,f.probability_set,f.scenario_cost_constants_usd,'DRO' if kwargs['method_id'] in ('EA','OR') else 'ARO')
        e=RawPlanningEvidence(kwargs['evidence_root'],f.case,f.tasks,p,f.scenarios.__getitem__,domain='development',
            input_binding={'fixture':f.name},resume=kwargs['resume_evidence'],resume_snapshot_reference=kwargs['resume_snapshot_reference'])
        return SimpleNamespace(problem=p,evidence=e,sink_factory=e.sink_factory,input_binding={'fixture':f.name})

    def invocation(self,number=1,remaining=20.,prior=0.,previous=None):
        d=self.root/f'attempt_{number}';d.mkdir();attempt_start=time.monotonic()
        value=dict(schema_version=1,domain='development',spec_id='S05',model_id='IEEE33-v54-Q1',item_id='EA',plan_id=self.request['plan_id'],
            item_request_reference=file_reference(self.request_path),remaining_seconds=remaining,cumulative_plan_seconds=prior,
            plan_remaining_seconds=30.-prior,plan_budget_seconds=30.,previous_checkpoint=None if previous is None else previous['checkpoint'],
            previous_result_reference=None if previous is None else file_reference(self.root/f'attempt_{number-1}'/'worker_result.json'),
            attempt_directory=str(d),result_path=str(d/'worker_result.json'),parent_job=None,
            created_at_utc=datetime.now(timezone.utc).isoformat(),attempt_started_monotonic=attempt_start,
            worker_deadline_monotonic=attempt_start+remaining)
        return self.write(d/'invocation.json',value)

    def fixture(self,prepare=None):return _DevelopmentFixture(prepare or self.prepare,BUDGET,OPTIONS,.05)

    def test_genuine_ea_aro_or_candidate_contains_best_own_raw_evidence_not_acceptance(self):
        for method in ('EA','ARO','OR'):
            if method!='EA':
                self.root=self.root/method.lower();self.root.mkdir();self.request_path=self.root/'request.json'
                self.request={**self.request,'method_id':method,'plan_id':'fixture.'+method,'output_root':str(self.root/'plan')};self.write(self.request_path,self.request)
            invocation=self.invocation();result=run_attempt(invocation,_development_fixture=self.fixture())
            self.assertEqual(result['status'],'COMPLETE');self.assertTrue(result['has_candidate'])
            self.assertFalse(result['full_support_verified']);self.assertFalse(result['scientific_acceptance'])
            _result(file_reference(self.root/'attempt_1'/'worker_result.json'),invocation)
            record=json.loads((self.root/'attempt_1'/'candidate_solver_record.json').read_text());candidate=record['candidate']
            self.assertEqual(record['status'],'PENDING_INDEPENDENT_NUMERICAL_AUDIT')
            self.assertAlmostEqual(candidate['upper_bound_usd'],self.f.expected['DRO' if method in ('EA','OR') else 'ARO']['objective_usd'],places=7)
            self.assertEqual(candidate['support_ids'],list(self.f.scenario_ids));self.assertEqual(len(candidate['q_usd']),3)
            self.assertTrue(candidate['own_raw_evidence']);self.assertEqual(result['raw_engine_status'],'CONVERGED')
            self.assertGreater(result['cumulative_accounted_seconds'],0)

    def test_or_dispatch_forwards_own_acceptance_to_physical_preparation(self):
        # Real small engine/evidence, with production prerequisites separately
        # covered by acceptance and queue tests; this checks the dispatch seam.
        self.request.update(method_id='OR',plan_id='fixture.OR')
        acceptance=self.write(self.root/'onset_acceptance.json',{'scope':'development fixture'})
        self.request['onset_acceptance_reference']=acceptance
        self.write(self.request_path,self.request)
        with patch('evcs_experiments.formal_planning_worker._formal_prerequisites',return_value=(BUDGET,OPTIONS)), \
             patch('evcs_experiments.formal_planning_worker.SEAL_RESERVE_SECONDS',.05), \
             patch('evcs_experiments.formal_onset.load_formal_onset_problem',side_effect=self.prepare) as factory:
            result=run_attempt(self.invocation())
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(factory.call_args.kwargs['onset_acceptance_reference'],acceptance)
        record=json.loads((self.root/'attempt_1/candidate_solver_record.json').read_text())
        self.assertEqual(record['method_id'],'OR')
        self.assertEqual(record['plan_id'],'fixture.OR')
        self.assertFalse(record['scientific_acceptance'])
        self.assertAlmostEqual(record['candidate']['upper_bound_usd'],self.f.expected['DRO']['objective_usd'],places=7)

    def test_unimplemented_saa_cannot_use_ccg_worker(self):
        self.request['method_id']='SAA';self.write(self.request_path,self.request)
        with self.assertRaisesRegex(ValueError,'Worker method/plan differs'):
            run_attempt(self.invocation(),_development_fixture=self.fixture())

    def test_short_parent_no_master_checkpoint_is_genuinely_resumable_next_attempt(self):
        first=self.invocation(remaining=4.)
        result=run_attempt(first,_development_fixture=self.fixture())
        self.assertEqual(result['status'],'TIME_LIMIT');self.assertFalse(result['has_candidate'])
        checkpoint=json.loads(Path(result['checkpoint']['path']).read_text())
        self.assertTrue(checkpoint['resume_without_engine_checkpoint']);self.assertIsNone(checkpoint['engine_checkpoint_snapshot'])
        _result(file_reference(self.root/'attempt_1'/'worker_result.json'),first)
        prior=result['cumulative_accounted_seconds']+.01
        second=self.invocation(2,remaining=20.,prior=prior,previous=result)
        completed=run_attempt(second,_development_fixture=self.fixture())
        self.assertEqual(completed['status'],'COMPLETE');self.assertTrue(completed['has_candidate'])
        self.assertGreater(completed['cumulative_accounted_seconds'],prior)
        # Earlier immutable receipt remains valid after new callbacks/checkpoints.
        _result(file_reference(self.root/'attempt_1'/'worker_result.json'),first)

    def test_partial_q_resume_keeps_previous_raw_lps_and_finishes_same_plan(self):
        count=[0]
        def bounded(template,plan,**kw):
            count[0]+=1;result=solve_fixed_plan(template,plan,**kw)
            if count[0]>1:
                return replace(result,optimal=False,solver_status=1,status='LIMIT_REACHED',
                    certificate={**result.certificate,'optimal_certificate_valid':False})
            return result
        first=self.invocation()
        with patch('evcs_experiments.ccg.solve_fixed_plan',side_effect=bounded):
            result=run_attempt(first,_development_fixture=self.fixture())
        self.assertEqual(result['status'],'TIME_LIMIT');self.assertFalse(result['has_candidate'])
        cp=json.loads(Path(result['checkpoint']['path']).read_text())
        old=json.loads(Path(cp['engine_checkpoint_snapshot']['path']).read_text())['state']
        self.assertEqual(len(old['pending']['q_usd']),1)
        saved_q=old['pending']['q_usd'][0];pending_plan=old['pending']['plan']
        second=self.invocation(2,prior=result['cumulative_accounted_seconds']+.01,previous=result)
        complete=run_attempt(second,_development_fixture=self.fixture())
        self.assertEqual(complete['status'],'TIME_LIMIT')
        self.assertTrue(complete['has_candidate'])
        third=self.invocation(3,prior=complete['cumulative_accounted_seconds']+.01,previous=complete)
        final=run_attempt(third,_development_fixture=self.fixture())
        self.assertEqual(final['status'],'COMPLETE')
        state=json.loads((self.root/'plan'/'driver'/'checkpoint.json').read_text())['state']
        resumed=next(row for row in state['trace'] if row['iteration']==old['pending']['iteration'])
        self.assertEqual(resumed['q_usd'][0],saved_q)
        self.assertEqual(json.loads(Path(cp['engine_checkpoint_snapshot']['path']).read_text())['state']['pending']['plan'],pending_plan)
        _result(file_reference(self.root/'attempt_1'/'worker_result.json'),first)

    def test_real_toy_failure_then_opening_cost_new_segment_limit_resume_complete(self):
        # Receipt-policy validation is covered by test_failure_carryover; inject
        # that verified boundary here and run the real tiny Q1 engine/exporter.
        failing=[True]
        def prepare(**kw):
            f=self.f
            def load(i):
                if failing[0]:raise ValueError('Formal feature semantic identity differs')
                return self.templates[i]
            cache=LazyRecourses(f.scenario_ids,[x.scenario_sha256 for x in f.scenarios],self.first.matrix_sha256,
                                {'fixture':f.name,'planning_source':{'adapter':'development'}},load,capacity=2)
            problem=CCGProblem(self.first,cache,f.probability_set,f.scenario_cost_constants_usd)
            e=RawPlanningEvidence(kw['evidence_root'],f.case,f.tasks,problem,f.scenarios.__getitem__,domain='development',
                input_binding={'fixture':f.name},resume=kw['resume_evidence'],resume_snapshot_reference=kw['resume_snapshot_reference'])
            return SimpleNamespace(problem=problem,evidence=e,sink_factory=e.sink_factory,input_binding={'fixture':f.name})
        with self.assertRaisesRegex(ValueError,'feature semantic'):
            run_attempt(self.invocation(),_development_fixture=self.fixture(prepare))
        oldroot=self.root/'plan';oldcheckpoint=json.loads((oldroot/'driver/checkpoint.json').read_text())
        self.assertEqual(oldcheckpoint['state']['trace'],[]);self.assertIsNotNone(oldcheckpoint['state']['phase1'])
        oldbytes=(oldroot/'driver/checkpoint.json').read_bytes();oldidentity=oldcheckpoint['state']['identity']
        failure_ref=file_reference(self.root/'attempt_1/worker_failure.json')
        self.root=self.root/'retry';self.root.mkdir();self.request_path=self.root/'request.json'
        proof=self.write(self.root/'carryover.json',{'old_failure':failure_ref,'accounted_seconds':16.85686})
        queue_ref=self.write(self.root/'queue.json',{'queue_root':str(self.root/'queue')})
        self.request={**self.request,'output_root':str(self.root/'plan'),'failure_carryover_reference':proof};self.write(self.request_path,self.request)
        verified=dict(reference=proof,accounted_seconds=16.85686,old_problem_identity=oldidentity)
        def invocation(n,remaining,prior,previous=None):
            ref=self.invocation(n,remaining=remaining,prior=prior,previous=previous)
            value=json.loads(Path(ref['path']).read_text());value.update(failure_carryover_reference=proof,opening_carryover_seconds=16.85686,request_reference=queue_ref)
            return self.write(Path(ref['path']),value)
        def verified_receipt(ref,request,root):
            self.assertEqual(ref,proof);self.assertEqual(file_reference(failure_ref['path']),failure_ref)
            return verified
        failing[0]=False
        with patch('evcs_experiments.formal_planning_worker.verify_failure_carryover',side_effect=verified_receipt):
            first=run_attempt(invocation(1,4.,16.85686),_development_fixture=self.fixture(prepare))
            self.assertEqual(first['status'],'TIME_LIMIT');self.assertFalse(first['has_candidate'])
            self.assertFalse((self.root/'plan/driver/checkpoint.json').exists())
            prior=first['cumulative_accounted_seconds']+.01
            done=run_attempt(invocation(2,10.,prior,first),_development_fixture=self.fixture(prepare))
        self.assertEqual(done['status'],'COMPLETE');self.assertTrue(done['has_candidate'])
        self.assertEqual(done['opening_carryover_seconds'],16.85686)
        self.assertGreater(done['cumulative_accounted_seconds'],prior)
        self.assertLess(done['cumulative_accounted_seconds'],30.)
        self.assertEqual((oldroot/'driver/checkpoint.json').read_bytes(),oldbytes)
        self.assertNotEqual(self.root/'plan/driver/checkpoint.json',oldroot/'driver/checkpoint.json')

    def test_unreviewed_positive_prior_without_checkpoint_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unbound existing work'):
            run_attempt(self.invocation(prior=16.85686),_development_fixture=self.fixture())

    def test_queue_preverification_is_charged_without_double_subtracting_deadline(self):
        ref=self.invocation(remaining=4.);value=json.loads(Path(ref['path']).read_text())
        value['attempt_started_monotonic']-=2.
        ref=self.write(Path(ref['path']),value)
        result=run_attempt(ref,_development_fixture=self.fixture())
        self.assertEqual(result['status'],'TIME_LIMIT')
        self.assertGreaterEqual(result['attempt_accounted_seconds'],2.)
        driver=json.loads((self.root/'plan'/'driver'/'driver_state.json').read_text())
        self.assertGreaterEqual(driver['cumulative_seconds'],2.)
        self.assertLess(result['attempt_accounted_seconds'],3.)

    def test_clock_reversal_cannot_reduce_elapsed_accounting(self):
        ref=self.invocation();value=json.loads(Path(ref['path']).read_text())
        value['created_at_utc']=(datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat()
        ref=self.write(Path(ref['path']),value)
        with self.assertRaisesRegex(ValueError,'Clock reversal'):
            run_attempt(ref,_development_fixture=self.fixture())

    def test_cumulative_budget_exhaustion_without_candidate_is_failure(self):
        first=self.invocation(remaining=4.);result=run_attempt(first,_development_fixture=self.fixture())
        second=self.invocation(2,remaining=3.,prior=27.,previous=result)
        with self.assertRaisesRegex(RuntimeError,'without a usable candidate'):
            run_attempt(second,_development_fixture=self.fixture())
        self.assertFalse((self.root/'attempt_2'/'worker_result.json').exists())
        self.assertTrue((self.root/'attempt_2'/'worker_failure.json').exists())

    def test_prior_time_and_snapshot_tampering_rejected_before_prepare(self):
        first=self.invocation(remaining=4.);result=run_attempt(first,_development_fixture=self.fixture())
        second=self.invocation(2,remaining=20.,prior=0.,previous=result)
        with self.assertRaisesRegex(ValueError,'cannot reset'):
            run_attempt(second,_development_fixture=self.fixture(lambda **k:(_ for _ in ()).throw(AssertionError('prepared'))))
        third=self.invocation(3,remaining=20.,prior=1.,previous=None)
        value=json.loads(Path(third['path']).read_text());value.update(previous_checkpoint=result['checkpoint'],
            previous_result_reference=file_reference(self.root/'attempt_1'/'worker_result.json'))
        third=self.write(Path(third['path']),value)
        cp=json.loads(Path(result['checkpoint']['path']).read_text());raw=Path(cp['driver_snapshot']['path']);raw.write_bytes(raw.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):
            run_attempt(third,_development_fixture=self.fixture())

    def test_preparation_and_final_export_are_inside_both_time_budgets(self):
        invocation=self.invocation(remaining=.15)
        value=json.loads(Path(invocation['path']).read_text())
        ticks=[value['attempt_started_monotonic']]
        clock=SimpleNamespace(monotonic=lambda:ticks[0])
        def slow(**kw):
            prepared=self.prepare(**kw)  # Real tiny Q1 input and raw first-stage export.
            ticks[0]+=.08
            return prepared
        def final_export(*args,**kwargs):
            ticks[0]+=.09
            return {'candidate':None}
        from evcs_experiments.planning_budget import run_budgeted_ccg
        with patch('evcs_experiments.formal_planning_worker.time',clock), \
             patch('evcs_experiments.planning_budget.time',clock), \
             patch('evcs_experiments.formal_planning_worker.run_budgeted_ccg',wraps=run_budgeted_ccg) as driver_call, \
             patch('evcs_experiments.planning_budget.run_ccg',side_effect=AssertionError('Tiny slice must not start a master')), \
             patch('evcs_experiments.formal_planning_worker._candidate',side_effect=final_export) as candidate:
            with self.assertRaisesRegex(ValueError,'Preparation/export exceeded'):
                run_attempt(invocation,_development_fixture=self.fixture(slow))
        candidate.assert_called_once()
        driver_call.assert_called_once()
        self.assertAlmostEqual(driver_call.call_args.kwargs['spent_before_seconds'],.08,places=7)
        self.assertAlmostEqual(driver_call.call_args.kwargs['available_wall_seconds'],.02,places=7)
        self.assertFalse((self.root/'attempt_1'/'worker_result.json').exists())
        failure=json.loads((self.root/'attempt_1'/'worker_failure.json').read_text())
        self.assertAlmostEqual(failure['cumulative_accounted_seconds'],.17,places=7)
        self.assertIn('Preparation/export exceeded',failure['error'])
        driver=json.loads((self.root/'plan/driver/driver_state.json').read_text())
        self.assertEqual(driver['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(driver['calls'],[])

    def test_candidate_changed_q_or_probability_cannot_borrow_other_raw_proof(self):
        invocation=self.invocation();result=run_attempt(invocation,_development_fixture=self.fixture())
        checkpoint=json.loads(Path(result['checkpoint']['path']).read_text())
        state=json.loads(Path(checkpoint['engine_checkpoint_snapshot']['path']).read_text())['state']
        driver=json.loads(Path(checkpoint['driver_snapshot']['path']).read_text())
        prepared=self.prepare(method_id='EA',evidence_root=self.root/'plan'/'raw',resume_evidence=True,
            resume_snapshot_reference=checkpoint['raw_evidence_snapshot'])
        for kind in ('q','probability','rmp'):
            changed=deepcopy(state);best=changed['best'];row=next(r for r in changed['trace'] if r['iteration']==best['iteration'])
            if kind=='q':best['q_usd'][0]+=.01;row['q_usd'][0]+=.01
            elif kind=='probability':
                best['probability']['wall_seconds']+=.01;row['probability_audit']['wall_seconds']+=.01
            else:row['rmp']['wall_seconds']+=.01
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                _candidate(changed,prepared,self.request,driver,{},1.)

    def test_formal_wrong_queue_item_and_parent_environment_rejected(self):
        request=dict(domain='formal',spec_id='S05',model_id='IEEE33-v54-Q1')
        invocation=dict(domain='formal',spec_id='S05',model_id='IEEE33-v54-Q1',batch_id='b',item_id='EA',plan_id='p',
            request_reference={'path':'/bound/request.json'},item_request_reference={'path':'/bound/item.json'},
            plan_budget_seconds=7200,argv=['/python'],cwd='/tmp',parent_job={'job_id':'job','directory':'/bound/job'})
        item=dict(item_id='EA',plan_id='p',request_reference={'path':'/wrong/item.json'},plan_budget_seconds=7200,argv=['/python'],cwd='/tmp')
        batch=dict(domain='formal',spec_id='S05',model_id='IEEE33-v54-Q1',batch_id='b',items=[item])
        with patch('evcs_experiments.formal_planning_worker.checked_request',return_value=(batch,invocation['request_reference'])):
            with self.assertRaisesRegex(ValueError,'item differs'):_formal_prerequisites(request,invocation)
            item['request_reference']=invocation['item_request_reference']
            with patch.dict('os.environ',{'EVCS_JOB_ID':'wrong','EVCS_JOB_DIRECTORY':'/wrong'}):
                with self.assertRaisesRegex(ValueError,'parent environment'):_formal_prerequisites(request,invocation)

    def test_formal_cli_and_dependency_injection_cannot_bypass_production_gates(self):
        invocation=self.invocation();value=json.loads(Path(invocation['path']).read_text());value['domain']='formal'
        invocation=self.write(Path(invocation['path']),value)
        with self.assertRaisesRegex(ValueError,'cannot bypass formal'):
            run_attempt(invocation,_development_fixture=self.fixture())
        value['domain']='development';invocation=self.write(Path(invocation['path']),value)
        with patch.dict('os.environ',{'EVCS_BATCH_ATTEMPT_REQUEST':invocation['path'],'EVCS_BATCH_INVOCATION_SHA256':invocation['sha256']}):
            with self.assertRaisesRegex(ValueError,'only accepts formal'):worker_main()


if __name__=='__main__':unittest.main()
