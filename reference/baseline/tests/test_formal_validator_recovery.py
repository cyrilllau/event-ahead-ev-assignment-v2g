"""Real ordinary EA/OR worker continuation through the explicit recovery entry.

Only physical fixtures, an initial cooperative stop, and local supervisor
observations are injected. Master, recourse, audit, seals, queue and child
processes are real; no formal numerical acceptance is claimed.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

from evcs_experiments import campaign_batch as batch,checkpoint_closure
from evcs_experiments import formal_planning_worker as fw
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases
from tests import test_campaign_batch as queue_helpers

HARNESS=r'''
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from evcs_experiments import formal_planning_worker as fw
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.ccg import CCGProblem
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.validation.s03_toys import build_golden_cases
invref=file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
inv=json.loads(Path(invref['path']).read_text())
req=json.loads(Path(inv['item_request_reference']['path']).read_text())
f=next(f for f in build_golden_cases() if f.name==req['target_id'])
templates=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
first=templates[0].first_stage
binding={'fixture':f.name,'historical_release':req['development_inventory_probe_reference']}
def prepare(**kw):
 cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],first.matrix_sha256,binding,templates.__getitem__,capacity=2)
 problem=CCGProblem(first,cache,f.probability_set,f.scenario_cost_constants_usd,'DRO')
 evidence=RawPlanningEvidence(kw['evidence_root'],f.case,f.tasks,problem,f.scenarios.__getitem__,domain='development',input_binding=binding,resume=kw['resume_evidence'],resume_snapshot_reference=kw['resume_snapshot_reference'])
 return SimpleNamespace(problem=problem,evidence=evidence,sink_factory=evidence.sink_factory,input_binding=binding)
fixture=fw._DevelopmentFixture(prepare,req['budget'],dict(master_backend='highs',optimizer_threads=1,oracle_time_limit_seconds=2.,mip_relative_gap=1e-9,optimality_relative_gap=1e-8),.05)
def old_closure(ref,seen=None):
 verify_reference(ref)
 if not ref['path'].endswith('.json'):return
 def walk(v):
  if isinstance(v,dict):
   if set(v)=={'path','sha256','size_bytes'}:old_closure(v)
   else:
    for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 walk(json.loads(Path(ref['path']).read_text()))
if len(sys.argv)>1:
 # Reproduce the inherited validator defect; the explicit adapter must replace it.
 fw._closure=old_closure
 try:old_closure(inv['previous_checkpoint'])
 except ValueError:pass
 else:raise AssertionError('Original generic validator unexpectedly accepted relative source inventory')
 from evcs_experiments.formal_validator_recovery import main
 main(_development_fixture=fixture)
else:
 if req['method_id']=='OR' and inv['previous_checkpoint'] is None:
  original_driver=fw.run_budgeted_ccg
  def short_driver(*args,**kwargs):
   return original_driver(*args,**dict(kwargs,available_wall_seconds=.2))
  fw.run_budgeted_ccg=short_driver
 result=fw.run_attempt(invref,_development_fixture=fixture)
 print(json.dumps({'status':result['status']}))
'''


class FormalValidatorRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.queue=self.root/'queue'
        self.parent=queue_helpers.CampaignBatchTests(methodName='runTest');self.parent.setUp()
        self.addCleanup(self.parent.tearDown)
        self.project=get_paths().project_root
        self.worker=self.root/'ordinary_worker.py';self.worker.write_text(HARNESS)
        self.budget=dict(cumulative_wall_seconds=30.,single_master_seconds=3.,
                         final_certificate_reserve_seconds=3.,shutdown_reserve_seconds=.1)
        historical=self.root/'old-release';historical.mkdir();(historical/'data.txt').write_text('original bytes')
        probe=self.write(historical/'RELEASE_MANIFEST.json',dict(schema_version=1,release_name=historical.name,
            created_at_utc='2026-09-09T00:00:00+00:00',git_head='0'*40,working_tree_snapshot=True,
            files=[dict(batch.file_reference(historical/'data.txt'),path='data.txt')]))
        self.sources=[batch.file_reference(p) for p in sorted((self.project/'src/evcs_experiments').rglob('*.py'))]
        numerical=dict(schema_version=1,postprocessing_reserve_seconds=.8,sources=dict(
            helper=batch.file_reference(self.project/'scripts/audits/audit_s05_pending_recovery.py'),
            numerical_kernel=batch.file_reference(self.project/'scripts/audits/audit_s05_candidate_recourse.py')))
        self.items=[];self.requests=[]
        for golden,method in zip(build_golden_cases()[:2],('EA','OR')):
            request=dict(schema_version=1,domain='development',spec_id='S05',model_id='IEEE33-v54-Q1',
                plan_id='recovery.'+method,target_id=golden.name,method_id=method,budget=self.budget,
                output_root=str(self.root/('output_'+method)),planning_source_references=self.sources,
                numerical_audit=numerical,development_inventory_probe_reference=probe)
            rr=self.write(self.root/(method+'.json'),request);self.requests.append(rr)
            self.items.append(dict(item_id=method,plan_id=request['plan_id'],plan_budget_seconds=30.,
                argv=[sys.executable,str(self.worker)],cwd=str(self.root),request_reference=rr))
        self.q=dict(schema_version=1,batch_id='ordinary-recovery',spec_id='S05',model_id='IEEE33-v54-Q1',domain='development',
            queue_root=str(self.queue),job_wall_seconds=60.,shutdown_reserve_seconds=.1,
            parent_wall_policy='verified_parent_remaining_v1',source_references=[batch.file_reference(batch.__file__),batch.file_reference(self.worker)],
            budget_reference=self.write(self.root/'budget.json',self.budget),items=self.items)
        self.qref=self.write(self.root/'queue.json',self.q)
        with self.parent.parent(59.):self.first=batch.run_batch(self.qref['path'],self.queue)
        self.assertEqual(self.first['status'],'TIME_LIMIT',self.diagnostics())
        self.assertEqual(self.first['completed_item_ids'],['EA'])
        self.assertEqual(self.first['pending_item_ids'],['OR'])
        self.old_or=batch.file_reference(self.queue/'OR__attempt_0001/worker_result.json')
        self.old_checkpoint=self.read(self.old_or)['checkpoint']
        self.assertTrue(self.read(self.old_checkpoint)['resume_without_engine_checkpoint'])
        self.assertIsNone(self.read(self.old_checkpoint)['engine_checkpoint_snapshot'])
        self.assertGreater(self.first['cumulative_plan_seconds']['recovery.OR'],0.)
        self.original_ea={str(p):batch.file_reference(p) for p in (self.queue/'EA__attempt_0001').rglob('*') if p.is_file()}
        self.original_or={str(p):batch.file_reference(p) for p in (self.queue/'OR__attempt_0001').rglob('*') if p.is_file()}
        self.module_manifest=self.project/('.test-ordinary-recovery-'+uuid.uuid4().hex+'.json')
        self.addCleanup(lambda:self.module_manifest.unlink(missing_ok=True))
        self.write(self.module_manifest,dict(files=[dict(path=Path(m.__file__).resolve().relative_to(self.project).as_posix(),
            sha256=batch.file_reference(m.__file__)['sha256']) for m in (batch,fw)]))

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
        return batch.file_reference(path)

    def read(self,ref):return json.loads(Path(ref['path']).read_text())

    def diagnostics(self):
        return {str(p):p.read_text()[-5000:] for p in self.queue.rglob('*.log')}

    def admit(self,**updates):
        from evcs_experiments import formal_validator_recovery as adapter
        self.parent_context=self.parent.parent(23.,elapsed=.35)
        self.parent_directory=self.parent_context.__enter__();self.addCleanup(lambda:self.parent_context.__exit__(None,None,None))
        parent=batch._parent(self.q);state=json.loads((self.queue/'state.json').read_text())
        self.a=dict(kind=adapter.KIND,domain='development',adapter_reference=batch.file_reference(adapter.__file__),
            validator_reference=batch.file_reference(checkpoint_closure.__file__),queue_reference=self.qref,
            worker_request_reference=self.requests[1],original_release_reference=batch.file_reference(self.module_manifest),
            baseline_ledger_reference=self.write(self.root/'baseline.json',state),queue_root=str(self.queue),plan_id='recovery.OR',
            original_module_references={name:batch.file_reference(m.__file__) for name,m in [('campaign_batch',batch),('formal_planning_worker',fw)]},
            original_python=sys.executable,parent_job_id=parent['job_id'],parent_job_directory=parent['directory'],
            baseline_receipt_references=state['attempt_receipts'],baseline_result_references=[self.read(r)['result_reference'] for r in state['attempt_receipts']],
            opening_cumulative_seconds=self.first['cumulative_plan_seconds']['recovery.OR'],
            parent_execution_path=str(self.root/'parent_execution.json'),parent_coverage_path=str(self.root/'parent_coverage.json'),
            development_entry_reference=batch.file_reference(self.worker),completed_item_ids=['EA'],pending_item_id='OR',**updates)
        return self.reseal_admission()

    def reseal_admission(self):
        from evcs_experiments import formal_validator_recovery as adapter
        review=dict(kind='FORMAL_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1',passed=True,admission_payload_sha256=canonical_hash(self.a),
            adapter_reference=self.a['adapter_reference'],validator_reference=self.a['validator_reference'])
        self.a['independent_review_reference']=self.write(self.root/'recovery_review.json',review)
        self.ref=self.write(self.root/'admission.json',self.a)
        manifest=json.loads((self.parent_directory/'manifest.json').read_text())
        manifest['argv']=[sys.executable,adapter.__file__,'parent','--admission',self.ref['path'],'--expected-sha256',self.ref['sha256']]
        (self.parent_directory/'manifest.json').write_text(json.dumps(manifest))
        return self.ref

    def run_parent(self):
        from evcs_experiments import formal_validator_recovery as adapter
        argv=[sys.executable,adapter.__file__,'parent','--admission',self.ref['path'],'--expected-sha256',self.ref['sha256']]
        clock=batch._parent(self.q)['wall_clock']
        with patch.object(adapter,'actual_argv',return_value=argv),patch.object(batch,'_supervisor_remaining',return_value=clock), \
             patch.object(batch,'process_identity',side_effect=queue_helpers.real_identity):return adapter.run_parent(self.ref)

    def test_real_parent_child_skips_completed_ea_and_continues_original_or_checkpoint(self):
        from evcs_experiments import formal_validator_recovery as adapter
        self.admit();out=self.run_parent();result=out['queue_result']
        self.assertEqual(result['status'],'COMPLETE_BATCH',(result,self.diagnostics()))
        self.assertEqual(result['new_attempt_item_ids'],['OR']);self.assertIn('EA',result['reused_item_ids'])
        self.assertEqual(result['cumulative_plan_seconds']['recovery.EA'],self.first['cumulative_plan_seconds']['recovery.EA'])
        self.assertGreater(result['cumulative_plan_seconds']['recovery.OR'],self.first['cumulative_plan_seconds']['recovery.OR'])
        for path,ref in {**self.original_ea,**self.original_or}.items():self.assertEqual(batch.file_reference(path),ref)
        attempt=self.queue/'OR__attempt_0002';invref=batch.file_reference(attempt/'invocation.json');inv=self.read(invref)
        self.assertEqual(inv['previous_checkpoint'],self.old_checkpoint)
        self.assertEqual(inv['cumulative_plan_seconds'],self.first['cumulative_plan_seconds']['recovery.OR'])
        final=adapter.validate_execution_result(batch.file_reference(attempt/'worker_result.json'),invref)
        self.assertIn(final['validation_recovery_execution_reference'],final['artifacts'])
        seal=self.read(batch.file_reference(attempt/'finalization.json'))
        numeric=self.read(seal['references']['solver_record'])
        self.assertTrue(numeric['full_support_verified']);self.assertTrue(numeric['final_budget_acceptance'])
        self.assertAlmostEqual(numeric['upper_bound_usd'],39.,places=7)
        self.assertFalse(numeric['formal_OOS'])
        cp=self.read(final['checkpoint']);self.assertEqual(cp['previous_checkpoint'],self.old_checkpoint)
        self.assertEqual(cp['request_reference'],self.requests[1])
        parent=self.read(batch.file_reference(self.a['parent_execution_path']))
        clock=parent['queue_clock_execution']
        self.assertEqual(inv['attempt_started_monotonic'],clock['accounting_start_monotonic'])
        self.assertEqual(clock['accounting_start_monotonic'],
                         clock['observed_parent_clock']['observed_monotonic']-clock['observed_parent_clock']['elapsed_seconds'])
        self.assertAlmostEqual(clock['observed_parent_clock']['elapsed_seconds'],.35,places=7)
        charged=result['cumulative_plan_seconds']['recovery.OR']-self.a['opening_cumulative_seconds']
        self.assertGreaterEqual(charged,.35)
        closing=self.write(self.root/'closing.json',json.loads((self.queue/'state.json').read_text()))
        # Jobctl terminal observations are explicit local fixtures; numerical work above is real.
        for name,update in (('undercoverage',{'elapsed_seconds':charged+.5}),
                            ('incomplete',{'complete':False}),('nonzero',{'returncode':7}),
                            ('live',{'supervisor':queue_helpers.real_identity(os.getpid())}),('covered',{})):
            terminal=dict(job_id=self.a['parent_job_id'],state='SUCCEEDED',returncode=0,complete=True,
                          elapsed_seconds=charged-.001)
            terminal.update(update)
            tr=self.write(self.parent_directory/'result.json',terminal)
            self.write(self.parent_directory/'status.json',terminal)
            coverage=self.read(adapter.validate_terminal_coverage(self.ref,tr,closing,
                output_path=self.root/('terminal_'+name+'.json')))
            self.assertEqual(coverage['passed'],name=='covered')
            self.assertFalse(coverage['automatic_resume_authorized'])
            self.assertEqual(coverage['original_cumulative_opening_seconds'],self.first['cumulative_plan_seconds']['recovery.OR'])
            if name=='undercoverage':self.assertAlmostEqual(coverage['uncovered_parent_seconds'],.5,places=7)
            if name=='live':self.assertEqual(coverage['status'],'HOLD_TERMINAL_JOB_NOT_QUIET')
        wrong=self.write(self.root/'wrong_terminal.json',terminal)
        with self.assertRaisesRegex(ValueError,'path'):
            adapter.validate_terminal_coverage(self.ref,wrong,closing,output_path=self.root/'wrong_terminal_coverage.json')

    def test_resealed_budget_reset_and_baseline_exemption_expansion_are_refused(self):
        from evcs_experiments import formal_validator_recovery as adapter
        self.admit()
        original=deepcopy(self.a)
        for mode in ('fees','completed','results','source','active','halted'):
            self.a=deepcopy(original);self.a.pop('independent_review_reference')
            if mode=='fees':self.a['opening_cumulative_seconds']=0.
            if mode=='completed':self.a['completed_item_ids']=['EA','OR']
            if mode=='results':self.a['baseline_result_references'].append(deepcopy(self.old_or))
            if mode=='source':self.a['validator_reference']['sha256']='0'*64
            if mode in ('active','halted'):
                state=self.read(self.a['baseline_ledger_reference']);state[mode]={'pid':123} if mode=='active' else 'requires review'
                self.a['baseline_ledger_reference']=self.write(self.root/('baseline_'+mode+'.json'),state)
            with self.subTest(mode=mode),self.assertRaises(ValueError):adapter.admission(self.reseal_admission())

    def test_missing_execution_receipt_is_rejected_after_real_resume(self):
        from evcs_experiments import formal_validator_recovery as adapter
        self.admit();out=self.run_parent();self.assertEqual(out['queue_result']['status'],'COMPLETE_BATCH',self.diagnostics())
        attempt=self.queue/'OR__attempt_0002';path=attempt/'worker_result.json';value=json.loads(path.read_text())
        receipt_ref=value['validation_recovery_execution_reference'];receipt=self.read(receipt_ref)
        receipt['actual_argv']=[sys.executable,'unreviewed_child.py']
        forged=self.write(attempt/'forged_receipt.json',receipt)
        forged_result=deepcopy(value);forged_result['validation_recovery_execution_reference']=forged
        forged_result['artifacts']=[forged if r==receipt_ref else r for r in forged_result['artifacts']]
        forged_ref=self.write(attempt/'forged_result.json',forged_result)
        with self.assertRaisesRegex(ValueError,'argv'):
            adapter.validate_execution_result(forged_ref,batch.file_reference(attempt/'invocation.json'))
        value.pop('validation_recovery_execution_reference');ref=self.write(path,value)
        with self.assertRaisesRegex(ValueError,'omitted|receipt'):
            adapter.validate_execution_result(ref,batch.file_reference(attempt/'invocation.json'))

    def test_prequeue_failure_cannot_retry_same_opening_with_new_job_and_admission(self):
        from evcs_experiments import formal_validator_recovery as adapter
        self.admit()
        original_state=json.loads((self.queue/'state.json').read_text())
        altered=deepcopy(original_state);altered['halted']='explicit test prequeue hold'
        self.write(self.queue/'state.json',altered)
        with self.assertRaisesRegex(ValueError,'history changed|unsettled'):
            self.run_parent()
        old_admission=deepcopy(self.a);old_ref=deepcopy(self.ref)
        manifest=json.loads((self.parent_directory/'manifest.json').read_text())
        claim_path=adapter.opening_claim_path(self.a,manifest)
        original_claim=batch.file_reference(claim_path)
        self.write(self.queue/'state.json',original_state)
        self.parent_context.__exit__(None,None,None)
        context=self.parent.parent(23.,elapsed=.4);directory=context.__enter__()
        self.addCleanup(lambda:context.__exit__(None,None,None))
        parent=batch._parent(self.q)
        value=deepcopy(old_admission);value.pop('independent_review_reference')
        value.update(parent_job_id=parent['job_id'],parent_job_directory=parent['directory'],
                     parent_execution_path=str(self.root/'second_parent.json'),parent_coverage_path=str(self.root/'second_coverage.json'))
        review=dict(kind='FORMAL_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1',passed=True,
                    admission_payload_sha256=canonical_hash(value),adapter_reference=value['adapter_reference'],validator_reference=value['validator_reference'])
        value['independent_review_reference']=self.write(self.root/'second_review.json',review)
        self.ref=self.write(self.root/'second_admission.json',value);self.a=value;self.parent_directory=directory
        manifest=json.loads((directory/'manifest.json').read_text())
        manifest['argv']=adapter.parent_argv(value,self.ref);self.write(directory/'manifest.json',manifest)
        with self.assertRaises(FileExistsError):self.run_parent()
        self.assertEqual(batch.file_reference(claim_path),original_claim)
        self.assertEqual(batch.file_reference(old_ref['path']),old_ref)
        self.assertFalse((self.queue/'OR__attempt_0002').exists())

    def test_wrong_actual_parent_argv_rejected_before_child_dispatch(self):
        from evcs_experiments import formal_validator_recovery as adapter
        self.admit()
        with patch.object(adapter,'actual_argv',return_value=[sys.executable,'unreviewed.py']), \
             patch.object(batch.subprocess,'Popen',side_effect=AssertionError('No child may launch')):
            with self.assertRaisesRegex(ValueError,'argv|entry'):adapter.run_parent(self.ref)
        self.assertFalse((self.queue/'OR__attempt_0002').exists())


if __name__=='__main__':unittest.main()
