"""First-recovery expense tests: immutable old debt, exclusive claim, failed-parent costs."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import campaign_batch as batch, saa_saved_model_expense as expense, saa_saved_model_recovery as worker
from evcs_experiments.random_streams import canonical_hash

class SavedModelExpenseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.op=self.root/'operations/new';self.op.mkdir(parents=True);self.old=self.root/'oldqueue';self.old.mkdir()
        self.plan='development.SAA';self.budget={'cumulative_wall_seconds':60}
        self.old_worker=self.put('old-worker.json',dict(domain='development',method_id='SAA',plan_id=self.plan,budget=self.budget))
        self.old_queue=self.put('old-queue.json',dict(items=[dict(request_reference=self.old_worker,plan_id=self.plan,plan_budget_seconds=60),dict(plan_id='development.ARO')],queue_root=str(self.old)))
        self.old_state=dict(request_reference=self.old_queue,attempt_receipts=[],halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
        (self.old/'state.json').write_text(json.dumps(self.old_state));self.snapshot=self.put('old-snapshot.json',self.old_state)
        self.old_terminal=self.put('old-terminal.json',dict(complete=True,state='FAILED',returncode=2,elapsed_seconds=2.5))
        self.fee_data=dict(kind='SAA_ASSEMBLED_MASTER_FAILED_PARENT_EXPENSE_V1',passed=True,accounting_PASS=True,status='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION',references={'worker_request':self.old_worker},original_queue_reference=self.old_queue,closing_snapshot_reference=self.snapshot,terminal_job_result_reference=self.old_terminal,plan_id=self.plan,SAA_charged_seconds=2.5,SAA_remaining_seconds=57.5,ARO_charged_seconds=0,ARO_no_invocation=True,ARO_no_worker_outputs=True,parent_charged_once=True,nested_seconds_added=False,cumulative_accounted_seconds_decimal='2.5',opening_cumulative_seconds_decimal='0')
        self.fee=self.put('fee.json',self.fee_data);self.decision_data=dict(kind='SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1',old_queue_permanently_quarantined=True,original_queue_rearm_prohibited=True,terminal_fee_settlement_reference=self.fee,original_queue_reference=self.old_queue,original_queue_snapshot_reference=self.snapshot)
        self.decision=self.put('decision.json',self.decision_data);self.policy=self.put('policy.json',dict(fee_reference=self.fee,isolation_decision_reference=self.decision,original_worker_reference=self.old_worker))
        self.newworker=self.put('worker.json',dict(domain='development',method_id='SAA',plan_id=self.plan,budget=self.budget,failure_carryover_reference=self.fee))
        budgetref=self.put('budget.json',self.budget);self.queue=self.put('queue.json',dict(schema_version=1,batch_id='new',spec_id='S05',model_id='IEEE33-v54-Q1',domain='development',queue_root=str(self.root/'newqueue'),job_wall_seconds=50,shutdown_reserve_seconds=1,parent_wall_policy='verified_parent_remaining_v1',source_references=[batch.file_reference(batch.__file__)],budget_reference=budgetref,items=[dict(item_id='saa',plan_id=self.plan,request_reference=self.newworker,plan_budget_seconds=60,failure_carryover_reference=self.fee,argv=[sys.executable,expense.__file__,'child'],cwd=str(self.root))]))
        self.admission_data=dict(kind=expense.KIND,domain='development',policy_reference=self.policy,expense_module_reference=batch.file_reference(expense.__file__),worker_module_reference=batch.file_reference(worker.__file__),worker_request_reference=self.newworker,queue_reference=self.queue,original_queue_source_reference=batch.file_reference(batch.__file__),original_python=sys.executable,campaign_root=str(self.root),parent_execution_path=str(self.op/'parent.json'),parent_job_id='recovery1',parent_timeout_seconds=50,parent_grace_seconds=1)
        self.admission=self.admit('admission',self.admission_data);self.parent=self.make_parent(self.admission)
    def put(self,name,value):
        p=self.op/name;p.write_text(json.dumps(value,sort_keys=True));return batch.file_reference(p)
    def admit(self,name,a):
        a=copy.deepcopy(a);a['independent_review_reference']=self.put(name+'-review.json',dict(kind=expense.REVIEW,passed=True,admission_payload_sha256=canonical_hash(a),expense_module_reference=a['expense_module_reference'],worker_module_reference=a['worker_module_reference']))
        return self.put(name+'.json',a)
    def make_parent(self,ref):
        a=expense.read(ref);job=self.root/'jobs'/a['parent_job_id'];job.mkdir(parents=True,exist_ok=True)
        m=dict(job_id=a['parent_job_id'],argv=expense.parent_argv(a,ref),outroot=str(self.root/'jobs'),threads=1,timeout_seconds=a['parent_timeout_seconds'],grace_seconds=a['parent_grace_seconds'])
        (job/'manifest.json').write_text(json.dumps(m));self.env=dict(EVCS_JOB_ID=a['parent_job_id'],EVCS_JOB_DIRECTORY=str(job));return dict(job_id=a['parent_job_id'],directory=str(job),manifest_reference=batch.file_reference(job/'manifest.json'),wall_clock=dict(clock_kind='linux_supervisor_birth_boottime_v1',observed_monotonic=10,elapsed_seconds=1))
    def reserve(self,ref=None,parent=None):
        ref=ref or self.admission;parent=parent or self.parent
        with patch.dict(os.environ,self.env),patch.object(expense,'actual_argv',return_value=expense.parent_argv(expense.read(ref),ref)):
            return expense.reserve_parent(ref,parent)
    def test_opening_preserves_old_parent_not_nested_clocks(self):
        o=expense.verify_opening(self.fee,self.decision,self.old_worker);self.assertEqual((o['accounted_seconds'],o['remaining_seconds']),(2.5,57.5));self.assertEqual((self.old/'state.json').read_text(),json.dumps(self.old_state))
        bad=dict(self.fee_data,SAA_charged_seconds=0,SAA_remaining_seconds=60);r=self.put('bad-fee.json',bad);d=self.put('bad-decision.json',dict(self.decision_data,terminal_fee_settlement_reference=r))
        with self.assertRaises(ValueError):expense.verify_opening(r,d,self.old_worker)
    def test_different_job_cannot_reopen_same_failed_parent(self):
        first=self.reserve();before=Path(first['path']).read_bytes();other=dict(self.admission_data,parent_job_id='recovery2',parent_execution_path=str(self.op/'parent2.json'));ref=self.admit('admission2',other);parent=self.make_parent(ref)
        with self.assertRaises(FileExistsError):self.reserve(ref,parent)
        self.assertEqual(Path(first['path']).read_bytes(),before)
    def test_prequeue_failure_charged_and_no_automatic_resume(self):
        reservation=self.reserve();job=Path(self.parent['directory']);terminal=dict(job_id=self.parent['job_id'],complete=True,state='FAILED',returncode=2,elapsed_seconds=3.)
        for name in ('status.json','result.json'):(job/name).write_text(json.dumps(terminal))
        with patch('evcs_experiments.runtime.jobctl.observation',return_value={'blocks_new_launch':False}):
            ref=expense.settle_parent(reservation,batch.file_reference(job/'result.json'));result=expense.read(ref)
            self.assertEqual(expense.settle_parent(reservation,batch.file_reference(job/'result.json')),ref)
        self.assertEqual(result['cumulative_spent_seconds'],5.5);self.assertEqual(result['remaining_seconds'],54.5);self.assertEqual(result['new_queue_charge_seconds'],0);self.assertFalse(result['passed']);self.assertFalse(result['automatic_resume_authorized']);self.assertFalse((self.root/'newqueue').exists())
    def test_oversized_parent_and_wrong_actual_argv_rejected(self):
        a=dict(self.admission_data,parent_timeout_seconds=57.5,parent_grace_seconds=1);ref=self.admit('too-long',a)
        with self.assertRaises(ValueError):expense._admission(ref)
        with patch.dict(os.environ,self.env),patch.object(expense,'actual_argv',return_value=['wrong']):
            with self.assertRaises(ValueError):expense.reserve_parent(self.admission,self.parent)
    def test_another_live_parent_blocks_execution_not_metadata(self):
        expense._admission(self.admission);p=self.root/'jobs/other';p.mkdir();(p/'status.json').write_text(json.dumps({'job_id':'other'}))
        with patch('evcs_experiments.runtime.jobctl.observation',return_value={'blocks_new_launch':True}):
            with self.assertRaises(ValueError):self.reserve()
        self.assertFalse((self.root/'operations/saa-saved-model-recovery-claims').exists())

    def test_unreserved_prior_launch_cannot_be_hidden_by_new_job(self):
        other=dict(self.admission_data,parent_job_id='recovery2',parent_execution_path=str(self.op/'parent2.json'))
        ref=self.admit('admission2',other);parent=self.make_parent(ref)
        with self.assertRaisesRegex(ValueError,'unreserved'):
            self.reserve(ref,parent)
        self.assertFalse((self.root/'operations/saa-saved-model-recovery-claims').exists())
    def test_typed_validator_scope_restores_original_aliases(self):
        from evcs_experiments import checkpoint_closure, saa_validator_recovery, formal_planning_worker, saa_planning_worker, saa_record_bridge
        original=(formal_planning_worker._closure,saa_planning_worker._closure,saa_record_bridge._closure)
        installer=batch.file_reference(saa_validator_recovery.__file__)
        old=self.put('validator-old-worker.json',dict(planning_source_references=[installer]))
        policy=self.put('validator-policy.json',dict(original_worker_reference=old,original_validator_installer_reference=installer,validator_reference=batch.file_reference(checkpoint_closure.__file__)))
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with expense._validator_scope(dict(domain='formal',policy_reference=policy)):
                self.assertIs(formal_planning_worker._closure,saa_planning_worker._closure)
                self.assertIsNot(formal_planning_worker._closure,original[0])
                leaf=self.put('typed-leaf.json',{'answer':42})
                saa_record_bridge._closure({'nested':leaf})
                raise RuntimeError('deliberate')
        self.assertEqual((formal_planning_worker._closure,saa_planning_worker._closure,saa_record_bridge._closure),original)
