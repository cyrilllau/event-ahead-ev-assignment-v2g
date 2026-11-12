"""Real separate saved-model parent/child, numerical seal and expense join.

Only historical failure metadata and the Linux clock on macOS are explicitly
development fixtures. No worker result, numerical witness, or receipt is mocked.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from evcs_experiments import campaign_batch as b, saa_saved_model_expense as expense
from evcs_experiments import saa_saved_model_recovery as recovery
from evcs_experiments import checkpoint_closure, saa_validator_recovery
from evcs_experiments.config import get_paths
from evcs_experiments.canonical_in_attempt import POLICY, verify_publication
from evcs_experiments.evaluation_lock import PROTOCOL_REFERENCE_KINDS, create_protocol_lock
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime import jobctl
import test_saa_saved_model_recovery as worker_fixture


ENTRY = r'''
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json, os, sys, time
from evcs_experiments import campaign_batch as b
from evcs_experiments.saa_planning_worker import _DevelopmentFixture
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases

def fixture(a):
    w=b.read_json(a['worker_request_reference']['path'])
    assert a['domain']==w['domain']=='development'
    toy=build_golden_cases()[0]
    class Recourses(tuple): pass
    rec=Recourses(assemble_recourse(toy.case,toy.tasks,s) for s in toy.scenarios)
    rec.scenario_ids=toy.scenario_ids
    rec.scenario_sha256s=tuple(s.scenario_sha256 for s in toy.scenarios)
    physical=SimpleNamespace(first_stage=rec[0].first_stage,case=toy.case,tasks=toy.tasks,
        bank=SimpleNamespace(load=toy.scenarios.__getitem__))
    prepared=SimpleNamespace(physical=physical,recourses=rec,input_binding={'fixture':'pipeline'})
    return _DevelopmentFixture(prepare=lambda **kw:prepared,budget=w['budget'],
        master_seconds=5.,audit_reserve_seconds=5.,seal_reserve_seconds=1.)

@contextmanager
def process_metadata(a):
    # Real parent and child PIDs/argv/session are still checked by the dispatcher.
    # Only the unavailable production Linux supervisor clock is fixture data.
    b.verify_reference(a['development_clock_reference'])
    clock=b.read_json(a['development_clock_reference']['path'])
    job=Path(a['campaign_root'])/'jobs'/a['parent_job_id']
    parent_pid=os.getpid() if os.environ.get('EVCS_BATCH_ATTEMPT_REQUEST') is None else os.getppid()
    parent_identity=b.process_identity(parent_pid)
    assert parent_identity is not None
    role='parent' if parent_pid==os.getpid() else 'child'
    trace=Path(a['parent_execution_path']).parent/('actual_'+role+'_process.json')
    with trace.open('x') as stream:
        json.dump(dict(role=role,pid=os.getpid(),ppid=os.getppid(),argv=[sys.executable,*sys.argv],
            process_identity=b.process_identity(os.getpid()),fixture_source_reference=b.file_reference(__file__),
            original_model_reference=b.read_json(a['policy_reference']['path'])['model_manifest_reference'],
            domain='development',process_metadata_only=True),stream,indent=2)
    def parent(q):
        now=time.monotonic();birth=clock['started_monotonic']
        return dict(job_id=a['parent_job_id'],directory=str(job),manifest_reference=b.file_reference(job/'manifest.json'),
            wall_clock=dict(clock_kind='linux_supervisor_birth_boottime_v1',development_process_clock_fixture=True,
                timeout_seconds=a['parent_timeout_seconds'],supervisor=parent_identity,
                observed_monotonic=now,elapsed_seconds=now-birth,
                conservative_deadline_monotonic=birth+a['parent_timeout_seconds']))
    active=sys.modules['__main__']
    original_scope=active._validator_scope
    @contextmanager
    def traced_scope(value):
        with original_scope(value):
            from evcs_experiments.saa_saved_model_recovery import verify_validator_installed
            policy=b.read_json(value['policy_reference']['path'])
            verify_validator_installed(policy)
            with trace.with_name('actual_'+role+'_typed_scope.json').open('x') as stream:
                json.dump(dict(role=role,pid=os.getpid(),validator_reference=policy['validator_reference'],
                    original_installer_reference=policy['original_validator_installer_reference'],
                    actual_aliases_verified=True),stream,indent=2)
            yield
    with patch.object(b,'_parent',side_effect=parent),patch.object(active,'_validator_scope',side_effect=traced_scope):
        yield
'''


class SavedModelDispatchTests(unittest.TestCase):
    def setUp(self):
        self.f=worker_fixture.SavedModelRecoveryTests('test_real_saved_arrays_complete_worker_without_reassembly_and_preserve_charge')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root=self.f.op; self.op=self.root/'operations/composed'; self.op.mkdir(parents=True)

    def put(self,name,value):
        p=self.op/name; p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        return b.file_reference(p)

    def prepare(self):
        f=self.f
        entry=self.op/'development_entry.py';entry.write_text(ENTRY)
        source=b.file_reference(expense.__file__)
        sources=f.old['planning_source_references']
        identity=dict(plan_id=f.plan_id,target_id='tiny',method_id='SAA',setting_id='baseline')
        config=self.put('configuration.json',dict(planning_budget=f.budget))
        budgetref=self.put('budget.json',f.budget)
        inputs=self.put('physical_inputs.json',dict(task_sha256=f.prepared.physical.tasks.task_sha256,
            config_hash=f.prepared.physical.tasks.config_hash,scenario_ids=list(f.prepared.recourses.scenario_ids)))
        protocol=dict(domain='development',model_version='IEEE33-v54-Q1',target_ids=['tiny'],method_ids=['SAA'],
            configuration=dict(planning_budget=f.budget),references={k:[config] for k in PROTOCOL_REFERENCE_KINDS},
            batches=[dict(batch_id='saved_model',expected_plans=[identity],sample_ids_by_target={'tiny':['never_opened']})])
        protocol['references']['source']=sources;protocol['references']['fixture']=[b.file_reference(entry)]
        pr=create_protocol_lock(self.root/'locks',protocol)
        old=deepcopy(f.old);old.update(config_reference=config,protocol_reference=pr,canonical_identity=identity,
            canonical_batch_id='saved_model',canonical_export_policy=POLICY,
            plan_input_references={k:inputs for k in ('tasks','routes','initial_state','planning_support')})
        oldref=self.put('original_worker.json',old)
        oldroot=self.root/'old_queue';oldroot.mkdir()
        oldq=self.put('old_queue.json',dict(queue_root=str(oldroot),items=[dict(request_reference=oldref,
            plan_id=f.plan_id,plan_budget_seconds=30),dict(plan_id='unstarted.ARO')]))
        oldstate=dict(request_reference=oldq,attempt_receipts=[],halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW')
        (oldroot/'state.json').write_text(json.dumps(oldstate));snapshot=self.put('old_closing_snapshot.json',oldstate)
        terminal=self.put('old_failed_terminal.json',dict(complete=True,state='FAILED',returncode=2,elapsed_seconds=2.5))
        completion=f.read(f.policy['model_completion_reference']);completion['request_reference']=oldref
        Path(f.policy['model_completion_reference']['path']).write_text(json.dumps(completion))
        completion_ref=b.file_reference(f.policy['model_completion_reference']['path'])
        fee=deepcopy(f.fee);fee.update(original_queue_reference=oldq,closing_snapshot_reference=snapshot,
            terminal_job_result_reference=terminal,ARO_charged_seconds=0,ARO_no_invocation=True,ARO_no_worker_outputs=True,
            parent_charged_once=True,nested_seconds_added=False,cumulative_accounted_seconds_decimal='2.5',opening_cumulative_seconds_decimal='0')
        fee['references'].update(worker_request=oldref,model_completion=completion_ref)
        fr=self.put('fee.json',fee)
        decision=f.read(f.policy['isolation_decision_reference']);decision.update(terminal_fee_settlement_reference=fr,
            terminal_job_result_reference=terminal,original_queue_reference=oldq,original_queue_snapshot_reference=snapshot,
            model_completion_reference=completion_ref)
        dr=self.put('isolation.json',decision)
        policy=deepcopy(f.policy);policy.update(original_worker_reference=oldref,fee_reference=fr,isolation_decision_reference=dr,
            model_completion_reference=completion_ref,validator_reference=b.file_reference(checkpoint_closure.__file__),
            original_validator_installer_reference=b.file_reference(saa_validator_recovery.__file__))
        policyref=self.put('policy.json',policy)
        req=deepcopy(old);req.update(output_root=policy['new_output_root'],model_input_mode=recovery.MODE,
            model_manifest_reference=policy['model_manifest_reference'],original_worker_reference=oldref,
            saved_model_recovery_policy_reference=policyref,failure_carryover_reference=fr)
        wr=self.put('worker.json',req)
        qr=self.put('queue.json',dict(schema_version=1,batch_id='saved_model',spec_id='S05',model_id='IEEE33-v54-Q1',
            domain='development',queue_root=str(self.root/'new_queue'),job_wall_seconds=26.,shutdown_reserve_seconds=1.,
            source_references=sources,budget_reference=budgetref, parent_wall_policy='verified_parent_remaining_v1',
            items=[dict(item_id='SAA',plan_id=f.plan_id,request_reference=wr,plan_budget_seconds=30.,
                failure_carryover_reference=fr,argv=[sys.executable,source['path'],'child'],cwd=str(get_paths().project_root))]))
        # This explicit simulated supervisor starts before process creation;
        # its positive prelude is billed, never subtracted from actual runtime.
        clock=self.put('development_clock.json',dict(started_monotonic=time.monotonic()-.25,
            domain='development',fixture=True,prelude_seconds=.25))
        a=dict(kind=expense.KIND,domain='development',policy_reference=policyref,expense_module_reference=source,
            worker_module_reference=b.file_reference(recovery.__file__),worker_request_reference=wr,queue_reference=qr,
            original_queue_source_reference=b.file_reference(b.__file__),original_python=sys.executable,campaign_root=str(self.root),
            parent_execution_path=str(self.op/'parent_execution.json'),parent_job_id='composed-recovery',
            parent_timeout_seconds=26.,parent_grace_seconds=1.,development_entry_reference=b.file_reference(entry),
            development_clock_reference=clock)
        review=self.put('source_review.json',dict(kind=expense.REVIEW,passed=True,admission_payload_sha256=canonical_hash(a),
            expense_module_reference=source,worker_module_reference=a['worker_module_reference'],
            development_fixture_only=True))
        a['independent_review_reference']=review; ar=self.put('admission.json',a)
        job=self.root/'jobs'/a['parent_job_id'];job.mkdir(parents=True)
        manifest=dict(job_id=a['parent_job_id'],argv=expense.parent_argv(a,ar),outroot=str(self.root/'jobs'),threads=1,
            timeout_seconds=26.,grace_seconds=1.,spec_id='S05')
        (job/'manifest.json').write_text(json.dumps(manifest))
        return a,ar,job,clock,oldroot,oldstate

    def test_real_composed_parent_child_canonical_and_terminal_expense(self):
        a,ar,job,clock,oldroot,oldstate=self.prepare()
        env=dict(os.environ,EVCS_JOB_ID=a['parent_job_id'],EVCS_JOB_DIRECTORY=str(job))
        for name in jobctl.THREAD_VARS:env[name]='1'
        # No production process, model, budget, or source changes are injected.
        launch=time.monotonic()
        with (self.op/'parent.stdout.log').open('x') as out,(self.op/'parent.stderr.log').open('x') as err:
            parent=subprocess.Popen(expense.parent_argv(a,ar),cwd=get_paths().project_root,env=env,
                stdout=out,stderr=err,start_new_session=True)
            identity=jobctl.process_identity(parent.pid)
            (job/'status.json').write_text(json.dumps(dict(job_id=a['parent_job_id'],state='RUNNING',
                supervisor=identity,command_process=identity,supervisor_sid=identity['sid'],command_pgid=identity['pgid'])))
            code=parent.wait(timeout=30)
        end=time.monotonic();wall=end-self.f.read(clock)['started_monotonic']
        status=dict(job_id=a['parent_job_id'],complete=True,state='SUCCEEDED' if code==0 else 'FAILED',returncode=code,
            elapsed_seconds=wall,actual_subprocess_seconds=end-launch,supervisor=identity,command_process=identity,
            supervisor_sid=identity['sid'],command_pgid=identity['pgid'],development_supervisor_metadata=True)
        for name in ('status.json','result.json'):(job/name).write_text(json.dumps(status))
        self.assertEqual(code,0,(self.op/'parent.stderr.log').read_text())
        queue=self.f.read(a['queue_reference']);root=Path(queue['queue_root']);state=b.read_json(root/'state.json')
        self.assertIsNone(state['active']);self.assertIsNone(state['halted'])
        self.assertEqual(len(state['attempt_receipts']),1);self.assertEqual(len(state['canonical_publication_receipts']),1)
        receipt_ref=state['attempt_receipts'][0];receipt=self.f.read(receipt_ref);result=self.f.read(receipt['result_reference'])
        self.assertEqual(result['status'],'COMPLETE');self.assertTrue(result['full_support_verified'])
        inv=self.f.read(receipt['invocation_reference']);execution=self.f.read(result['saved_model_execution_reference'])
        reservation_ref=execution['reservation_reference'];reservation=self.f.read(reservation_ref)
        parent_execution=self.f.read(execution['parent_execution_reference'])
        self.assertEqual(parent_execution['pid'],parent.pid)
        childtrace=b.read_json(self.op/'actual_child_process.json')
        parenttrace=b.read_json(self.op/'actual_parent_process.json')
        childpid=childtrace['pid']
        self.assertNotEqual(childpid,parent.pid)
        self.assertEqual(childtrace['ppid'],parent.pid)
        self.assertEqual(parenttrace['pid'],parent.pid)
        self.assertEqual(childtrace['argv'],inv['argv'])
        self.assertEqual(childtrace['original_model_reference'],self.f.policy['model_manifest_reference'])
        self.assertEqual(reservation['parent_pid'],parent.pid)
        self.assertEqual(result['queue_prior_attempt_seconds'],2.5)
        self.assertGreater(result['attempt_accounted_seconds'],.25)
        self.assertEqual(inv['attempt_started_monotonic'],self.f.read(clock)['started_monotonic'])
        parent_path=Path(a['parent_execution_path']);original_parent_bytes=parent_path.read_bytes()
        bad_parent=deepcopy(parent_execution);bad_parent['development_entry_reference']=None
        bad_ref=self.put('negative_parent_fixture_mismatch.json',bad_parent)
        try:
            parent_path.write_bytes(Path(bad_ref['path']).read_bytes())
            with self.assertRaisesRegex(ValueError,'Parent execution development fixture differs'):
                expense.verify_reservation(reservation_ref,invocation_reference=receipt['invocation_reference'])
        finally:
            parent_path.write_bytes(original_parent_bytes)
        self.assertEqual(b.file_reference(parent_path),execution['parent_execution_reference'])
        completion_ref=state['canonical_publication_receipts'][0];finalref=state['finalization_receipts'][0]
        publication=verify_publication(completion_ref,receipt_reference=receipt_ref,finalization_reference=finalref,
            worker_request_reference=a['worker_request_reference'])
        completion=self.f.read(completion_ref);canonical=self.f.read(completion['canonical_record_reference'])
        solver=self.f.read(canonical['references']['solver_record'])
        self.assertEqual(solver['prior_cumulative_wall_seconds'],2.5);self.assertEqual(solver['gap_relative'],0.)
        closing=self.put('closing_snapshot.json',state)
        original_scope=expense._validator_scope
        @contextmanager
        def terminal_scope(value):
            with original_scope(value):
                policy=self.f.read(value['policy_reference'])
                recovery.verify_validator_installed(policy)
                self.put('actual_terminal_typed_scope.json',dict(role='terminal_history',pid=os.getpid(),
                    validator_reference=policy['validator_reference'],actual_aliases_verified=True))
                yield
        with patch.object(expense,'_validator_scope',side_effect=terminal_scope):
            terminal=expense.settle_parent(reservation_ref,b.file_reference(job/'result.json'),closing)
        settled=self.f.read(terminal)
        self.assertTrue(settled['passed'],settled)
        self.assertEqual(settled['original_spent_seconds'],2.5)
        self.assertGreaterEqual(settled['new_queue_charge_seconds'],wall)
        self.assertEqual(settled['cumulative_spent_seconds'],solver['cumulative_wall_seconds'])
        self.assertFalse(settled['automatic_resume_authorized'])
        self.assertEqual(b.read_json(oldroot/'state.json'),oldstate)
        typedrefs=[b.file_reference(self.op/name) for name in ('actual_parent_typed_scope.json',
            'actual_child_typed_scope.json','actual_terminal_typed_scope.json')]
        self.assertTrue(all(self.f.read(ref)['actual_aliases_verified'] for ref in typedrefs))
        summary=dict(status='PASS_REAL_COMPOSED_SAVED_MODEL_DEVELOPMENT',parent_pid=parent.pid,child_pid=childpid,
            source_reference=b.file_reference(expense.__file__),worker_source_reference=b.file_reference(recovery.__file__),
            admission_reference=ar,parent_execution_reference=execution['parent_execution_reference'],
            receipt_reference=receipt_ref,worker_result_reference=receipt['result_reference'],canonical_completion_reference=completion_ref,
            actual_parent_process_reference=b.file_reference(self.op/'actual_parent_process.json'),
            actual_child_process_reference=b.file_reference(self.op/'actual_child_process.json'),
            actual_typed_scope_references=typedrefs,
            canonical_record_reference=completion['canonical_record_reference'],terminal_settlement_reference=terminal,
            solver_gap=solver['gap_relative'],lower_bound_usd=solver['lower_bound_usd'],upper_bound_usd=solver['upper_bound_usd'],
            parent_wall_seconds=wall,actual_subprocess_seconds=end-launch,original_debt_seconds=2.5,
            original_matrix_model_reference=self.f.policy['model_manifest_reference'],reassembly_performed=False,
            explicit_development_process_metadata=True,real_parent_and_child_processes=True,
            real_numerical_audit_and_canonical_seal=True,formal_run=False,formal_OOS=False)
        self.put('composed_summary.json',summary)

    def test_formal_fixture_field_and_tampered_development_source_rejected(self):
        a,ar,_,_,_,_=self.prepare()
        with self.assertRaisesRegex(ValueError,'Formal cannot inject'):
            with expense._development_scope({**a,'domain':'formal'}):
                self.fail('Formal fixture entered')
        p=Path(a['development_entry_reference']['path']);p.write_text(p.read_text()+'\n# changed fixture\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            with expense._development_scope(a):
                self.fail('Changed fixture entered')


if __name__=='__main__':unittest.main()
