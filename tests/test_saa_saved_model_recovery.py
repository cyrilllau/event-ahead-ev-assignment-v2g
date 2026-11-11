"""Independent saved-coefficient SAA recovery checks with real tiny physics.

Native memory-failure metadata and parent observations are explicit development
fixtures. Numerical master, fixed-Q and joined audit use the original real local
HiGHS path; this does not claim a commercial solver or formal recovery run.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments import campaign_batch as b
from evcs_experiments import native_gurobi as native
from evcs_experiments import saa_planning_worker as old_worker
from evcs_experiments import saa_saved_model_recovery as recovery
from evcs_experiments import saa_planning_pipeline, saa_record_bridge
from evcs_experiments.random_streams import canonical_hash
import test_saa_planning_pipeline as physical_fixture


class Recourses(tuple):
    pass


class SavedModelRecoveryTests(unittest.TestCase):
    def setUp(self):
        retained=os.environ.get('EVCS_SAA_RECOVERY_TEST_ARTIFACTS_ROOT')
        if retained:
            base=Path(retained).resolve();base.mkdir(parents=True,exist_ok=True)
            def factory(**kwargs):
                path=tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=kwargs.get('dir',base))
                return SimpleNamespace(name=path,cleanup=lambda:None)
        else:
            factory=tempfile.TemporaryDirectory
        self.outer=factory(); self.addCleanup(self.outer.cleanup)
        self.physical=physical_fixture.SAAPipelineTests()
        with patch('tempfile.TemporaryDirectory',side_effect=lambda **kw:factory(dir=self.outer.name)):
            self.physical.setUp()
        self.addCleanup(self.physical.doCleanups)
        self.args=self.physical.args
        self.root=self.physical.f.root.resolve()
        self.op=Path(self.outer.name).resolve()/'recovery_operation'
        self.op.mkdir()
        self.old_output=self.root
        self.new_output=self.op/'new_output'
        self.plan_id='development.saved_model.SAA'
        self.budget=dict(cumulative_wall_seconds=30.,single_master_seconds=5.,
            final_certificate_reserve_seconds=5.,shutdown_reserve_seconds=1.)
        rec=Recourses(self.args['recourses'])
        rec.scenario_ids=tuple(self.args['scenario_ids'])
        rec.scenario_sha256s=tuple(self.args['scenario_sha256s'])
        physical=SimpleNamespace(first_stage=self.args['first_stage'],case=self.args['case'],
            tasks=self.args['tasks'],bank=SimpleNamespace(load=self.args['scene_loader']))
        self.prepared=SimpleNamespace(physical=physical,recourses=rec,
            input_binding=self.args['input_binding'],
            empirical_probabilities=self.physical.model.manifest['empirical_probabilities'],
            scenario_cost_constants_usd=self.physical.model.manifest['scenario_cost_constants_usd'])
        self.fixture=old_worker._DevelopmentFixture(prepare=lambda **kwargs:self.prepared,budget=self.budget)
        self.old=dict(schema_version=1,method_id='SAA',domain='development',plan_id=self.plan_id,target_id='tiny',
            output_root=str(self.old_output),budget=self.budget,model_input_mode='assemble',max_coefficient_bytes=50_000_000,
            helper_reference=self.args['helper_reference'],template_references=self.args['template_references'],
            planning_source_references=self.args['source_references'],summary_reference={},audit_reference={})
        self.oldref=self.write('original_worker.json',self.old)
        self.fee=dict(kind='SAA_ASSEMBLED_MASTER_FAILED_PARENT_EXPENSE_V1',passed=True,accounting_PASS=True,
            status='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION',plan_id=self.plan_id,references={'worker_request':self.oldref},
            SAA_charged_seconds=2.5,SAA_remaining_seconds=27.5)
        feeref=self.write('fee.json',self.fee)
        decisionref=self.write('isolation.json',dict(kind='SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1',
            terminal_fee_settlement_reference=feeref,old_queue_permanently_quarantined=True,
            original_queue_rearm_prohibited=True,SAA_charged_seconds=2.5))
        def historical(name,value):
            p=self.old_output/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value));return b.file_reference(p)
        self.historical=historical
        modelref=self.args['model_manifest_reference']
        completion=historical('model_completion.json',dict(kind='SAA_COMPLETE_MODEL_CHECKPOINT_V1',request_reference=self.oldref,
            model_manifest_reference=modelref,model_input_mode='assemble',planning_source_references=self.old['planning_source_references'],
            input_binding=self.args['input_binding'],prior_cumulative_seconds=0,preparation_elapsed_seconds=2.))
        raw=historical('phases/master/native_result.json',dict(native_interface=dict(raw_model_status=17,solution_count=0,
            effective_options=dict(SoftMemLimit=24.,Threads=4))))
        master=historical('phases/master/result.json',dict(status='MEM_LIMIT',primal=None,lower_bound_usd=None,
            full_support_verified=False,model_manifest=modelref,raw_native_result=raw))
        phase=historical('phases/request.json',dict(model_manifest_reference=modelref,sources=self.old['planning_source_references'],
            input_binding=self.args['input_binding'],scenario_ids=list(self.args['scenario_ids']),scenario_sha256s=list(self.args['scenario_sha256s'])))
        cp=historical('phases/checkpoints/cp.json',dict(status='FAILED',master_result_reference=master,
            fixed_q_checkpoint_reference=None,candidate_audit_reference=None,request_reference=phase))
        terminal=self.write('terminal.json',dict(complete=True,state='FAILED',returncode=2,elapsed_seconds=2.5))
        queue=self.write('oldqueue.json',dict(domain='development'))
        self.fee['references'].update(model_manifest=modelref,model_completion=completion,master_result=master,
            native_master=raw,phase_checkpoint=cp,phase_request=phase)
        self.fee.update(terminal_job_result_reference=terminal,original_queue_reference=queue)
        feeref=self.write('fee.json',self.fee)
        decisionref=self.write('isolation.json',dict(kind='SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1',
            terminal_fee_settlement_reference=feeref,old_queue_permanently_quarantined=True,
            original_queue_rearm_prohibited=True,SAA_charged_seconds=2.5,model_manifest_reference=modelref,
            model_completion_reference=completion,native_failure_reference=raw,terminal_job_result_reference=terminal,
            original_queue_reference=queue))
        self.policy=dict(kind=recovery.KIND,domain='development',adapter_reference=b.file_reference(recovery.__file__),
            soft_memory_limit_gb=64.,original_worker_reference=self.oldref,fee_reference=feeref,isolation_decision_reference=decisionref,
            original_module_references={k:b.file_reference(m.__file__) for k,m in dict(native=native,pipeline=saa_planning_pipeline,
                worker=old_worker,bridge=saa_record_bridge).items()},new_output_root=str(self.new_output),model_manifest_reference=modelref,
            model_completion_reference=completion,failed_master_reference=master,failed_native_reference=raw,failed_phase_checkpoint_reference=cp)
        self.policyref,self.reqref=self.bind(self.policy,'original')
        self.reservationref=self.write('development_expense_witness.json',dict(plan_id=self.plan_id,accounted_seconds=2.5,
            policy_reference=self.policyref,private_fixture_only=True))
        # Parent/claim/terminal are independently tested by the expense component.
        # This explicit witness isolates worker integration; it is never a formal reservation.
        def witness(reference,*,invocation_reference,policy_reference):
            value=self.read(reference);inv=self.read(invocation_reference)
            if policy_reference!=value['policy_reference'] or inv['cumulative_plan_seconds']!=value['accounted_seconds']:
                raise ValueError('Development expense identity differs')
            return dict(plan_id=value['plan_id'],accounted_seconds=value['accounted_seconds'],module_reference=None)
        self.expense=SimpleNamespace(verify_reservation=witness)


    def write(self,name,value):
        path=self.op/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False))
        return b.file_reference(path)

    def read(self,reference):
        b.verify_reference(reference)
        return b.read_json(reference['path'])



    def bind(self,policy,name):
        pr=self.write(name+'.policy.json',policy)
        req=deepcopy(self.old)
        req.update(output_root=policy['new_output_root'],model_input_mode=recovery.MODE,
            model_manifest_reference=policy['model_manifest_reference'],original_worker_reference=self.oldref,
            saved_model_recovery_policy_reference=pr,failure_carryover_reference=policy['fee_reference'])
        return pr,self.write(name+'.worker.json',req)

    def invocation(self,**changes):
        attempt=self.op/'attempt';attempt.mkdir(exist_ok=True)
        now=time.monotonic();started=now-.25
        inv=dict(schema_version=1,domain='development',plan_id=self.plan_id,item_request_reference=self.reqref,
            attempt_started_monotonic=started,worker_deadline_monotonic=started+27.5,remaining_seconds=27.25,
            cumulative_plan_seconds=2.5,opening_carryover_seconds=2.5,failure_carryover_reference=self.policy['fee_reference'],
            plan_budget_seconds=30.,plan_remaining_seconds=27.5,previous_checkpoint=None,previous_result_reference=None,
            attempt_directory=str(attempt),result_path=str(attempt/'worker_result.json'))
        inv.update(changes);return self.write('invocation.json',inv)

    def run_recovery(self,inv):
        return recovery.run_attempt(inv,policy_reference=self.policyref,reservation_reference=self.reservationref,
            expense_module=self.expense,_development_fixture=self.fixture)

    def test_real_saved_arrays_complete_worker_without_reassembly_and_preserve_charge(self):
        refs=list(self.physical.model.manifest['numeric_artifacts'])+[self.oldref,self.policy['model_manifest_reference'],
            self.policy['model_completion_reference'],self.policy['failed_master_reference'],self.policy['failed_phase_checkpoint_reference']]
        with patch('evcs_experiments.saa_streaming.assemble_streaming_saa',side_effect=AssertionError('No assembly')),             patch('evcs_experiments.formal_saa.assemble_prepared_saa',side_effect=AssertionError('No assembly')):
            result=self.run_recovery(self.invocation())
        self.assertEqual(result['status'],'COMPLETE');self.assertTrue(result['full_support_verified'])
        self.assertEqual(result['raw_engine_status'],'OPTIMAL');self.assertFalse(result['formal_OOS'])
        self.assertEqual(result['queue_prior_attempt_seconds'],2.5)
        self.assertGreater(result['attempt_accounted_seconds'],.25)
        self.assertAlmostEqual(result['cumulative_accounted_seconds'],2.5+result['attempt_accounted_seconds'])
        audit=self.read(result['numerical_audit_reference'])
        self.assertEqual(audit['risk_certificate_kind'],'fixed_empirical')
        self.assertAlmostEqual(audit['upper_bound_usd'],-69.73333333333333)
        execution=self.read(result['saved_model_execution_reference'])
        self.assertFalse(execution['reassembly_performed']);self.assertFalse(execution['mathematical_parameters_changed'])
        cp=self.read(result['checkpoint']);phase=self.read(cp['phase_checkpoint_reference'])
        self.assertGreater(phase['prior_accounted_seconds'],2.75)
        request=self.read(phase['request_reference'])
        self.assertEqual(request['model_manifest_reference'],self.args['model_manifest_reference'])
        self.assertEqual(request['scenario_sha256s'],list(self.args['scenario_sha256s']))
        self.assertEqual(request['input_binding'],self.args['input_binding'])
        self.assertFalse((self.new_output/'model').exists())
        for ref in refs:self.assertEqual(b.file_reference(ref['path']),ref)
        self.write('positive_numeric_summary.json',dict(status='PASS_DEVELOPMENT_SAVED_MODEL_WORKER',
            worker_result_reference=b.file_reference(self.op/'attempt/worker_result.json'),
            model_manifest_reference=self.args['model_manifest_reference'],
            candidate_audit_reference=audit['candidate_audit_reference'],numerical_bridge_reference=result['numerical_audit_reference'],
            upper_bound_usd=audit['upper_bound_usd'],lower_bound_usd=audit['lower_bound_usd'],gap_relative=audit['gap_relative'],
            opening_seconds=2.5,charged_attempt_seconds=result['attempt_accounted_seconds'],
            cumulative_seconds=result['cumulative_accounted_seconds'],model_reassembly=False,
            original_fee_fixture=True,expense_reservation_fixture=True,formal_OOS=False,production_ready=False))
        with self.assertRaisesRegex(ValueError,'already exists'):self.run_recovery(self.invocation())

    def test_only_original_soft_memory_option_changes_and_restores(self):
        options=dict(time_limit=600,threads=4,mip_rel_gap=.001)
        original=native._parameters;before=original(options)
        with recovery._memory_policy():
            after=native._parameters(options)
            self.assertEqual(after,(before[0],dict(before[1],SoftMemLimit=64.)))
            self.assertNotIn('Presolve',after[1]);self.assertNotIn('PrePasses',after[1])
            with self.assertRaisesRegex(ValueError,'parameters'):native._parameters(dict(options,threads=1))
            with self.assertRaisesRegex(ValueError,'parameters'):native._parameters(dict(options,time_limit=601))
        self.assertIs(native._parameters,original)

    def test_policy_source_memory_phase_and_worker_scope_rejected(self):
        for index,change in enumerate((dict(soft_memory_limit_gb=128.),dict(new_output_root=str(self.old_output/'overwrite')),
            dict(adapter_reference=self.oldref),dict(domain='formal'))):
            with self.subTest(change=change):
                pr,wr=self.bind(dict(self.policy,**change),'bad'+str(index))
                with self.assertRaises(ValueError):recovery.verify_policy(pr,wr,_development_fixture=self.fixture)
        changed=self.read(self.reqref);changed['max_coefficient_bytes']+=1
        with self.assertRaisesRegex(ValueError,'changed fields'):
            recovery.verify_policy(self.policyref,self.write('changed_worker.json',changed),_development_fixture=self.fixture)
        with self.assertRaisesRegex(ValueError,'Public'):
            recovery.verify_policy(self.policyref,self.reqref)

    def test_changed_native_failure_or_saved_payload_refused(self):
        p=Path(self.policy['failed_native_reference']['path']);p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):recovery.verify_policy(self.policyref,self.reqref,_development_fixture=self.fixture)

    def test_original_debt_and_parent_grant_cannot_reset(self):
        for changes in (dict(cumulative_plan_seconds=0),dict(opening_carryover_seconds=0),dict(plan_remaining_seconds=30.),
                        dict(worker_deadline_monotonic=time.monotonic()+30),dict(previous_checkpoint=self.oldref)):
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'expense|grant'):
                self.run_recovery(self.invocation(**changes))
        self.assertFalse(self.new_output.exists())

    def test_saved_matrix_tamper_rejected_before_master(self):
        row=self.physical.model.manifest['numeric_artifacts'][0]
        path=Path(row['path']);value=bytearray(path.read_bytes());value[-1]^=1;path.write_bytes(value)
        with patch('evcs_experiments.saa_streaming_solve.solve_streaming_saa',side_effect=AssertionError('No master')):
            with self.assertRaises(ValueError):self.run_recovery(self.invocation())
        failure=self.read(b.file_reference(self.op/'attempt/worker_failure.json'))
        self.assertTrue(failure['partial_outputs_preserved']);self.assertGreater(failure['cumulative_accounted_seconds'],2.75)



    def test_actual_prepared_support_change_rejected_without_master(self):
        self.prepared.recourses.scenario_ids=tuple(reversed(self.prepared.recourses.scenario_ids))
        with patch('evcs_experiments.saa_streaming_solve.solve_streaming_saa',side_effect=AssertionError('No master')):
            with self.assertRaisesRegex(ValueError,'physical bindings'):
                self.run_recovery(self.invocation())
        self.assertFalse((self.new_output/'phases').exists())

    def test_resealed_policy_cannot_replace_fee_bound_failure(self):
        alternate=self.write('alternate_native.json',dict(native_interface=dict(raw_model_status=17,solution_count=0,
            effective_options=dict(SoftMemLimit=24.,Threads=4))))
        pr,wr=self.bind(dict(self.policy,failed_native_reference=alternate),'foreign_failure')
        with self.assertRaisesRegex(ValueError,'frozen failed-parent'):
            recovery.verify_policy(pr,wr,_development_fixture=self.fixture)



class TypedValidatorAliasTests(unittest.TestCase):
    def test_actual_installer_aliases_source_binding_and_restoration(self):
        from evcs_experiments import checkpoint_closure, saa_validator_recovery, formal_planning_worker
        policy=dict(validator_reference=b.file_reference(checkpoint_closure.__file__))
        original=(formal_planning_worker._closure,old_worker._closure,saa_record_bridge._closure)
        with self.assertRaisesRegex(ValueError,'installed'):recovery.verify_validator_installed(policy)
        restore=saa_validator_recovery.install_validator(policy)
        try:
            recovery.verify_validator_installed(policy)
            with patch.object(old_worker,'_closure',lambda value:None):
                with self.assertRaisesRegex(ValueError,'alias installer'):recovery.verify_validator_installed(policy)
            wrong=dict(validator_reference=dict(policy['validator_reference'],sha256='0'*64))
            with self.assertRaises(ValueError):recovery.verify_validator_installed(wrong)
        finally:restore()
        self.assertEqual((formal_planning_worker._closure,old_worker._closure,saa_record_bridge._closure),original)
        with self.assertRaises(ValueError):saa_validator_recovery.install_validator(wrong)
        self.assertEqual((formal_planning_worker._closure,old_worker._closure,saa_record_bridge._closure),original)


if __name__=='__main__':
    unittest.main()
