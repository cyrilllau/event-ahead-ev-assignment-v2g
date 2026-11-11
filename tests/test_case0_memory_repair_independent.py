"""Independent memory-repair evidence-path/admission tests; no study solves."""
import json
from unittest.mock import patch
from tests import test_case0_interruption_independent as prior
from evcs_experiments import case0_evaluation_runner as runner

class MemoryRepairEvidenceTests(prior.InterruptedEvidenceTests):
    def setUp(self):
        super().setUp()
        self.p.stop()
        destination=self.base/'memory-repair-1';self.attempt.rename(destination);self.attempt=destination
        self.transition['new_attempt_directory']=str(self.attempt)
        self.state['configuration_attempt_roots']={'size0/EA_DRO':str(self.attempt)}
        r=runner.file_reference(self.attempt/'solve/result.json')
        self.state['completed']['size0/EA_DRO'].update(path=r['path'],sha256=r['sha256'])
        (self.root/'state.json').write_text(json.dumps(self.state))
        self.memory_patch=patch('evcs_experiments.case0_memory_repair.validate_transition',return_value=(self.config,self.transition))
        self.memory_patch.start();self.addCleanup(self.memory_patch.stop)

    def test_completed_map_cannot_reuse_previous_attempt(self):
        self.state['configuration_attempt_roots']={'size0/EA_DRO':str(self.base/'continuation-1')}
        (self.root/'state.json').write_text(json.dumps(self.state))
        with self.assertRaises((ValueError,KeyError,FileNotFoundError)):
            runner.planning_evidence(self.root)

    def test_unrequested_zero_presolve_result_flag_is_rejected(self):
        p=self.attempt/'solve/result.json';result=runner.read(p);result['zero_presolve']=True;p.write_text(json.dumps(result))
        r=runner.file_reference(p);self.state['completed']['size0/EA_DRO'].update(path=r['path'],sha256=r['sha256'])
        (self.root/'state.json').write_text(json.dumps(self.state))
        with self.assertRaisesRegex(ValueError,'zero|presolve|reduction'):
            runner.planning_evidence(self.root)

    def test_new_producer_config_cannot_omit_requested_zero_proof_mode(self):
        config=runner.read(self.root/'config.lock.json');config['solver_zero_presolve']=True
        (self.root/'config.lock.json').write_text(json.dumps(config));self.config.update(config)
        sha=runner.file_reference(self.root/'config.lock.json')['sha256']
        for method in runner.METHODS:
            d=self.attempt if method=='EA_DRO' else self.root/'planning/size0'/method
            start=runner.read(d/'start.json');start['config_sha256']=sha;(d/'start.json').write_text(json.dumps(start))
        with self.assertRaisesRegex(ValueError,'zero|presolve|reduction'):
            runner.planning_evidence(self.root)

    def test_plan_attempt_rejects_reset_nested_budget(self):
        from evcs_experiments import case0_campaign as campaign
        self.transition.update(prior_configuration_seconds=406.8766127415018,prior_master_seconds=406.667808)
        state=dict(self.state,configuration_spent_seconds={'size0/EA_DRO':406.8766127415018},configuration_master_spent_seconds={'size0/EA_DRO':406.667808})
        folder,spent=campaign.configuration_attempt(self.root,state,'size0/EA_DRO')
        self.assertEqual(folder,self.attempt);self.assertEqual(spent,406.8766127415018)
        state['configuration_master_spent_seconds']['size0/EA_DRO']=0.
        with self.assertRaisesRegex(ValueError,'carryover ledger'):
            campaign.configuration_attempt(self.root,state,'size0/EA_DRO')

import unittest
from types import SimpleNamespace

class MemoryRepairAdmissionIndependentTests(unittest.TestCase):
    def setUp(self):
        from tests.test_case0_memory_repair import MemoryRepairTests
        self.f=MemoryRepairTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_missing_pending_diagnostic_authority_is_not_free_or_rechargeable(self):
        from evcs_experiments import case0_campaign as c
        f=self.f;state=c.read(f.root/'state.json');state.pop('pending_diagnostic_accounting_reference')
        c.save_state(f.root,state);f.refs['state']=f.ref(f.root/'state.json');f.seal_request()
        with self.assertRaisesRegex(ValueError,'diagnostic|prepared'):
            f.prepare()

    def test_same_bytes_different_prepared_ref_cannot_replace_authority(self):
        f=self.f;f.refs['prepared']=f.write(f.base/'duplicate-prepared.json',f.prepared);f.seal_request()
        with self.assertRaisesRegex(ValueError,'diagnostic|prepared'):
            f.prepare()

    def test_original_caps_and_explicit_solver_mode_reach_child(self):
        from evcs_experiments import case0_campaign as c
        f=self.f;state=f.prepare();state['active_configuration']='size0/EA_DRO';c.save_state(f.root,state)
        self.assertIsNone(state.get('pending_diagnostic_accounting_reference'))
        ctx=SimpleNamespace(first=object(),probability_set=object(),planning_ids=['s'],case=object(),tasks=object(),scenario=lambda i:object(),recourse=lambda i:object())
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}),patch('evcs_experiments.case0_inputs.Case0Context',return_value=ctx),patch('evcs_experiments.case0_planning.run_case0_planning',return_value=(None,'result')) as solver:
            self.assertEqual(c.plan_one('EA_DRO',1000.),'result')
        kwargs=solver.call_args.kwargs
        self.assertIs(kwargs['zero_presolve'],True);self.assertEqual(kwargs['solver_feasibility_tolerance'],1e-9)
        self.assertAlmostEqual(kwargs['master_seconds'],793.332192)
        self.assertAlmostEqual(kwargs['deadline'],1000.+1800.-406.8766127415018-.125)
        self.assertAlmostEqual(kwargs['configuration_started_at'],1000.-406.8766127415018-.125)
        self.assertEqual(kwargs['output_dir'],f.attempt/'memory-repair-1/solve')

    def test_memory_guard_has_distinct_exclusive_claim_and_fails_active_zero_exit(self):
        import os,sys
        from unittest.mock import Mock
        from evcs_experiments import case0_campaign as c,case0_interruption as guard
        f=self.f;state=f.prepare();op=f.root/'operations/memory-guard';job=f.root/'jobs/memory-guard';job.mkdir()
        argv=[sys.executable,'-m','evcs_experiments.case0_interruption','guarded-plan','--root',str(f.root),'--operation',str(op),'--unit','case0-expanded-memory-guard','--seconds','7180','--cpu','0']
        f.write(job/'manifest.json',dict(job_id='memory-guard',argv=argv,outroot=str(f.root/'jobs'),threads=1,timeout_seconds=7200,grace_seconds=5))
        def child(_):
            latest=c.read(f.root/'state.json');latest.update(status='PLANNING',active_configuration='size0/EA_DRO');c.save_state(f.root,latest)
            return Mock(wait=Mock(return_value=0))
        with patch.dict(os.environ,EVCS_JOB_ID='memory-guard',EVCS_JOB_DIRECTORY=str(job)),patch.object(guard,'linger_enabled',return_value=True),patch.object(guard,'resource_sample',return_value={}),patch.object(guard.subprocess,'Popen',side_effect=child):
            self.assertEqual(guard.guarded_plan(f.root,op,'case0-expanded-memory-guard'),1)
        claim=c.read(f.root/'revisions/memory-repair-1/guard_claim.json')
        self.assertEqual(claim['kind'],'CASE0_MEMORY_REPAIR_GUARD_CLAIM_V1');self.assertEqual(claim['transition_reference'],state['memory_repair'])
        result=c.read(op/'guard_failure.json');self.assertFalse(result['passed']);self.assertEqual(result['kind'],'CASE0_MEMORY_REPAIR_GUARD_RESULT_V1')
