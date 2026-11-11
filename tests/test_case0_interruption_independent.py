"""Independent evidence-path tests; transition authorization mocked, no study solve."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from tests import test_case0_evaluation_runner as fixtures
from evcs_experiments import case0_evaluation_runner as runner

class InterruptedEvidenceTests(unittest.TestCase):
    def setUp(self):
        fixtures.Case0AcceptanceContractTests.setUp(self)
        self.base=self.root/'planning/size0/EA_DRO'
        self.attempt=self.base/'continuation-1';self.attempt.mkdir()
        for name in ('solve','start.json','accounting.json'):
            (self.base/name).rename(self.attempt/name)
        (self.attempt/'inputs').symlink_to(self.base/'inputs',target_is_directory=True)
        self.transition=dict(new_attempt_directory=str(self.attempt),original_input_directory=str(self.base/'inputs'))
        self.config=runner.read(self.root/'config.lock.json')
        self.state=runner.read(self.root/'state.json')
        ref=runner.file_reference(self.attempt/'solve/result.json')
        self.state['completed']['size0/EA_DRO'].update(path=ref['path'],sha256=ref['sha256'])
        self.state['configuration_attempt_roots']={'size0/EA_DRO':str(self.attempt)}
        (self.root/'state.json').write_text(json.dumps(self.state))
        self.p=patch('evcs_experiments.case0_interruption.validate_transition',return_value=(self.config,self.transition))
        self.p.start();self.addCleanup(self.p.stop)

    def test_original_input_new_solve_acceptance_inventory(self):
        evidence=runner.planning_evidence(self.root)[2]
        self.assertEqual(evidence['EA_DRO']['input_identity_reference']['path'],str(self.base/'inputs/identity.json'))
        self.assertEqual(evidence['EA_DRO']['result_reference']['path'],str(self.attempt/'solve/result.json'))
        card=runner._read(self.card);audit=runner._read(card['numerical_audit_reference'])
        fields=dict(method_evidence=evidence,result_references={m:e['result_reference'] for m,e in evidence.items()})
        old_paths={str(self.base/'solve/result.json'),str(self.base/'start.json'),str(self.base/'accounting.json')}
        fields['artifact_references']=[r for r in card['artifact_references'] if r['path'] not in old_paths]
        fields['artifact_references'] += [runner.file_reference(self.attempt/name) for name in ('solve/result.json','start.json','accounting.json')]
        audit.update(fields);card.update(fields)
        card['numerical_audit_reference']=runner._write(self.root/'newaudit.json',audit)
        ref=runner._write(self.root/'newacceptance.json',card)
        _,_,results,_=runner.validate_case0_planning_acceptance(self.root,ref)
        self.assertEqual(results['EA_DRO']['method_id'],'EA_DRO')
        self.assertFalse((self.base/'solve').exists())

    def test_retargeted_input_symlink_rejected(self):
        (self.attempt/'inputs').unlink()
        (self.attempt/'inputs').symlink_to(self.root/'planning/size0/SAA/inputs',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'input/attempt transition'):
            runner.planning_evidence(self.root)

class OperationalScopeTests(unittest.TestCase):
    def test_mathematical_source_change_rejected(self):
        from evcs_experiments.case0_interruption import _sources
        old={'src/evcs_experiments/case0_campaign.py':'a','src/evcs_experiments/recourse.py':'b'}
        new=dict(old,**{'src/evcs_experiments/case0_campaign.py':'c','src/evcs_experiments/case0_interruption.py':'d'})
        _sources(old,new)
        new['src/evcs_experiments/recourse.py']='e'
        with self.assertRaisesRegex(ValueError,'mathematical'):_sources(old,new)

    def test_diagnostic_retention_preserves_default_and_scope(self):
        from evcs_experiments.case0_isolation import isolated_command
        args=('case0-expanded-review',['python','-V'],30,0)
        default=isolated_command(*args)
        retained=isolated_command(*args,retain_failed_unit=True)
        self.assertIn('--collect',default);self.assertNotIn('--collect',retained)
        self.assertEqual([x for x in default if x!='--collect'],retained)
        with self.assertRaisesRegex(ValueError,'expanded'):
            isolated_command('case0-original',['python','-V'],30,0,retain_failed_unit=True)

class InterruptedBudgetForwardingTests(unittest.TestCase):
    def test_continuation_child_receives_original_nested_caps(self):
        from tests import test_case0_interruption as fixture
        from evcs_experiments import case0_campaign as campaign
        from types import SimpleNamespace
        f=fixture.InterruptionTests(methodName='test_exact_carryover_reuses_original_inputs_and_preserves_all_science')
        f.setUp();self.addCleanup(f.doCleanups)
        state=f.prepare();state['active_configuration']='size0/EA_DRO';campaign.save_state(f.root,state)
        ctx=SimpleNamespace(first=object(),probability_set=object(),planning_ids=['s'],case=object(),tasks=object(),scenario=lambda i:object(),recourse=lambda i:object())
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}),patch('evcs_experiments.case0_inputs.Case0Context',return_value=ctx),patch('evcs_experiments.case0_planning.run_case0_planning',return_value=(None,'result')) as solve:
            self.assertEqual(campaign.plan_one('EA_DRO',1000.),'result')
        kwargs=solve.call_args.kwargs
        self.assertAlmostEqual(kwargs['deadline'],1000.+1800.-204.150118)
        self.assertAlmostEqual(kwargs['configuration_started_at'],1000.-204.150118)
        self.assertAlmostEqual(kwargs['master_seconds'],1200.-204.025118)
        self.assertEqual(kwargs['solver_feasibility_tolerance'],1e-9)
        self.assertEqual(kwargs['output_dir'],f.oldattempt/'continuation-1/solve')
