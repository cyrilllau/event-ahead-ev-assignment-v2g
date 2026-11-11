"""Input guard tests only: no actual formal bank, jobctl or Gurobi admission.

The preflight schema fixtures below deliberately mock the upstream worker
contract. Their labels are not evidence of actual160 numeric template loading.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_formal as formal
from evcs_experiments import saa_l_shaped_audit_bridge as bridge
from evcs_experiments.validation.s03_toys import build_golden_cases


class FormalSAAInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def write(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value)+'\n');return file_reference(p)

    def metadata(self):
        ids=[f'explicit_mock_support_{i}' for i in range(160)]
        return dict(kind=formal.PREFLIGHT_KIND,domain='formal',algorithm_id=formal.FORMAL_ALGORITHM_ID,
            worker_request_reference={'path':'/explicit-mock/worker.json','sha256':'0'*64,'size_bytes':0},
            source_references=[],formal_admission=False,scientific_acceptance=False,formal_OOS=False,
            solver_calls=0,original_model_unchanged=True,
            physical_identity=dict(scenario_ids=ids,scenario_sha256s=['1'*64]*160,
                template_sha256s=['2'*64]*160,constants_usd=[0.]*160,empirical_probabilities=[1/160]*160),
            rows=[{} for _ in ids])

    def test_development_problem_cannot_be_relabelled_into_formal_solver(self):
        t=build_golden_cases()[0]
        p=kernel.DevelopmentSAAProblem(t.case,t.tasks,t.scenarios,t.scenario_cost_constants_usd)
        object.__setattr__(p,'domain','formal')
        with patch.object(kernel,'_master',side_effect=AssertionError('No native call')):
            with self.assertRaisesRegex(ValueError,'typed formal'):
                kernel.run_saa_l_shaped(p)

    def test_forged_formal_instance_without_factory_is_rejected(self):
        p=object.__new__(formal.FormalStoredSAAProblem)
        with self.assertRaisesRegex(ValueError,'typed formal'):
            formal.validate_formal_problem(p)
        with self.assertRaisesRegex(ValueError,'factory'):
            formal.FormalStoredSAAProblem(None,None,self.root,_token=object())

    def test_bare_metadata_namespace_cannot_be_formal_problem(self):
        with self.assertRaisesRegex(ValueError,'typed formal'):
            formal.validate_formal_problem(SimpleNamespace(domain='formal',formal_admission=True))

    def test_formal_q_sink_refuses_synthetic_input_even_with_valid_sources(self):
        t=build_golden_cases()[0]
        p=kernel.DevelopmentSAAProblem(t.case,t.tasks,t.scenarios,t.scenario_cost_constants_usd)
        with self.assertRaisesRegex(ValueError,'typed formal'):
            bridge.FormalFixedQEvidence(problem=p,source_references=[],output_root=self.root/'q')

    def test_formal_deadline_cannot_be_missing_expired_or_nonfinite(self):
        for deadline in (time.monotonic()-1,float('inf'),float('nan'),True):
            with self.subTest(deadline=deadline),self.assertRaisesRegex(ValueError,'deadline'):
                formal.load_formal_l_shaped_problem({},evidence_root=self.root/'q',
                    preflight_path=self.root/'preflight.json',deadline=deadline)

    def test_preflight_cannot_claim_solver_or_production_acceptance(self):
        for key,value in (('formal_admission',True),('solver_calls',1),('scientific_acceptance',True),('formal_OOS',True)):
            body=self.metadata();body[key]=value;ref=self.write('scope.json',body)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'not a solver'):
                formal.validate_formal_input_reference(ref)

    def test_preflight_requires_all160_ordered_rows_and_exact_empirical_objective(self):
        changes=[lambda b:b['rows'].pop(),lambda b:b['physical_identity']['scenario_ids'].pop(),
            lambda b:b['physical_identity']['scenario_ids'].__setitem__(1,b['physical_identity']['scenario_ids'][0]),
            lambda b:b['physical_identity']['constants_usd'].__setitem__(0,1.),
            lambda b:b['physical_identity']['empirical_probabilities'].__setitem__(0,.5)]
        with patch.object(formal,'_worker_input',return_value={'planning_source_references':[]}),patch.object(formal,'_sources',return_value=[]):
            for n,change in enumerate(changes):
                body=self.metadata();change(body);ref=self.write(f'incomplete_{n}.json',body)
                with self.subTest(change=n),self.assertRaisesRegex(ValueError,'unchanged160'):
                    formal.validate_formal_input_reference(ref)

    def test_trusted_preflight_hash_cannot_be_recomputed_from_edited_bytes(self):
        ref=self.write('changed.json',self.metadata());Path(ref['path']).write_text('{}\n')
        with self.assertRaises(ValueError):formal.validate_formal_input_reference(ref)

    def test_input_worker_must_bind_original_model_method_and_formal_domain(self):
        for domain,method,model in (('development','SAA','IEEE33-v54-Q1'),('formal','EA','IEEE33-v54-Q1'),('formal','SAA','OTHER')):
            ref=self.write('worker.json',dict(schema_version=1,domain=domain,method_id=method,model_id=model,spec_id='S05'))
            with self.subTest(domain=domain,method=method,model=model),self.assertRaisesRegex(ValueError,'formal SAA worker'):
                formal._worker_input(ref)

    def test_formal_q_adapter_preserves_exact_physical_binding_separately_from_runtime_pin(self):
        # This checks real constructor argument handoff with the formal input
        # verifier explicitly mocked. It does not create or solve a formal bank.
        t=build_golden_cases()[0]
        p=kernel.DevelopmentSAAProblem(t.case,t.tasks,t.scenarios,t.scenario_cost_constants_usd)
        ident=dict(p.identity,domain='formal',algorithm_id=formal.FORMAL_ALGORITHM_ID,
            formal_input_reference={'path':'/explicit-mock/preflight.json','sha256':'0'*64})
        binding={'physical_source':'explicit_mock_constructor_handoff'}
        problem=SimpleNamespace(case=t.case,tasks=t.tasks,identity=ident,
            input_binding=binding,scene_loader=t.scenarios.__getitem__)
        with patch.object(formal,'validate_formal_problem',return_value={}),\
             patch.object(bridge.DevelopmentFixedQEvidence,'__init__',return_value=None) as parent:
            bridge.FormalFixedQEvidence(problem=problem,source_references=[],output_root=self.root/'events')
        kw=parent.call_args.kwargs
        self.assertIs(kw['input_binding'],binding)
        self.assertNotIn('formal_input_reference',kw['input_binding'])
        self.assertEqual(kw['scenario_ids'],ident['support_ids'])
        self.assertEqual(kw['domain'],'formal')

    def test_formal_resource_stop_is_not_a_native_timeout_or_partial_master(self):
        # Controlled guard-only fixture, not a formal data or Gurobi run.
        from evcs_experiments.saa_l_shaped_master_evidence import NativeDevelopmentMasterEvidence
        t=build_golden_cases()[0]
        p=kernel.DevelopmentSAAProblem(t.case,t.tasks,t.scenarios,t.scenario_cost_constants_usd)
        object.__setattr__(p,'domain','formal')
        sink=object.__new__(bridge.FormalFixedQEvidence);sink.problem=p
        master=NativeDevelopmentMasterEvidence(self.root/'masters',threads=1)
        with patch.object(formal,'validate_formal_problem',return_value={}),\
             patch.object(kernel,'_master',side_effect=AssertionError('No short formal master')) as native:
            state=kernel.run_saa_l_shaped(p,budget_seconds=7200,wall_seconds=30,
                master_time_limit_seconds=600,evidence_sink=sink,master_sink=master)
        native.assert_not_called()
        self.assertEqual(state['status'],'TIME_LIMIT')
        self.assertEqual(state['master_history'],[])
        self.assertEqual(state['iterations'],0)
        self.assertEqual(state['resource_stop_reason'],'INSUFFICIENT_REMAINING_FOR_FULL_FORMAL_MASTER_GRANT')
        self.assertFalse(state['formal_admission'])

    def test_actual_worker_accounting_payload_passes_original_receipt_settlement(self):
        # Reuses the original development settlement fixture solely for its
        # complete immutable receipt contract, not formal numerical admission.
        import test_formal_record_settlement as original
        from evcs_experiments import saa_l_shaped_worker as worker
        f=original.SettlementTests();f.setUp();self.addCleanup(f.doCleanups)
        accounting=worker._provisional_accounting({'cumulative_plan_seconds':3.,'plan_budget_seconds':100.},5.)
        args=f.fixture(prior=3.,elapsed=6.,mutate_result=lambda r:r.update(accounting))
        actual=original.settle_plan_record(**args)
        self.assertEqual(actual['solver_record']['cumulative_wall_seconds'],9.)
        self.assertEqual(actual['solver_record']['queue_verification_overhead_seconds'],1.)
        self.assertFalse(actual['solver_record']['formal_OOS'])

    def test_internal_worker_cannot_bypass_original_jobctl_prerequisites(self):
        from evcs_experiments import saa_l_shaped_worker as worker
        req=self.write('denied_worker.json',{'domain':'formal'})
        inv=self.write('denied_invocation.json',{'item_request_reference':req})
        with patch.object(worker,'_formal_prerequisites',side_effect=ValueError('No original jobctl admission')) as gate,\
             patch.object(worker,'load_formal_l_shaped_problem',side_effect=AssertionError('No numeric input before admission')):
            with self.assertRaisesRegex(ValueError,'jobctl'):
                worker.run_attempt(inv)
        gate.assert_called_once()
        self.assertFalse((self.root/'worker_result.json').exists())

    def test_restore_keeps_preflight_reference_and_never_starts_after_deadline(self):
        with self.assertRaisesRegex(ValueError,'deadline'):
            formal.restore_formal_l_shaped_problem({},evidence_root=self.root/'q',deadline=time.monotonic()-1)

if __name__=='__main__':unittest.main()
