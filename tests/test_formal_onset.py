"""Component admission tampering and a small genuine physical OR engine.

Signed JSON below is synthetic gate data, never an accepted research bank.
Only the physics preparation boundary is injected in the small engine test;
C&CG, full Q, raw evidence and probability arithmetic are real.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evcs_experiments.ccg import run_ccg
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.formal_planning import BaselinePhysicalPreparation, RawPlanningEvidence
from evcs_experiments.formal_onset import validate_onset_acceptance, load_formal_onset_problem, prepare_onset_physics
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases


class FormalOnsetTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.evidence=self.write('evidence',{'synthetic':True})
        self.protocol=self.write('protocol',{'synthetic':'unit gate only'})
        self.source=self.write('source',{'synthetic':'unit gate only'})
        self.summary=dict(status='COMPLETE_FORMAL_ONSET_INPUTS_PENDING_INDEPENDENT_AUDIT',atom_count=1920,
            formal_OOS=False,formal_plans_solved=False,source_manifest=self.source,formal_protocol_lock=self.protocol)
        self.summary_ref=self.write('summary',self.summary)
        self.audit=dict(status='PASS_INDEPENDENT_FORMAL_BANK_AUDIT',source_summary_sha256=self.summary_ref['sha256'],
            source_output=str(self.root),scenario_count=1920,formal_OOS=False,formal_plans_solved=False,
            full_S05_pass=False,bank_kind='onset_v1',lead_h=0.,audit_source_sha256=self.evidence['sha256'],
            formal_protocol_lock=self.protocol,source_manifest_sha256=self.source['sha256'])
        self.publish()

    def write(self,name,value):
        path=self.root/(name+'.json');path.write_text(json.dumps(value,sort_keys=True)+'\n');return file_reference(path)

    def publish(self):
        self.audit_ref=self.write('audit',self.audit)
        self.card=dict(whole_S05_PASS=False,status='PASS_FORMAL_ONSET_INPUT_BANK_COMPONENT',formal_OOS=False,formal_plans_solved=False,
            bank_summary=self.summary_ref,numerical_audit=self.audit_ref,auditor_source=self.evidence,
            development_review=self.evidence,production_review=self.evidence,operational_completion=self.evidence)
        self.card_ref=self.write('card',self.card)

    def kwargs(self):
        return dict(onset_acceptance_reference=self.card_ref,summary_reference=self.summary_ref,
                    audit_reference=self.audit_ref,protocol_reference=self.protocol)

    def test_explicit_component_scope_and_protocol(self):
        self.assertEqual(validate_onset_acceptance(**self.kwargs()),self.card)
        wrong=self.kwargs();wrong['protocol_reference']=self.evidence
        with self.assertRaisesRegex(ValueError,'worker protocol'):validate_onset_acceptance(**wrong)
        self.card['status']='PASS_COMPONENT';self.card_ref=self.write('card',self.card)
        with self.assertRaisesRegex(ValueError,'component acceptance'):validate_onset_acceptance(**self.kwargs())

    def test_resealed_wrong_epoch_kind_checker_and_incomplete_scope(self):
        original=deepcopy(self.audit)
        for key,value in [('lead_h',1.),('lead_h',False),('bank_kind','baseline_v1'),('audit_source_sha256','0'*64),
                          ('scenario_count',160),('formal_OOS',True),('status','DIAGNOSTIC_ONLY_NOT_BANK_PASS')]:
            self.audit=deepcopy(original);self.audit[key]=value;self.publish()
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_onset_acceptance(**self.kwargs())

    def test_changed_strong_reference_and_swapped_summary_fail(self):
        Path(self.evidence['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):validate_onset_acceptance(**self.kwargs())
        self.evidence=self.write('evidence',{'synthetic':True});self.publish()
        wrong=self.kwargs();wrong['summary_reference']=self.write('other_summary',self.summary)
        with self.assertRaisesRegex(ValueError,'different summary'):validate_onset_acceptance(**wrong)

    def test_preparation_rejects_resealed_baseline_index_before_loading_physics(self):
        ids=['target'+str(i) for i in range(8)]
        payload=dict(domain='formal',target_ids=ids,configuration=dict(baseline=dict(
            support_S=160,fleet_E=80,rho=1,lead_h=1,SOC_arr_min=.3,tau_blk_h=1)))
        self.protocol=self.write('protocol',dict(kind='protocol_lock',payload=payload,payload_sha256=canonical_hash(payload)))
        self.source=self.write('source',dict(files=[dict(path='evidence.json',sha256=self.evidence['sha256'])]))
        index=self.write('index',dict(target_event=ids[0],lead_h=1.,split='planning',task_source=self.evidence))
        self.summary.update(source_manifest=self.source,formal_protocol_lock=self.protocol,
            task_sources={key:self.evidence for key in ids},targets=[dict(target_event=key,H=1,planning_count=160,calibration_count=80,
                planning_index=index,calibration_index=self.evidence,moments=self.evidence,bootstrap=self.evidence) for key in ids])
        self.summary_ref=self.write('summary',self.summary)
        self.audit.update(source_summary_sha256=self.summary_ref['sha256'],source_manifest_sha256=self.source['sha256'],
            formal_protocol_lock=self.protocol,s01_acceptance=self.evidence,s04_acceptance=self.evidence)
        self.publish()
        kwargs=self.kwargs();kwargs.pop('protocol_reference')
        with patch('evcs_experiments.formal_onset.load_physical_case',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'lead0 index'):
                prepare_onset_physics(**kwargs,target_id=ids[0],method_id='OR')

    def test_wrong_method_refused_before_preparation(self):
        for method in ('EA','ARO','SAA','DSP'):
            with self.subTest(method=method),self.assertRaisesRegex(ValueError,'OR only'):
                load_formal_onset_problem(summary_reference=None,audit_reference=None,onset_acceptance_reference=None,
                    target_id='toy',method_id=method,evidence_root=self.root/'raw')
        with self.assertRaisesRegex(ValueError,'OR only'):
            prepare_onset_physics(summary_reference=None,audit_reference=None,onset_acceptance_reference=None,target_id='toy',method_id='EA')

    def test_small_physical_OR_factory_raw_evidence_and_resume(self):
        f=build_golden_cases()[1];first=assemble_first_stage(f.case,f.tasks)
        templates=[assemble_recourse(f.case,f.tasks,s) for s in f.scenarios]
        bank=SimpleNamespace(scenario_ids=f.scenario_ids,scenario_sha256s=tuple(s.scenario_sha256 for s in f.scenarios),load=f.scenarios.__getitem__)
        binding=dict(domain='synthetic unit fixture',method_id='OR',lead_h=0.,onset_acceptance=self.card_ref)
        physics=BaselinePhysicalPreparation(f.case,f.tasks,bank,first,f.probability_set,binding,templates.__getitem__,None,0.)
        args=dict(summary_reference=self.summary_ref,audit_reference=self.audit_ref,onset_acceptance_reference=self.card_ref,
            target_id='synthetic',method_id='OR',evidence_root=self.root/'raw')
        def development_exporter(*args,**kwargs):
            # Test-only domain injection keeps synthetic scene labels intact.
            kwargs['domain']='development'
            return RawPlanningEvidence(*args,**kwargs)
        with patch('evcs_experiments.formal_onset.prepare_onset_physics',return_value=physics) as prepare, \
             patch('evcs_experiments.formal_onset.RawPlanningEvidence',side_effect=development_exporter):
            prepared=load_formal_onset_problem(**args)
            self.assertEqual(prepare.call_args.kwargs['onset_acceptance_reference'],self.card_ref)
            self.assertEqual(prepared.problem.risk_model,'DRO');self.assertEqual(prepared.input_binding,binding)
            result=run_ccg(prepared.problem,evidence_sink=prepared.sink_factory(self.root/'call'),wall_seconds=20,
                rmp_time_limit_seconds=5,oracle_time_limit_seconds=5,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            self.assertEqual(result.status,'CONVERGED')
            self.assertAlmostEqual(result.state['upper_bound_usd'],f.expected['DRO']['objective_usd'],places=7)
            snapshot=prepared.evidence.snapshot(self.root/'snapshot.json')
            again=load_formal_onset_problem(**args,resume_evidence=True,resume_snapshot_reference=snapshot)
            self.assertEqual(again.problem.identity,prepared.problem.identity)
            self.assertEqual(set(again.evidence.q),set(prepared.evidence.q))
            Path(self.root/'raw/first_stage/identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):
                load_formal_onset_problem(**args,resume_evidence=True,resume_snapshot_reference=snapshot)

if __name__=='__main__':unittest.main()
