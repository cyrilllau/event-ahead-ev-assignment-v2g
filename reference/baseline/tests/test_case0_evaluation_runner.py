"""Acceptance wiring fixtures, not real Case0 numerical audit reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evcs_experiments import case0_evaluation_runner as m
from evcs_experiments.random_streams import canonical_hash


class Case0AcceptanceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        producer=self.root/'producer';producer.mkdir();source=producer/'core.py';source.write_text('# fixture')
        source_ref=m.file_reference(source)
        manifest=m._write(producer/'RELEASE_MANIFEST.json',dict(files=[dict(source_ref,path='core.py')]))
        config=dict(backend='gurobi',configuration_seconds=1800,gap_target=.005,sizes=[dict(fleet_per_depot=3,planning_variants=2),dict(fleet_per_depot=2,planning_variants=1)],variant_start=10000,source_sha256={'core.py':source_ref['sha256']})
        config_ref=m._write(self.root/'config.lock.json',config)
        refined=getattr(self,'refined',False);size_index=1 if refined else 0
        active_ref=config_ref
        if refined:
            revision=self.root/'revisions/numerical-refinement-1';revision.mkdir(parents=True)
            active_ref=m._write(revision/'config.json',dict(config,solver_feasibility_tolerance=1e-9))
        results={};inputs={};artifacts=[];completed={}
        for method in m.METHODS:
            d=self.root/'planning'/f'size{size_index}'/method
            if refined and method=='OR_DRO':d=d/'refinement-1'
            (d/'inputs').mkdir(parents=True);(d/'solve').mkdir()
            task=m._write(d/'inputs/tasks.json',{'fixture_only':True})
            identity=dict(task_sha256='a'*64,lead_h=0 if method=='OR_DRO' else 1,task_file=task,vehicle_ids=['fixture'])
            inputs[method]=m._write(d/'inputs/identity.json',identity)
            plan=dict(x=[0.],r=[[0.]],task_sha256='a'*64,config_hash='b'*64)
            result=dict(kind='CASE0_FULL_FORMULATION_NUMERICAL_RESULT_V1',method_id=method,numerical_valid=True,
                full_support_independently_evaluated=True,gap_passed=True,gap=0.,backend='gurobi',configuration_wall_seconds=1.,
                plan=plan,plan_sha256=canonical_hash(plan),scenario_ids=['fixture-planning'])
            results[method]=m._write(d/'solve/result.json',result)
            completed[f'size{size_index}/'+method]=dict(path=results[method]['path'],sha256=results[method]['sha256'])
            start=m._write(d/'start.json',dict(method=method,config_sha256=(config_ref if method=='EA_DRO' else active_ref)['sha256']))
            accounting=m._write(d/'accounting.json',dict(configuration_wall_seconds=1.))
            artifacts.extend([task,inputs[method],results[method],start,accounting])
        state=dict(size_index=size_index,active_configuration=None,status='PLANNING_REVIEW_REQUIRED',completed=completed)
        if refined:
            retry=self.root/'planning/size1/OR_DRO/refinement-1'
            transition=m._write(revision/'transition.json',dict(active_config_reference=active_ref,
                original_config_reference=config_ref,new_attempt_directory=str(retry)))
            state.update(active_config_reference=active_ref,numerical_refinement=transition,
                configuration_attempt_roots={'size1/OR_DRO':str(retry)})
        m._write(self.root/'state.json',state)
        sources=[m.file_reference(m.__file__)]
        review=m._write(self.root/'review.json',dict(kind='CASE0_NUMERICAL_AUDITOR_REVIEW_V1',passed=True,auditor_source_references=sources))
        artifacts.append(config_ref)
        if refined:artifacts.append(active_ref)
        fields=dict(config_reference=active_ref,original_config_reference=config_ref,method_evidence=m.planning_evidence(self.root)[2],size_index=size_index,result_references=results,input_identity_references=inputs,artifact_references=artifacts)
        audit=m._write(self.root/'audit.json',dict(kind='CASE0_INDEPENDENT_PLANNING_NUMERICAL_AUDIT_V1',passed=True,**fields,
            producer_source_manifest_references={method:manifest for method in m.METHODS},auditor_source_references=sources,independent_review_reference=review,original_master_coefficients_checked=True,
            full_q_kkt_physical_checked=True,risk_bounds_checked=True))
        evaluation_sources=[m.file_reference(p) for p in sorted(Path(m.__file__).parent.rglob('*.py'))]
        evaluation_review=m._write(self.root/'evaluation_review.json',dict(kind='CASE0_EVALUATOR_SOURCE_REVIEW_V1',passed=True,evaluation_source_references=evaluation_sources))
        self.card=m._write(self.root/'acceptance.json',dict(kind='CASE0_PLANNING_ACCEPTANCE_V1',passed=True,**fields,numerical_audit_reference=audit,evaluation_source_references=evaluation_sources,evaluation_review_reference=evaluation_review))

    def test_exact_fixture_acceptance_references_are_bound(self):
        _,_,results,_=m.validate_case0_planning_acceptance(self.root,self.card)
        self.assertEqual(set(results),set(m.METHODS))

    def test_valid_audit_card_cannot_open_oos_before_parent_settlement(self):
        state=m.read(self.root/'state.json');state['audit_reservation_reference']=m._write(self.root/'audit-reservation.json',{'pending':True})
        (self.root/'state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'accounting remains unsettled'):
            m.validate_case0_planning_acceptance(self.root,self.card)

    def test_missing_numerical_audit_cannot_open_oos(self):
        card=m._read(self.card);card.pop('numerical_audit_reference');p=self.root/'missing.json';p.write_text(json.dumps(card))
        with self.assertRaises(KeyError):m.validate_case0_planning_acceptance(self.root,m.file_reference(p))

    def test_changed_original_task_bytes_rejected(self):
        (self.root/'planning/size0/EA_DRO/inputs/tasks.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):m.validate_case0_planning_acceptance(self.root,self.card)

    def test_resealed_audit_scope_mismatch_rejected(self):
        card=m._read(self.card);audit=m._read(card['numerical_audit_reference']);audit['size_index']=1
        p=self.root/'foreign_audit.json';p.write_text(json.dumps(audit));card['numerical_audit_reference']=m.file_reference(p)
        q=self.root/'foreign_card.json';q.write_text(json.dumps(card))
        with self.assertRaisesRegex(ValueError,'scope'):m.validate_case0_planning_acceptance(self.root,m.file_reference(q))

class Case0RefinedAcceptanceTests(unittest.TestCase):
    refined=True
    setUp=Case0AcceptanceContractTests.setUp

    def test_active_config_and_exact_refined_or_old_ea_are_preserved(self):
        config,state,results,_=m.validate_case0_planning_acceptance(self.root,self.card)
        evidence=m.planning_evidence(self.root)[2]
        self.assertEqual(config['solver_feasibility_tolerance'],1e-9)
        self.assertIn('/OR_DRO/refinement-1/',evidence['OR_DRO']['result_reference']['path'])
        self.assertEqual(evidence['EA_DRO']['producer_config_reference'],m.file_reference(self.root/'config.lock.json'))
        self.assertEqual(evidence['SAA']['producer_config_reference'],state['active_config_reference'])

    def test_preparation_reopens_exact_refinement_inputs_and_reuses_identical_lock(self):
        calls=[]
        class ExistingInputs:
            def __init__(context,directory,**settings):
                calls.append((directory,settings));context.identity=m.read(directory/'identity.json')
                context.planning_ids=['fixture-planning']
            def sample_id(context,index,*,split,regime):
                self.assertEqual(split,'oos');return regime+':'+str(index)
        with patch.object(m,'campaign_root',return_value=self.root),patch.object(m,'Case0Context',ExistingInputs):
            first,_,plans=m.prepare_case0_evaluation_lock(acceptance_reference=self.card)
            original=Path(first['path']).read_bytes()
            second,_,_=m.prepare_case0_evaluation_lock(acceptance_reference=self.card)
        self.assertEqual(first,second);self.assertEqual(Path(first['path']).read_bytes(),original)
        self.assertTrue(any('/OR_DRO/refinement-1/inputs' in str(d) and kw['lead_hours']==0 for d,kw in calls))
        self.assertEqual(m.plan_identity(plans['DSP']),m.plan_identity(plans['EA_DRO']))
        lock=m._read(first);self.assertEqual(lock['config_reference'],m.read(self.root/'state.json')['active_config_reference'])
        self.assertEqual(lock['item_count'],768)

    def test_completed_map_cannot_redirect_to_failed_or_original_root(self):
        state=m.read(self.root/'state.json');state['configuration_attempt_roots']={}
        (self.root/'state.json').write_text(json.dumps(state))
        with self.assertRaises(FileNotFoundError):m.validate_case0_planning_acceptance(self.root,self.card)

    def test_old_config_cannot_label_refined_group(self):
        card=m._read(self.card);card['config_reference']=m.file_reference(self.root/'config.lock.json')
        ref=m._write(self.root/'oldlabel.json',card)
        with self.assertRaisesRegex(ValueError,'configuration/size'):m.validate_case0_planning_acceptance(self.root,ref)

if __name__=='__main__':unittest.main()
