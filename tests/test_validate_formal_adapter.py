"""Full 1920-call orchestration fixture; real physics is separately covered."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from evcs_experiments.config import get_paths

MODULE_PATH=get_paths().project_root/'scripts/verification/validate_formal_adapter.py'
spec=importlib.util.spec_from_file_location('formal_adapter_validation_tested_entry',MODULE_PATH)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class FormalAdapterValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.bank=self.root/'bank';self.bank.mkdir();self.source=self.root/'source';self.source.mkdir()
        self.cards=self.root/'cards';self.cards.mkdir();self.output=self.root/'output'
        self.events=[f'target{i}' for i in range(8)];self.loaded=[];self.fail_at=None;self.delay=0
        self.paths=SimpleNamespace(project_root=self.source,data_root=self.root/'data',frozen_source_root=self.root/'legacy',as_dict=lambda:{'fixture':True})
        self.entry=self.source/'scripts/verification/validate_formal_adapter.py';self.entry.parent.mkdir(parents=True);self.entry.write_text('# bound synthetic entry')
        self.adapter_file=self.source/'src/evcs_experiments/formal_physical_inputs.py';self.adapter_file.parent.mkdir(parents=True);self.adapter_file.write_text('# bound synthetic adapter')
        self.manifest=self.write(self.source/'RELEASE_MANIFEST.json',dict(files=[dict(path=str(p.relative_to(self.source)),sha256=v.sha256(p)) for p in (self.entry,self.adapter_file)]))
        self.protocol=self.write(self.cards/'protocol.json',dict(payload=dict(domain='formal',target_ids=self.events)))
        self.stub=self.write(self.bank/'raw.json',{'fixture':True})
        self.accepted=self.write(self.cards/'previous_acceptance.json',{'status':'PASS'})
        rows=[];taskrefs={};self.indices={}
        for target in self.events:
            taskrefs[target]=self.write(self.bank/(target+'.tasks.json'),dict(vehicle_summaries=[{}]*80,task_count=1))
            row=dict(target_event=target,H=1,planning_count=160,calibration_count=80,moments=self.stub,bootstrap=self.stub)
            for split,count in (('planning',160),('calibration',80)):
                items=[]
                for i in range(count):
                    sid=f'{target}/formal/{split}/base/v{i+1:03d}/R1'
                    items.append(dict(scenario_id=sid,scenario_sha256=v.canonical_hash(sid),manifest=self.stub))
                ir=self.write(self.bank/(target+'.'+split+'.json'),dict(target_event=target,split=split,task_source=taskrefs[target],scenarios=items))
                row[split+'_index']=ir;self.indices[ir['path']]=items
            rows.append(row)
        # The stub replaces only physical loading. Storage bytes remain bound.
        self.stub=self.write(self.bank/'raw.json',dict(files={'arrays':self.accepted,'ledger':self.accepted,'diagnostics':self.accepted}))
        for row in rows:
            row['moments']=row['bootstrap']=self.stub
            for split in ('planning','calibration'):
                ir=row[split+'_index'];index=json.loads(Path(ir['path']).read_text())
                for item in index['scenarios']:item['manifest']=self.stub
                row[split+'_index']=self.write(Path(ir['path']),index);self.indices[ir['path']]=index['scenarios']
        self.summary=self.write(self.bank/'summary.json',dict(status='COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT',
            atom_count=1920,formal_OOS=False,formal_plans_solved=False,source_manifest=self.manifest,formal_protocol_lock=self.protocol,
            targets=rows,task_sources=taskrefs))
        self.audit=self.write(self.cards/'audit.json',dict(status='PASS_INDEPENDENT_FORMAL_BANK_AUDIT',scenario_count=1920,
            source_summary_sha256=self.summary['sha256'],source_output=str(self.bank),source_manifest_sha256=self.manifest['sha256'],
            formal_protocol_lock=self.protocol,s01_acceptance=self.accepted,s04_acceptance=self.accepted))
        self.card=self.write(self.cards/'card.json',dict(status='PASS_COMPONENT',component_id='FORMAL_BASELINE_INPUTS',spec_id='S05',increment_id='S05-I01',
            scenario_count=1920,target_count=8,formal_OOS=False,formal_plans_solved=False,full_S05_pass=False,
            summary_reference=self.summary,audit_reference=self.audit,source_manifest_reference=self.manifest,protocol_reference=self.protocol,
            independent_review_reference=self.accepted,job_result_reference=self.accepted,artifact_manifest_reference=self.accepted))

    def write(self,path,value):path.write_text(json.dumps(value));return v.reference(path)

    def run_validation(self,wall=590):
        owner=self
        class FakeAdapter:
            def __init__(self,path,*a,**k):
                self.index=json.loads(Path(path).read_text());self.scenario_ids=tuple(r['scenario_id'] for r in self.index['scenarios'])
            def load(self,i,expected_scenario_sha256):
                if owner.delay:time.sleep(owner.delay)
                if len(owner.loaded)==owner.fail_at:raise ValueError('injected physical replay error')
                sid=self.scenario_ids[i];owner.loaded.append(sid)
                return SimpleNamespace(scenario_id=sid,scenario_sha256=expected_scenario_sha256,target_event=self.index['target_event'],
                    split=self.index['split'],development=False,task_ids=('task',),task_sha256='a'*64,
                    metadata=dict(verified_array_count=26,scenario_provenance=dict(array_hashes={str(j):'a'*64 for j in range(26)})))
        # Verify source origin against this explicit injected fixture module.
        FakeAdapter.__module__=__name__
        with ExitStack() as stack:
            stack.enter_context(patch.object(v,'get_paths',return_value=self.paths))
            stack.enter_context(patch.object(v,'loaded_source_references',return_value=[v.reference(self.adapter_file)]))
            stack.enter_context(patch.object(v,'__file__',str(self.entry)))
            stack.enter_context(patch(__name__+'.__file__',str(self.adapter_file)))
            stack.enter_context(patch.object(v,'FormalPhysicalBank',FakeAdapter))
            stack.enter_context(patch.object(v,'load_physical_case',return_value=object()))
            stack.enter_context(patch.object(v,'load_frozen_tasks',return_value=SimpleNamespace(task_ids=('task',))))
            return v.validate(self.summary['path'],self.summary['sha256'],self.card['path'],self.card['sha256'],
                self.source,self.manifest['sha256'],self.output,wall)

    def test_complete_matrix_invokes_adapter1920_times_and_seals_full_inventory(self):
        result=self.run_validation();self.assertEqual(result['scenario_count'],1920);self.assertEqual(len(set(self.loaded)),1920)
        self.assertEqual(result['checked_array_count'],49920);self.assertEqual(len(result['targets']),8)
        self.assertFalse(result['scientific_acceptance']);self.assertEqual(result['solver_calls'],0)
        for ref in json.loads((self.output/'artifact_manifest.json').read_text()):self.assertEqual(v.reference(ref['path']),ref)
        with self.assertRaises(FileExistsError):self.run_validation()

    def test_atom_failure_keeps_exact_prefix_and_no_pass(self):
        self.fail_at=7
        with self.assertRaisesRegex(ValueError,'physical replay error'):self.run_validation()
        r=json.loads((self.output/'result.json').read_text());self.assertEqual(r['scenario_count'],7)
        self.assertEqual(r['current']['scenario_index'],7);self.assertEqual(r['status'],'FAIL_FORMAL_ADAPTER_VALIDATION')
        self.assertEqual(len(json.loads((self.output/'completed_prefix.json').read_text())),7)

    def test_expired_wall_cannot_pass_or_skip_remainder(self):
        self.delay=.02
        with self.assertRaises(TimeoutError):self.run_validation(.1)
        r=json.loads((self.output/'result.json').read_text());self.assertEqual(r['status'],'TIME_LIMIT_INCOMPLETE_ADAPTER_VALIDATION')
        self.assertLess(r['scenario_count'],1920)

    def test_final_seal_timeout_preserves_unaccepted_pass_but_final_result_is_failure(self):
        real_clock=v.time.monotonic;offset=[0.];original=v.write_new
        def write(path,value):
            result=original(path,value)
            if Path(path).name=='artifact_manifest.json' and offset[0]==0:offset[0]=600.
            return result
        with patch.object(v.time,'monotonic',side_effect=lambda:real_clock()+offset[0]),patch.object(v,'write_new',side_effect=write):
            with self.assertRaises(TimeoutError):self.run_validation()
        result=json.loads((self.output/'result.json').read_text())
        self.assertEqual(result['status'],'TIME_LIMIT_INCOMPLETE_ADAPTER_VALIDATION')
        self.assertEqual(json.loads((self.output/'checkpoint.json').read_text())['status'],result['status'])
        self.assertTrue((self.output/'unaccepted_result.json').exists())
        for ref in json.loads((self.output/'artifact_manifest.json').read_text()):self.assertEqual(v.reference(ref['path']),ref)

    def test_mixed_imported_release_is_rejected(self):
        with patch.dict(v.sys.modules,{'evcs_experiments.foreign_source':SimpleNamespace(__file__='/tmp/foreign_release.py')}):
            with self.assertRaisesRegex(ValueError,'Mixed or unbound'):
                v.loaded_source_references(self.source,{self.adapter_file})

    def test_changed_acceptance_external_hash_is_rejected_before_calls(self):
        Path(self.card['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Externally pinned'):self.run_validation()
        self.assertEqual(self.loaded,[])

    def test_changed_acceptance_or_source_rejected_before_adapter_calls(self):
        self.adapter_file.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Release source changed'):self.run_validation()
        self.assertEqual(self.loaded,[])

    def test_output_inside_input_tree_rejected_without_writes(self):
        self.output=self.bank/'bad-output'
        with self.assertRaisesRegex(ValueError,'outside all'):self.run_validation()
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
