"""Independent tests of the actual r20 relative preflight inventory contract.

Fixture metadata retains the historical producer's exact keys and all ten
association overlay paths. Small immutable payloads replace source bytes; no
solver, source deployment, or scientific bank acceptance occurs here.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.checkpoint_closure import verify_checkpoint_closure
from evcs_experiments.config import get_paths


class CheckpointPreflightInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True)+'\n')
        return file_reference(path)

    def payload(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
        return file_reference(path)

    def relative(self,path,root):
        return dict(file_reference(path),path=path.relative_to(root).as_posix())

    def release(self,root,files,**extra):
        value=dict(schema_version=1,release_name=root.name,
            created_at_utc='2026-09-08T13:18:00+00:00',git_head='a'*40,
            working_tree_snapshot=True,files=files,**extra)
        return value,self.write(root/'RELEASE_MANIFEST.json',value)

    def fixture(self):
        project=get_paths().project_root
        card=json.loads((project/'provenance/operations/s05-template-association-20260908T091400Z/development_preparation.json').read_text())
        preflight=json.loads((project/'provenance/development/s05-unified-r20-20260908T131800Z/source_preflight.json').read_text())
        base=self.root/'releases/base';new=self.root/'releases/new'
        op=self.root/'operations/new';prepared_op=self.root/'operations/association'
        base_rows=[]
        core='src/evcs_experiments/unchanged_core.py'
        self.payload(base/core,'# unchanged kernel\n');base_rows.append(self.relative(base/core,base))
        for row in card['files']:
            inherited=row['inherited_reference']
            if inherited is not None:
                self.payload(base/inherited['path'],'# original '+inherited['path']+'\n')
                row['inherited_reference']=self.relative(base/inherited['path'],base)
                base_rows.append(row['inherited_reference'])
        _,base_ref=self.release(base,base_rows)
        card['base_release_manifest_sha256']=base_ref['sha256']
        for row in card['files']:
            rel=row['relative_path'];p=prepared_op/'prepared_source'/rel
            self.payload(p,'# reviewed overlay '+rel+'\n')
            r=file_reference(p);row['sha256']=r['sha256'];row['size_bytes']=r['size_bytes']
        card['test_logs']=[self.payload(prepared_op/Path(r['path']).name,'recorded original test output\n') for r in card['test_logs']]
        prep_ref=self.write(prepared_op/'development_preparation.json',card)
        merged={r['path']:dict(r) for r in base_rows}
        for r in base_rows:self.payload(new/r['path'],(base/r['path']).read_text())
        for r in card['files']:
            rel=r['relative_path'];self.payload(new/rel,(prepared_op/'prepared_source'/rel).read_text())
            merged[rel]=self.relative(new/rel,new)
        preparation_refs=[prep_ref];self.generic_log=None
        for part,stage in (('s05-generic-pipeline-20260908T103500Z','pipeline'),
                           ('s05-method-factories-20260908T115700Z','method')):
            body=json.loads((project/'provenance/development'/part/'development_preparation.json').read_text())
            stage_op=self.root/'operations'/stage
            refs=[]
            relative_names=['src/evcs_experiments/formal_planning.py']
            if stage=='method':relative_names.append('src/evcs_experiments/formal_bank.py')
            for relative in relative_names:
                r=self.payload(stage_op/'prepared_source'/relative,'# '+stage+' '+relative+'\n')
                refs.append(r);self.payload(new/relative,Path(r['path']).read_text())
                merged[relative]=self.relative(new/relative,new)
            body['source_references']=refs
            body['test_logs']={name:self.payload(stage_op/name,'original '+stage+' log\n')
                               for name in body['test_logs']}
            if stage=='pipeline':self.generic_log=next(iter(body['test_logs'].values()))
            if 'inherited_preparation' in body:body['inherited_preparation']=preparation_refs[-1]
            preparation_refs.append(self.write(stage_op/'development_preparation.json',body))
        patches=[]
        for relative in ('src/evcs_experiments/formal_bank.py','tests/test_bank_worker_deadline.py'):
            r=self.payload(op/'patches'/relative,'# final patch '+relative+'\n')
            patches.append(r);self.payload(new/relative,Path(r['path']).read_text())
            merged[relative]=self.relative(new/relative,new)
        preflight.update(base_manifest=base_ref,preparations=preparation_refs,patches=patches,
            files=[merged[rel] for rel in sorted(merged) if rel!=core],
            inherited_unchanged=1,source_file_count=len(merged),release_root=str(new))
        preflight_ref=self.write(op/'source_preflight.json',preflight)
        manifest,new_ref=self.release(new,list(merged.values()),inherited_manifest=base_ref,preflight_reference=preflight_ref)
        self.base,self.new,self.op,self.prepared_op=base,new,op,prepared_op
        self.card,self.preflight,self.manifest=card,preflight,manifest
        self.prep_ref,self.new_ref=prep_ref,new_ref
        return self.reseal()

    def reseal(self):
        self.prep_ref=self.write(self.prepared_op/'development_preparation.json',self.card)
        self.preflight['preparations'][0]=self.prep_ref
        pf=self.write(self.op/'source_preflight.json',self.preflight)
        self.manifest['preflight_reference']=pf
        self.new_ref=self.write(self.new/'RELEASE_MANIFEST.json',self.manifest)
        summary=self.write(self.root/'chain/summary.json',{'source_manifest':self.new_ref})
        bank=self.write(self.root/'chain/bank_acceptance.json',{'summary_reference':summary})
        worker=self.write(self.root/'chain/worker_request.json',{'bank_acceptance_reference':bank})
        return self.write(self.root/'chain/checkpoint.json',{'request_reference':worker})

    def test_legitimate_nested_immutable_inventory_is_verified_without_mutation(self):
        ref=self.fixture()
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        verify_checkpoint_closure(ref)
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        # Historical inherited source is an actual dependency, not a waived leaf.
        inherited=self.card['files'][0]['inherited_reference']
        (self.base/inherited['path']).write_text('# corrupted original source\n')
        with self.assertRaises(ValueError):verify_checkpoint_closure(ref)

    def test_preflight_unknown_schema_missing_duplicate_escape_and_wrong_root_rejected(self):
        for mode in ('unknown_status','unknown_field','missing','duplicate','escape','absolute','wrong_root','count'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode=='unknown_status':self.preflight['status']='PASS_ARBITRARY_SOURCE'
                if mode=='unknown_field':self.preflight['unchecked_relative_inventory']=[]
                if mode=='missing':self.preflight['files'].pop()
                if mode=='duplicate':self.preflight['files'].append(deepcopy(self.preflight['files'][0]))
                if mode=='escape':self.preflight['files'][0]['path']='../outside.py'
                if mode=='absolute':self.preflight['files'][0]['path']=str(self.new/self.preflight['files'][0]['path'])
                if mode=='wrong_root':self.preflight['release_root']=str(self.base)
                if mode=='count':self.preflight['source_file_count']+=1
                with self.assertRaises(ValueError):verify_checkpoint_closure(self.reseal())

    def test_preparation_unknown_missing_duplicate_and_mismatched_inherited_row_rejected(self):
        for mode in ('unknown_status','unknown_field','missing','duplicate','escape','wrong_inherited','wrong_base'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode=='unknown_status':self.card['status']='PASS_ARBITRARY_PREPARATION'
                if mode=='unknown_field':self.card['free_source_root']=str(self.prepared_op)
                if mode=='missing':self.card['files'].pop()
                if mode=='duplicate':self.card['files'].append(deepcopy(self.card['files'][0]))
                if mode=='escape':self.card['files'][0]['relative_path']='../outside.py'
                if mode=='wrong_inherited':self.card['files'][0]['inherited_reference']['sha256']='b'*64
                if mode=='wrong_base':self.card['base_release_manifest_sha256']='c'*64
                with self.assertRaises(ValueError):verify_checkpoint_closure(self.reseal())

    def test_wrong_prepared_bytes_and_symlink_with_equal_bytes_rejected(self):
        for mode in ('hash','symlink','directory_symlink'):
            with self.subTest(mode=mode):
                ref=self.fixture();rel=self.card['files'][0]['relative_path']
                p=self.prepared_op/'prepared_source'/rel
                if mode=='hash':p.write_text('# changed prepared bytes\n')
                elif mode=='symlink':
                    outside=self.root/'outside.py';outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside)
                else:
                    parent=p.parent;outside=self.root/'external_sources';parent.rename(outside);parent.symlink_to(outside,target_is_directory=True)
                with self.assertRaises(ValueError):verify_checkpoint_closure(ref)
                # Remove the symlink before rebuilding another fixture in this root.
                if mode=='symlink':p.unlink()
                if mode=='directory_symlink':parent.unlink();outside.rename(parent)

    def test_preflight_and_nested_absolute_dependencies_keep_generic_hash_checks(self):
        ref=self.fixture();verify_checkpoint_closure(ref)
        log=Path(self.generic_log['path']);log.write_text('tampered generic dependency log')
        with self.assertRaises(ValueError):verify_checkpoint_closure(ref)
        self.fixture()
        external=self.payload(self.root/'external_dependency.txt','immutable dependency')
        # Existing preflights without the typed inventory retain generic DAG rules.
        generic=self.write(self.op/'generic_preflight.json',{'original':external})
        self.manifest['preflight_reference']=generic
        manifest_ref=self.write(self.new/'RELEASE_MANIFEST.json',self.manifest)
        verify_checkpoint_closure(manifest_ref)
        Path(external['path']).write_text('changed')
        with self.assertRaises(ValueError):verify_checkpoint_closure(manifest_ref)

    def test_historical_log_role_never_waives_a_direct_dependency(self):
        self.fixture()
        historical=deepcopy(self.card['test_logs'][0])
        Path(historical['path']).write_text('later local log bytes')
        verify_checkpoint_closure(self.new_ref)
        both=self.write(self.root/'direct_historical_log.json',
                        {'release_reference':self.new_ref,'actual_dependency':historical})
        with self.assertRaises(ValueError):verify_checkpoint_closure(both)

    def test_generic_source_and_patch_inventory_roots_and_duplicates_are_strict(self):
        for mode in ('generic_wrong_root','generic_duplicate','generic_source_hash',
                     'patch_wrong_root','patch_duplicate','patch_missing','preparation_duplicate'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode.startswith('generic_'):
                    ref=self.preflight['preparations'][2]
                    body=json.loads(Path(ref['path']).read_text())
                    item=body['source_references'][0]
                    if mode=='generic_wrong_root':
                        copied=self.payload(self.root/'outside_source.py',Path(item['path']).read_text())
                        body['source_references'][0]=copied
                    elif mode=='generic_duplicate':body['source_references'].append(deepcopy(item))
                    else:Path(item['path']).write_text('# corrupted superseded method source')
                    self.preflight['preparations'][2]=self.write(Path(ref['path']),body)
                elif mode=='patch_wrong_root':
                    item=self.preflight['patches'][0]
                    copied=self.payload(self.root/'outside_patch.py',Path(item['path']).read_text())
                    self.preflight['patches'][0]=copied
                elif mode=='patch_duplicate':self.preflight['patches'][1]=deepcopy(self.preflight['patches'][0])
                elif mode=='patch_missing':self.preflight['patches'].pop()
                else:self.preflight['preparations'][2]=deepcopy(self.preflight['preparations'][1])
                with self.assertRaises(ValueError):verify_checkpoint_closure(self.reseal())

    def test_relative_rows_do_not_gain_semantics_outside_typed_preflight(self):
        self.fixture()
        with self.assertRaises(ValueError):verify_checkpoint_closure(self.prep_ref)
        ordinary=self.write(self.root/'ordinary.json',{'row':self.preflight['files'][0]})
        with self.assertRaises(ValueError):verify_checkpoint_closure(ordinary)


if __name__=='__main__':unittest.main()
