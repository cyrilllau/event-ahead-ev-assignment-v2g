"""Exact historical inventory contracts; ordinary relative refs remain errors."""
import json,tempfile,unittest
from pathlib import Path
from evcs_experiments.checkpoint_closure import verify_checkpoint_closure,_AUDIT_REPAIR_FILES,_AUDIT_REPAIR_SCOPE
from evcs_experiments.campaign_batch import file_reference
from tests import test_checkpoint_release_manifest_closure as baseline

class OperationInventoryTests(unittest.TestCase):
 def fixture(self,root):
  rows=[]
  for name in sorted(_AUDIT_REPAIR_FILES):
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
   path.write_text(json.dumps({'paths':{'project_root':str(root)}}) if name.endswith('.json') else 'exact source')
   rows.append(dict(file_reference(path),path=name))
  return {'scope':_AUDIT_REPAIR_SCOPE,'files':rows}
 def save(self,root,value):
  p=root/'source_manifest.json';p.write_text(json.dumps(value));return file_reference(p)
 def test_exact_actual_schema_and_changed_member(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();v=self.fixture(root);ref=self.save(root,v);verify_checkpoint_closure(ref)
   (root/'scripts/audits/audit_formal_bank.py').write_text('changed')
   with self.assertRaises(ValueError):verify_checkpoint_closure(ref)
 def test_missing_duplicate_escape_extra_field_and_bool_size(self):
  for mode in ('missing','duplicate','escape','extra','bool'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();v=self.fixture(root)
    if mode=='missing':v['files'].pop()
    if mode=='duplicate':v['files'][-1]=v['files'][0]
    if mode=='escape':v['files'][0]['path']='../outside.py'
    if mode=='extra':v['root']='invented'
    if mode=='bool':v['files'][0]['size_bytes']=True
    with self.assertRaises(ValueError):verify_checkpoint_closure(self.save(root,v))
 def test_resealed_wrong_declared_root_and_symlink(self):
  for symlink in (False,True):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve()/'op';root.mkdir();v=self.fixture(root)
    if symlink:
     p=root/'launch_retry.py';outside=Path(d)/'outside.py';outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside)
    else:
     p=root/'audit_test_config.json';p.write_text(json.dumps({'paths':{'project_root':str(root.parent)}}))
     v['files']=[dict(file_reference(p),path=p.name) if r['path']==p.name else r for r in v['files']]
    with self.assertRaises(ValueError):verify_checkpoint_closure(self.save(root,v))
 def test_unknown_scope_does_not_gain_relative_path_semantics(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();v=self.fixture(root);v['scope']='arbitrary inventory'
   with self.assertRaises(ValueError):verify_checkpoint_closure(self.save(root,v))

class InheritedReleaseTests(baseline.CheckpointReleaseManifestClosureTests):
 def test_explicit_inherited_release_and_preflight_both_verified(self):
  _,_,old=self.release('old');root,m,_=self.release('new')
  preflight=self.write(self.root/'preflight.json',{'original':old})
  m.update(inherited_manifest=old,preflight_reference=preflight)
  ref=self.write(root/'RELEASE_MANIFEST.json',m);verify_checkpoint_closure(ref)
  Path(preflight['path']).write_text('{}')
  with self.assertRaises(ValueError):verify_checkpoint_closure(ref)
 def test_partial_extension_unknown_field_and_relative_inherited_ref_rejected(self):
  for mode in ('partial','extra','relative'):
   root,m,_=self.release('new'+mode)
   r=self.write(self.root/(mode+'.json'),{})
   m['inherited_manifest']=r
   if mode!='partial':m['preflight_reference']=r
   if mode=='extra':m['ignored_context']=True
   if mode=='relative':m['inherited_manifest']=dict(r,path='old.json')
   with self.assertRaises(ValueError):verify_checkpoint_closure(self.write(root/'RELEASE_MANIFEST.json',m))
