"""Release inventory rows retain relative semantics only inside their schema."""
import json,tempfile,unittest
from pathlib import Path
from evcs_experiments.formal_planning_worker import _closure
from evcs_experiments.campaign_batch import file_reference

class CheckpointReleaseManifestClosureTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
 def write(self,p,value):
  p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return file_reference(p)
 def release(self,name):
  root=self.root/name;member=root/'src/evcs_experiments/core.py';member.parent.mkdir(parents=True);member.write_text('# original model\n')
  ref=file_reference(member);row=dict(ref,path='src/evcs_experiments/core.py')
  manifest=dict(schema_version=1,release_name=name,created_at_utc='2026-09-09T15:00:00+00:00',git_head='a'*40,working_tree_snapshot=True,files=[row])
  return root,manifest,self.write(root/'RELEASE_MANIFEST.json',manifest)
 def test_checkpoint_old_and_new_release_closure_and_runtime_exclusions(self):
  a,_,ar=self.release('old');b,_,br=self.release('new')
  (b/'.venv').mkdir();(b/'.venv/runtime').write_text('installed environment')
  (b/'configs').mkdir();(b/'configs/powerai.toml').write_text('runtime paths')
  old=self.write(self.root/'old-request.json',{'release_reference':ar})
  new=self.write(self.root/'new-request.json',{'release_reference':br,'old_request_reference':old})
  cp=self.write(self.root/'checkpoint.json',{'request_reference':new,'original_source_reference':ar})
  _closure(cp)
 def test_relative_normal_artifact_ref_is_still_rejected(self):
  root,manifest,_=self.release('source')
  cp=self.write(self.root/'checkpoint.json',{'artifact':manifest['files'][0]})
  with self.assertRaisesRegex(ValueError,'Frozen reference changed'):_closure(cp)
 def test_named_ordinary_result_cannot_opt_into_release_row_exception(self):
  ref=self.write(self.root/'RELEASE_MANIFEST.json',{'artifact':{'path':'x','sha256':'a'*64,'size_bytes':1}})
  with self.assertRaisesRegex(ValueError,'Malformed release'):_closure(ref)
 def test_tampered_member_rejected(self):
  root,_,ref=self.release('source');(root/'src/evcs_experiments/core.py').write_text('# changed model')
  with self.assertRaisesRegex(ValueError,'Frozen reference changed'):_closure(ref)
 def test_parent_escape_absolute_duplicate_and_noncanonical_paths_rejected(self):
  for value in ('../outside','/absolute','src/../core.py','./src/evcs_experiments/core.py'):
   with self.subTest(path=value):
    root,m,ref=self.release('s'+str(len(list(self.root.iterdir()))));m['files'][0]['path']=value
    ref=self.write(root/'RELEASE_MANIFEST.json',m)
    with self.assertRaisesRegex(ValueError,'path'):_closure(ref)
  root,m,_=self.release('duplicate');m['files'].append(m['files'][0]);ref=self.write(root/'RELEASE_MANIFEST.json',m)
  with self.assertRaisesRegex(ValueError,'duplicate'):_closure(ref)
 def test_unlisted_and_missing_members_are_rejected(self):
  root,m,ref=self.release('source');(root/'unexpected.py').write_text('x')
  with self.assertRaisesRegex(ValueError,'inventory'):_closure(ref)
  (root/'unexpected.py').unlink();(root/'src/evcs_experiments/core.py').unlink()
  with self.assertRaises(FileNotFoundError):_closure(ref)
 def test_symlink_even_with_matching_target_bytes_rejected(self):
  root,m,_=self.release('source');target=root/'src/evcs_experiments/core.py';real=self.root/'actual.py';real.write_bytes(target.read_bytes());target.unlink();target.symlink_to(real)
  ref=self.write(root/'RELEASE_MANIFEST.json',m)
  with self.assertRaisesRegex(ValueError,'symlink/escape'):_closure(ref)
 def test_malformed_schema_and_bool_size_rejected(self):
  for field,value in [('schema_version',True),('release_name','other'),('working_tree_snapshot',False),('git_head','invalid')]:
   root,m,_=self.release('s'+str(len(list(self.root.iterdir()))));m[field]=value;ref=self.write(root/'RELEASE_MANIFEST.json',m)
   with self.subTest(field=field),self.assertRaises(ValueError):_closure(ref)
  root,m,_=self.release('bool-size');m['files'][0]['size_bytes']=True;ref=self.write(root/'RELEASE_MANIFEST.json',m)
  with self.assertRaisesRegex(ValueError,'digest/size'):_closure(ref)
 def test_nested_release_member_is_hashed_as_inventory_content(self):
  root,m,_=self.release('source');member=self.write(root/'provenance/old/RELEASE_MANIFEST.json',{'historical':'content only'})
  m['files'].append(dict(member,path='provenance/old/RELEASE_MANIFEST.json'))
  ref=self.write(root/'RELEASE_MANIFEST.json',m);_closure(ref)
  (root/'provenance/old/RELEASE_MANIFEST.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'Frozen reference changed'):_closure(ref)

if __name__=='__main__':unittest.main()
