"""Historical verification index scope cannot suppress direct model evidence."""
import json,tempfile,unittest
from pathlib import Path
from evcs_experiments import checkpoint_closure as m
from evcs_experiments.campaign_batch import file_reference

class AssociationHistoryTests(unittest.TestCase):
 def write(self,p,v):p.write_text(json.dumps(v));return file_reference(p)
 def fixture(self,root):
  current=root/'current_model.bin';current.write_bytes(b'current numerical coefficient')
  old=root/'old_NEXT_ACTION.md';old.write_text('old mutable state')
  entries=[file_reference(old)]+[dict(path=str(root/f'historical_{i}.txt'),sha256='a'*64,size_bytes=1) for i in range(3203)]
  index=self.write(root/'checked_references.json',entries)
  proof=file_reference(current)
  cert=dict(status=m._ASSOCIATION_CHECK_STATUS,passed=True,index_reference=proof,proof_references={k:proof for k in ('bank_audit','numeric_audit','reader_acceptance')},acceptance_reference=proof,
   prior_review_reference=proof,source_manifest=proof,rows_reference=proof,checked_references=index,verified_template_count=160,
   verified_template_artifact_count=1280,hashed_unique_files=3204,hashed_bytes_including_repeats=26455406658,elapsed_seconds=25.8,
   solver_calls=0,recourse_reassembly_calls=0,full_numeric_audit_repeated=False,old_Q_imported=False,
   production_envelope_audit_accepted=False,formal_OOS=False,formal_plan_budget_changed=False)
  ref=self.write(root/'association_checks.json',cert)
  return ref,index,current,old,cert
 def test_historical_payload_not_current_dependency_and_direct_current_stays_checked(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
   old.write_text('subsequent recorded state');m.verify_checkpoint_closure(ref)
   old_ref=json.loads(Path(index['path']).read_text())[0]
   direct_old=self.write(root/'direct_old.json',{'certificate':ref,'current_direct_reference':old_ref})
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(direct_old)
   direct=self.write(root/'request.json',{'certificate':ref,'current':file_reference(current)})
   current.write_bytes(b'changed numerical coefficient')
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(direct)
 def test_certificate_and_index_bytes_each_immutable(self):
  for member in ('certificate','index'):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();ref,index,*_=self.fixture(root)
    Path((ref if member=='certificate' else index)['path']).write_text('{}')
    with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)
 def test_index_schema_scope_location_and_direct_reference_rejected(self):
  for mode in ('duplicate','foreign','scope','direct'):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
    if mode=='duplicate':
     rows=json.loads(Path(index['path']).read_text());rows[-1]=rows[0];cert['checked_references']=self.write(Path(index['path']),rows)
    if mode=='foreign':cert['checked_references']=self.write(root/'other_index.json',json.loads(Path(index['path']).read_text()))
    if mode=='scope':cert['unexpected']=True
    ref=self.write(root/'association_checks.json',cert)
    if mode=='direct':ref=self.write(root/'request.json',{'ordinary_index':index})
    with self.assertRaises((ValueError,FileNotFoundError)):m.verify_checkpoint_closure(ref)

class AssociationProofShapeTests(AssociationHistoryTests):
 def test_missing_malformed_proof_and_nonfinite_elapsed_rejected(self):
  for mode in ('missing','none','extra','float_bytes','bool_elapsed','infinite_elapsed'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
    if mode=='missing':cert['proof_references'].pop('numeric_audit')
    elif mode=='none':cert['proof_references']['reader_acceptance']=None
    elif mode=='extra':cert['proof_references']['extra']=file_reference(current)
    elif mode=='float_bytes':cert['hashed_bytes_including_repeats']=26.0
    elif mode=='bool_elapsed':cert['elapsed_seconds']=False
    else:cert['elapsed_seconds']=float('inf')
    ref=self.write(root/'association_checks.json',cert)
    with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)
