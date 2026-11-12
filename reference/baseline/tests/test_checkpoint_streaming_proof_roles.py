"""Streaming and completion consumers retain direct live dependencies."""
import copy,json,tempfile,unittest
from pathlib import Path
from evcs_experiments import checkpoint_closure as m
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.random_streams import canonical_hash

class StreamingProofTests(unittest.TestCase):
 def write(self,p,v):p.write_text(json.dumps(v));return file_reference(p)
 def fixture(self,root):
  current=root/'matrix.bin';current.write_bytes(b'original current coefficient bytes');direct=file_reference(current)
  old=root/'historical.txt';old.write_text('before');oldref=file_reference(old);old.write_text('after')
  bank=self.write(root/'bank.json',dict(m._BANK_PROOF,history=oldref))
  review=self.write(root/'review.json',dict(status='PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW',passed=True,domain='formal',formal_OOS=False,index_reference=direct,history=oldref))
  binding={k:direct for k in m._STREAM_BINDING_KEYS};binding.update(method_id='SAA',algorithm_id=m._STREAM_ALGORITHM,formal_OOS=False,audit=bank,template_provider=dict(template_index_reference=direct,template_association_review_reference=review))
  model={k:None for k in m._STREAMING_KEYS};model.update(schema_version=1,algorithm_id=m._STREAM_ALGORITHM,risk_model='SAA',domain='formal',input_binding=binding,status='ASSEMBLED_PENDING_RESOURCE_AND_INDEPENDENT_NUMERICAL_AUDIT',solver_calls=0,scientific_spec_pass=False,formal_OOS=False,solver_peak_rss_measured=False,shared_first_stage_unchanged=True,numeric_artifacts=[direct],scenarios=[{'artifacts':[direct]}])
  binding.update(summary=direct,formal_protocol_lock=direct,bank_source_manifest=direct,
   task_source=dict(path=direct['path'],sha256=direct['sha256'],bytes=direct['size_bytes']),
   planning_source={'source.py':'a'*64},target=dict(target_event='michael_2018',H=1,planning_count=160,calibration_count=80,
   planning_index=direct,calibration_index=direct,moments=direct,bootstrap=direct))
  numeric=[]
  for name in ('data','indices','indptr','row_lower','row_upper','lower_bounds','upper_bounds','c','integrality'):
   p=root/(name+'.npy');p.write_bytes(b'original');numeric.append(file_reference(p))
  ids=[f'scene{i}' for i in range(160)];hashes=['a'*64]*160;rows=[]
  for i in range(160):
   folder=root/'shards'/f'{i:03d}';folder.mkdir(parents=True);refs=[]
   for name in ('arrays.npz','matrix.npz','recourse_columns.npy','zero_proof.json'):
    p=folder/name;p.write_bytes(b'{}' if name.endswith('.json') else b'data');refs.append(file_reference(p))
   rows.append(dict(index=i,scenario_id=ids[i],scenario_sha256=hashes[i],template_sha256='b'*64,empirical_probability=1/160,
    original_recourse_columns=1,column_slice=[i,i+1],row_slice=[i,i+1],nnz=1,mapping_file=f'shards/{i:03d}/recourse_columns.npy',mapping_reference=refs[2],artifacts=refs))
  model.update(numeric_artifacts=numeric,scenario_ids=ids,scenario_sha256s=hashes,scenarios=rows)
  model['manifest_sha256']=canonical_hash({k:v for k,v in model.items() if k!='manifest_sha256'})
  ref=self.write(root/'manifest.json',model)
  cp={k:0 for k in m._COMPLETION_KEYS};cp.update(kind='SAA_COMPLETE_MODEL_CHECKPOINT_V1',request_reference=direct,model_manifest_reference=ref,model_input_mode='assemble',planning_source_references=[direct],input_binding=binding,numerical_audit_complete=False,scientific_acceptance=False,formal_OOS=False)
  return model,cp,current,review
 def test_manifest_and_completion_byte_bound_proofs_without_history_replay(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);model,cp,current,review=self.fixture(root)
   m.verify_checkpoint_closure(self.write(root/'completion.json',cp))
   # The role must never poison the ordinary dependency cache.
   direct=self.write(root/'ordinary.json',dict(completion=file_reference(root/'completion.json'),ordinary_review=review))
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(direct)
 def test_certificate_current_array_and_schema_tampering(self):
  for mode in ('proof_bytes','coefficient','unknown_field','binding_scope','provider_null','wrong_index','bad_digest','wrong_status','missing_array_bytes','numeric_null','numeric_extra','shards_null','sources_null','mapping_extra'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d);model,cp,current,review=self.fixture(root)
    if mode=='numeric_null':model['numeric_artifacts']=None
    elif mode=='numeric_extra':model['numeric_artifacts'][0]['extra']=True
    elif mode=='shards_null':model['scenarios']=None
    elif mode=='sources_null':model['input_binding']['planning_source']=None
    elif mode=='mapping_extra':model['scenarios'][0]['mapping_reference']['extra']=True
    elif mode=='proof_bytes':Path(review['path']).write_text('{}')
    elif mode=='coefficient':current.write_bytes(b'changed')
    elif mode=='unknown_field':model['unreviewed']=True
    elif mode=='binding_scope':model['input_binding']['new_proof']=review
    elif mode=='provider_null':model['input_binding']['template_provider']=None
    elif mode=='wrong_index':model['input_binding']['template_provider']['template_index_reference']=review
    elif mode=='wrong_status':model['status']='PASS'
    elif mode=='missing_array_bytes':current.unlink()
    if mode!='bad_digest':model['manifest_sha256']=canonical_hash({k:v for k,v in model.items() if k!='manifest_sha256'})
    else:model['manifest_sha256']='0'*64
    ref=self.write(root/'manifest.json',model)
    with self.assertRaises((ValueError,FileNotFoundError)):m.verify_checkpoint_closure(ref)
 def test_completion_mismatch_and_missing_source_rejected(self):
  for mode in ('binding','source','manifest_ref','extra'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d);model,cp,current,review=self.fixture(root);cp=copy.deepcopy(cp)
    if mode=='binding':cp['input_binding']['rho']=999
    elif mode=='source':cp['planning_source_references']=[]
    elif mode=='manifest_ref':cp['model_manifest_reference']=None
    else:cp['extra']=True
    with self.assertRaises(ValueError):m.verify_checkpoint_closure(self.write(root/'completion.json',cp))
 def test_development_preassembled_completion_keeps_ordinary_traversal(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);current=root/'data';current.write_text('current');ref=file_reference(current)
   manifest=self.write(root/'manifest.json',{'domain':'development','current':ref})
   cp=self.write(root/'completion.json',{'kind':'SAA_COMPLETE_MODEL_CHECKPOINT_V1','model_manifest_reference':manifest,'model_input_mode':'preassembled'})
   m.verify_checkpoint_closure(cp)
   current.write_text('tampered')
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(cp)
