"""Explicit current consumer roles; no historical-byte exemptions by hash/status."""
import json,tempfile,unittest
from pathlib import Path
from evcs_experiments import checkpoint_closure as m
from evcs_experiments.campaign_batch import file_reference

class CurrentProofRoleTests(unittest.TestCase):
 def write(self,p,v):p.write_text(json.dumps(v));return file_reference(p)
 def fixture(self,root):
  model=root/'current_model.bin';model.write_bytes(b'current mathematical data');direct=file_reference(model)
  old=root/'historical_NEXT_ACTION';old.write_text('then-current state');oldref=file_reference(old)
  def proof(name,header):return self.write(root/name,dict(header,historical_input=oldref))
  bank=proof('bank.json',m._BANK_PROOF)
  acceptance=proof('acceptance.json',{'status':'PASS_COMPONENT','formal_OOS':False})
  review=proof('review.json',{'status':'PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW','passed':True,'domain':'formal','formal_OOS':False,'index_reference':direct})
  carry=proof('carry.json',{'kind':'SAA_PRE_ASSEMBLY_FAILED_PARENT_CARRYOVER_V1'})
  req={k:direct for k in m._SAA_REQUEST_KEYS if k.endswith('_reference')}
  req.update(schema_version=1,domain='formal',spec_id='S05',model_id='IEEE33-v54-Q1',method_id='SAA',target_id='michael_2018',plan_id='risk_model.michael_2018.SAA.baseline',budget={},model_input_mode='assemble',max_coefficient_bytes=8589934592,output_root=str(root/'output'),planning_source_references=[direct],template_references=[direct]*160,model_checkpoint_reserve_seconds=60,model_reload_reserve_seconds=60,audit_reference=bank,bank_acceptance_reference=acceptance,template_association_review_reference=review,failure_carryover_reference=carry)
  ref=self.write(root/'worker_request.json',req);old.write_text('later historical state')
  return ref,req,model,bank,oldref
 def test_explicit_proof_role_preserves_hash_and_current_dependencies(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();ref,req,model,bank,old=self.fixture(root);m.verify_checkpoint_closure(ref)
   # The same proof under an ordinary current dependency is not cached as a leaf.
   other=self.write(root/'ordinary.json',{'request':ref,'ordinary_current_proof':bank})
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(other)
   model.write_bytes(b'changed current coefficients')
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)
 def test_modified_proof_bytes_or_current_ref_shapes_rejected(self):
  for mode in ('proof_bytes','direct_null','direct_extra','review_binding','extra_request_field'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();ref,req,model,bank,old=self.fixture(root)
    if mode=='proof_bytes':Path(bank['path']).write_text('{}')
    elif mode=='direct_null':req['helper_reference']=None
    elif mode=='direct_extra':req['helper_reference']=dict(req['helper_reference'],quote=True)
    elif mode=='review_binding':
     p=Path(req['template_association_review_reference']['path']);v=json.loads(p.read_text());v['index_reference']=bank;req['template_association_review_reference']=self.write(p,v)
    else:req['unexpected_role']=True
    ref=self.write(root/'worker_request.json',req)
    with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)
 def test_template_index_proofs_only_keep_ordered_rows_live(self):
  for mode in ('good','row_order','proof_scope','current_changed'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();_,req,model,bank,old=self.fixture(root);direct=file_reference(model)
    numeric=self.write(root/'numeric.json',dict(status='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT',passed=True,formal_OOS=False,scientific_acceptance=False,old=old))
    reader=self.write(root/'reader.json',dict(status='PASS_COMPONENT',old=old))
    ids=[f'scene{i}' for i in range(160)];hashes=['a'*64]*160
    rows=[dict(scenario_index=i,scenario_id=ids[i],scenario_sha256=hashes[i],template_sha256='b'*64,binding_reference=direct,original_template_reference=direct) for i in range(160)]
    index=dict(kind='ORIGINAL_RECOURSE_TEMPLATE_INDEX_V1',status='PENDING_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW',domain='formal',first_stage_sha256='c'*64,scenario_ids=ids,scenario_sha256s=hashes,rows=rows,original_request_reference=direct,original_snapshot_reference=direct,proof_references=dict(bank_audit=bank,numeric_audit=numeric,reader_acceptance=reader),formal_OOS=False,scientific_acceptance=False,old_q_reused=False)
    if mode=='row_order':index['rows'][0]['scenario_index']=1
    if mode=='proof_scope':index['proof_references'].pop('numeric_audit')
    if mode=='current_changed':model.write_text('modified current matrix')
    ref=self.write(root/'index.json',index)
    if mode=='good':m.verify_checkpoint_closure(ref)
    else:
     with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)
 def test_raw_request_audit_role_does_not_hide_other_inputs(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();_,req,model,bank,old=self.fixture(root)
   raw=dict(schema_version=1,domain='formal',problem_identity={},input_binding={'audit':bank,'current':file_reference(model)},exporter_sha256='a'*64,formal_OOS=False,independent_scientific_audit_required=True)
   ref=self.write(root/'raw_request.json',raw);m.verify_checkpoint_closure(ref)
   model.write_text('changed');
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)
