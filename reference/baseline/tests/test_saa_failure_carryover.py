"""Narrow original-evidence settlement fixtures; no numerical solver."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import saa_failure_carryover as c
from evcs_experiments.failure_carryover import ENGINE_FILES

class SAAFailureCarryoverTests(unittest.TestCase):
 def write(self,path,value):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return c.reference(path)
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
  self.refs={};self.oldsrc={};self.newsrc={};self.manifests=[]
  for label in ('old','new'):
   root=self.root/label;names=set(ENGINE_FILES)|{'recourse_template_index.py','recourse_store.py','campaign_batch.py','saa_planning_worker.py'}
   if label=='new':names.add('saa_failure_carryover.py')
   rows=[]
   for name in sorted(names):
    p=root/'src/evcs_experiments'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# '+name)
    ref=c.reference(p);rows.append(dict(path=str(p.relative_to(root)),sha256=ref['sha256'],size_bytes=ref['size_bytes']))
   self.manifests.append(self.write(root/'RELEASE_MANIFEST.json',{'files':rows}))
  self.plan='risk_model.michael_2018.SAA.baseline';self.oldqueue=self.root/'queues/old';self.attempt=self.oldqueue/'michael.SAA__attempt_0001';self.attempt.mkdir(parents=True)
  self.newqueue=self.root/'queues/new';self.output=self.root/'plans/old';self.output.mkdir(parents=True)
  dummy=self.write(self.root/'inputs.json',{'frozen':True});template=self.write(self.root/'template.json',{})
  self.index=self.write(self.root/'index.json',{'axes':['one']})
  oldimpl=[c.reference(self.root/'old/src/evcs_experiments'/n) for n in ('recourse_template_index.py','recourse_store.py')]
  self.oldreview=self.write(self.root/'association-old.json',dict(status='PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW',passed=True,index_reference=self.index,implementation_sources=oldimpl,scenario_ids=['one'],proof_references={'audit':dummy}))
  old=dict(schema_version=1,domain='formal',spec_id='S05',model_id='IEEE33-v54-Q1',method_id='SAA',target_id='michael_2018',plan_id=self.plan,model_input_mode='assemble',budget={'cumulative_wall_seconds':7200},output_root=str(self.output),planning_source_references=[c.reference(p) for p in sorted((self.root/'old/src/evcs_experiments').glob('*.py'))],helper_reference=dummy,template_references=[template],template_index_reference=self.index,template_association_review_reference=self.oldreview)
  old.update({k:dummy for k in ('protocol_reference','config_reference','bank_acceptance_reference','summary_reference','audit_reference')});self.old=old
  self.refs['worker_request']=self.write(self.root/'old-item.json',old)
  q=dict(queue_root=str(self.oldqueue),items=[dict(plan_id=self.plan,plan_budget_seconds=7200,request_reference=self.refs['worker_request'])]);self.refs['queue_request']=self.write(self.root/'old-queue.json',q)
  jobdir=self.root/'jobs/failed';job=dict(job_id='failed',state='FAILED',complete=True,returncode=2,elapsed_seconds=2.113231)
  self.refs['job_result']=self.write(jobdir/'result.json',job);self.refs['job_manifest']=self.write(jobdir/'manifest.json',dict(job_id='failed',argv=['python','--request',self.refs['queue_request']['path']]))
  inv=dict(plan_id=self.plan,item_request_reference=self.refs['worker_request'],request_reference=self.refs['queue_request'],cumulative_plan_seconds=0,opening_carryover_seconds=0,failure_carryover_reference=None,previous_checkpoint=None,previous_result_reference=None,attempt_directory=str(self.attempt),parent_job=dict(job_id='failed',manifest_reference=self.refs['job_manifest']))
  self.refs['invocation']=self.write(self.attempt/'invocation.json',inv)
  wf=dict(status='FAILED_SAA_WORKER_REQUIRES_DIAGNOSIS',error_type='ValueError',error=c.ERROR,cumulative_accounted_seconds=1.5)
  self.refs['worker_failure']=self.write(self.attempt/'worker_failure.json',wf)
  self.refs['failure']=self.write(self.attempt/'failure.json',dict(error='Worker exited nonzero: 1',elapsed_seconds=1.8))
  run=dict(request_reference=self.refs['queue_request'],status='FAILED_VALIDITY_OR_WORKER',unsettled_plan_seconds=True,elapsed_seconds=2.,cumulative_plan_seconds={self.plan:1.8})
  self.refs['batch_result']=self.write(self.oldqueue/'run1/result.json',run)
  state=dict(request_reference=self.refs['queue_request'],attempt_receipts=[],halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW',active=dict(invocation_reference=self.refs['invocation'],process=None),runs=[self.refs['batch_result']])
  self.refs['batch_state']=self.write(self.oldqueue/'state.json',state)
  diag=dict(status='DIAGNOSED_PRE_ASSEMBLY_SOURCE_ASSOCIATION_FAILURE',passed_diagnosis=True,job_result_reference=self.refs['job_result'],worker_failure_reference=self.refs['worker_failure'],queue_state_reference=self.refs['batch_state'],assembly_completed=False,master_calls=0,fixed_q_calls=0)
  self.refs['independent_diagnosis']=self.write(self.root/'diagnosis.json',diag)
  sett=dict(kind='SAA_PRE_ASSEMBLY_FAILED_PARENT_ACCOUNTING_V1',status='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION',accounting_PASS=True,plan_id=self.plan,source_manifest_reference=self.manifests[0],opening_cumulative_seconds_decimal='0',terminal_parent_seconds_decimal='2.113231',cumulative_accounted_seconds_decimal='2.113231',remaining_seconds_decimal='7197.886769',cumulative_cap_seconds=7200,parent_charged_once=True,nested_seconds_added=False)
  for a,b in {'parent_result_reference':'job_result','parent_manifest_reference':'job_manifest','worker_failure_reference':'worker_failure','queue_failure_reference':'failure','request_reference':'worker_request','queue_request_reference':'queue_request','independent_diagnosis_reference':'independent_diagnosis'}.items():sett[a]=self.refs[b]
  self.refs['settlement']=self.write(self.root/'settlement.json',sett)
  newimpl=[c.reference(self.root/'new/src/evcs_experiments'/n) for n in ('recourse_template_index.py','recourse_store.py')]
  rebound=json.loads(Path(self.oldreview['path']).read_text());rebound['implementation_sources']+=newimpl
  rebound['source_rebinding']=dict(previous_review_reference=self.oldreview,independent_diagnosis_reference=self.refs['independent_diagnosis'],old_index_and_axes_preserved=True,old_numerical_evidence_unchanged=True,new_numeric_work_performed=False,byte_identity_pairs=[dict(previous=a,current=b,exact_bytes_equal=True) for a,b in zip(oldimpl,newimpl)])
  self.new=copy.deepcopy(old);self.new.update(release_reference=self.manifests[1],output_root=str(self.root/'plans/new'),planning_source_references=[c.reference(p) for p in sorted((self.root/'new/src/evcs_experiments').glob('*.py'))],template_association_review_reference=self.write(self.root/'rebound.json',rebound))
  self.queue=dict(queue_root=str(self.newqueue),source_references=self.new['planning_source_references'],job_wall_seconds=7200,items=[dict(item_id='saa',plan_id=self.plan,plan_budget_seconds=7200,argv=['python','-m','saa'],cwd=str(self.root/'new'),request_reference={'path':str(self.root/'new-item.json')})])
  oldsources,_=c._release(self.manifests[0]);newsources,_=c._release(self.manifests[1]);changed={n for n in set(oldsources)|set(newsources) if oldsources.get(n)!=newsources.get(n)}
  self.record=dict(schema_version=1,kind=c.KIND,plan_id=self.plan,accounted_seconds=2.113231,references=self.refs,old_release_reference=self.manifests[0],new_release_reference=self.manifests[1],successor_queue_root=str(self.newqueue),successor_output_root=self.new['output_root'],successor_queue_request_path=str(self.root/'new-queue.json'),source_changes={n:dict(old=oldsources.get(n),new=newsources.get(n)) for n in sorted(changed)})
  self.seal()
  self.p1=patch.object(c,'observation',return_value={'blocks_new_launch':False});self.p1.start();self.addCleanup(self.p1.stop)
  self.p2=patch.object(c,'identity_alive',return_value=False);self.p2.start();self.addCleanup(self.p2.stop)
 def seal(self):
  self.record['successor_request_identity_sha256']=c.successor_request_identity(self.new)
  self.record['successor_queue_identity_sha256']=c.successor_queue_identity(self.queue)
  self.refs['independent_review']=self.write(self.root/'review.json',dict(status='PASS_SAA_FAILURE_CARRYOVER_REVIEW',candidate_payload_sha256=c.carryover_review_identity(self.record)))
  self.ref=self.write(self.root/'carryover.json',self.record);self.new['failure_carryover_reference']=self.ref
  self.queue['items'][0].update(request_reference=self.write(self.root/'new-item.json',self.new),failure_carryover_reference=self.ref)
  self.write(self.root/'new-queue.json',self.queue)
 def verify(self,**kw):return c.verify_saa_failure_carryover(self.ref,self.new,self.newqueue,**kw)
 def test_claim_exact_cost_once_and_old_bytes_preserved(self):
  oldbytes=Path(self.refs['batch_state']['path']).read_bytes();result=c.claim_saa_failure_carryover(self.ref,self.new,self.newqueue)
  self.assertEqual(result['accounted_seconds'],2.113231);self.assertEqual(self.verify(),result)
  self.assertEqual(Path(self.refs['batch_state']['path']).read_bytes(),oldbytes)
  with self.assertRaises(FileExistsError):c.claim_saa_failure_carryover(self.ref,self.new,self.newqueue)
 def test_reset_cost_rejected_even_reapproved(self):
  self.record['accounted_seconds']=0;self.seal()
  with self.assertRaisesRegex(ValueError,'cost'):self.verify(require_claim=False)
 def test_wrong_plan_and_scenario_refs_rejected(self):
  for key,value in [('plan_id','other'),('template_references',[])]:
   with self.subTest(key=key):
    old=self.new[key];self.new[key]=value;self.seal()
    with self.assertRaises(ValueError):self.verify(require_claim=False)
    self.new[key]=old
 def test_model_directory_or_receipt_blocks_carryover(self):
  (self.output/'model').mkdir()
  with self.assertRaisesRegex(ValueError,'Old model'):self.verify(require_claim=False)
  (self.output/'model').rmdir();(self.attempt/'receipt.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'Old model'):self.verify(require_claim=False)
 def test_changed_mathematical_source_and_rehashed_manifest_rejected(self):
  p=self.root/'new/src/evcs_experiments/recourse.py';p.write_text('# changed numerical model')
  m=json.loads(Path(self.manifests[1]['path']).read_text())
  for row in m['files']:
   if row['path'].endswith('/recourse.py'):row.update(sha256=c.reference(p)['sha256'],size_bytes=p.stat().st_size)
  self.record['new_release_reference']=self.write(Path(self.manifests[1]['path']),m);self.new['release_reference']=self.record['new_release_reference'];self.seal()
  with self.assertRaisesRegex(ValueError,'mathematical'):self.verify(require_claim=False)
 def test_old_claim_other_successor_rejected(self):
  p=c._claim_path(self.record);p.parent.mkdir();p.write_text('{}')
  with self.assertRaisesRegex(ValueError,'claimed'):self.verify()
 def test_mutated_successor_queue_argv_or_source_or_request_path_rejected(self):
  for mutate in (lambda q:q['items'][0].update(argv=['different']),lambda q:q.update(source_references=[]),lambda q:q['items'][0].update(request_reference={'path':'/other'})):
   q=copy.deepcopy(self.queue);mutate(q);self.write(self.root/'new-queue.json',q)
   with self.assertRaisesRegex(ValueError,'Successor queue'):self.verify(require_claim=False)
 def test_unbound_approval_and_active_predecessor_rejected(self):
  self.record['successor_output_root']='other';self.ref=self.write(self.root/'carryover.json',self.record)
  with self.assertRaisesRegex(ValueError,'review'):self.verify(require_claim=False)
 def test_reapproved_settlement_prior_nested_or_plan_change_rejected(self):
  original=c._read(self.refs['settlement'])
  for key,value in [('opening_cumulative_seconds_decimal','1'),('nested_seconds_added',True),('plan_id','other')]:
   with self.subTest(key=key):
    bad=dict(original);bad[key]=value
    self.refs['settlement']=self.write(self.root/'settlement.json',bad);self.seal()
    with self.assertRaisesRegex(ValueError,'cost'):self.verify(require_claim=False)
 def test_reapproved_association_support_proof_or_byte_pair_change_rejected(self):
  original=c._read(self.new['template_association_review_reference'])
  mutations=[lambda x:x.update(scenario_ids=['different']),lambda x:x.update(proof_references={}),
             lambda x:x['source_rebinding']['byte_identity_pairs'][0].update(exact_bytes_equal=False)]
  for mutate in mutations:
   bad=copy.deepcopy(original);mutate(bad)
   self.new['template_association_review_reference']=self.write(self.root/'rebound.json',bad);self.seal()
   with self.assertRaisesRegex(ValueError,'Association|rebound'):self.verify(require_claim=False)
 def test_inconsistent_release_reference_rejected_even_reapproved(self):
  self.new['release_reference']=self.manifests[0];self.seal()
  with self.assertRaisesRegex(ValueError,'release reference'):self.verify(require_claim=False)

 def test_alive_predecessor_rejected(self):
  with patch.object(c,'observation',return_value={'blocks_new_launch':True}):
   with self.assertRaisesRegex(ValueError,'alive'):self.verify(require_claim=False)

if __name__=='__main__':unittest.main()
