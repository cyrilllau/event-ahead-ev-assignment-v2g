"""Real tiny native/Q joint audit through receipt settlement and final sealing."""
import json
from pathlib import Path
import unittest
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments import saa_candidate_audit
from evcs_experiments.saa_record_bridge import build_saa_record_bridge,validate_saa_record_bridge
from evcs_experiments.formal_record_settlement import settle_plan_record,finalize_settled_record
import test_saa_candidate_audit as numerical

class SAASettlementTests(unittest.TestCase):
 def setUp(self):
  self.f=numerical.SAACandidateAuditTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.root=self.f.root
  self.budget=dict(cumulative_wall_seconds=100,single_master_seconds=10,final_certificate_reserve_seconds=20,shutdown_reserve_seconds=1)
 def write(self,name,value):
  p=self.root/name;p.write_text(json.dumps(value));return file_reference(p)
 def fixture(self,timelimit=False):
  if timelimit:
   n=self.root/'master/native_result.json';raw=json.loads(n.read_text());raw['status']=1;n.write_text(json.dumps(raw))
   p=self.root/'master/result.json';r=json.loads(p.read_text());r.update(raw_native_result=file_reference(n),solver_status=1,status='TIME_LIMIT',normal_convergence=False);p.write_text(json.dumps(r))
   self.f.args['master_result_reference']=file_reference(p)
  joint=saa_candidate_audit.audit_saa_candidate(**self.f.args);jr=self.write('joint.json',joint)
  request=dict(domain='development',method_id='SAA',plan_id='p',target_id='tiny',budget=self.budget,
   planning_source_references=self.f.sources,model_manifest_reference=joint['master_audit']['model_manifest'])
  req=self.write('worker_request.json',request)
  cr=dict(plan_id='p',driver_status='CONVERGED' if joint['normal_gap_met'] else 'TIME_LIMIT',raw_engine_status=joint['native_status'],
   candidate=dict(plan=joint['master_audit']['plan'],plan_sha256=joint['plan_sha256'],
    lower_bound_usd=joint['lower_bound_usd'],upper_bound_usd=joint['upper_bound_usd'],gap_relative=joint['gap_relative'],
    native_master_status=joint['native_status'],native_master_solver_status=joint['native_solver_status']))
  candidate=self.write('candidate.json',cr)
  audit=build_saa_record_bridge(req,candidate,jr,output_path=self.root/'bridge.json')
  queue=self.write('queue.json',dict(domain='development',items=[dict(item_id='i',plan_id='p',request_reference=req,plan_budget_seconds=100)]))
  inv=self.write('invocation.json',dict(domain='development',item_id='i',plan_id='p',request_reference=queue,item_request_reference=req,
   cumulative_plan_seconds=3,plan_remaining_seconds=97,plan_budget_seconds=100,result_path=str(self.root/'result.json')))
  result=self.write('result.json',dict(status='COMPLETE',invocation_sha256=inv['sha256'],numerical_audit_reference=audit,
   artifacts=[candidate,audit],attempt_accounted_seconds=5,queue_prior_attempt_seconds=3,cumulative_accounted_seconds=8,raw_engine_status=joint['native_status']))
  receipt=self.write('receipt.json',dict(item_id='i',plan_id='p',request_reference=queue,invocation_reference=inv,result_reference=result,
   elapsed_seconds=6,status='COMPLETE',returncode=0))
  self.bridge=json.loads(Path(audit['path']).read_text());self.request=request;self.candidate=cr;self.joint=joint
  return dict(receipt_reference=receipt,numerical_audit_reference=audit,prior_cumulative_seconds=3,planning_budget=self.budget)
 def test_real_joint_receipt_and_final_seal(self):
  args=self.fixture();answer=settle_plan_record(**args);record=answer['solver_record']
  self.assertEqual(record['status'],'CONVERGED');self.assertEqual(record['native_master_status'],'OPTIMAL')
  self.assertEqual(record['risk_certificate_kind'],'fixed_empirical');self.assertIsNone(record['original_rmp'])
  self.assertEqual(record['cumulative_wall_seconds'],9)
  sealed=finalize_settled_record(answer,actual_attempt_seconds_at_cutoff=7,final_seal_reserve_seconds=1,output_directory=self.root/'sealed')
  self.assertEqual(sealed['solver_record']['cumulative_wall_seconds'],11)
  self.assertFalse(sealed['solver_record']['scientific_spec_pass']);self.assertTrue(sealed['solver_record']['requires_seal_confirmation'])
 def test_native_timelimit_is_not_relabelled_by_small_gap(self):
  record=settle_plan_record(**self.fixture(timelimit=True))['solver_record']
  self.assertEqual(record['status'],'CONVERGED');self.assertEqual(record['raw_engine_status'],'TIME_LIMIT')
  self.assertEqual(record['native_master_solver_status'],1)
 def test_deep_original_primal_tamper_rejected(self):
  args=self.fixture();master=json.loads(Path(self.joint['master_audit']['result_reference']['path']).read_text())
  Path(master['primal']['path']).write_bytes(b'changed actual original primal')
  with self.assertRaises(ValueError):settle_plan_record(**args)
 def test_resealed_empirical_source_status_and_constant_tamper_rejected(self):
  import copy
  from evcs_experiments.random_streams import canonical_hash
  self.fixture()
  for kind in ('law','source','native','constant'):
   joint=copy.deepcopy(self.joint)
   if kind=='law':joint['fixed_q_audit']['empirical_probabilities'][0]=.2
   elif kind=='source':joint['source_references']=[]
   elif kind=='native':joint['native_status']='CONVERGED'
   else:joint['fixed_q_audit']['first_stage_cost_usd']+=1
   ref=self.write('changed_'+kind+'.json',joint);bridge=dict(self.bridge,candidate_audit_reference=ref)
   bridge.pop('bridge_sha256');bridge['bridge_sha256']=canonical_hash(bridge)
   with self.subTest(kind=kind),self.assertRaises(ValueError):validate_saa_record_bridge(bridge,self.request,self.candidate)
 def test_budget_reset_and_bridge_relabel_rejected(self):
  args=self.fixture();args['prior_cumulative_seconds']=0
  with self.assertRaises(ValueError):settle_plan_record(**args)
  self.candidate['raw_engine_status']='CONVERGED'
  with self.assertRaises(ValueError):validate_saa_record_bridge(self.bridge,self.request,self.candidate)

class SAAWorkerReceiptIntegrationTests(unittest.TestCase):
 def test_actual_worker_through_receipt_and_final_seal(self):
  import test_saa_planning_worker as workers
  f=workers.SAAWorkerTests(methodName='runTest');f.setUp();self.addCleanup(f.doCleanups)
  queue=f.write(f.root/'queue.json',dict(domain='development',items=[dict(item_id='saa',plan_id='tiny.saa',
   request_reference=f.reqref,plan_budget_seconds=30.)]))
  invref=f.invocation();inv=json.loads(Path(invref['path']).read_text());inv.update(item_id='saa',request_reference=queue)
  invref=f.write(Path(invref['path']),inv)
  result=f.run_worker(invref)
  elapsed=result['attempt_accounted_seconds']+.01
  receipt=f.write(f.root/'receipt.json',dict(item_id='saa',plan_id='tiny.saa',request_reference=queue,
   invocation_reference=invref,result_reference=file_reference(inv['result_path']),elapsed_seconds=elapsed,status=result['status'],returncode=0))
  answer=settle_plan_record(receipt_reference=receipt,numerical_audit_reference=result['numerical_audit_reference'],
   prior_cumulative_seconds=0.,planning_budget=f.budget)
  final=finalize_settled_record(answer,actual_attempt_seconds_at_cutoff=elapsed+.01,final_seal_reserve_seconds=1.,output_directory=f.root/'final')
  self.assertEqual(final['solver_record']['method_id'],'SAA')
  self.assertEqual(final['solver_record']['status'],'CONVERGED')
  self.assertEqual(final['solver_record']['cumulative_wall_seconds'],elapsed+.01+1.)
  self.assertTrue(final['solver_record']['requires_seal_confirmation'])
  self.assertFalse(final['solver_record']['formal_OOS'])
