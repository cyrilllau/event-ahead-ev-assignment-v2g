import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import saa_warm_start_recovery as memory,campaign_batch as b,native_gurobi as native,saa_planning_pipeline as pipeline,saa_planning_worker as worker

class WarmStartRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.output=self.root/'planning';old=self.output/'phases-memory-retry-1';(old/'master').mkdir(parents=True);(old/'checkpoints').mkdir()
  def write(name,v):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return b.file_reference(p)
  self.write=write;refs=[b.file_reference(m.__file__) for m in (native,pipeline,worker)]
  req=write('worker.json',dict(output_root=str(self.output),planning_source_references=refs,budget=dict(single_master_seconds=600,master_threads=4,normal_gap_relative=.005)))
  model=write('model.json',dict(domain='formal',scenario_ids=[str(i) for i in range(160)]))
  ni=dict(raw_model_status=9,solution_count=0,effective_options=dict(SoftMemLimit=64.,Threads=4))
  raw=write('planning/phases-memory-retry-1/master/native_result.json',dict(native_interface=ni))
  failed=write('planning/phases-memory-retry-1/master/result.json',dict(raw_native_result=raw,model_manifest=model,status='TIME_LIMIT',primal=None,lower_bound_usd=None,full_support_verified=False))
  preq=write('planning/phases-memory-retry-1/request.json',dict(model_manifest_reference=model,sources=refs,scenario_ids=[str(i) for i in range(160)]))
  cp=write('planning/phases-memory-retry-1/checkpoints/checkpoint_1.json',dict(status='FAILED',master_result_reference=failed,fixed_q_checkpoint_reference=None,candidate_audit_reference=None,request_reference=preq))
  terminal=write('terminal.json',dict(complete=True));settlement=write('settlement.json',dict(terminal_job_result_reference=terminal,new_queue_receipt_references=[],budget_within_limit=True))
  obs=write('resources.json',dict(available_memory_kib=256*1024*1024,native_soft_limit_gb=64.,not_a_process_rss_cap=True))
  self.p=dict(kind=memory.KIND,soft_memory_limit_gb=64.,worker_request_reference=req,settlement_reference=settlement,original_native_reference=refs[0],original_pipeline_reference=refs[1],original_worker_reference=refs[2],failed_master_reference=failed,failed_native_reference=raw,failed_phase_checkpoint_reference=cp,model_manifest_reference=model,terminal_result_reference=terminal,original_phase_root=str(self.output/'phases'),new_phase_root=str(self.output/'phases-warm-start-1'),resource_observation_reference=obs)
  from evcs_experiments import saa_candidate_warm_start as builder
  from tests.test_saa_candidate_warm_start import CandidateWarmStartTests
  fixture_root=self.root/'warmfixture';fixture_root.mkdir()
  model,evidence=CandidateWarmStartTests().fixture(fixture_root)
  self.p['builder_reference']=b.file_reference(builder.__file__);self.p['candidate_evidence']=evidence
  self.ref=write('policy.json',self.p);self.ad=dict(worker_request_reference=req,external_ledger=dict(expected_last_settlement_reference=settlement))
 def test_only_memory_and_new_phase_change(self):
  original=native._parameters;before=original(dict(time_limit=600,threads=4,mip_rel_gap=.001))
  def capture(**kw):return kw
  capture.__module__=pipeline.__name__
  with patch.object(worker,'run_saa_numerical_phases',capture):
   restore,record=memory.install(self.ref,self.ad)
   try:
    after=native._parameters(dict(time_limit=600,threads=4,mip_rel_gap=.001))
    self.assertEqual(after,(before[0],dict(before[1],SoftMemLimit=64.,Presolve=1,PrePasses=1)))
    token=object();kw=dict(output_directory=Path(self.p['original_phase_root']),resume_checkpoint_reference=None,model_manifest_reference=self.p['model_manifest_reference'],recourses=token,master_seconds=600)
    output=worker.run_saa_numerical_phases(**kw);self.assertEqual(output,dict(kw,output_directory=Path(self.p['new_phase_root'])));self.assertIs(output['recourses'],token)
    with self.assertRaises(ValueError):native._parameters(dict(time_limit=600,threads=1,mip_rel_gap=.001))
    Path(self.p['new_phase_root']).mkdir()
    with self.assertRaisesRegex(ValueError,'already exists'):worker.run_saa_numerical_phases(**kw)
   finally:restore()
  self.assertIs(native._parameters,original);self.assertEqual(record,memory.verify_policy(self.ref,self.ad))
 def test_resealed_scope_changes_rejected(self):
  for key,value in [('soft_memory_limit_gb',128.),('new_phase_root',str(self.root/'other')),('settlement_reference',self.write('otherhead.json',{}))]:
   with self.subTest(key=key),self.assertRaises(ValueError):memory.verify_policy(self.write('wrong.json',dict(self.p,**{key:value})),self.ad)
 def test_source_bytes_tamper_rejected(self):
  p=Path(self.p['failed_native_reference']['path']);p.write_text(p.read_text()+' ')
  with self.assertRaises(ValueError):memory.verify_policy(self.ref,self.ad)
 def test_refuses_wrong_old_native_limit(self):
  raw=b.read_json(self.p['failed_native_reference']['path']);raw['native_interface']['effective_options']['SoftMemLimit']=32.
  rawref=self.write('planning/phases-memory-retry-1/master/native_result.json',raw)
  failed=b.read_json(self.p['failed_master_reference']['path']);failed['raw_native_result']=rawref
  failedref=self.write('planning/phases-memory-retry-1/master/result.json',failed)
  cp=b.read_json(self.p['failed_phase_checkpoint_reference']['path']);cp['master_result_reference']=failedref
  cpref=self.write('planning/phases-memory-retry-1/checkpoints/checkpoint_1.json',cp)
  with self.assertRaisesRegex(ValueError,'native memory failure'):memory.verify_policy(self.write('badpolicy.json',dict(self.p,failed_native_reference=rawref,failed_master_reference=failedref,failed_phase_checkpoint_reference=cpref)),self.ad)
 def test_complete_hint_reaches_original_model_check_and_budget(self):
  import numpy as np
  from scipy.sparse import csr_matrix
  from tests.test_saa_candidate_warm_start import CandidateWarmStartTests
  fixture=self.root/'full';fixture.mkdir();model,evidence=CandidateWarmStartTests().fixture(fixture)
  self.p['candidate_evidence']=evidence;self.p['model_manifest_reference']=model
  # Isolate injection from policy admission already covered by the real fixtures.
  expected=memory.verify_policy(self.ref,self.ad)
  expected['new_phase_root']=str(self.output/'phases-warm-start-1')
  self.ref=self.write('injection.json',self.p)
  (Path(expected['new_phase_root'])/'master').mkdir(parents=True)
  seen={}
  def checked(*args,**kw):
   seen.update(kw)
   certificate=native._check_vector(*native._model_arrays(*args),kw['start'])
   if not certificate['passed']:raise ValueError('original model rejects hint')
   return certificate
  with patch.object(memory,'verify_policy',return_value=expected),patch.object(native,'solve_milp',checked):
   restore,_=memory.install(self.ref,self.ad)
   try:
    n=322;args=(np.ones(n),np.zeros(n),np.zeros(n),np.full(n,10.),csr_matrix(np.eye(n)),np.zeros(n),np.full(n,10.))
    result=native.solve_milp(*args,options=dict(time_limit=600,threads=4,mip_rel_gap=.001))
    self.assertTrue(result['passed']);self.assertGreater(seen['options']['time_limit'],0);self.assertLess(seen['options']['time_limit'],600)
    np.testing.assert_array_equal(seen['start'],np.r_[1,2,np.tile([5.,3.],160)])
   finally:restore()
