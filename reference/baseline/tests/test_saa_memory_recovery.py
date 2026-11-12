import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import saa_memory_recovery as memory,campaign_batch as b,native_gurobi as native,saa_planning_pipeline as pipeline,saa_planning_worker as worker

class MemoryRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.output=self.root/'planning';old=self.output/'phases';(old/'master').mkdir(parents=True);(old/'checkpoints').mkdir()
  def write(name,v):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return b.file_reference(p)
  self.write=write;refs=[b.file_reference(m.__file__) for m in (native,pipeline,worker)]
  req=write('worker.json',dict(output_root=str(self.output),planning_source_references=refs,budget=dict(single_master_seconds=600,master_threads=4,normal_gap_relative=.005)))
  model=write('model.json',dict(domain='formal',scenario_ids=[str(i) for i in range(160)]))
  ni=dict(raw_model_status=17,solution_count=0,effective_options=dict(SoftMemLimit=24.,Threads=4))
  raw=write('planning/phases/master/native_result.json',dict(native_interface=ni))
  failed=write('planning/phases/master/result.json',dict(raw_native_result=raw,model_manifest=model,status='MEM_LIMIT',primal=None,lower_bound_usd=None,full_support_verified=False))
  preq=write('planning/phases/request.json',dict(model_manifest_reference=model,sources=refs,scenario_ids=[str(i) for i in range(160)]))
  cp=write('planning/phases/checkpoints/checkpoint_1.json',dict(status='FAILED',master_result_reference=failed,fixed_q_checkpoint_reference=None,candidate_audit_reference=None,request_reference=preq))
  terminal=write('terminal.json',dict(complete=True));settlement=write('settlement.json',dict(terminal_job_result_reference=terminal,new_queue_receipt_references=[],budget_within_limit=True))
  obs=write('resources.json',dict(available_memory_kib=256*1024*1024,native_soft_limit_gb=64.,not_a_process_rss_cap=True))
  self.p=dict(kind=memory.KIND,soft_memory_limit_gb=64.,worker_request_reference=req,settlement_reference=settlement,original_native_reference=refs[0],original_pipeline_reference=refs[1],original_worker_reference=refs[2],failed_master_reference=failed,failed_native_reference=raw,failed_phase_checkpoint_reference=cp,model_manifest_reference=model,terminal_result_reference=terminal,original_phase_root=str(old),new_phase_root=str(self.output/'phases-memory-retry-1'),resource_observation_reference=obs)
  self.ref=write('policy.json',self.p);self.ad=dict(worker_request_reference=req,external_ledger=dict(expected_last_settlement_reference=settlement))
 def test_only_memory_and_new_phase_change(self):
  original=native._parameters;before=original(dict(time_limit=600,threads=4,mip_rel_gap=.001))
  def capture(**kw):return kw
  capture.__module__=pipeline.__name__
  with patch.object(worker,'run_saa_numerical_phases',capture):
   restore,record=memory.install(self.ref,self.ad)
   try:
    after=native._parameters(dict(time_limit=600,threads=4,mip_rel_gap=.001))
    self.assertEqual(after,(before[0],dict(before[1],SoftMemLimit=64.)))
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
  rawref=self.write('planning/phases/master/native_result.json',raw)
  failed=b.read_json(self.p['failed_master_reference']['path']);failed['raw_native_result']=rawref
  failedref=self.write('planning/phases/master/result.json',failed)
  cp=b.read_json(self.p['failed_phase_checkpoint_reference']['path']);cp['master_result_reference']=failedref
  cpref=self.write('planning/phases/checkpoints/checkpoint_1.json',cp)
  with self.assertRaisesRegex(ValueError,'native memory failure'):memory.verify_policy(self.write('badpolicy.json',dict(self.p,failed_native_reference=rawref,failed_master_reference=failedref,failed_phase_checkpoint_reference=cpref)),self.ad)
