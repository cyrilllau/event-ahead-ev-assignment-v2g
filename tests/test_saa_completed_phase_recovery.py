import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import saa_completed_phase_recovery as m,campaign_batch as b,saa_planning_worker as worker,saa_planning_pipeline as pipeline

class CompletedPhaseRecoveryTests(unittest.TestCase):
 def fixture(self,root):
  def write(name,obj):
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return b.file_reference(p)
  refs=[b.file_reference(worker.__file__),b.file_reference(pipeline.__file__)]
  req=write('request.json',dict(output_root=str(root/'planning'),planning_source_references=refs))
  ids=list(map(str,range(160)));model=write('model.json',dict(input_binding={},scenario_ids=ids,scenario_sha256s=ids))
  phase=write('planning/phases-warm-start-1/request.json',dict(model_manifest_reference=model,sources=refs,input_binding={},scenario_ids=ids,scenario_sha256s=ids))
  q=write('q.json',dict(completed_indices=list(range(160))))
  master=write('master.json',{})
  joined=write('joint.json',dict(status='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT',passed=True,full_support_recourse_verified=True,risk_certificate_kind='fixed_empirical',formal_OOS=False,fixed_q_audit=dict(checkpoint_reference=q),master_audit=dict(result_reference=master)))
  cp=dict(status='COMPLETE_FORMAL_SAA_NUMERICAL_PHASES',formal_admission=True,formal_OOS=False,scientific_spec_pass=False,request_reference=phase,candidate_audit_reference=joined,fixed_q_checkpoint_reference=q,master_result_reference=master)
  cpref=write('planning/phases-warm-start-1/checkpoints/checkpoint_000007.json',cp);write('planning/phases-warm-start-1/checkpoint.json',cp)
  terminal=write('terminal.json',dict(complete=True));settlement=write('settlement.json',dict(terminal_job_result_reference=terminal,new_queue_receipt_references=[],budget_within_limit=True,cumulative_extra_seconds=4000.,closing_queue_seconds=143.,effective_spent_seconds=4143.))
  p=dict(kind=m.KIND,worker_request_reference=req,settlement_reference=settlement,completed_phase_checkpoint_reference=cpref,original_worker_reference=refs[0],original_pipeline_reference=refs[1],model_manifest_reference=model,terminal_result_reference=terminal,original_phase_root=str(root/'planning/phases'),completed_phase_root=str(root/'planning/phases-warm-start-1'))
  return write('policy.json',p),dict(worker_request_reference=req,external_ledger=dict(expected_last_settlement_reference=settlement)),p,write
 def test_exact_binding_and_phase_selection_without_budget_rewrite(self):
  with tempfile.TemporaryDirectory() as d:
   ref,ad,p,write=self.fixture(Path(d).resolve());original=worker.run_attempt;phase=worker.run_saa_numerical_phases
   def capture(**kw):return kw
   capture.__module__=pipeline.__name__
   with patch.object(worker,'run_saa_numerical_phases',capture):
    restore,record=m.install(ref,ad)
    try:
     self.assertEqual(record['derived_worker_binding']['replacement_count'],1)
     self.assertEqual(worker.run_attempt.__kwdefaults__,original.__kwdefaults__)
     selected=worker.__dict__['_explicit_completed_saa_phase'](None,p['model_manifest_reference'],p['worker_request_reference'])
     kwargs=dict(output_directory=Path(p['original_phase_root']),resume_checkpoint_reference=selected,model_manifest_reference=p['model_manifest_reference'],prior_accounted_seconds=4301.,master_seconds=600,deadline=5000.)
     out=worker.run_saa_numerical_phases(**kwargs)
     self.assertEqual(out,dict(kwargs,output_directory=Path(p['completed_phase_root']),prior_accounted_seconds=8301.))
     self.assertEqual(kwargs['prior_accounted_seconds'],4301.)
     self.assertEqual(record['external_prior_offset_seconds'],4000.)
     source,derived,_=m.derived_binding(p['original_worker_reference'])
     self.assertEqual(derived.count('attempt_accounted_seconds=used'),source.count('attempt_accounted_seconds=used'))
     self.assertEqual(derived.count('cumulative_accounted_seconds=prior+used'),source.count('cumulative_accounted_seconds=prior+used'))
     with self.assertRaises(ValueError):worker.__dict__['_explicit_completed_saa_phase']({},p['model_manifest_reference'],p['worker_request_reference'])
    finally:restore()
   self.assertIs(worker.run_attempt,original);self.assertIs(worker.run_saa_numerical_phases,phase)
 def test_incomplete_or_changed_source_head_refused(self):
  for mode in ('checkpoint','source','head','path'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    ref,ad,p,write=self.fixture(Path(d).resolve())
    if mode=='checkpoint':Path(p['completed_phase_checkpoint_reference']['path']).write_text('{}')
    elif mode=='source':p['original_worker_reference']=p['original_pipeline_reference'];ref=write('policy2.json',p)
    elif mode=='head':ad['external_ledger']['expected_last_settlement_reference']=p['terminal_result_reference']
    else:p['completed_phase_root']=str(Path(d)/'other');ref=write('policy2.json',p)
    with self.assertRaises(ValueError):m.verify_policy(ref,ad)
 def test_original_completed_pipeline_restores_without_master_or_q_solves(self):
  from tests import test_saa_planning_pipeline as fixtures
  from evcs_experiments import saa_fixed_plan
  f=fixtures.SAAPipelineTests();f.setUp();self.addCleanup(f.doCleanups)
  first=f.run_pipeline();old=b.file_reference(first['checkpoint_reference']['path'])
  with patch.object(pipeline.saa_streaming_solve,'solve_streaming_saa',side_effect=AssertionError('No master replay')),patch.object(saa_fixed_plan,'solve_fixed_plan',side_effect=AssertionError('No recourse replay')):
   restored=f.run_pipeline(resume_checkpoint_reference=old,prior_accounted_seconds=first['cumulative_provisional_seconds']+.01)
  self.assertEqual(restored['status'],'COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES')
  self.assertEqual(restored['candidate_audit_reference'],first['candidate_audit_reference'])
  self.assertEqual(b.file_reference(old['path']),old)
