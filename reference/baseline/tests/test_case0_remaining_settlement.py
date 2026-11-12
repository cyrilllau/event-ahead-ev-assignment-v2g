"""Metadata-only parent/receipt failure scenarios; no subprocess or solver."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import case0_remaining_settlement as settle
from evcs_experiments.campaign_batch import file_reference

class RemainingSettlementTests(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)/settle.CAMPAIGN;self.root.mkdir();self.root=self.root.resolve();self.op=self.root/'operations/remaining';self.op.mkdir(parents=True)
  self.jobid='case0-expanded-remaining-test';self.jd=self.root/'jobs'/self.jobid;self.jd.mkdir(parents=True)
  self.source=self.save(self.op/'source.json',{});self.config=self.save(self.op/'config.json',{})
  self.opening=dict(size_index=0,compute_seconds=100.,configuration_spent_seconds={settle.EA:10.},configuration_master_spent_seconds={settle.EA:5.},accepted_approximate_plans={'exactEA':1},source_release_reference=self.source,active_config_reference=self.config,completed={},active_configuration=None,active_job_id=None,status='READY',evaluation_opened=False)
  self.before=self.save(self.op/'state_before.json',self.opening)
  self.job=dict(job_id=self.jobid,cwd=str(self.root),outroot=str(self.root/'jobs'),argv=['python','guard.py'],threads=1,timeout_seconds=100,grace_seconds=1)
  jr=self.save(self.op/'job.json',self.job);self.save(self.jd/'manifest.json',self.job)
  base=dict(kind='CASE0_REMAINING_METHODS_REQUEST_V1',methods=list(settle.METHODS),operation_directory=str(self.op),references={'state':self.before,'config':self.config})
  self.review=dict(kind='CASE0_REMAINING_METHODS_PRODUCTION_REVIEW_V1',passed=True,settler_reference=file_reference(settle.__file__),driver_sha256=self.source['sha256'],guard_reference=self.source,job_request_reference=jr,external_cost_references=[],request_without_review=base)
  rev=self.save(self.op/'production_review.json',self.review);req=self.save(self.op/'request.json',dict(base,review_reference=rev))
  self.ad=self.save(self.op/'launch_admission.json',dict(kind='CASE0_REMAINING_METHODS_ADMISSION_V1',passed=True,review_reference=rev,driver_reference=self.source,guard_reference=self.source,settler_reference=file_reference(settle.__file__),external_cost_references=[],accepted_approximate_plans=self.opening['accepted_approximate_plans'],state_before_reference=self.before,job_request_reference=jr,request_reference=req,unit=self.jobid,producer_source_reference=self.source,opening_compute_seconds=100.,opening_configuration_spent_seconds={settle.EA:10.},opening_configuration_master_spent_seconds={settle.EA:5.}))
  self.parent=dict(job_id=self.jobid,state='SUCCEEDED',complete=True,elapsed_seconds=20.,started_at_utc='2026-09-10T00:00:00+00:00',finished_at_utc='2026-09-10T00:00:20+00:00')
  self.pr=self.save(self.jd/'result.json',self.parent);self.state=copy.deepcopy(self.opening);self.state.update(active_job_id=self.jobid,pending_remaining_methods=self.ad,status='REMAINING_METHODS_PARENT_BOUNDARY',compute_seconds=110.)
  self.save(self.root/'state.json',self.state);self.save(self.op/'accepted_ea_comparison.json',{'accepted':True})
  self.mock=patch.object(settle,'owned_unit_dead',return_value={'live_members':[]});self.mock.start();self.addCleanup(self.mock.stop)
 def save(self,path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return file_reference(path)
 def method(self,receipt=True,state_saved=True):
  f=self.root/'planning/size0/OR_DRO';f.mkdir(parents=True)
  self.save(f/'start.json',dict(method='OR_DRO',started_at_utc='2026-09-10T00:00:05+00:00',source_release_reference=self.source,active_config_reference=self.config,prior_configuration_seconds=0.,prior_master_seconds=0.,accepted_ea_record_reference=file_reference(self.op/'accepted_ea_comparison.json')))
  if receipt:
   r=self.save(f/'solve/result.json',dict(method_id='OR_DRO',backend='gurobi',formal_OOS=False,numerical_valid=True,gap=.004,gap_passed=True,master_wall_seconds=3.))
   self.save(f/'accounting.json',dict(method='OR_DRO',configuration_wall_seconds=5.,master_seconds=3.,master_seconds_measured=True,master_charge_scope='actual',global_charge_nested=True,failed=False,result_reference=r))
   if state_saved:
    self.state['completed']['size0/OR_DRO']=dict(path=r['path'],sha256=r['sha256']);self.state['configuration_spent_seconds']['size0/OR_DRO']=5.;self.state['configuration_master_spent_seconds']['size0/OR_DRO']=3.
  else:self.state['active_configuration']='size0/OR_DRO'
  self.save(self.root/'state.json',self.state)
 def bind_extras(self,extra):
  self.review['external_cost_references']=list(extra);rev=self.save(self.op/'production_review.json',self.review)
  req=self.save(self.op/'request.json',dict(self.review['request_without_review'],review_reference=rev));ad=json.loads(Path(self.ad['path']).read_text());ad.update(external_cost_references=list(extra),review_reference=rev,request_reference=req)
  self.ad=self.save(self.op/'launch_admission.json',ad);self.state['pending_remaining_methods']=self.ad;self.save(self.root/'state.json',self.state)
 def prep(self,extra=()):return settle.prepare_remaining_settlement(self.root,self.ad,self.pr,external_cost_references=extra)
 def test_parent_residual_once_ea_and_method_actual_fees_unchanged(self):
  self.method();extra=self.save(self.op/'fee.json',{'actual_seconds':2.});self.bind_extras([extra]);p=self.prep([extra]);r=p['record'];s=p['updated_state']
  self.assertEqual(r['parent_residual_seconds'],10.);self.assertEqual(s['compute_seconds'],122.);self.assertEqual(s['configuration_spent_seconds']['size0/OR_DRO'],5.);self.assertEqual(s['configuration_master_spent_seconds'][settle.EA],5.)
 def test_killed_without_finally_uses_start_to_terminal_no_free_master(self):
  self.method(receipt=False);self.parent['state']='TIMED_OUT';self.pr=self.save(self.jd/'result.json',self.parent);p=self.prep();r=p['record']
  self.assertEqual(r['method_accounting'][0]['configuration_seconds'],15.);self.assertEqual(r['method_accounting'][0]['master_seconds'],15.)
  self.assertEqual(r['status'],'REMAINING_METHODS_STOPPED_FAILURE');self.assertIsNone(p['updated_state']['pending_remaining_methods']);self.assertEqual(p['updated_state']['compute_seconds'],120.)
 def test_saved_receipt_before_state_write_is_reconciled(self):
  self.method(state_saved=False);p=self.prep();self.assertIn('size0/OR_DRO',p['updated_state']['completed']);self.assertEqual(p['updated_state']['configuration_spent_seconds']['size0/OR_DRO'],5.)
 def test_interrupted_master_upper_bound_is_not_clamped(self):
  self.method(receipt=False);self.parent.update(state='FAILED',elapsed_seconds=2000.,finished_at_utc='2026-09-10T00:33:20+00:00');self.pr=self.save(self.jd/'result.json',self.parent);p=self.prep()
  self.assertEqual(p['record']['method_accounting'][0]['master_seconds'],1995.);self.assertEqual(p['record']['budget_exceeded_methods'],['OR_DRO'])
 def test_saved_interrupted_master_wall_preserved_and_malformed_is_conservative(self):
  self.method(receipt=False);path=self.root/'planning/size0/OR_DRO/solve/master.json';self.save(path,{'wall_seconds':4.})
  self.assertEqual(self.prep()['record']['method_accounting'][0]['master_seconds'],4.)
  path.write_text('unfinished master JSON')
  self.assertEqual(self.prep()['record']['method_accounting'][0]['master_seconds'],15.)
 def test_admitted_external_fee_cannot_be_omitted(self):
  fee=self.save(self.op/'fee.json',{'actual_seconds':2.});self.bind_extras([fee])
  with self.assertRaisesRegex(ValueError,'fees omitted'):self.prep()
 def test_another_settler_source_is_rejected(self):
  ad=json.loads(Path(self.ad['path']).read_text());ad['settler_reference']=self.source;self.ad=self.save(Path(self.ad['path']),ad)
  with self.assertRaisesRegex(ValueError,'settlement source'):self.prep()
 def test_live_unit_ea_mutation_duplicate_fee_rejected(self):
  with patch.object(settle,'owned_unit_dead',side_effect=ValueError('live')):
   with self.assertRaisesRegex(ValueError,'live'):self.prep()
  self.state['configuration_spent_seconds'][settle.EA]=11.;self.save(self.root/'state.json',self.state)
  with self.assertRaisesRegex(ValueError,'EA costs'):self.prep()
  self.state['configuration_spent_seconds'][settle.EA]=10.;self.save(self.root/'state.json',self.state);fee=self.save(self.op/'fee.json',{'actual_seconds':1.})
  self.bind_extras([fee,fee])
  with self.assertRaisesRegex(ValueError,'Duplicate'):self.prep([fee,fee])
 def test_reviewed_commit_is_exactly_once_and_receipt_precedes_state(self):
  p=self.prep();review=self.save(self.op/'review.json',dict(kind='CASE0_REMAINING_SETTLEMENT_REVIEW_V1',passed=True,proposal_sha256=p['proposal_sha256'],settlement_source_reference=file_reference(settle.__file__)))
  settle.commit_remaining_settlement(self.root,p,review);state=json.loads((self.root/'state.json').read_text());self.assertEqual(state['compute_seconds'],120.);self.assertTrue(Path(state['remaining_methods_settlement_reference']['path']).exists())
  with self.assertRaises((ValueError,FileExistsError)):settle.commit_remaining_settlement(self.root,p,review)
  self.assertEqual(json.loads((self.root/'state.json').read_text())['compute_seconds'],120.)

if __name__=='__main__':unittest.main()
