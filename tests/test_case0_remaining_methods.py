import copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from evcs_experiments.case0_remaining_methods import run_remaining

class RemainingMethodsTests(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)/'ieee33-v54-q1-case0-expanded';self.root.mkdir();self.op=self.root/'operations/new';self.op.mkdir(parents=True)
  self.s=dict(size_index=0,evaluation_opened=False,active_configuration=None,completed={},compute_seconds=1860.,configuration_spent_seconds={'size0/EA_DRO':1781.},configuration_master_spent_seconds={'size0/EA_DRO':1183.},source_release_reference={'sha256':'original'},active_config_reference={'sha256':'config'})
  self.config=dict(configuration_seconds=1800,master_seconds=1200,total_compute_seconds=21600,gap_target=.005,planning_methods=['EA_DRO','OR_DRO','SAA','ARO'],backend='gurobi')
  config_path=self.root/'config.json';config_path.write_text(json.dumps(self.config));self.s['active_config_reference']={'path':str(config_path),'sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'size_bytes':config_path.stat().st_size}
  self.calls=[];self.gap=.001;self.worker_fails=False
  def read(p):return json.loads(Path(p).read_text())
  def save(root,state):(root/'state.json').write_text(json.dumps(state))
  def worker(method,start,folder):
   self.assertNotEqual(method,'EA_DRO');self.calls.append(method)
   if self.worker_fails:raise RuntimeError('Expected isolated child failure')
   result=dict(method_id=method,backend='gurobi',numerical_valid=True,formal_OOS=False,gap_passed=self.gap<=.005,gap=self.gap,master_wall_seconds=0.)
   (folder/'solve').mkdir();(folder/'solve/result.json').write_text(json.dumps(result));return result
  self.api=SimpleNamespace(read=read,save_state=save,active_config=lambda r:self.config,verify_sources=lambda c:None,configuration_attempt=lambda r,s,k:(r/'planning'/k,0.),bounded_configuration=worker,digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),utcnow=lambda:'2026-09-10T17:21:00+00:00')
  self.accepted=SimpleNamespace(load_accepted_plan=lambda r,s:dict(comparison_result=dict(accepted_for_comparison=True,gap_passed=False,gap=.007101288)))
  save(self.root,self.s)
 def run_it(self,seconds=5500):return run_remaining(self.root,self.op,seconds=seconds,accepted_api=self.accepted,api=self.api)
 def test_only_remaining_three_keep_ea_gap_budget_and_close_oos(self):
  s=self.run_it();self.assertEqual(self.calls,['OR_DRO','SAA','ARO']);self.assertNotIn('size0/EA_DRO',s['completed']);self.assertEqual(s['configuration_spent_seconds']['size0/EA_DRO'],1781.);self.assertEqual(s['configuration_master_spent_seconds']['size0/EA_DRO'],1183.)
  self.assertFalse(s['evaluation_opened']);self.assertEqual(s['status'],'FOUR_METHOD_RESULTS_PENDING_ADAPTED_INDEPENDENT_AUDIT');self.assertTrue(s['remaining_methods_parent_pending_settlement'])
  a=json.loads((self.op/'inner_accounting.json').read_text());self.assertEqual(a['closing_inner_compute_seconds'],s['compute_seconds']);self.assertEqual(len(a['method_accounting_references']),3)
 def test_ea_exception_never_waives_other_gap(self):
  self.gap=.02;s=self.run_it();self.assertEqual(self.calls,['OR_DRO']);self.assertEqual(s['status'],'DIAGNOSIS_REQUIRED');self.assertFalse(s['completed']['size0/OR_DRO']['gap_passed'])
 def test_failed_child_retains_fees_and_no_automatic_retry(self):
  self.worker_fails=True;s=self.run_it();self.assertEqual(self.calls,['OR_DRO']);self.assertEqual(s['status'],'FAILURE_REQUIRES_REVIEW');self.assertIsNone(s['active_configuration']);self.assertGreater(s['configuration_spent_seconds']['size0/OR_DRO'],0.)
  r=json.loads((self.root/'planning/size0/OR_DRO/accounting.json').read_text());self.assertFalse(r['master_seconds_measured']);self.assertEqual(r['master_seconds'],r['configuration_wall_seconds'])
 def test_parent_boundary_does_not_start_incomplete_configuration(self):
  s=self.run_it(100);self.assertEqual(self.calls,[]);self.assertEqual(s['status'],'REMAINING_METHODS_PARENT_BOUNDARY')
 def test_no_ea_acceptance_or_forged_normal_flag_rejected(self):
  self.accepted.load_accepted_plan=lambda r,s:None
  with self.assertRaisesRegex(ValueError,'accepted EA'):self.run_it()
  self.assertEqual(self.calls,[])
 def test_unsettled_parent_blocks_another_batch(self):
  self.s['remaining_methods_parent_pending_settlement']=True;self.api.save_state(self.root,self.s)
  with self.assertRaisesRegex(ValueError,'unsettled parent'):self.run_it()
  self.assertEqual(self.calls,[])
 def test_old_failed_remaining_result_cannot_be_skipped(self):
  p=self.root/'badresult.json';p.write_text(json.dumps(dict(method_id='OR_DRO',backend='gurobi',numerical_valid=True,gap_passed=False,gap=.02)))
  self.s['completed']['size0/OR_DRO']=dict(path=str(p),sha256=self.api.digest(p));self.api.save_state(self.root,self.s)
  with self.assertRaisesRegex(ValueError,'normally accepted'):self.run_it()
  self.assertEqual(self.calls,[])
if __name__=='__main__':unittest.main()
