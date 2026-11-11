import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference

PATH=get_paths().project_root/'scripts/reports/case0_report_job.py'
spec=importlib.util.spec_from_file_location('case0_report_job_under_test',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Case0ReportJobTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
  self.op=self.root/'operations/report';self.op.mkdir(parents=True);self.job=self.root/'jobs/report';self.job.mkdir(parents=True)
  self.config=self.root/'config.json';self.config.write_text('{}')
  self.script=self.root/'case0_summary.py';self.script.write_text("import argparse,pathlib\np=argparse.ArgumentParser();p.add_argument('--audit');p.add_argument('--sha256');p.add_argument('--output');a=p.parse_args();o=pathlib.Path(a.output);o.mkdir();(o/'stats.json').write_text('{}')\n")
  dummy=self.write(self.root/'evidence.json',{})
  self.audit=dict(kind='CASE0_INDEPENDENT_EVALUATION_NUMERICAL_AUDIT_V1',passed=True,item_count=768,items=[{}]*768,formal_OOS=False,Case1_acceptance=False,**{k:True for k in ('original_coefficients_checked','fixed_plan_rhs_checked','full_primal_dual_kkt_checked','physical_metrics_checked','paired_axes_checked')},**{k:dummy for k in ('lock_reference','planning_acceptance_reference','evaluation_parent_settlement_reference')})
  self.ar=self.write(self.root/'audit.json',self.audit)
  self.accept=dict(kind='CASE0_EVALUATION_ACCEPTANCE_V1',passed=True,item_count=768,numerical_audit_reference=self.ar,formal_OOS=False,Case1_acceptance=False,audit_parent_settlement_reference=dummy,**{k:dummy for k in ('lock_reference','planning_acceptance_reference','evaluation_parent_settlement_reference')})
  self.acc=self.write(self.root/'accepted.json',self.accept)
  source=file_reference(self.script);review=self.write(self.root/'review.json',dict(kind='CASE0_SUMMARY_SOURCE_REVIEW_V1',passed=True,summary_script_reference=source))
  self.req=dict(kind='CASE0_REPORT_JOB_REQUEST_V1',operation_root=str(self.op),output_directory=str(self.op/'summary'),seconds=20,evaluation_acceptance_reference=self.acc,numerical_audit_reference=self.ar,summary_script_reference=source,summary_review_reference=review,wrapper_reference=file_reference(PATH))
  self.state=dict(status='CASE0_NUMERICAL_PASS_REPORT_PENDING',evaluation_opened=True,active_configuration=None,compute_seconds=4000.,evaluation_acceptance=self.acc)
  self.write(self.root/'state.json',self.state);self.seal_request()
  self.write(self.job/'status.json',dict(state='RUNNING'))
  for p in (patch.object(m,'campaign_root',return_value=self.root),patch.object(m,'active_config',return_value={'total_compute_seconds':21600}),patch.object(m,'verify_current_cgroup',return_value={'verified':True}),patch.object(m,'_supervisor_remaining',return_value={'conservative_deadline_monotonic':time.monotonic()+50}),patch.dict(os.environ,EVCS_JOB_ID='report',EVCS_JOB_DIRECTORY=str(self.job))):p.start();self.addCleanup(p.stop)
 def write(self,p,x):p.write_text(json.dumps(x));return file_reference(p)
 def seal_request(self):
  self.ref=self.write(self.op/'request.json',self.req)
  self.write(self.job/'manifest.json',dict(job_id='report',outroot=str(self.root/'jobs'),timeout_seconds=60,grace_seconds=10,argv=['isolation','--',sys.executable,str(PATH),'--config',str(self.config),'--request',self.ref['path'],'--sha256',self.ref['sha256']]))
 def test_reservation_precedes_real_child_and_original_settlement_shape(self):
  run=m.subprocess.run
  def check(*a,**k):
   state=json.loads((self.root/'state.json').read_text());reservation=json.loads((self.op/'reservation.json').read_text())
   self.assertEqual(reservation['kind'],'CASE0_AUDIT_RESERVATION_V1');self.assertEqual(state['audit_reservation_reference'],file_reference(self.op/'reservation.json'));return run(*a,**k)
  with patch.object(m.subprocess,'run',side_effect=check):result=m.run_report_job(self.ref,self.config)
  self.assertFalse(result['numerical_acceptance']);self.assertTrue(result['terminal_parent_settlement_required'])
  self.assertEqual(json.loads((self.root/'state.json').read_text())['compute_seconds'],4000.)
 def test_child_failure_remains_reserved_for_full_parent_settlement(self):
  with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=7)):
   with self.assertRaisesRegex(ValueError,'child failed'):m.run_report_job(self.ref,self.config)
  self.assertTrue((self.op/'failure.json').exists());self.assertTrue(json.loads((self.root/'state.json').read_text())['audit_reservation_reference'])
 def test_wrong_acceptance_rejected_before_child(self):
  self.accept['passed']=False;self.acc=self.write(self.root/'accepted.json',self.accept);self.req['evaluation_acceptance_reference']=self.acc;self.state['evaluation_acceptance']=self.acc;self.write(self.root/'state.json',self.state);self.seal_request()
  with patch.object(m.subprocess,'run',side_effect=AssertionError('Must not launch')):
   with self.assertRaisesRegex(ValueError,'accepted'):m.run_report_job(self.ref,self.config)
  self.assertTrue((self.op/'reservation.json').exists())
 def test_wrong_parent_identity_or_request_hash_cannot_launch(self):
  manifest=json.loads((self.job/'manifest.json').read_text());manifest['job_id']='other';self.write(self.job/'manifest.json',manifest)
  with patch.object(m.subprocess,'run',side_effect=AssertionError('Must not launch')):
   with self.assertRaisesRegex(ValueError,'manifest'):m.run_report_job(self.ref,self.config)
   with self.assertRaisesRegex(ValueError,'hash'):m.main(['--config',str(self.config),'--request',self.ref['path'],'--sha256','0'*64])
  self.assertFalse((self.op/'reservation.json').exists())

class ReportingLinuxClockCompatibilityTests(unittest.TestCase):
 def test_script_does_not_import_new_campaign_private_clock(self):
  import ast
  tree=ast.parse(PATH.read_text())
  imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module=='evcs_experiments.campaign_batch']
  self.assertEqual({alias.name for node in imports for alias in node.names},{'file_reference','verify_reference'})
 def test_original_conservative_linux_clock_and_pid_reuse_rejection(self):
  identity=dict(pid=100,sid=100,pgid=100,token='linux:testboot:10000')
  status=dict(state='RUNNING',supervisor=identity,supervisor_sid=100)
  with patch.object(m,'identity_alive',return_value=True),patch.object(m,'process_identity',return_value=identity),patch.object(m.os,'sysconf',return_value=100),patch.object(m.time,'CLOCK_BOOTTIME',7,create=True),patch.object(m.time,'clock_gettime',return_value=105.),patch.object(m.time,'monotonic',return_value=200.):
   clock=m._supervisor_remaining({'timeout_seconds':60},status)
   self.assertEqual(clock['elapsed_seconds'],5.);self.assertEqual(clock['remaining_seconds'],55.)
   self.assertEqual(clock['conservative_deadline_monotonic'],255.)
   with patch.object(m,'process_identity',return_value=dict(identity,token='linux:testboot:10001')):
    with self.assertRaisesRegex(ValueError,'identity'):m._supervisor_remaining({'timeout_seconds':60},status)
   with self.assertRaisesRegex(ValueError,'active'):m._supervisor_remaining({'timeout_seconds':60},dict(status,state='SUCCEEDED'))
