"""Independent real-metadata continuation/commit/guard exercise, no study solver."""
import copy,json,os,sys,unittest
from unittest.mock import Mock,patch
from tests import test_case0_representation as fixture
from evcs_experiments import case0_memory_repair as m,case0_campaign as c,case0_interruption as guard

class ContinuationIndependentTests(unittest.TestCase):
 def setUp(self):
  f=fixture.RepresentationTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f
  # Reproduce a genuinely incomplete original admission: its reservation survives.
  with patch.object(m,'_representation_ancestry',side_effect=ValueError('historical mirror mismatch')):
   with self.assertRaises(ValueError):f.prepare()
  original=copy.deepcopy(f.refs);before=copy.deepcopy(f.state)
  saved_path=f.base/'failed-state-before.json';saved_path.write_bytes((f.root/'state.json').read_bytes());saved=f.ref(saved_path)
  pre=f.write(f.base/'preflight.json',dict(passed=True,development_verification_seconds=16.21209406852722))
  wall=f.write(f.base/'failed-wall.json',dict(actual_seconds=.13020952977240086,requires_accounting_before_any_resume=True))
  reserve=f.ref(f.root/'revisions/compact-metadata-1/reservation.json')
  self.fee=dict(kind='CASE0_FAILED_REPRESENTATION_ADMISSION_SETTLEMENT_V1',solver_started=False,scientific_pass=False,
   reservation_reference=reserve,state_before_reference=saved,source_release_reference=before['source_release_reference'],
   preflight_reference=pre,failed_prepare_reference=wall,opening_compute_seconds=before['compute_seconds'],
   added_global_seconds=16.21209406852722+.13020952977240086,
   closing_compute_seconds=before['compute_seconds']+16.21209406852722+.13020952977240086,
   added_configuration_seconds=.13020952977240086,
   closing_configuration_seconds=before['configuration_spent_seconds'][m.KEY]+.13020952977240086,
   closing_master_seconds=before['configuration_master_spent_seconds'][m.KEY])
  fee=f.write(f.base/'failed-settlement.json',self.fee)
  state=dict(before,failed_representation_admission_settlement_reference=fee,failed_representation_reservation_reference=reserve,
   compute_seconds=self.fee['closing_compute_seconds'],configuration_spent_seconds={m.KEY:self.fee['closing_configuration_seconds']})
  c.save_state(f.root,state);self.state=state
  self.refs={k:v for k,v in original.items() if k not in ('review','state')};self.refs.update(state=f.ref(f.root/'state.json'),failed_admission_settlement=fee)
  f.new_sources[m.MODULE]='e'*64;self.seal()
 def seal(self):
  f=self.f;self.refs.pop('review',None)
  self.refs['review']=f.write(f.base/'continuation-review.json',dict(kind='CASE0_REPRESENTATION_CONTINUATION_REVIEW_V1',passed=True,revision=m.REPRESENTATION,references=copy.deepcopy(self.refs),new_source_sha256=f.new_sources))
  self.request=f.write(f.base/'continuation-request.json',dict(kind='CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1',revision=m.REPRESENTATION,references=self.refs,new_source_sha256=f.new_sources))
 def check(self):return m.check_representation_continuation(self.request['path'],self.request['sha256'])
 def prepare(self):return m.prepare_representation_continuation(self.request['path'],self.request['sha256'])
 def test_checker_no_writes_then_full_commit_and_advanced_validation(self):
  f=self.f;before={str(p):p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
  self.assertTrue(self.check()['passed']);self.assertEqual(before,{str(p):p.read_bytes() for p in f.root.rglob('*') if p.is_file()})
  state=self.prepare();config,t=m.validate_representation(f.root,state)
  self.assertIs(config['solver_compact_recourses'],True);self.assertEqual(t['opening_compute_seconds'],self.fee['closing_compute_seconds'])
  self.assertEqual(t['prior_master_seconds'],613.165128)
  state['compute_seconds']+=.1;m.validate_representation(f.root,state)
  with self.assertRaises(ValueError):self.check()
 def test_resealed_global_fee_reset_rejected(self):
  f=self.f;state=copy.deepcopy(self.state);state['compute_seconds']-=16.21209406852722;c.save_state(f.root,state)
  self.refs['state']=f.ref(f.root/'state.json');self.seal()
  with self.assertRaisesRegex(ValueError,'budget'):self.check()
 def test_second_source_change_rejected(self):
  self.f.new_sources['src/evcs_experiments/case0_planning.py']='f'*64;self.seal()
  with self.assertRaisesRegex(ValueError,'component|more than'):self.check()
 def test_postcommit_guard_claim_and_zero_exit_stale_state_fail_closed(self):
  f=self.f;state=self.prepare();op=f.root/'operations/guard';job=f.root/'jobs/guard';job.mkdir()
  argv=[sys.executable,'-m','evcs_experiments.case0_interruption','guarded-plan','--root',str(f.root),'--operation',str(op),'--unit','case0-expanded-compact-test','--seconds','7180','--cpu','0']
  f.write(job/'manifest.json',dict(job_id='guard',argv=argv,outroot=str(f.root/'jobs'),threads=1,timeout_seconds=7200,grace_seconds=5))
  def child(_):
   latest=c.read(f.root/'state.json');latest.update(status='PLANNING',active_configuration=m.KEY);c.save_state(f.root,latest);return Mock(wait=Mock(return_value=0))
  with patch.dict(os.environ,EVCS_JOB_ID='guard',EVCS_JOB_DIRECTORY=str(job)),patch.object(guard,'linger_enabled',return_value=True),patch.object(guard,'resource_sample',return_value={}),patch.object(guard.subprocess,'Popen',side_effect=child):
   self.assertEqual(guard.guarded_plan(f.root,op,'case0-expanded-compact-test'),1)
  self.assertEqual(c.read(op/'guard_failure.json')['kind'],'CASE0_REPRESENTATION_GUARD_RESULT_V1')
