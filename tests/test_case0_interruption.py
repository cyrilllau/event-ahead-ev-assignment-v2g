"""Synthetic operational recovery only; no solver or production state."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from evcs_experiments import case0_campaign as c
from evcs_experiments import case0_interruption as m
from evcs_experiments.case0_isolation import isolated_command


class InterruptionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name).resolve();self.root=self.base/'ieee33-v54-q1-case0-expanded';self.root.mkdir()
        (self.root/'MANIFEST_SHA256.csv').write_text('synthetic manifest')
        self.release=self.base/'old_release';self.release.mkdir()
        self.old={}
        for name in sorted(m.CHANGED_SOURCES-{'src/evcs_experiments/case0_interruption.py'}|{'src/evcs_experiments/recourse.py'}):
            p=self.release/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('old '+name)
            self.old[name]=c.digest(p)
        self.current=dict(self.old)
        paths=SimpleNamespace(campaign_root=self.root,data_root=self.root)
        for obj,name,kw in [(c,'get_paths',dict(return_value=paths)),(c,'sources',dict(side_effect=lambda:dict(self.current))),
                            (m,'linger_enabled',dict(return_value=True)),(m,'observation',dict(return_value=dict(blocks_new_launch=False,supervisor_alive=False,command_alive=False,live_group_members=[])))]:
            p=patch.object(obj,name,**kw);p.start();self.addCleanup(p.stop)
        state=c.prepare('gurobi','expanded48-v1');self.config=m.file_reference(self.root/'config.lock.json')
        self.oldattempt=self.root/'planning/size0/EA_DRO';(self.oldattempt/'inputs').mkdir(parents=True)
        self.write(self.oldattempt/'inputs/identity.json',{'fixture':'unchanged'});self.write(self.oldattempt/'inputs/tasks.json',{'fixture':'unchanged'})
        self.parent=self.root/'jobs/old-parent';self.parent.mkdir(parents=True)
        self.parentref=self.write(self.parent/'result.json',dict(job_id='old-parent',complete=True,state='SUCCEEDED',elapsed_seconds=204.025118))
        self.manifest=self.write(self.parent/'manifest.json',dict(job_id='old-parent'))
        self.write(self.parent/'status.json',dict(job_id='old-parent',state='SUCCEEDED'))
        launch=self.write(self.base/'launch.json',{})
        before=dict(state,status='PLANNING',active_job_id='old-parent',active_configuration=m.KEY,
                    compute_seconds=11.148362567648292,planning_launch_admission_reference=launch)
        befref=self.write(self.base/'before.json',before)
        invref=self.write(self.base/'attempt.json',dict(root=str(self.oldattempt),files=[m.file_reference(p) for p in sorted(self.oldattempt.rglob('*')) if p.is_file()]))
        self.closing=215.16411233612823;opening=self.closing-204.025118;already=before['compute_seconds']-opening
        self.settlement=dict(kind='CASE0_INTERRUPTED_PARENT_SETTLEMENT_V1',scientific_pass=False,OOS_opened=False,
            configuration_key=m.KEY,parent_reference=self.parentref,launch_admission_reference=launch,
            state_before_reference=befref,original_attempt_manifest_reference=invref,observation_reference=self.write(self.base/'observation.json',{}),
            parent_elapsed_seconds=204.025118,prior_configuration_seconds=204.025118,prior_master_seconds=204.025118,
            opening_compute_seconds=opening,previously_accounted_compute_seconds=before['compute_seconds'],already_charged_parent_seconds=already,
            added_compute_seconds=204.025118-already,closing_compute_seconds=self.closing,original_attempt_directory=str(self.oldattempt))
        self.setref=self.write(self.base/'settlement.json',self.settlement)
        state.update(status='INTERRUPTION_REVIEW_REQUIRED',compute_seconds=self.closing,active_configuration=None,active_job_id=None,
            planning_parent_pending_settlement=False,planning_parent_settlement_reference=self.setref,
            configuration_spent_seconds={m.KEY:204.025118},configuration_master_spent_seconds={m.KEY:204.025118})
        c.save_state(self.root,state)
        sr=self.write(self.release/'RELEASE_MANIFEST.json',dict(files=[dict(path=k,sha256=v) for k,v in self.old.items()]))
        self.current.update({'src/evcs_experiments/case0_campaign.py':'b'*64,'src/evcs_experiments/case0_interruption.py':'c'*64})
        self.refs=dict(config=self.config,state=m.file_reference(self.root/'state.json'),original_source_manifest=sr,
            parent_manifest=self.manifest,parent_result=self.parentref,settlement=self.setref)
        self.seal_request()

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return m.file_reference(path)

    def seal_request(self):
        self.refs.pop('review',None)
        review=self.write(self.base/'review.json',dict(kind='CASE0_INTERRUPTION_REVIEW_V1',passed=True,
            references=dict(self.refs),new_source_sha256=dict(self.current)))
        self.refs['review']=review
        self.request=self.write(self.base/'request.json',dict(kind='CASE0_INTERRUPTION_REQUEST_V1',references=self.refs,
            original_release_root=str(self.release),new_source_sha256=dict(self.current)))

    def prepare(self):
        with patch.object(m.time,'monotonic',side_effect=[100.,100.125]):
            return m.prepare_interruption(self.request['path'],self.request['sha256'])

    def test_exact_carryover_reuses_original_inputs_and_preserves_all_science(self):
        original={str(p):p.read_bytes() for p in self.oldattempt.rglob('*') if p.is_file()}
        state=self.prepare();self.assertEqual(state['compute_seconds'],self.closing+.125)
        self.assertEqual(state['configuration_spent_seconds'][m.KEY],204.025118+.125)
        self.assertEqual(state['configuration_master_spent_seconds'][m.KEY],204.025118)
        revised=c.active_config(self.root);old=json.loads(Path(self.config['path']).read_text())
        self.assertEqual({k:v for k,v in revised.items() if k!='source_sha256'},{k:v for k,v in old.items() if k!='source_sha256'})
        folder,spent=c.configuration_attempt(self.root,state,m.KEY)
        self.assertEqual(folder,self.oldattempt/'continuation-1');self.assertEqual(spent,204.150118)
        for p,raw in original.items():self.assertEqual(Path(p).read_bytes(),raw)
        with self.assertRaises(FileExistsError):self.prepare()

    def test_mathematical_source_change_rejected_and_claim_prevents_free_retry(self):
        self.current['src/evcs_experiments/recourse.py']='d'*64;self.seal_request()
        with self.assertRaisesRegex(ValueError,'mathematical'):self.prepare()
        self.assertTrue((self.root/'revisions/interruption-1/reservation.json').exists())
        self.assertEqual(c.read(self.root/'state.json')['compute_seconds'],self.closing)
        with self.assertRaises(FileExistsError):self.prepare()

    def test_changed_old_input_rejected_without_overwrite(self):
        (self.oldattempt/'inputs/tasks.json').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'reference changed'):self.prepare()

    def test_unsettled_live_parent_cannot_continue(self):
        with patch.object(m,'observation',return_value=dict(blocks_new_launch=False,supervisor_alive=True,command_alive=False,live_group_members=[])):
            with self.assertRaisesRegex(ValueError,'live'):self.prepare()

    def test_nested_budget_reset_rejected_even_with_rehashed_review(self):
        self.settlement['prior_master_seconds']=0.;self.refs['settlement']=self.write(self.base/'settlement.json',self.settlement)
        state=c.read(self.root/'state.json');state['planning_parent_settlement_reference']=self.refs['settlement'];c.save_state(self.root,state)
        self.refs['state']=m.file_reference(self.root/'state.json');self.seal_request()
        with self.assertRaisesRegex(ValueError,'nested budget'):self.prepare()

    def guard_fixture(self,final_status='PLANNING',active=m.KEY):
        self.prepare();operation=self.root/'operations/guard';job=self.root/'jobs/recovery';job.mkdir()
        argv=[sys.executable,'-m','evcs_experiments.case0_interruption','guarded-plan','--root',str(self.root),
              '--operation',str(operation),'--unit','case0-expanded-recovery','--seconds','7180','--cpu','0']
        self.write(job/'manifest.json',dict(job_id='recovery',argv=argv,outroot=str(self.root/'jobs'),threads=1,timeout_seconds=7200,grace_seconds=5))
        def child(_):
            state=c.read(self.root/'state.json');state.update(status=final_status,active_configuration=active)
            if final_status=='PLANNING_REVIEW_REQUIRED':state['completed']={f'size0/{v}':{} for v in c.METHODS}
            c.save_state(self.root,state);return Mock(wait=Mock(return_value=0))
        return operation,job,child

    def test_child_zero_cannot_hide_active_scientific_work(self):
        op,job,child=self.guard_fixture()
        with patch.dict(os.environ,EVCS_JOB_ID='recovery',EVCS_JOB_DIRECTORY=str(job)),patch.object(m.subprocess,'Popen',side_effect=child),patch.object(m,'resource_sample',return_value={'memory.peak':123}):
            self.assertEqual(m.guarded_plan(self.root,op,'case0-expanded-recovery'),1)
        self.assertTrue((op/'guard_failure.json').exists())

    def test_fixed_global_guard_claim_prevents_second_operation(self):
        op,job,child=self.guard_fixture('PLANNING_REVIEW_REQUIRED',None)
        with patch.dict(os.environ,EVCS_JOB_ID='recovery',EVCS_JOB_DIRECTORY=str(job)),patch.object(m.subprocess,'Popen',side_effect=child),patch.object(m,'resource_sample',return_value={}):
            self.assertEqual(m.guarded_plan(self.root,op,'case0-expanded-recovery'),0)
            state=c.read(self.root/'state.json');state.update(status='INTERRUPTION_PREPARED',completed={});c.save_state(self.root,state)
            with self.assertRaises(FileExistsError):m.guarded_plan(self.root,op,'case0-expanded-recovery')

    def test_retain_failed_unit_is_explicit_and_legacy_default_unchanged(self):
        old=isolated_command('case0-test',['python'],100,0);self.assertIn('--collect',old)
        new=isolated_command('case0-expanded-recovery',['python'],100,0,retain_failed_unit=True)
        self.assertNotIn('--collect',new);self.assertIn('MemoryMax=8G',new)
        with self.assertRaises(ValueError):isolated_command('case0-other',['python'],100,0,retain_failed_unit=True)

    def test_telemetry_rejects_foreign_cgroup_and_retains_last_counters(self):
        output=SimpleNamespace(stdout='ControlGroup=/user.slice/foreign.service\nResult=oom-kill\n',returncode=0)
        with patch.object(m.subprocess,'run',return_value=output):
            with self.assertRaisesRegex(ValueError,'Foreign'):m.resource_sample('case0-expanded-recovery')
        output.stdout='ControlGroup=\nResult=oom-kill\nExecMainStatus=9\n'
        with patch.object(m.subprocess,'run',return_value=output):
            r=m.resource_sample('case0-expanded-recovery',{'memory.peak':800,'memory.events':{'oom_kill':1}})
        self.assertEqual(r['memory.events']['oom_kill'],1);self.assertEqual(r['memory.peak'],800)
        self.assertEqual(r['unit_properties']['Result'],'oom-kill')


if __name__=='__main__':unittest.main()
