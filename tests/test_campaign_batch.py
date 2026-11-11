"""Short subprocess integration checks for the immutable bounded queue."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from evcs_experiments import campaign_batch as batch

real_identity = batch.process_identity

WORKER = '''import json,os,time,sys,hashlib
from pathlib import Path
inv=json.loads(Path(os.environ['EVCS_BATCH_ATTEMPT_REQUEST']).read_text())
req=json.loads(Path(inv['item_request_reference']['path']).read_text())
root=Path(inv['attempt_directory'])
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
with Path(req['counter']).open('a') as f:f.write(inv['item_id']+'\\n')
mode=req.get('mode','complete')
if mode=='nonzero':sys.exit(7)
if mode=='sleep':time.sleep(10)
if mode=='descendant':
 import subprocess
 subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
time.sleep(req.get('delay',0))
artifact=root/'artifact.json';artifact.write_text(json.dumps(inv))
result=dict(status='COMPLETE',invocation_sha256=os.environ['EVCS_BATCH_INVOCATION_SHA256'],artifacts=[ref(artifact)])
if mode in ('resume','always_resume') and (mode=='always_resume' or inv['previous_checkpoint'] is None):
 result.update(status='TIME_LIMIT',resumable=True,checkpoint=ref(artifact))
if mode=='invalid_limit':result.update(status='TIME_LIMIT',resumable=True)
Path(inv['result_path']).write_text(json.dumps(result))
'''

class CampaignBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='evcs-batch-test-');self.root=Path(self.temp.name)
        self.worker=self.root/'worker.py';self.worker.write_text(WORKER)
        self.budget=self.root/'budget.json';self.budget.write_text('{"cumulative_seconds":7200}')
        self.counter=self.root/'counter';self.queue=self.root/'queue';self.request=self.root/'request.json'
        self.children=[]

    def tearDown(self):
        for pid in self.children:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:os.waitpid(pid,0)
            except ChildProcessError:pass
        for process in batch._UNSETTLED_PROCESSES:
            process.poll()
        self.temp.cleanup()

    def prepare(self,modes=('complete',),*,wall=4.,reserve=.1,plan_budget=4.,shared=False,delay=0.):
        items=[]
        for index,mode in enumerate(modes):
            path=self.root/f'input-{index}.json';path.write_text(json.dumps(dict(mode=mode,counter=str(self.counter),delay=delay)))
            items.append(dict(item_id='item'+str(index),plan_id='shared' if shared else 'plan'+str(index),
                argv=[sys.executable,str(self.worker)],cwd=str(self.root),request_reference=batch.file_reference(path),plan_budget_seconds=plan_budget))
        request=dict(schema_version=1,batch_id='test',spec_id='S05',model_id='IEEE33-v54-Q1',domain='development',queue_root=str(self.queue),
            source_references=[batch.file_reference(batch.__file__),batch.file_reference(self.worker)],budget_reference=batch.file_reference(self.budget),
            job_wall_seconds=wall,shutdown_reserve_seconds=reserve,items=items)
        self.request.write_text(json.dumps(request));return request

    def run_queue(self):return batch.run_batch(self.request,self.queue)

    def test_serial_completion_then_verified_reuse(self):
        self.prepare(('complete','complete'))
        first=self.run_queue();second=self.run_queue()
        self.assertEqual(first['status'],'COMPLETE_BATCH');self.assertEqual(first['new_attempt_item_ids'],['item0','item1'])
        self.assertEqual(second['new_attempt_item_ids'],[]);self.assertEqual(second['reused_item_ids'],['item0','item1'])
        self.assertEqual(self.counter.read_text().splitlines(),['item0','item1'])
        self.assertEqual(first['cumulative_plan_seconds'],second['cumulative_plan_seconds'])
        self.assertFalse(second['scientific_acceptance']);self.assertGreater(second['verification_and_queue_overhead_seconds'],0.)

    def test_explicit_resume_preserves_partial_and_skips_completed(self):
        self.prepare(('complete','resume','complete'))
        first=self.run_queue();self.assertEqual(first['status'],'TIME_LIMIT')
        original=batch.verify_reference(first['checkpoint'])
        second=self.run_queue();third=self.run_queue()
        self.assertEqual(second['status'],'COMPLETE_BATCH');self.assertEqual(second['reused_item_ids'],['item0'])
        self.assertEqual(second['new_attempt_item_ids'],['item1','item2'])
        self.assertEqual(third['new_attempt_item_ids'],[])
        self.assertEqual(self.counter.read_text().splitlines(),['item0','item1','item1','item2'])
        batch.verify_reference(original)
        inv=json.loads((self.queue/'item1__attempt_0002/invocation.json').read_text())
        self.assertEqual(inv['previous_checkpoint'],original)
        self.assertGreater(inv['cumulative_plan_seconds'],0)
        self.assertLess(inv['plan_remaining_seconds'],inv['plan_budget_seconds'])
        self.assertGreater(second['cumulative_plan_seconds']['plan1'],first['cumulative_plan_seconds']['plan1'])

    def test_same_plan_items_share_budget_and_job_deadline(self):
        self.prepare(('complete','complete'),shared=True,delay=.03)
        result=self.run_queue();self.assertEqual(result['status'],'COMPLETE_BATCH')
        first=json.loads((self.queue/'item0__attempt_0001/invocation.json').read_text())
        second=json.loads((self.queue/'item1__attempt_0001/invocation.json').read_text())
        self.assertLess(second['remaining_seconds'],first['remaining_seconds'])
        self.assertGreater(second['cumulative_plan_seconds'],0)
        self.assertEqual(set(result['cumulative_plan_seconds']),{'shared'})
        for invocation in (first,second):
            self.assertLess(invocation['attempt_started_monotonic'], invocation['worker_deadline_monotonic'])
            self.assertLess(invocation['attempt_started_monotonic'], time.monotonic())
            # The start includes request/history verification before the
            # remaining-wall snapshot; workers share this host monotonic clock.
            self.assertGreaterEqual(invocation['worker_deadline_monotonic']-invocation['attempt_started_monotonic'],
                                    invocation['remaining_seconds'])

    def test_changed_source_rejected_before_first_worker(self):
        self.prepare();self.worker.write_text(WORKER+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'Frozen reference changed'):self.run_queue()
        self.assertFalse(self.counter.exists())

    def test_changed_completed_artifact_prevents_reuse_or_new_item(self):
        self.prepare(('complete','resume'));self.run_queue()
        (self.queue/'item0__attempt_0001/artifact.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Frozen reference changed'):self.run_queue()
        self.assertEqual(self.counter.read_text().splitlines(),['item0','item1'])

    def test_changed_old_checkpoint_prevents_resume(self):
        self.prepare(('resume',));first=self.run_queue()
        Path(first['checkpoint']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Frozen reference changed'):self.run_queue()
        self.assertEqual(self.counter.read_text().splitlines(),['item0'])

    def test_budget_cannot_reset_by_different_root_or_deleted_receipt(self):
        self.prepare();self.run_queue()
        with self.assertRaisesRegex(ValueError,'Queue root differs'):batch.run_batch(self.request,self.root/'other')
        state_path=self.queue/'state.json';state=json.loads(state_path.read_text());state['attempt_receipts']=[];state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'Receipt inventory changed'):self.run_queue()
        self.assertEqual(self.counter.read_text().splitlines(),['item0'])

    def test_invalid_time_limit_halts_and_preserves_partial(self):
        self.prepare(('invalid_limit','complete'))
        result=self.run_queue();self.assertEqual(result['status'],'FAILED_VALIDITY_OR_WORKER')
        self.assertTrue((self.queue/'item0__attempt_0001/artifact.json').exists())
        self.assertEqual(self.counter.read_text().splitlines(),['item0'])
        with self.assertRaisesRegex(RuntimeError,'unreviewed'):self.run_queue()

    def test_nonzero_exit_is_not_time_limit(self):
        self.prepare(('nonzero','complete'))
        result=self.run_queue();self.assertEqual(result['status'],'FAILED_VALIDITY_OR_WORKER')
        self.assertIn('nonzero',result['reason']);self.assertEqual(self.counter.read_text().splitlines(),['item0'])

    def test_hard_timeout_blocks_second_launch_while_worker_alive(self):
        self.prepare(('sleep','complete'),wall=.45,reserve=.05,plan_budget=1.)
        result=self.run_queue();self.assertEqual(result['status'],'FAILED_HARD_TIMEOUT')
        pid=result['active_process']['pid'];self.children.append(pid)
        self.assertTrue(batch.identity_alive(result['active_process']))
        with self.assertRaisesRegex(RuntimeError,'unreviewed'):self.run_queue()
        self.assertEqual(self.counter.read_text().splitlines(),['item0'])
        self.assertTrue(result['unsettled_plan_seconds'])
        self.assertGreater(result['cumulative_plan_seconds']['plan0'],0)

    def test_cumulative_budget_exhaustion_does_not_launch_again(self):
        # The real worker returns a real immutable checkpoint. Account for a
        # deterministic long prior attempt without sleeping or depending on
        # subsecond process startup speed on a loaded remote machine.
        self.prepare(('always_resume',),wall=200.,reserve=10.,plan_budget=100.)
        offset=[0.];group_checks=[0]
        actual_monotonic=time.monotonic;actual_group_members=batch.group_members
        clock=SimpleNamespace(monotonic=lambda:actual_monotonic()+offset[0])
        def observed_members(*args):
            members=actual_group_members(*args);group_checks[0]+=1
            if group_checks[0]==2:
                # This is the first worker's post-exit descendant check: its
                # elapsed wall accounting is now 95 seconds plus real overhead.
                offset[0]+=95.
            return members
        with patch.object(batch,'time',clock),patch.object(batch,'group_members',observed_members):
            first=self.run_queue();self.assertEqual(first['status'],'TIME_LIMIT')
            self.assertGreaterEqual(first['cumulative_plan_seconds']['plan0'],95.)
            self.assertLess(first['cumulative_plan_seconds']['plan0'],100.)
            second=self.run_queue()
        self.assertEqual(second['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(second['new_attempt_item_ids'],[])
        self.assertEqual(second['cumulative_plan_seconds'],first['cumulative_plan_seconds'])
        self.assertEqual(self.counter.read_text().splitlines(),['item0'])
        self.assertEqual(len(list(self.queue.glob('item0__attempt_*/invocation.json'))),1)
        batch.verify_reference(first['checkpoint'])

    def test_concurrent_invocation_cannot_launch_second_worker(self):
        self.prepare(('complete',),delay=.25)
        argv=[sys.executable,'-m','evcs_experiments.campaign_batch','--request',str(self.request),'--outputroot',str(self.queue)]
        first=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+2
            while not self.counter.exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(self.counter.exists())
            second=subprocess.run(argv,capture_output=True,text=True,timeout=2)
            self.assertNotEqual(second.returncode,0)
            self.assertIn('already active',second.stderr)
            stdout,stderr=first.communicate(timeout=3)
            self.assertEqual(first.returncode,0,stderr)
            self.assertEqual(json.loads(stdout)['status'],'COMPLETE_BATCH')
            self.assertEqual(self.counter.read_text().splitlines(),['item0'])
        finally:
            if first.poll() is None:first.kill();first.wait()

    def test_worker_cannot_leave_live_descendant_then_launch_next_item(self):
        self.prepare(('descendant','complete'))
        before={(p['pid'],p['token']) for p in batch.group_members(os.getpgrp(),os.getsid(0))}
        try:
            result=self.run_queue()
            self.assertEqual(result['status'],'FAILED_VALIDITY_OR_WORKER')
            self.assertIn('live group descendants',result['reason'])
            self.assertEqual(self.counter.read_text().splitlines(),['item0'])
        finally:
            self.children.extend(p['pid'] for p in batch.group_members(os.getpgrp(),os.getsid(0)) if (p['pid'],p['token']) not in before)

    def test_formal_without_jobctl_parent_rejected(self):
        request=self.prepare();request['domain']='formal';self.request.write_text(json.dumps(request))
        # This case requires no parent even when the suite itself runs in jobctl.
        with patch.dict(os.environ):
            os.environ.pop('EVCS_JOB_DIRECTORY',None)
            os.environ.pop('EVCS_JOB_ID',None)
            with self.assertRaisesRegex(ValueError,'jobctl parent'):self.run_queue()
        self.assertFalse(self.counter.exists())

    def test_reviewed_opening_cost_is_counted_once_across_real_subprocess_resume(self):
        self.prepare(('resume',),wall=30.,reserve=.1,plan_budget=30.)
        proof_path=self.root/'reviewed-predecessor.json';proof_path.write_text('{"accounted_seconds":16.85686}')
        opening={'plan0':dict(reference=batch.file_reference(proof_path),accounted_seconds=16.85686)}
        with patch.object(batch,'_carryovers',return_value=opening):
            first=self.run_queue();self.assertEqual(first['status'],'TIME_LIMIT')
            second=self.run_queue();self.assertEqual(second['status'],'COMPLETE_BATCH')
            third=self.run_queue()
        inv1=json.loads((self.queue/'item0__attempt_0001/invocation.json').read_text())
        inv2=json.loads((self.queue/'item0__attempt_0002/invocation.json').read_text())
        self.assertEqual(inv1['cumulative_plan_seconds'],16.85686)
        self.assertAlmostEqual(inv1['plan_remaining_seconds'],13.14314)
        self.assertEqual(inv2['cumulative_plan_seconds'],first['cumulative_plan_seconds']['plan0'])
        receipts=[json.loads(p.read_text()) for p in self.queue.glob('*__attempt_*/receipt.json')]
        self.assertAlmostEqual(second['cumulative_plan_seconds']['plan0'],16.85686+sum(r['elapsed_seconds'] for r in receipts))
        self.assertEqual(second['cumulative_plan_seconds'],third['cumulative_plan_seconds'])
        self.assertEqual(third['new_attempt_item_ids'],[])
        self.assertEqual(self.counter.read_text().splitlines(),['item0','item0'])
        self.assertIsNone(inv1['previous_checkpoint']);self.assertIsNotNone(inv2['previous_checkpoint'])
        self.assertEqual(inv1['failure_carryover_reference'],opening['plan0']['reference'])

    def test_carryover_history_cannot_drop_original_opening_reference(self):
        self.prepare();first=self.run_queue()
        state=json.loads((self.queue/'state.json').read_text())
        with self.assertRaisesRegex(ValueError,'carryover binding'):
            batch._history(state,batch.file_reference(self.request),self.queue,{'plan0':dict(accounted_seconds=1.,reference={'different':True})})

    def test_string_shell_command_rejected(self):
        request=self.prepare();request['items'][0]['argv']='echo unexpected';self.request.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError,'argument list'):self.run_queue()
        self.assertFalse(self.counter.exists())

    @contextlib.contextmanager
    def parent(self,timeout,*,elapsed=0.,token='linux:testboot:10000'):
        directory=self.root/('parent-'+str(time.monotonic_ns()));directory.mkdir()
        identity=dict(pid=os.getsid(0),sid=os.getsid(0),pgid=os.getsid(0),token=token,state='S')
        (directory/'manifest.json').write_text(json.dumps(dict(job_id=directory.name,spec_id='S05',timeout_seconds=timeout)))
        (directory/'status.json').write_text(json.dumps(dict(supervisor=identity,supervisor_sid=os.getsid(0),state='RUNNING')))
        with patch.dict(os.environ,EVCS_JOB_DIRECTORY=str(directory),EVCS_JOB_ID=directory.name), \
             patch.object(batch,'identity_alive',return_value=True), \
             patch.object(batch,'process_identity',side_effect=lambda pid: identity if pid==identity['pid'] else real_identity(pid)), \
             patch.object(batch.os,'sysconf',return_value=100), \
             patch.object(batch.time,'CLOCK_BOOTTIME',7,create=True), \
             patch.object(batch.time,'clock_gettime',return_value=100.+elapsed,create=True):
            yield directory

    def verified_request(self,modes=('resume',),**kwargs):
        request=self.prepare(modes,**kwargs);request['parent_wall_policy']='verified_parent_remaining_v1'
        self.request.write_text(json.dumps(request));return request

    def test_short_parent_then_long_parent_same_queue_real_worker_resume(self):
        self.verified_request(wall=30.,plan_budget=30.)
        original=batch.file_reference(self.request)
        with self.parent(2.,elapsed=.3):first=self.run_queue()
        self.assertEqual(first['status'],'TIME_LIMIT')
        with self.parent(20.,elapsed=.7):second=self.run_queue()
        self.assertEqual(second['status'],'COMPLETE_BATCH')
        a=json.loads((self.queue/'item0__attempt_0001/invocation.json').read_text())
        b=json.loads((self.queue/'item0__attempt_0002/invocation.json').read_text())
        self.assertEqual(a['request_reference'],original);self.assertEqual(b['request_reference'],original)
        self.assertEqual(b['previous_checkpoint'],first['checkpoint'])
        self.assertEqual(b['cumulative_plan_seconds'],first['cumulative_plan_seconds']['plan0'])
        self.assertLess(a['remaining_seconds'],1.6);self.assertGreater(b['remaining_seconds'],18.)
        self.assertGreater(second['cumulative_plan_seconds']['plan0'],first['cumulative_plan_seconds']['plan0'])

    def test_verified_time_limit_seal_is_charged_once(self):
        self.verified_request(wall=30.,reserve=.1,plan_budget=30.)
        with self.parent(2.):first=self.run_queue()
        receipt=json.loads((self.queue/'item0__attempt_0001/receipt.json').read_text())
        self.assertEqual(receipt['closing_reserve_seconds'],.1)
        self.assertEqual(receipt['charged_elapsed_seconds'],receipt['elapsed_seconds']+.1)
        with self.parent(20.):self.run_queue()
        inv=json.loads((self.queue/'item0__attempt_0002/invocation.json').read_text())
        self.assertEqual(inv['cumulative_plan_seconds'],receipt['charged_elapsed_seconds'])

    def test_future_charged_finalization_cannot_make_next_attempt_start_future(self):
        self.verified_request(('complete','complete'),wall=30.,plan_budget=30.)
        # Exercise real child execution but inject a predecessor's legitimate
        # future reserved endpoint; isolate finalization receipt validation.
        original_result=batch._result;history=batch._history
        def result(ref,inv):
            value=original_result(ref,inv)
            if 'item0__attempt_0001' in ref['path']:value['numerical_audit_reference']={'fixture':True}
            return value
        def finalize(*args):
            path=args[4]/'finalization.json';path.write_text('{}')
            return 10.,batch.file_reference(path)
        def replay(state,request_ref,root,carryovers):
            with patch.object(batch,'_finalization_charge',return_value=None):return history(state,request_ref,root,carryovers)
        with self.parent(20.),patch.object(batch,'_result',side_effect=result), \
             patch.object(batch,'_finalize_attempt',side_effect=finalize),patch.object(batch,'_history',side_effect=replay):
            outcome=self.run_queue()
        self.assertEqual(outcome['status'],'COMPLETE_BATCH')
        inv=json.loads((self.queue/'item1__attempt_0001/invocation.json').read_text())
        self.assertLessEqual(inv['attempt_started_monotonic'],time.monotonic())
        self.assertLessEqual(inv['remaining_seconds'],inv['plan_remaining_seconds'])

    def test_parent_elapsed_consumes_grant_before_worker(self):
        self.verified_request(wall=30.,plan_budget=30.)
        with self.parent(2.,elapsed=1.95):result=self.run_queue()
        self.assertEqual(result['status'],'BUDGET_EXHAUSTED');self.assertFalse(self.counter.exists())
        self.assertEqual(result['new_attempt_item_ids'],[])
        self.assertTrue(result['unsettled_plan_seconds'])
        self.assertGreater(result['pre_worker_cost']['elapsed_seconds'],0.)
        with self.parent(20.):
            with self.assertRaisesRegex(RuntimeError,'unreviewed'):self.run_queue()

    def test_verified_leading_source_verification_cost_is_charged(self):
        self.verified_request(wall=30.,plan_budget=30.)
        original=batch.checked_request;calls=[0]
        def checking(path):
            calls[0]+=1
            if calls[0]==1:time.sleep(.04)
            return original(path)
        with self.parent(2.),patch.object(batch,'checked_request',side_effect=checking):result=self.run_queue()
        inv=json.loads((self.queue/'item0__attempt_0001/invocation.json').read_text())
        self.assertGreaterEqual(result['cumulative_plan_seconds']['plan0'],.04)
        self.assertGreaterEqual(inv['parent_job']['wall_clock']['observed_monotonic']-inv['attempt_started_monotonic'],.04)
        self.assertLess(result['verification_and_queue_overhead_seconds'],.04)

    def test_parent_grant_above_frozen_max_rejected(self):
        self.verified_request(wall=30.)
        with self.parent(31.):
            with self.assertRaisesRegex(ValueError,'frozen queue maximum'):self.run_queue()
        self.assertFalse(self.counter.exists())

    def test_parent_clock_identity_unavailable_rejected(self):
        self.verified_request()
        with self.parent(2.,token='ps:some start'):
            with self.assertRaisesRegex(ValueError,'Linux'):self.run_queue()
        with self.parent(2.),patch.object(batch,'identity_alive',return_value=False):
            with self.assertRaisesRegex(ValueError,'not alive'):self.run_queue()
        self.assertFalse(self.counter.exists())

    def test_terminal_parent_and_wrong_identity_rejected(self):
        self.verified_request()
        with self.parent(2.) as parent:
            status=json.loads((parent/'status.json').read_text());status['state']='STOPPING'
            (parent/'status.json').write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'active work'):self.run_queue()
        with self.parent(2.) as parent:
            manifest=json.loads((parent/'manifest.json').read_text());manifest['job_id']='unrelated'
            (parent/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Parent job/spec/budget'):self.run_queue()
        self.assertFalse(self.counter.exists())

    def test_invalid_grant_and_unknown_policy_rejected(self):
        self.verified_request()
        for value in (0.,-1.,True,7201.):
            with self.parent(value):
                with self.assertRaises(ValueError):self.run_queue()
        request=json.loads(self.request.read_text());request['parent_wall_policy']='unreviewed'
        self.request.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError,'Unknown parent wall policy'):self.run_queue()

    def test_new_wall_policy_cannot_be_changed_on_resume(self):
        self.verified_request(wall=30.,plan_budget=30.)
        with self.parent(2.):self.run_queue()
        request=json.loads(self.request.read_text());request.pop('parent_wall_policy');self.request.write_text(json.dumps(request))
        with self.parent(30.):
            with self.assertRaisesRegex(ValueError,'request'):self.run_queue()
        self.assertEqual(self.counter.read_text().splitlines(),['item0'])

    def test_legacy_short_parent_still_rejected(self):
        self.prepare(wall=30.)
        with self.parent(2.):
            with self.assertRaisesRegex(ValueError,'Parent job/spec/budget'):self.run_queue()

    def test_verified_mode_without_parent_is_rejected(self):
        self.verified_request()
        with patch.dict(os.environ):
            os.environ.pop('EVCS_JOB_DIRECTORY',None)
            with self.assertRaisesRegex(ValueError,'requires jobctl parent'):self.run_queue()

if __name__=='__main__':unittest.main()
