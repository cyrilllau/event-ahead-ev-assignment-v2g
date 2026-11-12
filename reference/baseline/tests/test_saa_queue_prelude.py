"""Actual queue subprocess -> tiny SAA worker, including verification prelude."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
from evcs_experiments import campaign_batch as batch
from evcs_experiments.campaign_batch import file_reference
import test_campaign_batch as queues
import test_saa_planning_worker as workers

HARNESS='''import json,os
from pathlib import Path
from types import SimpleNamespace
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.saa_planning_worker import run_attempt,_DevelopmentFixture
class Recourses(tuple):pass
f=build_golden_cases()[0]
r=Recourses(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
r.scenario_ids=tuple(f.scenario_ids);r.scenario_sha256s=tuple(s.scenario_sha256 for s in f.scenarios)
p=SimpleNamespace(physical=SimpleNamespace(first_stage=r[0].first_stage,case=f.case,tasks=f.tasks,bank=SimpleNamespace(load=f.scenarios.__getitem__)),recourses=r,input_binding={'fixture':'pipeline'})
ref=file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
inv=json.loads(Path(ref['path']).read_text());req=json.loads(Path(inv['item_request_reference']['path']).read_text())
run_attempt(ref,_development_fixture=_DevelopmentFixture(prepare=lambda **kw:p,budget=req['budget']))
'''

class SAAQueuePreludeTests(unittest.TestCase):
 def setUp(self):
  self.f=workers.SAAWorkerTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.parent=queues.CampaignBatchTests(methodName='runTest');self.parent.setUp();self.addCleanup(self.parent.tearDown)
 def test_actual_queue_prelude_saa_worker_and_receipt_finalization(self):
  f=self.f;h=f.root/'tiny_worker.py';h.write_text(HARNESS);queue=f.root/'queue'
  budget=f.write(f.root/'budget.json',f.budget)
  request=dict(schema_version=1,batch_id='saa-prelude',spec_id='S05',model_id='IEEE33-v54-Q1',domain='development',
   queue_root=str(queue),job_wall_seconds=30.,shutdown_reserve_seconds=1.,parent_wall_policy='verified_parent_remaining_v1',
   source_references=[file_reference(batch.__file__),file_reference(h)],budget_reference=budget,
   items=[dict(item_id='saa',plan_id='tiny.saa',plan_budget_seconds=30.,argv=[sys.executable,str(h)],cwd=str(f.root),request_reference=f.reqref)])
  ref=f.write(f.root/'queue_request.json',request)
  actual=batch.checked_request;calls=[0]
  def delayed(path):
   calls[0]+=1
   if calls[0]==1:time.sleep(.04)
   return actual(path)
  with self.parent.parent(29.),patch.object(batch,'checked_request',side_effect=delayed):
   result=batch.run_batch(ref['path'],queue)
  self.assertEqual(result['status'],'COMPLETE_BATCH',result)
  attempt=queue/'saa__attempt_0001';inv=json.loads((attempt/'invocation.json').read_text())
  worker=json.loads((attempt/'worker_result.json').read_text());receipt=json.loads((attempt/'receipt.json').read_text())
  self.assertGreaterEqual(inv['parent_job']['wall_clock']['observed_monotonic']-inv['attempt_started_monotonic'],.04)
  self.assertGreaterEqual(worker['attempt_accounted_seconds'],.04)
  self.assertGreaterEqual(receipt['elapsed_seconds'],worker['attempt_accounted_seconds'])
  self.assertLessEqual(result['cumulative_plan_seconds']['tiny.saa'],30.)
  self.assertTrue(json.loads((queue/'state.json').read_text())['finalization_receipts'])
 def test_worker_refuses_true_plan_span_or_dispatch_grant_overrun(self):
  for index,key in enumerate(('dispatch_grant','plan_span'),1):
   ref=self.f.invocation(index=index);inv=json.loads(Path(ref['path']).read_text())
   if key=='dispatch_grant':inv['remaining_seconds']=.000001
   else:
    now=time.monotonic();inv.update(attempt_started_monotonic=now-29.,worker_deadline_monotonic=now+2.,remaining_seconds=3.)
   ref=self.f.write(Path(ref['path']),inv)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'grant|deadline'):
    self.f.run_worker(ref)
