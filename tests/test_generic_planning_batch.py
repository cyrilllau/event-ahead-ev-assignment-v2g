"""Three genuine small plans complete audits and accounting in one local batch."""
import json
from pathlib import Path
import sys,tempfile,unittest
from evcs_experiments.campaign_batch import file_reference,run_batch
from evcs_experiments.config import get_paths
import evcs_experiments.campaign_batch as queue

WORKER='''
import json,os
from types import SimpleNamespace
from pathlib import Path
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.formal_planning_worker import run_attempt,_DevelopmentFixture
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.ccg import CCGProblem
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.validation.s03_toys import build_golden_cases
f=build_golden_cases()[0];templates=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios);first=templates[0].first_stage
inv=file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST']);iv=json.loads(Path(inv['path']).read_text());req=json.loads(Path(iv['item_request_reference']['path']).read_text())
def prepare(**kw):
 cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],first.matrix_sha256,{'fixture':f.name},templates.__getitem__,capacity=2)
 p=CCGProblem(first,cache,f.probability_set,f.scenario_cost_constants_usd,'DRO' if kw['method_id'] in ('EA','OR') else 'ARO')
 e=RawPlanningEvidence(kw['evidence_root'],f.case,f.tasks,p,f.scenarios.__getitem__,domain='development',input_binding={'fixture':f.name},resume=kw['resume_evidence'],resume_snapshot_reference=kw['resume_snapshot_reference'])
 return SimpleNamespace(problem=p,evidence=e,sink_factory=e.sink_factory,input_binding={'fixture':f.name})
fixture=_DevelopmentFixture(prepare,req['budget'],dict(master_backend='highs',optimizer_threads=1,oracle_time_limit_seconds=2.,mip_relative_gap=1e-9,optimality_relative_gap=1e-8),.05)
run_attempt(inv,_development_fixture=fixture)
'''
class GenericPlanningBatchTests(unittest.TestCase):
 def test_three_configurations_finalize_and_reuse_without_relaunch(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();worker=root/'worker.py';worker.write_text(WORKER)
   def write(name,v):
    p=root/name;p.write_text(json.dumps(v));return file_reference(p)
   project=get_paths().project_root
   budget=dict(cumulative_wall_seconds=30.,single_master_seconds=3.,final_certificate_reserve_seconds=2.,shutdown_reserve_seconds=.1)
   br=write('budget.json',budget);items=[]
   for method in ('EA','ARO','OR'):
    req=dict(schema_version=1,domain='development',spec_id='S05',model_id='IEEE33-v54-Q1',plan_id='toy.'+method,target_id='toy',method_id=method,budget=budget,output_root=str(root/method),numerical_audit=dict(schema_version=1,postprocessing_reserve_seconds=.8,sources=dict(helper=file_reference(project/'scripts/audits/audit_s05_pending_recovery.py'),numerical_kernel=file_reference(project/'scripts/audits/audit_s05_candidate_recourse.py'))))
    items.append(dict(item_id=method,plan_id=req['plan_id'],request_reference=write(method+'.json',req),plan_budget_seconds=30.,argv=[sys.executable,str(worker)],cwd=str(project)))
   q=write('queue.json',dict(schema_version=1,domain='development',spec_id='S05',model_id='IEEE33-v54-Q1',batch_id='three-plans',queue_root=str(root/'queue'),job_wall_seconds=60.,shutdown_reserve_seconds=.1,source_references=[file_reference(queue.__file__)],budget_reference=br,items=items))
   result=run_batch(q['path'],root/'queue');self.assertEqual(result['status'],'COMPLETE_BATCH',result)
   self.assertEqual(result['completed_item_ids'],['EA','ARO','OR'])
   for name in ('EA','ARO','OR'):
    d=root/'queue'/(name+'__attempt_0001');seal=json.loads((d/'finalization.json').read_text());numeric=json.loads((d/'final_record/solver_record.json').read_text())
    self.assertEqual(numeric['method_id'],name);self.assertEqual(numeric['plan_id'],'toy.'+name);self.assertTrue(numeric['full_support_verified']);self.assertTrue(numeric['final_budget_acceptance']);self.assertLessEqual(seal['actual_attempt_seconds_before_confirmation_write'],seal['charged_attempt_seconds']);self.assertFalse(numeric['formal_OOS'])
   reused=run_batch(q['path'],root/'queue');self.assertEqual(reused['new_attempt_item_ids'],[]);self.assertEqual(reused['cumulative_plan_seconds'],result['cumulative_plan_seconds'])

   # Re-sealing the inner files cannot replace the state-pinned confirmation.
   d=root/'queue'/'EA__attempt_0001'
   sealpath=d/'finalization.json';seal=json.loads(sealpath.read_text())
   changed=seal['charged_attempt_seconds']+.001
   for key in ('solver_record','budget'):
    path=Path(seal['references'][key]['path']);value=json.loads(path.read_text())
    value['charged_attempt_seconds']=changed
    value['cumulative_wall_seconds']=value['prior_cumulative_wall_seconds']+changed
    path.write_text(json.dumps(value));seal['references'][key]=file_reference(path)
   seal['charged_attempt_seconds']=changed;sealpath.write_text(json.dumps(seal))
   with self.assertRaisesRegex(ValueError,'Frozen reference changed'):
    run_batch(q['path'],root/'queue')
