"""Synthetic terminal files only: no formal bank generation or solver access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from evcs_experiments import failure_carryover as fc

class CarryoverTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
        self.c=self.root/'campaign';self.plan=self.c/'plans/old';self.plan.mkdir(parents=True)
        self.refs={};self.jobdir=self.c/'jobs/job';self.jobdir.mkdir(parents=True)
        self.oldrelease=self.release('old');self.newrelease=self.release('new',True)
        self.config=self.put(self.root/'config.json',{'planning_budget':{'cumulative_wall_seconds':7200}})
        old=dict(domain='formal',spec_id='S05',model_id='IEEE33-v54-Q1',plan_id='p',target_id='target',method_id='EA',implementation_method_mapping={'EA':'EA_DRO'},budget={'cumulative_wall_seconds':7200},config_reference=self.config,
                 protocol_reference=self.config,summary_reference=self.config,audit_reference=self.config,bank_acceptance_reference=self.config,frozen_plan_identity={'plan_id':'p'},release_reference=self.oldrelease,output_root=str(self.plan))
        self.new={**old,'release_reference':self.newrelease,'output_root':str(self.c/'plans/new')};self.queue=self.c/'batches/new'
        self.add('worker_request',old)
        qpath=self.c/'batches/old/state.json';qpath.parent.mkdir(parents=True)
        self.add('queue_request',dict(items=[dict(plan_id='p',request_reference=self.refs['worker_request'])],queue_root=str(qpath.parent)))
        self.add('job_manifest',dict(job_id='job',argv=['python','--request',self.refs['queue_request']['path']]))
        attempt=qpath.parent/'item__attempt_0001';attempt.mkdir()
        self.add('invocation',dict(cumulative_plan_seconds=0,previous_checkpoint=None,previous_result_reference=None,attempt_directory=str(attempt),item_request_reference=self.refs['worker_request'],request_reference=self.refs['queue_request'],parent_job=dict(manifest_reference=self.refs['job_manifest'],job_id='job')))
        self.add('failure',dict(error='Worker exited nonzero: 1',elapsed_seconds=16.6))
        self.add('worker_failure',dict(status='FAILED_WORKER_REQUIRES_DIAGNOSIS',error='Formal feature semantic identity differs',cumulative_accounted_seconds=16.))
        self.add('batch_result',dict(request_reference=self.refs['queue_request'],status='FAILED_VALIDITY_OR_WORKER',unsettled_plan_seconds=True,elapsed_seconds=16.7,cumulative_plan_seconds={'p':16.6}))
        self.refs['batch_state']=self.put(qpath,dict(attempt_receipts=[],halted='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW',active={'invocation_reference':self.refs['invocation']},runs=[self.refs['batch_result']],request_reference=self.refs['queue_request']))
        self.refs['job_result']=self.put(self.jobdir/'result.json',dict(job_id='job',complete=True,state='FAILED',returncode=2,elapsed_seconds=16.85686))
        self.identity={'lazy_input_binding':{'binding':{'planning_source':{'formal_physical_inputs.py':'old'},'case':'frozen'}},'first_stage':'same','scenario_ids':['a','b']}
        _,engine=fc._release(self.oldrelease)
        state=dict(identity=self.identity,problem_sha256=fc.canonical_hash(self.identity),engine_sha256=engine,trace=[],pending=None,best=None,lower_bound_usd=None,upper_bound_usd=None,next_iteration=1,status='RUNNING',phase1={'status':'FEASIBLE_BASIC'})
        self.refs['engine_checkpoint']=self.put(self.plan/'driver/checkpoint.json',dict(state=state,sha256=fc.canonical_hash(state)))
        self.refs['driver_state']=self.put(self.plan/'driver/driver_state.json',dict(status='FAILED_EXCEPTION',calls=[]))
        self.refs['driver_request']=self.put(self.plan/'driver/request.json',dict(problem_identity=self.identity))
        self.refs['raw_request']=self.put(self.plan/'raw/request.json',dict(problem_identity=self.identity))
        phase=self.put(self.plan/'raw/events/000000_phase1/solver.json',state['phase1'])
        event=self.put(self.plan/'raw/events/000000_phase1/record.json',dict(event='phase1',iteration=0,artifacts=[phase]))
        self.put(self.plan/'raw/history.json',[event]);self.put(self.plan/'raw/first_stage/record.json',dict(artifacts=[]))
        self.refs['plan_inventory']=self.put(self.root/'inventory.json',[fc.reference(p) for p in sorted(self.plan.rglob('*')) if p.is_file()])
        self.add('independent_review',dict(status='PASS_FAILURE_CARRYOVER_REVIEW',job_result=self.refs['job_result'],old_release=self.oldrelease,new_release=self.newrelease,plan_id='p'))
        oldsrc,_=fc._release(self.oldrelease);newsrc,_=fc._release(self.newrelease)
        self.record=dict(schema_version=1,kind=fc.KIND,plan_id='p',references=self.refs,old_release_reference=self.oldrelease,new_release_reference=self.newrelease,
                         successor_queue_root=str(self.queue),successor_output_root=self.new['output_root'],accounted_seconds=16.85686,source_changes={'formal_physical_inputs.py':dict(old=oldsrc['formal_physical_inputs.py'],new=newsrc['formal_physical_inputs.py'])})
        self.path=self.root/'carryover.json';self.pin=self.put(self.path,self.record)
        self.patch=patch.object(fc,'observation',return_value={'blocks_new_launch':False});self.patch.start();self.addCleanup(self.patch.stop)
    def put(self,p,x):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return fc.reference(p)
    def add(self,k,x):self.refs[k]=self.put(self.root/(k+'.json'),x)
    def release(self,name,new=False):
        root=self.root/name;rows=[]
        for n in (*fc.ENGINE_FILES,'formal_physical_inputs.py'):
            p=root/'src/evcs_experiments'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(n+(' changed' if new and n=='formal_physical_inputs.py' else ''))
            r=fc.reference(p);rows.append(dict(path=str(p.relative_to(root)),sha256=r['sha256'],size_bytes=r['size_bytes']))
        return self.put(root/'RELEASE_MANIFEST.json',dict(files=rows))
    def check(self,**kw):return fc.verify_failure_carryover(self.pin,self.new,self.queue,**kw)
    def test_exact_cost_and_unique_claim_then_repeat_read_is_idempotent(self):
        with self.assertRaises(FileNotFoundError):self.check()
        out=fc.claim_failure_carryover(self.pin,self.new,self.queue)
        self.assertEqual(out['accounted_seconds'],16.85686)
        self.assertAlmostEqual(7200-out['accounted_seconds'],7183.14314)
        self.assertEqual(self.check(),self.check())
        with self.assertRaises(FileExistsError):fc.claim_failure_carryover(self.pin,self.new,self.queue)
    def test_cost_cannot_be_reset_or_charged_twice(self):
        for amount in (0,16.,33.71372):
            self.pin=self.put(self.path,{**self.record,'accounted_seconds':amount})
            with self.subTest(amount=amount),self.assertRaisesRegex(ValueError,'cost reset'):self.check(require_claim=False)
    def test_other_successor_plan_budget_and_input_ref_rejected(self):
        for key,value in (('plan_id','other'),('budget',{'cumulative_wall_seconds':7300}),('summary_reference',self.refs['failure'])):
            old=self.new;self.new={**old,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(require_claim=False)
            self.new=old
        with self.assertRaisesRegex(ValueError,'successor'):fc.verify_failure_carryover(self.pin,self.new,self.c/'batches/other',require_claim=False)
    def test_raw_inventory_add_delete_and_byte_change_rejected(self):
        p=self.plan/'extra';p.write_text('x')
        with self.assertRaisesRegex(ValueError,'inventory changed'):self.check(require_claim=False)
        p.unlink();p=self.plan/'raw/history.json';p.write_text('[]')
        with self.assertRaisesRegex(ValueError,'reference changed'):self.check(require_claim=False)
    def test_live_or_nonterminal_predecessor_rejected(self):
        with patch.object(fc,'observation',return_value={'blocks_new_launch':True}):
            with self.assertRaisesRegex(ValueError,'alive'):self.check(require_claim=False)
        ref=self.refs['job_result'];x=json.loads(Path(ref['path']).read_text());x['complete']=False;self.refs['job_result']=self.put(Path(ref['path']),x)
        self.pin=self.put(self.path,self.record)
        with self.assertRaises(ValueError):self.check(require_claim=False)
    def test_checkpoint_rmp_progress_rejected_even_if_envelope_resigned(self):
        ref=self.refs['engine_checkpoint'];cp=json.loads(Path(ref['path']).read_text());cp['state']['trace']=[{'iteration':1}];cp['sha256']=fc.canonical_hash(cp['state']);self.refs['engine_checkpoint']=self.put(Path(ref['path']),cp);self.pin=self.put(self.path,self.record)
        with self.assertRaisesRegex(ValueError,'pre-RMP'):self.check(require_claim=False)
    def test_prepared_scientific_identity_can_only_change_provenance(self):
        verified=self.check(require_claim=False);new=deepcopy(self.identity);new['lazy_input_binding']['binding']['planning_source']={'adapter':'new'};fc.verify_prepared_problem(verified,new)
        new['scenario_ids'].reverse()
        with self.assertRaisesRegex(ValueError,'scientific problem'):fc.verify_prepared_problem(verified,new)
    def test_engine_change_cannot_be_hidden_in_source_manifest(self):
        manifest=json.loads(Path(self.newrelease['path']).read_text());row=next(r for r in manifest['files'] if r['path'].endswith('/ccg.py'));p=Path(self.newrelease['path']).parent/row['path'];p.write_text('changed');ref=fc.reference(p);row.update(sha256=ref['sha256'],size_bytes=ref['size_bytes']);self.newrelease=self.put(Path(self.newrelease['path']),manifest);self.new['release_reference']=self.newrelease;self.record['new_release_reference']=self.newrelease;self.pin=self.put(self.path,self.record)
        with self.assertRaisesRegex(ValueError,'engine/source'):self.check(require_claim=False)


    def test_actual_receipt_verifier_connects_to_queue_opening_gate(self):
        from evcs_experiments import campaign_batch as batch
        self.new['failure_carryover_reference']=self.pin
        request_path=self.root/'new-worker.json';self.put(request_path,self.new)
        fc.claim_failure_carryover(self.pin,self.new,self.queue)
        queue=dict(items=[dict(plan_id='p',request_reference=fc.reference(request_path),failure_carryover_reference=self.pin)],
                   queue_root=str(self.queue),source_references=[fc.reference(fc.__file__)])
        verified=batch._carryovers(queue)
        self.assertEqual(verified['p']['accounted_seconds'],16.85686)
        queue['items'].append(dict(queue['items'][0]))
        with self.assertRaisesRegex(ValueError,'one unique'):batch._carryovers(queue)

    def test_unlisted_python_and_extra_prior_attempt_rejected(self):
        extra=Path(self.newrelease['path']).parent/'src/evcs_experiments/unlisted.py';extra.write_text('x')
        with self.assertRaisesRegex(ValueError,'Python inventory'):self.check(require_claim=False)
        extra.unlink();(self.c/'batches/old/item__attempt_0002').mkdir()
        with self.assertRaisesRegex(ValueError,'attempt inventory'):self.check(require_claim=False)
    def test_resigned_phase1_state_cannot_borrow_original_raw(self):
        p=Path(self.refs['engine_checkpoint']['path']);cp=json.loads(p.read_text());cp['state']['phase1']['fake']=True;cp['sha256']=fc.canonical_hash(cp['state']);self.refs['engine_checkpoint']=self.put(p,cp)
        self.refs['plan_inventory']=self.put(self.root/'inventory.json',[fc.reference(p) for p in sorted(self.plan.rglob('*')) if p.is_file()]);self.pin=self.put(self.path,self.record)
        with self.assertRaisesRegex(ValueError,'Phase-I raw'):self.check(require_claim=False)

if __name__=='__main__':unittest.main()
