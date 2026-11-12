"""Genuine tiny SAA phases through the queue worker; no commercial solver."""
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace,MappingProxyType
import unittest
from unittest.mock import patch

from evcs_experiments import saa_planning_worker as w
from evcs_experiments.campaign_batch import file_reference,_result
from evcs_experiments.recourse import solve_fixed_plan
import test_saa_planning_pipeline as fixtures


class Recourses(tuple):pass


class SAAWorkerTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.SAAPipelineTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        args=self.f.args;self.root=self.f.f.root/'worker_tests';self.root.mkdir()
        self.budget=dict(cumulative_wall_seconds=30.,single_master_seconds=5.,final_certificate_reserve_seconds=5.,shutdown_reserve_seconds=1.)
        rec=Recourses(args['recourses']);rec.scenario_ids=tuple(args['scenario_ids']);rec.scenario_sha256s=tuple(args['scenario_sha256s'])
        bank=SimpleNamespace(load=args['scene_loader'])
        prepared=SimpleNamespace(physical=SimpleNamespace(first_stage=args['first_stage'],case=args['case'],tasks=args['tasks'],bank=bank),
                                 recourses=rec,input_binding=args['input_binding'],empirical_probabilities=self.f.model.manifest['empirical_probabilities'],
                                 scenario_cost_constants_usd=self.f.model.manifest['scenario_cost_constants_usd'])
        self.fixture=w._DevelopmentFixture(prepare=lambda **kw:prepared,budget=self.budget)
        self.req=dict(schema_version=1,method_id='SAA',domain='development',plan_id='tiny.saa',output_root=str(self.root/'output'),
            budget=self.budget,model_input_mode='preassembled',model_manifest_reference=args['model_manifest_reference'],helper_reference=args['helper_reference'],
            template_references=args['template_references'],planning_source_references=args['source_references'],
            summary_reference={},audit_reference={},target_id='tiny')
        self.reqref=self.write(self.root/'request.json',self.req)
    def write(self,p,v):p.write_text(json.dumps(v));return file_reference(p)
    def invocation(self,index=0,prior=0.,previous=None):
        d=self.root/f'attempt{index}';d.mkdir();now=time.monotonic()
        inv=dict(schema_version=1,domain='development',plan_id='tiny.saa',item_request_reference=self.reqref,
            attempt_started_monotonic=now,worker_deadline_monotonic=now+30-prior,remaining_seconds=30-prior,
            plan_remaining_seconds=30-prior,plan_budget_seconds=30.,cumulative_plan_seconds=prior,
            attempt_directory=str(d),result_path=str(d/'worker_result.json'),
            previous_checkpoint=None if previous is None else previous['checkpoint'],
            previous_result_reference=None if previous is None else file_reference(self.root/f'attempt{index-1}'/'worker_result.json'))
        return self.write(d/'invocation.json',inv)
    def run_worker(self,inv):return w.run_attempt(inv,_development_fixture=self.fixture)
    def test_real_phases_worker_queue_result_and_saa_bridge(self):
        inv=self.invocation();result=self.run_worker(inv)
        self.assertEqual(result['status'],'COMPLETE');self.assertTrue(result['full_support_verified'])
        self.assertEqual(result['raw_engine_status'],'OPTIMAL')
        self.assertEqual(_result(file_reference(self.root/'attempt0/worker_result.json'),inv)['status'],'COMPLETE')
        bridge=json.loads(Path(result['numerical_audit_reference']['path']).read_text())
        self.assertEqual(bridge['risk_certificate_kind'],'fixed_empirical')
        self.assertAlmostEqual(bridge['upper_bound_usd'],-69.73333333333333)
    def test_partial_q_resume_reuses_master_and_original_checkpoint(self):
        calls=[]
        def partial(template,plan,**kw):
            calls.append(template.scenario_sha256);r=solve_fixed_plan(template,plan,**kw)
            if len(calls)==2:return replace(r,status='LIMIT_REACHED',solver_status=1,optimal=False,solution=None,objective_usd=None,
                                           certificate=MappingProxyType({'optimal_certificate_valid':False}))
            return r
        with patch('evcs_experiments.saa_fixed_plan.solve_fixed_plan',side_effect=partial):old=self.run_worker(self.invocation())
        self.assertEqual(old['status'],'TIME_LIMIT');oldhash=file_reference(old['checkpoint']['path'])
        prior=old['cumulative_accounted_seconds']+.1
        with patch('evcs_experiments.saa_streaming_solve.solve_streaming_saa',side_effect=AssertionError('No master replay')):
            result=self.run_worker(self.invocation(1,prior,old))
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(file_reference(oldhash['path']),oldhash)
        self.assertGreater(result['cumulative_accounted_seconds'],prior)
    def test_fresh_assembly_runs_inside_worker_and_is_bound_in_checkpoint(self):
        self.req.pop('model_manifest_reference');self.req.update(model_input_mode='assemble',max_coefficient_bytes=50_000_000)
        self.reqref=self.write(self.root/'request.json',self.req)
        result=self.run_worker(self.invocation())
        self.assertEqual(result['status'],'COMPLETE')
        cp=json.loads(Path(result['checkpoint']['path']).read_text())
        self.assertEqual(Path(cp['model_manifest_reference']['path']),(self.root/'output/model/manifest.json').resolve())
        self.assertGreater(result['attempt_accounted_seconds'],0.)

    def test_formal_fixture_injection_is_rejected(self):
        invref=self.invocation();inv=json.loads(Path(invref['path']).read_text());inv['domain']='formal'
        ref=self.write(Path(invref['path']),inv)
        with self.assertRaisesRegex(ValueError,'Fixture'):self.run_worker(ref)
    def test_prior_budget_reset_and_uncommitted_root_rejected(self):
        invref=self.invocation();inv=json.loads(Path(invref['path']).read_text());inv['cumulative_plan_seconds']=1.
        with self.assertRaisesRegex(ValueError,'budget reset'):self.run_worker(self.write(Path(invref['path']),inv))
        (self.root/'output').mkdir()
        with self.assertRaisesRegex(ValueError,'Uncommitted'):self.run_worker(self.invocation(1))


if __name__=='__main__':unittest.main()

class SAAPreparationReceiptTests(unittest.TestCase):
    def test_actual_receipt_producer_binding_and_negative_cases(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            def write(name,obj):
                p=root/name;p.write_text(json.dumps(obj));return file_reference(p)
            model=write('model.json',{'numeric_artifacts':[]})
            queue=write('queue.json',{})
            invocation=dict(plan_id='p',request_reference=queue,item_id='prepare',result_path=str(root/'result.json'))
            iv=write('inv.json',invocation)
            result=dict(status='COMPLETE',invocation_sha256=iv['sha256'],artifacts=[model])
            rr=write('result.json',result)
            receipt=dict(plan_id='p',request_reference=queue,item_id='prepare',invocation_reference=iv,
                         result_reference=rr,status='COMPLETE',returncode=0,elapsed_seconds=2.)
            ref=write('receipt.json',receipt)
            args=dict(plan_id='p',queue_reference=queue,model_reference=model)
            self.assertEqual(w._preparation_receipt(ref,**args),(2.,True))
            other=write('other.json',{})
            self.assertEqual(w._preparation_receipt(ref,**dict(args,model_reference=other)),(2.,False))
            result['invocation_sha256']='wrong';rr=write('result.json',result)
            receipt['result_reference']=rr;ref=write('receipt.json',receipt)
            with self.assertRaisesRegex(ValueError,'invocation'):w._preparation_receipt(ref,**args)
            result['invocation_sha256']=iv['sha256'];rr=write('result.json',result)
            receipt['result_reference']=rr;receipt['returncode']=1;ref=write('receipt.json',receipt)
            with self.assertRaisesRegex(ValueError,'exit'):w._preparation_receipt(ref,**args)
