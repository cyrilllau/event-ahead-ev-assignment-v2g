import copy
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from evcs_experiments.validation.s04_raw_fixture import raw_fixture,frozen_fixture,physical_fixture,locked_factory
from evcs_experiments.evaluation_metrics import evaluate_development_plan,plan_identity
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.evaluation_batch import evaluate_locked_batch
from evcs_experiments.evaluation_lock import create_protocol_lock,lock_plan_batch,file_reference

class RawEvaluationTests(unittest.TestCase):
    def test_production_replay_preserves_plan_and_delays_dsp(self):
        raw=raw_fixture();sid=next(iter(raw['samples']));ea,re=physical_fixture(raw,sid,-1);dsp,rd=physical_fixture(raw,sid,0)
        self.assertEqual(plan_identity(ea.plan),plan_identity(dsp.plan))
        self.assertAlmostEqual(re.arrival_h[0],-.75);self.assertAlmostEqual(rd.arrival_h[0],.25)
        self.assertTrue(re.mu_full[0,0]);self.assertFalse(rd.mu_full[0,0]);self.assertTrue(rd.mu_full[0,1])
        a,_=evaluate_development_plan(ea.case,ea.tasks,ea.scenario,ea.plan)
        d,_=evaluate_development_plan(dsp.case,dsp.tasks,dsp.scenario,dsp.plan)
        self.assertEqual(a['plan_sha256_after'],d['plan_sha256_after'])
        self.assertAlmostEqual(d['costs']['Fshed_usd']-a['costs']['Fshed_usd'],1.5)
        self.assertAlmostEqual(d['metrics']['first_reserve_h'],1.)
        self.assertAlmostEqual(a['metrics']['first_reserve_h'],0.)
    def test_delay_changes_soc_without_state_reset(self):
        raw=raw_fixture();sid=list(raw['samples'])[2];f,re=physical_fixture(raw,sid,-1.)
        self.assertAlmostEqual(re.arrival_h[0],.25)
        self.assertAlmostEqual(re.arrival_soc[0],.8-(.3+1.)/80.)
        self.assertEqual(raw['states']['EV1']['soc'],.8)
    def test_tampered_route_rejected(self):
        raw=raw_fixture();raw['bank']['tasks'][0]['arc_ids']=[9]
        with self.assertRaises(ValueError):physical_fixture(raw,next(iter(raw['samples'])),-1.)

class LockedBatchTests(unittest.TestCase):
    def setup_lock(self,root,dsp_change=None):
        raw=raw_fixture();f=frozen_fixture(raw)
        def write(name,value):
            path=root/name;path.write_text(json.dumps(value));return file_reference(path)
        refs={name:write(name+'.json',{'kind':name}) for name in
              ('source','evaluator','config','random_streams','experiment_matrix','budgets',
               'planning_support','solver_record','budget')}
        refs['tasks']=write('raw.json',raw)
        refs['routes']=write('routes.json',raw['bank']['tasks'][0]['arc_ids'])
        refs['initial_state']=write('state.json',raw['states'])
        rows=[]
        for method in ('EA','DSP'):
            meta=dict(x=f.plan.x.tolist(),r=f.plan.r.tolist(),task_sha256=f.plan.task_sha256,
                      config_hash=f.plan.config_hash,execution_mode=method,departure_h=-1 if method=='EA' else 0)
            if method=='DSP':meta['ea_source_plan_id']='EA'
            if method=='DSP' and dsp_change=='x':meta['x'][0]=0
            if method=='DSP' and dsp_change=='r':meta['r'][0][0]=2
            if method=='DSP' and dsp_change=='source':meta['ea_source_plan_id']='missing'
            if method=='DSP' and dsp_change=='mode':meta['execution_mode']='EA'
            if method=='DSP' and dsp_change=='departure':meta['departure_h']=-1
            plan=FirstStagePlan(meta['x'],meta['r'],meta['task_sha256'],meta['config_hash'])
            identity=dict(plan_id=method,target_id=raw['target_id'],method_id=method,setting_id='baseline')
            record_refs={k:refs[k] for k in ('tasks','routes','initial_state','planning_support','solver_record','budget')}
            record_refs.update(plan=write(method+'.json',meta),planning_source=[refs['source']])
            if method=='DSP' and dsp_change in ('tasks','routes','initial_state'):
                record_refs[dsp_change]=write('different-'+dsp_change+'.json',{'changed':True})
            rows.append(dict(**identity,plan_sha256=plan_identity(plan),references=record_refs))
        scope={raw['target_id']:[next(iter(raw['samples']))]}
        protocol=dict(domain='development',model_version='v54-q1',target_ids=[raw['target_id']],
            method_ids=['EA','DSP'],references={k:[refs[k]] for k in
                ('source','evaluator','config','random_streams','experiment_matrix','budgets')},
            batches=[dict(batch_id='paired',expected_plans=[{k:r[k] for k in
                ('plan_id','target_id','method_id','setting_id')} for r in rows],sample_ids_by_target=scope)])
        # Both production fixture and independent negative factories are frozen.
        protocol['references']['evaluator'] += [file_reference(__file__),
            file_reference(Path(__import__('inspect').getsourcefile(locked_factory)))]
        lock=root/'locks';create_protocol_lock(lock,protocol);lock_plan_batch(lock,'paired',rows)
        return lock,scope,refs

    def test_locked_pair_runs_raw_replay_and_exports_matching_plan_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lock,scope,_=self.setup_lock(root)
            with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No first-stage solve')):
                result=evaluate_locked_batch(lock,'paired',locked_factory,root/'out',domain='development',sample_ids_by_target=scope)
            self.assertEqual(result['item_count'],2);self.assertFalse(result['formal_OOS'])
            a=json.loads((root/'out/item_000000/metrics.json').read_text())
            b=json.loads((root/'out/item_000001/metrics.json').read_text())
            self.assertEqual(a['sample_id'],b['sample_id'])
            self.assertEqual(a['plan_sha256_after'],b['plan_sha256_after'])
            self.assertTrue((root/'out/metrics_table.csv').is_file())

    def test_scope_domain_and_source_changes_rejected_before_sample_access(self):
        for change in ('sample','domain','source'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);lock,scope,refs=self.setup_lock(root);calls=[]
                def factory(*args):calls.append(args);return locked_factory(*args)
                domain='development'
                if change=='sample':scope={next(iter(scope)):['unlocked-sample']}
                if change=='domain':domain='formal'
                if change=='source':Path(refs['source']['path']).write_text('changed')
                with self.assertRaises(ValueError):
                    evaluate_locked_batch(lock,'paired',factory,root/'out',domain=domain,sample_ids_by_target=scope)
                self.assertEqual(calls,[])
                self.assertFalse((lock/'access-paired.json').exists())

    def test_dsp_changed_source_plan_routes_state_mode_or_departure_rejected_before_access(self):
        for change in ('x','r','source','tasks','routes','initial_state','mode','departure'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);lock,scope,_=self.setup_lock(root,dsp_change=change);calls=[]
                def factory(*args):calls.append(args);return locked_factory(*args)
                with self.assertRaises(ValueError):
                    evaluate_locked_batch(lock,'paired',factory,root/'out',domain='development',sample_ids_by_target=scope)
                self.assertEqual(calls,[])
                self.assertFalse((lock/'access-paired.json').exists())

    def test_factory_wrong_sample_domain_or_departure_preserves_failure(self):
        for change in ('sample','domain','departure'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);lock,scope,_=self.setup_lock(root);calls=[]
                def factory(*args):
                    calls.append(args);case,tasks,scene=locked_factory(*args)
                    if change=='sample':scene=replace(scene,scenario_id='wrong-sample')
                    if change=='domain':scene=replace(scene,development=False)
                    if change=='departure':scene=replace(scene,metadata={**scene.metadata,'departure_h':99})
                    return case,tasks,scene
                with self.assertRaises(ValueError):
                    evaluate_locked_batch(lock,'paired',factory,root/'out',domain='development',sample_ids_by_target=scope)
                self.assertEqual(len(calls),1)
                self.assertTrue((root/'out/item_000000/failure.json').is_file())
                self.assertFalse((root/'out/item_000000/metrics.json').exists())

    def test_uncertified_lp_preserves_solver_status_without_metrics_or_later_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lock,scope,_=self.setup_lock(root);calls=[]
            def factory(*args):calls.append(args);return locked_factory(*args)
            failure=SimpleNamespace(optimal=False,solution=None,status='TIME_LIMIT',certificate={},
                as_dict=lambda:dict(status='TIME_LIMIT',optimal=False,certificate={}))
            with patch('evcs_experiments.evaluation_batch.solve_fixed_plan',return_value=failure):
                with self.assertRaisesRegex(RuntimeError,'TIME_LIMIT'):
                    evaluate_locked_batch(lock,'paired',factory,root/'out',domain='development',sample_ids_by_target=scope)
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((root/'out/item_000000/solver.json').read_text())['status'],'TIME_LIMIT')
            self.assertTrue((root/'out/item_000000/failure.json').is_file())
            self.assertFalse((root/'out/item_000000/metrics.json').exists())
            self.assertFalse((root/'out/item_000001').exists())


if __name__=='__main__':unittest.main()
