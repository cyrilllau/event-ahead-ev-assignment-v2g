"""Small genuine Q1 goldens exercise every raw callback; no formal solve/OOS."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import load_npz

from evcs_experiments.ccg import CCGProblem,run_ccg
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.formal_planning import RawPlanningEvidence,load_formal_problem
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.probability import solve_worst_distribution
from evcs_experiments.probability_phase1 import solve_phase1
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.validation.s03_toys import build_golden_cases


class FormalPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=build_golden_cases()[0];f=cls.fixture
        cls.first=assemble_first_stage(f.case,f.tasks)
        cls.templates=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def prepare(self,risk='DRO',resume=False,history_ref=None):
        f=self.fixture
        cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],self.first.matrix_sha256,
            dict(fixture=f.name),self.templates.__getitem__,capacity=2)
        p=CCGProblem(self.first,cache,f.probability_set,f.scenario_cost_constants_usd,risk)
        if resume and history_ref is None:
            history_ref=self.snapshot_ref
        e=RawPlanningEvidence(self.root/'evidence',f.case,f.tasks,p,f.scenarios.__getitem__,
            domain='development',input_binding=dict(fixture=f.name),resume=resume,
            resume_snapshot_reference=history_ref if resume else None)
        return p,e

    def test_real_golden_callbacks_preserve_all_raw_coefficients_and_certificates(self):
        for risk in ('DRO','ARO'):
            with self.subTest(risk=risk):
                if (self.root/'evidence').exists():
                    self.root=self.root/'aro';self.root.mkdir()
                problem,e=self.prepare(risk)
                result=run_ccg(problem,evidence_sink=e.sink_factory(self.root/'call'),
                    wall_seconds=20,rmp_time_limit_seconds=5,oracle_time_limit_seconds=5,
                    mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
                self.assertEqual(result.status,'CONVERGED')
                self.assertAlmostEqual(result.state['upper_bound_usd'],self.fixture.expected[risk]['objective_usd'],places=7)
                events=[json.loads(Path(ref['path']).read_text()) for ref in e.events]
                self.assertEqual({r['event'] for r in events},{'phase1','rmp','recourse','probability','iteration'})
                for ref,row in zip(e.events,events):
                    d=Path(ref['path']).parent
                    for raw in row['artifacts']:self.assertEqual(file_reference(raw['path']),raw)
                    if row['event']=='recourse':
                        with np.load(d/'dual_kkt.npz',allow_pickle=False) as raw:
                            self.assertLess(np.max(abs(raw['stationarity'])),1e-8)
                            self.assertIn('inequality',raw.files);self.assertIn('upper',raw.files)
                        self.assertTrue(json.loads((d/'physical_check.json').read_text())['passed'])
                    if row['event']=='probability':
                        q=json.loads((d/'q_input.json').read_text());self.assertEqual(q['support_ids'],list(self.fixture.scenario_ids))
                        self.assertEqual(len(q['recourse_evidence']),3)
                        self.assertTrue(json.loads((d/'probability_check.json').read_text())['passed'])
                for i,t in enumerate(self.templates):
                    d=e.root/'templates'/f'{i:03d}'
                    self.assertEqual((load_npz(d/'Br_ub.npz')!=t.Br_ub).nnz,0)
                    with np.load(d/'arrays.npz') as raw:np.testing.assert_array_equal(raw['c'],t.c)
                self.assertLessEqual(len(problem.recourses.statistics()['resident_indices']),2)

    def test_restart_retains_q_evidence_and_rejects_any_changed_raw_artifact(self):
        p,e=self.prepare();result=run_ccg(p,evidence_sink=e.sink_factory(self.root/'call'),wall_seconds=20)
        self.assertEqual(result.status,'CONVERGED')
        self.snapshot_ref=e.snapshot(self.root/'accepted_snapshot.json')
        _,again=self.prepare(resume=True);self.assertEqual(set(again.q),set(e.q))
        raw=e.root/'templates'/'000'/'arrays.npz';raw.write_bytes(raw.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'changed'):self.prepare(resume=True)

    def test_rewritten_callback_and_mutable_journal_cannot_replace_pinned_history(self):
        p,e=self.prepare();sink=e.sink_factory(self.root/'call');sink('phase1',0,result=solve_phase1(p.ambiguity))
        pinned=e.snapshot(self.root/'accepted_snapshot.json')
        record=Path(e.events[0]['path']);row=json.loads(record.read_text());row['iteration']=999
        record.write_text(json.dumps(row));journal=e.root/'history.json';journal.write_text(json.dumps([file_reference(record)]))
        with self.assertRaisesRegex(ValueError,'changed|pinned'):self.prepare(resume=True,history_ref=pinned)

    def test_snapshot_rejects_uncommitted_callback_directory(self):
        p,e=self.prepare();(e.root/'events'/'000000_recourse').mkdir()
        with self.assertRaisesRegex(ValueError,'Uncommitted callback'):
            e.snapshot(self.root/'must_not_exist.json')
        self.assertFalse((self.root/'must_not_exist.json').exists())

    def test_snapshot_binds_template_records_and_survives_future_callback_append(self):
        p,e=self.prepare();sink=e.sink_factory(self.root/'call')
        sink('phase1',0,result=solve_phase1(p.ambiguity));e.template(0,self.templates[0])
        pin=e.snapshot(self.root/'first_snapshot.json');before=Path(pin['path']).read_bytes()
        sink('phase1',0,result=solve_phase1(p.ambiguity))
        self.assertEqual(Path(pin['path']).read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'history differs'):self.prepare(resume=True,history_ref=pin)
        pin=e.snapshot(self.root/'second_snapshot.json')
        d=e.root/'templates'/'000';raw=d/'arrays.npz';raw.write_bytes(raw.read_bytes()+b'x')
        record=d/'record.json';data=json.loads(record.read_text())
        data['artifacts']=[file_reference(r['path']) for r in data['artifacts']];record.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'changed'):self.prepare(resume=True,history_ref=pin)

    def test_missing_full_support_q_cannot_be_forged_by_probability_callback(self):
        p,e=self.prepare();plan=self.first.evaluate
        from evcs_experiments.first_stage import FirstStagePlan
        candidate=FirstStagePlan(np.zeros(3),np.zeros((3,2)),self.fixture.tasks.task_sha256,self.fixture.tasks.config_hash)
        q=[solve_fixed_plan(t,candidate).objective_usd+c for t,c in zip(self.templates,p.scenario_cost_constants_usd)]
        result=solve_worst_distribution(p.ambiguity,q,q_support_ids=p.ambiguity.support_ids)
        with self.assertRaisesRegex(ValueError,'incomplete'):e.sink_factory(self.root/'call')('probability',1,plan=candidate,result=result)
        event=json.loads(Path(e.events[-1]['path']).read_text())
        self.assertEqual(event['status'],'FAILED_EVIDENCE_CHECK')
        self.assertTrue((Path(e.events[-1]['path']).parent/'solver.json').exists())

    def test_unknown_event_overwrite_and_deleted_callback_history_rejected(self):
        p,e=self.prepare();sink=e.sink_factory(self.root/'call')
        with self.assertRaises(ValueError):sink('silently_drop_this',1)
        with self.assertRaises(FileExistsError):self.prepare()
        sink('phase1',0,result=solve_phase1(p.ambiguity))
        self.snapshot_ref=e.snapshot(self.root/'accepted_snapshot.json')
        Path(e.events[0]['path']).unlink()
        with self.assertRaises(FileNotFoundError):self.prepare(resume=True)

    def test_failed_original_physical_check_retains_raw_solution_and_failure(self):
        p,e=self.prepare()
        with patch('evcs_experiments.formal_planning.check_first_stage',return_value={'passed':False,'reason':'injected independent failure'}):
            with self.assertRaisesRegex(ValueError,'first-stage check'):
                run_ccg(p,evidence_sink=e.sink_factory(self.root/'call'),wall_seconds=10)
        last=Path(e.events[-1]['path']);row=json.loads(last.read_text())
        self.assertEqual(row['status'],'FAILED_EVIDENCE_CHECK')
        self.assertTrue((last.parent/'matrix.npz').exists());self.assertTrue((last.parent/'solution.npz').exists())
        self.assertTrue((last.parent/'failure.json').exists())

    def test_factory_retains_full_ordered160_support_without_eager_templates(self):
        # This tests orchestration against a synthetic bound storage fixture.
        # PhysicalBank's real formal arrays/namespace are covered independently;
        # no production atom is relabelled or optimized here.
        from types import SimpleNamespace
        from contextlib import ExitStack
        from evcs_experiments.config import get_paths
        from evcs_experiments.random_streams import canonical_hash
        from evcs_experiments.joint_scenarios import array_hash
        config=json.loads((get_paths().project_root/'configs/formal_experiments.json').read_text())
        targets=config['target_ids'];target='michael_2018'
        def write(name,value):
            path=self.root/name;path.write_text(json.dumps(value));return file_reference(path)
        source_file=self.root/'frozen_source.py';source_file.write_text('# synthetic source closure fixture')
        source=write('RELEASE_MANIFEST.json',dict(files=[dict(path=source_file.name,sha256=file_reference(source_file)['sha256'])]))
        protocol=write('protocol.json',dict(payload=dict(domain='formal',target_ids=targets,configuration=config)))
        acceptance=write('acceptance.json',dict(status='PASS'))
        task=write('tasks.json',dict(vehicle_summaries=[{}]*80,task_count=3))
        phi=np.array([.2,.1,.1,.1,1.]);arraypath=self.root/'features.npz';np.savez_compressed(arraypath,features=phi)
        diag=write('diagnostics.json',{});rows=[]
        for v in range(20):
            for base in targets:
                sid=f'{target}/formal/planning/{base}/v{v+1:03d}/R1'
                ledger=dict(scenario_id=sid,array_hashes=dict(features=array_hash(phi)))
                ledger['scenario_sha256']=canonical_hash(ledger)
                lr=write(f'ledger_{len(rows)}.json',ledger)
                mr=write(f'manifest_{len(rows)}.json',dict(scenario_id=sid,scenario_sha256=ledger['scenario_sha256'],
                    files=dict(arrays=file_reference(arraypath),ledger=lr,diagnostics=diag)))
                rows.append(dict(scenario_id=sid,scenario_sha256=ledger['scenario_sha256'],manifest=mr))
        index=write('planning.index.json',dict(task_source=task,split='planning',scenarios=rows,assignment_state_sha256='f'*64))
        moment=write('moments.json',dict(status='PASS_MOMENT_CALIBRATION',formal_OOS=False,
            bounds_by_rho={'1.0':dict(rho=1.,lower=[0,0,0,0,.91],upper=[1,1,1,1,1.09],feasibility=dict(valid_witness=True))}))
        summary=write('summary.json',dict(status='COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT',
            atom_count=1920,formal_OOS=False,source_manifest=source,formal_protocol_lock=protocol,
            task_sources={t:task for t in targets},targets=[dict(target_event=t,H=3,planning_count=160,calibration_count=80,
                planning_index=index,calibration_index=index,moments=moment,bootstrap=file_reference(arraypath)) for t in targets]))
        audit=write('audit.json',dict(status='PASS_INDEPENDENT_FORMAL_BANK_AUDIT',source_summary_sha256=summary['sha256'],
            source_output=str(self.root),formal_OOS=False,formal_plans_solved=False,scenario_count=1920,
            source_manifest_sha256=source['sha256'],formal_protocol_lock=protocol,s01_acceptance=acceptance,s04_acceptance=acceptance))
        bank=SimpleNamespace(scenario_ids=tuple(r['scenario_id'] for r in rows),
            scenario_sha256s=tuple(r['scenario_sha256'] for r in rows),load=lambda i:(_ for _ in ()).throw(AssertionError('eager scene load')))
        tasks=SimpleNamespace(**vars(self.fixture.tasks));tasks.config={**tasks.config,'basic_soc':.3}
        with ExitStack() as stack:
            stack.enter_context(patch('evcs_experiments.formal_planning.load_physical_case',return_value=self.fixture.case))
            stack.enter_context(patch('evcs_experiments.formal_planning.load_frozen_tasks',return_value=tasks))
            stack.enter_context(patch('evcs_experiments.formal_planning.FormalPhysicalBank',return_value=bank))
            stack.enter_context(patch('evcs_experiments.formal_planning.assemble_recourse',side_effect=AssertionError('eager matrix assembly')))
            for method in ('EA','ARO'):
                prepared=load_formal_problem(summary_reference=summary,audit_reference=audit,target_id=target,
                    method_id=method,evidence_root=self.root/method)
                self.assertEqual(len(prepared.problem.recourses),160)
                self.assertEqual(prepared.problem.recourses.capacity,2)
                self.assertEqual(prepared.problem.recourses.loads,0)
                self.assertEqual(prepared.problem.probability_set.support_ids,bank.scenario_ids)
                np.testing.assert_array_equal(prepared.problem.probability_set.features,np.tile(phi,(160,1)))
                self.assertEqual(len(prepared.problem.ambiguity.feature_ids),5 if method=='EA' else 0)
                self.assertGreater(prepared.preparation_seconds,0)
            source_file.write_text('source mutated')
            with self.assertRaisesRegex(ValueError,'source bytes changed'):
                load_formal_problem(summary_reference=summary,audit_reference=audit,target_id=target,
                    method_id='EA',evidence_root=self.root/'mutated')

    def test_factory_rejects_unsupported_method_and_unbound_or_failed_audit_before_physics(self):
        summary=self.root/'summary.json';summary.write_text('{}');audit=self.root/'audit.json'
        for status,digest in [('FAIL_INDEPENDENT_FORMAL_BANK_AUDIT',file_reference(summary)['sha256']),
                              ('PASS_INDEPENDENT_FORMAL_BANK_AUDIT','0'*64)]:
            audit.write_text(json.dumps(dict(status=status,source_summary_sha256=digest,source_output=str(self.root))))
            with patch('evcs_experiments.formal_planning.load_physical_case',side_effect=AssertionError('no physical access')):
                with self.assertRaises(ValueError):load_formal_problem(summary_reference=file_reference(summary),audit_reference=file_reference(audit),
                    target_id='michael_2018',method_id='EA',evidence_root=self.root/'evidence')
        with self.assertRaises(ValueError):load_formal_problem(summary_reference={},audit_reference={},target_id='x',method_id='SAA',evidence_root=self.root/'evidence')


if __name__=='__main__':unittest.main()
