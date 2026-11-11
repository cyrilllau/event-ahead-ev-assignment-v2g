"""Development fixtures for template reuse and new fixed-plan Q checkpoints."""
import importlib.util
import os
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.random_streams import canonical_hash
p=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))/'scripts/diagnostics/run_s05_candidate_evaluation.py'
s=importlib.util.spec_from_file_location('candidate_component',p);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)


def first_stage():
    case=SimpleNamespace(periods=(1,2),station_ids=('CS1',),station_bus_ids=(2,),station_plugs=np.array([1]),station_interface_kw=np.array([10.]),dt_h=.5)
    tasks=SimpleNamespace(task_ids=('task',),vehicle_ids=('ev',),station_ids=('CS1',),chi_occ=np.ones((1,2)),chi_res=np.ones((1,2)),block_mask=np.ones((1,2)),p_dis_kw=np.array([4.]),p_block_kw=np.array([2.]),eta_dis=np.array([.8]),e_forecast_dc_kwh=np.array([3.]),c_act_usd=np.array([5.]),c_cap_usd_per_kwh=np.array([[.1,.2]]),task_sha256='tasks',config_hash='config')
    return case,tasks,assemble_first_stage(case,tasks)


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.case,self.tasks,self.first=first_stage();self.ids=tuple('s'+str(i) for i in range(160));self.sha=tuple('scene'+str(i) for i in range(160))
        old=self.root/'old';old.mkdir();history=[]
        for i in range(2):history.append(d.write(old/f'prefix{i}.json',{}))
        self.artifacts=[];self.templates=[]
        for i in range(160):
            td=old/f't{i:03d}';td.mkdir();a=d.write(td/'identity.json',{'frozen':i});self.artifacts.append(a)
            tr=d.write(td/'record.json',dict(kind='recourse_template',scenario_index=i,scenario_id=self.ids[i],template_sha256='template'+str(i),artifacts=[a]))
            pointer=d.write(td/'template_reference.json',tr)
            history.append(d.write(td/'event.json',dict(event='recourse',scenario_index=i,scenario_id=self.ids[i],artifacts=[pointer])))
            self.templates.append(SimpleNamespace(template_sha256='template'+str(i),scenario_sha256=self.sha[i],first_stage=self.first))
        history.append(d.write(old/'probability.json',{}));hr=d.write(old/'history.json',history)
        identity=dict(first_stage_sha256=self.first.matrix_sha256,scenario_ids=list(self.ids))
        state=dict(identity=identity,problem_sha256=canonical_hash(identity),pending=dict(recourse_certificates=[dict(template_sha256=t.template_sha256) for t in self.templates]))
        cp=d.write(old/'checkpoint.json',dict(state=state,sha256=canonical_hash(state)))
        card=d.write(old/'acceptance.json',dict(status='PASS_SAVED160_COMPONENT_REVIEW',complete_lp_count=160,references=dict(history=hr,checkpoint=cp),problem_sha256=state['problem_sha256']))
        er=self.root/'raw';er.mkdir();d.write(er/'history.json',[])
        evidence=SimpleNamespace(root=er,ids=self.ids,events=[],domain='development')
        evidence.template=lambda *args:(_ for _ in ()).throw(AssertionError('Do not export new identity'))
        class Cache:
            scenario_sha256s=self.sha
            def __getitem__(cache,i):evidence.template(i,self.templates[i]);return self.templates[i]
        problem=SimpleNamespace(identity=identity,first_stage=self.first,recourses=Cache(),probability_set=SimpleNamespace(support_ids=self.ids),scenario_cost_constants_usd=np.ones(160))
        evidence.problem=problem
        def sink_factory(folder):
            Path(folder).mkdir(parents=True,exist_ok=True)
            def sink(event,iteration,*,index,plan,result):
                evidence.template(index,self.templates[index])
                ref=d.write(er/f'event{len(evidence.events):03d}.json',dict(scenario_index=index,plan_sha256=canonical_hash(d.serial_plan(plan)),q_usd=result.objective_usd+1,optimal_certificate_valid=result.optimal))
                evidence.events.append(ref);d.atomic_json(er/'history.json',evidence.events)
            return sink
        evidence.sink_factory=sink_factory
        self.prepared=SimpleNamespace(case=self.case,tasks=self.tasks,problem=problem,evidence=evidence)
        self.adapter=d.ReusedTemplateEvidenceAdapter(evidence,old_history_reference=hr,old_checkpoint_reference=cp,saved160_acceptance_reference=card,output=self.root/'reuse').install()
        result=d.write(self.root/'native_result.json',{'status':'native development fixture'})
        vector=self.root/'vector.npz';np.savez_compressed(vector,vector=np.array([1.,2.,2.,99.]))
        self.binding=d.write(self.root/'candidate_binding.json',dict(result_reference=result,vector_reference=d.reference(vector),vector_key='vector',first_stage_slice=[0,3],first_stage_sha256=self.first.matrix_sha256))
        self.plan=d.extract_candidate(self.binding,self.first);self.calls=[]
    def hooks(self,fail=None):
        def solve(template,plan,time_limit_seconds):
            i=self.templates.index(template);self.calls.append(i)
            return SimpleNamespace(optimal=i!=fail,certificate={'optimal_certificate_valid':i!=fail},objective_usd=10.+i)
        return {'check_first_stage':lambda c,t,p:{'passed':True},'solve':solve}
    def run(self,**kw):return d.evaluate_prepared_candidate(self.prepared,self.plan,candidate_binding_reference=self.binding,output=self.root/'eval',deadline=time.monotonic()+60,_development_hooks=self.hooks(),**kw)


class CandidateEvaluationTests(unittest.TestCase):
    def test_reuse_verified_once_without_new_template_identity_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            with patch.object(d,'verify',wraps=d.verify) as verify:
                a=f.adapter.template(0,f.templates[0]);n=verify.call_count
                self.assertEqual(f.adapter.template(0,f.templates[0]),a);self.assertEqual(verify.call_count,n)
            self.assertEqual(f.adapter.inventory()['new_identity_json_exports'],0)
            self.assertFalse(f.adapter.inventory()['old_q_reused'])

    def test_changed_original_artifact_or_new_template_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);Path(f.artifacts[0]['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'reference'):f.adapter.template(0,f.templates[0])
            f.templates[1].template_sha256='changed'
            with self.assertRaisesRegex(ValueError,'template differs'):f.adapter.template(1,f.templates[1])

    def test_source_bound_candidate_has_no_rounding_or_plan_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);self.assertEqual(f.plan.x.tolist(),[1.]);self.assertEqual(f.plan.r.tolist(),[[2.,2.]])
            b=d.read(f.binding);v=Path(tmp)/'fractional.npz';np.savez_compressed(v,vector=np.array([.5,1.,1.]))
            b['vector_reference']=d.reference(v);br=d.write(Path(tmp)/'fractional_binding.json',b)
            with self.assertRaisesRegex(ValueError,'integer first-stage'):d.extract_candidate(br,f.first)
            v.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'reference'):d.extract_candidate(br,f.first)

    def test_full160_pass_produces_new_q_and_no_probability_or_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);r=f.run()
            self.assertEqual(f.calls,list(range(160)));self.assertEqual(r['completed_count'],160)
            self.assertEqual(r['q_usd'],list(np.arange(160)+11.));self.assertEqual(len(r['recourse_event_references']),160)
            self.assertEqual(r['status'],'FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY')
            self.assertFalse(r['probability_evaluated']);self.assertFalse(r['full_support_verified']);self.assertFalse(r['automatic_resume_supported'])
            self.assertEqual(len(f.adapter.used),160);self.assertTrue((Path(tmp)/'eval/full_q.json').is_file())

    def test_failed_new_lp_stops_and_preserves_completed_q_without_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            with self.assertRaisesRegex(ValueError,'optimal certificate'):
                d.evaluate_prepared_candidate(f.prepared,f.plan,candidate_binding_reference=f.binding,output=Path(tmp)/'eval',deadline=time.monotonic()+60,_development_hooks=f.hooks(fail=3))
            cp=json.loads((Path(tmp)/'eval/checkpoint.json').read_text());self.assertEqual(cp['status'],'FAILED_RECOURSE')
            self.assertEqual(cp['completed_count'],3);self.assertEqual(cp['q_usd'][:4],[11.,12.,13.,None]);self.assertEqual(f.calls,[0,1,2,3])
            self.assertFalse((Path(tmp)/'eval/full_q.json').exists())

    def test_exhausted_clock_checkpoints_without_calling_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);r=d.evaluate_prepared_candidate(f.prepared,f.plan,candidate_binding_reference=f.binding,output=Path(tmp)/'eval',deadline=time.monotonic()-1,_development_hooks=f.hooks())
            self.assertEqual(r['status'],'TIME_LIMIT');self.assertEqual(r['completed_count'],0);self.assertEqual(f.calls,[])

    def test_changed_prepared_model_identity_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);f.prepared.problem.identity=dict(f.prepared.problem.identity,changed_ambiguity=True)
            with self.assertRaisesRegex(ValueError,'model identity changed'):f.adapter.template(0,f.templates[0])

    def test_development_hooks_cannot_enter_formal_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);f.prepared.evidence.domain='formal'
            with self.assertRaisesRegex(ValueError,'No formal dependency injection'):f.run()
            self.assertEqual(f.calls,[])

if __name__=='__main__':unittest.main()
