"""Synthetic saved-record fixtures; only the tiny probability LP is real."""
import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
p=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))/'scripts/diagnostics/run_s05_candidate_probability.py'
s=importlib.util.spec_from_file_location('candidate_probability_component',p);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)


class Fixture:
    def __init__(self,path):
        self.root=Path(path);self.refs={};self.serial=0
        def save(name,data):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);return d.write(p,data)
        self.save=save
        ids=['development-synthetic-'+str(i) for i in range(160)]
        P=d.probability.ProbabilitySet(tuple(ids),('f',),np.zeros((160,1)),np.array([0.]),np.array([0.]))
        identity=dict(first_stage_sha256='first',scenario_ids=ids,input_ambiguity_sha256=P.ambiguity_sha256,effective_ambiguity_sha256=P.ambiguity_sha256,constants_usd=[2.]*160,risk_model='DRO')
        state=dict(identity=identity,problem_sha256=d.canonical_hash(identity),pending=dict(recourse_certificates=[dict(template_sha256='t'+str(i)) for i in range(160)]))
        self.refs['original_checkpoint']=save('old/checkpoint.json',dict(state=state,sha256=d.canonical_hash(state)))
        self.refs['original_probability_input']=save('old/probability_input.json',dict(input_ambiguity=P.as_dict(),effective_ambiguity=P.as_dict()))
        oldevent=save('old/probability.json',dict(event='probability',artifacts=[self.refs['original_probability_input']]))
        self.refs['original_history']=save('old/history.json',[oldevent]*163)
        self.refs['saved160_acceptance']=save('old/card.json',dict(status='PASS_SAVED160_COMPONENT_REVIEW',complete_lp_count=160,problem_sha256=state['problem_sha256'],references=dict(checkpoint=self.refs['original_checkpoint'],history=self.refs['original_history'])))
        plan=dict(x=[1.],r=[[2.,2.]],task_sha256='tasks',config_hash='config');ph=d.canonical_hash(plan);planref=save('plan.json',plan)
        np.savez_compressed(self.root/'vector.npz',vector=np.array([1.,2.,2.,99.]))
        native=save('native.json',dict(valid_incumbent=True,certificate_failures=[]));req=save('request.json',{'development_fixture':True})
        binding=dict(first_stage_sha256='first',problem_sha256=state['problem_sha256'],vector_reference=d.reference(self.root/'vector.npz'),vector_key='vector',first_stage_slice=[0,3],plan_sha256=ph,source_request=req,source_native_result=native,formal_OOS=False,lockable=False)
        self.refs['original_binding']=save('binding.json',binding)
        self.refs['candidate_result']=save('candidate_result.json',dict(candidate_available=True,candidate_vector=binding['vector_reference'],request_reference=req,candidate_plan_sha256=ph,candidate_first_cost_usd=3.,problem_sha256=state['problem_sha256'],first_stage_sha256='first'))
        self.refs['candidate_review']=save('candidate_review.json',dict(status='PASS_INDEPENDENT_EPIGRAPH_AUXILIARY_RESULT_REVIEW',passed=True,references=dict(result=self.refs['candidate_result'],candidate_binding=self.refs['original_binding'],native_solver=native),candidate_plan_sha256=ph,original_problem_sha256=state['problem_sha256']))
        adapted=save('adapted.json',dict(binding,result_reference=native,original_binding_reference=self.refs['original_binding'],candidate_result_reference=self.refs['candidate_result'],candidate_review_reference=self.refs['candidate_review']))
        self.refs['first_stage_check']=save('first_stage_check.json',dict(passed=True,task_sha256='tasks',config_hash='config',costs=dict(Fpre_usd=3.)))
        self.refs['new_raw_request']=save('raw/request.json',dict(problem_identity=identity,domain='formal',formal_OOS=False))
        events=[];templates=[]
        # Synthetic physical/KKT records deliberately do not pretend to be real recourse solves.
        shared=[]
        for name in ('fixed_rhs.npz','primal_vector.npz','solution.npz','dual_kkt.npz'):
            np.savez_compressed(self.root/name,synthetic=np.array([0.]));shared.append(d.reference(self.root/name))
        for i in range(160):
            root=f'events/{i:03d}/';tr=save(root+'template.json',dict(scenario_id=ids[i],scenario_index=i,template_sha256='t'+str(i)))
            pointer=save(root+'template_reference.json',tr)
            artifacts=[pointer,save(root+'plan.json',plan),save(root+'physical_check.json',dict(passed=True)),save(root+'solver.json',dict(optimal=True,has_solution=True,objective_usd=float(i),certificate=dict(optimal_certificate_valid=True,primal_feasible_within_tolerance=True,template_sha256='t'+str(i),certificate_status='PASS_NUMERIC_LP_CERTIFICATE',optimal_cost_bound_valid=True,max_inequality_violation=0.,max_equality_violation=0.,max_bound_violation=0.,max_dual_stationarity_violation=0.,max_dual_sign_violation=0.,max_invalid_infinite_bound_marginal=0.,dual_objective_usd=float(i))))]+shared
            events.append(save(root+'event.json',dict(event='recourse',domain='formal',scenario_index=i,scenario_id=ids[i],plan_sha256=ph,q_usd=float(i+2),optimal_certificate_valid=True,artifacts=artifacts)))
            templates.append(dict(index=i,scenario_id=ids[i],all_original_artifacts_hashed=True,template_reference=tr))
        full=dict(status='FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY',completed_count=160,next_index=160,error=None,formal_OOS=False,probability_evaluated=False,plan_reference=planref,plan_sha256=ph,candidate_binding_reference=adapted,
            template_reuse_inventory=dict(references=dict(old_checkpoint=self.refs['original_checkpoint'],old_history=self.refs['original_history'],saved160_acceptance=self.refs['saved160_acceptance']),old_q_reused=False,templates=templates),support_ids=ids,q_usd=list(np.arange(160,dtype=float)+2),recourse_event_references=events,raw_history_reference=save('raw/history.json',events))
        self.full=full;self.refs['full_q']=save('full_q.json',full);self.refs['checkpoint']=save('checkpoint.json',full)
        self.refs['evaluation_result']=save('evaluation_result.json',dict(request_reference=req,formal_OOS=False,checkpoint_reference=self.refs['checkpoint'],completed_count=160,original_candidate_binding=self.refs['original_binding'],plan_sha256=ph,problem_sha256=state['problem_sha256']))
        self.refs['execution']=save('execution.json',dict(request_reference=req,probability_calls=0,master_calls=0,formal_OOS=False,source_references=dict(old_checkpoint=self.refs['original_checkpoint'],candidate_binding=self.refs['original_binding'],saved160_acceptance=self.refs['saved160_acceptance']),plan_sha256=ph,original_problem_sha256=state['problem_sha256'],candidate_binding_reference=adapted))
        for key,path in [('probability_source',d.probability.__file__),('checker_source',d.probability_check.__file__),('driver_source',d.__file__)]:self.refs[key]=d.reference(path)
    def full_change(self,change):
        self.serial+=1;change(self.full)
        self.refs['full_q']=self.save(f'full{self.serial}.json',self.full);self.refs['checkpoint']=self.save(f'cp{self.serial}.json',self.full)
        result=d.read(self.refs['evaluation_result']);result['checkpoint_reference']=self.refs['checkpoint'];self.refs['evaluation_result']=self.save(f'eval{self.serial}.json',result)
    def run(self):return d.complete_candidate_probability(self.refs,output=self.root/'out')


class CandidateProbabilityTests(unittest.TestCase):
    def test_real_probability_full160_original_constants_once_and_pending_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);r=f.run()
            self.assertEqual(r['upper_bound_usd'],164.) # max q=159+2, plus Fpre=3
            self.assertTrue(r['probability_independent_check_passed']);self.assertIsNone(r['lower_bound_usd'])
            for k in ('lockable','scientific_acceptance','full_support_verified','formal_OOS','original_matrix_independent_reaudit_performed'):self.assertFalse(r[k])
            self.assertTrue(r['parent_budget_settlement_required'])
    def test_partial_or_reordered_full_q_rejected_before_solver(self):
        for change in (lambda r:r.update(completed_count=159),lambda r:r['support_ids'].reverse()):
            with tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);f.full_change(change)
                with patch.object(d.probability,'solve_worst_distribution') as solve:
                    with self.assertRaises(ValueError):f.run()
                    solve.assert_not_called()
    def test_changed_artifact_and_signed_wrong_constant_rejected(self):
        for mode in ('artifact','constant'):
            with tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp)
                if mode=='artifact':(Path(tmp)/'events/000/physical_check.json').write_text('{}')
                else:f.full_change(lambda r:r['q_usd'].__setitem__(0,4.))
                with patch.object(d.probability,'solve_worst_distribution') as solve:
                    with self.assertRaises(ValueError):f.run()
                    solve.assert_not_called()
    def test_resigned_failed_raw_certificate_and_double_added_constant_rejected(self):
        for mode in ('dual','physical','constant'):
            with tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);event=d.read(f.full['recourse_event_references'][0])
                if mode=='constant':
                    event['q_usd']=4.;f.full['q_usd'][0]=4.
                else:
                    name='solver.json' if mode=='dual' else 'physical_check.json'
                    old=d.artifact(event,name);value=d.read(old)
                    if mode=='dual':value['certificate']['max_dual_stationarity_violation']=1e-6
                    else:value['passed']=False
                    new=f.save('revised/'+name,value);event['artifacts']=[new if r==old else r for r in event['artifacts']]
                f.full['recourse_event_references'][0]=f.save('resigned_event.json',event)
                f.full['raw_history_reference']=f.save('raw/revised_history.json',f.full['recourse_event_references'])
                f.full_change(lambda r:None)
                with patch.object(d.probability,'solve_worst_distribution') as solve:
                    with self.assertRaises(ValueError):f.run()
                    solve.assert_not_called()
    def test_old_probability_source_ref_and_native_alias_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);f.refs['probability_source']=f.save('old_probability.py',{'old_solver':True})
            with self.assertRaisesRegex(ValueError,'old R4'):f.run()
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);b=d.read(f.refs['original_binding']);b['result_reference']=b.pop('source_native_result')
            f.refs['original_binding']=f.save('wrongbinding.json',b)
            with self.assertRaises(ValueError):f.run()
    def test_probability_checker_failure_never_exports_ub(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            from types import SimpleNamespace
            failed=SimpleNamespace(passed=False,as_dict=lambda:{'passed':False})
            with patch.object(d.probability_check,'check_probability_solution',return_value=failed):r=f.run()
            self.assertEqual(r['status'],'PROBABILITY_FAILED');self.assertIsNone(r['upper_bound_usd'])
    def test_time_cap_rejects_boolean_or_unbounded(self):
        for value in (True,31,float('inf'),0):
            with self.assertRaisesRegex(ValueError,'time cap'):d.complete_candidate_probability({},output='unused',time_limit_seconds=value)

if __name__=='__main__':unittest.main()
