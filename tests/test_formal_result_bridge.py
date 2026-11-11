"""No solver: truthful status, exact budget and real append-only gate fixtures."""
import copy,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference,verify_formal_numerical_records
from evcs_experiments.random_streams import canonical_hash
P=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))/'src/evcs_experiments/formal_result_bridge.py'
spec=importlib.util.spec_from_file_location('evcs_experiments.formal_result_bridge_tested',P)
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


def fixture():
    refs={k:dict(path='/tmp/'+k+'.json',sha256='0'*64,size_bytes=0) for k in bridge.REQUIRED}
    ident=dict(plan_id='mechanism.michael_2018.EA_DRO.baseline',target_id='michael_2018',method_id='EA_DRO',setting_id='baseline')
    pb=dict(cumulative_wall_seconds=7200,single_master_seconds=600,final_certificate_reserve_seconds=1790,shutdown_reserve_seconds=10,normal_gap_relative=.005)
    protocol=dict(domain='formal',configuration=dict(planning_budget=pb),batches=[dict(batch_id='paired',expected_plans=[ident,dict(ident,plan_id='other',method_id='SAA')])])
    plan=dict(x=[1.,0.],r=[[0.],[0.]],task_sha256='t',config_hash='c');ph=canonical_hash(plan)
    identity=dict(scenario_ids=['s'+str(i) for i in range(160)],method='EA');problem=canonical_hash(identity)
    state=dict(identity=identity,problem_sha256=problem,status='PROBABILITY_FAILED',pending=dict(rmp=dict(status='TIME_LIMIT',optimal=False,lower_bound_usd=None)))
    D=dict(protocol=dict(kind='protocol_lock',payload=protocol,payload_sha256=canonical_hash(protocol)),candidate_plan=plan,
        original_checkpoint=dict(state=state,sha256=canonical_hash(state)),original_raw_request=dict(domain='formal',problem_identity=identity))
    D['recourse_audit']=dict(status='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT',passed=True,complete_lp_count=160,support_ids=identity['scenario_ids'],q_usd=[10.]*160,Fpre_usd=2.,original_problem_sha256=problem,plan_sha256=ph)
    D['probability_component']=dict(probability_independent_check_passed=True,complete_lp_count=160,first_stage_cost_usd=2.,worst_recourse_upper_bound_usd=98.,upper_bound_usd=100.,original_problem_sha256=problem,plan_sha256=ph)
    D['completion_result']=dict(status='FULL_SUPPORT_NUMERICAL_COMPLETION_PENDING_FORMAL_PLAN_LOCK',full_support_verified=True,recourse_independent_check_passed=True,probability_independent_check_passed=True,complete_lp_count=160,recourse_audit=refs['recourse_audit'],probability_result=refs['probability_component'],upper_bound_usd=100.,original_problem_sha256=problem,plan_sha256=ph)
    D['auxiliary_result']=dict(lower_bound_usd=20.,problem_sha256=problem,candidate_plan_sha256=ph)
    D['auxiliary_review']=dict(status='PASS_INDEPENDENT_EPIGRAPH_AUXILIARY_RESULT_REVIEW',passed=True,references=dict(result=refs['auxiliary_result']),numerical_lower_bound_usd=20.,original_problem_sha256=problem,candidate_plan_sha256=ph,native_status='OPTIMAL')
    D['previous_zero_probability_acceptance']=dict(status='PASS_PROBABILITY_RECOVERY_COMPONENT_REVIEW',probability_certificate_PASS=True,problem_sha256=problem,upper_bound_usd=105.)
    D['settlement']=dict(kind='S05_CANDIDATE_COMPLETION_PARENT_ACCOUNTING_V1',status='SETTLED_NUMERICAL_CANDIDATE_PENDING_FORMAL_PLAN_LOCK',accounting_PASS=True,parent_charged_once=True,cumulative_seconds_provisional=False,nested_seconds_added=False,opening_cumulative_seconds_decimal='5337.02842836330092',terminal_parent_seconds_decimal='101.795211',cumulative_accounted_seconds_decimal='5438.82363936330092',remaining_seconds_decimal='1761.17636063669908',cumulative_accounted_seconds=5438.823639363301,remaining_seconds=1761.176360636699,cumulative_cap_seconds=7200,original_cumulative_cap_seconds=7200,plan_id=ident['plan_id'],references=dict(completion_result=refs['completion_result']))
    D['completion_review']=dict(status='PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW',passed=True,scientific_candidate_complete=True,references=refs.copy(),original_problem_sha256=problem,plan_sha256=ph,lower_bound_usd=20.,upper_bound_usd=100.,previous_zero_upper_bound_usd=105.)
    sources=[file_reference(P)]
    D['adapter_decision']=dict(status='APPROVED_FORMAL_RESULT_BRIDGE',protocol_reference=refs['protocol'],original_problem_sha256=problem,candidate_plan_sha256=ph,auxiliary_bound_allowed=True,ccg_convergence_claimed=False,formal_OOS=False,references=refs.copy(),planning_source=sources)
    return dict(batch_id='paired',identity=ident,references=refs,planning_source=sources),D


class BridgeTests(unittest.TestCase):
    def test_truthful_time_limit_and_unspent_reserve(self):
        request,D=fixture();n,b=bridge.build_records(request,D)
        self.assertEqual(n['status'],'TIME_LIMIT');self.assertEqual(n['stop_reason'],bridge.REASON)
        self.assertFalse(n['budget_exhausted']);self.assertFalse(n['full_CCG_converged']);self.assertFalse(n['normal_gap_met'])
        self.assertIsNone(n['original_native_master_lower_bound_usd']);self.assertEqual(n['auxiliary_master_status'],'OPTIMAL')
        self.assertEqual(n['gap_relative'],.8);self.assertEqual(n['candidate_improvement_over_zero_usd'],5.)
        self.assertEqual(n['remaining_wall_seconds_decimal'],'1761.17636063669908');self.assertEqual(n['complete_iteration_reserve_seconds_decimal'],'2410')
        self.assertEqual(b['planning_budget'],D['protocol']['payload']['configuration']['planning_budget'])
    def test_bad_scientific_evidence(self):
        cases=[('recourse_audit','complete_lp_count',159),('probability_component','original_problem_sha256','foreign'),('completion_result','plan_sha256','foreign'),('auxiliary_review','passed',False),('adapter_decision','formal_OOS',True),('completion_review','scientific_candidate_complete',False)]
        for name,key,value in cases:
            with self.subTest(name=name,key=key):
                r,D=fixture();D[name][key]=value
                with self.assertRaises(ValueError):bridge.build_records(r,D)
    def test_worse_candidate_and_missing_native_bound_not_relabelled(self):
        r,D=fixture();D['previous_zero_probability_acceptance']['upper_bound_usd']=90.;D['completion_review']['previous_zero_upper_bound_usd']=90.
        with self.assertRaises(ValueError):bridge.build_records(r,D)
        r,D=fixture();D['original_checkpoint']['state']['pending']['rmp']['lower_bound_usd']=20.
        D['original_checkpoint']['sha256']=canonical_hash(D['original_checkpoint']['state'])
        with self.assertRaises(ValueError):bridge.build_records(r,D)
    def test_budget_cannot_reset_or_double_charge(self):
        for key,val in [('parent_charged_once',False),('nested_seconds_added',True),('opening_cumulative_seconds_decimal','0'),('cumulative_cap_seconds',8000),('cumulative_seconds_provisional',True)]:
            with self.subTest(key=key):
                r,D=fixture();D['settlement'][key]=val
                with self.assertRaises(ValueError):bridge.build_records(r,D)
    def test_room_for_complete_iteration_rejects_this_stop_reason(self):
        r,D=fixture();s=D['settlement'];s.update(opening_cumulative_seconds_decimal='3900',terminal_parent_seconds_decimal='100',cumulative_accounted_seconds_decimal='4000',remaining_seconds_decimal='3200',cumulative_accounted_seconds=4000.,remaining_seconds=3200.)
        with self.assertRaises(ValueError):bridge.build_records(r,D)
    def test_changed_review_reference_rejected(self):
        r,D=fixture();D['completion_review']['references']['candidate_plan']=dict(r['references']['candidate_plan'],sha256='f'*64)
        with self.assertRaises(ValueError):bridge.build_records(r,D)
    def test_real_export_and_existing_gate_without_paired_group_lock(self):
        r,D=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'original';inputs.mkdir();op=root/'op';op.mkdir()
            def put(name,doc,directory=inputs):
                p=directory/(name+'.json');p.write_text(json.dumps(doc));return file_reference(p)
            taskref=put('tasks',dict(task_sha256='t',tasks=[],queries=[],vehicle_summaries=[]));supportref=put('support',dict(ids=D['recourse_audit']['support_ids']))
            legacy=dict(path=taskref['path'],sha256=taskref['sha256'],bytes=taskref['size_bytes'])
            D['original_raw_request']['input_binding']=dict(task_source=legacy,target=dict(planning_index=supportref))
            r['plan_input_references']=dict(tasks=taskref,routes=taskref,initial_state=taskref,planning_support=supportref)
            order=['protocol','candidate_plan','original_checkpoint','original_raw_request','recourse_audit','probability_component','auxiliary_result','auxiliary_review','previous_zero_probability_acceptance','completion_result','settlement','completion_review','adapter_decision']
            for name in order:
                if name=='auxiliary_review':D[name]['references']['result']=r['references']['auxiliary_result']
                if name=='completion_result':D[name].update(recourse_audit=r['references']['recourse_audit'],probability_result=r['references']['probability_component'])
                if name=='settlement':D[name]['references']['completion_result']=r['references']['completion_result']
                if name in ('completion_review','adapter_decision'):D[name]['references']=r['references'].copy()
                if name=='adapter_decision':D[name]['protocol_reference']=r['references']['protocol']
                r['references'][name]=put(name,D[name],op if name in ('settlement','completion_review','adapter_decision') else inputs)
            before={p:p.read_bytes() for p in root.rglob('*.json')}
            out=op/'prepared';result=bridge.prepare_formal_result(r,output=out)
            record=json.loads(Path(result['plan_record_reference']['path']).read_text())
            verify_formal_numerical_records(D['protocol']['payload'],{'plans':[record]})
            from evcs_experiments.evaluation_batch import read_locked_plan
            loaded,metadata=read_locked_plan(record)
            self.assertEqual(metadata['execution_mode'],'EA');self.assertEqual(record['plan_sha256'],canonical_hash(D['candidate_plan']))
            self.assertFalse(result['formal_plan_locked']);self.assertFalse(result['formal_OOS']);self.assertEqual(result['solver_calls'],0)
            self.assertEqual({p.name for p in out.iterdir()},{'plan.json','solver_record.json','planning_budget.json','plan_record.json','artifact_manifest.json'})
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))
            with self.assertRaises(FileExistsError):bridge.prepare_formal_result(r,output=out)
            Path(taskref['path']).write_text('{}')
            with self.assertRaises(ValueError):bridge.prepare_formal_result(r,output=op/'tampered')

if __name__=='__main__':unittest.main()
