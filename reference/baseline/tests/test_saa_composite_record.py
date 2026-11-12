import copy
import unittest
from evcs_experiments.saa_composite_record import build_saa_composite_record, REQUIRED, PINNED
from evcs_experiments.random_streams import canonical_hash


def fixture():
    refs={k:dict(path='/evidence/'+k,sha256=PINNED.get(k,k),size_bytes=1) for k in REQUIRED}
    ident=dict(plan_id='risk_model.michael_2018.SAA.baseline',target_id='michael_2018',method_id='SAA',setting_id='risk_objective=baseline')
    pb=dict(cumulative_wall_seconds=7200,normal_gap_relative=.005)
    p=dict(domain='formal',configuration=dict(planning_budget=pb),batches=[dict(batch_id='risk',expected_plans=[ident])])
    plan=dict(x=[1],r=[[2]],task_sha256='task',config_hash='config'); ph=canonical_hash(plan)
    sources=[dict(path='/old.py',sha256='old',size_bytes=1)]; adapters=[dict(path='/new.py',sha256='new',size_bytes=1)]
    taskref=dict(path='/tasks',sha256='tasks',size_bytes=1); support=dict(path='/support',sha256='support',size_bytes=1)
    inputs=dict(tasks=taskref,routes=taskref,initial_state=taskref,planning_support=support)
    request=dict(kind='SAA_SAME_PLAN_AUXILIARY_RECORD_REQUEST_V1',identity=ident,batch_id='risk',references=refs,
                 plan_input_references=inputs,planning_source=sources,adapter_sources=adapters)
    queue=1357.4711459395228; total=5683.899147049283; extra=total-queue
    lb=205364.35583154118; ub=225329.06402995915; oldlb=13690.39555159417; gap=(ub-lb)/ub
    old=dict(domain='formal',**{k:ident[k] for k in ('plan_id','target_id','method_id')},plan_sha256=ph,
        candidate_record_reference=refs['candidate'],request_reference=refs['original_worker_request'],
        numerical_audit_reference=refs['original_bridge'],candidate_audit_reference=refs['original_joint_audit'],
        full_support_verified=True,risk_certificate_kind='fixed_empirical',final_budget_acceptance=True,
        requires_seal_confirmation=True,actual_attempt_seconds_at_cutoff=20,charged_attempt_seconds=30,final_seal_allocated_seconds=10,
        status='TIME_LIMIT',native_master_status='TIME_LIMIT',raw_engine_status='TIME_LIMIT',lower_bound_usd=oldlb,
        upper_bound_usd=ub,gap_relative=(ub-oldlb)/ub,cumulative_wall_seconds=queue)
    seal=dict(status='PASS_FINAL_RECORD_SEAL',references=dict(solver_record=refs['original_solver'],budget=refs['original_budget']),
              actual_attempt_seconds_before_confirmation_write=21,charged_attempt_seconds=30,seal_allocation_seconds=10)
    binds=dict(final_solver_record_reference=refs['original_solver'],finalization_reference=refs['original_finalization'],
        candidate_audit_reference=refs['original_joint_audit'],numerical_bridge_reference=refs['original_bridge'],
        passed=True,settlement_reference=dict(path='/settlement6'))
    ids=list(range(160)); fixed=dict(support_ids=ids,empirical_probabilities=[1/160]*160)
    auxrefs=dict(result=refs['auxiliary_result'],settlement=refs['latest_settlement'],parent_result=dict(path='/parent8'))
    d=dict(protocol=dict(kind='protocol_lock',payload=p,payload_sha256=canonical_hash(p)),
      original_worker_request=dict(domain='formal',**{k:ident[k] for k in ('plan_id','target_id','method_id')},budget=pb,
          protocol_reference=refs['protocol'],planning_source_references=sources),
      candidate=dict(candidate=dict(plan=plan,plan_sha256=ph)),original_solver=old,
      original_budget=dict(planning_budget=pb,charged_attempt_seconds=30,final_budget_acceptance=True),original_finalization=seal,
      original_bridge=dict(candidate_record_reference=refs['candidate'],candidate_audit_reference=refs['original_joint_audit'],
          request_reference=refs['original_worker_request'],plan_sha256=ph),
      original_joint_audit=dict(plan_sha256=ph,status='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT',full_support_recourse_verified=True,
          upper_bound_usd=ub,fixed_q_audit=fixed),original_terminal_review=copy.deepcopy(binds),
      original_join=dict(binds,gate=dict(anchor_reference=refs['ledger_anchor'],candidate_reference=refs['candidate']),
          original_queue_charged_seconds=queue,effective_spent_seconds=5541.351921049283),
      latest_settlement=dict(terminal_job_result_reference=auxrefs['parent_result'],effective_spent_seconds=total,
          cumulative_extra_seconds=extra,closing_queue_seconds=queue,budget_seconds=7200),
      auxiliary_acceptance=dict(kind='SAA_AUXILIARY_BOUND_ACCEPTANCE_V1',passed=True,review_reference=refs['auxiliary_review'],
          references=auxrefs,plan_sha256=ph,lower_bound_usd=lb,upper_bound_usd=ub,gap_relative=gap,
          original_native_status='TIME_LIMIT',auxiliary_native_status='OPTIMAL',effective_spent_seconds=total,remaining_seconds=7200-total),
      auxiliary_review=dict(passed=True,references=auxrefs,original_physical_acceptance_inherited=refs['original_joint_audit'],
          all_160_raw_dual_stationarity_sign_gradient_intercept_checks=True,original_integer_domain_retained=True,
          full_support_count=160,plan_sha256=ph,lower_bound_usd=lb,upper_bound_usd=ub,gap=gap,auxiliary_native_status='OPTIMAL'),
      auxiliary_result=dict(full_support_count=160,original_plan_sha256=ph,source_binding=dict(candidate_audit=refs['original_joint_audit'],
          final_record=refs['original_solver'],final_budget_join=refs['original_join'],final_acceptance_review=refs['original_terminal_review']),
          lower_bound_usd=lb,upper_bound_usd=ub,previous_lower_bound_usd=oldlb,original_native_status='TIME_LIMIT',auxiliary_native_status='OPTIMAL'),
      ledger_checked=dict(latest_settlement=refs['latest_settlement'],pending=False,extra=extra),
      summary=dict(task_sources=dict(michael_2018=taskref),targets=[dict(target_event='michael_2018',planning_index=support)]),
      tasks=dict(task_sha256='task',tasks=[],queries=[],vehicle_summaries=[]),
      fixed_q_request=dict(scenario_ids=ids,scenario_sha256s=['s']*160,plan=plan,plan_sha256=ph),
      model_manifest=dict(scenario_ids=ids,scenario_sha256s=['s']*160,empirical_probabilities=[1/160]*160))
    d['planning_support']=dict(target_event='michael_2018',split='planning',development=False,regime='R1',lead_h=1,task_source=taskref,task_sha256='task',scenarios=[dict(scenario_id=i,scenario_sha256='s') for i in ids])
    for k in REQUIRED:d.setdefault(k,{})
    d['adoption_decision']=dict(kind='SAA_COMPOSITE_ADOPTION_DECISION_V1',approved=True,identity=ident,
        references={k:v for k,v in refs.items() if k!='adoption_decision'},planning_source=sources,adapter_sources=adapters,
        plan_input_references=inputs,no_new_Q=True,formal_OOS=False,administration_accounting='separate_measured_not_solver_time')
    return request,d

class CompositeTests(unittest.TestCase):
    def test_preserves_native_and_uses_current_total(self):
        r,d=fixture(); result=build_saa_composite_record(r,d)
        self.assertEqual(result['solver_record']['original_native_record'],d['original_solver'])
        self.assertAlmostEqual(result['budget_record']['effective_spent_seconds'],5683.899147049283)
        self.assertEqual(result['plan']['execution_mode'],'EA')
        self.assertFalse(result['solver_record']['normal_gap_met'])

    def test_reject_mutations(self):
        mutations=[lambda r,d:d['original_finalization']['references'].update(solver_record=refs_fake()),
          lambda r,d:d['latest_settlement'].update(effective_spent_seconds=5541.351921049283),
          lambda r,d:d['ledger_checked'].update(pending=True),
          lambda r,d:d['ledger_checked'].update(extra=0),
          lambda r,d:d['original_solver'].update(status='OPTIMAL'),
          lambda r,d:d['auxiliary_acceptance'].update(upper_bound_usd=1),
          lambda r,d:d['auxiliary_review'].update(passed=False),
          lambda r,d:d['model_manifest']['scenario_sha256s'].__setitem__(0,'changed'),
          lambda r,d:d['tasks'].update(task_sha256='other'),
          lambda r,d:r['references']['auxiliary_acceptance'].update(sha256='forged'),
          lambda r,d:d['original_terminal_review']['finalization_reference'].update(sha256='replaced')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                r,d=copy.deepcopy(fixture()); mutate(r,d)
                with self.assertRaises(ValueError):build_saa_composite_record(r,d)

def refs_fake():return dict(path='/different',sha256='different',size_bytes=1)
