"""Synthetic immutable evidence graph: no numerical proofs or solver are mocked by production.

These metadata tests exercise binding/acceptance semantics with small explicit
previously-accepted proof fixtures; original numerical replay is not this helper's job.
"""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import case0_accepted_plan as bridge
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.random_streams import canonical_hash

class AcceptedPlanTests(unittest.TestCase):
 def setUp(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.parent=Path(tmp.name);self.root=self.parent/bridge.CAMPAIGN;self.root.mkdir();self.op=self.root/'operations/accept';self.op.mkdir(parents=True)
  self.source=self.parent/'release';(self.source/'src').mkdir(parents=True);self.sp=self.source/'src/fixture.py';self.sp.write_text('# original producer fixture\n')
  self.plan=dict(x=[1.],r=[[2.]],task_sha256='t'*64,config_hash='c'*64);self.planhash=canonical_hash(self.plan)
  self.patch=patch.object(bridge,'PLAN_SHA',self.planhash);self.patch.start();self.addCleanup(self.patch.stop)
  self.ids=['zeta_2020/case0/development/planning/s'+str(i)+'/v30001/R1' for i in range(8)];q=bridge.UPPER-1.;self.q=[q]*8
  config=dict(profile='expanded48-v1',target_event='zeta_2020',backend='gurobi',sizes=[dict(fleet_per_depot=12,planning_variants=1)],variant_start=30000,gap_target=.005,configuration_seconds=1800,master_seconds=1200,source_sha256={'src/fixture.py':file_reference(self.sp)['sha256']})
  configref=self.save('config.json',config);manifestref=self.save('manifest.json',{'files':[dict(path='src/fixture.py',sha256=file_reference(self.sp)['sha256'])]},self.source)
  taskref=self.save('tasks.json',{'synthetic':'task identity'});identityref=self.save('identity.json',dict(profile='expanded48-v1',target='zeta_2020',lead_h=1,vehicle_ids=[str(i) for i in range(48)],variant_offset=30000,variants_per_base={'planning':1},formal_OOS=False,task_sha256='t'*64,task_file=taskref))
  self.header=dict(model_sha256=bridge.MODEL_SHA,scenario_ids=self.ids,recourse_template_sha256s=[str(i)*64 for i in range(8)],ambiguity_sha256='a'*64)
  headerref=self.save('model.json',self.header);masterref=self.save('master.json',{'status':'TIME_LIMIT','fixture':True})
  entries=[]
  for i in range(8):
   checks=self.save(f'checks{i}.json',dict(kkt={'passed':True,'qop_usd':q},physical={'passed':True,'costs':{'Qop':q}}))
   entries.append(dict(index=i,scenario_id=self.ids[i],template_sha256=str(i)*64,q_usd=q,files={'fixed_solver':self.save(f'solver{i}.json',{'fixture':'already accepted dual/primal'}),'primal':self.save(f'primal{i}.json',{'fixture':'already accepted vector'}),'checks':checks}))
  self.ub=dict(kind='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1',passed=True,method_id='EA_DRO',source_binding={'master':masterref,'model':headerref,'input_identity':identityref,'config':configref,'source_manifest':manifestref},plan=self.plan,plan_sha256=self.planhash,model_sha256=bridge.MODEL_SHA,upper_bound_usd=bridge.UPPER,gap_passed=False,planning_acceptance=False,full_numerical_acceptance=False,first_stage_physical_passed=True,full_support_fixed_Q_and_KKT_passed=True,probability_primal_dual_passed=True,scenario_ids=self.ids,q_usd=self.q,completed_scenario_references=entries,Fpre_usd=1.)
  ubref=self.save('ub.json',self.ub)
  pref=self.save('probability.json',dict(certified_optimal=True,support_ids=self.ids,ambiguity_sha256='a'*64,probabilities=[.125]*8,primal_objective_usd=q,dual_objective_usd=q))
  self.closure=dict(kind='CASE0_ASSEMBLED_ORIGINAL_EXTENSIVE_WITNESS_CHECK_V1',passed=True,plan_sha256=self.planhash,model_sha256=bridge.MODEL_SHA,original_array_check=dict(passed=True,max_row_violation=0.,max_bound_violation=0.,max_integrality_violation=0.),original_native_primal_recovered=False,assembled_from_independent_Q_primal_and_P_dual=True,planning_acceptance=False,new_Q_calls=0,new_master_calls=0,objective_usd=bridge.UPPER,source_binding={'ub_result':ubref,'probability':pref},assembled_primal_reference=self.save('assembled.json',{'fixture':'accepted extensive vector'}))
  cr=self.save('closure.json',self.closure);ctr=self.save('closure_terminal.json',dict(kind='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_TERMINAL_REVIEW_V1',passed=True,numerical_recheck_passed=True,references={'result':cr}))
  csr=self.save('closure_sett.json',dict(passed=True,result_reference=cr,original_best_ub_reference=ubref,terminal_review_reference=ctr))
  native=self.save('native.json',dict(passed=True,native_status='OPTIMAL',lower_bound_usd=bridge.LOWER))
  self.bound=dict(kind='CASE0_PARTIAL_CHAIN_MASTER_BOUND_RESULT_V1',model_sha256=bridge.MODEL_SHA,old_plan_sha256=self.planhash,lower_bound_usd=bridge.LOWER,upper_bound_usd=bridge.UPPER,gap=bridge.GAP,gap_passed=False,planning_acceptance=False,source_binding={'old_ub':ubref},native_audit_reference=native,native_status='OPTIMAL',new_native_lower_bound_usd=bridge.LOWER)
  br=self.save('bound.json',self.bound);btr=self.save('bound_terminal.json',dict(passed=True,numerical_recheck_passed=True,references={'result':br}))
  self.sett=dict(passed=True,result_reference=br,best_ub_reference=ubref,best_lower_bound_usd=bridge.LOWER,best_upper_bound_usd=bridge.UPPER,best_gap=bridge.GAP,terminal_review_reference=btr,closing_compute_seconds=1600.,closing_configuration_seconds=1500.,closing_master_seconds=1100.)
  sr=self.save('bound_sett.json',self.sett)
  self.refs=dict(saved_candidate_ub_reference=ubref,original_extensive_closure_result_reference=cr,original_extensive_closure_settlement_reference=csr,partial_chain_master_bound_result_reference=br,partial_chain_master_bound_settlement_reference=sr,source_release_reference=manifestref,active_config_reference=configref)
  dp=self.op/'decision.md';dp.write_text('Specific user accepted this plan, not the normal threshold.');self.dp=patch.object(bridge,'DECISION_SHA',file_reference(dp)['sha256']);self.dp.start();self.addCleanup(self.dp.stop)
  self.accept=dict(kind='CASE0_EXACT_PLAN_APPROXIMATE_ACCEPTANCE_V1',accepted_for_comparison=True,normal_gap_passed=False,formal_OOS=False,user_instruction='就用0.71 可以',plan_sha256=self.planhash,model_sha256=bridge.MODEL_SHA,upper_bound_usd=bridge.UPPER,lower_bound_usd=bridge.LOWER,gap=bridge.GAP,references=self.refs,decision_document_reference=file_reference(dp),opening_compute_seconds=1600.,opening_configuration_seconds=1500.,metadata_seconds=1.,closing_compute_seconds=1601.,closing_configuration_seconds=1501.,master_seconds=0.)
  ar=self.save('acceptance.json',self.accept);self.state=dict(size_index=0,accepted_approximate_plans={bridge.KEY:ar},compute_seconds=1601.,configuration_spent_seconds={bridge.KEY:1501.},configuration_master_spent_seconds={bridge.KEY:1100.},**self.refs)
 def save(self,name,obj,folder=None):
  p=(folder or self.op)/name;p.write_text(json.dumps(obj));return file_reference(p)
 def load(self):
  # A test-specific exact acceptance file replaces only the production fixed
  # acceptance digest. All inner semantic/reference checks run unchanged.
  with patch.object(bridge,'ACCEPTANCE_SHA',self.state['accepted_approximate_plans'][bridge.KEY]['sha256']):return bridge.load_accepted_plan(self.root,self.state)
 def test_distinct_accepted_view_preserves_original_normal_failure(self):
  before={p:p.read_bytes() for p in self.parent.rglob('*') if p.is_file()};e=self.load();r=e['comparison_result']
  self.assertTrue(r['accepted_for_comparison']);self.assertFalse(r['gap_passed'] or r['normal_planning_acceptance']);self.assertEqual(r['gap'],.007101288085421298)
  self.assertEqual(e['result_reference'],self.refs['saved_candidate_ub_reference']);self.assertNotEqual(r['kind'],'CASE0_FULL_FORMULATION_NUMERICAL_RESULT_V1');self.assertEqual(before,{p:p.read_bytes() for p in before})
  self.assertIsNone(bridge.load_accepted_plan(self.root,self.state,method='OR_DRO'))
 def test_rounded_threshold_must_not_reject_exact_accepted_point(self):
  self.assertGreater(bridge.GAP,.0071);self.assertTrue(self.load()['comparison_result']['accepted_for_comparison'])
 def test_missing_or_tampered_exact_decision_does_not_skip_ea(self):
  state=copy.deepcopy(self.state);state['accepted_approximate_plans']={};self.assertIsNone(bridge.load_accepted_plan(self.root,state))
  Path(self.accept['decision_document_reference']['path']).write_text('different user decision')
  with self.assertRaises(ValueError):self.load()
 def test_rehashed_acceptance_cannot_enable_normal_pass_or_change_gap(self):
  for k,v in [('normal_gap_passed',True),('gap',.005),('plan_sha256','wrong')]:
   old=copy.deepcopy(self.accept);old[k]=v;self.state['accepted_approximate_plans'][bridge.KEY]=self.save('acceptance.json',old)
   with self.assertRaises(ValueError):self.load()
 def test_missing_eighth_or_reordered_scenario_evidence_rejected(self):
  # Rebind the UB and dependent acceptance references to test semantic gates,
  # rather than only relying on the immutable outer digest.
  bad=copy.deepcopy(self.ub);bad['completed_scenario_references']=bad['completed_scenario_references'][:-1];ref=self.save('ub.json',bad)
  self.refs['saved_candidate_ub_reference']=ref;self.state['saved_candidate_ub_reference']=ref;self.state['accepted_approximate_plans'][bridge.KEY]=self.save('acceptance.json',self.accept)
  with self.assertRaisesRegex(ValueError,'eight-support'):self.load()
 def test_byte_changed_q_witness_is_rejected(self):
  p=Path(self.ub['completed_scenario_references'][3]['files']['fixed_solver']['path']);p.write_text('changed')
  with self.assertRaises(ValueError):self.load()
 def test_ea_budget_reset_or_further_master_refinement_rejected(self):
  for k,val in [('configuration_spent_seconds',1499.),('configuration_spent_seconds',1502.),('configuration_master_spent_seconds',1101.)]:
   old=copy.deepcopy(self.state);self.state[k][bridge.KEY]=val
   with self.assertRaisesRegex(ValueError,'reset or refined'):self.load()
   self.state=old

if __name__=='__main__':unittest.main()
