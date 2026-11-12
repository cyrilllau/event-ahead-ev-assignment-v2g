import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import load_npz,save_npz
from evcs_experiments.case0_planning import run_case0_planning
from evcs_experiments.case0_planning_audit import audit_case0_method
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases

class Case0IndependentPlanningAuditTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.f=build_golden_cases()[0]
  for s in self.f.scenarios:s.split='planning'
  self.first=assemble_first_stage(self.f.case,self.f.tasks);self.templates=[assemble_recourse(self.f.case,self.f.tasks,s) for s in self.f.scenarios]
 def produce(self,method='EA_DRO'):
  now=time.monotonic();self.out=self.root/method
  run_case0_planning(self.first,self.templates,self.f.scenario_ids,self.f.probability_set,case=self.f.case,tasks=self.f.tasks,scenes=self.f.scenarios,method_id=method,output_dir=self.out,backend='highs',deadline=now+1800,configuration_started_at=now,master_seconds=5,empirical_probabilities=np.full(len(self.templates),1/len(self.templates)))
 def audit(self):
  with patch('evcs_experiments.case0_planning.solve_extensive',side_effect=AssertionError('No MILP audit')),patch('evcs_experiments.case0_planning.solve_fixed_plan',side_effect=AssertionError('No Q reoptimization')),patch('evcs_experiments.case0_planning.solve_worst_distribution',side_effect=AssertionError('No risk optimization')):
   return audit_case0_method(result_reference=file_reference(self.out/'result.json'),case=self.f.case,tasks=self.f.tasks,scenes=self.f.scenarios,probability_set=self.f.probability_set,deadline=time.monotonic()+20)
 def edit(self,name,fun):
  p=self.out/name;x=json.loads(p.read_text());fun(x);p.write_text(json.dumps(x))
 def test_all_four_real_forms_and_saved_witnesses(self):
  for method in ('EA_DRO','OR_DRO','SAA','ARO'):
   self.produce(method);result=self.audit();self.assertTrue(result['passed']);self.assertLessEqual(result['gap'],.005);self.assertEqual(result['scenario_count'],3)
 def test_original_matrix_tamper_rejected_despite_true_flags(self):
  self.produce();a=load_npz(self.out/'matrix.npz');a.data[0]+=1;save_npz(self.out/'matrix.npz',a)
  with self.assertRaisesRegex(ValueError,'coefficient'):self.audit()
 def test_raw_dual_tamper_rejected(self):
  self.produce();self.edit('scenario_000/fixed_solver.json',lambda x:x['certificate']['dual_lower_marginals'].__setitem__(0,100.))
  with self.assertRaisesRegex(ValueError,'KKT'):self.audit()
 def test_native_bound_offset_tamper_rejected(self):
  self.produce();self.edit('master.json',lambda x:x['raw_solver'].__setitem__('raw_lower_bound_without_constant_usd',1e7))
  with self.assertRaisesRegex(ValueError,'lower bound'):self.audit()
 def test_rehashed_result_risk_claim_tamper_rejected(self):
  self.produce();self.edit('result.json',lambda x:x.__setitem__('upper_bound_usd',0.))
  with self.assertRaisesRegex(ValueError,'upper_bound'):self.audit()
 def test_nonuniform_saa_coefficients_and_wrong_domain_rejected(self):
  self.produce('SAA');self.edit('model.json',lambda x:x.__setitem__('empirical_probabilities',[.2,.3,.5]))
  with self.assertRaisesRegex(ValueError,'uniform'):self.audit()
  self.f.scenarios[0].split='oos'
  with self.assertRaisesRegex(ValueError,'domain'):self.audit()
 def test_deadline_rejects_before_read(self):
  with self.assertRaises(TimeoutError):audit_case0_method(result_reference={},case=None,tasks=None,scenes=(),probability_set=None,deadline=time.monotonic()-1)

if __name__=='__main__':unittest.main()

class Case0GroupAdmissionTests(unittest.TestCase):
 def test_changed_request_evidence_rejected_before_context_or_solver(self):
  from evcs_experiments.case0_planning_audit import audit_case0_planning
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'config.lock.json').write_text('{}')
   ref=file_reference(root/'config.lock.json');request=dict(root=d,config_reference=ref,original_config_reference=ref,method_evidence={'wrong':{}})
   with patch('evcs_experiments.case0_evaluation_runner.planning_evidence',return_value=({},dict(status='PLANNING_REVIEW_REQUIRED',size_index=1),{})),patch('evcs_experiments.case0_inputs.Case0Context',side_effect=AssertionError('No context before evidence admission')):
    with self.assertRaisesRegex(ValueError,'method evidence'):audit_case0_planning(request=request,deadline=time.monotonic()+10)
 def test_audit_source_review_cannot_relabel_source(self):
  from evcs_experiments.case0_planning_audit import audit_case0_planning
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'config.lock.json').write_text('{}');(root/'review.json').write_text(json.dumps(dict(kind='CASE0_NUMERICAL_AUDITOR_REVIEW_V1',passed=True,auditor_source_references=[])))
   ref=file_reference(root/'config.lock.json');request=dict(root=d,config_reference=ref,original_config_reference=ref,method_evidence={},independent_review_reference=file_reference(root/'review.json'))
   with patch('evcs_experiments.case0_evaluation_runner.planning_evidence',return_value=({},dict(status='PLANNING_REVIEW_REQUIRED',size_index=1),{})):
    with self.assertRaisesRegex(ValueError,'exact source'):audit_case0_planning(request=request,deadline=time.monotonic()+10)
