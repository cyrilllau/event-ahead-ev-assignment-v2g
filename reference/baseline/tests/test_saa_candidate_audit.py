"""Real small SAA master and independent fixed-Q evidence joined without a solve."""
import json
from pathlib import Path
import tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from evcs_experiments import saa_candidate_audit as join
from evcs_experiments import saa_fixed_plan as producer
from evcs_experiments import saa_fixed_plan_audit as qa
from evcs_experiments import saa_master_audit as ma
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.saa_streaming import assemble_streaming_saa
from evcs_experiments.saa_streaming_solve import solve_streaming_saa
from evcs_experiments.validation.s03_toys import build_golden_cases

class SAACandidateAuditTests(unittest.TestCase):
 def setUp(self):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
  f=build_golden_cases()[0];self.f=f;self.ts=[assemble_recourse(f.case,f.tasks,s) for s in f.scenarios]
  constants=f.scenario_cost_constants_usd-100
  self.model=assemble_streaming_saa(self.ts[0].first_stage,self.ts,scenario_ids=f.scenario_ids,
   scenario_sha256s=[s.scenario_sha256 for s in f.scenarios],empirical_probabilities=f.empirical_probabilities,
   scenario_cost_constants_usd=constants,output_directory=self.root/'model',max_coefficient_bytes=50000000)
  native=solve_streaming_saa(self.model,self.ts,case=f.case,tasks=f.tasks,scene_loader=f.scenarios.__getitem__,
   output_directory=self.root/'master',backend='highs',time_limit_seconds=5,deadline=time.monotonic()+30,evidence_reserve_seconds=5)
  plan=self.model.plan(np.load(native['primal']['path']))
  self.sources=[file_reference(p) for p in sorted(Path(join.__file__).parent.glob('*.py'))]
  q=producer.complete_saa_fixed_plan_q(plan=plan,case=f.case,tasks=f.tasks,scenario_ids=f.scenario_ids,
   scenario_sha256s=[s.scenario_sha256 for s in f.scenarios],template_sha256s=[t.template_sha256 for t in self.ts],
   scenario_cost_constants_usd=constants,template_loader=self.ts.__getitem__,scene_loader=f.scenarios.__getitem__,
   source_references=self.sources,input_binding={'model_manifest':file_reference(self.model.root/'manifest.json')},
   output_directory=self.root/'q',deadline=time.monotonic()+30,cumulative_budget_seconds=30)
  refs=[]
  for i,t in enumerate(self.ts):
   d=self.root/f'template_{i}';d.mkdir()
   for n in qa.recourse_store.MATRICES:save_npz(d/(n+'.npz'),getattr(t,n))
   np.savez_compressed(d/'arrays.npz',**{n:getattr(t,n) for n in qa.recourse_store.ARRAYS})
   qa._write(d/'identity.json',t.as_dict())
   refs.append(qa._write(d/'record.json',dict(kind='recourse_template',scenario_index=i,scenario_id=f.scenario_ids[i],
    template_sha256=t.template_sha256,artifacts=[file_reference(p) for p in sorted(d.iterdir())])))
  env=qa.bind_saa_template_envelope(request_reference=q['request_reference'],template_references=refs,
   first_stage_sha256=self.ts[0].first_stage.matrix_sha256,source_references=self.sources,output_path=self.root/'envelope.json')
  self.args=dict(model=self.model,recourses=self.ts,case=f.case,tasks=f.tasks,scene_loader=f.scenarios.__getitem__,
   master_result_reference=file_reference(self.root/'master/result.json'),checkpoint_reference=q['checkpoint_reference'],
   fixed_q_request_reference=q['request_reference'],template_envelope_reference=env,
   helper_reference=file_reference(get_paths().project_root/'scripts/audits/audit_s05_pending_recovery.py'),
   source_references=self.sources,deadline=time.monotonic()+30)
 def test_real_joint_native_lower_and_empirical_upper_no_solver(self):
  with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('audit cannot solve')):
   r=join.audit_saa_candidate(**self.args)
  self.assertAlmostEqual(r['lower_bound_usd'],-69.73333333333333,places=7)
  self.assertAlmostEqual(r['upper_bound_usd'],r['lower_bound_usd'],places=7)
  self.assertEqual(r['risk_certificate_kind'],'fixed_empirical');self.assertEqual(r['solver_calls'],0)
  self.assertFalse(r['budget_settled']);self.assertFalse(r['formal_OOS'])
 def test_other_support_constants_or_plan_cannot_borrow_master(self):
  original=json.loads(Path(self.args['fixed_q_request_reference']['path']).read_text())
  for key,value in [('scenario_ids',list(reversed(original['scenario_ids']))),('constants_usd',[0,0,0]),('plan_sha256','f'*64)]:
   req={**original,key:value};p=self.root/'other.json';p.write_text(json.dumps(req))
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'differ|same usable'):
    join.audit_saa_candidate(**{**self.args,'fixed_q_request_reference':file_reference(p)})
 def test_native_time_limit_remains_visible_even_when_gap_is_small(self):
  n=self.root/'master/native_result.json';native=json.loads(n.read_text());native['status']=1;n.write_text(json.dumps(native))
  p=self.root/'master/result.json';r=json.loads(p.read_text());r.update(raw_native_result=file_reference(n),solver_status=1,status='TIME_LIMIT',normal_convergence=False);p.write_text(json.dumps(r))
  result=join.audit_saa_candidate(**{**self.args,'master_result_reference':file_reference(p)})
  self.assertEqual(result['native_status'],'TIME_LIMIT');self.assertEqual(result['native_solver_status'],1)
  self.assertTrue(result['normal_gap_met']);self.assertFalse(result['scientific_spec_pass'])
 def test_expired_and_missing_own_source_rejected(self):
  with self.assertRaisesRegex(ValueError,'deadline'):join.audit_saa_candidate(**{**self.args,'deadline':time.monotonic()-1})
  refs=[r for r in self.sources if r['path']!=join.__file__]
  with self.assertRaisesRegex(ValueError,'source missing'):join.audit_saa_candidate(**{**self.args,'source_references':refs})

if __name__=='__main__':unittest.main()
