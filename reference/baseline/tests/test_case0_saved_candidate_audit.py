import dataclasses,importlib.util,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.extensive import assemble_extensive,solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases

path=get_paths().project_root/'scripts/audits/audit_case0_saved_candidate.py'
spec=importlib.util.spec_from_file_location('case0_candidate_audit_test_module',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class SavedCandidateTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.f=build_golden_cases()[0]
        for scene in self.f.scenarios:scene.split='planning'
        first=assemble_first_stage(self.f.case,self.f.tasks)
        templates=[assemble_recourse(self.f.case,self.f.tasks,s) for s in self.f.scenarios]
        model=assemble_extensive(first,templates,risk_model='DRO',scenario_ids=self.f.scenario_ids,probability_set=self.f.probability_set)
        saved=solve_extensive(model,time_limit_seconds=5.)
        self.master=saved.as_dict();self.master.update(status='TIME_LIMIT',optimal=False,lower_bound_usd=None,mip_gap=None)
        self.header=model.as_dict();self.context=SimpleNamespace(first=first,case=self.f.case,tasks=self.f.tasks,
            planning_ids=self.f.scenario_ids,scenario=lambda i:self.f.scenarios[i],probability_set=self.f.probability_set)
        self.expected=float(saved.objective_usd)
    def run_audit(self,folder='audit',resume=None):
        out=self.root/folder;out.mkdir()
        with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No candidate master solve')):
            return audit.audit_candidate(master=self.master,header=self.header,context=self.context,output_dir=out,
                deadline=time.monotonic()+20,binding={'master':'unchanged evidence'},resume=resume)
    def test_saved_plan_full_q_ub_does_not_create_lb_or_planning_pass(self):
        r=self.run_audit();self.assertAlmostEqual(r['upper_bound_usd'],self.expected)
        self.assertTrue(r['first_stage_physical_passed'] and r['full_support_fixed_Q_and_KKT_passed'])
        self.assertIsNone(r['lower_bound_usd']);self.assertIsNone(r['gap'])
        self.assertFalse(r['gap_passed'] or r['planning_acceptance'] or r['full_numerical_acceptance'] or r['formal_OOS'])
        self.assertEqual(r['scenario_ids'],list(self.f.scenario_ids))
        self.assertFalse(r['original_master_vector_independently_replayed'])
    def test_invalid_saved_plan_is_not_rescued_by_true_flags(self):
        self.master['solution']['x'][0]=.5
        with self.assertRaisesRegex(ValueError,'first-stage'):self.run_audit()
        self.assertTrue((self.root/'audit/candidate.json').exists())
    def test_reordered_support_rejected_before_lp(self):
        self.header['scenario_ids']=list(reversed(self.header['scenario_ids']))
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('No LP')):
            with self.assertRaisesRegex(ValueError,'identity'):self.run_audit()
    def test_original_template_identity_and_oos_domain_rejected(self):
        self.header['recourse_template_sha256s'][0]='0'*64
        with self.assertRaisesRegex(ValueError,'recourse support'):self.run_audit()
        self.header['recourse_template_sha256s'][0]=assemble_recourse(self.f.case,self.f.tasks,self.f.scenarios[0]).template_sha256
        self.f.scenarios[0].split='oos'
        with self.assertRaisesRegex(ValueError,'planning R1'):self.run_audit('second')
    def test_each_completed_q_checkpoint_survives_later_failure(self):
        original=audit.solve_fixed_plan;calls=0
        def limited(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise TimeoutError('bounded second Q')
            return original(*args,**kwargs)
        with patch.object(audit,'solve_fixed_plan',side_effect=limited):
            with self.assertRaisesRegex(TimeoutError,'second Q'):self.run_audit()
        cp=json.loads((self.root/'audit/checkpoint.json').read_text());self.assertEqual(len(cp['completed']),1)
        r=self.run_audit('continued',resume=cp);self.assertTrue(r['completed_scenario_references'][0]['reused_from_checkpoint'])
    def test_checkpoint_core_rechecks_duals_before_reusing_any_q(self):
        self.run_audit();cp=json.loads((self.root/'audit/checkpoint.json').read_text())
        row=cp['completed'][0];p=Path(row['files']['fixed_solver']['path']);saved=json.loads(p.read_text())
        saved['certificate']['dual_lower_marginals'][0]+=100.;p.write_text(json.dumps(saved))
        row['files']['fixed_solver']=audit.file_reference(p)
        with patch.object(audit,'solve_fixed_plan',side_effect=AssertionError('No new Q before reused certificate')):
            with self.assertRaisesRegex(ValueError,'KKT'):self.run_audit('tampered',resume=cp)
    def test_no_bound_audit_does_not_accept_bound_relabel(self):
        self.master['lower_bound_usd']=1.
        with self.assertRaisesRegex(ValueError,'no-bound'):self.run_audit()

if __name__=='__main__':unittest.main()
