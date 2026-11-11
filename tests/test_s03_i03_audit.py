"""Adversarial saved-evidence replay, with tiny synthetic fixtures only."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from scipy.sparse import load_npz, save_npz

from evcs_experiments.config import get_paths
from evcs_experiments.validation.validate_s03_ccg import export_ccg
from evcs_experiments.validation.validate_s03_extensive import export_toys

spec=importlib.util.spec_from_file_location('ccg_saved_audit_tests',get_paths().project_root/'scripts/audits/audit_s03_i03.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class CCGSavedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.reference=cls.root/'reference';cls.reference.mkdir();export_toys(cls.reference,5.)
        cls.evidence=cls.root/'ccg';cls.evidence.mkdir();export_ccg(cls.evidence,5.,cls.reference)
        source=get_paths().project_root/'src/evcs_experiments'
        cls.engine=audit.canonical({n:audit.sha256(source/n) for n in
            ('ccg.py','extensive.py','first_stage.py','recourse.py','probability.py','probability_phase1.py','random_streams.py',
             'lazy_recourses.py','zero_presolve.py','native_highs.py','native_gurobi.py')})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_fixture(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        folder=Path(t.name)/'three_ev_moment'
        shutil.copytree(self.evidence/'toys/three_ev_moment',folder)
        return folder

    def run_audit(self,folder):
        return audit.audit_run(folder,'DRO','resumed',self.reference,self.engine)

    def rewrite(self,path,change):
        value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))

    def test_all_saved_matrices_and_certificates_pass(self):
        rows=[audit.audit_run(self.evidence/'toys'/n,r,m,self.reference,self.engine)
              for n in audit.old.GOLDENS for r in ('DRO','ARO') for m in ('continuous','resumed')]
        self.assertEqual(sum(x['RMP_matrices'] for x in rows),24)
        self.assertEqual(sum(x['fixed_LP_certificates'] for x in rows),178)

    def test_backend_or_thread_identity_cannot_be_relabelled(self):
        folder=self.evidence/'toys/three_ev_moment'
        for backend,threads in (('gurobi',4),('highs',4)):
            with self.subTest(backend=backend,threads=threads):
                with self.assertRaisesRegex(ValueError,'backend/thread'):
                    audit.audit_run(folder,'DRO','resumed',self.reference,self.engine,
                        expected_master_backend=backend,expected_optimizer_threads=threads)

    def test_trace_cannot_substitute_incumbent_for_solver_bound(self):
        folder=self.copy_fixture();p=folder/'DRO/resumed/result.json'
        self.rewrite(p,lambda v:v['trace'][0].update(lower_bound_usd=v['trace'][0]['lower_bound_usd']+1.))
        with self.assertRaisesRegex(ValueError,'Historical LB'):self.run_audit(folder)

    def test_trace_cannot_replace_original_rmp_dual(self):
        folder=self.copy_fixture();p=folder/'DRO/resumed/result.json'
        self.rewrite(p,lambda v:v['trace'][0]['raw_rmp_dual'].update(theta0=999.))
        with self.assertRaisesRegex(ValueError,'original RMP/dual'):self.run_audit(folder)

    def test_checkpoint_cannot_reset_cumulative_time_even_with_new_checksum(self):
        folder=self.copy_fixture();p=folder/'DRO/resumed/checkpoint_session_2.json'
        def poison(v):
            v['state']['elapsed_seconds']=0.;v['sha256']=audit.canonical(v['state'])
        self.rewrite(p,poison)
        with self.assertRaisesRegex(ValueError,'cumulative time'):self.run_audit(folder)

    def test_returned_plan_must_belong_to_best_bound(self):
        folder=self.copy_fixture();p=folder/'DRO/resumed/result.json'
        self.rewrite(p,lambda v:v['best'].update(iteration=999))
        with self.assertRaisesRegex(ValueError,'historical best'):self.run_audit(folder)

    def test_recourse_certificate_cannot_hide_bad_stationarity(self):
        folder=self.copy_fixture();d=folder/'DRO/resumed'
        events=json.loads((d/'events.json').read_text());e=next(e for e in events if e['event']=='recourse')
        p=d/e['path']/'solver.json'
        def poison(v):v['certificate']['dual_lower_marginals'][0]+=1.
        self.rewrite(p,poison)
        with self.assertRaisesRegex(ValueError,'raw KKT'):self.run_audit(folder)

    def test_scenario_addition_cannot_ignore_full_support_residual(self):
        folder=self.copy_fixture();p=folder/'DRO/resumed/result.json'
        self.rewrite(p,lambda v:v['trace'][0].update(added_scenario_index=None))
        with self.assertRaisesRegex(ValueError,'Scenario addition'):self.run_audit(folder)

    def test_intermediate_checkpoint_cannot_substitute_another_plan(self):
        folder=self.copy_fixture();p=folder/'DRO/resumed/checkpoint_session_1.json'
        def poison(v):
            v['state']['best']['plan']['x'][0]=1.-v['state']['best']['plan']['x'][0]
            v['sha256']=audit.canonical(v['state'])
        self.rewrite(p,poison)
        with self.assertRaisesRegex(ValueError,'Checkpoint best plan'):self.run_audit(folder)

    def test_template_parameter_sign_cannot_change_from_accepted_reference(self):
        folder=self.copy_fixture();p=folder/'template_0/Bx_ub.npz'
        a=load_npz(p);a.data[0]*=-1;save_npz(p,a)
        with self.assertRaisesRegex(ValueError,'Physical Bx changed'):self.run_audit(folder)


if __name__=='__main__':unittest.main()
