import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.config import get_paths
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases

spec=importlib.util.spec_from_file_location('pending_audit',get_paths().project_root/'scripts/audits/audit_s05_pending_recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PendingAuditTests(unittest.TestCase):
    def test_real_template_digest_from_saved_coefficients(self):
        f=build_golden_cases()[0];t=assemble_recourse(f.case,f.tasks,f.scenarios[0])
        matrices={k:getattr(t,k) for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
        arrays={k:getattr(t,k) for k in ('b_ub','b_eq','lower_bounds','upper_bounds','component_objectives')}
        self.assertEqual(m.template_digest(t.as_dict(),matrices,arrays,t.first_stage.matrix_sha256),t.template_sha256)
        arrays={**arrays,'component_objectives':arrays['component_objectives']+1.}
        self.assertNotEqual(m.template_digest(t.as_dict(),matrices,arrays,t.first_stage.matrix_sha256),t.template_sha256)

    def test_large_descriptive_suffix_still_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'identity.json';p.write_text(json.dumps(dict(task_sha256='x',variable_map=['label']*3000),indent=2))
            a=m.Audit(10);r=a.ref(p);self.assertEqual(m.metadata_header(p),{'task_sha256':'x'})
            p.write_bytes(p.read_bytes().replace(b'label',b'other'))
            with self.assertRaisesRegex(ValueError,'changed'):a.verify(r)

    def test_missing_metadata_delimiter_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'identity header'):m.metadata_header(p)

    def fixture(self):
        empty=csr_matrix((0,1));mat={k:empty for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
        v=dict(b_ub=np.array([]),b_eq=np.array([]),lower_bounds=np.array([0.]),upper_bounds=np.array([np.inf]),c=np.array([1.]))
        dual=dict(inequality=np.array([]),equality=np.array([]),lower=np.array([1.]),upper=np.array([0.]),
                  stationarity=np.array([0.]),inequality_slack=np.array([]),equality_residual=np.array([]),
                  inequality_complementarity=np.array([]),lower_complementarity=np.array([0.]),upper_complementarity=np.array([]))
        cert={'dual_'+k+'_marginals':dual[k].tolist() for k in ('inequality','equality','lower','upper')}
        cert.update(dual_objective_usd=0.,dual_bound_usd=0.,primal_bound_usd=0.,options={'threads':1})
        solver=dict(status='OPTIMAL',optimal=True,solver_status=0,objective_usd=0.,certificate=cert)
        return mat,v,SimpleNamespace(x=np.array([0.]),r=np.array([[0.]])),dual,solver

    def test_kkt_success_and_wrong_primal_or_dual_rejected(self):
        mat,v,plan,dual,solver=self.fixture();fixed={k:v[k] for k in ('b_ub','b_eq')}
        self.assertEqual(m.audit_kkt(mat,v,plan,np.array([0.]),dual,solver,fixed)['duality_gap_usd'],0)
        with self.assertRaisesRegex(ValueError,'KKT'):m.audit_kkt(mat,v,plan,np.array([-1.]),dual,solver,fixed)
        dual['lower']=np.array([0.]);solver['certificate']['dual_lower_marginals']=[0.]
        with self.assertRaisesRegex(ValueError,'KKT'):m.audit_kkt(mat,v,plan,np.array([0.]),dual,solver,fixed)

    def test_limited_lp_not_promoted_by_feasible_primal(self):
        mat,v,plan,dual,solver=self.fixture();solver.update(status='TIME_LIMIT',optimal=False)
        with self.assertRaisesRegex(ValueError,'not OPTIMAL'):m.audit_kkt(mat,v,plan,np.array([0.]),dual,solver,{k:v[k] for k in ('b_ub','b_eq')})

    def test_rmp_residuals_do_not_round_integral_decisions(self):
        v=dict(row_lower=np.array([0.]),row_upper=np.array([1.]),lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),integrality=np.array([1]))
        self.assertEqual(m.rmp_residuals(csr_matrix([[1.]]),v,np.array([0.]))['max_row_violation'],0)
        with self.assertRaisesRegex(ValueError,'primal'):m.rmp_residuals(csr_matrix([[1.]]),v,np.array([.1]))

    def test_budget_timeout_is_exception(self):
        a=m.Audit(10)
        with patch.object(m.time,'monotonic',return_value=a.started+10.01):
            with self.assertRaises(TimeoutError):a.remaining()


if __name__=='__main__':unittest.main()
