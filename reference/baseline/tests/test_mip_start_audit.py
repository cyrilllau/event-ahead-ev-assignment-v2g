import importlib.util,unittest
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.config import get_paths
from evcs_experiments.native_highs import solve_with_start
spec=importlib.util.spec_from_file_location('start_audit',get_paths().project_root/'scripts/audits/audit_mip_start.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class StartAuditTests(unittest.TestCase):
    def test_original_start_and_reject_nonbinary_poison(self):
        arrays=dict(c=np.array([1.]),integrality=np.array([1]),lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),row_lower=np.array([-np.inf]),row_upper=np.array([1.]))
        A=csr_matrix([[1.]])
        r=solve_with_start(arrays['c'],arrays['integrality'],arrays['lower_bounds'],arrays['upper_bounds'],A,arrays['row_lower'],arrays['row_upper'],np.array([1.]),{'time_limit':1.,'threads':1})
        solver={'raw_solver':{'native_interface':r.native_record}}
        self.assertEqual(a.audit_mip_start(A,arrays,[1.],solver)['status'],'PASS_ORIGINAL_MIP_START')
        with self.assertRaisesRegex(ValueError,'integrality'):a.audit_mip_start(A,arrays,[.5],solver)
        with self.assertRaisesRegex(ValueError,'hash'):a.audit_mip_start(A,arrays,[0.],solver)
if __name__=='__main__':unittest.main()
