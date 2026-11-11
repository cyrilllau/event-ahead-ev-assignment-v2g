import copy,importlib.util,unittest
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.zero_presolve import reduce_zero_columns
spec=importlib.util.spec_from_file_location('zero_auditor',get_paths().project_root/'scripts/audits/audit_zero_reduction.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ProofTests(unittest.TestCase):
    def setUp(self):
        self.a=csr_matrix([[1.,0,0],[-1,1,0],[0,0,1],[0,0,-1]])
        self.args=(self.a,[-np.inf]*4,[0]*4,[0,0,-np.inf],[1,1,np.inf])
        self.proof=reduce_zero_columns(*self.args).as_dict()
    def test_original_proof_passes(self):
        self.assertEqual(audit.audit_zero_reduction(*self.args,self.proof)['zero_columns'],3)
    def test_forged_witness_rejected_even_after_checksum_recomputed(self):
        d=copy.deepcopy(self.proof);d['proof_steps'][1]['witness_rows']=[1];d.pop('sha256');d['sha256']=canonical_hash(d)
        with self.assertRaises(ValueError):audit.audit_zero_reduction(*self.args,d)
    def test_original_rhs_change_rejected(self):
        args=list(self.args);args[2]=[1e-25,0,0,0]
        with self.assertRaises(ValueError):audit.audit_zero_reduction(*args,self.proof)

if __name__=='__main__':unittest.main()
