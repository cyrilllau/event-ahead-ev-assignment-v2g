"""Independent edge checks for Case0's unchanged exact-zero proof bridge."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments import case0_planning_audit as audit
from evcs_experiments.zero_presolve import reduce_zero_columns
from evcs_experiments.random_streams import canonical_hash

class ZeroProofBridgeIndependentTests(unittest.TestCase):
    def setUp(self):
        self.a=csr_matrix([[1.,0.],[0.,1.]])
        self.arrays=dict(c=np.array([4.,5.]),row_lower=np.array([0.,-np.inf]),row_upper=np.array([0.,3.]),lower_bounds=np.zeros(2),upper_bounds=np.array([2.,3.]),objective_constant_usd=np.array(7.))
        self.proof=reduce_zero_columns(self.a,self.arrays['row_lower'],self.arrays['row_upper'],self.arrays['lower_bounds'],self.arrays['upper_bounds']).as_dict()
    def call(self,vector,proof=None):
        return audit.audit_case0_zero_reduction(self.a,self.arrays,self.proof if proof is None else proof,np.array(vector))
    def test_original_objective_constant_and_exact_zero_lift(self):
        result=self.call([0.,2.])
        self.assertTrue(result['passed']);self.assertEqual(result['original_objective_constant_usd'],7.)
        with self.assertRaisesRegex(ValueError,'exactly zero'):
            self.call([1e-30,2.])
    def test_resealed_threshold_cannot_turn_exact_proof_approximate(self):
        proof=copy.deepcopy(self.proof);proof['coefficient_threshold']=1e-20
        proof['sha256']=canonical_hash({k:v for k,v in proof.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'Reduction kind'):self.call([0.,2.],proof)
    def test_pinned_checker_cannot_be_silently_replaced(self):
        with patch.object(audit,'ZERO_REDUCTION_AUDITOR_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'checker source'):self.call([0.,2.])

    def test_native_metadata_must_describe_retained_rows_not_original(self):
        import hashlib
        empty=hashlib.sha256(np.array([],dtype='<i8').tobytes()).hexdigest()
        one=hashlib.sha256(np.array([0],dtype='<i8').tobytes()).hexdigest()
        native=dict(original_matrix_shape=[1,1],original_matrix_nnz=1,model_columns=1,model_rows=1,row_map_summary=dict(eq=dict(count=0,source_rows_sha256=empty),le=dict(count=1,source_rows_sha256=one),ge=dict(count=0,source_rows_sha256=empty),original_rows=1,translated_rows=1,free_rows=0,ranged_rows=0))
        audit.audit_case0_zero_reduction(self.a,self.arrays,self.proof,np.array([0.,2.]),native)
        native['original_matrix_shape']=[2,2]
        with self.assertRaisesRegex(ValueError,'Native reduced matrix'):
            audit.audit_case0_zero_reduction(self.a,self.arrays,self.proof,np.array([0.,2.]),native)
