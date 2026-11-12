import copy
import time
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.bounded_row_relaxation import relax_bounded_rows
from evcs_experiments.bounded_row_relaxation_audit import audit_bounded_row_relaxation
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.relaxation_bound import coefficient_sha256

class IndependentRelaxationTests(unittest.TestCase):
    def setUp(self):
        self.m=dict(matrix=csr_matrix([[-15.,1.],[1.,-1e-14],[1e-14,1.]]),
            c=np.array([1.,0.]),lower_bounds=np.array([0.,0.]),upper_bounds=np.array([1.,np.inf]),
            integrality=np.array([1,0]),row_lower=np.array([-np.inf,0.,-1.]),
            row_upper=np.array([0.,2.,np.inf]),objective_constant_usd=3.)
        self.kw=dict(allowed_rows=(1,3),allowed_columns=(0,2),bound_witnesses={1:0},witness_rows=(0,1))
        self.r,self.record=relax_bounded_rows(self.m,**self.kw)
    def audit(self):
        return audit_bounded_row_relaxation(self.m,self.r,self.record,deadline=time.monotonic()+5,**self.kw)
    def reseal(self):
        self.record['relaxed_coefficient_sha256']=coefficient_sha256(self.r)
        self.record['original_coefficient_sha256']=coefficient_sha256(self.m)
        self.record.pop('relaxation_sha256',None)
        self.record['relaxation_sha256']=canonical_hash(self.record)
    def test_exact_witness_and_outer_rows(self):
        out=self.audit();self.assertTrue(out['passed']);self.assertEqual(out['removed_entry_count'],2)
        self.assertEqual(out['bound_witness_count'],1);self.assertTrue(np.isinf(self.r['upper_bounds'][1]))
    def test_matrix_domain_constant_tamper(self):
        for key in ('matrix','lower_bounds','objective_constant_usd'):
            with self.subTest(key=key):
                self.setUp()
                if key=='matrix':self.r[key]=self.r[key].copy();self.r[key][0,0]=-14.
                elif key=='lower_bounds':self.r[key]=np.array([.1,0.])
                else:self.r[key]=4.
                self.reseal()
                with self.assertRaises(ValueError):self.audit()
    def test_tightened_rhs_resealed(self):
        self.r['row_upper'][1]=self.m['row_upper'][1]-1e-6;self.reseal()
        with self.assertRaisesRegex(ValueError,'strengthens'):self.audit()
    def test_missing_deletion_and_record_tamper(self):
        self.r['matrix']=self.m['matrix'].copy();self.reseal()
        with self.assertRaises(ValueError):self.audit()
        self.setUp();self.record['removed_entries'].pop();self.reseal()
        with self.assertRaises(ValueError):self.audit()
    def test_unbounded_without_witness(self):
        self.kw['bound_witnesses']={};self.record['bound_witnesses']=[];self.reseal()
        with self.assertRaisesRegex(ValueError,'Unbounded'):self.audit()
    def test_resealed_unsafe_witness(self):
        self.record['bound_witnesses'][0]['upper_bound']=14.;self.reseal()
        with self.assertRaisesRegex(ValueError,'Unsafe'):self.audit()
    def test_wrong_prefix_and_fraction(self):
        self.kw['witness_rows']=(1,2)
        with self.assertRaises(ValueError):self.audit()
        self.setUp();self.record['rows'][0]['min_sum_exact']['numerator']='0';self.reseal()
        with self.assertRaises(ValueError):self.audit()
    def test_deadline(self):
        with self.assertRaises(TimeoutError):
            audit_bounded_row_relaxation(self.m,self.r,self.record,deadline=time.monotonic()-1,**self.kw)
