import time,unittest
import numpy as np
from scipy.sparse import csr_matrix,vstack,hstack
from tests.test_case0_saa_saved_arrays import SavedArraysTests,rehash
from evcs_experiments.case0_aro_saved_arrays import derive_saved_aro_cuts
from evcs_experiments.relaxation_bound import solve_relaxation_bound
class AROArraysTests(unittest.TestCase):
 def setUp(self):
  SavedArraysTests.setUp(self)
  self.A=vstack([hstack([self.A,csr_matrix((4,1))]),csr_matrix([[0.,1.,0.,-1.],[0.,0.,1.,-1.]])],format='csr')
  self.d.update(c=np.array([.1,0.,0.,1.]),row_lower=np.full(6,-np.inf),row_upper=np.r_[self.d['row_upper'],0.,0.],lower_bounds=np.r_[self.d['lower_bounds'],-np.inf],upper_bounds=np.r_[self.d['upper_bounds'],np.inf],integrality=np.zeros(4,dtype=int))
  self.h.update(risk_model='ARO',empirical_probabilities=None,shape=list(self.A.shape),nnz=self.A.nnz);self.h['variable_slices']['risk_threshold']=[3,4]
  self.h['block_map'] += [dict(scope='s'+str(i),kind='risk_epigraph',row_start=4+i,row_stop=5+i,source_sha256='t'+str(i)) for i in range(2)]
  rehash(self.A,self.d,self.h);self.u.update(method_id='ARO',model_sha256=self.h['model_sha256'])
 def run_model(self):return derive_saved_aro_cuts(self.A,self.d,self.h,self.u,self.e,deadline=time.monotonic()+5)
 def test_max_epigraph_original_costs_and_bound(self):
  m,r,c=self.run_model();self.assertEqual(m['matrix'].shape,(2,2));np.testing.assert_array_equal(m['c'],[.1,1.]);np.testing.assert_array_equal(m['row_upper'],[-2.,-2.])
  result,_=solve_relaxation_bound(m,time_limit_seconds=2);self.assertAlmostEqual(result['lower_bound_usd'],.2)
 def test_changed_epigraph_sign_rejected(self):
  self.A[4,3]=1.;rehash(self.A,self.d,self.h);self.u['model_sha256']=self.h['model_sha256']
  with self.assertRaisesRegex(ValueError,'risk row'):self.run_model()
 def test_wrong_max_risk_rejected(self):
  self.u['q_usd']=[2.,3.];self.u['risk_value_usd']=2.5
  with self.assertRaisesRegex(ValueError,'worst-case'):self.run_model()
