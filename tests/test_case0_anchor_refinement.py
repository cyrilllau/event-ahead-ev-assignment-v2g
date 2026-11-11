import unittest
import numpy as np
from scipy.sparse import csr_matrix
from tests.test_dual_cut_bound import scenario
from evcs_experiments.case0_anchor_refinement import solve_anchor_q,audit_anchor_q,build_cut_bundle
from evcs_experiments.relaxation_bound import solve_relaxation_bound
class AnchorTests(unittest.TestCase):
 def test_two_anchors_recover_known_SAA_and_ARO_minimum(self):
  f=dict(matrix=csr_matrix((0,1)),c=np.array([.1]),integrality=np.array([0]),lower_bounds=np.array([0.]),upper_bounds=np.array([2.]),row_lower=np.array([]),row_upper=np.array([]),objective_constant_usd=0.)
  groups=[]
  for x in (0.,2.):
   group={}
   for i in range(2):
    c,g,a,v,r=solve_anchor_q(scenario('s'+str(i)),np.array([x]),2);self.assertTrue(a['passed']);group[i]=(c,g)
   groups.append(group)
  for risk in ('SAA','ARO'):
   m,_=build_cut_bundle(f,['s0','s1'],groups,risk);r,v=solve_relaxation_bound(m,time_limit_seconds=2);self.assertAlmostEqual(r['lower_bound_usd'],1.1)
   m0,_=build_cut_bundle(f,['s0','s1'],groups[:1],risk);r0,_=solve_relaxation_bound(m0,time_limit_seconds=2);self.assertAlmostEqual(r0['lower_bound_usd'],.2)
 def test_invalid_new_primal_and_dual_cannot_create_cut(self):
  t=scenario();c,g,a,v,r=solve_anchor_q(t,np.array([0.]),2);d={k:v[k].copy() for k in ('inequality','equality','lower','upper')}
  with self.assertRaisesRegex(ValueError,'primal'):audit_anchor_q(t,[0.],[0.],0.,d)
  d['inequality'][0]=0.
  with self.assertRaisesRegex(ValueError,'dual'):audit_anchor_q(t,[0.],[2.],2.,d)
 def test_incomplete_group_rejected(self):
  f=dict(c=[0.])
  with self.assertRaisesRegex(ValueError,'Incomplete'):build_cut_bundle(f,['s0'],[{}],'SAA')
