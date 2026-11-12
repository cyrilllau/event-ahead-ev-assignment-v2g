import copy
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.saa_dual_bound import build_saa_dual_bound
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.relaxation_bound import solve_relaxation_bound
from tests.test_dual_cut_bound import scenario,dual

class SAADualBoundTests(unittest.TestCase):
    def fixture(self):
        first=dict(matrix=csr_matrix((0,1)),row_lower=[],row_upper=[],c=np.array([.1]),lower_bounds=[0.],upper_bounds=[2.],integrality=[0],objective_constant_usd=3.)
        cuts={i:build_dual_cut(scenario('s'+str(i)),dual(i),scenario_constant_usd=7.*i,anchor_first_stage=[2.*i],anchor_q_usd=2.+7.*i) for i in range(2)}
        return first,cuts
    def test_uniform_known_bound_constants_once_and_original_domain(self):
        p,cuts=self.fixture();m,r=build_saa_dual_bound(p,['s0','s1'],cuts,constants_usd=[0,7])
        self.assertIs(m['matrix'],p['matrix']);self.assertIs(m['integrality'],p['integrality'])
        self.assertEqual(m['objective_constant_usd'],7.5);np.testing.assert_allclose(m['c'],[.1])
        result,_=solve_relaxation_bound(m,time_limit_seconds=3)
        self.assertAlmostEqual(result['lower_bound_usd'],7.5)
        self.assertEqual(r['risk_certificate_kind'],'fixed_empirical');self.assertFalse(r['scientific_acceptance'])
        # Exact full SAA objective is 3+.1*x+max(2-x,x)+3.5, whose optimum is7.6.
        for x in np.linspace(0,2,11):self.assertLessEqual(m['c'][0]*x+m['objective_constant_usd'],6.5+.1*x+max(2-x,x)+1e-12)
    def test_wrong_support_constant_gradient_and_missing_cut_rejected(self):
        p,c=self.fixture()
        for ids,cs,k in [(['s1','s0'],[0,7],c),(['s0','s1'],[0,8],c),(['s0','s1'],[0,7],{0:c[0]})]:
            with self.assertRaises(ValueError):build_saa_dual_bound(p,ids,k,constants_usd=cs)
        d=copy.deepcopy(c);d[0][1][0]+=1
        with self.assertRaisesRegex(ValueError,'Gradient'):build_saa_dual_bound(p,['s0','s1'],d,constants_usd=[0,7])
    def test_no_ambiguity_or_nonuniform_override(self):
        p,c=self.fixture()
        with self.assertRaises(TypeError):build_saa_dual_bound(p,['s0','s1'],c,constants_usd=[0,7],probability_set=object())
    def test_original_cut_bytes_and_cost_unchanged(self):
        p,c=self.fixture();before=copy.deepcopy(c)
        build_saa_dual_bound(p,['s0','s1'],c,constants_usd=[0,7])
        np.testing.assert_array_equal(p['c'],[.1]);self.assertEqual(c[0][0],before[0][0])
