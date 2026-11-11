import copy
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.dual_cut_bound import build_dual_cut,build_weighted_dual_problem
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.relaxation_bound import solve_relaxation_bound


def scenario(name='s0'):
    # min z subject to z>=2-x, z>=x, z>=0: Q(x)=max(2-x,x).
    return dict(A_ub=csr_matrix([[-1.],[-1.]]),A_eq=csr_matrix((0,1)),Bx_ub=csr_matrix([[1.],[-1.]]),
        Bx_eq=csr_matrix((0,1)),Br_ub=csr_matrix((2,0)),Br_eq=csr_matrix((0,0)),b_ub=np.array([-2.,0.]),b_eq=np.array([]),
        c=np.array([1.]),lower_bounds=np.array([0.]),upper_bounds=np.array([np.inf]),scenario_id=name,template_sha256=name+'-fixed')


def dual(which=0):return dict(inequality=np.array([-1.,0.] if which==0 else [0.,-1.]),equality=np.array([]),lower=np.array([0.]),upper=np.array([0.]))


class DualCutTests(unittest.TestCase):
    def cut(self,name='s0',which=0,constant=0.):return build_dual_cut(scenario(name),dual(which),scenario_constant_usd=constant,anchor_first_stage=[0. if which==0 else 2.],anchor_q_usd=2.+constant)

    def test_lower_bound_at_distinct_plans_not_only_anchor(self):
        record,g=self.cut();self.assertTrue(record['passed']);self.assertEqual(record['intercept_usd'],2.);np.testing.assert_array_equal(g,[-1.])
        for x in (0.,.5,1.,1.5,2.):self.assertLessEqual(record['intercept_usd']+g[0]*x,max(2-x,x))
        self.assertLess(record['intercept_usd']+g[0]*1.5,1.5)

    def test_constant_once_and_second_dual_orientation(self):
        record,g=self.cut(which=1,constant=7.)
        self.assertTrue(record['passed']);self.assertEqual(record['intercept_usd'],7.);np.testing.assert_array_equal(g,[1.])
        self.assertEqual(record['anchor_affine_value_usd'],9.)

    def test_bound_dual_and_equality_constants(self):
        t=dict(A_ub=csr_matrix((0,2)),A_eq=csr_matrix([[1.,0.]]),Bx_ub=csr_matrix((0,1)),Bx_eq=csr_matrix([[2.]]),
            Br_ub=csr_matrix((0,0)),Br_eq=csr_matrix((1,0)),b_ub=[],b_eq=[1.],c=[3.,-2.],lower_bounds=[-np.inf,-np.inf],upper_bounds=[np.inf,4.],scenario_id='eq')
        d=dict(inequality=[],equality=[3.],lower=[0.,0.],upper=[0.,-2.])
        record,g=build_dual_cut(t,d,scenario_constant_usd=5.,anchor_first_stage=[1.],anchor_q_usd=6.)
        self.assertTrue(record['passed']);self.assertEqual(record['intercept_usd'],0.);np.testing.assert_array_equal(g,[6.])

    def test_bad_stationarity_sign_infinite_marginal_and_anchor_fail(self):
        for change in ('station','sign','infinite','anchor'):
            d=dual();q=2.
            if change=='station':d['inequality'][0]-=2e-8
            elif change=='sign':d['inequality'][1]=1.
            elif change=='infinite':d['upper'][0]=1.
            else:q+=1.
            record,_=build_dual_cut(scenario(),d,anchor_first_stage=[0.],anchor_q_usd=q)
            with self.subTest(change=change):self.assertFalse(record['passed'])

    def combined(self,weights=(.25,.75,0.)):
        a=ProbabilitySet(('s0','s1','unused'),('mean',),np.array([[0.],[1.],[2.]]),np.array([.75]),np.array([.75]))
        first=dict(matrix=csr_matrix((0,1)),row_lower=[],row_upper=[],c=np.array([.1]),lower_bounds=[0.],upper_bounds=[2.],integrality=[0],objective_constant_usd=3.)
        cuts={0:self.cut('s0'),1:self.cut('s1',which=1)}
        return first,a,cuts,build_weighted_dual_problem(first,a.support_ids,weights,cuts,probability_set=a)

    def test_feasible_p_weighting_first_stage_cost_and_real_small_lp(self):
        first,a,cuts,(problem,record)=self.combined()
        np.testing.assert_allclose(problem['c'],[.6]);self.assertEqual(problem['objective_constant_usd'],3.5)
        self.assertEqual(record['positive_indices'],[0,1]);self.assertEqual(record['exact_zero_weight_count'],1)
        self.assertIs(problem['matrix'],first['matrix']);np.testing.assert_array_equal(first['c'],[.1])
        result,_=solve_relaxation_bound(problem,time_limit_seconds=3)
        self.assertEqual(result['status'],'OPTIMAL');self.assertAlmostEqual(result['lower_bound_usd'],3.5)
        for x in (0.,.5,1.,1.5,2.):self.assertLessEqual(problem['c'][0]*x+problem['objective_constant_usd'],3.+.1*x+max(2-x,x))

    def test_probability_axis_negative_mass_and_moment_violation_rejected(self):
        first,a,cuts,_=self.combined()
        for p in ((.5,.5,0.),(.25,.75,-1e-16),(.25,.74,0.)):
            with self.assertRaises(ValueError):build_weighted_dual_problem(first,a.support_ids,p,cuts,probability_set=a)
        with self.assertRaises(ValueError):build_weighted_dual_problem(first,tuple(reversed(a.support_ids)),[.25,.75,0.],cuts,probability_set=a)

    def test_missing_positive_extra_zero_wrong_scenario_and_gradient_tamper_rejected(self):
        first,a,cuts,_=self.combined()
        for bad in ({0:cuts[0]},{**cuts,2:cuts[0]},{0:cuts[1],1:cuts[0]}):
            with self.assertRaises(ValueError):build_weighted_dual_problem(first,a.support_ids,[.25,.75,0.],bad,probability_set=a)
        record,g=cuts[0];g[0]+=1.
        with self.assertRaisesRegex(ValueError,'gradient changed'):build_weighted_dual_problem(first,a.support_ids,[.25,.75,0.],cuts,probability_set=a)

    def test_reserve_coupling_included_without_plan_fixing(self):
        t=scenario();t['Br_ub']=csr_matrix([[2.],[-2.]]);t['Br_eq']=csr_matrix((0,1))
        record,g=build_dual_cut(t,dual(),anchor_first_stage=[0.,0.],anchor_q_usd=2.)
        np.testing.assert_array_equal(g,[-1.,-2.]);self.assertTrue(record['passed'])

    def test_saved_intercept_mutation_rejected(self):
        first,a,cuts,_=self.combined();cuts[0][0]['intercept_usd']+=1.
        with self.assertRaisesRegex(ValueError,'cut record changed'):build_weighted_dual_problem(first,a.support_ids,[.25,.75,0.],cuts,probability_set=a)


if __name__=='__main__':unittest.main()
