import copy
import unittest
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.epigraph_master import certify_box_floor,build_epigraph_master
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.relaxation_bound import solve_relaxation_bound,coefficient_sha256


def fixture(sid='s0',constant=5.):
    # min 2*y+5, y>=3-5*x, 1<=y<=10. True Q=max(6-10*x,2)+5.
    t=dict(A_ub=csr_matrix([[-1.]]),A_eq=csr_matrix((0,1)),Bx_ub=csr_matrix([[5.]]),Bx_eq=csr_matrix((0,1)),
        Br_ub=csr_matrix((1,0)),Br_eq=csr_matrix((0,0)),b_ub=[-3.],b_eq=[],c=[2.],lower_bounds=[1.],upper_bounds=[10.],
        scenario_id=sid,template_sha256=sid+'-original')
    d=dict(inequality=[-2.],equality=[],lower=[0.],upper=[0.])
    cut=build_dual_cut(t,d,scenario_constant_usd=constant,anchor_first_stage=[0.],anchor_q_usd=6.+constant)
    floor=certify_box_floor(t,scenario_constant_usd=constant)
    return t,cut,floor


def first(integer=1):return dict(matrix=csr_matrix([[1.]]),row_lower=[0.],row_upper=[1.],c=[1.],lower_bounds=[0.],upper_bounds=[1.],integrality=[integer],objective_constant_usd=3.)


class EpigraphTests(unittest.TestCase):
    def model(self,integer=1):
        t,c,f=fixture();a=ProbabilitySet(('s0',),(),np.empty((1,0)),[],[])
        p=first(integer);return p,a,{0:c},{0:f},build_epigraph_master(p,a.support_ids,{0:c},probability_set=a,floors={0:f})

    def test_floor_required_and_strengthens_bound_at_new_plan(self):
        p,a,c,f,(m,r)=self.model()
        self.assertEqual(f[0]['floor_usd'],7.);self.assertEqual(c[0][0]['intercept_usd'],11.)
        # Old cut alone at x=1 equals1; floor raises the true lower cost to7.
        self.assertEqual(float(c[0][0]['intercept_usd']+c[0][1]@np.array([1.])),1.)
        with self.assertRaises(ValueError):build_epigraph_master(p,a.support_ids,c,probability_set=a,floors={})
        z=np.array([1.,7.]);self.assertTrue(np.all(m['matrix']@z>=m['row_lower']))
        z=np.array([1.,1.]);self.assertFalse(np.all(m['matrix']@z>=m['row_lower']))

    def test_integer_and_continuous_against_original_recourse_hand_optimum(self):
        p,a,c,f,(m,r)=self.model()
        answer=milp(m['c'],integrality=m['integrality'],bounds=Bounds(m['lower_bounds'],m['upper_bounds']),
            constraints=LinearConstraint(m['matrix'],m['row_lower'],m['row_upper']),options={'time_limit':3.,'threads':1})
        self.assertEqual(answer.status,0)
        # Original binary alternatives: x=0 costs14, x=1 costs11, including Fpre constant3.
        original=min(3.+x+2.*max(1.,3.-5*x)+5. for x in (0.,1.))
        self.assertAlmostEqual(answer.fun+m['objective_constant_usd'],original);self.assertAlmostEqual(original,11.)
        lp,_=solve_relaxation_bound(m,time_limit_seconds=3.)
        self.assertEqual(lp['status'],'OPTIMAL');self.assertAlmostEqual(lp['lower_bound_usd'],10.4)
        self.assertLessEqual(lp['lower_bound_usd'],original)
        self.assertEqual(r['variable_slices']['theta0'],[1,2])

    def test_nonzero_moment_features_dual_signs_and_full_axis(self):
        p=first();_,c0,f0=fixture('s0',5.);_,c1,f1=fixture('s1',9.)
        a=ProbabilitySet(('s0','s1'),('p1',),np.array([[0.],[1.]]),[.25],[.25])
        m,r=build_epigraph_master(p,a.support_ids,{0:c0,1:c1},probability_set=a,floors={0:f0,1:f1})
        self.assertEqual(m['matrix'].shape,(5,4));np.testing.assert_array_equal(m['c'],[1.,1.,-.25,.25])
        np.testing.assert_array_equal(m['matrix'].toarray()[1:,1:],[[1.,0.,0.],[1.,-1.,1.],[1.,0.,0.],[1.,-1.,1.]])
        result,_=solve_relaxation_bound(m,time_limit_seconds=3.);self.assertEqual(result['status'],'OPTIMAL')
        self.assertAlmostEqual(result['lower_bound_usd'],11.4) # additive weighted constant difference .25*4
        with self.assertRaises(ValueError):build_epigraph_master(p,('s1','s0'),{0:c0,1:c1},probability_set=a,floors={0:f0,1:f1})

    def test_original_first_stage_immutable_including_integrality_and_constant(self):
        p,a,c,f,(m,r)=self.model();before=coefficient_sha256(p)
        np.testing.assert_array_equal(m['matrix'].toarray()[:1,:1],p['matrix'].toarray())
        for key in ('lower_bounds','upper_bounds','integrality','c'):np.testing.assert_array_equal(m[key][:1],p[key])
        self.assertEqual(m['objective_constant_usd'],3.);self.assertEqual(before,r['original_first_stage_coefficient_sha256'])
        self.assertEqual(before,coefficient_sha256(p));self.assertEqual(m['integrality'].tolist(),[1,0])
        _,_,_,_,(continuous,_)=self.model(integer=0);self.assertEqual(continuous['integrality'].tolist(),[0,0])

    def test_general_sign_aware_floor_and_zero_cost_free_columns(self):
        t=dict(c=[2.,-3.,0.],lower_bounds=[1.,-np.inf,-np.inf],upper_bounds=[np.inf,4.,np.inf],scenario_id='s',template_sha256='t')
        f=certify_box_floor(t,scenario_constant_usd=5.);self.assertEqual(f['floor_usd'],-5.)
        self.assertEqual(f['negative_cost_columns'],1);self.assertTrue(f['negative_values_not_clipped'])
        for k,index,value in [('lower_bounds',0,-np.inf),('upper_bounds',1,np.inf)]:
            bad=copy.deepcopy(t);bad[k][index]=value
            with self.assertRaises(ValueError):certify_box_floor(bad)

    def test_full_axis_and_scenario_floor_identity_rejected(self):
        p,a,c,f,_=self.model()
        for cuts,floors in [({},f),({0:c[0],1:c[0]},f),(c,{0:f[0],1:f[0]}),({True:c[0]},f)]:
            with self.assertRaises(ValueError):build_epigraph_master(p,a.support_ids,cuts,probability_set=a,floors=floors)
        wrong=certify_box_floor(dict(c=[2.],lower_bounds=[1.],upper_bounds=[10.],scenario_id='other',template_sha256='wrong'),scenario_constant_usd=5.)
        with self.assertRaises(ValueError):build_epigraph_master(p,a.support_ids,c,probability_set=a,floors={0:wrong})

    def test_changed_cut_gradient_or_floor_checksums_rejected(self):
        for kind in ('cut','gradient','floor'):
            p,a,c,f,_=self.model()
            if kind=='cut':c[0][0]['intercept_usd']+=1.
            elif kind=='gradient':c[0][1][0]+=1.
            else:f[0]['floor_usd']+=1.
            with self.subTest(kind=kind),self.assertRaises(ValueError):build_epigraph_master(p,a.support_ids,c,probability_set=a,floors=f)

    def test_floor_must_match_template_and_constant_even_if_separately_valid(self):
        p,a,c,f,_=self.model();t,_,_=fixture();t['template_sha256']='different'
        wrong=certify_box_floor(t,scenario_constant_usd=5.)
        with self.assertRaises(ValueError):build_epigraph_master(p,a.support_ids,c,probability_set=a,floors={0:wrong})
        t,_,_=fixture();wrong=certify_box_floor(t,scenario_constant_usd=4.)
        with self.assertRaises(ValueError):build_epigraph_master(p,a.support_ids,c,probability_set=a,floors={0:wrong})


if __name__=='__main__':unittest.main()
