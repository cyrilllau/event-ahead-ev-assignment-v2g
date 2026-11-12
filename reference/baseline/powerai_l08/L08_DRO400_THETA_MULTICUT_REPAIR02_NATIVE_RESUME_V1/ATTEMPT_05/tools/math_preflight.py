#!/usr/bin/env python3
"""Synthetic mathematical tests only. Does not load or solve the user's EV model.
Requires numpy/scipy. Run: python math_preflight.py --output results.json
"""
from __future__ import annotations
import argparse, itertools, json, math, unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint, linprog


def frac(x):
    x=float(x)
    if not math.isfinite(x): raise ValueError('nonfinite')
    return F.from_float(x)


def downward(x: F) -> float:
    v=float(x)
    if not math.isfinite(v): raise ValueError('float overflow')
    if frac(v)>x: v=math.nextafter(v,-math.inf)
    return v


def aggregate_safely(p, a, b, lo, hi):
    """Exact binary-float arithmetic + safe downward intercept.
    This validates simplex only. Production MUST additionally certify original P.
    Inputs are already globally valid scenario affine lower bounds.
    """
    p=np.asarray(p,float); a=np.asarray(a,float); b=np.asarray(b,float)
    lo=np.asarray(lo,float); hi=np.asarray(hi,float)
    if p.ndim!=1 or a.shape!=p.shape or b.ndim!=2 or b.shape[0]!=len(p): raise ValueError('dimension')
    if lo.shape!=(b.shape[1],) or hi.shape!=lo.shape: raise ValueError('box dimension')
    if any(not np.isfinite(v).all() for v in (p,a,b,lo,hi)): raise ValueError('nonfinite')
    if np.any(p<0) or sum(map(frac,p),F(0))!=1: raise ValueError('not exact simplex')
    if np.any(lo>hi): raise ValueError('invalid box')
    pf=list(map(frac,p)); af=sum((w*frac(z) for w,z in zip(pf,a)),F(0))
    bf=[sum((pf[s]*frac(b[s,j]) for s in range(len(p))),F(0)) for j in range(b.shape[1])]
    stored=np.array([float(v) for v in bf]); excess=F(0)
    for j in range(len(bf)):
        d=frac(stored[j])-bf[j]
        excess+=max(d*frac(lo[j]),d*frac(hi[j]))
    alpha=downward(af-excess)
    return alpha,stored,{'exact_alpha':str(af),'coefficient_adjustment':str(excess),'stored_alpha_hex':alpha.hex()}


def vertices():
    # Original P: simplex, 0<=p_i<=3/4, 1/4<=p0+p1<=3/4.
    A=np.vstack([-np.eye(4),np.eye(4),[1,1,0,0],[-1,-1,0,0]])
    h=np.r_[np.zeros(4),np.full(4,.75),.75,-.25]
    pts=[]
    for ids in itertools.combinations(range(len(h)),3):
        M=np.vstack([np.ones(4),A[list(ids)]]); rhs=np.r_[1,h[list(ids)]]
        if abs(np.linalg.det(M))<1e-10:continue
        p=np.linalg.solve(M,rhs)
        if np.max(A@p-h)<=1e-10:
            p[np.abs(p)<1e-12]=0
            if not any(np.max(np.abs(p-q))<1e-10 for q in pts): pts.append(p)
    return np.array(pts)

P=vertices()
# Q_s(x) = max of three affine functions, one of them zero. x=(z1,z2,e).
A=np.array([[0,14,8],[0,10,13],[0,16,6],[0,9,15]],float)
B=np.array([[[0,0,0],[-9,-2,-2],[-1,-6,-.5]],
            [[0,0,0],[-2,-8,-1],[-7,-1,-2]],
            [[0,0,0],[-6,-7,-1.5],[-1,-3,-.5]],
            [[0,0,0],[-7,-4,-1],[-2,-8,-1.5]]],float)
C=np.array([3.6,3.3,.65]); BOXLO=np.zeros(3); BOXHI=np.array([1,1,3.])


def recourse(x):
    vals=A+np.einsum('sij,j->si',B,x); k=vals.argmax(axis=1)
    return vals[np.arange(4),k],A[np.arange(4),k],B[np.arange(4),k]


def original_extensive(relax=False):
    # Variables x(3), q(4), rho(1); one common first stage and all scenarios.
    rows=[]; lower=[]
    r=np.zeros(8);r[:3]=[2,2,-1];rows.append(r);lower.append(0)
    for s in range(4):
        for t in range(3):
            r=np.zeros(8);r[:3]=-B[s,t];r[3+s]=1;rows.append(r);lower.append(A[s,t])
    for p in P:
        r=np.zeros(8);r[3:7]=-p;r[-1]=1;rows.append(r);lower.append(0)
    obj=np.r_[C,np.zeros(4),1]
    res=milp(obj,integrality=np.r_[([0,0] if relax else [1,1]),np.zeros(6)],
        bounds=Bounds(np.zeros(8),np.r_[BOXHI,np.full(5,np.inf)]),
        constraints=LinearConstraint(np.array(rows),lower,np.inf),options={'mip_rel_gap':0.0})
    if not res.success: raise RuntimeError(res.message)
    return res


def run_aggregate():
    cuts=[]; history=[]; best_u=math.inf; best_x=None
    for k in range(40):
        rows=[np.array([2.,2.,-1.,0.])]; lower=[0.]
        for alpha,beta in cuts: rows.append(np.r_[-beta,1]);lower.append(alpha)
        m=milp(np.r_[C,1.],integrality=[1,1,0,0],bounds=Bounds(np.zeros(4),np.r_[BOXHI,np.inf]),
               constraints=LinearConstraint(np.array(rows),lower,np.inf),options={'mip_rel_gap':0.0})
        if not m.success:raise RuntimeError(m.message)
        x=m.x[:3];q,a,b=recourse(x);worst=P@(q);p=P[np.argmax(worst)]
        u=float(C@x+max(worst))
        if u<best_u:best_u=u;best_x=x.copy()
        alpha,beta,_=aggregate_safely(p,a,b,BOXLO,BOXHI)
        history.append({'round':k,'master_lower':float(m.fun),'point_upper':u,'best_upper':best_u,
                        'cut_at_point':float(alpha+beta@x),'risk':float(max(worst))})
        if best_u-m.fun<=1e-7: return {'objective':best_u,'x':best_x.tolist(),'history':history,'cuts':cuts}
        if alpha+beta@x-m.x[3]<=1e-8:raise AssertionError('uncertified stall')
        cuts.append((alpha,beta))
    raise RuntimeError('synthetic did not converge')


class Tests(unittest.TestCase):
    def test_01_probabilities(self):
        self.assertGreater(len(P),1)
        for p in P:
            self.assertEqual(sum(map(frac,p),F(0)),1)
            self.assertTrue(np.all(p>=0));self.assertTrue(np.max(p)<=.75)
            self.assertGreaterEqual(p[:2].sum(),.25);self.assertLessEqual(p[:2].sum(),.75)
    def test_02_extensive_equals_aggregate(self):
        direct=original_extensive();r=run_aggregate()
        self.assertAlmostEqual(direct.fun,r['objective'],7)
        self.assertGreater(len(r['history']),1)
    def test_03_all_bound_directions(self):
        opt=original_extensive().fun
        for h in run_aggregate()['history']:
            self.assertLessEqual(h['master_lower'],opt+1e-7)
            self.assertGreaterEqual(h['best_upper'],opt-1e-7)
    def test_04_cuts_valid_on_grid(self):
        for al,be in run_aggregate()['cuts']:
            for z in itertools.product((0.,1.),repeat=2):
                for e in np.linspace(0,min(3,2*sum(z)),13):
                    x=np.array([*z,e]);q,*_=recourse(x)
                    self.assertLessEqual(al+be@x,np.max(P@q)+1e-9)
    def test_05_point_support_is_tight(self):
        for h in run_aggregate()['history']:
            self.assertAlmostEqual(h['cut_at_point'],h['risk'],8)
    def test_06_probability_set_not_simplex(self):
        q=np.array([100.,0,0,0]);self.assertAlmostEqual(max(P@q),75.);self.assertNotEqual(max(P@q),max(q))
    def test_07_rounded_aggregate_exact_at_corners(self):
        p=[.25,.75];a=[.1,.9];b=np.array([[.1,-.7],[.3,.2]])
        lo=[-10,-3];hi=[20,9];aa,bb,_=aggregate_safely(p,a,b,lo,hi)
        for x in itertools.product(*zip(lo,hi)):
            lhs=frac(aa)+sum((frac(v)*frac(xx) for v,xx in zip(bb,x)),F(0))
            rhs=sum((frac(p[s])*(frac(a[s])+sum((frac(v)*frac(xx) for v,xx in zip(b[s],x)),F(0))) for s in range(2)),F(0))
            self.assertLessEqual(lhs,rhs)
    def test_08_bad_simplex_rejected(self):
        for p in ([.5,.5000000001],[-.25,1.25],[float('nan'),1]):
            with self.assertRaises(ValueError):aggregate_safely(p,[1,2],[[1],[2]],[0],[1])
    def test_09_finite_bound_term(self):
        # min -y for 0<=y<=2 has lower bound -2, not 0 from missing bound term.
        c=np.array([-1.]);ylo=np.array([0.]);yhi=np.array([2.]);box=np.minimum(c*ylo,c*yhi).sum()
        self.assertEqual(box,-2.)
    def test_10_missing_scenario_is_not_upper(self):
        p=np.array([.25,.75]);q=np.array([0.,100.]);self.assertEqual(p@q,75.);self.assertNotEqual(p@q,p@np.zeros(2))
    def test_11_integer_stage_not_relaxed(self):
        self.assertLess(original_extensive(True).fun,original_extensive().fun-1e-6)
    def test_12_exact_duplicate_cost_not_probability_merge(self):
        # identical Q values can share one oracle, while their probabilities remain distinct.
        q=np.array([5.,5.,1.,0.]);g=np.array([0,0,1,2]);r=np.array([5.,1.,0.])
        self.assertTrue(np.array_equal(q,r[g]));self.assertAlmostEqual(max(P@q),max(P@r[g]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    dr=original_extensive(); cr=original_extensive(True);ag=run_aggregate();ag.pop('cuts')
    payload={'synthetic_only':True,'user_EV_optimization_run':False,'gurobi_used':False,
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'passed':result.wasSuccessful(),'scenario_count':4,'probability_vertices':len(P),
             'direct_integer_objective':float(dr.fun),'direct_LP_objective':float(cr.fun),
             'aggregated_decomposition':ag}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)
