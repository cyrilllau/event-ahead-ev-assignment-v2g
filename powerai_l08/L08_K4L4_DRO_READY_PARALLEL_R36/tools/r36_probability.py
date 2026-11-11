from r36_common import *
from scipy import sparse as sp
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB
@functools.lru_cache(None)
def moments():
 d=R29/'05_DRO_MOMENTS';co=read(d/'COMPRESSION_COMPLETE.json');assert co['raw']==322311 and co['effective']==45139
 for name,key in [('SUPPORT_UNIQUE.npy','rows_sha'),('SUPPORT_INVERSE.npy','inverse_sha'),('EFFECTIVE_INTERVALS.npz','bounds_sha')]:assert sha(d/name)==co[key]
 raw=np.load(d/'X_SUPPORT.npy',mmap_mode='r');phi=np.load(d/'SUPPORT_UNIQUE.npy',mmap_mode='r');inv=np.load(d/'SUPPORT_INVERSE.npy');bounds=loadnp(d/'EFFECTIVE_INTERVALS.npz');rb=loadnp(d/'CALIBRATED_RAW_BOUNDS_K4L4.npz');assert raw.shape==(400,322311) and phi.shape==(45139,400)
 low=np.full(45139,-np.inf);high=np.full(45139,np.inf);np.maximum.at(low,inv,rb['lower']);np.minimum.at(high,inv,rb['upper']);assert np.array_equal(low,bounds['lower']) and np.array_equal(high,bounds['upper'])
 for i in range(0,322311,2048):assert np.array_equal(raw[:,i:i+2048].T,phi[inv[i:i+2048]])
 scale=np.maximum(np.max(np.abs(phi),axis=1),np.maximum(np.maximum(abs(low),abs(high)),1e-3))
 return dict(phi=sp.csr_matrix(phi/scale[:,None]),lower=low/scale,upper=high/scale,scale=scale,raw=raw,raw_bounds=rb,inverse=inv,identity=digest(dict(compression=co,raw_bounds=sha(d/'CALIBRATED_RAW_BOUNDS_K4L4.npz'),support=sha(d/'X_SUPPORT.npy'))))
def raw_scan(p,a):
 assert np.asarray(p).shape==(400,) and np.isfinite(p).all();ex=p@a['raw'];rb=a['raw_bounds'];v=max(float(np.max(rb['lower']-ex,initial=0)),float(np.max(ex-rb['upper'],initial=0)),0.)
 return dict(raw_residual=v,simplex_residual=abs(float(p.sum())-1),min_p=float(p.min()),all_raw_rows=322311)
def dual_certificate(q,b0,bp,bm,a):
 assert len(q)==400 and len(bp)==len(bm)==45139 and np.isfinite(q).all()
 bp=np.maximum(bp,0);bm=np.maximum(bm,0);planes=b0+np.asarray(a['phi'].T@(bp-bm)).ravel();magnitude=abs(b0)+float(abs(a['upper']*bp).sum()+abs(a['lower']*bm).sum())
 guard=max(1e-7,256*np.finfo(float).eps*max(1,magnitude));shift=max(0,float(np.max(q-planes)))+guard;b0+=shift;planes+=shift
 upper=float(np.nextafter(b0+a['upper']@bp-a['lower']@bm+guard,np.inf));assert np.min(planes-q)>=0
 return dict(beta0=float(b0),plus=bp,minus=bm,planes=planes,risk_upper=upper,intercept_correction=shift,guard=guard,max_plane_violation=float(np.max(q-planes)))
def split_point(q,out,a):
 out=Path(out);normal=max(1.,float(abs(q).max()));key=digest(dict(q=q,risk=a['identity']))
 if (out/'SPLIT_POINT.npz').exists():assert read(out/'SPLIT_END.json')['key']==key;return loadnp(out/'SPLIT_POINT.npz')
 assert not (out/'SPLIT_BEGIN.json').exists(),'PRESERVE_INCOMPLETE_SPLIT_NO_BLIND_RETRY'
 n=len(a['lower']);m=gp.Model('R36_FIXED_Q_SPLIT');m.Params.Threads=1;m.Params.Method=2;m.Params.Crossover=1;m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.BarConvTol=1e-10;m.Params.LogFile=str(out/'SPLIT_NATIVE.log')
 b0=m.addVar(lb=-GRB.INFINITY,name='beta0');bp=m.addMVar(n,name='plus');bm=m.addMVar(n,name='minus');plane=m.addConstr(b0+a['phi'].T@(bp-bm)>=q/normal);m.setObjective(b0+a['upper']@bp-a['lower']@bm);m.update();dump(out/'SPLIT_BEGIN.json',dict(epoch=time.time(),key=key,rows=m.NumConstrs,variables=m.NumVars));m.optimize()
 dump(out/'SPLIT_END.json',dict(key=key,epoch=time.time(),status=m.Status,runtime=m.Runtime,objective=m.ObjVal*normal if m.SolCount else None));assert m.Status==GRB.OPTIMAL
 point=dict(beta0=np.array(b0.X*normal),plus=bp.X*normal,minus=bm.X*normal,p=plane.Pi,objective=np.array(m.ObjVal*normal));npz(out/'SPLIT_POINT.npz',**point);m.dispose();return point
def solve_risk(q,out,compare_split=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);a=moments();key=digest(dict(q=np.asarray(q),risk=a['identity']))
 if (out/'COMPLETE.json').exists():
  rec=read(out/'COMPLETE.json');assert rec['key']==key and rec['certificate_sha']==sha(out/'CERTIFICATE.npz');return rec
 q=np.asarray(q,float);assert q.shape==(400,) and np.isfinite(q).all();normal=max(1.,float(abs(q).max()));A=sp.vstack([a['phi'],-a['phi']],format='csc');b=np.r_[a['upper'],-a['lower']]
 if (out/'PRIMAL_RESULT.npz').exists():
  old=read(out/'PRIMAL_END.json');assert old['key']==key and old['success'];z=loadnp(out/'PRIMAL_RESULT.npz');p,du,be=z['p'],z['inequality_dual'],z['equality_dual']
 elif (out/'PRIMAL_END.json').exists() and read(out/'PRIMAL_END.json')['success']:
  assert read(out/'PRIMAL_END.json')['key']==key
  # The completed HiGHS call is not repeated. The independently planned split LP
  # supplies both beta and its row-dual probability vector, then both are audited.
  z=split_point(q,out,a);p=z['p'];du=-np.r_[z['plus'],z['minus']]/normal;be=np.array([-float(z['beta0'])/normal]);npz(out/'PRIMAL_RESULT.npz',p=p,inequality_dual=du,equality_dual=be)
  dump(out/'PRIMAL_POINT_RECOVERY.json',dict(source='Independent planned Gurobi split dual row multipliers',original_HiGHS_not_repeated=True,original_vector_not_saved=True,reason='Original npz path argument named p conflicted with probability array keyword'))
 else:
  dump(out/'PRIMAL_BEGIN.json',dict(epoch=time.time(),key=key,owner=ctrl.proc(os.getpid()),method='highs-ipm',threads=1,probabilities=400,moment_sides=len(b)));start=time.time()
  res=linprog(-q/normal,A_ub=A,b_ub=b,A_eq=np.ones((1,400)),b_eq=[1],bounds=(0,None),method='highs-ipm',options=dict(threads=1,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-10,disp=True))
  dump(out/'PRIMAL_END.json',dict(key=key,status=res.status,success=res.success,message=res.message,elapsed=time.time()-start,epoch=time.time()));assert res.success,res.message
  p=res.x;du=res.ineqlin.marginals;be=res.eqlin.marginals;npz(out/'PRIMAL_RESULT.npz',p=p,inequality_dual=du,equality_dual=be)
 check=raw_scan(p,a);assert check['raw_residual']<=1e-8 and check['simplex_residual']<=1e-9 and check['min_p']>=0,check
 n=45139;cert=dual_certificate(q,-float(be[0])*normal,-du[:n]*normal,-du[n:]*normal,a);risk=float(p@q);assert -1e-5<=cert['risk_upper']-risk<=1e-3,(risk,cert['risk_upper'])
 if compare_split:
  z=split_point(q,out,a);dc=dual_certificate(q,float(z['beta0']),z['plus'],z['minus'],a);assert abs(float(z['objective'])-risk)<=1e-3 and dc['risk_upper']-risk<=1e-3
  npz(out/'INDEPENDENT_SPLIT.npz',beta0=dc['beta0'],plus=dc['plus'],minus=dc['minus'])
 npz(out/'CERTIFICATE.npz',p=p,beta0=cert['beta0'],plus=cert['plus'],minus=cert['minus'],planes=cert['planes'],q=q,scale=a['scale'])
 rec=dict(passed=True,key=key,risk_identity=a['identity'],risk_primal=risk,risk_upper=cert['risk_upper'],signed_gap=cert['risk_upper']-risk,raw_scan=check,dual_max_violation=cert['max_plane_violation'],intercept_guard=cert['intercept_correction'],certificate_sha=sha(out/'CERTIFICATE.npz'),all_support=400,independent_split=compare_split,signed_formal=False)
 dump(out/'COMPLETE.json',rec);return rec
def ambiguity_tests():
 a=moments();w=loadnp(R29/'06_NONEMPTY/FEASIBLE_WITNESS.npz')['p'];check=raw_scan(w,a);assert check['raw_residual']<=1e-8 and check['simplex_residual']<=1e-9
 q=loadnp(R/'evaluations/ZERO/Q.npz')['qU'];res=solve_risk(q,R/'probability/ZERO',True)
 dump(R/'probability/REPRESENTATION.json',dict(formal='SPLIT_BETA',signed_used=False,reason='Signed production interfaces and same-pool benchmark are not used; retain original split formulation',all400planes=True))
 return dict(passed=True,raw=322311,effective=45139,witness_positive=int((w>0).sum()),witness_check=check,ZERO_risk=res)
