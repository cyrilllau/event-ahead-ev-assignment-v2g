"""Full per-scenario theta representation and unbounded-direction certificates."""
from pathlib import Path
from fractions import Fraction as F
import math, hashlib
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
from envelope_sparse import enclosure, digest, simplex

def down(q):
 v=float(q)
 return np.nextafter(v,-np.inf) if F(v)>q else v

def upper(q):
 v=float(q)
 return np.nextafter(v,np.inf) if F(v)<q else v

def compensate(alpha, source, stored, lo, hi):
 delta=(stored-source).tocsr();delta.eliminate_zeros()
 loss=sum((F(float(v))*F(float(hi[j] if v>=0 else lo[j])) for j,v in zip(delta.indices,delta.data)),F(0))
 # A nonnegative guard is deliberately conservative, never strengthens source.
 loss=max(F(0),loss)
 rhs=down(F(float(alpha))-loss)
 return rhs,dict(changed_coefficients=delta.nnz,box_loss_upper=upper(loss),alpha_loss=float(alpha)-rhs)

def add_supports(m, variables, nfirst, theta_col, B, alpha, ids, lo, hi, prefix):
 B=B.tocsr();nr=B.shape[0];nc=len(variables)
 A=sp.hstack([-B,sp.csr_matrix((nr,nc-nfirst))],format='lil')
 A[:,theta_col]=1.;A=A.tocsr()
 start=m.NumConstrs
 rows=m.addMConstr(A,gp.MVar.fromlist(variables),'>',alpha,name=prefix)
 m.update();cons=rows.tolist();stored=[];actualalpha=[];proof=[]
 for i,co in enumerate(cons):
  r=m.getRow(co);ii=[];vv=[];theta=0.
  for k in range(r.size()):
   j=r.getVar(k).index;v=r.getCoeff(k)
   if j==theta_col:theta=v
   else:
    if j>=nfirst:raise ValueError('SUPPORT_OTHER_UNBOUNDED_COLUMN')
    ii.append(j);vv.append(-v)
  if theta!=1. or co.Sense!='>':raise ValueError('SUPPORT_THETA_OR_SENSE')
  actual=sp.csr_matrix((vv,([0]*len(ii),ii)),shape=(1,nfirst));actual.sort_indices()
  rhs,pr=compensate(float(alpha[i]),B.getrow(i),actual,lo,hi)
  co.RHS=rhs;co.ConstrName=prefix+'/'+ids[i]
  stored.append(actual);actualalpha.append(rhs);proof.append(dict(pr,id=ids[i],row=start+i,actual_getRow=True))
 m.update()
 return sp.vstack(stored,format='csr'),np.asarray(actualalpha),proof

def add_probability(m,variables,rho_col,theta_map,p,name):
 p=simplex(p);items=sorted(theta_map.items());nz=p[p>0]
 k=0
 while nz.size and math.ldexp(float(nz.min()),k)<1e-12:k+=1
 scale=math.ldexp(1.,k)
 if scale>1e6 or (nz.size and np.max(np.ldexp(nz,k))>1e6):raise ValueError('PROBABILITY_FAITHFUL_SCALING_UNAVAILABLE')
 co=m.addConstr(scale*variables[rho_col]-gp.LinExpr([math.ldexp(float(p[s]),k) for s,j in items],[variables[j] for s,j in items])>=0,name=name)
 m.update();row=m.getRow(co);actual={row.getVar(i).index:row.getCoeff(i) for i in range(row.size())}
 expected={rho_col:scale,**{j:-math.ldexp(float(p[s]),k) for s,j in items if p[s]!=0}}
 if actual!=expected or co.RHS!=0 or co.Sense!='>':raise ValueError('PROBABILITY_ACTUAL_ROW_NOT_FAITHFUL')
 for s,j in items:
  if F(math.ldexp(float(p[s]),k))!=F(float(p[s]))*(2**k):raise ValueError('POWER2_NOT_EXACT')
 return dict(row=co.index,power2=k,scale=scale,faithful=True,coefficient_loss=0.,full400_p_sha=hashlib.sha256(p.tobytes()).hexdigest(),zero_scenario_mass=float(sum(p[s] for s in range(len(p)) if s not in theta_map)))

def model_arrays(m):
 m.update();v=m.getVars();r=m.getConstrs()
 return m.getA().tocsr(),dict(cost=np.asarray(m.getAttr('Obj',v)),lo=np.asarray(m.getAttr('LB',v)),hi=np.asarray(m.getAttr('UB',v)),vtype=np.asarray(m.getAttr('VType',v)),rhs=np.asarray(m.getAttr('RHS',r)),sense=np.asarray(m.getAttr('Sense',r)),objcon=np.array(m.ObjCon)),dict(rows=m.getAttr('ConstrName',r),cols=m.getAttr('VarName',v))

def lift(A,a,x,nfirst,rho_col,theta_map,stored):
 v=np.zeros(A.shape[1]);v[:nfirst]=x
 for sid,col in theta_map.items():
  B,alpha=stored[sid];_,u=enclosure(B,x)
  val=np.max(np.asarray(alpha,np.longdouble)+u,initial=np.longdouble(0))
  v[col]=max(0.,float(np.nextafter(float(val),np.inf)))
 # All nonphysical rows with rho have positive coefficient. No physical row uses rho.
 c=A[:,rho_col].tocoo();keep=c.row
 work=A[keep].copy().tolil();work[:,rho_col]=0.;work=work.tocsr()
 lower,_=enclosure(work,v);need=(np.asarray(a['rhs'][keep],np.longdouble)-lower)/np.asarray(c.data,np.longdouble)
 v[rho_col]=max(0.,float(np.nextafter(float(np.max(need,initial=0)),np.inf)))
 if not np.isfinite(v).all():raise ValueError('LIFT_NONFINITE')
 return v

def lagrangian(A,a,rawpi,nfirst,rho_col):
 if rawpi is None:return dict(LB=None,lower_certificate_verified=False,status='NO_PI'),None
 if len(rawpi)!=A.shape[0] or not np.isfinite(rawpi).all():raise ValueError('PI_IDENTITY_OR_FINITE')
 sense=a['sense'];pi=np.where(sense=='<',np.minimum(rawpi,0),np.where(sense=='>',np.maximum(rawpi,0),rawpi)).copy()
 Ac=A.tocsc();corrections=[];checks=[]
 # rho first: its rows fund all theta dual demands. Then scale only positive theta demand.
 for j in [rho_col]+[k for k in range(nfirst,A.shape[1]) if k!=rho_col]:
  begin,end=Ac.indptr[j:j+2];ii=Ac.indices[begin:end];vv=Ac.data[begin:end]
  exact=lambda: F(float(a['cost'][j]))-sum((F(float(c))*F(float(pi[i])) for i,c in zip(ii,vv)),F(0))
  residual=exact()
  if residual<0:
   demand=[(i,c) for i,c in zip(ii,vv) if c>0 and pi[i]>0]
   total=sum((F(float(c))*F(float(pi[i])) for i,c in demand),F(0));supply=total+residual
   if supply<0 or total<=0:raise ValueError('NEGATIVE_UNBOUNDED_DIRECTION')
   factor=max(0.,float(supply/total)*(1-1e-12))
   for i,c in demand:pi[i]*=factor
   residual=exact();corrections.append(dict(column=j,factor=factor,rows=len(demand)))
  if residual<0:raise ValueError('UNBOUNDED_RESIDUAL_NOT_CERTIFIED')
  if a['lo'][j]!=0 or a['hi'][j]<gp.GRB.INFINITY:raise ValueError('UNBOUNDED_BOX_IDENTITY')
  checks.append(dict(column=j,residual_lower=down(residual),exact_dyadic=[str(residual.numerator),str(residual.denominator)]))
 l,u=enclosure(A[:,:nfirst].T,pi);cost=np.asarray(a['cost'][:nfirst],np.longdouble)
 # Directed subtraction plus conservative finite-box residual enclosure.
 eps=np.finfo(np.longdouble).eps;guard=4*eps*(abs(cost)+np.maximum(abs(l),abs(u)))
 rl=cost-u-guard;ru=cost-l+guard;lo=a['lo'][:nfirst];hi=a['hi'][:nfirst]
 if not np.isfinite(lo).all() or not np.isfinite(hi).all():raise ValueError('UNPROVEN_FINITE_FIRST_BOX')
 terms=np.minimum.reduce([rl*lo,rl*hi,ru*lo,ru*hi]);pl,_=enclosure(sp.csr_matrix(a['rhs'][None,:]),pi)
 lower=np.longdouble(a['objcon'])+pl[0]+np.sum(terms,dtype=np.longdouble)-(8*nfirst+32)*eps*np.sum(abs(terms),dtype=np.longdouble)
 LB=float(np.nextafter(float(lower),-np.inf))
 if not math.isfinite(LB):raise ValueError('CERTIFICATE_NONFINITE')
 return dict(LB=LB,lower_certificate_verified=True,status='CERTIFIED',unbounded_directions=len(checks),checks=checks,corrections=corrections,method='full directed finite-box residual plus exact dyadic unbounded-direction checks'),pi

def external_gap(lb,ub):
 if lb is None or ub is None:return dict(LB=lb,UB=ub,absolute_gap=None,relative_gap=None,target=False)
 if not math.isfinite(lb) or not math.isfinite(ub):raise ValueError('NONFINITE_BOUND')
 gap=ub-lb
 if gap<0:raise ValueError('NEGATIVE_EXTERNAL_GAP')
 return dict(LB=lb,UB=ub,absolute_gap=gap,relative_gap=gap/max(1.,abs(ub)),target=bool(gap<=1 or gap/max(1.,abs(ub))<=.01))
