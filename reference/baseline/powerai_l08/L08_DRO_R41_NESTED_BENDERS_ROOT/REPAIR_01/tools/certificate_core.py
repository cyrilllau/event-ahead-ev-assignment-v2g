import numpy as np
from scipy import sparse as sp
from r41_common import dump

def cut_certificate(W,T,h,c,lo,hi,xlo,xhi,pi,x,obj,out):
 # All constants and multipliers are binary64; long-double accumulation error is
 # bounded by gamma_n times the absolute sum. Output coefficients are rounded
 # to binary64 with a uniform correction over the complete first-stage bounds.
 ld=np.longdouble;eps=np.finfo(ld).eps
 def matbound(A,v):
  A=A.astype(ld);v=np.asarray(v,dtype=ld);mid=A@v;mass=abs(A)@abs(v);n=max(1,A.shape[1]);gam=ld(n)*eps/(1-ld(n)*eps);return mid,gam*mass+eps*abs(mid)
 def dotbound(a,b):
  a=np.asarray(a,dtype=ld);b=np.asarray(b,dtype=ld);mid=a@b;n=len(a);g=ld(n)*eps/(1-ld(n)*eps);return mid,g*(abs(a)@abs(b))+eps*abs(mid)
 wt,we=matbound(W.T,pi);r=np.asarray(c,ld)-wt;we+=eps*(abs(c)+abs(wt));rl=r-we;ru=r+we
 assert np.isfinite(lo).all() and np.isfinite(hi).all(),'UNBOUNDED_RECOURSE_RESIDUAL_NOT_CERTIFIED'
 terms=np.minimum.reduce([rl*lo,rl*hi,ru*lo,ru*hi]);termerr=4*eps*np.maximum(abs(terms),1)
 aa,ae=dotbound(pi,h);beta,be=matbound(T.T,-pi);bf=np.asarray(beta,float);be+=abs(np.asarray(bf,ld)-beta)
 assert np.isfinite(xlo).all() and np.isfinite(xhi).all(),'UNBOUNDED_FIRST_STAGE_NOT_CERTIFIED'
 correction=np.sum(be*np.maximum(abs(xlo),abs(xhi)),dtype=ld);alpha=aa-ae+np.sum(terms-termerr,dtype=ld)-correction
 alpha-=eps*len(terms)*np.sum(abs(terms)+termerr,dtype=ld)+ld('1e-8');af=float(np.nextafter(float(alpha),-np.inf))
 pointlower=float(af+np.asarray(bf,ld)@np.asarray(x,ld));gap=float(obj-pointlower)
 np.savez_compressed(out/'CUT.npz',alpha=af,beta=bf,Pi=pi,lo=lo,hi=hi,xlo=xlo,xhi=xhi)
 rec=dict(passed=gap>=-1e-6 and gap<=1e-3,objective=obj,cut_at_point=pointlower,signed_certificate_gap=gap,all_first_stage_columns=len(bf),all_recourse_columns=len(c),all_scenario_rows=len(h),finite_bound_correction=True,global_domain_not_fixed_plan=True,guard=float(ae+correction),proof='Dual sign-feasible Pi; c-W^T Pi residual bounded over full original/derived recourse bounds. Round beta with correction over complete first-stage bounds. theta>=alpha+beta*x valid for every legal first-stage x, not only test point.')
 dump(out/'CERTIFICATE.json',rec);return rec

def finite_bounds(A,b,sense,lo,hi,names,out):
 from fractions import Fraction as F
 A=A.tocsr();lo=lo.copy();hi=hi.copy();proof=[]
 for sweep in range(4):
  if np.isfinite(lo).all() and np.isfinite(hi).all():break
  for i in range(A.shape[0]):
   if sense[i]=='=':directions=[1,-1]
   else:directions=[1 if sense[i]=='<' else -1]
   js=A.indices[A.indptr[i]:A.indptr[i+1]];co=A.data[A.indptr[i]:A.indptr[i+1]]
   for sign in directions:
    cc=sign*co
    missing=[k for k,j in enumerate(js) if (cc[k]>0 and not np.isfinite(hi[j])) or (cc[k]<0 and not np.isfinite(lo[j]))]
    for k in missing:
     terms=[];valid=True
     for z,j in enumerate(js):
      if z==k:continue
      endpoint=lo[j] if cc[z]>0 else hi[j]
      if not np.isfinite(endpoint):valid=False;break
      terms.append(F(float(cc[z]))*F(float(endpoint)))
     if not valid:continue
     bd=(F(float(sign*b[i]))-sum(terms,F(0)))/F(float(cc[k]));j=js[k];val=float(bd)
     if cc[k]>0:hi[j]=float(np.nextafter(val,np.inf));kind='upper'
     else:lo[j]=float(np.nextafter(val,-np.inf));kind='lower'
     proof.append(dict(variable=names[j],row=int(i),direction=sign,kind=kind,exact=[bd.numerator,bd.denominator],outward=float(hi[j] if kind=='upper' else lo[j])))
 dump(out/'IMPLIED_BOUNDS.json',dict(records=proof,remaining_unbounded=[names[j] for j in np.flatnonzero(~np.isfinite(lo)|~np.isfinite(hi))]))
 return lo,hi
