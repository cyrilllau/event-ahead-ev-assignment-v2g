"""Scheduling only: never rounds a plan or aliases a certificate."""
import hashlib
from fractions import Fraction as F
import numpy as np
from scipy import sparse as sp
from envelope_sparse import enclosure
NEAR_ENERGY=1e-7
SEPARATION=1e-4
class NearIndex:
 def __init__(self,known,n,energy_columns):
  self.mask=np.ones(n,dtype=bool);self.energy=np.array(energy_columns,dtype=int);self.mask[self.energy]=False;self.index={}
  for key,v in known.items():
   if 'x' not in v:continue
   x=np.asarray(v['x'],float)
   if x.shape!=(n,):raise ValueError('CACHE_POINT_DIMENSION')
   h=hashlib.sha256(x[self.mask].tobytes()).hexdigest();self.index.setdefault(h,[]).append((key,x[self.energy].copy()))
 def near(self,x):
  if self.energy.size==0:return None
  h=hashlib.sha256(np.asarray(x)[self.mask].tobytes()).hexdigest()
  for key,e in self.index.get(h,[]):
   if np.max(abs(np.asarray(x)[self.energy]-e))<=NEAR_ENERGY:return key
  return None

def classify(v,known,n,mapping,near,force_final=False):
 x=np.asarray(v[:n],float)
 if not np.isfinite(v).all():raise ValueError('NONFINITE_CANDIDATE')
 key=hashlib.sha256(x.tobytes()).hexdigest();k=known.get(key)
 if k is not None:
  needed=v[n]<k['risk_L']-SEPARATION or any(v[j]<k['qL'][s]-SEPARATION for s,j in mapping.items())
  return ('CERTIFIED_SEPARATION' if needed else 'EXACT_AUDITED_NO_SEPARATION'),key
 if force_final:return 'EXACT_FINAL_AUDIT',key
 hit=near.near(x)
 return ('DEFER_NEAR_NO_CERTIFICATE_REUSE' if hit else 'NEW_MATERIAL_POINT'),hit

def support_decision(beta,alpha,x,theta):
 B=sp.csr_matrix(np.asarray(beta,dtype=float).reshape(1,-1));lower,upper=enclosure(B,np.asarray(x,dtype=float))
 low=F(float(np.nextafter(float(lower[0]),-np.inf)))+F(float(alpha))-F(float(theta))
 high=F(float(np.nextafter(float(upper[0]),np.inf)))+F(float(alpha))-F(float(theta))
 return dict(activate=high>F(SEPARATION),residual_lo=float(low),residual_hi=float(np.nextafter(float(high),np.inf)),threshold=SEPARATION,source_bound_then_original_storage_guard=True)

def probability_decision(p,v,n,mapping):
 high=sum((F(float(p[s]))*F(float(v[j])) for s,j in mapping.items()),F(0))-F(float(v[n]))
 return dict(activate=high>F(SEPARATION),residual_exact=[high.numerator,high.denominator],residual_hi=float(np.nextafter(float(high),np.inf)),threshold=SEPARATION)
