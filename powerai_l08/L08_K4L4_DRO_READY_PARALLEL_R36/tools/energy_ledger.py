"""R10 experimental onboard auxiliary load; all times hours and energy DC except stated AC."""
import numpy as np
POLICY='ONBOARD_FULL_DWELL_TO_RESERVED_END_V1'
def forecast(A,a,k,r,aux=1.):
 K=k/4;J=(r+1)/4
 assert np.isfinite(a) and 0<=a<=K<=23 and r<96 and r-k+1>=4 and A>=150
 assert aux>=0 and np.isfinite(aux)
 raw=A-100-aux*(J-a)
 if raw<0:raise ValueError('AUXILIARY_BATTERY_INFEASIBLE')
 energy=.95*raw;power=90*(J-K)
 return dict(cap_ac=min(energy,power),battery_ac=energy,power_ac=power,aux_dc=aux*(J-a),binding='energy' if energy<power else 'duration')
def actual(A,a,k,r,success=True,aux=1.):
 assert 0<=k<=r<96 and aux>=0 and np.isfinite(aux)
 mu=np.zeros(96,bool);K=k/4;J=(r+1)/4
 if not success:return dict(qualified=False,mu=mu,use_dc=0.,available_ac=0.,aux_dc=None,raw_dc=None)
 assert np.isfinite(a) and a>=0
 dwell=max(0,J-a);raw=A-100-aux*dwell
 if a<J and raw<0:raise ValueError('AUXILIARY_BATTERY_INFEASIBLE')
 g=bool(A>=150 and max(K,a)<=23 and r/4>=a)
 if g:mu=(np.arange(96)>=k)&(np.arange(96)<=r)&(np.arange(96)/4>=a)
 use=max(0,raw) if g else 0.
 return dict(qualified=g,mu=mu,use_dc=use,available_ac=min(.95*use,22.5*mu.sum()),aux_dc=aux*dwell,raw_dc=raw)
def battery_trace(A,a,J,p,aux=1.,extra=()):
 p=np.asarray(p);assert p.shape==(96,) and np.min(p)>=-1e-7 and 0<=a<J<=24
 times=sorted(set([float(a),float(J)]+[t/4 for t in range(97) if a<t/4<J]+[float(t) for t in extra if a<=t<=J]))
 dc=[float(A-aux*(u-a)-np.sum(p*np.maximum(0,np.minimum(u,np.arange(96)/4+.25)-np.arange(96)/4))/.95) for u in times]
 assert max(np.diff(dc),default=0)<=1e-7
 return dict(times_h=times,dc_kWh=dc,soc_pct=[x/5 for x in dc],min_dc=min(dc),terminal_dc=dc[-1],aux_dc=aux*(J-a),V2G_ac=float(.25*p.sum()))
