"""Exact integer route/window domain; no task-index binary array."""
from r4_common import *
import gurobipy as gp
from gurobipy import GRB
def add_first(m,qs,E,C,M,starts=None,ends=None):
 starts=list(range(93)) if starts is None else list(starts);ends=list(range(1,97)) if ends is None else list(ends)
 assert np.asarray(M).shape==(E,) and all(q['B']>=50 for q in qs)
 v=m.addVars(len(qs),vtype=GRB.BINARY,name='route');s=m.addVars(E,starts,vtype=GRB.BINARY,name='start');f=m.addVars(E,ends,vtype=GRB.BINARY,name='end')
 u=m.addVars(E,ub=1,name='selected');K=m.addVars(E,ub=23,name='K');J=m.addVars(E,ub=24,name='J');en=m.addVars(E,ub=M.tolist(),name='E')
 w=m.addVars(E,96,ub=1,name='window');b=m.addVars(E,C,ub=1,name='station');wc=m.addVars(E,C,96,ub=1,name='reserved')
 by_e=[[i for i,q in enumerate(qs) if q['e']==e] for e in range(E)]
 for e,ids in enumerate(by_e):
  m.addConstr(u[e]==gp.quicksum(v[i] for i in ids),name=f'one_route/{e}')
  m.addConstr(s.sum(e,'*')==u[e]);m.addConstr(f.sum(e,'*')==u[e])
  m.addConstr(K[e]==gp.quicksum(.25*k*s[e,k] for k in starts));m.addConstr(J[e]==gp.quicksum(.25*j*f[e,j] for j in ends));m.addConstr(J[e]-K[e]>=u[e])
  ah=gp.LinExpr([qs[i]['arrival'] for i in ids],[v[i] for i in ids]);bb=gp.LinExpr([qs[i]['B'] for i in ids],[v[i] for i in ids])
  m.addConstr(ah<=K[e]);m.addConstr(en[e]>=15*u[e]);m.addConstr(en[e]<=M[e]*u[e]);m.addConstr(en[e]<=.95*(bb-CFG['aux_dc_kw']*(J[e]-ah)));m.addConstr(en[e]<=90*(J[e]-K[e]))
  for t in range(96):m.addConstr(w[e,t]-(w[e,t-1] if t else 0)==(s[e,t] if t in starts else 0)-(f[e,t] if t in ends else 0),name=f'window_recurrence/{e}/{t}')
  for c in range(C):
   m.addConstr(b[e,c]==gp.quicksum(v[i] for i in ids if qs[i]['c']==c))
   for t in range(96):
    m.addConstr(wc[e,c,t]<=b[e,c]);m.addConstr(wc[e,c,t]<=w[e,t]);m.addConstr(wc[e,c,t]>=b[e,c]+w[e,t]-1)
 for c in range(C):
  for t in range(96):m.addConstr(gp.quicksum(wc[e,c,t] for e in range(E))<=CFG['ports'],name=f'ports/{c}/{t}')
 parts=[25*gp.quicksum(u.values()),30*gp.quicksum(J.values())-gp.LinExpr([30*q['d']/4 for q in qs],list(v.values())),gp.quicksum(J[e]-K[e] for e in range(E)),.02*gp.quicksum(en.values())]
 return dict(qs=qs,Ecount=E,Ccount=C,M=np.asarray(M),v=v,s=s,f=f,u=u,K=K,J=J,E=en,w=w,b=b,wc=wc,by_e=by_e,parts=parts,objective=sum(parts),starts=starts,ends=ends)
def decode(fs,values=None):
 get=(lambda x:x.X) if values is None else (lambda x:values[x.index])
 plans=[]
 for e,ids in enumerate(fs['by_e']):
  sel=[i for i in ids if get(fs['v'][i])>.5]
  if not sel:
   assert abs(get(fs['E'][e]))<=1e-6;continue
  assert len(sel)==1
  ks=[k for k in fs['starts'] if get(fs['s'][e,k])>.5];js=[j for j in fs['ends'] if get(fs['f'][e,j])>.5];assert len(ks)==len(js)==1
  h=task(fs['qs'][sel[0]],ks[0],js[0]-1);h['contract']=float(get(fs['E'][e]));plans.append(h)
 return plans
def fix_first(fs,plans):
 sel={h['e']:h for h in plans}
 for i,q in enumerate(fs['qs']):
  h=sel.get(q['e']);val=int(h is not None and (q['c'],q['d'])==(h['c'],h['d']));fs['v'][i].LB=fs['v'][i].UB=val
 for e in range(fs['Ecount']):
  h=sel.get(e)
  for k in fs['starts']:fs['s'][e,k].LB=fs['s'][e,k].UB=int(h is not None and h['k']==k)
  for j in fs['ends']:fs['f'][e,j].LB=fs['f'][e,j].UB=int(h is not None and h['r']+1==j)
  fs['E'][e].LB=fs['E'][e].UB=h['contract'] if h else 0
