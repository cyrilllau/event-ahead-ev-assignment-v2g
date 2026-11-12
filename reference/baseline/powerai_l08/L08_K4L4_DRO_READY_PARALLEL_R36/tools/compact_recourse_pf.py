"""Continuous lifting; arrival groups contain each route once per scenario."""
from compact_first_stage import *
from contract_matrix import assemble
from exact_transform import compact_base,original_expressions
def add_recourse(m,fs,net,scene,states,sid):
 E=fs['Ecount'];C=fs['Ccount'];qs=fs['qs'];v=fs['v'];M=fs['M'];pre=f's{sid}'
 # Fresh original network rows; no old matrix, solution, or certificate is loaded.
 original,base,pfmap=compact_base(net,scene,E,M)
 y=m.addMVar(len(base.cost),lb=base.lo,ub=base.hi,name=pre+'_y');yy=original_expressions(y,pfmap);ix=original.index
 retain=[i for i,n in enumerate(base.row_names['ub']) if not n.startswith(('discharge_','short_le_','p_availability/','incident_capability/'))]
 m.addMConstr(base.A[retain],y,'<',base.b[retain],name=pre+'_network_ub');m.addMConstr(base.H,y,'=',base.d,name=pre+'_network_eq')
 a=m.addVars(E,96,ub=1,name=pre+'_arrived');over=m.addVars(E,96,ub=1,name=pre+'_overlap');W=m.addVars(E,ub=24,name=pre+'_AUX_HOURS');cap=m.addVars(E,name=pre+'_capacity_aux');D=m.addVars(E,ub=1,name=pre+'_incident');H=m.addVars(E,name=pre+'_activated_contract')
 groups={};good=[]
 for i,(q,st) in enumerate(zip(qs,states)):
  g=bool(st['success'] and np.isfinite(st['arrival']) and q['departure_dc']-1.2*q['distance']>=150 and st['arrival']<=23)
  good.append(g)
  if g:
   assert st['arrival']>=0
   t=int(np.searchsorted(np.arange(96)/4,st['arrival']));groups.setdefault((q['e'],t),[]).append(i)
 zeta=np.asarray(base.metadata['zeta'],bool)
 for e,ids in enumerate(fs['by_e']):
  valid=[i for i in ids if good[i]];g=gp.quicksum(v[i] for i in valid);ar=gp.LinExpr([states[i]['arrival'] for i in valid],[v[i] for i in valid]);B=gp.LinExpr([qs[i]['B'] for i in valid],[v[i] for i in valid])
  for t in range(96):
   m.addConstr(a[e,t]-(a[e,t-1] if t else 0)==gp.quicksum(v[i] for i in groups.get((e,t),[])),name=f'{pre}/arrival/{e}/{t}')
   m.addConstr(over[e,t]<=a[e,t]);m.addConstr(over[e,t]<=fs['w'][e,t]);m.addConstr(over[e,t]>=a[e,t]+fs['w'][e,t]-1)
   m.addConstr(gp.quicksum(yy[ix['p'][e,c,t]] for c in range(C))<=90*over[e,t])
   for c in range(C):m.addConstr(yy[ix['p'][e,c,t]]<=90*fs['b'][e,c])
  m.addConstr(W[e]<=24*g);m.addConstr(W[e]>=fs['J'][e]-ar-24*(fs['u'][e]-g))
  m.addConstr(cap[e]<=.95*(B-CFG['aux_dc_kw']*W[e]));m.addConstr(cap[e]<=22.5*gp.quicksum(over[e,t] for t in range(96)))
  energy=.25*gp.quicksum(yy[i] for i in ix['p'][e].flat);m.addConstr(energy<=cap[e]);m.addConstr(energy<=fs['E'][e])
  incident_vars=[fs['wc'][e,c,t] for c,t in zip(*np.where(zeta))]
  m.addConstr(D[e]<=fs['u'][e]);m.addConstr(D[e]<=gp.quicksum(incident_vars))
  for wc in incident_vars:m.addConstr(D[e]>=wc)
  m.addConstr(H[e]<=fs['E'][e]);m.addConstr(H[e]<=M[e]*D[e]);m.addConstr(H[e]>=fs['E'][e]-M[e]*(1-D[e]))
  m.addConstr(yy[ix['short'][e]]>=H[e]-cap[e]);m.addConstr(yy[ix['short'][e]]<=fs['E'][e])
 parts=[gp.LinExpr(row.tolist(),y.tolist()) for row in base.parts]
 return dict(y=y,base=base,original=original,pfmap=pfmap,parts=parts,objective=sum(parts),arrived=a,overlap=over,W=W,capacity=cap,D=D,H=H,arrival_group_route_count=sum(map(len,groups.values())),qualified_route_count=sum(good),network_retained_rows=retain)
def build(qs,E,C,M,net,scenes,states,method='DET',prob=None,pool=None,log=None,starts=None,ends=None):
 begun=time.time();m=gp.Model('R4_'+method);m.Params.OutputFlag=bool(log);m.Params.Threads=4;m.Params.MIPGap=0;m.Params.MIPGapAbs=1;m.Params.FeasibilityTol=1e-9;m.Params.IntFeasTol=1e-9;m.Params.OptimalityTol=1e-9
 if log:m.Params.LogFile=str(log)
 fs=add_first(m,qs,E,C,M,starts,ends);obj=fs['objective'];risk={}
 if method=='ARO':risk['theta']=m.addVar(lb=-GRB.INFINITY);obj+=risk['theta']
 elif method=='Proposed':
  phi,lo,hi=prob;risk['theta']=m.addVar(lb=-GRB.INFINITY);risk['minus']=m.addVars(len(lo));risk['plus']=m.addVars(len(lo));obj+=risk['theta']+gp.quicksum(hi[i]*risk['plus'][i]-lo[i]*risk['minus'][i] for i in range(len(lo)))
 copies={}
 for sid in (range(len(scenes)) if pool is None else pool):
  rc=add_recourse(m,fs,net,scenes[sid],states[sid],sid);copies[sid]=rc;q=rc['objective']
  if method=='DET':obj+=q
  elif method=='SAA':obj+=q/len(scenes)
  elif method=='ARO':m.addConstr(risk['theta']>=q)
  else:m.addConstr(risk['theta']+gp.quicksum((risk['plus'][i]-risk['minus'][i])*phi[i,sid] for i in range(len(lo)))>=q)
 m.setObjective(obj);m.update()
 size=dict(variables=m.NumVars,binaries=m.NumBinVars,rows=m.NumConstrs,nonzeros=m.NumNZs,route_binaries=len(qs),start_end_binaries=E*(len(fs['starts'])+len(fs['ends'])),build_elapsed_s=time.time()-begun,peak_rss_bytes=rss(),arrival_group_counts={s:rc['arrival_group_route_count'] for s,rc in copies.items()})
 return m,fs,copies,risk,size
