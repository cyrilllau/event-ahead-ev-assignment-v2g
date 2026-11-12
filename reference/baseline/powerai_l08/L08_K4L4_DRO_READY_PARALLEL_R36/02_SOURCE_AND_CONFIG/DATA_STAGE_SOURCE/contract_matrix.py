"""Explicit R3 matrices. z=[x_h, E_e]; all y continuous. No reserve variables."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog
from r3_common import CFG,tou,incident,qualify,first_cost,dump,sha,npz
class Rows:
 def __init__(self,n,m):self.n=n;self.m=m;self.rows=[];self.cols=[];self.val=[];self.zr=[];self.zc=[];self.zv=[];self.b=[];self.names=[]
 def add(self,name,y=(),z=(),rhs=0):
  i=len(self.b);self.names.append(name);self.b.append(rhs)
  for j,v in y:
   if v:self.rows.append(i);self.cols.append(j);self.val.append(v)
  for j,v in z:
   if v:self.zr.append(i);self.zc.append(j);self.zv.append(v)
 def matrices(self):
  return sp.coo_matrix((self.val,(self.rows,self.cols)),shape=(len(self.b),self.n)).tocsr(),sp.coo_matrix((self.zv,(self.zr,self.zc)),shape=(len(self.b),self.m)).tocsr(),np.asarray(self.b,float)
@dataclass
class Matrix:
 cost:object;parts:object;A:object;B:object;b:object;H:object;G:object;d:object;lo:object;hi:object;index:dict;row_names:dict;metadata:dict
 def solve(self,z):
  v=linprog(self.cost,A_ub=self.A,b_ub=self.b-self.B@z,A_eq=self.H,b_eq=self.d-self.G@z,bounds=list(zip(self.lo,self.hi)),method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  if not v.success:raise RuntimeError(v.message)
  return dict(objective=float(v.fun),y=v.x,components=self.parts@v.x,residual=self.residual(v.x,z))
 def residual(self,y,z):
  return max(0.,float(np.max(self.A@y+self.B@z-self.b,initial=0)),float(np.max(abs(self.H@y+self.G@z-self.d),initial=0)),float(np.max(self.lo-y,initial=0)),float(np.max(y-self.hi,initial=0)))
 def save(self,folder):
  folder.mkdir(parents=True,exist_ok=True);arrays=dict(cost=self.cost,parts=self.parts,b=self.b,d=self.d,lo=self.lo,hi=self.hi)
  for n in ['A','B','H','G']:
   m=getattr(self,n);arrays.update({n+'_data':m.data,n+'_indices':m.indices,n+'_indptr':m.indptr,n+'_shape':np.array(m.shape)})
  npz(folder/'MATRIX.npz',**arrays)
  dump(folder/'MAPS.json',dict(index={k:v.tolist() for k,v in self.index.items()},rows=self.row_names,metadata=self.metadata,sha256=sha(folder/'MATRIX.npz')))
def first_stage(tasks,E,C,global_M=None):
 H=len(tasks);rows=Rows(0,H+E);parts=np.zeros((4,H+E));ee=np.array([h['e'] for h in tasks]);M=np.asarray(global_M,dtype=float).copy();assert M.shape==(E,) and np.all(np.isfinite(M)) and np.all(M>=0), "Explicit immutable global_M required"
 for j,h in enumerate(tasks):parts[:,j]=[25,30*(h['r']+1-h['d'])/4,(h['r']-h['k']+1)/4,0];assert h['Ebar']<=M[h['e']]+1e-9
 parts[3,H:]=.02
 for e in range(E):
  ids=np.flatnonzero(ee==e)
  rows.add(f'one_task/{e}',z=[(j,1) for j in ids],rhs=1)
  rows.add(f'contract_min/{e}',z=[(j,15) for j in ids]+[(H+e,-1)])
  rows.add(f'contract_max/{e}',z=[(j,-tasks[j]['Ebar']) for j in ids]+[(H+e,1)])
 for c in range(C):
  for t in range(96):rows.add(f'ports/{c}/{t}',z=[(j,1) for j,h in enumerate(tasks) if h['c']==c and h['k']<=t<=h['r']],rhs=CFG['ports'])
 _,B,b=rows.matrices();return B,b,parts,M,rows.names
def assemble(net,tasks,scene,route_state,E,global_M=None):
 H=len(tasks);C=len(net['station_buses']);N=len(net['buses']);L=len(net['from']);T=96
 assert scene['line_operating_outages'].shape==(96,L) and scene['load_p_kw'].shape==(96,N)
 assert len(route_state)==H
 shapes=dict(p=(E,C,T),short=(E,),shed=(T,N),line_p=(T,L),line_q=(T,L),u=(T,N),grid_p=(T,),grid_q=(T,));ix={};count=0
 for k,s in shapes.items():ix[k]=np.arange(count,count+np.prod(s)).reshape(s);count+=np.prod(s)
 count=int(count);lo=np.zeros(count);hi=np.full(count,np.inf);parts=np.zeros((4,count))
 off=scene['line_operating_outages'].astype(bool)
 for name,key in [('line_p','p_limit'),('line_q','q_limit')]:
  cap=(~off)*np.asarray(net[key]);lo[ix[name]]=-cap;hi[ix[name]]=cap
 lo[ix['u']]=net['u_min'];hi[ix['u']]=net['u_max'];hi[ix['shed']]=scene['load_p_kw'];hi[ix['p']]=90
 parts[0,ix['shed']]=.25*np.asarray(net['shed_cost']);parts[1,ix['short']]=1;parts[2,ix['p']]=.25*tou();parts[3,ix['p']]=.25*.05/.95
 ub=Rows(count,H+E);eq=Rows(count,H+E);zeta,refs=incident(net,scene)
 mu=[];use=[];av=[]
 for h,state in zip(tasks,route_state):
  q,m,u,v=qualify(h['departure_dc'],h['distance'],state['arrival'],h['k'],h['r'],state['success']);mu.append(m);use.append(u);av.append(v)
 mu=np.asarray(mu,bool);use=np.asarray(use);av=np.asarray(av)
 _,_,_,M,_=first_stage(tasks,E,C,global_M=global_M)
 for e in range(E):
  ids=[j for j,h in enumerate(tasks) if h['e']==e]
  ub.add(f'discharge_contract/{e}',[(i,.25) for i in ix['p'][e].flat],[(H+e,-1)])
  ub.add(f'discharge_physical/{e}',[(i,.25) for i in ix['p'][e].flat],[(j,-.95*use[j]) for j in ids])
  ub.add(f'short_le_contract/{e}',[(ix['short'][e],1)],[(H+e,-1)])
  for c in range(C):
   for t in range(T):ub.add(f'p_availability/{e}/{c}/{t}',[(ix['p'][e,c,t],1)],[(j,-90) for j in ids if tasks[j]['c']==c and mu[j,t]])
 for j,h in enumerate(tasks):
  e=h['e'];I=float(zeta[h['c'],h['k']:h['r']+1].any())
  ub.add(f'incident_capability/{j}',[(ix['short'][e],-1)],[(j,M[e]),(H+e,I)],rhs=av[j]+M[e])
 for c in range(C):
  for t in range(T):ub.add(f'station_power/{c}/{t}',[(ix['p'][e,c,t],1) for e in range(E)],rhs=1800)
 bi={b:i for i,b in enumerate(net['buses'])}
 for t in range(T):
  for n,b in enumerate(net['buses']):
   p=[(ix['shed'][t,n],1)];q=[(ix['shed'][t,n],.33)]
   for c,sb in enumerate(net['station_buses']):
    if sb==b:
     p += [(ix['p'][e,c,t],1) for e in range(E)];q += [(ix['p'][e,c,t],.33) for e in range(E)]
   if b==1:p.append((ix['grid_p'][t],1));q.append((ix['grid_q'][t],1))
   for l,(u,v) in enumerate(zip(net['from'],net['to'])):
    sign=int(v==b)-int(u==b)
    if sign:p.append((ix['line_p'][t,l],sign));q.append((ix['line_q'][t,l],sign))
   eq.add(f'P_balance/{b}/{t}',p,rhs=scene['load_p_kw'][t,n]);eq.add(f'Q_balance/{b}/{t}',q,rhs=scene['load_q_kvar'][t,n])
  for l,(u,v) in enumerate(zip(net['from'],net['to'])):
   terms=[(ix['u'][t,bi[v]],1),(ix['u'][t,bi[u]],-1),(ix['line_p'][t,l],net['mp'][l]),(ix['line_q'][t,l],net['mq'][l])]
   ub.add(f'voltage_plus/{l}/{t}',terms,rhs=net['big_m']*off[t,l]);ub.add(f'voltage_minus/{l}/{t}',[(i,-v) for i,v in terms],rhs=net['big_m']*off[t,l])
  for b in refs[t]:eq.add(f'reference/{b}/{t}',[(ix['u'][t,bi[b]],1)],rhs=net['u_ref'])
 A,B,b=ub.matrices();Hmat,G,d=eq.matrices()
 return Matrix(parts.sum(0),parts,A,B,b,Hmat,G,d,lo,hi,ix,dict(ub=ub.names,eq=eq.names),dict(dt=.25,x_count=H,E_count=E,mu=mu.tolist(),Euse=use.tolist(),Eav=av.tolist(),zeta=zeta.tolist(),refs=refs,M=M.tolist(),active_vars=list(ix),current_outage=True,road_repair_clearing=False))
def fallback(mat,z,scene,net):
 y=np.zeros(len(mat.cost));ix=mat.index;y[ix['u']]=1;y[ix['shed']]=scene['load_p_kw'];root=net['buses'].index(1);y[ix['shed'][:,root]]=0;y[ix['grid_p']]=scene['load_p_kw'][:,root];y[ix['grid_q']]=scene['load_q_kvar'][:,root];y[ix['short']]=z[-len(ix['short']):]
 return y
def compress(mat):
 fixed=np.flatnonzero(np.isfinite(mat.hi)&(mat.lo==mat.hi));keep=np.setdiff1d(np.arange(len(mat.cost)),fixed);values=mat.lo[fixed];constant=float(mat.cost[fixed]@values)
 reduced=Matrix(mat.cost[keep],mat.parts[:,keep],mat.A[:,keep],mat.B,mat.b-mat.A[:,fixed]@values,mat.H[:,keep],mat.G,mat.d-mat.H[:,fixed]@values,mat.lo[keep],mat.hi[keep],{},mat.row_names,dict(original=mat.metadata,keep=keep.tolist(),fixed=fixed.tolist(),fixed_values=values.tolist(),objective_constant=constant,rows_retained=True))
 return reduced,keep,fixed,values,constant
def restore(reduced_y,keep,fixed,values,size):
 y=np.empty(size);y[keep]=reduced_y;y[fixed]=values;return y
def extensive(tasks,E,C,matrices,method,prob=None,pool=None,log=None,global_M=None):
 import gurobipy as gp
 from gurobipy import GRB
 H=len(tasks);F,b,parts,M,names=first_stage(tasks,E,C,global_M=global_M);m=gp.Model('R3_'+method);m.Params.Threads=4;m.Params.OutputFlag=bool(log);m.Params.MIPGap=0;m.Params.MIPGapAbs=1;m.Params.FeasibilityTol=1e-9;m.Params.IntFeasTol=1e-9;m.Params.OptimalityTol=1e-9
 if log:m.Params.LogFile=str(log)
 z=m.addMVar(H+E,lb=0,ub=np.r_[np.ones(H),M],vtype=np.array([GRB.BINARY]*H+[GRB.CONTINUOUS]*E),name='z')
 m.addConstr(F@z<=b,name='first_stage');obj=parts.sum(0)@z
 if method=='ARO':theta=m.addVar(lb=-GRB.INFINITY,name='theta');obj+=theta
 elif method=='Proposed':
  phi,low,up=prob;theta=m.addVar(lb=-GRB.INFINITY,name='beta0');minus=m.addMVar(len(low),lb=0,name='beta_minus');plus=m.addMVar(len(up),lb=0,name='beta_plus');obj+=theta-low@minus+up@plus
 ys={}
 for s in (range(len(matrices)) if pool is None else pool):
  a=matrices[s];y=m.addMVar(len(a.cost),lb=a.lo,ub=a.hi,name=f'y_{s}');ys[s]=y
  m.addConstr(a.A@y+a.B@z<=a.b);m.addConstr(a.H@y+a.G@z==a.d);q=a.cost@y
  if method=='DET':obj+=q
  elif method=='SAA':obj+=q/len(matrices)
  elif method=='ARO':m.addConstr(theta>=q)
  else:m.addConstr(theta+(plus-minus)@phi[:,s]>=q)
 m.setObjective(obj,GRB.MINIMIZE);m.optimize()
 result=dict(status=m.Status,solutions=m.SolCount,bestbound=m.ObjBound,solver_runtime=m.Runtime)
 if m.SolCount:
  result.update(objective=m.ObjVal,z=z.X,components_first=parts@z.X,y={s:y.X for s,y in ys.items()})
  if method=='Proposed':result.update(beta0=theta.X,beta_minus=minus.X,beta_plus=plus.X)
 return result
