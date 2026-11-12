"""R7 isolated algebra; original physics and old files are never edited."""
from r10_common import *
import scipy.sparse as sp
from dataclasses import replace
from contract_matrix import Matrix
import gurobipy as gp
from gurobipy import GRB
def identity_map(mat,reason):
 n=len(mat.cost)
 return mat,dict(applicable=False,reason=reason,S=sp.eye(n,format='csr'),keep=np.arange(n),dropped_eq=[],original_columns=n,reduced_columns=n)
def pf_reduce(mat,net,scene,rho=.33):
 # Fail closed: a tree plus the single substation injection has no circulation nullspace.
 if list(net['from'])!=list(net['line_from']) or list(net['to'])!=list(net['line_to']):return identity_map(mat,'NOT_APPLICABLE_ENDPOINT_METADATA_MISMATCH')
 buses=net['buses'];nb=len(buses);nl=len(net['line_ids']);bi={b:i for i,b in enumerate(buses)}
 if rho!=.33 or nl!=nb-1 or 1 not in bi:return identity_map(mat,'NOT_APPLICABLE_NONRADIAL_OR_RHO')
 if any(u==v for u,v in zip(net['line_from'],net['line_to'])):return identity_map(mat,'NOT_APPLICABLE_SELF_LOOP')
 incidence=np.zeros((nb,nl+1))
 for j,(u,v) in enumerate(zip(net['line_from'],net['line_to'])):
  incidence[bi[u],j]-=1;incidence[bi[v],j]+=1
 incidence[bi[1],-1]=1
 if np.linalg.matrix_rank(incidence)!=nb:return identity_map(mat,'NOT_APPLICABLE_INCIDENCE_NULLSPACE')
 if not np.array_equal(np.asarray(scene['load_q_kvar']),rho*np.asarray(scene['load_p_kw'])):return identity_map(mat,'NOT_APPLICABLE_NONUNIFORM_LOAD_PF')
 q=np.r_[mat.index['line_q'].ravel(),mat.index['grid_q'].ravel()].astype(int)
 p=np.r_[mat.index['line_p'].ravel(),mat.index['grid_p'].ravel()].astype(int)
 keep=np.setdiff1d(np.arange(len(mat.cost)),q);oldnew=np.full(len(mat.cost),-1,int);oldnew[keep]=np.arange(len(keep))
 S=sp.coo_matrix((np.r_[np.ones(len(keep)),np.full(len(q),rho)],(np.r_[keep,q],np.r_[np.arange(len(keep)),oldnew[p]])),shape=(len(mat.cost),len(keep))).tocsr()
 hs=mat.H@S;names=mat.row_names['eq'];lookup={n:i for i,n in enumerate(names)};drop=[]
 for i,n in enumerate(names):
  if n.startswith('Q_balance/'):
   pn='P_balance/'+n[len('Q_balance/'):]
   if pn not in lookup:return identity_map(mat,'NOT_APPLICABLE_MISSING_P_ROW')
   j=lookup[pn];delta=hs[i]-rho*hs[j];zg=mat.G[i]-rho*mat.G[j]
   if np.any(delta.data!=0) or np.any(zg.data!=0) or mat.d[i]!=rho*mat.d[j]:return identity_map(mat,'NOT_APPLICABLE_REACTIVE_BALANCE_NOT_REDUNDANT')
   drop.append(i)
 if len(drop)!=96*nb:return identity_map(mat,'NOT_APPLICABLE_BALANCE_COUNT')
 lo=mat.lo[keep].copy();hi=mat.hi[keep].copy()
 for oq,op in zip(q,p):
  j=oldnew[op];lo[j]=max(lo[j],mat.lo[oq]/rho);hi[j]=min(hi[j],mat.hi[oq]/rho)
 if np.any(lo>hi):return identity_map(mat,'NOT_APPLICABLE_EMPTY_BOUND_INTERSECTION')
 eqkeep=np.setdiff1d(np.arange(len(mat.d)),drop)
 md=dict(mat.metadata,representation='R7_PF_EXACT',rho=rho,original_rows_sha=digest(mat.row_names))
 red=Matrix(np.asarray(S.T@mat.cost).ravel(),np.asarray(mat.parts@S),mat.A@S,mat.B,mat.b,hs[eqkeep],mat.G[eqkeep],mat.d[eqkeep],lo,hi,{'retained_original_columns':keep},dict(ub=mat.row_names['ub'],eq=[names[i] for i in eqkeep]),md)
 return red,dict(applicable=True,reason='TREE_UNIFORM_PF_REDUNDANCY_PROVED',S=S,keep=keep,q_columns=q,p_columns=p,dropped_eq=drop,original_columns=len(mat.cost),reduced_columns=len(keep),bound_intersections=len(q),voltage_coefficients=(np.asarray(net['mp'])+rho*np.asarray(net['mq'])).tolist(),rho=rho)
def lift(y,mapping):return np.asarray(mapping['S']@y).ravel()
def add_strengthening(m,fs,enabled=(1,2,3)):
 groups={}
 for i,q in enumerate(fs['qs']):
  t=int(np.searchsorted(np.arange(97)/4,q['arrival']));groups.setdefault((q['e'],t),[]).append(i)
 if sum(map(len,groups.values()))!=len(fs['qs']):raise ValueError('ROUTE_GROUPING')
 A=m.addVars(fs['Ecount'],97,lb=0,ub=1,name='r7_arrival_prefix')
 S=m.addVars(fs['Ecount'],97,lb=0,ub=1,name='r7_start_prefix')
 F=m.addVars(fs['Ecount'],97,lb=0,ub=1,name='r7_end_prefix')
 for e in range(fs['Ecount']):
  for t in range(97):
   m.addConstr(A[e,t]-(A[e,t-1] if t else 0)==gp.quicksum(fs['v'][i] for i in groups.get((e,t),[])),name=f'R7/Arec/{e}/{t}')
   m.addConstr(S[e,t]-(S[e,t-1] if t else 0)==(fs['s'][e,t] if t in fs['starts'] else 0),name=f'R7/Srec/{e}/{t}')
   m.addConstr(F[e,t]-(F[e,t-1] if t else 0)==(fs['f'][e,t] if t in fs['ends'] else 0),name=f'R7/Frec/{e}/{t}')
   if 1 in enabled:m.addConstr(S[e,t]<=A[e,t],name=f'R7/VI1/{e}/{t}')
   if 2 in enabled:m.addConstr(F[e,t]<=(S[e,t-4] if t>=4 else 0),name=f'R7/VI2/{e}/{t}')
  for t in range(96):
   if 3 in enabled:m.addConstr(gp.quicksum(fs['wc'][e,c,t] for c in range(fs['Ccount']))==fs['w'][e,t],name=f'R7/VI3/{e}/{t}')
 return dict(A=A,S=S,F=F,route_occurrences=sum(map(len,groups.values())),group_count=len(groups),proof='integer_route_start_end_prefix_and_exclusive_station')
def compact_base(net,scene,E,M):
 from contract_matrix import assemble
 original=assemble(net,[],scene,[],E,global_M=M)
 reduced,mapping=pf_reduce(original,net,scene)
 return original,reduced,mapping
def original_expressions(y,mapping):
 keep=mapping['keep'];pos={int(k):i for i,k in enumerate(keep)};yy=[None]*mapping['original_columns'];vv=y.tolist()
 for k,i in pos.items():yy[k]=vv[i]
 if mapping['applicable']:
  for q,p in zip(mapping['q_columns'],mapping['p_columns']):yy[int(q)]=mapping['rho']*vv[pos[int(p)]]
 assert all(v is not None for v in yy)
 return yy
def save_mapping(p,mapping):
 numeric={k:v for k,v in mapping.items() if isinstance(v,np.ndarray)}
 S=mapping['S'];numeric.update(S_data=S.data,S_indices=S.indices,S_indptr=S.indptr,S_shape=np.asarray(S.shape))
 c.npz(Path(p).with_suffix('.npz'),**numeric)
 dump(Path(p).with_suffix('.json'),{k:v for k,v in mapping.items() if k!='S' and not isinstance(v,np.ndarray)})
