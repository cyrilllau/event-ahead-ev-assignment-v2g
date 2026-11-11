from pathlib import Path
import os,sys,json,time,hashlib,resource,shutil,re
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];B=R.parent;OLD=B/'L08_K4L4_FORECAST44_DOMAIN_AND_DET_R30'
sys.path.insert(0,str(OLD/'tools'))
import r3_common as c
import numpy as np,scipy.sparse as sp,gurobipy as gp
from gurobipy import GRB
from compact_first_stage import add_first
from compact_recourse_pf import add_recourse
from exact_transform import add_strengthening
from contract_matrix import assemble,first_stage
read,sha,dump,npz=c.read,c.sha,c.dump,c.npz
V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
sys.path.insert(0,str(V/'tools'));import control_core as ctrl
sys.path.insert(0,str(B/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'));import r23_common as rr
LEASE=V/'leases/R31_DET_FINITE_DIAGNOSTIC.json';rr.LEASE=LEASE

def params(m,threads=1,focus=0,seconds=None,log=None):
 d=dict(Threads=threads,MIPFocus=focus,Method=1,NodeMethod=1,PreSparsify=1,Seed=0,MIPGap=0,MIPGapAbs=1e-7,FeasibilityTol=1e-9,IntFeasTol=1e-9,OptimalityTol=1e-9,SoftMemLimit=10.)
 if seconds is not None:d.update(TimeLimit=seconds,MIPGapAbs=1.)
 for k,v in d.items():m.setParam(k,v)
 if log:m.Params.LogFile=str(log)
 else:m.Params.OutputFlag=0
 return d

def matrix(m):
 m.update();return dict(A=m.getA(),b=np.array(m.getAttr('RHS')),sense=np.array(m.getAttr('Sense')),names=m.getAttr('VarName'),rows=m.getAttr('ConstrName'),lo=np.array(m.getAttr('LB')),hi=np.array(m.getAttr('UB')),types=np.array(m.getAttr('VType')),cost=np.array(m.getAttr('Obj')),constant=m.ObjCon)

def audit_matrix(a,x,integer=True):
 d=a['A']@x-a['b'];s=a['sense'];z=np.where(s=='=',abs(d),np.where(s=='<',np.maximum(d,0),np.maximum(-d,0)));it=a['types']!='C'
 out=dict(row_residual=float(z.max(initial=0)),bound_residual=float(max(np.max(a['lo']-x,initial=0),np.max(x-a['hi'],initial=0))),integer_residual=float(np.max(abs(x[it]-np.rint(x[it])),initial=0)),objective=float(a['cost']@x+a['constant']))
 assert out['row_residual']<=1e-6 and out['bound_residual']<=1e-6,out
 if integer:assert out['integer_residual']<=1e-9,out
 return out

def prerequisites(qs,states):
 assert len(qs)==len(states)
 for q,s in zip(qs,states):
  assert s['success'] and np.isfinite(s['arrival']),'NOT_APPLICABLE_FAILED_ROUTE'
  assert abs(s['arrival']-q['arrival'])<=1e-10,'NOT_APPLICABLE_DIFFERENT_ARRIVAL'
  assert q['departure_dc']-1.2*q['distance']>=150 and 0<=s['arrival']<=23,'NOT_APPLICABLE_SOC_OR_QUALIFICATION'
  assert abs(q['B']-(q['departure_dc']-1.2*q['distance']-100))<=1e-10,'NOT_APPLICABLE_ENERGY_IDENTITY'

def transform(original,qs,E,C,short_indices,p_indices,linked=False):
 """Remove exactly the verified generic auxiliary block; retain all core columns."""
 m=original.copy();a=matrix(m);names=a['names'];ci={n:i for i,n in enumerate(names)};A=a['A'];CSC=A.tocsc()
 remove=[i for i,n in enumerate(names) if n.startswith(('s0_arrived[','s0_overlap[','s0_AUX_HOURS[','s0_capacity_aux[','s0_incident[','s0_activated_contract[')) or n in {f's0_y[{k}]' for k in short_indices}]
 touched=set(A[:,remove].nonzero()[0]);gate=[];pnames={f's0_y[{k}]' for k in p_indices.flat}
 if linked:
  for i in range(len(a['rows'])):
   js=A.indices[A.indptr[i]:A.indptr[i+1]];vs=A.data[A.indptr[i]:A.indptr[i+1]]
   if len(js)==2 and a['sense'][i]=='<' and a['b'][i]==0:
    ns=[names[j] for j in js]
    if any(n.startswith('station[') for n in ns) and any(n in pnames for n in ns):
     dd=dict(zip(ns,vs));assert any(v==-90 for n,v in dd.items() if n.startswith('station['));assert any(v==1 for n,v in dd.items() if n.startswith('s0_y['));gate.append(i)
  assert len(gate)==E*C*96,('STATION_GATE_COUNT',len(gate))
 deleted=sorted(touched|set(gate));deleted_set=set(deleted);m.remove([m.getConstrs()[i] for i in deleted]);m.remove([m.getVars()[i] for i in remove]);m.update();vv={v.VarName:v for v in m.getVars()}
 for e in range(E):
  for t in range(96):
   if linked:
    for cc in range(C):m.addConstr(vv[f's0_y[{p_indices[e,cc,t]}]']<=90*vv[f'reserved[{e},{cc},{t}]'],name=f'R31/p_reserved/{e}/{cc}/{t}')
   else:m.addConstr(gp.quicksum(vv[f's0_y[{p_indices[e,cc,t]}]'] for cc in range(C))<=90*vv[f'window[{e},{t}]'],name=f'R31/p_window/{e}/{t}')
 m.update();mapping=dict(original_names=names,new_names=m.getAttr('VarName'),removed_columns=[names[i] for i in remove],removed_rows=[a['rows'][i] for i in deleted],kept_original_rows=[a['rows'][i] for i in range(len(a['rows'])) if i not in deleted_set],linked=linked,E=E,C=C,short_indices=np.array(short_indices).tolist(),p_indices=np.array(p_indices).tolist(),scope='FORECAST_EQUALS_ACTUAL_ONLY',construction='arrived=A_prefix; overlap=w; W=J-arrival; cap=E; incident=max incident reserved or0; activated=E*incident; short=0')
 assert m.NumBinVars==original.NumBinVars
 for pre in ['R7/VI1/','R7/VI2/','R7/VI3/']:assert sum(n.startswith(pre) for n in a['rows'])==sum(n.startswith(pre) for n in m.getAttr('ConstrName'))
 return m,mapping

def lift(values,names,original_names,qs,E,C,zeta,short_indices):
 v=dict(zip(names,values));out={n:v.get(n,0.) for n in original_names}
 for e in range(E):
  ar=sum(q['arrival']*out[f'route[{i}]'] for i,q in enumerate(qs) if q['e']==e);en=out[f'E[{e}]']
  for t in range(96):out[f's0_arrived[{e},{t}]']=out[f'r7_arrival_prefix[{e},{t}]'];out[f's0_overlap[{e},{t}]']=out[f'window[{e},{t}]']
  d=max([0.]+[out[f'reserved[{e},{cc},{t}]'] for cc,t in zip(*np.where(zeta))]);out[f's0_incident[{e}]']=d;out[f's0_activated_contract[{e}]']=en*d;out[f's0_AUX_HOURS[{e}]']=out[f'J[{e}]']-ar;out[f's0_capacity_aux[{e}]']=en;out[f's0_y[{short_indices[e]}]']=0.
 return np.array([out[n] for n in original_names])

def physical_audit(x,names,qs,E,net,scene,M,pfmap,integer=True):
 v=dict(zip(names,x));plans=[]
 for e in range(E):
  chosen=[i for i,q in enumerate(qs) if q['e']==e and v[f'route[{i}]']>.5]
  if not chosen:assert abs(v[f'E[{e}]'])<1e-6;continue
  assert len(chosen)==1;q=qs[chosen[0]];ks=[k for k in range(93) if v.get(f'start[{e},{k}]',0)>.5];js=[j for j in range(1,97) if v.get(f'end[{e},{j}]',0)>.5];assert len(ks)==len(js)==1
  h=c.task(q,ks[0],js[0]-1);h['contract']=float(v[f'E[{e}]']);assert 15-1e-6<=h['contract']<=h['Ebar']+1e-6;plans.append(h)
 mat=assemble(net,plans,scene,[dict(success=True,arrival=h['arrival']) for h in plans],E,M);y=np.empty(len(mat.cost));keep=np.asarray(pfmap['keep']);y[keep]=[v[f's0_y[{i}]'] for i in range(len(keep))];y[np.array(pfmap['q_columns'])]=.33*y[np.array(pfmap['p_columns'])]
 z=np.r_[np.ones(len(plans)),[v[f'E[{e}]'] for e in range(E)]];f,b,parts,_,_=first_stage(plans,E,len(net['station_buses']),M);fr=float(np.max(f@z-b,initial=0));res=mat.residual(y,z);assert max(fr,res)<=1e-6,(fr,res)
 traces=[]
 for h in plans:
  tr=c.battery_trace(h['B']+100,h['arrival'],(h['r']+1)/4,y[mat.index['p']][h['e']].sum(0));assert tr['min_dc']>=100-1e-6;traces.append(dict(ev=h['ev'],min_dc=tr['min_dc'],terminal_dc=tr['terminal_dc'],aux_dc=tr['aux_dc']))
 obj=float(parts.sum(0)@z+mat.cost@y);guard=max(1e-7,10*max(fr,res)*float(abs(mat.cost).sum()),128*np.finfo(float).eps*max(1,2*abs(obj)))
 return dict(passed=True,original_PQ_residual=res,original_first_stage_residual=fr,objective=obj,numerical_guard=guard,original_feasible_UB=float(np.nextafter(obj+guard,np.inf)),selected=len(plans),plans=plans,SOC=traces,network_source='original contract_matrix.assemble, all P/Q/voltage/line/energy constraints',V2G_ac=float(.25*y[mat.index['p']].sum()),shed_ac=float(.25*y[mat.index['shed']].sum()))

def model_sizes(m):return dict(variables=m.NumVars,rows=m.NumConstrs,nonzeros=m.NumNZs,binaries=m.NumBinVars)

def native_fixture(qs,E,net,scene,M,starts,ends):
 prerequisites(qs,[dict(success=True,arrival=q['arrival']) for q in qs]);m=gp.Model('R31_SYNTHETIC_ORIGINAL');params(m);fs=add_first(m,qs,E,len(net['station_buses']),M,starts,ends);add_strengthening(m,fs);rc=add_recourse(m,fs,net,scene,[dict(success=True,arrival=q['arrival']) for q in qs],0);m.setObjective(fs['objective']+rc['objective']);m.update();return m,fs,rc
