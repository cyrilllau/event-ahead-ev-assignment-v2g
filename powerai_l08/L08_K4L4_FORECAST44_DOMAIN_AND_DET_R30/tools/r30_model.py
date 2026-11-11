from r3_common import *
from compact_first_stage import add_first,decode,fix_first
from compact_recourse_pf import add_recourse
from exact_transform import add_strengthening,lift
from contract_matrix import assemble,first_stage
import gurobipy as gp
from gurobipy import GRB

def parameters(m,log,abs_gap=1.):
 m.Params.Threads=1;m.Params.MIPGap=0;m.Params.MIPGapAbs=abs_gap;m.Params.FeasibilityTol=1e-9;m.Params.IntFeasTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.Method=1;m.Params.NodeMethod=1;m.Params.PreSparsify=1;m.Params.TimeLimit=GRB.INFINITY
 m.Params.SoftMemLimit=read(R/'02_SOURCE_AND_CONFIG/RESOURCE_ADMISSION.json')['soft_mem_limit_GB'];m.Params.LogFile=str(log)
def construct(qs,E,net,scene,states,M,name,starts=None,ends=None,test=False):
 m=gp.Model(name);parameters(m,R/'logs'/(name+'.log'),1e-7 if test else 1.);fs=add_first(m,qs,E,len(net['station_buses']),M,starts,ends);vi=add_strengthening(m,fs);rc=add_recourse(m,fs,net,scene,states,0);m.setObjective(fs['objective']+rc['objective']);m.update();return m,fs,rc

def save_raw(m,fs,rc,out,net,scene,states):
 out.mkdir(parents=True,exist_ok=True);names=m.getAttr('VarName');v=np.array(m.getAttr('X'));npz(out/'COMPACT_RAW.npz',values=v);dump(out/'VARIABLE_NAMES.json',names)
 mp=rc['pfmap'];npz(out/'PF_LIFT.npz',keep=mp['keep'],q_columns=mp.get('q_columns',np.array([],int)),p_columns=mp.get('p_columns',np.array([],int)))
 y=lift(rc['y'].X,mp);plan=decode(fs);qd={q['route_id']:st for q,st in zip(fs['qs'],states)};contracts=np.array([fs['E'][e].X for e in range(fs['Ecount'])]);z=np.r_[np.ones(len(plan)),contracts]
 mat=assemble(net,plan,scene,[qd[q['route_id']] for q in plan],fs['Ecount'],fs['M']);fm,b,parts,_,_=first_stage(plan,fs['Ecount'],fs['Ccount'],fs['M']);fr=max(0,float(np.max(fm@z-b,initial=0)));res=mat.residual(y,z)
 integers=[abs(v[i]-round(v[i])) for i,x in enumerate(m.getVars()) if x.VType in [GRB.BINARY,GRB.INTEGER]]
 irr=max(integers,default=0);assert fr<=1e-6 and res<=1e-6 and irr<=1e-9,(fr,res,irr)
 obj=float(parts.sum(0)@z+mat.cost@y);assert abs(obj-m.ObjVal)<=1e-5,(obj,m.ObjVal)
 npz(out/'LIFTED_ORIGINAL.npz',y=y,z=z);dump(out/'PLAN.json',plan);dump(out/'RAW_AUDIT.json',dict(passed=True,integer_residual=irr,first_stage_residual=fr,original_residual=res,objective=obj,native_objective=m.ObjVal,raw_sha=sha(out/'COMPACT_RAW.npz'),variables_sha=sha(out/'VARIABLE_NAMES.json'),lift_sha=sha(out/'PF_LIFT.npz')))
 return plan,y
