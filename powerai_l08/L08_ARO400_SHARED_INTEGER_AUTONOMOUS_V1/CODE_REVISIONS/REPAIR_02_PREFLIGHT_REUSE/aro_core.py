"""Finite-support integer ARO: shared first stage and one max epigraph."""
from pathlib import Path
import os,sys,time,json,hashlib,gc,shutil,uuid
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(os.environ['ARO_ROOT']).resolve();REV=Path(__file__).resolve().parent;BASE=Path('/home/Shixin/ev-assignment');R36=BASE/'L08_K4L4_DRO_READY_PARALLEL_R36';R41=BASE/'L08_DRO_R41_NESTED_BENDERS_ROOT/REPAIR_01'
sys.path.insert(0,str(REV/'lifecycle'))
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import durable as d
import runtime as rt
import engine as e
from memory_policy import ResourceRecoveryNeeded,ResourceRetryExhausted,SchedulerStopped
sys.path.insert(0,str(R36/'tools'))
import r36_model as orig
sys.path.insert(0,str(R41/'tools'))
import certificate_core as certificates
import r41_common
r41_common.R=ROOT
read=d.read;sha=d.sha
def native(model,kind):return e.optimize(model,kind,{})
def write(path,value,immutable=False):
 p=Path(path);assert ROOT in p.resolve().parents;d.atomic(p,value,immutable)
def ref(path):return rt.ref(path)
def checked():
 if time.monotonic()>=rt.deadline(read(ROOT/'RUN.json'))-90:raise SchedulerStopped({'kind':'GLOBAL_WALL_LIMIT'})
 if (ROOT/'USER_STOP.json').exists():raise SchedulerStopped({'kind':'USER_REQUESTED_STOP'})
 if (ROOT/'PAUSE_RESOURCE.json').exists():raise SchedulerStopped({'kind':'WAIT_RESOURCE'})
def first():return sp.load_npz(R41/'identity/FIRST_A.npz').tocsr(),dict(np.load(R41/'identity/FIRST_ARRAYS.npz')),read(R41/'identity/FIRST_NAMES.json')
def make_master(A,a,names,integer=True,env=None):
 m=gp.Model('L08_ARO400_FINITE_SUPPORT_SHARED_INTEGER',env=env);m.Params.OutputFlag=0
 x=m.addMVar(len(names),lb=a['lo'],ub=a['hi'],vtype=a['vtype'] if integer else 'C',name='x');m.update();m.setAttr('VarName',x.tolist(),names)
 m.addMConstr(A,x,a['sense'],a['rhs'],name='common');eta=m.addVar(lb=0,ub=gp.GRB.INFINITY,name='ARO_ETA');m.setObjective(a['cost']@x+eta);m.update();return m,x,eta
def add_support(m,x,eta,alpha,beta,name):
 nz=np.flatnonzero(beta);columns=x.tolist() if isinstance(x,gp.MVar) else x;return m.addConstr(eta>=float(alpha)+gp.LinExpr(np.asarray(beta)[nz].tolist(),[columns[int(j)] for j in nz]),name=name)
def full_bound(F,records,ids,science=None,point_sha=None):
 if not np.isfinite(F):raise ValueError('NONFINITE_FIRST_COST')
 if len(records)!=len(ids) or {int(r['sid']) for r in records}!=set(ids):raise ValueError('PARTIAL_FULL400_EVALUATION')
 if any(r.get('qU_audited') is None or not np.isfinite(r['qU_audited']) for r in records):raise ValueError('MISSING_OR_NONFINITE_Q_UPPER')
 if science is not None and any(r.get('scientific_id')!=science for r in records):raise ValueError('MIXED_SCIENCE')
 if point_sha is not None and any(r.get('point_sha')!=point_sha for r in records):raise ValueError('MIXED_ACCEPTED_POINT')
 return float(np.nextafter(np.longdouble(F)+max(np.longdouble(r['qU_audited']) for r in records),np.inf))
def bound_update(old_lb,native_bound):
 if native_bound is None or not np.isfinite(native_bound) or abs(native_bound)>1e90:return old_lb
 return max(old_lb,float(np.nextafter(native_bound-1e-6,-np.inf)))
def canonical(raw,folder,deadline):
 if not np.isfinite(raw).all():raise ValueError('NONFINITE_CANDIDATE')
 checked();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);A,a,names=first();binary=a['vtype']=='B';dist=float(np.max(abs(raw[binary]-np.rint(raw[binary])),initial=0))
 if dist>1e-9:raise ValueError('NONINTEGER_FIRST_STAGE_CANDIDATE')
 x=np.asarray(raw,float).copy();x[binary]=np.rint(x[binary]);check=e.b.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],x)
 if max(check.values())>1e-9:
  env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=gp.Model('FIXED_DISCRETE_CONTINUOUS_REPAIR',env=env);m.Params.OutputFlag=0;y=m.addMVar(len(x),lb=a['lo'],ub=a['hi']);m.addMConstr(A,y,a['sense'],a['rhs']);y.LB=np.where(binary,x,a['lo']);y.UB=np.where(binary,x,a['hi']);delta=m.addMVar(len(x));m.addConstr(delta>=y-x);m.addConstr(delta>=x-y);m.setObjective(delta.sum());m.Params.Threads=1;m.Params.Method=1;m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.SoftMemLimit=3;m.Params.TimeLimit=max(.01,deadline-time.monotonic());e.CONTEXT.clear();e.CONTEXT['task']='CANONICAL_'+uuid.uuid4().hex;native(m,'FIXED_INTEGER_REPAIR');
  if m.Status!=2:raise ValueError('FIXED_INTEGER_REPAIR_NOT_OPTIMAL')
  x=y.X.copy();m.dispose();env.dispose();check=e.b.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],x)
 if max(check.values())>1e-9 or np.max(abs(x[binary]-np.rint(x[binary])),initial=0)>1e-9:raise ValueError('CANONICAL_FIRST_STAGE_AUDIT_FAILED')
 key=hashlib.sha256(x.tobytes()).hexdigest();d.publish_npz(folder/'POINT.npz',accepted=x,raw=raw);write(folder/'AUDIT.json',{'passed':True,'binary_distance_raw':dist,'residual':check,'point_sha':key,'only_fixed_discrete_continuous_repair_allowed':True},True);return x,key

def build_oracles(ids=range(400)):
 A,fa,names=first();entries=[]
 for sid in ids:
  checked();out=ROOT/'oracles'/f'{sid:04d}';out.mkdir(parents=True,exist_ok=True)
  if (out/'CACHE_COMPLETE.json').exists():
   validate_oracle(sid);entries.append(read(out/'CACHE_COMPLETE.json'));continue
  old=R41/'oracles'/f'{sid:04d}'
  if (old/'CACHE_COMPLETE.json').exists():
   for name in ('W.npz','T.npz','ARRAYS.npz','NAMES.json','IMPLIED_BOUNDS.json','QBAR_PROOF.json','CACHE_COMPLETE.json','PF_LIFT.npz','PF_LIFT.json'):
    shutil.copyfile(old/name,out/('CACHE_PENDING.json' if name=='CACHE_COMPLETE.json' else name))
   entries.append(read(out/'CACHE_PENDING.json'));write(out/'SOURCE_REUSE.json',{'source':ref(old/'CACHE_COMPLETE.json'),'reused_without_optimize':True},True)
  else:
   env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=gp.Model('FULL_DOMAIN_'+str(sid),env=env);m.Params.OutputFlag=0
   fs=orig.add_first(m,orig.routes(),30,8,orig.full_bounds());orig.add_strengthening(m,fs);m.setObjective(fs['objective']);m.update();nx=m.NumVars;nr=m.NumConstrs
   assert nx==60780 and nr==90768 and m.NumBinVars==25770
   assert (m.getA()-A).nnz==0 and np.array_equal(m.getAttr('Obj'),fa['cost']) and np.array_equal(m.getAttr('VType'),fa['vtype'])
   rc=orig.add_scene(m,fs,None,sid);m.setObjective(rc['objective']);m.update();allA=m.getA().tocsr();rhs=np.array(m.getAttr('RHS'));sense=np.array(m.getAttr('Sense'));cost=np.array(m.getAttr('Obj'));lo=np.array(m.getAttr('LB'));hi=np.array(m.getAttr('UB'));lo[lo<=-1e90]=-np.inf;hi[hi>=1e90]=np.inf;allnames=m.getAttr('VarName')
   assert m.ObjCon==0 and not np.any(cost[:nx]) and np.min(cost[nx:])>=0 and all(t=='C' for t in m.getAttr('VType')[nx:])
   fl,fh=certificates.finite_bounds(allA,rhs,sense,lo,hi,allnames,out);assert np.isfinite(fl).all() and np.isfinite(fh).all(),'UNPROVEN_FULL_DOMAIN_BOUND'
   from exact_transform import save_mapping
   save_mapping(out/'PF_LIFT',rc['pfmap']);W=allA[nr:,nx:];T=allA[nr:,:nx];sp.save_npz(out/'W.npz',W);sp.save_npz(out/'T.npz',T);np.savez_compressed(out/'ARRAYS.npz',h=rhs[nr:],sense=sense[nr:],c=cost[nx:],lo=lo[nx:],hi=hi[nx:],finite_lo=fl[nx:],finite_hi=fh[nx:],xlo=fl[:nx],xhi=fh[:nx]);write(out/'NAMES.json',{'first':allnames[:nx],'recourse':allnames[nx:],'rows':m.getAttr('ConstrName')[nr:]},True)
   qbar=float(np.nextafter(float(np.asarray(cost[nx:],np.longdouble)@np.asarray(fh[nx:],np.longdouble))+1e-6,np.inf));write(out/'QBAR_PROOF.json',{'Qbar':qbar,'bounds_proof_sha':sha(out/'IMPLIED_BOUNDS.json'),'not_used_as_eta_upper':True},True)
   record={'sid':sid,'nx':nx,'ny':len(cost)-nx,'rows':W.shape[0],'nonzeros':W.nnz+T.nnz,'physical':ref(orig.scene_path('TRAIN',sid)/'physical.npz'),'recovery':ref(orig.scene_path('TRAIN',sid)/'recovery.npz'),'replay':ref(orig.verify_replay('TRAIN',sid)),'matrix_generation_once':True,'d_s_and_constant_zero_verified':True};write(out/'CACHE_PENDING.json',record,True);entries.append(record);m.dispose();env.dispose();del allA,W,T,fs,rc;gc.collect()
  seal=out/'PAYLOAD_SHA.json'
  if not seal.exists():write(seal,{'files':[ref(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('CACHE_COMPLETE.json','CACHE_PENDING.json','PAYLOAD_SHA.json')]},True)
  for reference in read(seal)['files']:rt.verify_ref(reference)
  if not (out/'CACHE_COMPLETE.json').exists():write(out/'CACHE_COMPLETE.json',read(out/'CACHE_PENDING.json'),True)
  validate_oracle(sid);write(ROOT/'BUILD_PROGRESS.json',{'completed':len(entries),'total':len(ids),'sid':sid,'epoch':time.time()})
 if len(ids)==400:write(ROOT/'ORACLE_MANIFEST.json',entries,True)

def validate_oracle(sid):
 out=ROOT/'oracles'/f'{sid:04d}';seal=read(out/'PAYLOAD_SHA.json')
 for r in seal['files']:rt.verify_ref(r)
 complete=read(out/'CACHE_COMPLETE.json')
 for k in ('physical','recovery','replay'):rt.verify_ref(complete[k])
 names=read(out/'NAMES.json');assert names['first']==first()[2]
 a=dict(np.load(out/'ARRAYS.npz'));W=sp.load_npz(out/'W.npz');T=sp.load_npz(out/'T.npz');assert T.shape[1]==60780 and W.shape[0]==T.shape[0]==len(a['h']) and W.shape[1]==len(a['c']);assert np.min(a['c'])>=0 and np.isfinite(a['finite_lo']).all() and np.isfinite(a['finite_hi']).all();return complete

class AROOracle(e.Oracle):
 def __init__(self,sid,root,env=None,context=None):
  validate_oracle(sid);self.sid=int(sid);self.root=Path(root);self.out=self.root/'oracles'/f'{sid:04d}';self.out.mkdir(parents=True,exist_ok=True);src=ROOT/'oracles'/f'{sid:04d}';self.a=dict(np.load(src/'ARRAYS.npz'));self.W=sp.load_npz(src/'W.npz').tocsr();self.T=sp.load_npz(src/'T.npz').tocsr();self.qbar=read(src/'QBAR_PROOF.json')['Qbar'];self.model=self.y=self.rows=None;self.owned_env=env;self.lifecycle_context=context or {};self.dispatch=None;self.memory_provider=None;self.native_kind='RECOURSE_ORACLE'

def parameters(m,log,remaining):
 checked()
 for k,v in dict(Threads=4,MIPFocus=0,Method=1,NodeMethod=1,PreSparsify=1,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,MIPGap=0,MIPGapAbs=.1,SoftMemLimit=8,Seed=0,TimeLimit=min(300,max(.01,remaining))).items():setattr(m.Params,k,v)
 m.Params.OutputFlag=1;m.Params.LogFile=str(log)
