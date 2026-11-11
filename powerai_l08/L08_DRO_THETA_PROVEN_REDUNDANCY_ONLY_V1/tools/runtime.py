from pathlib import Path
import os,sys,json,time,hashlib,uuid,math
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import theta_kernel as tk
from envelope_sparse import enclosure
R=Path(__file__).resolve().parents[1]
DEPS={}
ALLOWED={'TEST_SMALL','A_LP','B_LP','A_MIP','B_MIP'}
NATIVE_OPTIMIZE=gp.Model.optimize
def forbidden(*args,**kwargs):raise RuntimeError('UNREGISTERED_OPTIMIZER_CALL')
gp.Model.optimize=forbidden
def cv(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)
def dump(path,v):
 p=Path(path);p=p if p.is_absolute() else R/p
 if R not in p.resolve().parents:raise ValueError('WRITE_OUTSIDE_OUT')
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,default=cv,allow_nan=False,indent=2));os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def source(p,expected=None):
 p=Path(p);v=DEPS.get(str(p))
 if v is None:v=ref(p);DEPS[str(p)]=v
 if expected is not None:assert v['sha256']==expected,('SOURCE_SHA',str(p))
 return p
def verify(r):return source(r['path'],r['sha256'])
def read(p):return json.loads(Path(p).read_text())
def js(p):return read(source(p))
def nz(p):return dict(np.load(source(p)))
def sm(p):return sp.load_npz(source(p)).tocsr()
def matrix_sha(A):return hashlib.sha256(A.data.tobytes()+A.indices.tobytes()+A.indptr.tobytes()).hexdigest()
def check():
 run=read(R/'RUN.json')
 if time.monotonic()>=run['deadline_monotonic']-180:raise TimeoutError('GLOBAL_BUDGET')
 if (R/'USER_STOP.json').exists():raise RuntimeError('USER_STOP')
def status(phase,**v):dump('STATUS.json',dict(phase=phase,epoch=time.time(),**v));check()
def save(p,**v):
 p=R/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.savez_compressed(f,**v)
 return ref(p)
def event(v):
 with (R/'OPTIMIZER_CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(v,default=cv,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def attrs(m):
 result={};errors={}
 for k in ['Status','SolCount','Runtime','Work','MaxMemUsed','ObjVal','ObjBound','ObjBoundC','NodeCount','IterCount','IsMIP']:
  try:
   v=getattr(m,k)
   if isinstance(v,(int,float)) and (not math.isfinite(v) or abs(v)>=1e99):result[k]=None;errors[k]='NONFINITE_OR_INFINITY'
   else:result[k]=v
  except Exception as e:result[k]=None;errors[k]=str(e)
 return result,errors
def optimize(m,kind,folder,native_seconds,wall_deadline):
 assert kind in ALLOWED
 check();p=R/folder;p.mkdir(parents=True,exist_ok=True)
 remaining=min(native_seconds,wall_deadline-time.monotonic(),read(R/'RUN.json')['deadline_monotonic']-180-time.monotonic())
 if remaining<=0:raise TimeoutError('NO_OPTIMIZE_BUDGET')
 m.Params.TimeLimit=remaining;m.Params.OutputFlag=1;m.Params.LogFile=str(p/'NATIVE.log')
 ps={k:getattr(m.Params,k) for k in ['Threads','Method','NodeMethod','PreSparsify','Presolve','Aggregate','NumericFocus','ScaleFlag','MIPFocus','Seed','FeasibilityTol','OptimalityTol','IntFeasTol','SoftMemLimit','MIPGap','MIPGapAbs','TimeLimit','SolutionLimit']};ps['Cutoff']='INFINITY' if abs(m.Params.Cutoff)>=1e99 else m.Params.Cutoff
 dump(p/'PARAMETERS.json',ps)
 entry=dict(id=uuid.uuid4().hex,kind=kind,phase='BEGIN',epoch=time.time(),monotonic=time.monotonic(),folder=str(p),callback=False)
 dump(p/'BEGIN.json',entry);event(entry);error=None;raw={};unavailable={}
 try:NATIVE_OPTIMIZE(m)
 except BaseException as e:error=repr(e)
 finally:
  values,errs=attrs(m);end=dict(entry,phase='END',epoch=time.time(),wall=time.monotonic()-entry['monotonic'],exception=error,attributes=values,attribute_errors=errs)
  dump(p/'END.json',end);event(end)
  for k in ['X','Pi','RC','VBasis','CBasis']:
   try:
    if k=='X' and not values['SolCount']:raise ValueError('NO_SOLUTION')
    raw[k]=np.asarray(m.getAttr(k))
   except Exception as e:unavailable[k]=str(e)
  if raw:save(Path(folder)/'RAW_NATIVE.npz',**raw)
  dump(p/'RAW_SCHEMA.json',dict(available={k:len(v) for k,v in raw.items()},unavailable=unavailable))
 if error:raise RuntimeError(error)
 return end,raw
def params(m,mip):
 values=dict(Threads=4 if mip else 1,Method=1,NodeMethod=1,MIPFocus=0,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,SoftMemLimit=8,MIPGap=.001,MIPGapAbs=1)
 if mip:values['PreSparsify']=1
 for k,v in values.items():setattr(m.Params,k,v)
def build(A,a,n,env):
 m=gp.Model(env=env);m.Params.OutputFlag=0
 x=m.addMVar(A.shape[1],lb=a['lo'],ub=a['hi'],vtype=a['vtype'],obj=a['cost']);m.update();m.setAttr('VarName',x.tolist(),n['cols'])
 rows=m.addMConstr(A,x,a['sense'],a['rhs']);m.update();m.setAttr('ConstrName',rows.tolist(),n['rows']);m.ObjCon=float(a['objcon']);m.update()
 return m
def exact_model(m,A,a,n):
 B,b,nn=tk.model_arrays(m);d=B-A;d.eliminate_zeros();assert d.nnz==0 and B.shape==A.shape and nn==n
 for k in a:assert np.array_equal(a[k],b[k]),k
 assert m.NumSOS==0 and m.NumGenConstrs==0 and m.NumQConstrs==0 and m.ModelSense==1
 return dict(passed=True,rows=B.shape[0],columns=B.shape[1],nnz=B.nnz,binary=m.NumBinVars,matrix_sha=matrix_sha(B),all_arrays_names_exact=True)
def export(folder,A,a,n):
 p=R/folder;p.mkdir(parents=True,exist_ok=False);sp.save_npz(p/'A.npz',A);save(Path(folder)/'ARRAYS.npz',**a);dump(p/'NAMES.json',n)
 ident=dict(matrix_sha=matrix_sha(A),rows=A.shape[0],cols=A.shape[1],nnz=A.nnz,binaries=int(sum(a['vtype']=='B')),files=[ref(p/x) for x in ['A.npz','ARRAYS.npz','NAMES.json']]);dump(p/'IDENTITY.json',ident);return ident
def activity(A,a,x):
 lo,hi=enclosure(A,x);rhs=np.asarray(a['rhs'],np.longdouble)
 vio=np.where(a['sense']=='<',hi-rhs,np.where(a['sense']=='>',rhs-lo,np.maximum(abs(hi-rhs),abs(lo-rhs))))
 bounds=max(float(np.max(a['lo']-x,initial=0)),float(np.max(x-a['hi'],initial=0)))
 return dict(max_row_violation_upper=float(np.max(vio,initial=0)),max_bound_violation=bounds,strict_point_feasible=False),vio
