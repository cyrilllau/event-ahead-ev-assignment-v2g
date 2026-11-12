"""Isolated full-support DRO aggregation; no old launcher is invoked."""
from pathlib import Path
import os,sys,time,json,hashlib,functools,uuid
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
OUT=None;BASE=Path('/home/Shixin/ev-assignment')
ARO=BASE/'L08_ARO400_SHARED_INTEGER_AUTONOMOUS_V1'
R36=BASE/'L08_K4L4_DRO_READY_PARALLEL_R36'
R41=BASE/'L08_DRO_R41_NESTED_BENDERS_ROOT/REPAIR_01'
# Lifecycle module search path is supplied explicitly by the parsed entry point.
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import durable as d, runtime as rt, engine as e
from probability_patch_adapter import certify_candidate_compat
sys.path.insert(0,str(R36/'tools'))
import r36_model as orig
SCIENCE='c1fe0c77564e0b350782b0375447b2f637f2e5241871b28601a68cf6aa543765'
read=d.read;sha=d.sha;b=e.b

class ExecutionContext:
 def __init__(self, execution_root, source_root, config, run, deadline=None):
  self.root=Path(execution_root).resolve();self.source_root=Path(source_root).resolve();self.config=config;self.run=run
  self.deadline=min(run['deadline_monotonic']-120,deadline or run['deadline_monotonic']-120);self.prep=self.deadline
  assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==run['boot_id']
  assert config['science']==SCIENCE and config['source_root']==str(self.source_root)
 def check(self,preparation=False):
  if time.monotonic()>=self.deadline:raise TimeoutError('GLOBAL_WALL_LIMIT')
  if (self.root/'PAUSE_RESOURCE.json').exists():raise RuntimeError('PAUSED_RESOURCE')
  if (self.root/'USER_STOP.json').exists():raise RuntimeError('USER_REQUESTED_STOP')
CTX=None
def bind_context(context):
 global OUT,CTX
 CTX=context;OUT=context.root
 e.configure(OUT,dict(context.run,source=str(R41),execution_root=str(OUT),source_root=str(context.source_root),config=context.config),context.deadline)
 return context
def write(path,obj,immutable=False):
 p=Path(path);p=p if p.is_absolute() else OUT/p
 if OUT.resolve() not in p.resolve().parents:raise ValueError('WRITE_OUTSIDE_ISOLATED_ROOT')
 d.atomic(p,obj,immutable)
def save(path,**arrays):return d.publish_npz(Path(path),**arrays)
def ref(p):return rt.ref(p)
@functools.lru_cache(None)
def first():return sp.load_npz(R41/'identity/FIRST_A.npz').tocsr(),dict(np.load(R41/'identity/FIRST_ARRAYS.npz')),read(R41/'identity/FIRST_NAMES.json')
def master(name='DRO400_AGGREGATED_INTEGER',integer=True):
 A,a,names=first();m=gp.Model(name);m.Params.OutputFlag=0
 x=m.addMVar(len(names),lb=a['lo'],ub=a['hi'],vtype=a['vtype'] if integer else 'C');m.update();m.setAttr('VarName',x.tolist(),names)
 m.addMConstr(A,x,a['sense'],a['rhs'],name='common');rho=m.addVar(lb=0,name='DRO_RHO');m.setObjective(a['cost']@x+rho);m.update()
 assert m.NumVars==60781 and (not integer or m.NumBinVars==25770)
 return m,x,rho
def add_cut(m,x,rho,alpha,beta,name):
 ix=np.flatnonzero(beta);cols=x.tolist();return m.addConstr(rho>=alpha+gp.LinExpr(beta[ix].tolist(),[cols[int(i)] for i in ix]),name=name)
def bounded(value):return float(value)-1e-6 if value is not None and np.isfinite(value) and abs(value)<1e90 else None
def merge_bound(old,new):return old if new is None or not np.isfinite(new) or abs(new)>=1e90 else max(old,float(new))
def full_vector(records,key):
 if len(records)!=400 or {int(r['sid']) for r in records}!=set(range(400)):raise ValueError('PARTIAL_400_NOT_A_BOUND')
 z=np.zeros(400)
 for r in records:z[int(r['sid'])]=r[key]
 if not np.isfinite(z).all():raise ValueError('NONFINITE_FULL400_VALUE')
 return z
def aggregate(p,records,lo,hi,probability_passed):
 if not probability_passed or np.any(p<0) or not np.isfinite(p).all():raise ValueError('UNCERTIFIED_PROBABILITY')
 if len(records)!=len(p) or {int(r['sid']) for r in records}!=set(range(len(p))):raise ValueError('MISSING_CUT_SUPPORT')
 if not np.isfinite(lo).all() or not np.isfinite(hi).all():raise ValueError('UNPROVEN_FINITE_FIRST_BOX')
 ld=np.longdouble;total=np.zeros(len(lo),ld);magnitude=np.zeros(len(lo),ld);aa=ld(0);amag=ld(0);sources=[]
 for r in records:
  sid=int(r['sid']);weight=ld(p[sid]);sources.append({'sid':sid,'weight':float(weight),'source':r.get('cut'),'zero':r.get('universal_zero',False)})
  if r.get('universal_zero'):continue
  if not r.get('global_cut_valid'):raise ValueError('FIXED_PLAN_DUAL_IS_NOT_GLOBAL')
  z=np.load(r['cut']);beta=z['beta'];alpha=float(z['alpha']);nz=np.flatnonzero(beta)
  terms=weight*np.asarray(beta[nz],ld);total[nz]+=terms;magnitude[nz]+=abs(terms);aa+=weight*ld(alpha);amag+=abs(weight*ld(alpha))
 # Each dot has at most n products and n additions. Conservative gamma,
 # followed by directed cast error and a full finite-box compensation.
 eps=np.finfo(ld).eps;gamma=ld(8*len(p)+32)*eps
 err=gamma*magnitude+np.finfo(ld).tiny
 bhat=np.asarray(total,float);delta=np.asarray(bhat,ld)-total
 dl=np.nextafter(delta-err,-np.inf);du=np.nextafter(delta+err,np.inf)
 compensation=np.maximum.reduce([dl*lo,dl*hi,du*lo,du*hi])
 E=np.sum(compensation,dtype=ld);E+=ld(8*len(lo)+32)*eps*np.sum(abs(compensation),dtype=ld)
 al=aa-gamma*amag-E;al-=32*eps*(abs(aa)+abs(E)+1)
 ahat=float(np.nextafter(float(al),-np.inf))
 return ahat,bhat,dict(compensation=float(E),alpha_round_guard=float(aa-ld(ahat)),gamma=float(gamma),source_count=len(p),sources=sources,finite_box=True,method='streaming_longdouble_gamma_outward_full_box')

class Oracle(e.Oracle):
 def __init__(self,sid,root,env=None,context=None):
  self.sid=int(sid);self.root=Path(root);self.out=self.root/'oracles'/f'{sid:04d}';self.out.mkdir(parents=True,exist_ok=True)
  src=ARO/'oracles'/f'{sid:04d}';self.a=dict(np.load(src/'ARRAYS.npz'));self.W=sp.load_npz(src/'W.npz').tocsr();self.T=sp.load_npz(src/'T.npz').tocsr();self.qbar=read(src/'QBAR_PROOF.json')['Qbar'];self.model=self.y=self.rows=None;self.owned_env=env;self.lifecycle_context=context or {};self.dispatch=None;self.memory_provider=None;self.native_kind='FULL_DOMAIN_GLOBAL_SUPPORT'

def plan_point(plan):
 A,a,names=first();ix={n:i for i,n in enumerate(names)};x=np.zeros(len(names));qs=orig.routes();by={h['e']:h for h in plan}
 for i,q in enumerate(qs):x[ix[f'route[{i}]']]=int(q['e'] in by and q['route_id']==by[q['e']]['route_id'])
 for ev,h in by.items():
  k=h['k'];j=h['r']+1;c=h['c'];arrival=h['arrival']
  for n,v in [(f'start[{ev},{k}]',1),(f'end[{ev},{j}]',1),(f'selected[{ev}]',1),(f'K[{ev}]',k/4),(f'J[{ev}]',j/4),(f'E[{ev}]',h['contract']),(f'station[{ev},{c}]',1)]:x[ix[n]]=v
  for t in range(k,j):x[ix[f'window[{ev},{t}]']]=x[ix[f'reserved[{ev},{c},{t}]']]=1
  for t in range(97):
   x[ix[f'r7_arrival_prefix[{ev},{t}]']]=int(arrival<=t/4+1e-12)
   x[ix[f'r7_start_prefix[{ev},{t}]']]=int(k<=t)
   x[ix[f'r7_end_prefix[{ev},{t}]']]=int(j<=t)
 check=b.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],x)
 if max(check.values())>1e-9:raise ValueError(('PLAN_MAPPING_FAILED',check))
 return x

def canonical(raw,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 A,a,names=first();binary=a['vtype']=='B';raw=np.asarray(raw,float)
 if not np.isfinite(raw).all() or np.max(abs(raw[binary]-np.rint(raw[binary])),initial=0)>1e-9:raise ValueError('NONINTEGER_NATIVE_CANDIDATE')
 x=raw.copy();x[binary]=np.rint(x[binary]);check=b.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],x)
 if max(check.values())>1e-9:
  m=gp.Model('FIXED_INTEGER_REPAIR');m.Params.OutputFlag=0;y=m.addMVar(len(x),lb=np.where(binary,x,a['lo']),ub=np.where(binary,x,a['hi']));m.addMConstr(A,y,a['sense'],a['rhs']);dev=m.addMVar(len(x));m.addConstr(dev>=y-x);m.addConstr(dev>=x-y);m.setObjective(dev.sum());b.configure_oracle(m,Path(folder)/'REPAIR.log',CTX.deadline-time.monotonic()-30);e.optimize(m,'FIXED_INTEGER_REPAIR',{})
  if m.Status!=2:raise ValueError('FIXED_INTEGER_REPAIR_FAILED')
  x=y.X.copy();m.dispose();check=b.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],x)
 if max(check.values())>1e-9:raise ValueError(('FIRST_STAGE_AUDIT_FAILED',check))
 save(Path(folder)/'POINT.npz',raw=raw,accepted=x);write(Path(folder)/'AUDIT.json',dict(passed=True,residual=check,point_sha=hashlib.sha256(x.tobytes()).hexdigest()))
 return x

class Probability(b.Probability):
 def solve_full(self,qL,qU,qbars,label):
  folder=self.out/label;folder.mkdir(parents=True,exist_ok=False);normal=max(1.,float(abs(qL).max()));self.model.setObjective((qL/normal)@self.p,gp.GRB.MAXIMIZE);history=[];closed=False
  for k in range(200):
   CTX.check();call=folder/f'CALL_{k:03d}';call.mkdir()
   for key,val in dict(Threads=1,Method=1,FeasibilityTol=1e-9,OptimalityTol=1e-9,SoftMemLimit=3,TimeLimit=max(.01,CTX.deadline-time.monotonic()-30),OutputFlag=1,LogFile=str(call/'NATIVE.log')).items():setattr(self.model.Params,key,val)
   cid=e.optimize(self.model,'FULL400_PROBABILITY',dict(iteration=k,label=label))
   if self.model.Status!=2:raise RuntimeError('PROBABILITY_NATIVE_NOT_OPTIMAL')
   p=np.asarray(self.p.X);bp=np.zeros(45139);bm=np.zeros(45139)
   for (kind,i),row in self.active.items():
    if kind=='U':bp[i]=max(0,row.Pi)*normal
    else:bm[i]=max(0,-row.Pi)*normal
   raw=dict(p=p,qL=qL,qU=qU,beta0=float(self.simplex.Pi)*normal,plus=bp,minus=bm,active_kind=np.array([v[0] for v in self.active]),active_id=np.array([v[1] for v in self.active]))
   save(call/'RAW_NATIVE.npz',**raw)
   ex=self.a['phi']@p;viol=sorted([(float(v),'L',int(i)) for i,v in enumerate(self.a['lower']-ex) if v>1e-9]+[(float(v),'U',int(i)) for i,v in enumerate(ex-self.a['upper']) if v>1e-9],reverse=True);history.append(dict(iteration=k,violations=len(viol),max=viol[0][0] if viol else 0,call_id=cid));write(folder/'HISTORY.json',history)
   if not viol:closed=True;break
   new=[(kind,i) for _,kind,i in viol if (kind,i) not in self.active][:500]
   if not new:break
   self.add(new)
  save(folder/'RAW_FINAL.npz',**raw)
  risk=b.risk_upper(qU,float(raw['beta0']),bp,bm)
  p0=np.load(b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz')['p']
  strict=certify_candidate_compat(p,qbars,list(range(400)),folder/'STRICT',p0=p0,dd=b.moment_data(),check=b.probability_check,enclosure=b.enclosure,owned_root=OUT)
  ps=np.load(folder/'STRICT/PROBABILITY.npz')['pstar'];check=b.probability_check(ps)
  if not check['passed']:raise ValueError('P_STAR_NOT_IN_ORIGINAL_P')
  riskL=float(b.enclosure(sp.csr_matrix(qL[None,:]),ps)[0][0]);save(folder/'CERTIFICATE.npz',pstar=ps,qL=qL,qU=qU,beta0=risk['beta0'],plus=bp,minus=bm)
  info=dict(risk_L=riskL,risk_U=risk['risk_upper'],rowgen_closed=closed,strict_probability=strict,probability_check=check,certificate=ref(folder/'CERTIFICATE.npz'),dual_covers_all400=True)
  write(folder/'CERTIFICATE.json',info);return ps,info
