"""Production callback, native archiving and commit. No scientific model changes."""
from pathlib import Path
import time,math,hashlib,uuid,traceback,threading
import numpy as np
import gurobipy as gp
from scheduling_fix import NearIndex,classify
from repair_guard_helpers import safe_native_observations
WRITE_LOCK=threading.RLock()
C=gp.GRB.Callback
WHERE={getattr(C,n):n for n in ('POLLING','PRESOLVE','SIMPLEX','MIP','MIPSOL','MIPNODE','MESSAGE','BARRIER','MULTIOBJ','IIS') if hasattr(C,n)}
def no_finite_cutoff(value):
 value=float(value)
 return bool(value>0 and (math.isinf(value) or value>=gp.GRB.INFINITY))

def point_sha(x):return hashlib.sha256(np.asarray(x,dtype=float).tobytes()).hexdigest()
def pending(v,known,nfirst,mapping):
 key=point_sha(v[:nfirst]);k=known.get(key)
 if k is None:return True
 if v[nfirst]<k['risk_L']-1e-4:return True
 return any(v[col]<k['qL'][sid]-1e-4 for sid,col in mapping.items())
class Callback:
 def __init__(self,variables,nfirst,mapping,known,deadline,stop,emit,call_id,revision,stage,clock=time.monotonic,energy_columns=()):
  self.variables=variables;self.n=nfirst;self.mapping=mapping;self.known=known;self.deadline=deadline;self.stop=stop;self.emit=emit;self.call_id=call_id;self.revision=revision;self.stage=stage;self.clock=clock;self.start=clock();self.last=-math.inf;self.root_end=None;self.reason=None;self.error=None;self.query=None;self.queue={};self.observed=set();self.telemetry_errors=[];self.callback_wall=0.;self.near=NearIndex(known,nfirst,energy_columns);self.actionable=set();self.deferred=set();self.scheduling_events=[]
 def get(self,m,what,name):self.query=name;return m.cbGet(what)
 def __call__(self,m,where):
  t=self.clock();self.observed.add(int(where));self.query=None
  try:
   stop=self.stop()
   if t>=self.deadline or stop:
    self.reason=stop or 'GLOBAL_WALL_LIMIT';m.terminate();return
   if where==C.POLLING:return
   nodes=bound=phase=None
   if where==C.MIPSOL:
    self.query='cbGetSolution';v=np.asarray(m.cbGetSolution(self.variables),float)
    obj=float(self.get(m,C.MIPSOL_OBJ,'MIPSOL_OBJ'))
    if v.shape!=(len(self.variables),) or not np.isfinite(v).all() or not math.isfinite(obj):raise ValueError('INVALID_CALLBACK_CANDIDATE')
    action,reference=classify(v,self.known,self.n,self.mapping,self.near)
    key=point_sha(v[:self.n])
    if action!='EXACT_AUDITED_NO_SEPARATION':self.queue[key]=(v,obj,'MIPSOL')
    if action=='DEFER_NEAR_NO_CERTIFICATE_REUSE':self.deferred.add(key)
    elif action in ('CERTIFIED_SEPARATION','NEW_MATERIAL_POINT'):self.actionable.add(key)
    if len(self.scheduling_events)<1000:self.scheduling_events.append(dict(point_sha=key,action=action,reference=reference,certificate_aliased=False))
   elif where==C.MIP:
    nodes=self.get(m,C.MIP_NODCNT,'MIP_NODCNT');bound=self.get(m,C.MIP_OBJBND,'MIP_OBJBND');phase=self.get(m,C.MIP_PHASE,'MIP_PHASE')
    if nodes>0 and self.root_end is None:self.root_end=t
    if self.stage=='MASTER_MIP' and self.root_end is not None and t-self.root_end>=120 and self.actionable:
     self.reason='POST_ROOT_PENDING_EVALUATION_OR_SEPARATION';m.terminate()
   # Other events are deliberately unused: no API queries, including MIPNODE.
   if where in (C.MIP,C.MIPSOL) and t-self.last>=10:
    value=dict(call_id=self.call_id,math_revision=self.revision,epoch=time.time(),wall_since_optimize=t-self.start,root_completed=self.root_end is not None,pending=len(self.queue),actionable=len(self.actionable),deferred_near=len(self.deferred),where=int(where),where_name=WHERE.get(where,'UNKNOWN'),native_bound=bound if bound is not None and math.isfinite(bound) and abs(bound)<1e90 else None,nodes=nodes,phase=phase)
    try:
     with WRITE_LOCK:self.emit(value)
    except OSError as exc:
     if len(self.telemetry_errors)<3:self.telemetry_errors.append(dict(error=repr(exc),epoch=time.time()))
    self.last=t
  except BaseException as exc:
   if self.error is None:self.error=dict(call_id=self.call_id,math_revision=self.revision,where=int(where),where_name=WHERE.get(where,'UNKNOWN'),query_name=self.query,exception=repr(exc),traceback=traceback.format_exc(),epoch=time.time(),wall_since_optimize=self.clock()-self.start)
   self.reason='CALLBACK_API_OR_CANDIDATE_ERROR';m.terminate()
  finally:self.callback_wall+=self.clock()-t

def commit(result,write,folder=''):
 prefix=Path(folder)
 record=dict(call_id=result['id'],math_id=result['math_id'],status=result['status'],reason=result['reason'],partial=result['status']!=gp.GRB.OPTIMAL,raw=result.get('raw'),candidate_refs=result['candidates'],bound=result['bound'],end=result['end_ref'],Pi_basis_available=False,branch_tree_restored=False)
 write(prefix/'LAST_RETURNED.json',record)
 if result.get('has_X'):write(prefix/'LAST_SOLVED.json',record)
 return 'IMPLEMENTATION_ERROR' if result.get('callback_error') or result.get('exception') or result.get('array_errors') else 'RETURNED'

def native(m,stage,folder,ctx,run,math_id,revision,nfirst,mapping,known,write,save,ref,append,seconds=None,simulated=False):
 ctx.check();folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);m.update();vars=m.getVars();mip=bool(m.IsMIP)
 limit=ctx.deadline-time.monotonic()
 if seconds is not None:limit=min(limit,seconds)
 if limit<=0:raise TimeoutError('NO_NATIVE_BUDGET')
 params=dict(Threads=4 if stage!='TEST' else 1,Method=1,NodeMethod=1,PreSparsify=1,MIPFocus=0,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,MIPGap=.001,MIPGapAbs=1,SoftMemLimit=24,TimeLimit=limit)
 for k,v in params.items():setattr(m.Params,k,v)
 m.Params.OutputFlag=1;m.Params.LogFile=str(folder/'NATIVE.log')
 actual={k:getattr(m.Params,k) for k in params};assert actual==params
 assert no_finite_cutoff(m.Params.Cutoff) and m.Params.SolutionLimit==m.getParamInfo('SolutionLimit')[5]
 write(folder/'PARAMETERS.json',dict(actual,Cutoff='INFINITY',SolutionLimit=m.Params.SolutionLimit,OutputFlag=m.Params.OutputFlag,LogFile=m.Params.LogFile))
 cid=uuid.uuid4().hex;t=time.monotonic();begin=dict(id=cid,phase='BEGIN',scope='TEST' if stage=='TEST' else 'SCIENCE',simulated_native=simulated,stage=stage,kind=stage,run_id=run['run_id'],epoch=time.time(),monotonic=t,math_id=math_id,revision=revision,folder=str(folder))
 write(folder/'BEGIN.json',begin,True);append(begin)
 def stop():
  if (ctx.root/'USER_STOP.json').exists():return 'USER_REQUESTED_STOP'
  if (ctx.root/'PAUSE_RESOURCE.json').exists():return 'RESOURCE_PAUSE'
  return None
 cb=Callback(vars,nfirst,mapping,known,ctx.deadline,stop,lambda v:write(folder/'PROGRESS.json',v),cid,revision,stage,energy_columns=ctx.config.get('energy_columns',()))
 exc=None
 try:m.optimize(cb)
 except BaseException as e:exc=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 obs=safe_native_observations(m,infinity=gp.GRB.INFINITY);get=lambda k:obs.get(k,{}).get('value')
 end=dict(begin,phase='END',epoch=time.time(),wall=time.monotonic()-t,status=get('Status'),runtime=get('Runtime'),work=get('Work'),max_mem_GB=get('MaxMemUsed'),sol_count=get('SolCount'),raw_observations=obs,reason=cb.reason or ('NATIVE_EXCEPTION' if exc else 'NATIVE_RETURN'),exception=exc,callback_error=cb.error,callback_wall=cb.callback_wall,telemetry_errors=cb.telemetry_errors,observed_where=[dict(value=i,name=WHERE.get(i,'UNKNOWN')) for i in sorted(cb.observed)],unobserved_where=[n for i,n in WHERE.items() if i not in cb.observed],root_completed=cb.root_end is not None,scheduling_events=cb.scheduling_events,deferred_near=len(cb.deferred),actionable=len(cb.actionable))
 write(folder/'NATIVE_END.json',end,True)
 raw={};array_errors={};unavailable={}
 attrs=['X'] if get('SolCount') is not None and get('SolCount')>0 else []
 if not mip:attrs+=['Pi','RC','VBasis','CBasis']
 for a in attrs:
  try:ar=np.asarray(m.getAttr(a))
  except (gp.GurobiError,AttributeError) as e:unavailable[a]=repr(e);continue
  size=m.NumConstrs if a in ('Pi','CBasis') else m.NumVars
  if ar.shape!=(size,) or not np.isfinite(ar).all():array_errors[a]='INVALID_SHAPE_OR_NONFINITE';continue
  raw[a]=ar
 end['raw']=save(folder/'RAW_NATIVE.npz',**raw);end['has_X']='X'in raw;end['array_errors']=array_errors;end['unavailable_arrays']=unavailable
 bound=get('ObjBoundC');allowed=(gp.GRB.OPTIMAL,gp.GRB.INTERRUPTED,gp.GRB.TIME_LIMIT,gp.GRB.NODE_LIMIT,gp.GRB.ITERATION_LIMIT,gp.GRB.SOLUTION_LIMIT,gp.GRB.MEM_LIMIT)
 end['raw_bound']=bound;end['bound']=float(bound)-1e-6 if mip and end['status'] in allowed and bound is not None and not array_errors else None;end['bound_level']='SOLVER_SUPPORTED_GUARDED_ONCE' if end['bound'] is not None else 'NOT_AVAILABLE'
 queue=cb.queue
 if 'X'in raw:
  key=point_sha(raw['X'][:nfirst]);queue.pop(key,None);queue={key:(raw['X'],get('ObjVal'),'FINAL_NATIVE'),**queue}
 candidates=[]
 for i,(key,(v,obj,source)) in enumerate(sorted(queue.items(),key=lambda kv:(0 if kv[1][2]=='FINAL_NATIVE' else 1,kv[1][1] if kv[1][1] is not None else math.inf,kv[0]))):
  candidates.append(dict(point=save(folder/f'CANDIDATE_{i:04d}.npz',x=v[:nfirst],full=v),point_sha=key,objective=obj,source=source))
 write(folder/'CANDIDATES.json',candidates,True);end['candidates']=candidates;write(folder/'END.json',end,True);append(end);end['end_ref']=ref(folder/'END.json')
 # The caller always commits this structured result BEFORE acting on errors.
 return end,raw
