"""Execution adapters around the preserved CONTINUE_01 scientific core."""
from pathlib import Path
import os,sys,time,uuid,hashlib,traceback,multiprocessing as mp
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import durable as d
import runtime as rt
import core
from memory_policy import RecourseMemLimit,SchedulerStopped
b=core.b
OUT=None;RUN=None;CONTEXT={};DEADLINE=None;PREFLIGHT=False

def configure(out,run,deadline):
 global OUT,RUN,DEADLINE
 OUT=Path(out);RUN=run;DEADLINE=deadline;b.OUT=OUT;core.OUT=OUT;core.SRC=Path(run['source'])
 b.MAX_BATCHES=None;core.LIMIT=None;b.WALL_LIMIT=None
 b.atomic_json=owned_json;b.tracked_optimize=optimize;core.save_npz=save_npz
def owned_json(path,x):
 path=Path(path)
 if OUT.resolve() not in path.resolve().parents:raise ValueError('WRITE_OUTSIDE_RUN')
 d.atomic(path,x)
def save_npz(path,**arrays):return d.publish_npz(path,**arrays)
def opt_raw(call):
 p=Path(call)/'RAW_STATE.npz'
 return dict(np.load(rt.verify_ref(rt.ref(p))))
def optimize(model,kind,context,extra=None):
 if PREFLIGHT:raise RuntimeError('PREFLIGHT_OPTIMIZER_FORBIDDEN')
 if time.monotonic()>=DEADLINE or (OUT/'USER_STOP.json').exists() or (OUT/'PAUSE_RESOURCE.json').exists():raise TimeoutError('COOPERATIVE_STOP_BEFORE_OPTIMIZE')
 cid=uuid.uuid4().hex;call=OUT/'CALLS'/cid;call.mkdir(parents=True)
 ctx=dict(CONTEXT,**context);begin={'id':cid,'phase':'BEGIN','kind':kind,'context':ctx,'epoch':time.time(),'owner':rt.proc(os.getpid()),'remaining':DEADLINE-time.monotonic()}
 d.atomic(call/'BEGIN.json',begin,True);rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',begin)
 # Every native call observes the same immutable global deadline.
 model.Params.TimeLimit=min(float(model.Params.TimeLimit),max(.001,DEADLINE-time.monotonic()))
 last_poll=[0.]
 def control(m,where):
  now=time.monotonic()
  if now>=DEADLINE:m.terminate();return
  if now-last_poll[0]>=1:
   last_poll[0]=now
   if (OUT/'USER_STOP.json').exists() or (OUT/'PAUSE_RESOURCE.json').exists():m.terminate()
 try:
  model.optimize(control)
  raw={}
  for attr in ('X','Pi','RC','VBasis','CBasis'):
   try:raw[attr]=np.asarray(model.getAttr(attr))
   except gp.GurobiError:pass
  raw_ref=d.publish_npz(call/'RAW_STATE.npz',**raw)
  end={'id':cid,'phase':'END','kind':kind,'context':ctx,'epoch':time.time(),'status':int(model.Status),'runtime':float(model.Runtime),'iterations':float(model.IterCount),'work':float(model.Work),'max_mem_GB':float(model.MaxMemUsed),'raw':raw_ref}
 except BaseException as exc:
  end={'id':cid,'phase':'END','kind':kind,'context':ctx,'epoch':time.time(),'exception':type(exc).__name__,'message':str(exc)}
  d.atomic(call/'END.json',end,True);rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',end);raise
 d.atomic(call/'END.json',end,True);rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',end)
 return cid

def raw_for(task,kind):
 for p in sorted((OUT/'CALLS').glob('*/END.json'),key=lambda p:p.stat().st_mtime,reverse=True):
  r=d.read(p)
  if r['kind']==kind and r['context'].get('task')==task and r.get('status')==gp.GRB.OPTIMAL:
   z=dict(np.load(rt.verify_ref(r['raw'])));return z,r
 return None,None
def task_id(science,sid,x,phase):return d.digest({'science':science,'sid':sid,'x':hashlib.sha256(np.asarray(x,dtype=float).tobytes()).hexdigest(),'oracle':'CONTINUE_01_frozen_v1','phase':phase})

class Oracle(b.Oracle):
 def __init__(self,sid,root,env=None,context=None):
  super().__init__(sid,root)
  self.owned_env=env;self.lifecycle_context=context or {};self.dispatch=None;self.memory_provider=None
  self.native_kind='RECOURSE_ORACLE'
 def close(self,discard=False):
  model=self.model
  self.model=None;self.y=None;self.rows=None;self.memory_provider=None;self.dispatch=None
  if model is not None:model.dispose()
  if discard:self.W=None;self.T=None;self.a=None;self.owned_env=None
 def solve_task(self,x,task,folder,deadline):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);result=folder/'RESULT.json'
  if result.exists():return self.validate_result(x,result)
  CONTEXT.clear();CONTEXT.update({**(self.dispatch or {}),'task':task,'sid':self.sid})
  raw,call=raw_for(task,self.native_kind)
  if raw is None:
   if self.model is None:
    if self.owned_env is None:raise ValueError('ORACLE_NATIVE_ENV_MUST_BE_EXPLICIT')
    self.model=gp.Model(f'R44_ORACLE_{self.sid}',env=self.owned_env);self.model.Params.OutputFlag=0
    self.y=self.model.addMVar(len(self.a['c']),lb=self.a['lo'],ub=self.a['hi']);self.rows=self.model.addMConstr(self.W,self.y,self.a['sense'],self.a['h']);self.model.setObjective(self.a['c']@self.y)
   self.rows.RHS=self.a['h']-self.T@x
   b.configure_oracle(self.model,folder/f'NATIVE_{time.time_ns()}.log',deadline-time.monotonic())
   if self.lifecycle_context.get('test_mode'):
    if 'TESTS' not in OUT.parts:raise ValueError('TEST_MEMORY_OVERRIDE_OUTSIDE_TEST_NAMESPACE')
    self.model.Params.SoftMemLimit=self.test_soft_limit()
   CONTEXT['configured_limit_decimal_GB']=float(self.model.Params.SoftMemLimit)
   if self.dispatch is not None:self.dispatch['configured_limit_decimal_GB']=float(self.model.Params.SoftMemLimit)
   cid=optimize(self.model,self.native_kind,{'sid':self.sid,'task':task})
   call=d.read(OUT/'CALLS'/cid/'END.json')
   if call.get('status')!=gp.GRB.OPTIMAL:
    if call.get('status')==gp.GRB.MEM_LIMIT:
     raise RecourseMemLimit(dict(self.dispatch or {},kind='RECOURSE_MEM_LIMIT',native_status=int(gp.GRB.MEM_LIMIT),sid=self.sid,task_id=task,call_id=call['id'],memory=self.memory_provider() if self.memory_provider else None,corrective_action_required=True))
    if call.get('status') in (gp.GRB.TIME_LIMIT,gp.GRB.INTERRUPTED):
     kind='GLOBAL_WALL_LIMIT' if time.monotonic()>=deadline else 'USER_REQUESTED_STOP' if (OUT/'USER_STOP.json').exists() else 'WAIT_RESOURCE' if (OUT/'PAUSE_RESOURCE.json').exists() else 'LOCAL_NATIVE_BUDGET'
     raise SchedulerStopped({'kind':kind,'native_status':int(call['status']),'sid':self.sid,'task_id':task})
    raise RuntimeError('RECOURSE_NATIVE_NOT_OPTIMAL_'+str(call.get('status')))
   raw=dict(np.load(rt.verify_ref(call['raw'])))
  check,alpha,beta,pi,ql,qu=self.validate_point(x,raw['X'],raw['Pi'])
  point=folder/'POINT.npz';cut=folder/'CUT.npz'
  if not point.exists():d.publish_npz(point,x=x,y=raw['X'],Pi=pi,RC=raw['RC'])
  if not cut.exists():d.publish_npz(cut,alpha=alpha,beta=beta,Pi=pi)
  rec={'sid':self.sid,'point_sha':hashlib.sha256(x.tobytes()).hexdigest(),'qL':ql,'qU_audited':qu,'strict_UB':None,'residual':check,'global_cut_valid':True,'primal':str(point),'cut':str(cut),'sources':[rt.ref(point),rt.ref(cut)],'task_id':task,'native_source':call['id'],'status':'NUMERICALLY_AUDITED_ONLY'}
  d.atomic(result,rec,True);return rec
 def validate_result(self,x,result):
  r=d.read(result)
  for ref in r['sources']:rt.verify_ref(ref)
  z=np.load(r['primal']);cut=np.load(r['cut'])
  if not np.array_equal(x,z['x']) or r['point_sha']!=hashlib.sha256(x.tobytes()).hexdigest() or r['sid']!=self.sid:raise ValueError('CACHE_POINT_IDENTITY_FAILURE')
  check,a,bb,pi,ql,qu=self.validate_point(x,z['y'],z['Pi'])
  if a!=float(cut['alpha']) or not np.array_equal(bb,cut['beta']) or qu!=r['qU_audited']:raise ValueError('CACHE_CERTIFICATE_FAILURE')
  return r

def get_oracle(objects,sid,root,cls=Oracle):
 if sid not in objects:objects[sid]=cls(sid,root)
 return objects[sid]
def worker_loop(conn,out,run,deadline,oracle_cls=Oracle):
 configure(out,run,deadline);objects={}
 try:
  while True:
   msg=conn.recv()
   if msg['op']=='close':break
   try:
    oracle=get_oracle(objects,msg['sid'],Path(out)/'ORACLE_OBJECTS',oracle_cls);rec=oracle.solve_task(msg['x'],msg['task'],msg['folder'],deadline)
    conn.send({'passed':True,'sid':msg['sid'],'record':rec})
   except BaseException as exc:
    conn.send({'passed':False,'sid':msg['sid'],'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
 finally:
  for obj in objects.values():obj.close()
  conn.close()

class Workers:
 def __init__(self,out,run,deadline,attempt,oracle_cls=Oracle):self.out=Path(out);self.run=run;self.deadline=deadline;self.attempt=Path(attempt);self.items=[];self.seq=0;self.cache_hits=0;self.oracle_cls=oracle_cls
 def spawn(self):
  ctx=mp.get_context('spawn');pc,cc=ctx.Pipe();p=ctx.Process(target=worker_loop,args=(cc,str(self.out),self.run,self.deadline,self.oracle_cls));p.start();cc.close();self.seq+=1
  rec={'p':p,'conn':pc,'id':rt.wait_process(p.pid),'number':self.seq,'busy':None};self.items.append(rec);d.atomic(self.attempt/'workers'/f'{self.seq}_PROCESS.json',{'process':rec['id'],'parent':rt.proc(os.getpid())},True);return rec
 def evaluate(self,x,pool,science,on_progress):
  result={};pending=[]
  for sid in pool:
   task=task_id(science,sid,x,'RECOURSE');folder=self.out/'EVALUATION_CACHE'/task
   if (folder/'RESULT.json').exists():
    oracle=self.oracle_cls(sid,self.out/'CACHE_VALIDATION');result[sid]=oracle.validate_result(x,folder/'RESULT.json');oracle.close();self.cache_hits+=1
   else:pending.append({'op':'solve','sid':sid,'x':x,'task':task,'folder':str(folder)})
  while len(self.items)<min(4,len(pool)):self.spawn()
  while pending or any(it['busy'] for it in self.items):
   if time.monotonic()>=self.deadline or (self.out/'USER_STOP.json').exists() or (self.out/'PAUSE_RESOURCE.json').exists():raise TimeoutError('WORKER_DEADLINE_OR_COOPERATIVE_PAUSE')
   for it in list(self.items):
    if it['busy'] is None and pending:
     it['busy']=pending.pop(0);it['conn'].send(it['busy'])
    if it['conn'].poll(.02):
     try:ans=it['conn'].recv()
     except EOFError:ans=None
     if ans:
      task=it['busy'];it['busy']=None
      if ans['passed']:result[ans['sid']]=ans['record']
      else:
       d.atomic(self.attempt/'worker_failures'/f'{time.time_ns()}.json',ans,True)
       allowed,n=rt.retry_allowed(self.out,task['task'],ans['type']+ans['message'],3,ans['type'] in ('OSError','FileNotFoundError','TimeoutError'))
       if not allowed:raise RuntimeError('WORKER_'+ans['type']+': '+ans['message'])
       time.sleep(min([2,10,30][n-1],max(0,self.deadline-time.monotonic())));pending.append(task)
    if not it['p'].is_alive():
     task=it['busy'];it['p'].join();self.record_wait(it);self.items.remove(it)
     if task:
      allowed,n=rt.retry_allowed(self.out,task['task'],'WORKER_EXIT',3,True)
      if not allowed:raise RuntimeError('WORKER_RETRY_LIMIT')
      pending.append(task)
     self.spawn()
   on_progress(len(result))
  if set(result)!=set(pool):raise ValueError('INCOMPLETE_93_EVALUATION')
  return [result[s] for s in pool]
 def record_wait(self,it):
  d.atomic(self.attempt/'workers'/f'{it["number"]}_WAIT.json',{'pid':it['p'].pid,'identity':it['id'],'actual_parent_wait':True,'returncode':it['p'].exitcode,'epoch':time.time()},True)
 def close(self):
  for it in self.items:
   if it['p'].is_alive():
    try:it['conn'].send({'op':'close'})
    except (BrokenPipeError,EOFError):pass
  for it in self.items:
   it['p'].join(timeout=max(0,min(30,DEADLINE+90-time.monotonic())))
   if it['p'].is_alive():
    if not rt.same(it['id'],rt.proc(it['p'].pid)):raise ValueError('OWN_WORKER_IDENTITY_MISMATCH')
    it['p'].terminate();it['p'].join(timeout=15)
   if it['p'].is_alive():raise RuntimeError('OWN_WORKER_CLEANUP_FAILED')
   self.record_wait(it);it['conn'].close()
  self.items=[]

class Probability(core.Probability):
 def __init__(self,out,side_source):
  Path(out).parent.mkdir(parents=True,exist_ok=True)
  b.Probability.__init__(self,out);previous=np.load(rt.verify_ref(side_source));self.add([(str(k),int(i)) for k,i in zip(previous['active_kind'],previous['active_id'])])
  owned_json(Path(out)/'RESTORED_ACTIVE_SIDES.json',{'source':side_source,'basis_restored':False,'active_sides':len(self.active)})

def replay_risk(folder,q):
 folder=Path(folder);raw=np.load(folder/'RAW_FINAL.npz')
 if not np.array_equal(raw['q'],q):raise ValueError('RISK_Q_IDENTITY_FAILURE')
 ident=d.read(folder/'RAW_FINAL_IDENTITY.json')
 if ident['status']!='ROWGEN_CLOSED':raise RuntimeError('RISK_ROWGEN_NUMERICAL_PARTIAL')
 risk=b.risk_upper(q,float(raw['beta0']),raw['plus'],raw['minus'])
 strict_folder=folder/'STRICT_CUT'
 if (strict_folder/'PROBABILITY.npz').exists() and (folder/'CERTIFICATE.json').exists():
  rec=d.read(folder/'CERTIFICATE.json');strict=rec['strict'];z=np.load(strict_folder/'PROBABILITY.npz')
  if not b.probability_check(z['pstar'])['passed']:raise ValueError('RISK_PROBABILITY_CERTIFICATE_FAILURE')
 else:
  strict_folder=folder/('STRICT_RECOVERY_'+uuid.uuid4().hex);strict=core.certify(raw['p'],np.array(b.GLOBAL_QBARS),b.GLOBAL_POOL,strict_folder)
 if abs(risk['risk_upper']-d.read(folder/'RISK_UPPER.json')['risk_upper'])>1e-8:raise ValueError('RISK_UPPER_REPLAY_MISMATCH')
 return raw['p'],risk,strict,{'raw_final':rt.ref(folder/'RAW_FINAL.npz'),'rowgen_closed':True,'strict':strict,'risk_upper':risk['risk_upper'],'strict_folder':str(strict_folder)}

def solve_master(state,model,v,folder,task,deadline,require_basis=True):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);CONTEXT.clear();CONTEXT.update(task=task)
 raw,call=raw_for(task,'MASTER')
 if raw is None:
  log=folder/f'NATIVE_{time.time_ns()}.log';b.master_config(model,log,deadline-time.monotonic());cid=optimize(model,'MASTER',{'task':task,'directory':str(folder),'matrix':d.identity(state)})
  call=d.read(OUT/'CALLS'/cid/'END.json');raw=dict(np.load(rt.verify_ref(call['raw'])))
  if require_basis and not b.basis_accepted(log):
   # One explicitly logged same-matrix cold-basis retry, never disguised as warm.
   marker=folder/'COLD_BASIS_RETRY.json'
   if marker.exists():raise RuntimeError('BASIS_REJECTED_AFTER_COLD_RETRY')
   d.atomic(marker,{'identity':d.identity(state),'first_call':cid,'cold_start':True},True);model.reset();return solve_master(state,model,v,folder,task+'/COLD',deadline,False)
 if 'Pi' in raw:
  lb,cert=b.lagrangian(state['A'],state['rhs'],state['sense'],state['cost'],state['lo'],state['hi'],raw['Pi'])
  owned_json(folder/'LOWER_CERTIFICATE.json',{'LB':lb,'identity':d.identity(state),'call':call['id'],'kind':'LAGRANGIAN_COMPLETE_VARIABLE_BOX'})
 else:raise RuntimeError('MASTER_NO_VALID_DUAL')
 if call.get('status')!=gp.GRB.OPTIMAL:raise TimeoutError('MASTER_NATIVE_PARTIAL_'+str(call.get('status')))
 z={'x':raw['X'],'Pi':raw['Pi'],'RC':raw['RC'],'VBasis':raw['VBasis'].astype(int),'CBasis':raw['CBasis'].astype(int)}
 rv=b.directed_row_violations(state['A'],state['rhs'],state['sense'],z['x']);non_scene=np.array([not b.scenario_name(n) for n in state['names']['rows']]);check=b.row_audit(state['A'],state['rhs'],state['sense'],state['lo'],state['hi'],z['x'])
 owned_json(folder/'PRIMAL_AUDIT.json',{'all':check,'physical_probability':float(max(0,np.max(rv[non_scene]))),'benders_separate':float(max(0,np.max(rv[~non_scene])))})
 if max(0,np.max(rv[non_scene]))>1e-9 or check['bound']>1e-9 or not np.isfinite(z['x']).all():raise RuntimeError('FIRST_STAGE_OR_NUMERIC_FAILED')
 return z,lb,rt.ref(OUT/'CALLS'/call['id']/'END.json')

# The original scheduler remains readable above; this is the active formal implementation.
from memory_worker import Workers
