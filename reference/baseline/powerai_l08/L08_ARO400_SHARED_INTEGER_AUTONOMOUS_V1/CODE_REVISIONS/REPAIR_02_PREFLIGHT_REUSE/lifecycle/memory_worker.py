"""Four fixed slots, one explicit Env each, bounded caches and typed recovery."""
from pathlib import Path
import multiprocessing as mp
from multiprocessing.connection import wait as ipc_wait
import os,time,uuid,traceback,collections,hashlib
import numpy as np
import gurobipy as gp
import durable as d
import runtime as rt
from memory_policy import (ModelCache,RetryBook,stable_map,accept_token,RecourseMemLimit,ResourceRecoveryNeeded,ResourceRetryExhausted,SchedulerStopped,PolicyViolation)

def policy_for(out,policy=None):
 if policy is not None:return policy
 revision=d.read(Path(out)/'REVISION.json');p=d.read(revision['policy']['path'])
 if d.sha(revision['policy']['path'])!=revision['policy']['sha256']:raise PolicyViolation('PATCH_POLICY_SHA_FAILURE')
 if p['production_soft_mem_decimal_GB']!=3:raise PolicyViolation('PRODUCTION_MEMORY_LIMIT_CHANGE_FORBIDDEN')
 return p

def worker_entry(conn,out,run,deadline,slot,generation,allowed,policy,logroot,oracle_cls=None):
 import engine as e
 e.configure(out,run,deadline);env=None;cache=None;exitcode=0;cleanup=[];reason='NORMAL';env_id=None
 folder=Path(logroot);folder.mkdir(parents=True,exist_ok=True);modules={'engine':e.__file__,'worker':__file__}
 try:
  env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
  def factory(sid,owned_env):
   cls=e.Oracle if oracle_cls is None else oracle_cls
   return cls(sid,Path(out)/'ORACLE_OBJECTS',env=owned_env,context={'slot':slot,'generation':generation,'test_mode':bool(policy.get('test_mode'))})
  cache=ModelCache(slot,generation,env,allowed,factory,policy,folder/'LIFECYCLE.jsonl');env_id=cache.env_id
  ready={'event':'READY','run_id':run['run_id'],'slot':slot,'generation':generation,'env_id':env_id,'owned_env_created_after_spawn':True,'process':rt.proc(os.getpid()),'modules':modules,'memory':cache.memory()};d.atomic(folder/'READY.json',ready,True);conn.send(ready)
  while True:
   msg=conn.recv()
   if msg['op']=='close':break
   token={k:msg[k] for k in ('sid','slot','generation','dispatch_token')}
   if msg['slot']!=slot or msg['generation']!=generation:raise PolicyViolation('DISPATCH_SLOT_GENERATION_MISMATCH')
   if time.monotonic()>=deadline:conn.send(dict(token,event='STOP',kind='GLOBAL_DEADLINE'));break
   if Path(out,'USER_STOP.json').exists():conn.send(dict(token,event='STOP',kind='USER_REQUESTED_STOP'));break
   if Path(out,'PAUSE_RESOURCE.json').exists():conn.send(dict(token,event='STOP',kind='LEASE_PAUSE'));break
   oracle=cache.acquire(msg['sid']);oracle.dispatch=dict(token,task_id=msg['task'],env_id=env_id,configured_limit_decimal_GB=3,clean_recheck=bool(msg.get('clean_recheck')));oracle.memory_provider=cache.memory
   if msg.get('clean_recheck') and len(cache.objects)!=1:raise PolicyViolation('CLEAN_ENV_RECHECK_NOT_EXCLUSIVE')
   conn.send(dict(token,event='STARTED',task_id=msg['task'],env_id=env_id,memory=cache.memory(),exclusive_oracle=len(cache.objects)==1))
   try:
    rec=oracle.solve_task(msg['x'],msg['task'],msg['folder'],deadline);conn.send(dict(token,event='RESULT',record=rec,env_id=env_id,memory=cache.memory()))
   except RecourseMemLimit as exc:
    reason='RECOURSE_MEM_LIMIT';exitcode=75;event=dict(exc.event,**token,event='MEM_LIMIT',env_id=env_id,memory=cache.memory(),corrective_action_required=True);d.atomic(folder/f'MEM_LIMIT_{msg["dispatch_token"]}.json',event,True);conn.send(event);break
   except SchedulerStopped as exc:
    reason=exc.event['kind'];conn.send(dict(token,event='STOP',**exc.event));break
   except BaseException as exc:
    reason='WORKER_ERROR';exitcode=70;event=dict(token,event='ERROR',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());d.atomic(folder/'ERROR.json',event,True);conn.send(event);break
   finally:
    cache.release_busy()
    if oracle is not None:oracle.memory_provider=None;oracle.dispatch=None
 except BaseException as exc:
  reason='WORKER_ENTRY_OR_CHANNEL_ERROR';exitcode=70
  d.atomic(folder/'ENTRY_ERROR.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},True)
  try:conn.send({'event':'ENTRY_FAILURE','slot':slot,'generation':generation,'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
  except (BrokenPipeError,EOFError,OSError):pass
 finally:
  if cache is not None:
   try:cache.close()
   except BaseException as exc:cleanup.append({'phase':'models','type':type(exc).__name__,'message':str(exc)})
  if env is not None:
   try:env.dispose()
   except BaseException as exc:cleanup.append({'phase':'Env.dispose','type':type(exc).__name__,'message':str(exc)})
  d.atomic(folder/'ENV_CLOSED.json',{'slot':slot,'generation':generation,'env_id':env_id,'all_models_closed':not cleanup,'Env_dispose_returned':not cleanup,'cleanup_errors':cleanup,'exit_reason':reason,'exitcode':exitcode,'epoch':time.time()},True)
  conn.close()
 if cleanup:exitcode=70
 raise SystemExit(exitcode)

class Workers:
 def __init__(self,out,run,deadline,attempt,oracle_cls=None,policy=None):
  self.out=Path(out);self.run=run;self.deadline=deadline;self.attempt=Path(attempt);self.oracle_cls=oracle_cls;self.policy=policy_for(out,policy);self.pool=list(map(int,self.policy['frozen_pool']));self.owner=stable_map(self.pool);self.book=RetryBook(self.out,self.policy);self.items={};self.generations={};self.cache_hits=0;self.last_accounting={};self.closed=False;self.seq=0
  mapping={'run_id':run['run_id'],'frozen_pool_sorted':sorted(self.pool),'owner':self.owner,'counts':[sum(v==s for v in self.owner.values()) for s in range(4)],'rule':'index_in_sorted_frozen93 %4','no_work_stealing':True}
  d.atomic(self.out/'STABLE_SLOT_MAPPING.json',mapping,True)
  path=self.out/'WORKER_GENERATIONS.json';self.generations=d.read(path) if path.exists() else {str(s):-1 for s in range(4)}
 def spawn(self,slot,replacement=None):
  if slot in self.items:raise PolicyViolation('DUPLICATE_LIVE_SLOT')
  self.generations[str(slot)]+=1;generation=self.generations[str(slot)];d.atomic(self.out/'WORKER_GENERATIONS.json',self.generations)
  allowed=[sid for sid in sorted(self.pool) if self.owner[str(sid)]==slot];folder=self.attempt/'workers'/f'SLOT_{slot}_GEN_{generation}';folder.mkdir(parents=True,exist_ok=False);pc,cc=mp.get_context('spawn').Pipe();p=mp.get_context('spawn').Process(target=worker_entry,args=(cc,str(self.out),self.run,self.deadline,slot,generation,allowed,self.policy,str(folder),self.oracle_cls));p.start();cc.close()
  identity=rt.wait_process(p.pid)
  item={'p':p,'conn':pc,'identity':identity,'slot':slot,'generation':generation,'folder':folder,'ready':None,'busy':None,'state':'STARTING','replacement':replacement,'quarantine_at':None,'spawn_at':time.monotonic()};self.items[slot]=item;d.atomic(folder/'PROCESS.json',{'pid':p.pid,'identity':identity,'run_id':self.run['run_id'],'parent':rt.proc(os.getpid()),'slot':slot,'generation':generation,'allowed_sids':allowed,'module':str(Path(__file__).resolve())},True);return item
 def event(self,kind,**kwargs):rt.append_event(self.attempt/'SCHEDULER_EVENTS.jsonl',{'event':kind,'epoch':time.time(),'monotonic':time.monotonic(),**kwargs})
 def wait_record(self,it):
  it['p'].join(timeout=0);r={'actual_parent_wait':True,'returncode':it['p'].exitcode,'pid':it['p'].pid,'identity':it['identity'],'slot':it['slot'],'generation':it['generation'],'epoch':time.time()};d.atomic(it['folder']/'WAIT.json',r,True);return rt.ref(it['folder']/'WAIT.json')
 def evaluate(self,x,pool,science,on_progress):
  import engine as e
  if set(map(int,pool))!=set(self.pool):raise PolicyViolation('EVALUATION_POOL_CHANGED')
  start=time.monotonic();results={};pending={s:[] for s in range(4)};self.last_accounting={'active_wall':0.,'waiting_wall':0.,'backoff_wall':0.,'cleanup_wall':0.,'total_wall':0.,'task_backoff_sum_overlaps_active':0.,'loop_intervals_mutually_exclusive':True};last=start;terminal=None
  try:
   for sid in pool:
    task=e.task_id(science,sid,x,'RECOURSE');folder=self.out/'EVALUATION_CACHE'/task
    if (folder/'RESULT.json').exists():
     cls=e.Oracle if self.oracle_cls is None else self.oracle_cls;oracle=cls(sid,self.attempt/'CACHE_VALIDATION',env=None,context={'validation_only':True,'test_mode':self.policy.get('test_mode',False)})
     try:results[sid]=oracle.validate_result(x,folder/'RESULT.json');self.cache_hits+=1
     finally:oracle.close(discard=True)
    else:pending[self.owner[str(sid)]].append({'op':'solve','sid':sid,'x':x,'task':task,'folder':str(folder),'ready_at':start,'clean_recheck':False})
   self.event('PARTIAL_CACHE_AUDIT',point_sha=hashlib.sha256(x.tobytes()).hexdigest(),hits=len(results),missing=sorted(set(pool)-set(results)),task_ids={str(t['sid']):t['task'] for tasks in pending.values() for t in tasks})
   for slot in range(4):
    if slot not in self.items:self.spawn(slot)
   while any(pending.values()) or any(it['busy'] or it['state'] in ('STARTING','QUARANTINED') for it in self.items.values()):
    now=time.monotonic();dt=now-last;last=now
    category='active_wall' if any(it['busy'] and it['state']=='BUSY' for it in self.items.values()) else 'cleanup_wall' if any(it['state']=='QUARANTINED' for it in self.items.values()) else 'backoff_wall' if any(t['ready_at']>now for q in pending.values() for t in q) else 'waiting_wall';self.last_accounting[category]+=dt
    if now>=self.deadline:raise SchedulerStopped({'kind':'GLOBAL_WALL_LIMIT'})
    if (self.out/'USER_STOP.json').exists():raise SchedulerStopped({'kind':'USER_REQUESTED_STOP'})
    if (self.out/'PAUSE_RESOURCE.json').exists():raise SchedulerStopped({'kind':'WAIT_RESOURCE','reason':'LEASE_PAUSE'})
    # Reap dead generations before any dispatch. Others continue to report.
    for slot,it in list(self.items.items()):
     if it['p'].is_alive():
      if it['state']=='STARTING' and now-it['spawn_at']>30:raise ResourceRecoveryNeeded({'kind':'WORKER_READY_TIMEOUT','slot':slot,'generation':it['generation']})
      if it['state']=='QUARANTINED' and now-it['quarantine_at']>30:
       if not rt.same(it['identity'],rt.proc(it['p'].pid)):raise PolicyViolation('OWN_QUARANTINED_PID_IDENTITY_MISMATCH')
       it['p'].terminate();self.event('OWN_QUARANTINED_WORKER_TERMINATED_AFTER_TIMEOUT',slot=slot,generation=it['generation'])
      continue
     # Drain committed native events before interpreting the process exit.
     if not it.get('channel_eof') and it['conn'].poll():continue
     wait_ref=self.wait_record(it);closed=d.read(it['folder']/'ENV_CLOSED.json') if (it['folder']/'ENV_CLOSED.json').exists() else {'Env_dispose_returned':False,'note':'unexpected process exit; native address space released by OS'}
     task=it['busy'];recovery=it.get('recovery');it['conn'].close();del self.items[slot]
     self.event('GENERATION_EXITED',slot=slot,generation=it['generation'],WAIT=wait_ref,Env_closed=closed)
     if terminal is not None:continue
     if task is not None:
      if recovery:
       self.book.corrective(task,slot,task.get('clean_recheck',False));task=dict(task,clean_recheck=True,ready_at=now+2,backoff_start=now);self.book.recycle(slot);replacement={'task':task,'old_wait':wait_ref,'old_env_closed':closed,'cause':recovery}
      else:
       n=self.book.infrastructure(task,'WORKER_EXIT');self.book.recycle(slot);task=dict(task,ready_at=now+[2,10,30][n-1],backoff_start=now);replacement={'task':task,'old_wait':wait_ref,'old_env_closed':closed,'cause':{'kind':'WORKER_EXIT'}}
      self.spawn(slot,replacement)
     elif it['state'] not in ('CLOSING',):
      self.book.recycle(slot);self.spawn(slot)
    if terminal is not None:
     # Preserve any results already independently committed by other slots.
     for it in self.items.values():
      if it['busy'] is None and it['state']=='IDLE':
       try:it['conn'].send({'op':'close'});it['state']='CLOSING'
       except (BrokenPipeError,EOFError,OSError):pass
     if not any(it['busy'] for it in self.items.values()):raise terminal
    # No work stealing. Retry readiness never sleeps in this scheduler loop.
    for slot,it in list(self.items.items()):
     if terminal is not None or it['state']!='IDLE' or not it['p'].is_alive():continue
     queue=pending[slot]
     # Reserve a fresh Env for its corrective task while its backoff elapses.
     # Other slots continue; this slot must not warm another model first.
     eligible=(0 if queue[0]['ready_at']<=now else None) if queue and queue[0].get('first_after_recycle') else next((i for i,t in enumerate(queue) if t['ready_at']<=now),None)
     if eligible is None:continue
     task=queue.pop(eligible);token=uuid.uuid4().hex;task.update(slot=slot,generation=it['generation'],dispatch_token=token)
     if 'backoff_start' in task:self.last_accounting['task_backoff_sum_overlaps_active']+=now-task['backoff_start']
     try:it['conn'].send(task);it['busy']=task;it['state']='BUSY';self.event('DISPATCH',slot=slot,generation=it['generation'],sid=task['sid'],task_id=task['task'],dispatch_token=token,clean_recheck=task.get('clean_recheck',False))
     except (BrokenPipeError,EOFError,OSError):it['busy']=task;it['state']='QUARANTINED';it['quarantine_at']=now
    conns=[it['conn'] for it in self.items.values() if it['state']!='CLOSING'];ready=ipc_wait(conns,timeout=.05) if conns else []
    for conn in ready:
     it=next((item for item in self.items.values() if item['conn'] is conn),None)
     if it is None:continue
     try:ans=conn.recv()
     except (EOFError,OSError):it['channel_eof']=True;continue
     if ans.get('event')=='ENTRY_FAILURE':
      if ans['error_type'] in ('OSError','FileNotFoundError'):
       it['state']='QUARANTINED';it['quarantine_at']=now;self.event('WHITELIST_INFRASTRUCTURE_ENTRY_FAILURE',message=ans);continue
      raise PolicyViolation('WORKER_ENTRY_FAILURE '+ans['error_type']+': '+ans['message'])
     if ans.get('event')=='READY':
      if ans.get('slot')!=it['slot'] or ans.get('generation')!=it['generation']:raise PolicyViolation('READY_GENERATION_IDENTITY')
      it['identity']=ans['process'];it['ready']=ans;it['state']='IDLE';replacement=it.get('replacement');self.event('READY',**{k:v for k,v in ans.items() if k!='event'})
      if replacement:
       task=dict(replacement['task'],first_after_recycle=True);old_wait=d.read(rt.verify_ref(replacement['old_wait']))
       if old_wait['returncode'] is None:raise PolicyViolation('CORRECTIVE_RETRY_WITHOUT_REAL_OLD_WAIT')
       if replacement['cause'].get('kind')=='RECOURSE_MEM_LIMIT':self.book.commit_corrective(task,replacement['old_wait'],rt.ref(it['folder']/'READY.json'))
       pending[it['slot']].insert(0,task);self.event('RESOURCE_STATE_CHANGED_BEFORE_RETRY',slot=it['slot'],old_generation_wait=replacement['old_wait'],new_ready=rt.ref(it['folder']/'READY.json'),task_id=task['task'],native_env_released_and_recreated=True);it['replacement']=None
      continue
     if not accept_token(it['busy'],ans):self.event('LATE_OR_DUPLICATE_MESSAGE_IGNORED',message=ans);continue
     task=it['busy']
     if ans['event']=='STARTED':
      self.book.legacy_initial(task,{'new_READY':rt.ref(it['folder']/'READY.json'),'STARTED':{k:v for k,v in ans.items() if k!='record'},'old_ATTEMPT_000_WAIT':self.policy.get('old_attempt_wait')});continue
     if ans['event']=='RESULT':results[ans['sid']]=ans['record'];it['busy']=None;it['state']='IDLE';self.event('RESULT_ACCEPTED',sid=ans['sid'],slot=it['slot'],generation=it['generation'],task_id=task['task'],memory=ans['memory']);continue
     if ans['event']=='MEM_LIMIT':
      it['state']='QUARANTINED';it['quarantine_at']=now;it['recovery']=ans;self.event('QUARANTINED_NATIVE_MEM_LIMIT',message=ans)
      try:self.book.corrective(task,it['slot'],task.get('clean_recheck',False))
      except ResourceRetryExhausted as exc:terminal=exc
      continue
     if ans['event']=='STOP':raise SchedulerStopped({'kind':ans['kind'],'slot':it['slot']})
     if ans['event']=='ERROR':
      if ans['error_type'] in ('OSError','FileNotFoundError'):
       it['state']='QUARANTINED';it['quarantine_at']=now;self.event('WHITELIST_INFRASTRUCTURE_WORKER_EXIT',message=ans)
      else:raise RuntimeError('WORKER_COMPUTATION_ERROR '+ans['error_type']+': '+ans['message']+'\n'+ans['traceback'])
    on_progress(len(results))
   if terminal is not None:raise terminal
   if set(results)!=set(pool):raise PolicyViolation('INCOMPLETE_FROZEN93_EVALUATION')
   self.event('BATCH_COMPLETE',point_sha=hashlib.sha256(x.tobytes()).hexdigest(),results=len(results),slots={str(s):{'generation':it['generation'],'ready':it['ready']} for s,it in self.items.items()});return [results[s] for s in pool]
  finally:
   self.last_accounting['total_wall']=time.monotonic()-start;d.atomic(self.attempt/f'EVALUATION_TIMING_{time.time_ns()}.json',self.last_accounting,True)
 def close(self):
  if self.closed:return
  t=time.monotonic();errors=[]
  for it in self.items.values():
   if it['p'].is_alive():
    try:it['conn'].send({'op':'close'})
    except (BrokenPipeError,EOFError,OSError):pass
  for slot,it in list(self.items.items()):
   it['p'].join(timeout=max(0,min(30,self.deadline+90-time.monotonic())))
   if it['p'].is_alive():
    if not rt.same(it['identity'],rt.proc(it['p'].pid)):errors.append({'slot':slot,'reason':'OWN_PID_IDENTITY_MISMATCH'});continue
    it['p'].terminate();it['p'].join(timeout=15)
   if it['p'].is_alive():errors.append({'slot':slot,'reason':'OWN_WORKER_DID_NOT_EXIT'});continue
   if not (it['folder']/'WAIT.json').exists():self.wait_record(it)
   it['conn'].close();del self.items[slot]
  self.closed=True;self.last_accounting['close_cleanup_wall']=time.monotonic()-t
  d.atomic(self.attempt/'WORKER_CLEANUP.json',{'wall':time.monotonic()-t,'errors':errors,'all_owned_workers_waited':not errors,'production_signals_sent':[]},True)
  if errors:raise ResourceRecoveryNeeded({'kind':'OWN_WORKER_CLEANUP_FAILURE','errors':errors})
