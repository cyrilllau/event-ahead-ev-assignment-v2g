"""Typed resource policy and bounded native-model ownership; no math changes."""
from pathlib import Path
from collections import OrderedDict
import time,uuid,gc
import durable as d
import runtime as rt

class ResourceEvent(Exception):
 def __init__(self,event):self.event=event;super().__init__(event.get('kind','RESOURCE_EVENT'))
class RecourseMemLimit(ResourceEvent):pass
class ResourceRecoveryNeeded(ResourceEvent):pass
class ResourceRetryExhausted(ResourceEvent):pass
class SchedulerStopped(ResourceEvent):pass
class PolicyViolation(Exception):pass

def stable_map(pool):
 values=sorted(map(int,pool))
 if len(values)!=len(set(values)):raise PolicyViolation('DUPLICATE_FROZEN_SID')
 return {str(sid):i%4 for i,sid in enumerate(values)}
def accept_token(expected,event):return bool(expected and event.get('dispatch_token')==expected.get('dispatch_token') and event.get('generation')==expected.get('generation') and event.get('sid')==expected.get('sid') and event.get('slot')==expected.get('slot'))
RECOVERABLE={'WAIT_RESOURCE','RETRYABLE_NATIVE_OR_INFRASTRUCTURE','RESOURCE_RECOVERY_NEEDED'}
NORMAL={'CERTIFIED_CONVERGED','NUMERICAL_TARGET_REACHED_CERTIFICATION_PENDING','GLOBAL_WALL_LIMIT','USER_REQUESTED_STOP','NO_PROGRESS_DIAGNOSTIC','TEST_COMPLETED'}
def controller_exit(status):return 75 if status in RECOVERABLE else 0 if status in NORMAL else 70
def launcher_action(returncode,status):
 if returncode==0 and status not in NORMAL:return 'EXIT_PROTOCOL_FAILURE'
 if returncode==75 and status in RECOVERABLE:return 'RESUME_IF_ORIGINAL_BUDGET_AND_ATTEMPTS_ALLOW'
 if returncode==0 and status in NORMAL:return 'FINISH'
 return 'STOP_FAILURE'

class RetryBook:
 def __init__(self,out,policy):
  self.out=Path(out);self.policy=policy;self.path=self.out/'MEMORY_RETRY_STATE.json';self.state=d.read(self.path) if self.path.exists() else {'legacy':{},'corrective':{},'local':{},'recycles_total':0,'recycles_by_slot':{str(s):0 for s in range(4)}}
  self.legacy=d.read(self.out/'RETRIES.json') if (self.out/'RETRIES.json').exists() else {};self.legacy_sha=d.sha(self.out/'RETRIES.json') if (self.out/'RETRIES.json').exists() else None
 def save(self):d.atomic(self.path,self.state)
 def legacy_initial(self,task,evidence=None):
  if task['task'] not in self.policy.get('legacy_task_ids',[]):return False
  if task['task'] in self.state['legacy']:return False
  sig='TimeoutErrorRECOURSE_NATIVE_RETRYABLE_17';old=self.legacy.get(d.digest([task['task'],sig]),0)
  self.state['legacy'][task['task']]={'sid':task['sid'],'old_native17_timeout_retry_count':old,'old_signature':sig,'old_RETRIES_sha':self.legacy_sha,'new_alias':'RECOURSE_MEM_LIMIT','one_time_MEMORY_POLICY_REPAIR_V1_attempt_used':True,'evidence':evidence,'epoch':time.time()};self.save();return True
 def corrective(self,task,slot,clean_recheck):
  if clean_recheck:raise ResourceRetryExhausted({'kind':'SINGLE_ORACLE_RESOURCE_LIMIT','sid':task['sid'],'task_id':task['task'],'slot':slot,'production_SoftMemLimit_decimal_GB':3,'reason':'MEM_LIMIT repeated after verified disposal and fresh exclusive Env'})
  if task['task'] in self.state['legacy']:raise ResourceRetryExhausted({'kind':'SINGLE_ORACLE_RESOURCE_LIMIT','sid':task['sid'],'task_id':task['task'],'slot':slot,'reason':'legacy task already used the explicitly authorized corrective policy attempt'})
  count=self.state['corrective'].get(task['task'],0)
  if count>=self.policy['max_local_retries']:raise ResourceRetryExhausted({'kind':'RESOURCE_RETRY_EXHAUSTED','task_id':task['task'],'sid':task['sid'],'slot':slot})
  return count+1
 def commit_corrective(self,task,old_wait,new_ready):
  n=self.state['corrective'].get(task['task'],0)+1;self.state['corrective'][task['task']]=n
  self.state.setdefault('corrective_evidence',{})[task['task']]={'old_generation_real_WAIT':old_wait,'new_generation_READY':new_ready,'changed_resource_state':True,'epoch':time.time()};self.save();return n
 def recycle(self,slot):
  if self.state['recycles_total']>=self.policy['max_worker_recycles_total'] or self.state['recycles_by_slot'][str(slot)]>=self.policy['max_worker_recycles_per_slot']:raise ResourceRetryExhausted({'kind':'WORKER_RECYCLE_LIMIT','slot':slot,'state':self.state})
  self.state['recycles_total']+=1;self.state['recycles_by_slot'][str(slot)]+=1;self.save()
 def infrastructure(self,task,kind):
  key=d.digest([task['task'],kind]);n=self.state['local'].get(key,0)
  if n>=self.policy['max_local_retries']:raise ResourceRetryExhausted({'kind':'INFRASTRUCTURE_RETRY_EXHAUSTED','task_id':task['task'],'sid':task['sid'],'cause':kind})
  self.state['local'][key]=n+1;self.save();return n+1

class ModelCache:
 def __init__(self,slot,generation,owned_env,allowed,factory,policy,log,measure_override=None):
  self.slot=slot;self.generation=generation;self.env=owned_env;self.env_id=uuid.uuid4().hex;self.allowed=set(allowed);self.factory=factory;self.policy=policy;self.log=Path(log);self.objects=OrderedDict();self.busy=None;self.closed=False;self.measure_override=measure_override
 def emit(self,event,**kwargs):
  rt.append_event(self.log,{'event':event,'slot':self.slot,'generation':self.generation,'env_id':self.env_id,'epoch':time.time(),'monotonic':time.monotonic(),'object_count':len(self.objects),'native_model_count':sum(getattr(o,'model',None) is not None for o in self.objects.values()),'resident_sids':list(self.objects),'busy_sid':self.busy,**kwargs})
 def memory(self):
  if self.measure_override is not None:return self.measure_override(self)
  current=peak=None;unavailable='no live model exposing Env memory attributes'
  for oracle in self.objects.values():
   if getattr(oracle,'model',None) is None:continue
   try:current=float(oracle.model.MemUsed);peak=float(oracle.model.MaxMemUsed);unavailable=None;break
   except Exception as exc:unavailable=type(exc).__name__+': '+str(exc)
  p=rt.proc(__import__('os').getpid())
  return {'MemUsed_decimal_GB':current,'MaxMemUsed_decimal_GB':peak,'metric_unavailable':unavailable,'rss':None if p is None else p['rss'],'pss':None if p is None else p['pss'],'accounting':'one live Model.MemUsed for this shared Env; not sum of per-model copies; MaxMemUsed is peak only'}
 def evict(self,sid,reason):
  if sid==self.busy:raise PolicyViolation('EVICT_BUSY_ORACLE')
  before=self.memory();obj=self.objects.pop(sid);obj.close(discard=True);del obj;gc.collect();self.emit('MODEL_EVICTED',sid=sid,reason=reason,before=before,after=self.memory())
 def acquire(self,sid):
  if self.closed:raise PolicyViolation('CACHE_ALREADY_CLOSED')
  if sid not in self.allowed:raise PolicyViolation('SID_OUTSIDE_STABLE_SLOT')
  if self.busy is not None:raise PolicyViolation('CONCURRENT_TASK_IN_ONE_SLOT')
  current=self.memory();self.emit('BEFORE_ACQUIRE',sid=sid,memory=current)
  if current['MemUsed_decimal_GB'] is not None and current['MemUsed_decimal_GB']>self.policy['evict_above_decimal_GB']:
   while self.objects:
    current=self.memory()
    if current['MemUsed_decimal_GB'] is not None and current['MemUsed_decimal_GB']<=self.policy['evict_target_decimal_GB']:break
    self.evict(next(iter(self.objects)),'CURRENT_ENV_MEMORY_PRESSURE')
  if sid not in self.objects:
   while len(self.objects)>=self.policy['max_models_per_worker']:self.evict(next(iter(self.objects)),'HARD_COUNT_LIMIT')
   self.objects[sid]=self.factory(sid,self.env);self.emit('ORACLE_OBJECT_CREATED',sid=sid,memory=self.memory())
  else:self.objects.move_to_end(sid);self.emit('ORACLE_OBJECT_REUSED',sid=sid,memory=self.memory())
  self.busy=sid;return self.objects[sid]
 def release_busy(self):self.busy=None;self.emit('AFTER_TASK',memory=self.memory())
 def close(self):
  if self.closed:return
  self.busy=None
  while self.objects:self.evict(next(iter(self.objects)),'WORKER_ENV_SHUTDOWN')
  self.emit('ALL_MODELS_DISPOSED',memory=self.memory());self.closed=True
