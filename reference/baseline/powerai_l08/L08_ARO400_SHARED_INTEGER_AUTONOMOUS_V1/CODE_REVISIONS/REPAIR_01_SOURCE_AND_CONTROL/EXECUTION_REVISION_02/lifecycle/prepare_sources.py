"""Build an isolated revision from hash-verified original tools, never overwrite them."""
from pathlib import Path
import hashlib,json,shutil,difflib,ast
ROOT=Path('/home/Shixin/ev-assignment/L08_DRO_R44_POOLED_ROOT_CONTINUATION/AUTONOMOUS_CONVERGENCE')
REV=Path(__file__).resolve().parent
if (REV/'CODE_SHA.json').exists():raise SystemExit('REVISION_SOURCE_ALREADY_FROZEN')
original=json.loads((ROOT/'CODE_SHA.json').read_text())
for name,value in original.items():
 source=ROOT/'tools'/name
 if hashlib.sha256(source.read_bytes()).hexdigest()!=value:raise ValueError('ORIGINAL_SOURCE_SHA_MISMATCH '+name)
 if name not in ('engine.py','autonomous_controller.py'):shutil.copyfile(source,REV/name)
def replace(source,old,new):
 if source.count(old)!=1:raise ValueError('SOURCE_ANCHOR_COUNT '+old[:80])
 return source.replace(old,new,1)
old_engine=(ROOT/'tools/engine.py').read_text();engine=old_engine
engine=replace(engine,'import core\n','import core\nfrom memory_policy import RecourseMemLimit,SchedulerStopped\n')
engine=replace(engine,'class Oracle(b.Oracle):\n','''class Oracle(b.Oracle):
 def __init__(self,sid,root,env=None,context=None):
  super().__init__(sid,root)
  self.owned_env=env;self.lifecycle_context=context or {};self.dispatch=None;self.memory_provider=None
  self.native_kind='RECOURSE_ORACLE'
 def close(self,discard=False):
  model=self.model
  self.model=None;self.y=None;self.rows=None;self.memory_provider=None;self.dispatch=None
  if model is not None:model.dispose()
  if discard:self.W=None;self.T=None;self.a=None;self.owned_env=None
''')
engine=replace(engine,"CONTEXT.clear();CONTEXT.update(task=task,sid=self.sid)","CONTEXT.clear();CONTEXT.update({**(self.dispatch or {}),'task':task,'sid':self.sid})")
engine=replace(engine,"raw,call=raw_for(task,'RECOURSE_ORACLE')","raw,call=raw_for(task,self.native_kind)")
engine=replace(engine,"self.model=gp.Model(f'R44_ORACLE_{self.sid}');self.model.Params.OutputFlag=0","if self.owned_env is None:raise ValueError('ORACLE_NATIVE_ENV_MUST_BE_EXPLICIT')\n    self.model=gp.Model(f'R44_ORACLE_{self.sid}',env=self.owned_env);self.model.Params.OutputFlag=0")
engine=replace(engine,"cid=optimize(self.model,'RECOURSE_ORACLE',{'sid':self.sid,'task':task})","""if self.lifecycle_context.get('test_mode'):
    if 'TESTS' not in OUT.parts:raise ValueError('TEST_MEMORY_OVERRIDE_OUTSIDE_TEST_NAMESPACE')
    self.model.Params.SoftMemLimit=self.test_soft_limit()
   CONTEXT['configured_limit_decimal_GB']=float(self.model.Params.SoftMemLimit)
   if self.dispatch is not None:self.dispatch['configured_limit_decimal_GB']=float(self.model.Params.SoftMemLimit)
   cid=optimize(self.model,self.native_kind,{'sid':self.sid,'task':task})""")
engine=replace(engine,"if call.get('status') in (gp.GRB.TIME_LIMIT,gp.GRB.MEM_LIMIT,gp.GRB.INTERRUPTED):raise TimeoutError('RECOURSE_NATIVE_RETRYABLE_'+str(call.get('status')))","""if call.get('status')==gp.GRB.MEM_LIMIT:
     raise RecourseMemLimit(dict(self.dispatch or {},kind='RECOURSE_MEM_LIMIT',native_status=int(gp.GRB.MEM_LIMIT),sid=self.sid,task_id=task,call_id=call['id'],memory=self.memory_provider() if self.memory_provider else None,corrective_action_required=True))
    if call.get('status') in (gp.GRB.TIME_LIMIT,gp.GRB.INTERRUPTED):
     kind='GLOBAL_WALL_LIMIT' if time.monotonic()>=deadline else 'USER_REQUESTED_STOP' if (OUT/'USER_STOP.json').exists() else 'WAIT_RESOURCE' if (OUT/'PAUSE_RESOURCE.json').exists() else 'LOCAL_NATIVE_BUDGET'
     raise SchedulerStopped({'kind':kind,'native_status':int(call['status']),'sid':self.sid,'task_id':task})""")
engine+='\n# The original scheduler remains readable above; this is the active formal implementation.\nfrom memory_worker import Workers\n'
(REV/'engine.py').write_text(engine)
old_controller=(ROOT/'tools/autonomous_controller.py').read_text();controller=old_controller
controller=replace(controller,'b=e.b\n','b=e.b\nfrom memory_policy import ResourceRecoveryNeeded,ResourceRetryExhausted,SchedulerStopped,controller_exit\n')
controller=replace(controller,"code={p.name:d.sha(p) for p in (out/'tools').glob('*.py')};frozen=d.read(out/'CODE_SHA.json')","code={p.name:d.sha(p) for p in Path(__file__).resolve().parent.glob('*.py')};frozen=d.read(Path(__file__).resolve().parent/'CODE_SHA.json')")
controller=replace(controller,"x=np.load(rt.verify_ref(meta['accepted']))['accepted'];t=time.monotonic();records=workers.evaluate(x,b.GLOBAL_POOL,meta['science_id'],lambda n:publish(n));meta['timing']['parallel_recourse_wall']=meta['timing'].get('parallel_recourse_wall',0)+time.monotonic()-t","""x=np.load(rt.verify_ref(meta['accepted']))['accepted'];t=time.monotonic()
    try:records=workers.evaluate(x,b.GLOBAL_POOL,meta['science_id'],lambda n:publish(n))
    finally:
     meta['timing']['parallel_recourse_wall']=meta['timing'].get('parallel_recourse_wall',0)+time.monotonic()-t
     meta['timing']['last_evaluation_components']=dict(workers.last_accounting)""")
controller=replace(controller,' except TimeoutError as exc:\n  status=',""" except ResourceRetryExhausted as exc:
  status=exc.event['kind'];d.atomic(attempt/'RESOURCE_TERMINAL.json',{'event':exc.event,'phase':meta['phase'],'traceback':traceback.format_exc()},True)
 except ResourceRecoveryNeeded as exc:
  status='RESOURCE_RECOVERY_NEEDED';d.atomic(attempt/'RECOVERABLE_RESOURCE.json',{'event':exc.event,'phase':meta['phase'],'traceback':traceback.format_exc()},True)
 except SchedulerStopped as exc:
  status=exc.event['kind'] if exc.event['kind'] in ('GLOBAL_WALL_LIMIT','USER_REQUESTED_STOP','WAIT_RESOURCE') else 'RETRYABLE_NATIVE_OR_INFRASTRUCTURE';d.atomic(attempt/'SCHEDULER_STOP.json',{'event':exc.event,'phase':meta['phase']},True)
 except TimeoutError as exc:
  status=""")
controller=replace(controller,"  workers.close()\n  if prob is not None:prob.close()\n  if model is not None:model.dispose()","""  cleanup_errors=[]
  for label,fn in [('workers',workers.close),('probability',None if prob is None else prob.close),('master',None if model is None else model.dispose)]:
   if fn is None:continue
   try:fn()
   except BaseException as cleanup_exc:cleanup_errors.append({'component':label,'type':type(cleanup_exc).__name__,'message':str(cleanup_exc)})
  d.atomic(attempt/'CLEANUP_ERRORS.json',{'errors':cleanup_errors,'original_status_retained':status,'worker_timing':workers.last_accounting},True)
  if cleanup_errors and status=='RUNNING':status='RESOURCE_RECOVERY_NEEDED'""")
controller=replace(controller,"return 75 if status in ('WAIT_RESOURCE','RETRYABLE_NATIVE_OR_INFRASTRUCTURE') else 0","return controller_exit(status)")
(REV/'autonomous_controller.py').write_text(controller)
diff=''.join(difflib.unified_diff(old_engine.splitlines(True),engine.splitlines(True),fromfile='ROOT/tools/engine.py',tofile='REV/engine.py'))+'\n'+''.join(difflib.unified_diff(old_controller.splitlines(True),controller.splitlines(True),fromfile='ROOT/tools/autonomous_controller.py',tofile='REV/autonomous_controller.py'))
(REV/'PATCH_DIFF.patch').write_text(diff)
def functions(src):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
before,after=functions(old_controller),functions(controller)
unchanged=[name for name in before if before[name]==after[name]];changes=[name for name in before if before[name]!=after[name]]
if changes!=['scientific']:raise ValueError('UNEXPECTED_SCIENTIFIC_FUNCTION_CHANGE '+repr(changes))
eb,ea=functions(old_engine),functions(engine)
for name in ('task_id','replay_risk','solve_master','optimize','raw_for'):
 if eb[name]!=ea[name]:raise ValueError('UNAUTHORIZED_ENGINE_MATH_CHANGE '+name)
record={'unchanged_original_files':{name:value for name,value in original.items() if name not in ('engine.py','autonomous_controller.py')},'controller_unchanged_functions':unchanged,'controller_changed_functions':changes,'engine_unchanged_task_id_probability_master_native_recorder':True,'old_SOURCE_CODE_SHA':original,'scope':'native ownership/cache scheduling, typed resources and cleanup/exit timing only'}
(REV/'SOURCE_COMPATIBILITY.json').write_text(json.dumps(record,indent=2))
for p in REV.glob('*.py'):compile(p.read_text(),str(p),'exec')
print(json.dumps({'prepared':str(REV),'changed_controller_functions':changes,'task_key_preserved':True,'science_calls':0}))
