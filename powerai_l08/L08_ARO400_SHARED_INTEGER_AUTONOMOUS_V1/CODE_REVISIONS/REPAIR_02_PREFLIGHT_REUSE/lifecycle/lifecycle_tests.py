from pathlib import Path
import sys,os,time,json,copy,uuid,subprocess,traceback
import numpy as np
import gurobipy as gp
import durable as d
import runtime as rt
import engine as e
from memory_policy import *
from testfixtures import TinyOracle,FakeOracle

def unit_tests(root,run,policy):
 root.mkdir(parents=True,exist_ok=True);mapping=stable_map(policy['frozen_pool']);assert [sum(v==s for v in mapping.values()) for s in range(4)]==[24,23,23,23];records=[];caches=[];objects=[]
 def measured(cache):
  cur=.01*len(cache.objects);return {'MemUsed_decimal_GB':cur,'MaxMemUsed_decimal_GB':.24,'rss':None,'pss':None,'metric_unavailable':None}
 for slot in range(4):
  allowed=[int(sid) for sid,owner in mapping.items() if owner==slot];caches.append(ModelCache(slot,0,None,allowed,lambda sid,env:FakeOracle(sid,env),policy,root/f'UNIT_SLOT_{slot}.jsonl',measured))
 for outer in range(12):
  for sid in sorted(policy['frozen_pool']):
   cache=caches[mapping[str(sid)]];oracle=cache.acquire(sid);objects.append(oracle);cache.release_busy()
  assert [len(c.objects) for c in caches]==[24,23,23,23]
 for c in caches:
  allowed=next(sid for sid in policy['frozen_pool'] if str(sid) in mapping and mapping[str(sid)]!=c.slot)
  try:c.acquire(allowed);raise AssertionError('cross-slot accepted')
  except PolicyViolation:pass
  c.close();c.close()
 assert all(o.model is None and o.y is None and o.rows is None and o.W is None and o.T is None and o.a is None and o.disposed==1 for o in objects)
 # Current memory pressure, distinct from a deliberately high historic peak.
 def pressure(cache):return {'MemUsed_decimal_GB':.16*len(cache.objects),'MaxMemUsed_decimal_GB':9.,'metric_unavailable':None,'rss':None,'pss':None}
 cache=ModelCache(0,0,None,list(range(30)),lambda sid,env:FakeOracle(sid,env),policy,root/'PRESSURE.jsonl',pressure)
 for sid in range(25):cache.acquire(sid);cache.release_busy()
 assert len(cache.objects)<24;cache.close()
 # Missing metrics must retain the hard-count gate.
 def unavailable(cache):return {'MemUsed_decimal_GB':None,'MaxMemUsed_decimal_GB':None,'metric_unavailable':'injected unavailable','rss':None,'pss':None}
 cache=ModelCache(0,0,None,list(range(30)),lambda sid,env:FakeOracle(sid,env),policy,root/'NO_METRIC.jsonl',unavailable)
 for sid in range(30):cache.acquire(sid);cache.release_busy();assert len(cache.objects)<=24
 cache.close();assert controller_exit('UNHANDLED_IMPLEMENTATION_ERROR')==70 and controller_exit('RESOURCE_RECOVERY_NEEDED')==75 and controller_exit('GLOBAL_WALL_LIMIT')==0
 assert launcher_action(0,'UNHANDLED_IMPLEMENTATION_ERROR')=='EXIT_PROTOCOL_FAILURE'
 task={'sid':359,'task':'task-stable'};p=copy.deepcopy(policy);p['legacy_task_ids']=['task-stable'];book=RetryBook(root/'RETRY_TEST',p);assert book.legacy_initial(task,{'test':True});assert not RetryBook(root/'RETRY_TEST',p).legacy_initial(task)
 try:book.corrective(task,0,False);raise AssertionError('legacy allowance reset')
 except ResourceRetryExhausted:pass
 p['legacy_task_ids']=[];book=RetryBook(root/'CLEAN_TEST',p)
 try:book.corrective(task,0,True);raise AssertionError('clean Env failed retry was unlimited')
 except ResourceRetryExhausted as exc:assert exc.event['kind']=='SINGLE_ORACLE_RESOURCE_LIMIT'
 expected={'sid':1,'slot':0,'generation':1,'dispatch_token':'new'};assert accept_token(expected,dict(expected));assert not accept_token(expected,dict(expected,generation=0,dispatch_token='old'))
 assert rt.deadline(run)==run['deadline_monotonic'];assert controller_exit('TEST_COMPLETED')==0
 result={'passed':True,'scope':'formal policy/cache/exit functions with labeled fake-native unit objects; not scientific Q','12_rounds_93_stable_owners':True,'slot_counts':[24,23,23,23],'no_cross_slot':True,'models_reused_without_unconditional_rebuild':True,'hard_count_and_pressure_and_missing_metrics_checked':True,'close_idempotent_native_refs_cleared':True,'current_MemUsed_not_historic_peak_controls_eviction':True,'old_failure_status0_rejected':True,'typed_75_70_0_protocol':True,'late_tokens_rejected':True,'retry_allowance_persistent':True,'clean_Env_second_limit_stops':True,'original_deadline_preserved':True,'native_calls':0};d.atomic(root/'UNIT_REPORT.json',result,True);return result

def tiny_child(root,run,policy):
 root=Path(root);deadline=min(rt.deadline(run)-120,time.monotonic()+240);e.configure(root,run,deadline);policy=dict(policy,test_mode=True,legacy_task_ids=[]);x=np.array([.5]);pool=policy['frozen_pool'];science='TEST_ONLY_MEMORY_LIFECYCLE_'+run['run_id'];cached_sid=sorted(pool)[1]
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();oracle=TinyOracle(cached_sid,root/'PREEXISTING_FIXTURE',env,{'test_mode':True,'slot':1,'generation':1});task=e.task_id(science,cached_sid,x,'RECOURSE');oracle.dispatch={'slot':1,'generation':1,'dispatch_token':'preexisting_fixture','task_id':task};record=oracle.solve_task(x,task,root/'EVALUATION_CACHE'/task,deadline);oracle.close(discard=True);env.dispose();before={ref['path']:ref['sha256'] for ref in record['sources']};w=e.Workers(root,run,deadline,root/'CONTROLLER',oracle_cls=TinyOracle,policy=policy);status='TEST_COMPLETED';failure=None
 try:
  records=w.evaluate(x,pool,science,lambda n:None)
  assert len(records)==93 and w.cache_hits==1
 except (ResourceRecoveryNeeded,ResourceRetryExhausted,SchedulerStopped) as exc:status='RESOURCE_RECOVERY_NEEDED' if isinstance(exc,ResourceRecoveryNeeded) else exc.event['kind'];failure=exc.event
 except BaseException:status='UNHANDLED_IMPLEMENTATION_ERROR';failure=traceback.format_exc()
 finally:w.close()
 events=[json.loads(line) for line in (root/'CONTROLLER/SCHEDULER_EVENTS.jsonl').read_text().splitlines()];mem=[r for r in events if r['event']=='QUARANTINED_NATIVE_MEM_LIMIT'];changed=[r for r in events if r['event']=='RESOURCE_STATE_CHANGED_BEFORE_RETRY'];waits=[d.read(p) for p in (root/'CONTROLLER/workers').glob('*/WAIT.json')];closes=[d.read(p) for p in (root/'CONTROLLER/workers').glob('*/ENV_CLOSED.json')]
 if status=='TEST_COMPLETED':
  assert mem and mem[0]['message']['native_status']==int(gp.GRB.MEM_LIMIT) and changed
  assert len(mem)==1 and len(changed)==1 and len(waits)==5
  assert not list((root/'CONTROLLER/workers').glob('*/ENTRY_ERROR.json'))
  assert all(v==d.sha(p) for p,v in before.items());assert any(r['returncode']==75 and r['actual_parent_wait'] for r in waits);assert all(r['actual_parent_wait'] for r in waits);assert all(r['Env_dispose_returned'] for r in closes)
  ledger=[json.loads(line) for line in (root/'OPTIMIZER_CALL_LEDGER.jsonl').read_text().splitlines()];ends=[r for r in ledger if r['phase']=='END'];assert any(r.get('status')==gp.GRB.MEM_LIMIT for r in ends);assert sum(r.get('status')==gp.GRB.OPTIMAL for r in ends)==93
  result={'status':status,'passed':True,'real_native_MEM_LIMIT':True,'native_mem_limit_events':len(mem),'successful_fixture_native_calls':93,'fixtures_are_not_scientific_Q':True,'stable_pool':pool,'preserved_preexisting_cache':True,'completed_fixture_results':93,'corrective_recovery_new_generation':True,'true_worker_WAIT_count':len(waits),'old_failed_worker_WAIT75':True,'explicit_Env_disposed':True,'original_deadline_utc':run['deadline_utc'],'elapsed_test':time.time(),'worker_cleanup':w.last_accounting}
 else:result={'status':status,'passed':False,'failure':failure}
 d.atomic(root/'RESULT.json',result,True);return controller_exit(status)

def run_all(rev,run,policy):
 root=Path(rev)/'TESTS'/time.strftime('%Y%m%dT%H%M%S',time.gmtime());root.mkdir(parents=True,exist_ok=False);started=time.monotonic();unit=unit_tests(root/'UNIT',run,policy);testroot=root/'NATIVE';testroot.mkdir();d.atomic(testroot/'RUN_CONTEXT.json',run,True);d.atomic(testroot/'TEST_POLICY.json',dict(policy,test_mode=True,legacy_task_ids=[]),True)
 command=[sys.executable,'-B',str(Path(__file__).resolve()),'--tiny-child',str(testroot)]
 with (root/'STDOUT.log').open('xb') as so,(root/'STDERR.log').open('xb') as se:
  child=subprocess.Popen(command,stdout=so,stderr=se);owner=rt.wait_process(child.pid);d.atomic(root/'PROCESS.json',{'child':owner,'parent':rt.proc(os.getpid()),'command':command},True)
  try:rc=child.wait(timeout=max(1,min(300,rt.deadline(run)-time.monotonic()-120)))
  except subprocess.TimeoutExpired:
   if not rt.same(owner,rt.proc(child.pid)):raise PolicyViolation('TEST_CONTROLLER_PID_IDENTITY_MISMATCH')
   child.terminate();rc=child.wait(timeout=30)
  d.atomic(root/'WAIT.json',{'actual_parent_wait':True,'returncode':rc,'child':owner},True)
 result=d.read(testroot/'RESULT.json') if (testroot/'RESULT.json').exists() else {'status':'MISSING_TEST_RESULT','passed':False}
 action=launcher_action(rc,result['status']);summary={'passed':unit['passed'] and result.get('passed') and rc==0 and action=='FINISH','unit':unit,'native_integration':result,'real_test_parent_WAIT':rt.ref(root/'WAIT.json'),'formal_launcher_action':action,'global_original_run_id':run['run_id'],'original_deadline_utc':run['deadline_utc'],'wall':time.monotonic()-started,'directory':str(root),'science_Q_calls_in_this_test':0};d.atomic(root/'TEST_REPORT.json',summary,True);d.atomic(Path(rev)/'TEST_REPORT.json',summary)
 if not summary['passed']:raise RuntimeError('REAL_MEM_LIMIT_INTEGRATION_NOT_ACCEPTED '+str(root))
 return summary

if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--tiny-child':
  root=Path(sys.argv[2]);raise SystemExit(tiny_child(root,d.read(root/'RUN_CONTEXT.json'),d.read(root/'TEST_POLICY.json')))
 raise SystemExit('use run_all from the revision coordinator')
