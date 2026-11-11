#!/usr/bin/env python3
"""Authorized MEMORY_RECOVERY_V1 launcher. Original RUN/config/code remain intact."""
from pathlib import Path
import os,sys,time,json,uuid,hashlib,subprocess,traceback,zipfile,argparse,shutil
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import durable as d
import runtime as rt
import engine as e
import autonomous_controller as ac
from memory_policy import launcher_action,controller_exit
ROOT=Path('/home/Shixin/ev-assignment/L08_DRO_R44_POOLED_ROOT_CONTINUATION/AUTONOMOUS_CONVERGENCE');REV=Path(__file__).resolve().parent;RUN_ID='R44_AUTO_beaee595eb5646398a86a72992c20955';MISSING=[359,360,365,369,374,378,379,380,385,389];POINT_SHA='d3d5f53f3000f5247e1f73027722a4b929a49def2320654e835e8e6ccf5bafcd';MODEL_SHA='f15a6b577d1c06f5ebe63e77ee51084ad54c63f5b0b42ecc4cfe95bbc95865a2'
def run_context():
 run=d.read(ROOT/'RUN.json');cfg=d.read(ROOT/'CONFIG.json')
 if run['run_id']!=RUN_ID or run['deadline_utc']!=1793697286.356393:raise ValueError('ORIGINAL_RUN_OR_DEADLINE_IDENTITY')
 if d.sha(ROOT/'CONFIG.json')!=run['config_sha'] or d.sha(ROOT/'AUTHORIZATION.json')!=run['authorization_sha']:raise ValueError('ORIGINAL_CONFIG_AUTH_HASH_CHANGED')
 run['source']=cfg['source'];e.configure(ROOT,run,rt.deadline(run)-120);return run,cfg
def owned_attempts_quiet():
 result=[]
 for p in sorted(ROOT.glob('ATTEMPT_*/PROCESS.json')):
  r=d.read(p);owner=r['child'];alive=rt.same(owner,rt.proc(owner['pid']));wait=d.read(p.parent/'WAIT.json') if (p.parent/'WAIT.json').exists() else None;result.append({'directory':str(p.parent),'live':alive,'owner':owner,'WAIT':wait})
 return result
def current_state_gate():
 store=d.Store(ROOT);s,meta=store.replay();identity=d.identity(s)
 if identity['matrix_sha']!=MODEL_SHA or s['A'].shape!=(91246,60874) or s['A'].nnz!=1475125 or len(s['dnames'])!=5137:raise ValueError('RECOVERY_MODEL_SNAPSHOT_MISMATCH')
 if meta['outer']!=29 or meta['phase']!='EVALUATE_93' or meta['best_LB']!=11874.599055028833:raise ValueError('RECOVERY_PHASE_BOUND_MISMATCH')
 if sum(e.b.scenario_name(n) for n in s['names']['rows'])!=218 or sum(e.b.probability_name(n) for n in s['names']['rows'])!=260:raise ValueError('RECOVERY_ROW_CLASSIFICATION')
 x=np.load(rt.verify_ref(meta['accepted']))['accepted']
 if hashlib.sha256(x.tobytes()).hexdigest()!=POINT_SHA:raise ValueError('RECOVERY_ACCEPTED_POINT_MISMATCH')
 _,z,proof=ac.solution_load(store,meta['last_solved'])
 if abs(proof['LB']-11874.599055029172)>1e-8:raise ValueError('RECOVERY_LB_REPLAY_MISMATCH')
 return store,s,meta,x,proof
def audit_cache(meta,x):
 pool=d.read(ROOT/'SOURCES.json')['pool'];valid=[];missing=[];old_objects={};all_valid=[]
 for reference in d.read(ROOT/'SOURCES.json')['dependencies']:rt.verify_ref(reference)
 for sid in pool:
  task=e.task_id(meta['science_id'],sid,x,'RECOURSE');folder=ROOT/'EVALUATION_CACHE'/task
  if not (folder/'RESULT.json').exists():missing.append(sid);continue
  oracle=e.Oracle(sid,REV/'VALIDATION',env=None,context={'validation_only':True})
  try:r=oracle.validate_result(x,folder/'RESULT.json')
  finally:oracle.close(discard=True)
  valid.append({'sid':sid,'task_id':task,'RESULT':rt.ref(folder/'RESULT.json'),'sources':r['sources'],'qL':r['qL'],'qU_audited':r['qU_audited']})
 # All original176 successful cache payloads are protected by exact SHA, not rerun.
 for p in (ROOT/'EVALUATION_CACHE').glob('*/RESULT.json'):
  r=d.read(p);paths=[p]+[Path(q['path']) for q in r['sources']]
  for path in paths:old_objects[str(path)]=d.sha(path)
  all_valid.append({'sid':r['sid'],'point_sha':r['point_sha'],'result':rt.ref(p)})
 ledger=[json.loads(line) for line in (ROOT/'OPTIMIZER_CALL_LEDGER.jsonl').read_text().splitlines()];ends=[r for r in ledger if r['phase']=='END'];counts={'all_BEGIN':sum(r['phase']=='BEGIN' for r in ledger),'success_recourse':sum(r['kind']=='RECOURSE_ORACLE' and r.get('status')==2 for r in ends),'mem_limit_recourse':sum(r['kind']=='RECOURSE_ORACLE' and r.get('status')==17 for r in ends),'probability_calls':sum(r['kind']=='PROBABILITY_ROWGEN' for r in ends),'master_calls':sum(r['kind']=='MASTER' for r in ends)}
 if any(sid not in MISSING for sid in missing):raise ValueError('UNEXPECTED_MISSING_RECOVERY_SID')
 result={'passed':True,'point_sha':POINT_SHA,'scientific_id':meta['science_id'],'cache_hits':len(valid),'missing':sorted(missing),'valid':valid,'old_native_counts':counts,'old_successful_results':all_valid,'old_payload_sha':old_objects,'cache_validation_optimizer_calls':0};d.atomic(REV/'PARTIAL93_CACHE_AUDIT.json',result,True);return result
def archive_old_controls():
 archive=REV/'ORIGINAL_STATE';archive.mkdir(exist_ok=False);mapping=[]
 names=['LAST_COMMITTED.json','CONTROLLER_STATE.json','PROGRESS.json','HEARTBEAT.json','LAUNCHER_OWNER.json','RETRIES.json','OPTIMIZER_CALL_LEDGER.jsonl','PAUSE_RESOURCE.json','RESOURCE_WAIT.json','CODE_SHA.json','RUN.json','CONFIG.json','AUTHORIZATION.json','PRODUCTION_PROTECTION_AFTER.json']
 for name in names:
  p=ROOT/name
  if p.exists():target=archive/name;shutil.copyfile(p,target);mapping.append({'original':str(p),'archived':str(target),'sha256':d.sha(p),'archived_sha256':d.sha(target)})
 for p in (ROOT/'FINAL').rglob('*'):
  if p.is_file():target=archive/'FINAL'/p.relative_to(ROOT/'FINAL');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);mapping.append({'original':str(p),'archived':str(target),'sha256':d.sha(p),'archived_sha256':d.sha(target)})
 d.atomic(archive/'OLD_TO_ARCHIVE_MAP.json',{'files':mapping,'old_sealed_manifest_snapshot_remains_authoritative_for_its_old_bytes':True,'old_manifest_not_claimed_to_match_future_live_mutable_pointers':True},True);return rt.ref(archive/'OLD_TO_ARCHIVE_MAP.json')
def prepare():
 run,cfg=run_context()
 if rt.deadline(run)-time.monotonic()<150:
  d.atomic(REV/'EXPIRED.json',{'status':'ORIGINAL_GLOBAL_DEADLINE_EXPIRED','run':run,'science_started':False},True);return
 with rt.cc.lock(ROOT/'RUN.lock',nonblock=True):
  attempts=owned_attempts_quiet()
  if any(r['live'] for r in attempts):d.atomic(REV/'EXISTING_LIVE_CONTROLLER.json',{'attempts':attempts,'new_writer_started':False},True);return
  old_failure=d.read(ROOT/'ATTEMPT_000/FAILURE.json');old_wait=d.read(ROOT/'ATTEMPT_000/WAIT.json');old_final=d.read(ROOT/'FINAL/REPORT.json')
  if old_failure['phase']!='EVALUATE_93' or old_failure['message']!='WORKER_TimeoutError: RECOURSE_NATIVE_RETRYABLE_17' or old_wait['returncode']!=0 or not old_wait['actual_parent_wait'] or old_final['status']!='UNHANDLED_IMPLEMENTATION_ERROR':raise ValueError('AUTHORIZED_TERMINAL_REOPEN_BINDING_FAILED')
  original_manifest=d.read(ROOT/'FINAL/SEALED_MANIFEST.json');checked=0
  for ref in original_manifest['files']:rt.verify_ref(ref);checked+=1
  d.atomic(REV/'ORIGINAL_FINAL_MANIFEST_CHECK.json',{'passed':True,'checked':checked,'performed_before_dynamic_pointer_updates':True,'manifest':rt.ref(ROOT/'FINAL/SEALED_MANIFEST.json')},True)
  store,s,meta,x,proof=current_state_gate();cache=audit_cache(meta,x);protection=ac.protect(ROOT);d.atomic(REV/'PROTECTION_BEFORE.json',protection,True)
  policy={'policy_version':'MEMORY_RECOVERY_V1','run_id':RUN_ID,'original_deadline_utc':run['deadline_utc'],'frozen_pool':d.read(ROOT/'SOURCES.json')['pool'],'production_soft_mem_decimal_GB':3.,'max_models_per_worker':24,'evict_above_decimal_GB':2.25,'evict_target_decimal_GB':2.,'max_local_retries':3,'max_worker_recycles_total':32,'max_worker_recycles_per_slot':8,'max_controller_attempts_including_old':5,'legacy_missing_sids_authorized':MISSING,'legacy_task_ids':[e.task_id(meta['science_id'],sid,x,'RECOURSE') for sid in MISSING],'old_attempt_wait':rt.ref(ROOT/'ATTEMPT_000/WAIT.json'),'old_failure':rt.ref(ROOT/'ATTEMPT_000/FAILURE.json'),'scientific_id_unchanged':meta['science_id'],'oracle_mathematical_version':'CONTINUE_01_frozen_v1','test_mode':False,'resource_reservation_bytes':24*1024**3}
  d.atomic(REV/'PATCH_POLICY.json',policy,True);lease,admission=rt.acquire(ROOT,run,protection)
  if not lease:d.atomic(REV/'RESOURCE_NOT_AVAILABLE.json',admission);return
  try:
   from lifecycle_tests import run_all
   tests=run_all(REV,run,policy)
  finally:rt.release(ROOT,lease)
  if not tests['passed']:raise RuntimeError('MEMORY_REPAIR_TESTS_NOT_ACCEPTED')
  frozen={p.name:d.sha(p) for p in REV.glob('*.py')};d.atomic(REV/'CODE_SHA.json',frozen,True)
  archive=archive_old_controls();resolution={'authorized_once':True,'authority_request':rt.ref(REV/'REQUEST.txt'),'run_id':RUN_ID,'phase_to_resume':'EVALUATE_93','outer_to_resume':29,'accepted_sha':POINT_SHA,'matrix_sha':MODEL_SHA,'old_FAILURE':rt.ref(ROOT/'ATTEMPT_000/FAILURE.json'),'old_WAIT':rt.ref(ROOT/'ATTEMPT_000/WAIT.json'),'old_final':rt.ref(ROOT/'FINAL/REPORT.json'),'old_CODE_SHA':rt.ref(ROOT/'CODE_SHA.json'),'archive_mapping':archive,'new_code_sha':rt.ref(REV/'CODE_SHA.json'),'policy':rt.ref(REV/'PATCH_POLICY.json'),'real_MEM_LIMIT_test':rt.ref(REV/'TEST_REPORT.json'),'original_deadline_utc':run['deadline_utc'],'original_deadline_monotonic':run['deadline_monotonic'],'budget_reset':False,'failure_chain':['native17','TimeoutError','retry without releasing overloaded resource state','RuntimeError wrapper','generic failure status','return0','historical final sealed despite failure'],'old_WAIT0_not_rewritten':True,'old_final_retained':True,'resumption_allowed_only_for_this_matched_terminal':True}
  d.atomic(ROOT/'RECOVERY_RESOLUTION.json',resolution,True);d.atomic(ROOT/'REVISION.json',{'revision':'MEMORY_RECOVERY_V1','run_id':RUN_ID,'path':str(REV),'code_sha':rt.ref(REV/'CODE_SHA.json'),'policy':rt.ref(REV/'PATCH_POLICY.json'),'recovery_resolution':rt.ref(ROOT/'RECOVERY_RESOLUTION.json')},True)
  meta['status']='RECOVERY_READY';meta['memory_repair_baseline_LB']=meta['best_LB'];meta['memory_policy_revision']='MEMORY_RECOVERY_V1';store.commit(s,meta)
  d.atomic(REV/'RECOVERY_READY.json',{'run_id':RUN_ID,'phase':meta['phase'],'outer':meta['outer'],'cache_hits':cache['cache_hits'],'missing':cache['missing'],'LB_retained':meta['best_LB'],'current_LB_replayed':proof['LB'],'original_remaining_s':rt.deadline(run)-time.monotonic(),'science_calls_during_prepare':0,'test_calls_separately_recorded':True},True)
  print(json.dumps(d.read(REV/'RECOVERY_READY.json')))

def verify_revision():
 revision=d.read(ROOT/'REVISION.json')
 if revision['run_id']!=RUN_ID or Path(revision['path'])!=REV:raise ValueError('ACTIVE_REVISION_IDENTITY')
 for name,value in d.read(rt.verify_ref(revision['code_sha'])).items():
  if d.sha(REV/name)!=value:raise ValueError('REVISION_SOURCE_SHA_FAILURE '+name)
 resolution=d.read(rt.verify_ref(revision['recovery_resolution']))
 for key in ('old_FAILURE','old_WAIT','old_final','old_CODE_SHA','policy'):rt.verify_ref(resolution[key])
 if resolution['old_FAILURE']['sha256']!=d.sha(ROOT/'ATTEMPT_000/FAILURE.json'):raise ValueError('OLD_FAILURE_RECOVERY_BINDING_CHANGED')
 if not d.read(REV/'TEST_REPORT.json')['passed']:raise ValueError('NATIVE_MEM_TEST_NOT_ACCEPTED')
 return revision
def quiescent(attempt):
 live=[]
 for p in (attempt/'workers').glob('*/PROCESS.json'):
  r=d.read(p);owner=r.get('identity');ready=p.parent/'READY.json'
  if ready.exists():owner=d.read(ready)['process']
  if owner and rt.same(owner,rt.proc(owner['pid'])):live.append(owner)
 return live
def final_revision(run,cfg,status,attempt):
 store=d.Store(ROOT);s,meta=store.replay();folder=ROOT/'FINAL_REVISIONS/MEMORY_RECOVERY_V1';folder.mkdir(parents=True,exist_ok=False);result=ac.summary(ROOT,run,s,meta,attempt);audit=d.read(REV/'PARTIAL93_CACHE_AUDIT.json');unchanged=all(d.sha(path)==value for path,value in audit['old_payload_sha'].items());new_calls=[]
 ledger=[json.loads(line) for line in (ROOT/'OPTIMIZER_CALL_LEDGER.jsonl').read_text().splitlines()];old_ids=set(json.loads(line)['id'] for line in (REV/'ORIGINAL_STATE/OPTIMIZER_CALL_LEDGER.jsonl').read_text().splitlines());new_ends=[r for r in ledger if r['phase']=='END' and r['id'] not in old_ids]
 result.update(status=status,repair_LB_gain=meta['best_LB']-11874.599055028833,old176_payloads_unchanged=unchanged,original_deadline_utc=run['deadline_utc'],original_total_wall=time.time()-run['start_utc'],revision='MEMORY_RECOVERY_V1',new_native_success_recourse=sum(r['kind']=='RECOURSE_ORACLE' and r.get('status')==2 for r in new_ends),new_native_MEM_LIMIT=sum(r['kind']=='RECOURSE_ORACLE' and r.get('status')==17 for r in new_ends),strict_UB_still_null=meta.get('strict_UB') is None,original_final_retained=True,production_signals_sent=[])
 controls=folder/'CONTROL_SNAPSHOT';controls.mkdir()
 for name in ('LAST_COMMITTED.json','PROGRESS.json','HEARTBEAT.json','CONTROLLER_STATE.json','MEMORY_RETRY_STATE.json','WORKER_GENERATIONS.json','OPTIMIZER_CALL_LEDGER.jsonl','LATEST_FINAL.json'):
  if (ROOT/name).exists():shutil.copyfile(ROOT/name,controls/name)
 d.atomic(folder/'REPORT.json',result,True);d.atomic(folder/'STATE_REFERENCES.json',{'last_committed':rt.ref(controls/'LAST_COMMITTED.json'),'last_solved':meta['last_solved'],'working_matrix_identity':d.identity(s),'phase':meta['phase'],'source_policy':rt.ref(REV/'PATCH_POLICY.json')},True)
 d.atomic(folder/'CONTROL_PROTOCOL_TABLE.json',{'old_failure_WAIT':0,'old_failure_status':'UNHANDLED_IMPLEMENTATION_ERROR','new_recoverable_status_exit':75,'new_failed_terminal_exit':70,'new_normal_terminal_exit':0,'native_MEM_LIMIT_is_not_Q_upper':True,'WAIT0_with_failed_status':'EXIT_PROTOCOL_FAILURE'},True)
 # Preserve all new attempt/Env/native records and the exact end-state pointers.
 refs=[]
 for base in (REV,attempt,folder):
  refs.extend(rt.ref(p) for p in sorted(base.rglob('*')) if p.is_file() and p.name!='SEALED_MANIFEST.json' and not p.name.endswith(('.tmp','.lock')) and p.name not in ('launcher.log','prepare.log'))
 d.atomic(folder/'SEALED_MANIFEST.json',{'files':refs,'sealed_after_true_WAIT_and_lease_release':True,'snapshot_of_end_controls':True},True);d.atomic(ROOT/'LATEST_FINAL.json',{'revision':'MEMORY_RECOVERY_V1','report':rt.ref(folder/'REPORT.json'),'manifest':rt.ref(folder/'SEALED_MANIFEST.json'),'status':status})
 delivery=folder/'delivery';delivery.mkdir();target=delivery/'R44_MEMORY_LIFECYCLE_REPAIR_V1_PRO_ANALYSIS.zip'
 with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for base,prefix in ((REV,'revision'),(attempt,attempt.name),(folder,'final')):
   for p in sorted(base.rglob('*')):
    if p.is_file() and 'delivery' not in p.parts and not p.name.endswith(('.tmp','.lock')) and p.stat().st_size<50*1024**2:z.write(p,prefix+'/'+str(p.relative_to(base)))
 d.atomic(delivery/'PACKAGE_INDEX.json',{'analysis':rt.ref(target),'full_evidence_server_ROOT':str(ROOT),'original_data_reference':rt.ref(ROOT/'delivery/PACKAGE_RESULT.json'),'current_state_reference':rt.ref(folder/'STATE_REFERENCES.json'),'omitted_large_payloads_are_not_claimed_independent_full_science_package':True},True)

def launcher():
 run,cfg=run_context();verify_revision()
 if (ROOT/'FINAL_REVISIONS/MEMORY_RECOVERY_V1/REPORT.json').exists():print('MATCHED_REVISION_ALREADY_TERMINAL');return
 try:lock=rt.cc.lock(ROOT/'RUN.lock',nonblock=True);lock.__enter__()
 except BlockingIOError:print('EXISTING_WRITER_LOCK_HELD_NO_DUPLICATE');return
 lease=None;attempt=None;status='UNHANDLED_IMPLEMENTATION_ERROR'
 try:
  if any(r['live'] for r in owned_attempts_quiet()):print('EXISTING_CONTROLLER_ALIVE_NO_DUPLICATE');return
  protection=d.read(REV/'PROTECTION_BEFORE.json');d.atomic(ROOT/'LAUNCHER_OWNER.json',{'process':rt.proc(os.getpid()),'run_id':RUN_ID,'revision':'MEMORY_RECOVERY_V1','source':str(Path(__file__).resolve()),'epoch':time.time()});d.atomic(ROOT/'LATEST_FINAL.json',{'state':'RECOVERY_RUNNING','previous_final':rt.ref(ROOT/'FINAL/REPORT.json'),'recovery_resolution':rt.ref(ROOT/'RECOVERY_RESOLUTION.json')})
  for index in range(1,5):
   attempt=ROOT/f'ATTEMPT_{index:03d}'
   if (attempt/'WAIT.json').exists():continue
   if attempt.exists() and any(attempt.iterdir()):raise ValueError('UNREGISTERED_NONEMPTY_ATTEMPT_NO_OVERWRITE')
   wait_start=time.monotonic()
   while lease is None and time.monotonic()<rt.deadline(run)-120:
    lease,inventory=rt.acquire(ROOT,run,protection)
    if lease:break
    d.atomic(REV/'RESOURCE_WAIT.json',{'inventory':inventory,'elapsed':time.monotonic()-wait_start,'epoch':time.time()})
    if time.monotonic()-wait_start>=900:status='RESOURCE_NOT_AVAILABLE';break
    time.sleep(10)
   if lease is None:status='GLOBAL_WALL_LIMIT' if time.monotonic()>=rt.deadline(run)-120 else 'RESOURCE_NOT_AVAILABLE';break
   attempt.mkdir(exist_ok=True);d.atomic(attempt/'BEGIN.json',{'run_id':RUN_ID,'revision':'MEMORY_RECOVERY_V1','revision_source':str(REV),'epoch':time.time(),'original_remaining_s':rt.deadline(run)-time.monotonic(),'lease':lease},True);command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',attempt.name]
   with (attempt/'STDOUT.log').open('xb') as so,(attempt/'STDERR.log').open('xb') as se:child=subprocess.Popen(command,stdout=so,stderr=se,start_new_session=True)
   owner=rt.wait_process(child.pid,expected=str(Path(__file__).resolve()));d.atomic(attempt/'PROCESS.json',{'child':owner,'parent':rt.proc(os.getpid()),'command':command,'revision_code_sha':rt.ref(REV/'CODE_SHA.json')},True);renew=time.monotonic()+6900;cooperative=False
   while child.poll() is None:
    now=time.monotonic()
    if now>=renew and not cooperative:
     new,inv=rt.acquire(ROOT,run,protection,renew=lease)
     if new:lease=new;renew=now+6900
     else:d.atomic(ROOT/'PAUSE_RESOURCE.json',{'reason':'MEMORY_RECOVERY_V1 lease renewal readmission failed','epoch':time.time(),'inventory':inv});cooperative=True
    if now>=rt.deadline(run)-90 and not cooperative:d.atomic(ROOT/'PAUSE_RESOURCE.json',{'reason':'original global deadline cleanup','epoch':time.time()});cooperative=True
    if now>=rt.deadline(run)+5:
     if rt.same(owner,rt.proc(child.pid)):child.terminate()
     break
    time.sleep(2)
   rc=child.wait(timeout=60);d.atomic(attempt/'WAIT.json',{'actual_parent_wait':True,'returncode':rc,'child':owner,'epoch':time.time()},True)
   if quiescent(attempt):raise RuntimeError('OWN_GENERATION_STILL_LIVE_NO_RESOURCE_RELEASE_OR_SECOND_ATTEMPT')
   rt.release(ROOT,lease);lease=None;result=d.read(attempt/'RESULT.json') if (attempt/'RESULT.json').exists() else {'status':'MISSING_ATTEMPT_RESULT'};status=result['status'];action=launcher_action(rc,status);d.atomic(attempt/'EXIT_PROTOCOL.json',{'real_returncode':rc,'status':status,'action':action,'revision':'MEMORY_RECOVERY_V1'},True)
   if action!='RESUME_IF_ORIGINAL_BUDGET_AND_ATTEMPTS_ALLOW':
    if action=='EXIT_PROTOCOL_FAILURE':status=action
    break
   if time.monotonic()>=rt.deadline(run)-120:status='GLOBAL_WALL_LIMIT';break
   if (ROOT/'PAUSE_RESOURCE.json').exists():
    paused=REV/'PAUSE_RECORDS';paused.mkdir(exist_ok=True);shutil.move(str(ROOT/'PAUSE_RESOURCE.json'),str(paused/(str(time.time_ns())+'.json')))
  else:status='RETRY_LIMIT'
  final_revision(run,cfg,status,attempt)
 except BaseException:
  d.atomic(REV/f'LAUNCHER_FAILURE_{time.time_ns()}.json',{'traceback':traceback.format_exc(),'status':status,'epoch':time.time()},True);raise
 finally:
  if lease:
   if attempt is None or not quiescent(attempt):rt.release(ROOT,lease)
  lock.__exit__(None,None,None)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--start-or-resume',action='store_true');parser.add_argument('--launcher',action='store_true');parser.add_argument('--child');args=parser.parse_args();run,cfg=run_context()
 if args.prepare:return prepare()
 verify_revision()
 if args.child:
  attempt=ROOT/args.child
  try:raise SystemExit(ac.scientific(ROOT,cfg,run,attempt))
  except SystemExit:raise
  except BaseException as exc:d.atomic(attempt/'ENTRY_FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},True);raise
 if args.launcher:return launcher()
 if args.start_or_resume:
  if (ROOT/'FINAL_REVISIONS/MEMORY_RECOVERY_V1/REPORT.json').exists():print('REVISION_TERMINAL_NO_RESTART');return
  owners=owned_attempts_quiet()
  if any(r['live'] for r in owners):print(json.dumps({'status':'ATTACHED_EXISTING','attempts':owners}));return
  logpath=REV/('launcher_'+str(time.time_ns())+'.log')
  with logpath.open('xb') as log:p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--launcher'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps({'launcher_pid':p.pid,'run_id':RUN_ID,'revision':str(REV),'original_deadline_utc':run['deadline_utc'],'remaining_s':rt.deadline(run)-time.monotonic()}));return
 raise SystemExit('choose --prepare or --start-or-resume')

if __name__=='__main__':main()
