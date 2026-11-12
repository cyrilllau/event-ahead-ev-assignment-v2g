from pathlib import Path
import sys,time,shutil,json
ROOT=Path('/home/Shixin/ev-assignment/L08_DRO_R44_POOLED_ROOT_CONTINUATION/AUTONOMOUS_CONVERGENCE');OLD=ROOT/'CODE_REVISIONS/MEMORY_RECOVERY_V1';REV=OLD/'EXECUTION_REVISION_02';sys.path.insert(0,str(REV));sys.dont_write_bytecode=True
import durable as d
import runtime as rt
import engine as e
import autonomous_controller as ac
from revision_controller import run_context,owned_attempts_quiet
from lifecycle_tests import run_all
run,cfg=run_context()
with rt.cc.lock(ROOT/'RUN.lock',nonblock=True):
 if any(x['live'] for x in owned_attempts_quiet()):raise ValueError('OWN_CONTROLLER_STILL_LIVE')
 launch=d.read(ROOT/'LAUNCHER_OWNER.json')['process']
 if rt.same(launch,rt.proc(launch['pid'])):raise ValueError('OWN_LAUNCHER_STILL_LIVE')
 wait=d.read(ROOT/'ATTEMPT_001/WAIT.json')
 if not wait['actual_parent_wait'] or wait['returncode']!=75:raise ValueError('AMENDMENT_REQUIRES_TRUE_COOPERATIVE_WAIT75')
 for p in (ROOT/'ATTEMPT_001/workers').glob('*/READY.json'):
  owner=d.read(p)['process']
  if rt.same(owner,rt.proc(owner['pid'])):raise ValueError('OWN_OLD_WORKER_STILL_LIVE')
 for name,value in d.read(OLD/'CODE_SHA.json').items():
  if d.sha(OLD/name)!=value:raise ValueError('PREVIOUS_FROZEN_REVISION_CHANGED')
 old_config=d.read(ROOT/'REVISION.json');d.atomic(REV/'PREVIOUS_REVISION_CONFIG.json',old_config,True)
 for name in ('PARTIAL93_CACHE_AUDIT.json','PATCH_POLICY.json','PROTECTION_BEFORE.json','REQUEST.txt'):
  shutil.copyfile(OLD/name,REV/name)
 (REV/'ORIGINAL_STATE').symlink_to(OLD/'ORIGINAL_STATE',target_is_directory=True)
 store=d.Store(ROOT);state,meta=store.replay();_,_,proof=ac.solution_load(store,meta['last_solved']);head=d.read(ROOT/'LAST_COMMITTED.json');d.atomic(REV/'PRE_AMENDMENT_PINNED_HEAD.json',head,True)
 protection=ac.protect(ROOT);lease,admission=rt.acquire(ROOT,run,protection)
 if lease is None:raise RuntimeError('AMENDMENT_TEST_RESOURCE_NOT_AVAILABLE')
 try:test=run_all(REV,run,d.read(REV/'PATCH_POLICY.json'))
 finally:rt.release(ROOT,lease)
 if not test['passed']:raise ValueError('AMENDED_NATIVE_MEM_TEST_FAILED')
 frozen={p.name:d.sha(p) for p in REV.glob('*.py')};d.atomic(REV/'CODE_SHA.json',frozen,True)
 evidence={'scope':'same authorized memory lifecycle repair, append-only implementation amendment; no mathematical or budget changes','reason':'native test generation1 received unrelated tasks during corrective-task ready_at; exclusive-clean check correctly rejected it; original integration test failed to reject unexpected worker exit','previous_test_kept':rt.ref(OLD/'TEST_REPORT.json'),'previous_test_additional_ENTRY_ERROR':rt.ref(OLD/'TESTS/20261103T010419/NATIVE/CONTROLLER/workers/SLOT_0_GEN_1/ENTRY_ERROR.json'),'fixes':['reserve fresh slot for first corrective task during nonblocking ready_at','typed unknown entry failure terminates rather than whitelist-retry','native acceptance now requires exactly five worker WAIT records, exactly one recovery and no ENTRY_ERROR'],'old_frozen_sources_unchanged':True,'previous_controller_real_WAIT':rt.ref(ROOT/'ATTEMPT_001/WAIT.json'),'all_previous_workers_dead':True,'pre_amendment_phase':meta['phase'],'pre_amendment_outer':meta['outer'],'pre_amendment_LB':meta['best_LB'],'pre_amendment_replayed_LB':proof['LB'],'preserved_head':rt.ref(REV/'PRE_AMENDMENT_PINNED_HEAD.json'),'new_code_sha':rt.ref(REV/'CODE_SHA.json'),'new_real_native_test':rt.ref(REV/'TEST_REPORT.json'),'original_run_id':run['run_id'],'original_deadline_utc':run['deadline_utc'],'original_config_sha':d.sha(ROOT/'CONFIG.json'),'original_config_hash_unchanged':d.sha(ROOT/'CONFIG.json')==run['config_sha'],'production_signals_sent':[]}
 d.atomic(REV/'IMPLEMENTATION_AMENDMENT.json',evidence,True)
 updated=dict(old_config,path=str(REV),code_sha=rt.ref(REV/'CODE_SHA.json'),policy=rt.ref(REV/'PATCH_POLICY.json'),implementation_amendment=rt.ref(REV/'IMPLEMENTATION_AMENDMENT.json'))
 d.atomic(ROOT/'REVISION.json',updated)
 hold=ROOT/'ATTEMPT_002/MAINTENANCE_HOLD.json';archive=REV/'MAINTENANCE_ARCHIVE';archive.mkdir();hold.rename(archive/'MAINTENANCE_HOLD.json');(ROOT/'ATTEMPT_002').rmdir()
 if (ROOT/'PAUSE_RESOURCE.json').exists():raise ValueError('OLD_COOPERATIVE_PAUSE_NOT_ARCHIVED_BY_PARENT')
 if d.read(ROOT/'LAST_COMMITTED.json')!=head:raise ValueError('SCIENCE_HEAD_CHANGED_DURING_TEST')
 print(json.dumps({'status':'CORRECTED_REVISION_TESTED_READY','source':str(REV),'resume_outer':meta['outer'],'resume_phase':meta['phase'],'original_remaining_s':rt.deadline(run)-time.monotonic(),'new_test':test['native_integration']}))
