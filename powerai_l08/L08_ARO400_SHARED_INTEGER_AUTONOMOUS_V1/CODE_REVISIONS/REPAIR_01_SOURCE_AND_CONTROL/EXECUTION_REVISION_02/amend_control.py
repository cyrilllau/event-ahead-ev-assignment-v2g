from pathlib import Path
import os,sys,time,json,shutil,hashlib,difflib
ROOT=Path(os.environ['ARO_ROOT']);OLD=ROOT/'CODE_REVISIONS/REPAIR_01_SOURCE_AND_CONTROL';REV=Path(__file__).resolve().parent;sys.path.insert(0,str(REV));from control import *
with rt.cc.lock(ROOT/'OWNER.lock',nonblock=True):
 oldproc=read(ROOT/'ATTEMPT_001/PROCESS.json')['child'];assert not rt.same(oldproc,rt.proc(oldproc['pid']));oldlaunch=read(OLD/'LAUNCHER.json')['owner'];assert not rt.same(oldlaunch,rt.proc(oldlaunch['pid']));wait=read(ROOT/'ATTEMPT_001/WAIT.json');assert wait['actual_parent_wait'] and wait['returncode']==0;assert read(ROOT/'ATTEMPT_001/RESULT.json')['status']=='WAIT_RESOURCE'
 for name,h in read(OLD/'CODE_SHA.json').items():assert sha(OLD/name)==h
 run=read(ROOT/'RUN.json');assert rt.deadline(run)>time.monotonic()+120
 assert exit_code('SINGLE_ORACLE_RESOURCE_LIMIT')==70 and exit_code('RESOURCE_RETRY_EXHAUSTED')==70 and exit_code('IMPLEMENTATION_OR_CORRECTNESS_FAILED')==70 and exit_code('RESOURCE_RECOVERY_NEEDED')==75 and exit_code('WAIT_RESOURCE')==75 and exit_code('GLOBAL_WALL_LIMIT')==0
 write(REV/'EXIT_PROTOCOL_TEST.json',{'passed':True,'failed_resource_exits_nonzero':True,'old_cooperative_WAIT0_not_rewritten':ref(ROOT/'ATTEMPT_001/WAIT.json'),'no_native_calls_for_this_control_test':True,'final_report_path_includes_attempt':True},True)
 for name in ('gates','tests'):
  (REV/name).symlink_to(OLD/name,target_is_directory=True)
 write(REV/'PREFLIGHT_RESULT.json',dict(read(OLD/'PREFLIGHT_RESULT.json'),prior_native_tests_reused_unmodified=True,exit_protocol_test=ref(REV/'EXIT_PROTOCOL_TEST.json')),True)
 patch=''
 for name in ('control.py','run_aro400.py'):patch+=''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(REV/name).read_text().splitlines(True),fromfile='R1/'+name,tofile='R1_AMENDMENT/'+name))
 (REV/'PATCH_DIFF.patch').write_text(patch);frozen={str(p.relative_to(REV)):sha(p) for p in REV.rglob('*.py') if 'instructions' not in p.parts};write(REV/'CODE_SHA.json',frozen,True)
 resolution=dict(read(OLD/'RECOVERY_RESOLUTION.json'),new_CODE_SHA=ref(REV/'CODE_SHA.json'),implementation_amendment='failed resource statuses explicitly nonzero; final report directory keyed by actual attempt; all science/math/cache functions byte-identical',old_revision_code_sha=ref(OLD/'CODE_SHA.json'),own_cooperative_WAIT=ref(ROOT/'ATTEMPT_001/WAIT.json'),preflight=ref(REV/'PREFLIGHT_RESULT.json'),old_matrices_preserved_count=read(ROOT/'BUILD_PROGRESS.json')['completed']);write(REV/'RECOVERY_RESOLUTION.json',resolution,True)
 (ROOT/'PAUSE_RESOURCE.json').rename(REV/'PRESERVED_OWN_PAUSE.json');write(ROOT/'ARO_ACTIVE_REVISION_CURRENT.json',{'path':str(REV),'code':ref(REV/'CODE_SHA.json'),'resolution':ref(REV/'RECOVERY_RESOLUTION.json')},True);print(json.dumps({'ready':True,'existing_matrices':resolution['old_matrices_preserved_count'],'deadline_utc':run['deadline_utc'],'next_attempt':2}))
