from pathlib import Path
import os,sys,json,time,hashlib,shutil,subprocess,contextlib,fcntl,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];B=R.parent;D=B/'L08_K4L4_FORECAST44_DOMAIN_AND_DET_R30';DR=B/'L08_K4L4_DRO_READY_PARALLEL_R36';V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
I=R/'instructions/L08_DET_CAPACITY_CUTS_AND_DRO_RECOVERY_R38';E=I/'evidence'
sys.path.insert(0,str(V/'tools'));import control_core as ctrl
sys.path.insert(0,str(B/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'));import r23_common as rr
LEASE=V/'leases/R38_FINITE_DIAGNOSTICS.json';rr.LEASE=LEASE
read,sha,dump=ctrl.read,ctrl.sha,ctrl.dump
def sources():return {str(p):sha(p) for q in [D,DR] for p in (q/'tools').rglob('*.py')}
def protected():
 z=read(R/'LOCK.json');assert sources()==z['sources'];cp=read(DR/'formal/CHECKPOINT.json')
 return dict(epoch=time.time(),source_unchanged=True,original_DET_same_owner=ctrl.same(z['DET_owner']),DRO_owner_alive=ctrl.same(cp.get('owner')),signals_sent=[],old_case_writes=0)
def inventory():
 cp=read(DR/'formal/CHECKPOINT.json');det=read(D/'04_MODEL_AND_NATIVE_SOLVE/NATIVE_BEGIN.json')['owner'];res=rr.resources()
 for r in read(I/'SOURCE_EVIDENCE.json')['evidence_files']:assert sha(I/r['path'])==r['sha256']
 assert sha(D/'04_MODEL_AND_NATIVE_SOLVE/DET.mps.gz')==read(E/'DET_INTEGER_CAPACITY_CUTS.json')['base_model_sha256']
 ps=ctrl.allproc();live=[p for p in ps if any('ev-assignment' in a for a in p['argv']) and any(a.endswith('.py') for a in p['argv'])]
 z=dict(epoch=time.time(),sources=sources(),DET_owner=det,DET_live=ctrl.proc(det['pid']),DRO=cp,DRO_live=ctrl.same(cp['owner']),resources=res,live_processes=live,scope=read(I/'R38_SCOPE_LOCKS.json'))
 if (R/'LOCK.json').exists():assert read(R/'LOCK.json')['sources']==z['sources']
 else:dump(R/'LOCK.json',z)
 dump(R/'INVENTORY.json',z)
 if not z['DRO_live']:
  waits=[dict(path=str(p),record=read(p)) for p in (DR/'operations').glob('PARENT_*/start-if-safe.WAIT.json') if read(p).get('child_pid',read(p).get('child'))==cp['owner']['pid']]
  assert len(waits)==1 and waits[0]['record']['actual_parent_wait'];dump(R/'DRO_PARENT_WAIT.json',waits[0])
 dump(R/'DET_ONLY_SCOPE.json',dict(new_DET_calls_fixed400=False,new_status_report_package_solve=False,legacy_all_has_fixed400='evaluate-fixed400' in (D/'tools/run_l08_k4l4_forecast44_domain_det_r30.py').read_text(),live_scope_guard='LIVE_SCOPE_GUARD_NOT_EFFECTIVE',no_live_hook=True))
def stage_done(stage):return (R/'execution'/stage/'WAIT.json').exists() and read(R/'execution'/stage/'WAIT.json')['returncode']==0
@contextlib.contextmanager
def admission(stage,gib):
 with ctrl.lock(V/'DISPATCH.lock'):
  assert not ctrl.same(read(LEASE).get('owner'))
  res=rr.resources();ok=not res['unknown_scheduler_limits'] and res['signed_headroom_bytes']>=gib*1024**3
  dump(R/'admission'/f'{stage}_{time.time_ns()}.json',dict(admitted=ok,GiB=gib,resources=res))
  if not ok:raise RuntimeError('RESOURCE_BLOCKED')
  ctrl.dump(LEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=gib*1024**3,threads=4 if stage.startswith('det') else 1,method='R38_FINITE_DIAGNOSTIC',stage=stage,root=str(R)))
 try:yield
 finally:
  with ctrl.lock(V/'DISPATCH.lock'):
   if (read(LEASE).get('owner') or {}).get('pid')==os.getpid():ctrl.dump(LEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time()))
