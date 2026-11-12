from pathlib import Path
import os,sys,json,time,hashlib,subprocess,traceback,contextlib,math
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];B=R.parents[1];DR=B/'L08_K4L4_DRO_READY_PARALLEL_R36';D38=B/'L08_DET_CAPACITY_CUTS_AND_DRO_RECOVERY_R38';DET=B/'L08_DET_R38_CAPACITY_CONTINUATION_V1/RECOVERY_01';V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2';LEASE=V/'leases/R41_NESTED_ROOT.json'
sys.path.insert(0,str(V/'tools'));import control_core as ctrl
sys.path.insert(0,str(B/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'));import r23_common as rr
sys.path.insert(0,str(DR/'tools'))
read=ctrl.read;sha=ctrl.sha

def dump(p,z):
 p=Path(p);assert R==p or R in p.parents
 ctrl.dump(p,z)
def record(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def snapshot():
 own=read(DET/'PROCESS.json')['owner'];sources={str(p):sha(p) for root in [DR/'tools',D38/'tools'] for p in root.glob('*.py')}
 return dict(epoch=time.time(),owner=own,actual=ctrl.proc(own['pid']),owner_live=ctrl.same(own),model=record(DET/'MODEL.mps.gz'),start=record(DET/'START.npz'),bounds=read(DET/'BOUNDS.json'),progress=read(DET/'PROGRESS.json'),formal_checkpoint=record(DR/'formal/CHECKPOINT.json'),sources=sources,signals_sent=[],production_writes=[])
def protection():
 before=read(R/'PROTECTION_BEFORE.json');after=snapshot();after.update(signals_sent=[],production_files_modified=(before['sources']!=after['sources'] or before['model']!=after['model'] or before['start']!=after['start'] or before['formal_checkpoint']!=after['formal_checkpoint']),DET_identity_unchanged=ctrl.same(before['owner']))
 dump(R/'PROTECTION_AFTER.json',after);assert not after['production_files_modified'] and after['DET_identity_unchanged'];return after

def stage(name):
 ex=R/'execution'/name;assert not ex.exists(),'NO_RETRY_EXISTING_STAGE';ex.mkdir()
 with ctrl.lock(R/'R41_OWNER.lock',nonblock=True):
  rr.LEASE=LEASE
  with ctrl.lock(V/'DISPATCH.lock'):
   assert not ctrl.same(read(LEASE).get('owner'))
   live=[]
   for p in ctrl.allproc():
    a=' '.join(p['argv'])
    if '/ev-assignment/' in a and ('.py' in a) and 'python' in a and p['pid']!=os.getpid() and str(R) not in a and 'resume_det_callback_fix.py' not in a:live.append(p)
   rs=rr.resources();ok=not live and not rs['unknown_scheduler_limits'] and rs['signed_headroom_bytes']>=32*1024**3
   dump(ex/'ADMISSION.json',dict(admitted=ok,other_heavy=live,request_bytes=32*1024**3,max_workers=8,threads_each=1,resources=rs))
   if not ok:dump(R/'DEFERRED_RESOURCE.json',dict(stage=name,details=read(ex/'ADMISSION.json')));return
   ctrl.dump(LEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=32*1024**3,threads=8,method='R41_ISOLATED_CONTINUOUS_ROOT',root=str(R)))
  if not (R/'PROTECTION_BEFORE.json').exists():dump(R/'PROTECTION_BEFORE.json',snapshot())
  try:
   cmd=[sys.executable,'-B',str(R/'tools/run_r41.py'),'worker',name];t=time.monotonic()
   with (ex/'stdout.log').open('xb') as so,(ex/'stderr.log').open('xb') as se:
    c=subprocess.Popen(cmd,stdout=so,stderr=se,cwd=R);dump(ex/'PROCESS.json',dict(child=ctrl.proc(c.pid),parent=ctrl.proc(os.getpid()),command=cmd));peak=0;last=0
    while c.poll() is None:
     ps=ctrl.descendants(ctrl.proc(c.pid),ctrl.allproc()) if ctrl.proc(c.pid) else [];rss=sum(p['rss'] for p in ps);peak=max(peak,rss)
     if name in ['root','root_boundary_resume'] and rss>32*1024**3 and not (R/'MEMORY_STOP.json').exists():dump(R/'MEMORY_STOP.json',dict(epoch=time.time(),rss=rss,limit=32*1024**3,own_native_cooperative_stop_only=True))
     if time.monotonic()-last>30:
      dump(R/'STATUS.json',dict(stage=name,elapsed=time.monotonic()-t,tree_rss=rss,peak_rss=peak,child=ctrl.proc(c.pid),epoch=time.time()));last=time.monotonic()
     time.sleep(2)
    rc=c.wait()
   dump(ex/'WAIT.json',dict(actual_parent_wait=True,returncode=rc,elapsed=time.monotonic()-t,peak_rss=peak,child_pid=c.pid,parent_pid=os.getpid(),epoch=time.time()))
   if rc:dump(R/'FAILURE.json',dict(stage=name,returncode=rc,automatic_retry=False));raise RuntimeError('STAGE_FAILED_NO_RETRY '+name)
  finally:
   protection()
   with ctrl.lock(V/'DISPATCH.lock'):
    ctrl.dump(LEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time()))
