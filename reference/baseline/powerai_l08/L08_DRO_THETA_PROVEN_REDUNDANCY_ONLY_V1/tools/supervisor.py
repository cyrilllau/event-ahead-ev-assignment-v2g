from pathlib import Path
import os,sys,json,time,subprocess,uuid,hashlib,traceback,fcntl
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'))
import r23_common as rr
c=rr.ctrl;rr.LEASE=rr.V/'leases/DRO_THETA_PROVEN_REDUNDANCY_ONLY_V1.json'
def write(n,v): c.dump(R/n,v)
def protect():
 out=[]
 for e in rr.resources()['entries']:
  o=e.get('owner')
  if not o or not c.same(o):continue
  files=[]
  for arg in o.get('argv',[]):
   p=Path(arg)
   if p.suffix=='.py' and p.is_file():files.append(dict(path=str(p),sha256=c.sha(p)))
  out.append(dict(owner=o,source=e['source'],scripts=files))
 return out
def finish_protection(before):
 return [dict(**e,alive_same_identity=c.same(e['owner']),scripts_unchanged=all(c.sha(x['path'])==x['sha256'] for x in e['scripts'])) for e in before]
def main():
 lock=(R/'OWNER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run=c.read(R/'RUN.json');assert run['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 lid=uuid.uuid4().hex;child=None;identity=None;admitted=False;before=protect();write('PROTECTED_BEFORE.json',before);rc=2
 try:
  t=time.monotonic()
  while True:
   with c.lock(rr.V/'DISPATCH.lock'):
    old=c.read(rr.LEASE)
    if old.get('reserved_bytes',0)>0:raise RuntimeError('EXISTING_OWN_RESERVATION')
    res=rr.resources();admitted=not res['unknown_scheduler_limits'] and res['available_for_new_bytes']>=12*1024**3
    write('ADMISSION.json',dict(lease_id=lid,admitted=admitted,requested_bytes=12*1024**3,cores=4,resources=res))
    if admitted:
     c.dump(rr.LEASE,dict(owner=c.proc(os.getpid()),lease_id=lid,reserved_bytes=12*1024**3,threads=4,root=str(R),method='PROVEN_REDUNDANCY_FINITE_TEST'));break
   if time.monotonic()-t>=300:raise RuntimeError('RESOURCE_NOT_ADMITTED')
   time.sleep(3)
  write('SUPERVISOR_PROCESS.json',c.proc(os.getpid()))
  while not (R/'READY.json').exists():
   if time.monotonic()>=run['deadline_monotonic']-180:raise RuntimeError('PREPARATION_BUDGET_EXHAUSTED')
   time.sleep(2)
  ready=c.read(R/'READY.json')
  for x in ready['code']:assert c.sha(x['path'])==x['sha256']
  cfg=c.read(R/'CONFIG.json');cmd=[sys.executable,'-B',str(R/'tools/run_proven_redundancy_only.py'),'--source-snapshot',cfg['snapshot'],'--source-root',cfg['source_root'],'--execution-root',str(R),'--config',str(R/'CONFIG.json'),'--execute']
  with (R/'STDOUT.log').open('xb') as so,(R/'STDERR.log').open('xb') as se:
   child=subprocess.Popen(cmd,stdout=so,stderr=se,cwd=R,start_new_session=True)
   identity=c.proc(child.pid);write('PROCESS.json',dict(parent=c.proc(os.getpid()),child=identity,argv=cmd));last=0
   while child.poll() is None:
    now=time.monotonic();stage=c.read(R/'ACTIVE_STAGE.json');end=min(run['deadline_monotonic']-180,stage.get('wall_deadline',run['deadline_monotonic']-180))
    if now-last>10:
     family=c.descendants(c.proc(os.getpid()),c.allproc());used=sum(p['pss'] or 0 for p in family)
     write('RESOURCE_PROGRESS.json',dict(epoch=time.time(),pss_bytes=used,reservation_bytes=12*1024**3,tree=family));last=now
     if used>12*1024**3:end=now
    if now>=end:
     write('SUPERVISOR_LIMIT.json',dict(epoch=time.time(),stage=stage,reason='OWN_WALL_OR_RESOURCE_LIMIT',child=identity))
     # Only this verified task session, never any production PID.
     if c.same(identity):os.killpg(child.pid,15)
     break
    time.sleep(1)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:
    if c.same(identity):os.killpg(child.pid,9)
    rc=child.wait()
  write('WAIT.json',dict(actual_parent_wait=True,returncode=rc,child=identity,epoch=time.time(),wall=time.monotonic()-run['start_monotonic']))
 except BaseException as exc:write('SUPERVISOR_ERROR.json',dict(error=str(exc),traceback=traceback.format_exc()))
 finally:
  if child is not None and child.poll() is None:
   if c.same(identity):os.killpg(child.pid,15)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:
    if c.same(identity):os.killpg(child.pid,9)
    rc=child.wait()
   write('WAIT.json',dict(actual_parent_wait=True,returncode=rc,child=identity,epoch=time.time()))
  if admitted:
   with c.lock(rr.V/'DISPATCH.lock'):
    v=c.read(rr.LEASE);assert v['lease_id']==lid
    c.dump(rr.LEASE,dict(owner=None,lease_id=lid,reserved_bytes=0,released=time.time()))
  write('LEASE_RELEASED.json',dict(lease_id=lid,admitted=admitted,production_signals_sent=[]))
  write('PROTECTED_AFTER.json',finish_protection(before));write('PARENT_END.json',dict(returncode=rc,epoch=time.time(),wall=time.monotonic()-run['start_monotonic'],production_signals_sent=[]))
  if not (R/'FINAL_RESULT.json').exists():write('FINAL_RESULT.json',dict(status='INFRASTRUCTURE_OR_BUDGET_STOP',remaining_stages='NOT_RUN',real_wait=rc))
 return rc
if __name__=='__main__':raise SystemExit(main())
