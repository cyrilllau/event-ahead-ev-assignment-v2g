from pathlib import Path
import argparse,sys,os,time,json,uuid,hashlib,subprocess,traceback

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-root',required=True);p.add_argument('--source-root',required=True);a=p.parse_args()
 out=Path(a.execution_root).resolve();base=Path(a.source_root).resolve().parent
 sys.path.insert(0,str(base/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'))
 import r23_common as rr
 c=rr.ctrl;run=c.read(out/'RUN.json');lease=rr.V/'leases/DRO400_AGG_REPAIR01_FULL_CYCLE_V1.json';rr.LEASE=lease
 need=24*1024**3;admitted=False;child=None;status='PREPARING';lease_id=uuid.uuid4().hex
 def protected():
  rows=[]
  for f in (rr.V/'leases').glob('*.json'):
   if f==lease:continue
   z=c.read(f)
   if c.same(z.get('owner')):rows.append(dict(lease=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),owner=z['owner'],actual=c.proc(z['owner']['pid'])))
  return rows
 with c.lock(out/'OWNER.lock',nonblock=True):
  c.dump(out/'PARENT_PROCESS.json',c.proc(os.getpid()));c.dump(out/'PROTECTED_BEFORE.json',protected())
  try:
   wait_start=time.monotonic()
   while not admitted:
    with c.lock(rr.V/'DISPATCH.lock'):
     assert not c.same(c.read(lease).get('owner'))
     res=rr.resources();admitted=not res['unknown_scheduler_limits'] and res['available_for_new_bytes']>=need
     c.dump(out/'ADMISSION.json',dict(admitted=admitted,request_bytes=need,resources=res,lease_id=lease_id))
     if admitted:c.dump(lease,dict(owner=c.proc(os.getpid()),lease_id=lease_id,reserved_bytes=need,threads=4,root=str(out),run_id=run['run_id']))
    if admitted:break
    if time.monotonic()-wait_start>=900:raise RuntimeError('RESOURCE_WAIT_900_EXHAUSTED')
    time.sleep(10)
   while not (out/'SOURCE_FROZEN.json').exists():
    if time.monotonic()>=run['deadline_monotonic']-120:raise TimeoutError('GLOBAL_DEADLINE_BEFORE_SOURCE_FROZEN')
    c.dump(out/'PARENT_STATUS.json',dict(status='WAITING_CODE_FREEZE',epoch=time.time(),remaining=run['deadline_monotonic']-time.monotonic()))
    time.sleep(2)
   freeze=c.read(out/'SOURCE_FROZEN.json')
   for r in freeze['files']:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
   attempt=out/'ATTEMPTS/000';attempt.mkdir(parents=True,exist_ok=False)
   cmd=[sys.executable,'-B',str(out/'tools/run_dro400_aggregate_repair.py'),'--execution-root',str(out),'--source-root',a.source_root,'--config',str(out/'CONFIG.json'),'--start-or-resume']
   with (attempt/'stdout.log').open('xb') as so,(attempt/'stderr.log').open('xb') as se:
    child=subprocess.Popen(cmd,cwd=out,stdout=so,stderr=se,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    identity=c.proc(child.pid);c.dump(attempt/'PROCESS.json',dict(parent=c.proc(os.getpid()),child=identity,argv=cmd,code=freeze));start=time.monotonic()
    while child.poll() is None:
     # No signals to production. Own child cooperates via its immutable deadline.
     if time.monotonic()>=run['deadline_monotonic']-120:c.dump(out/'USER_STOP.json',dict(reason='GLOBAL_CLEANUP_RESERVE',run_id=run['run_id']))
     if time.monotonic()>=run['deadline_monotonic']+30 and c.same(identity):child.terminate()
     time.sleep(2)
    rc=child.wait()
   c.dump(attempt/'WAIT.json',dict(actual_parent_wait=True,returncode=rc,elapsed=time.monotonic()-start,pid=child.pid,argv=cmd))
   final=c.read(out/'FINAL_RESULT.json');status=final.get('status','ENTRY_FAILED' if rc else 'NO_SCIENTIFIC_FINAL')
   c.dump(out/'PARENT_RESULT.json',dict(status=status,returncode=rc,child_success_not_inferred_from_zero=True,epoch=time.time()))
  except BaseException as exc:
   status='INFRASTRUCTURE_FAILED';c.dump(out/'PARENT_FAILURE.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
   raise
  finally:
   if child is not None and child.poll() is None:
    c.dump(out/'USER_STOP.json',dict(reason='PARENT_CLEANUP',run_id=run['run_id']))
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:
     if c.same(identity):child.terminate()
     child.wait(timeout=15)
    c.dump(out/'CLEANUP_WAIT.json',dict(actual_parent_wait=True,returncode=child.returncode,pid=child.pid))
   if admitted:
    with c.lock(rr.V/'DISPATCH.lock'):
     z=c.read(lease)
     if z.get('lease_id')==lease_id:c.dump(lease,dict(owner=None,reserved_bytes=0,lease_id=lease_id,run_id=run['run_id'],released_epoch=time.time()))
   c.dump(out/'LEASE_RELEASED.json',dict(lease_id=lease_id,epoch=time.time(),production_signals_sent=[]))
   c.dump(out/'PROTECTED_AFTER.json',protected())
   c.dump(out/'PARENT_END.json',dict(status=status,wall=time.monotonic()-run['start_monotonic'],epoch=time.time(),production_signals_sent=[]))

if __name__=='__main__':main()
