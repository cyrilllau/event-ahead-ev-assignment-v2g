from pathlib import Path
import sys,os,time,uuid,json,hashlib,subprocess,traceback
BASE=Path('/home/Shixin/ev-assignment');OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'))
import r23_common as rr
c=rr.ctrl;lease=rr.V/'leases/DRO400_THETA_MULTICUT_REPAIR02_NATIVE_RESUME_V1.json';rr.LEASE=lease
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protected():
 rows=[]
 for p in sorted((rr.V/'leases').glob('*.json')):
  if p==lease:continue
  z=c.read(p);o=z.get('owner')
  if c.same(o):rows.append(dict(lease=str(p),owner=o,scripts=[dict(path=s,sha256=sha(s)) for s in o['argv'] if s.endswith('.py') and Path(s).exists()]))
 return rows
def own_tree(owner):
 if not c.same(owner):return []
 parents={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   text=(p/'stat').read_text();tail=text[text.rfind(') ')+2:].split();parents[int(p.name)]=int(tail[1])
  except (OSError,ValueError,IndexError):pass
 ids={owner['pid']};old=0
 while len(ids)!=old:
  old=len(ids);ids.update(i for i,pp in parents.items() if pp in ids)
 return [q for q in (c.proc(i) for i in ids) if q]
def main():
 child=None;identity=None;admitted=False;lid=uuid.uuid4().hex;rc=2;before=[];t=time.monotonic();begin=time.time()
 with c.lock(OUT/'OWNER.lock',nonblock=True):
  cfg=c.read(OUT/'CONFIG.json');assert cfg['total_seconds']==14400+cfg['credited_downtime_seconds']
  testroot=Path(cfg['scheduling_test_root']);tests=c.read(testroot/'NATIVE_TESTS/RESULT.json');assert tests['passed'] and tests['native_integration'] and c.read(testroot/'TEST_WAIT.json')['actual_parent_wait'] and c.read(testroot/'TEST_WAIT.json')['returncode']==0
  assert c.read(testroot/'TEST_LEASE_RELEASED.json')['production_signals_sent']==[]
  for name in ('scheduling_fix.py','native_repair02.py'):assert sha(OUT/'tools'/name)==sha(testroot/'tools'/name)
  attempt_start=t
  assert cfg['recovery_mode']=='REVISION1_NATIVE_RESULT'
  previous=Path(cfg['memory_resume_source']);prevwait=c.read(previous/'WAIT.json');prevprocess=c.read(previous/'PROCESS.json');failure=c.read(previous/'FINAL_RESULT.json')
  assert prevwait['actual_parent_wait'] and prevwait['returncode'] in (0,2) and not c.same(prevprocess['parent']) and not c.same(prevprocess['child'])
  assert c.read(previous/'LEASE_RELEASED.json')['admitted'] and (failure.get('error')=='USER_REQUESTED_STOP' or 'USER_REQUESTED_STOP' in failure['status'] or failure['status']=='GLOBAL_BUDGET_PARTIAL')
  assert c.read(OUT/'RESUME_AUTHORIZATION.json')['authorized_memory_resume']
  previous_run=c.read(previous/'RUN.json');run=c.read(OUT/'RUN.json');assert run['run_id']==previous_run['run_id'] and run['start_monotonic']==previous_run['start_monotonic'] and run==previous_run and c.read(OUT/'ATTEMPT.json')['attempt_number']==5
  assert run['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and time.monotonic()<run['deadline_monotonic']-180
  t=run['start_monotonic'];c.dump(OUT/'BEGIN.json',dict(run,phase='AUTHORIZED_SCHEDULING_FIX_ATTEMPT_05',attempt_start_monotonic=attempt_start,attempt_start_epoch=begin))
  before=protected();c.dump(OUT/'PROTECTED_BEFORE.json',before)
  try:
   old=Path(cfg['recovery_source']);w=c.read(old/'WAIT.json');pr=c.read(old/'PROCESS.json')
   assert w['actual_parent_wait'] and w['returncode']==2 and not c.same(pr['parent']) and not c.same(pr['child']) and c.read(old/'LEASE_RELEASED.json')['admitted']
   c.dump(OUT/'RECOVERY_PROCESS_GATE.json',dict(passed=True,old_wait=w,old_parent_alive=False,old_child_alive=False,old_lease_released=True,new_RUN=True,old_failure_type_not_used=True,production_signals_sent=[]))
   for r in c.read(OUT/'CODE_SHA.json'):assert sha(r['path'])==r['sha256']
   while True:
    with c.lock(rr.V/'DISPATCH.lock'):
     assert not c.same(c.read(lease).get('owner'))
     res=rr.resources();admitted=not res['unknown_scheduler_limits'] and res['available_for_new_bytes']>=40*1024**3
     c.dump(OUT/'ADMISSION.json',dict(admitted=admitted,requested_bytes=40*1024**3,threads=4,resources=res,lease_id=lid))
     if admitted:
      c.dump(lease,dict(owner=c.proc(os.getpid()),lease_id=lid,reserved_bytes=40*1024**3,threads=4,root=str(OUT),method='THETA_REPAIR02_ISOLATED'));break
    if time.monotonic()-attempt_start>=900 or time.monotonic()>=run['deadline_monotonic']-180:
     c.dump(OUT/'FINAL_RESULT.json',dict(status='RESOURCE_NOT_ADMITTED',wait_seconds=time.monotonic()-t));return 2
    time.sleep(5)
   cmd=[sys.executable,'-B',str(OUT/'tools/run_theta_repair02.py'),'--execution-root',str(OUT),'--resume-source',cfg['recovery_source'],'--snapshot',cfg['snapshot'],'--native-result',cfg['native_result'],'--config',str(OUT/'CONFIG.json'),'--start-or-resume']
   with (OUT/'STDOUT.log').open('xb') as so,(OUT/'STDERR.log').open('xb') as se:
    child=subprocess.Popen(cmd,cwd=OUT,stdout=so,stderr=se,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'TEMP')))
    identity=c.proc(child.pid);
    for _ in range(20):
     current=c.proc(child.pid)
     if current and current.get('argv'):identity=current;break
     time.sleep(.05)
    c.dump(OUT/'PROCESS.json',dict(parent=c.proc(os.getpid()),child=identity,argv=cmd));last=0
    while child.poll() is None:
     now=time.monotonic()
     if now>=run['deadline_monotonic']-180:c.dump(OUT/'USER_STOP.json',dict(reason='GLOBAL_CLEANUP_RESERVE'))
     if now-last>=30:
      tree=own_tree(c.proc(os.getpid()));used=sum(p['pss'] if p.get('pss') is not None else p.get('rss',0) for p in tree)
      c.dump(OUT/'RESOURCE_PROGRESS.json',dict(epoch=time.time(),own_tree_pss_bytes=used,limit_bytes=40*1024**3,processes=tree,method='direct verified proc tree, not admission list which excludes own lease'))
      if used>40*1024**3:c.dump(OUT/'PAUSE_RESOURCE.json',dict(reason='OWN_PROCESS_TREE_RESERVATION_EXCEEDED',actual_bytes=used))
      last=now
     if now>=run['deadline_monotonic'] and c.same(identity):child.terminate()
     time.sleep(2)
    rc=child.wait()
   c.dump(OUT/'WAIT.json',dict(actual_parent_wait=True,returncode=rc,child=identity,wall=time.monotonic()-attempt_start,total_RUN_wall=time.monotonic()-t,epoch=time.time()))
  except BaseException as exc:
   c.dump(OUT/'SUPERVISOR_ERROR.json',dict(error=str(exc),traceback=traceback.format_exc()));rc=2
  finally:
   if child is not None and child.poll() is None:
    c.dump(OUT/'USER_STOP.json',dict(reason='OWN_PARENT_CLEANUP'))
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
     if c.same(identity):child.terminate()
     child.wait(timeout=30)
    c.dump(OUT/'CLEANUP_WAIT.json',dict(actual_parent_wait=True,returncode=child.returncode,child=identity))
   if admitted:
    with c.lock(rr.V/'DISPATCH.lock'):
     z=c.read(lease);assert z['lease_id']==lid;c.dump(lease,dict(owner=None,reserved_bytes=0,lease_id=lid,released_epoch=time.time()))
   c.dump(OUT/'LEASE_RELEASED.json',dict(lease_id=lid,admitted=admitted,production_signals_sent=[]))
   c.dump(OUT/'PROTECTED_AFTER.json',protected());checks=[]
   for b in before:checks.append(dict(lease=b['lease'],same_owner=c.read(Path(b['lease'])).get('owner')==b['owner'],source_unchanged=all(sha(s['path'])==s['sha256'] for s in b['scripts'])))
   c.dump(OUT/'PRODUCTION_PROTECTION.json',dict(checks=checks,production_signals_sent=[]))
   c.dump(OUT/'PARENT_END.json',dict(returncode=rc,epoch=time.time(),wall=time.monotonic()-t,production_signals_sent=[],real_child_wait=None if child is None else child.returncode))
 return rc
if __name__=='__main__':raise SystemExit(main())
