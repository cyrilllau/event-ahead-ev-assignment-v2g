"""Explicit user-authorized DET handover. Never touches a DRO process/model."""
from pathlib import Path
import os,sys,time,json,shutil,subprocess,signal,math,traceback,fcntl,contextlib
R=Path('/home/Shixin/ev-assignment/L08_DET_CAPACITY_CUTS_AND_DRO_RECOVERY_R38')
sys.path.insert(0,str(R/'tools'))
from r38_det import *
C=B/'L08_DET_R38_CAPACITY_CONTINUATION_V1'
H=R/'det_handover/PREPARED_R38_V1'
NEWLEASE=V/'leases/R38_DET_PRODUCTION.json'
OLDLEASE=V/'leases/R30_FORECAST44_DET.json'
OLDWAIT=D/'logs/PARENT_1793474083028700536/WAIT.json'
OLDPROCESS=OLDWAIT.with_name('PROCESS.json')
AUTHOR='User explicitly accepted switching DET to R38, retaining saved evidence and best plan, losing unexportable branch tree; DRO untouched.'

def finite(x):
 return x is not None and math.isfinite(float(x)) and abs(float(x))<1e50

def progress_snapshot(mm,w):
 # POLLING/MESSAGE and other non-progress stages must never query runtime.
 if w not in [GRB.Callback.MIP,GRB.Callback.PRESOLVE,GRB.Callback.SIMPLEX]:return None
 z=dict(epoch=time.time(),runtime=mm.cbGet(GRB.Callback.RUNTIME))
 if w==GRB.Callback.MIP:z.update(phase='MIP',native_LB=float(mm.cbGet(GRB.Callback.MIP_OBJBND)),native_UB=float(mm.cbGet(GRB.Callback.MIP_OBJBST)),nodes=float(mm.cbGet(GRB.Callback.MIP_NODCNT)))
 elif w==GRB.Callback.PRESOLVE:z.update(phase='PRESOLVE',rows_removed=int(mm.cbGet(GRB.Callback.PRE_ROWDEL)),cols_removed=int(mm.cbGet(GRB.Callback.PRE_COLDEL)))
 else:z.update(phase='SIMPLEX',iterations=mm.cbGet(GRB.Callback.SPX_ITRCNT),simplex_objective_not_global_LB=mm.cbGet(GRB.Callback.SPX_OBJVAL))
 for k,v in list(z.items()):
  if isinstance(v,float) and not finite(v):z[k]=None
 return z

def resource_check():
 # The legacy reader omits some newer leases. Conservatively charge every
 # live lease it did not enumerate at its full reservation, without RSS credit.
 res=rr.resources();seen={str(x['source']) for x in res['entries']};extra=[]
 for p in (V/'leases').glob('*.json'):
  z=read(p)
  if str(p) not in seen and p!=NEWLEASE and ctrl.same(z.get('owner')):
   extra.append(dict(path=str(p),reserved_bytes=z.get('reserved_bytes',0),owner=z['owner']))
 remaining=res['signed_headroom_bytes']-sum(x['reserved_bytes'] for x in extra)
 assert not res['unknown_scheduler_limits'] and remaining>=12*1024**3,'RESOURCE_WAIT_NO_OLD_SIGNAL'
 return dict(reader=res,extra_live_lease_reservations=extra,conservative_headroom_bytes=remaining,request_bytes=12*1024**3)

def expected():
 a=read(H/'READY.json');assert a['status']=='PREPARED_NOT_ACTIVATED'
 assert sha(H/'CAPACITY.mps.gz')==a['model_sha']
 assert sha(H/'START.npz')==a['start_sha']
 assert read(OUT/'CUT_PROOF.json')['passed'] and read(OUT/'CUT_PROOF.json')['count']==84
 assert read(OUT/'LP_CERTIFICATE_ACCEPTANCE.json')['model_sha']==a['model_sha']
 assert old.science()==a['science']
 return a

def snapshot_old(folder):
 folder.mkdir(parents=True,exist_ok=True)
 for p in (D/'04_MODEL_AND_NATIVE_SOLVE').iterdir():
  if p.is_file() and p.name not in ['DET.mps.gz','ZERO.mst']:
   shutil.copy2(p,folder/p.name)
 shutil.copytree(D/'04_MODEL_AND_NATIVE_SOLVE/candidates',folder/'candidates',dirs_exist_ok=True)
 dump(folder/'BOUND.json',old.base_bound())
 dump(folder/'MANIFEST.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})

def preflight():
 C.mkdir(exist_ok=True);a=expected();res=resource_check()
 m=gp.read(str(H/'CAPACITY.mps.gz'));x=np.load(H/'START.npz')['values'];old.new_start(m,x);m.update()
 assert m.getAttr('VarName')==read(D/'04_MODEL_AND_NATIVE_SOLVE/VARIABLE_NAMES.json')
 assert m.NumVars==99234 and m.NumConstrs==140611 and m.NumBinVars==25770
 assert np.max(np.abs(np.array(m.getAttr('Start'))-x))==0
 check=old.audit_matrix(old.matrix(m),x);m.dispose()
 assert max(a['global_LB'],1200)==a['global_LB']
 assert finite(1294) and not finite(float('inf')) and not finite(None)
 dump(C/'PREFLIGHT.json',dict(passed=True,epoch=time.time(),model_sha=a['model_sha'],complete_start=check,resource=res,science=a['science'],native_solves=0,det_only=True,callback_never_audits=True,default_no_fixed400=True))
 print('PREFLIGHT_PASS',flush=True)

def audit_candidates():
 for p in sorted((C/'candidates').glob('*.npz')):
  target=p.with_suffix('.AUDIT.json');failed=p.with_suffix('.REJECTED.json')
  if target.exists() or failed.exists():continue
  try:
   z=old.original_audit(np.load(p)['values'],target)
   z['raw_sha']=sha(p);dump(target,z)
  except Exception:
   dump(failed,dict(error=traceback.format_exc(),raw_sha=sha(p),epoch=time.time()))

def bounds_update():
 z=read(C/'INHERITED_BOUNDS.json');prior=read(C/'BOUNDS.json')
 ls=[z['LB']]+([prior['LB']] if finite(prior.get('LB')) else [])
 for n in ['PROGRESS.json','NATIVE_END.json']:
  p=read(C/n)
  if finite(p.get('native_LB')):ls.append(p['native_LB']-1e-6)
 choices=[(z['UB'],str(C/'START.npz'),str(C/'START_ORIGINAL_AUDIT.json'))]
 for p in (C/'candidates').glob('*.AUDIT.json'):
  a=read(p)
  if a.get('passed'):choices.append((a['UB'],str(p).replace('.AUDIT.json','.npz'),str(p)))
 ub,best,cert=min(choices);lb=max(ls);gap=ub-lb
 assert gap>=-1e-6,('SIGNED_GAP_INVALID',lb,ub)
 result=dict(epoch=time.time(),LB=lb,UB=ub,signed_gap=gap,certified=0<=gap<=1,best_point=best,original_certificate=cert,science=z['science'],branch_tree_restored=False,scope='FORECAST44_DET_ONLY',inherited_bound=z['LB'])
 dump(C/'BOUNDS.json',result);return result

def worker():
 assert ctrl.same(read(NEWLEASE).get('owner')),'PARENT_LEASE_REQUIRED'
 assert not ctrl.same(expected()['old_DET_owner'])
 assert not (C/'NATIVE_BEGIN.json').exists(),'NO_AUTOMATIC_COLD_RESTART'
 m=gp.read(str(C/'MODEL.mps.gz'));assert sha(C/'MODEL.mps.gz')==expected()['model_sha']
 old.params(m,4,0,None,C/'NATIVE.log');m.Params.MIPGap=0;m.Params.MIPGapAbs=1;m.Params.TimeLimit=GRB.INFINITY
 m.Params.OutputFlag=1;m.Params.LogFile=str(C/'NATIVE.log')
 x=np.load(C/'START.npz')['values'];old.new_start(m,x);m.update()
 assert np.max(np.abs(np.array(m.getAttr('Start'))-x))==0
 (C/'candidates').mkdir(exist_ok=True)
 dump(C/'NATIVE_BEGIN.json',dict(owner=ctrl.proc(os.getpid()),epoch=time.time(),model_sha=sha(C/'MODEL.mps.gz'),start_sha=sha(C/'START.npz'),Threads=4,MIPFocus=0,TimeLimit='Infinity',MIPGap=0,MIPGapAbs=1,SoftMemLimit_decimal_GB=m.Params.SoftMemLimit,branch_tree_restored=False,legacy_fixed400_dispatch=False))
 last=[0.];poll=[0.];best=[float('inf')];seq=[0];reason=[None]
 def cb(mm,w):
  try:
   now=time.monotonic()
   if w==GRB.Callback.MIPSOL:
    val=float(mm.cbGet(GRB.Callback.MIPSOL_OBJ))
    if val<best[0]-1e-8:
     name=f'{seq[0]:05d}';seq[0]+=1
     tmp=C/'candidates'/f'{name}.tmp.npz';p=C/'candidates'/f'{name}.npz'
     np.savez_compressed(tmp,values=np.array(mm.cbGetSolution(mm.getVars())));os.replace(tmp,p)
     dump(p.with_suffix('.json'),dict(native_objective=val,epoch=time.time(),sha256=sha(p)));best[0]=val
   if now-poll[0]>=2:
    poll[0]=now;co=read(C/'CONTROL.json');bd=read(C/'BOUNDS.json')
    if co.get('action')=='PAUSE' or bd.get('certified'):
     reason[0]='USER_COOPERATIVE_PAUSE' if co.get('action')=='PAUSE' else 'ORIGINAL_FORM_COMBINED_GAP_MET';mm.terminate()
   if now-last[0]<15:return
   z=progress_snapshot(mm,w)
   if z is None:return
   dump(C/'PROGRESS.json',z);last[0]=now
  except Exception:
   dump(C/'CALLBACK_FAILURE.json',dict(error=traceback.format_exc(),epoch=time.time()));reason[0]='CALLBACK_SAVE_FAILURE';mm.terminate()
 m.optimize(cb)
 end=dict(status=m.Status,runtime=m.Runtime,native_LB=float(m.ObjBound) if finite(m.ObjBound) else None,native_UB=float(m.ObjVal) if m.SolCount else None,nodes=m.NodeCount,work=m.Work,solver_max_memory_GB=m.MaxMemUsed,reason=reason[0],epoch=time.time())
 if m.SolCount:
  np.savez_compressed(C/'candidates/FINAL.npz',values=np.array(m.getAttr('X')))
 dump(C/'NATIVE_END.json',end);m.dispose()

def parent():
 C.mkdir(exist_ok=True)
 with ctrl.lock(C/'OWNER.lock',nonblock=True):
  assert not (C/'NATIVE_BEGIN.json').exists() and not (C/'SIGNAL_SENT.json').exists(),'EXISTING_HANDOVER_REQUIRES_REVIEW'
  a=expected();assert read(C/'PREFLIGHT.json')['passed']
  target=a['old_DET_owner'];assert ctrl.same(target),'OLD_OWNER_CHANGED_REVIEW_REQUIRED'
  dro_before=ctrl.proc(957854);assert dro_before and str(dro_before['start_ticks'])=='127332390'
  assert read(OLDPROCESS)['pid']==target['pid'] and read(OLDPROCESS)['parent']==target['ppid']
  assert not OLDWAIT.exists()
  dump(C/'USER_AUTHORIZATION.json',dict(text=AUTHOR,epoch=time.time(),old_owner=target,DRO_protected=dro_before))
  snapshot_old(C/'old_before_signal')
  with ctrl.lock(V/'DISPATCH.lock'):
   assert not ctrl.same(read(NEWLEASE).get('owner'));res=resource_check();dump(C/'ADMISSION.json',res)
   assert ctrl.same(target) and ctrl.same(dro_before)
   dump(NEWLEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=12*1024**3,threads=4,stage='DET_HANDOVER',root=str(C),method='R38_FORECAST44_DET'))
   dump(C/'SIGNAL_INTENT.json',dict(signal='SIGTERM',only_pid=target['pid'],identity=ctrl.proc(target['pid']),epoch=time.time(),authorization=AUTHOR))
   os.kill(target['pid'],signal.SIGTERM)
   dump(C/'SIGNAL_SENT.json',dict(signal='SIGTERM',only_pid=target['pid'],epoch=time.time()))
  try:
   until=time.monotonic()+60
   while not OLDWAIT.exists() and time.monotonic()<until:time.sleep(.5)
   assert OLDWAIT.exists(),'ORIGINAL_PARENT_WAIT_NOT_AVAILABLE'
   w=read(OLDWAIT);assert w['actual_parent_wait'] and w['returncode']==-signal.SIGTERM and not ctrl.same(target)
   dump(C/'ORIGINAL_PARENT_WAIT.json',dict(source=str(OLDWAIT),record=w,process=read(OLDPROCESS),record_sha=sha(OLDWAIT)))
   snapshot_old(C/'old_after_wait')
   with ctrl.lock(D/'RUN.lock',nonblock=True):
    with ctrl.lock(V/'DISPATCH.lock'):
     assert not ctrl.same(read(OLDLEASE).get('owner'))
     dump(C/'OLD_LEASE_ARCHIVE.json',read(OLDLEASE));dump(OLDLEASE,dict(owner=None,reserved_bytes=0,reason='USER_AUTHORIZED_R38_HANDOVER',parent_wait=str(OLDWAIT),superseded_by=str(NEWLEASE),epoch=time.time()))
     assert ctrl.same(dro_before)
    shutil.copy2(H/'CAPACITY.mps.gz',C/'MODEL.mps.gz')
    # Preserve all latest saved candidates, audit only the latest best again if its identity changed.
    rows=sorted((read(p)['objective'],p) for p in (C/'old_after_wait/candidates').glob('*.json'))
    _,p=rows[0];point=p.with_suffix('.npz');assert sha(point)==read(p)['raw_sha']
    audit=old.original_audit(np.load(point)['values'],C/'LATEST_OLD_ORIGINAL_AUDIT.json')
    if audit['UB']<a['safe_UB']:
     shutil.copy2(point,C/'START.npz');shutil.copy2(C/'LATEST_OLD_ORIGINAL_AUDIT.json',C/'START_ORIGINAL_AUDIT.json');ub=audit['UB']
    else:
     shutil.copy2(H/'START.npz',C/'START.npz');ca=old.original_audit(np.load(C/'START.npz')['values'],C/'START_ORIGINAL_AUDIT.json');ub=ca['UB']
    m=gp.read(str(C/'MODEL.mps.gz'));old.new_start(m,np.load(C/'START.npz')['values']);m.update();m.write(str(C/'COMPLETE_START.mst'));m.dispose()
    bound=old.base_bound();lb=max(a['global_LB'],bound['value'])
    dump(C/'INHERITED_BOUNDS.json',dict(LB=lb,UB=ub,science=a['science'],R38_bound_source=str(OUT/'GLOBAL_BOUNDS.json'),R30_bound=bound,old_parent_elapsed_s=w['elapsed_s'],old_runtime_history_preserved=True,R38_diagnostic_runtime_s=read(OUT/'MIP_END.json')['runtime'],branch_tree_restored=False))
    (C/'candidates').mkdir(exist_ok=True);bounds_update()
    with (C/'stdout.log').open('xb') as out,(C/'stderr.log').open('xb') as err:
     cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker'];child=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=C)
     dump(C/'PROCESS.json',dict(owner=ctrl.proc(child.pid),pid=child.pid,parent=os.getpid(),argv=cmd,epoch=time.time()))
     dump(C/'HANDOVER_COMPLETE.json',dict(epoch=time.time(),old_owner=target,old_parent_wait=w,new_parent=ctrl.proc(os.getpid()),new_child=ctrl.proc(child.pid),DRO_before=dro_before,DRO_after=ctrl.proc(957854),DRO_same=ctrl.same(dro_before),signals_sent=[dict(pid=target['pid'],signal='SIGTERM')],det_only=True,no_fixed400=True,branch_tree_restored=False))
     began=time.monotonic();monitor_errors=[]
     while child.poll() is None:
      try:audit_candidates();bounds_update()
      except Exception:
       monitor_errors.append(dict(epoch=time.time(),error=traceback.format_exc()));dump(C/'MONITOR_ERRORS.json',monitor_errors)
      time.sleep(10)
     rc=child.wait();dump(C/'WAIT.json',dict(returncode=rc,actual_parent_wait=True,child=child.pid,parent=os.getpid(),elapsed_s=time.monotonic()-began,epoch=time.time()))
    audit_candidates();z=bounds_update()
    dump(C/'RESULT.json',dict(status='CERTIFIED' if z['certified'] and rc==0 else 'STOPPED_NOT_CERTIFIED',bounds=z,native=read(C/'NATIVE_END.json'),WAIT=read(C/'WAIT.json'),DRO_signals_sent=[],fixed400_started=False,branch_tree_restored=False,automatic_restart=False))
  finally:
   # Never release a reservation belonging to a still-running orphan child.
   proc=read(C/'PROCESS.json').get('owner');live=ctrl.same(proc)
   with ctrl.lock(V/'DISPATCH.lock'):
    if live:dump(NEWLEASE,dict(owner=proc,reserved_bytes=12*1024**3,threads=4,stage='MONITOR_FAILED_CHILD_ALIVE',root=str(C)))
    elif (read(NEWLEASE).get('owner') or {}).get('pid')==os.getpid():dump(NEWLEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time()))

if __name__=='__main__':
 try:
  if sys.argv[1]=='preflight':preflight()
  elif sys.argv[1]=='worker':worker()
  elif sys.argv[1]=='activate-parent':parent()
  elif sys.argv[1]=='status':print(json.dumps({n:read(C/n) for n in ['HANDOVER_COMPLETE.json','NATIVE_BEGIN.json','PROGRESS.json','BOUNDS.json','WAIT.json','RESULT.json']},default=str))
  else:raise ValueError('Unknown explicit command')
 except Exception:
  C.mkdir(exist_ok=True);dump(C/f'FAILURE_{sys.argv[1]}_{time.time_ns()}.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
