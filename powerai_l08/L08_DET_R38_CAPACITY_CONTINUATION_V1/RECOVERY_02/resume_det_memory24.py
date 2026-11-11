"""One explicit DET-only cold resume after real MEM_LIMIT/WAIT0.
Reuse unchanged R38 mathematical/worker implementation. Only runtime memory
settings, additive evidence directory, parent monitoring and provenance change.
No optimizer call in preflight. No scientific all/fixed400 entry is reachable.
"""
from pathlib import Path
import os,sys,time,json,shutil,subprocess,traceback,inspect,math
sys.dont_write_bytecode=True
R=Path('/home/Shixin/ev-assignment/L08_DET_CAPACITY_CUTS_AND_DRO_RECOVERY_R38')
BASE=R.parent/'L08_DET_R38_CAPACITY_CONTINUATION_V1';OLD=BASE/'RECOVERY_01';C=BASE/'RECOVERY_02'
sys.path.insert(0,str(R/'tools'))
import run_det_r38_continuation as s
from run_det_r38_continuation import ctrl,read,dump,sha,expected,V,D,NEWLEASE,gp,GRB
assert Path(__file__).resolve()==C/'resume_det_memory24.py'
s.C=C
RESERVE=32*1024**3;SOFT_GB=24.;NODE_GB=.5
original_params=s.old.params
def memory_params(model,*args,**kwargs):
 original_params(model,*args,**kwargs)
 if len(args)>=4 and Path(args[3])==C/'NATIVE.log':
  model.Params.SoftMemLimit=SOFT_GB;model.Params.NodefileStart=NODE_GB;model.Params.NodefileDir=str(C/'nodefiles')
  dump(C/'RUNTIME_MEMORY_SETTINGS.json',{'SoftMemLimit_decimal_GB':model.Params.SoftMemLimit,'NodefileStart_decimal_GB':model.Params.NodefileStart,'NodefileDir':model.Params.NodefileDir,'threads':model.Params.Threads,'mathematical_model_unchanged':True,'TimeLimit_will_be_Infinity_in_original_worker':True,'epoch':time.time()})
s.old.params=memory_params
def source_hashes():
 modules=[s,s.old,sys.modules['r38_det'],sys.modules['r38_common'],ctrl,sys.modules['r23_common']]
 files={str(Path(m.__file__).resolve()) for m in modules};files.add(str(Path(__file__).resolve()))
 files.update(str(p) for folder in (D/'tools',s.old.R/'tools') for p in folder.glob('*.py'))
 return {p:sha(p) for p in sorted(files)}
def refs():
 return {str(p):sha(p) for p in [OLD/'MODEL.mps.gz',OLD/'START.npz',OLD/'START_ORIGINAL_AUDIT.json',OLD/'BOUNDS.json',OLD/'WAIT.json',OLD/'NATIVE_END.json',OLD/'RESULT.json',R/'det/CUT_PROOF.json',R/'det/LP_CERTIFICATE_ACCEPTANCE.json']}
def old_gate():
 w=read(OLD/'WAIT.json');native=read(OLD/'NATIVE_END.json');process=read(OLD/'PROCESS.json')
 assert w['actual_parent_wait'] is True and w['returncode']==0
 assert native['status']==17,'expected actual MEM_LIMIT'
 assert not ctrl.same(process['owner'])
 parent=read(BASE/'CURRENT_CONTINUATION.json')['parent'];assert not ctrl.same(parent)
 assert not ctrl.same(read(NEWLEASE).get('owner'))
 live=ctrl.allproc();workers=[]
 for p in live:
  if p['pid']==os.getpid():continue
  scripts=[a for a in p['argv'] if a.endswith('.py')]
  if scripts and any(name in scripts[0] for name in ['run_det_r38_continuation.py','resume_det_callback_fix.py','resume_det_memory24.py']):workers.append(p)
 assert not workers,('EXISTING_DET_OWNER',workers)
 return {'old_parent_wait':w,'old_native_end':native,'old_owner_dead':True,'old_parent_dead':True,'no_other_DET_entry_process':True}
def resource():
 rr=s.rr;prev=rr.LEASE;rr.LEASE=NEWLEASE
 try:z=rr.resources()
 finally:rr.LEASE=prev
 assert not z['unknown_scheduler_limits'],'UNKNOWN_RESOURCE_LIMIT'
 assert z['signed_headroom_bytes']>=RESERVE,'WAIT_RESOURCE_NO_OTHER_TASK_STOP'
 disk=shutil.disk_usage(C)
 assert disk.free>=50*1024**3,'WAIT_NODEFILE_DISK'
 return {'reader':z,'request_bytes':RESERVE,'SoftMemLimit_decimal_GB':SOFT_GB,'nodefile_threshold_decimal_GB':NODE_GB,'nodefile_free_bytes':disk.free,'admitted':True}
def preflight():
 assert not (C/'NATIVE_BEGIN.json').exists() and not (C/'PROCESS.json').exists()
 gate=old_gate();a=expected();assert sha(OLD/'MODEL.mps.gz')==a['model_sha']
 bounds=read(OLD/'BOUNDS.json');assert bounds['science']==a['science'];audit=read(OLD/'START_ORIGINAL_AUDIT.json');assert audit['passed'] is True and audit['UB']==bounds['UB']
 # Preserve all old files; each copied artifact is bound by its original digest.
 for name in ('MODEL.mps.gz','START.npz','START_ORIGINAL_AUDIT.json'):
  dest=C/name
  if not dest.exists():shutil.copy2(OLD/name,dest)
  assert sha(dest)==sha(OLD/name)
 (C/'nodefiles').mkdir(exist_ok=True);(C/'candidates').mkdir(exist_ok=True)
 # All validation uses the real installed matrix/auditor, with optimize forbidden.
 original_optimize=gp.Model.optimize
 def forbid(*args,**kwargs):raise AssertionError('PREFLIGHT_MUST_NOT_OPTIMIZE')
 gp.Model.optimize=forbid
 try:
  m=gp.read(str(C/'MODEL.mps.gz'));x=s.np.load(C/'START.npz')['values'];s.old.new_start(m,x);m.update()
  assert m.NumVars==99234 and m.NumConstrs==140611 and m.NumBinVars==25770
  assert m.getAttr('VarName')==read(D/'04_MODEL_AND_NATIVE_SOLVE/VARIABLE_NAMES.json')
  assert s.np.max(abs(s.np.array(m.getAttr('Start'))-x))==0
  original_params(m,4,0,None,C/'PREFLIGHT_MODEL.log');m.Params.SoftMemLimit=SOFT_GB;m.Params.NodefileStart=NODE_GB;m.Params.NodefileDir=str(C/'nodefiles');m.Params.MIPGap=0;m.Params.MIPGapAbs=1;m.Params.TimeLimit=GRB.INFINITY
  settings={name:m.getParamInfo(name)[2] for name in ['Threads','Method','NodeMethod','MIPFocus','SoftMemLimit','NodefileStart','NodefileDir','FeasibilityTol','OptimalityTol','IntFeasTol','MIPGap','MIPGapAbs']}
  assert settings['SoftMemLimit']==SOFT_GB and settings['NodefileStart']==NODE_GB and settings['Threads']==4
  assert s.np.isinf(m.Params.TimeLimit)
  matrix_check=s.old.audit_matrix(s.old.matrix(m),x);m.dispose()
  original_audit=s.old.original_audit(x,C/'START_REVALIDATED_ORIGINAL_AUDIT.json');assert original_audit['passed'] is True
  assert abs(original_audit['UB']-bounds['UB'])<1e-9
 finally:gp.Model.optimize=original_optimize
 inherited=read(OLD/'INHERITED_BOUNDS.json');inherited.update(LB=bounds['LB'],UB=bounds['UB'],science=bounds['science'],previous_bound_source={'path':str(OLD/'BOUNDS.json'),'sha256':sha(OLD/'BOUNDS.json')},previous_native_end={'path':str(OLD/'NATIVE_END.json'),'sha256':sha(OLD/'NATIVE_END.json')},previous_true_parent_wait=gate['old_parent_wait'],prior_native_runtime_s=gate['old_native_end']['runtime'],prior_elapsed_preserved=True,branch_tree_restored=False)
 dump(C/'INHERITED_BOUNDS.json',inherited);s.bounds_update()
 with ctrl.lock(V/'DISPATCH.lock'):admit=resource()
 record={'passed':True,'epoch':time.time(),'gate':gate,'source_hashes':source_hashes(),'input_hashes':refs(),'model_sha':sha(C/'MODEL.mps.gz'),'start_sha':sha(C/'START.npz'),'original_worker_SHA':sha(Path(s.__file__)),'same_unchanged_original_worker':True,'matrix':matrix_check,'original_audit':original_audit,'runtime_parameters':settings,'TimeLimit':'Infinity','resource_check':admit,'optimizer_calls':0,'branch_tree_restored':False,'scope':'FORECAST44_DET_ONLY','fixed400_dispatch':False,'other_tasks_restarted':False,'signals_sent':[]}
 dump(C/'PREFLIGHT.json',record);print('PREFLIGHT_PASS',flush=True)
def parent():
 p=read(C/'PREFLIGHT.json');assert p['passed'] and p['optimizer_calls']==0;assert source_hashes()==p['source_hashes'] and refs()==p['input_hashes']
 with ctrl.lock(BASE/'OWNER.lock',nonblock=True),ctrl.lock(D/'RUN.lock',nonblock=True):
  gate=old_gate();assert not (C/'PROCESS.json').exists() and not (C/'NATIVE_BEGIN.json').exists()
  with ctrl.lock(V/'DISPATCH.lock'):
   assert not ctrl.same(read(NEWLEASE).get('owner'));admit=resource();dump(C/'ADMISSION.json',admit)
   dump(NEWLEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=RESERVE,threads=4,stage='DET_MEMORY24_RESUME',root=str(C),method='R38_FORECAST44_DET'))
  child=None;errors=[]
  try:
   with (C/'stdout.log').open('xb') as out,(C/'stderr.log').open('xb') as err:
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker'];child=subprocess.Popen(command,stdout=out,stderr=err,cwd=C)
    owner=None
    for _ in range(40):
     owner=ctrl.proc(child.pid)
     if owner and owner['argv']==command:break
     time.sleep(.025)
    assert owner and owner['argv']==command,'CHILD_IDENTITY_NOT_OBSERVED'
    dump(C/'PROCESS.json',dict(owner=owner,pid=child.pid,parent=os.getpid(),argv=command,epoch=time.time()))
    dump(BASE/'CURRENT_CONTINUATION.json',dict(path=str(C),owner=owner,parent=ctrl.proc(os.getpid()),reason='Explicit user continuation after real MEM_LIMIT; same model/start/LB, memory24GB and nodefile0.5GB',signals_sent=[],branch_tree_restored=False,epoch=time.time()))
    start=time.monotonic()
    while child.poll() is None:
     try:s.audit_candidates();s.bounds_update()
     except Exception:errors.append({'epoch':time.time(),'error':traceback.format_exc()});dump(C/'MONITOR_ERRORS.json',errors)
     time.sleep(10)
    rc=child.wait();dump(C/'WAIT.json',dict(returncode=rc,actual_parent_wait=True,child=child.pid,parent=os.getpid(),elapsed_s=time.monotonic()-start,epoch=time.time()))
   s.audit_candidates();bounds=s.bounds_update();dump(C/'RESULT.json',dict(status='CERTIFIED' if bounds['certified'] and rc==0 else 'STOPPED_NOT_CERTIFIED',bounds=bounds,native=read(C/'NATIVE_END.json'),WAIT=read(C/'WAIT.json'),scope='FORECAST44_DET_ONLY',branch_tree_restored=False,prior_runtime_s=gate['old_native_end']['runtime'],fixed400_started=False,automatic_restart=False,signals_sent=[]))
  finally:
   owner=read(C/'PROCESS.json').get('owner')
   with ctrl.lock(V/'DISPATCH.lock'):
    if ctrl.same(owner):dump(NEWLEASE,dict(owner=owner,reserved_bytes=RESERVE,threads=4,stage='MONITOR_FAILED_CHILD_ALIVE',root=str(C)))
    elif (read(NEWLEASE).get('owner') or {}).get('pid')==os.getpid():dump(NEWLEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time()))
def worker():
 p=read(C/'PREFLIGHT.json');assert source_hashes()==p['source_hashes'] and refs()==p['input_hashes']
 # The original scientific worker is reused verbatim, including candidate capture,
 # finite callback guards, audit-based termination, and infinite native TimeLimit.
 s.worker()
if __name__=='__main__':
 try:
  if sys.argv[1]=='preflight':preflight()
  elif sys.argv[1]=='parent':parent()
  elif sys.argv[1]=='worker':worker()
  else:raise ValueError('explicit preflight/parent/worker only')
 except Exception:
  dump(C/('FAILURE_'+sys.argv[1]+'_'+str(time.time_ns())+'.json'),dict(error=traceback.format_exc(),epoch=time.time()));raise
