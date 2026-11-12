from r36_model import *
from r36_queue import enqueue,pause_required
from r36_control import Control,batch_add,next_action,set_start
import itertools
GATES=['inventory','verify-payload','build-common-domain','replay400','recourse-tests','zero400','ambiguity-tests','control-tests','resource-check','semantic-tests','integration-tests','incident-semantics-tests','build-rmp','delivery-tests']
def stage_gates():
 for name in GATES:assert read(R/'gates'/f'{name}.json')['passed'],name
 assert read(R/'RESOURCE_PROFILE.json')['admitted'];assert read(R/'ROUTE_POLICY_ACCEPTED.json')['passed']
 return {name:sha(R/'gates'/f'{name}.json') for name in GATES}
def capture(m,fs,risk,get,embedded,queue,kind):
 # Only first-stage decision coordinates and the probability plane are captured.
 qvars=list(fs['v'].values());sv=list(fs['s'].values());fv=list(fs['f'].values());ev=list(fs['E'].values());vs=qvars+sv+fv+ev;vals=get(vs);mapping={v.index:float(x) for v,x in zip(vs,vals)}
 from compact_first_stage import decode
 pp=decode(fs,mapping);pvals=np.asarray(get(risk['plus'].tolist()));mvals=np.asarray(get(risk['minus'].tolist()));b0=float(get([risk['beta0']])[0]);planes=b0+np.asarray(moments()['phi'].T@(pvals-mvals)).ravel()
 return enqueue(queue,pp,planes,embedded,kind)
def run():
 gates=stage_gates();profile=read(R/'RESOURCE_PROFILE.json');root=R/'formal';root.mkdir(exist_ok=True);queue=root/'ASYNC';queue.mkdir(exist_ok=True)
 stop=queue/'STOP_WHEN_IDLE.json'
 if stop.exists():stop.rename(queue/('STOP_HISTORY_'+str(time.time_ns())+'.json'))
 old=read(root/'CHECKPOINT.json') if (root/'CHECKPOINT.json').exists() else None;pool=set(old['pool'] if old else read(R/'INITIAL_POOL.json')['pool']);zero=set(read(R/'INITIAL_POOL.json')['universal_zero']);assert not pool&zero
 attempt=root/('ATTEMPT_'+str(time.time_ns()));attempt.mkdir()
 if old:
  modelpath=Path(old['attempt'])/'LATEST_MODEL.mps.gz'
 else:
  modelpath=R/'prepared/RMP.mps.gz';assert sha(modelpath)==read(R/'gates/build-rmp.json')['sha256']
 m=gp.read(str(modelpath));fs,risk=bind(m);copies={};validate_model(m,fs,pool);parameters(m,attempt/'NATIVE.log',4,profile['soft_mem_GB'])
 a=moments();zr=read(R/'probability/ZERO/COMPLETE.json');best=old['best'] if old else dict(safe_UB=zr['risk_upper'],plan=[],plan_hash=plan_digest([]),risk=zr,F=0,certificate=str(R/'probability/ZERO/CERTIFICATE.npz'))
 if 'certificate' not in best:best['certificate']=str(Path(best['evaluation'])/'probability/CERTIFICATE.npz')
 cert=loadnp(best['certificate']);start=set_start(m,fs,best['plan'],risk,cert);LB=old.get('LB') if old else 0.;seen=set();accepted=[]
 start_identity=(best['plan_hash'],tuple(sorted(pool)))
 # Existing valid ADDs are consumed before the first new optimize, even without an incumbent.
 def poll():
  nonlocal best
  for p in (queue/'results').glob('*.json'):
   if p.name in seen:continue
   result=read(p);assert result['verified'] and result['science']==SCIENCE and result['risk_identity']==a['identity'] and result['full_support']==400
   req=read(queue/'pending'/p.name);assert result['plan_hash']==req['plan_hash'];accepted.append(result);seen.add(p.name)
   if result['safe_UB']<best['safe_UB']:
    best=dict(result,plan=req['plan'],certificate=str(Path(result['evaluation'])/'probability/CERTIFICATE.npz'));dump(queue/'required'/p.name,dict(kind='BEST',epoch=time.time()))
  return batch_add(accepted,pool|zero)
 def checkpoint(status,extra=None):
  dump(root/'CHECKPOINT.json',dict(science=SCIENCE,status=status,epoch=time.time(),owner=ctrl.proc(os.getpid()),pool=sorted(pool),universal_zero=sorted(zero),LB=LB,UB=best['safe_UB'],gap=best['safe_UB']-LB if LB is not None else None,best=best,attempt=str(attempt),gates=gates,**(extra or {})))
 def add(asks):
  for ask in asks:
   sid=ask['sid'];assert sid not in pool and sid not in zero
   tx=root/'transactions'/f'{sid:04d}';dump(tx/'REQUEST.json',ask)
   current=resource_reader.resources();growth=5*profile['slope_bytes_per_scene'];owner_reserved=profile['request_GiB']*1024**3
   if rss()+growth+4*1024**3>owner_reserved:checkpoint('PAUSED_RESOURCE_GROWTH',dict(needed_bytes=rss()+growth+4*1024**3,reserved_bytes=owner_reserved));return False
   copies[sid]=add_scene(m,fs,risk,sid);m.update();pool.add(sid);validate_model(m,fs,pool);m.write(str(attempt/'LATEST_MODEL.mps.gz'));checkpoint('POOL_COMMITTED');dump(tx/'ACK.json',dict(sid=sid,pool=sorted(pool),model_sha=sha(attempt/'LATEST_MODEL.mps.gz'),epoch=time.time()))
  return True
 if not add(poll()):m.dispose();return
 workerlog=attempt/'AUDIT_WORKER';workerlog.mkdir();so=(workerlog/'stdout').open('wb');se=(workerlog/'stderr').open('wb');worker=subprocess.Popen([sys.executable,'-B',str(R/'tools/r36_queue.py'),str(queue)],stdout=so,stderr=se);dump(workerlog/'PROCESS.json',dict(pid=worker.pid,parent=os.getpid(),cmd=worker.args,epoch=time.time()))
 dump(attempt/'START.json',dict(start=start,branch_tree_restored=False,pool=sorted(pool),owner=ctrl.proc(os.getpid()),scientific_identity=SCIENCE,old_tasks=protected(),gates=gates))
 # Optional DET candidate is audited asynchronously; it supplies no inherited bound.
 enqueue(queue,plans(),np.asarray(cert['planes']),pool|zero,'CANDIDATE')
 try:
  for epoch in itertools.count():
   # Reuse the latest accepted certificate; do not run recourse or risk LP here.
   refreshed_start=(best['plan_hash'],tuple(sorted(pool)))!=start_identity
   if refreshed_start:
    cert=loadnp(best['certificate']);set_start(m,fs,best['plan'],risk,cert);start_identity=(best['plan_hash'],tuple(sorted(pool)))
   ctl=Control(poll);it=attempt/f'EPOCH_{epoch:04d}';it.mkdir();lastwrite=0.;candidate_objective=np.inf;errors=[]
   dump(it/'WARM_START.json',dict(plan_hash=best['plan_hash'],safe_UB=best['safe_UB'],certificate=best['certificate'],certificate_sha=sha(best['certificate']),refreshed_this_epoch=refreshed_start,all_previous_slots_cleared=True,unknown_auxiliaries='GRB.UNDEFINED',epoch=time.time()))
   def callback(mm,where):
    nonlocal lastwrite,candidate_objective,LB
    try:
     ctl.callback(mm,where)
     if where==GRB.Callback.MIPSOL:
      obj=mm.cbGet(GRB.Callback.MIPSOL_OBJ)
      if obj<candidate_objective-1e-6:
       try:capture(mm,fs,risk,mm.cbGetSolution,pool|zero,queue,'CANDIDATE');candidate_objective=obj
       except Exception as ex:errors.append(dict(kind='LOCAL_CAPTURE_FAILED',error=repr(ex)))
     if ctl.bound is not None:LB=max(LB if LB is not None else -np.inf,ctl.bound)
     if time.time()-lastwrite>=30:checkpoint('RUNNING_NATIVE',dict(phase=where,observed_phases=sorted(ctl.observed),root_ready=ctl.root_ready,root_started_epoch=ctl.root_started,root_progress_policy='ROOT_PROGRESS_V1',pending_batch_size=len(ctl.asks)));lastwrite=time.time()
    except Exception as ex:errors.append(dict(kind='GLOBAL_CONTROL_ERROR',error=repr(ex)));ctl.reason='GLOBAL_CONTROL_ERROR';mm.terminate()
   begin=dict(epoch=time.time(),owner=ctrl.proc(os.getpid()),pool=sorted(pool),params=dict(Threads=m.Params.Threads,TimeLimit='Infinity',SoftMemLimit=m.Params.SoftMemLimit),scientific_identity=SCIENCE)
   dump(it/'BEGIN.json',begin);dump(root/'NATIVE_BEGIN.json',begin);m.optimize(callback)
   bound=float(m.ObjBound) if np.isfinite(m.ObjBound) and abs(m.ObjBound)<1e90 else None
   if bound is not None:LB=max(LB if LB is not None else -np.inf,bound-1e-6)
   dump(it/'END.json',dict(epoch=time.time(),status=m.Status,SolCount=m.SolCount,LB=bound,objective=m.ObjVal if m.SolCount else None,runtime=m.Runtime,work=m.Work,reason=ctl.reason,errors=errors,observed_phases=sorted(ctl.observed)));m.write(str(attempt/'LATEST_MODEL.mps.gz'))
   final=capture(m,fs,risk,lambda vs:m.getAttr('X',vs),pool|zero,queue,'FINAL_NATIVE') if m.SolCount else None;asks=poll();action=next_action(m.SolCount,asks,m.Status)
   if ctl.reason=='GLOBAL_CONTROL_ERROR':checkpoint('PREPARED_TEST_BLOCKED',dict(errors=errors));return
   if asks:
    if not add(asks):return
    continue
   if action!='AUDIT_FINAL':checkpoint(action,dict(required_final=final));return
   # Outside optimize: wait for required final, while preserving pending requests.
   while not (queue/'results'/f'{final}.json').exists():
    if worker.poll() is not None or (queue/'failed'/f'{final}.json').exists():checkpoint('PAUSED_PENDING_REQUIRED',dict(request=final));return
    time.sleep(1);asks=poll()
    if asks:break
   if asks:
    if not add(asks):return
    continue
   poll();answer=read(queue/'results'/f'{final}.json');gap=best['safe_UB']-LB;assert gap>=-1e-6
   if gap<=1 and answer['max_upper_residual']<=1e-4:
    checkpoint('DRO_CERTIFIED');dump(root/'CERTIFIED.json',dict(passed=True,LB=LB,UB=best['safe_UB'],gap=gap,full400_separation=answer['max_upper_residual'],best=best,epoch=time.time()));return
   if answer['max_upper_residual']>1e-4:checkpoint('UNRESOLVED_EMBEDDED_OR_INTERVAL_VIOLATION');return
   m.Params.MIPGapAbs=0;checkpoint('CONTINUE_SAME_RMP',dict(no_reset=True))
 finally:
  dump(stop,dict(epoch=time.time(),finish_active_preserve_all_pending=True));pause_required(queue);rc=worker.wait();dump(workerlog/'WAIT.json',dict(returncode=rc,actual_parent_wait=True,epoch=time.time()));so.close();se.close();m.dispose()
