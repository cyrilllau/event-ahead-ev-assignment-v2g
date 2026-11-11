import argparse,os,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--config',required=True);parser.add_argument('--role',choices=['start','launcher','child','preflight'],default='start');parser.add_argument('--attempt');args=parser.parse_args();os.environ['ARO_ROOT']=str(Path(args.root).resolve())
from control import *
import subprocess,traceback,difflib
def context():
 run=read(ROOT/'RUN.json');assert run['run_id']=='R44_AUTO_6e708d658e074a2988bea6e49319ba2d' and run['deadline_utc']==1793715496.6286445;assert sha(ROOT/'CONFIG.json')==run['config_sha'];assert sha(ROOT/'AUTHORIZATION.json')==run['authorization_sha'];run['source']=read(ROOT/'CONFIG.json')['source'];deadline=rt.deadline(run)-90;e.configure(ROOT,run,deadline);return run,deadline
def code_check():
 if args.role=='preflight' and not (REV/'CODE_SHA.json').exists():return {'module':str(Path(__file__).resolve()),'not_yet_frozen':True,'root_explicit':str(ROOT)}
 for name,value in read(REV/'CODE_SHA.json').items():assert sha(REV/name)==value,('REV_CODE_SHA',name)
 return {'module':str(Path(__file__).resolve()),'code_sha':ref(REV/'CODE_SHA.json'),'root_explicit':str(ROOT)}
def preflight():
 from repair_tests import control_tests
 if not (REV/'tests/CONTROL_RESULT.json').exists():control_tests()
 if not (ROOT/'SOURCE_LOCK.json').exists():source_gates()
 verify_source_lock()
 from aro_tests import synthetic,real_small
 if not (ROOT/'TESTS/SYNTHETIC/RESULT.json').exists():synthetic()
 build_oracles([1,385])
 if not (ROOT/'TESTS/REAL_SMALL/RESULT.json').exists():real_small(rt.deadline(read(ROOT/'RUN.json'))-90)
 write(REV/'PREFLIGHT_RESULT.json',{'passed':True,'control':ref(REV/'tests/CONTROL_RESULT.json'),'synthetic':ref(ROOT/'TESTS/SYNTHETIC/RESULT.json'),'real_small':ref(ROOT/'TESTS/REAL_SMALL/RESULT.json'),'frozen_source':ref(ROOT/'SOURCE_LOCK.json')},True)
def child(attempt):
 run,deadline=context();modules=code_check();write(attempt/'MODULE_IDENTITY.json',modules,True);state=read_state();master=None;status='RUNNING';cleanup=[]
 def publish(phase,**kw):write(ROOT/'PROGRESS.json',dict(phase=phase,outer=state['outer'],epoch=time.time(),remaining_global_s=max(0,deadline-time.monotonic()),best_LB=state['best_LB'],UB_audited=state['best_UB_audited'],**kw))
 try:
  if state['phase']=='PREFLIGHT':
   publish('REPAIR_PREFLIGHT');preflight();publish('BUILD_FULL400_ORACLES');build_oracles();state=commit_state(dict(state,phase='ZERO_SEED'))
  science=verify_source_lock();A,a,names=first();master,xvar,eta=make_master(A,a,names);assert master.NumVars==60781 and master.NumBinVars==25770;assert np.array_equal(master.getAttr('Obj'),np.r_[a['cost'],1]);assert master.ObjCon==0;assert len(orig.routes())==20100 and len(set(q['c'] for q in orig.routes()))==8
  for r in state['cuts']:
   z=np.load(rt.verify_ref(r['cut']));add_support(master,xvar,eta,float(z['alpha']),z['beta'],r['name'])
  master.update();write(attempt/'FORMAL_MASTER_AUDIT.json',{'columns':master.NumVars,'binary':master.NumBinVars,'routes':20100,'stations':8,'eta_count':1,'objective':'frozen_cost@x+eta','DRO_risk_variables':0,'tree_restored':False,'test_fixings':0},True)
  if state['phase']=='ZERO_SEED':
   x,pid=canonical(np.zeros(60780),ROOT/'points/ZERO_REPAIR',deadline);state=commit_state(dict(state,phase='EVALUATE',point_id=pid,point=ref(ROOT/'points/ZERO_REPAIR/POINT.npz'),eta_at_candidate=0))
  while time.monotonic()<deadline:
   checked();phase=state['phase'];outer=state['outer'];folder=ROOT/f'outer_{outer:05d}';folder.mkdir(exist_ok=True)
   action=check_external_gap(state)
   if action=='BOUND_INCONSISTENCY_REVIEW':raise ValueError('UNEXPLAINED_NEGATIVE_EXTERNAL_GAP')
   if action=='FINAL_CERTIFY' or phase=='FINAL_CERTIFY':
    state=commit_state(dict(state,phase='FINAL_CERTIFY'));publish('FINAL_CERTIFY');cert=strict_certify(state,deadline);write(folder/'STRICT_RESULT.json',cert,True)
    if not cert['passed']:status='NUMERICAL_TARGET_REACHED_CERTIFICATION_PENDING';break
    state=commit_state(dict(state,best_UB_certified=cert['UB_certified']))
    if cert['UB_certified']-state['best_LB']<0:raise ValueError('STRICT_NEGATIVE_EXTERNAL_GAP')
    if cert['UB_certified']-state['best_LB']<=1:status='ARO400_SOLVER_GAP_TARGET_REACHED';break
    state=commit_state(dict(state,phase='SOLVE'));continue
   if phase=='EVALUATE':
    x=np.load(rt.verify_ref(state['point']))['accepted'];records=evaluate(x,state['point_id'],attempt,science,run,deadline,publish);_,Fup=e.b.enclosure(sp.csr_matrix(a['cost'][None,:]),x);F=float(np.nextafter(float(Fup[0]),np.inf));ub=full_bound(F,records,list(range(400)),science,state['point_id']);new=copy.deepcopy(state)
    if new['best_UB_audited'] is None or ub<new['best_UB_audited']:new.update(best_UB_audited=ub,best_plan=state['point'],best_plan_id=state['point_id'])
    new['phase']='EVALUATE';state=commit_state(new);gap_action=check_external_gap(state)
    if gap_action=='BOUND_INCONSISTENCY_REVIEW':raise ValueError('UNEXPLAINED_NEGATIVE_GAP_AFTER_FULL400')
    if gap_action=='FINAL_CERTIFY':state=commit_state(dict(state,phase='FINAL_CERTIFY'));continue
    added=separate(master,xvar,eta,records,x,state['eta_at_candidate'],state['cuts']);new=copy.deepcopy(state);new['cuts']+=added;model_id=d.digest([r['fingerprint'] for r in new['cuts']]);new['processed'][state['point_id']]={'model_id':model_id,'eta':state['eta_at_candidate'],'q_interval_max':max(r['qU_audited']-r['qL'] for r in records)};new['phase']='SOLVE';write(folder/'CUT_TRANSACTION.json',{'added':added,'model_id':model_id},True);new['history'].append({'outer':outer,'LB':state['best_LB'],'UB':state['best_UB_audited'],'new_cuts':len(added),'point':state['point_id']});state=commit_state(new)
    pending=ROOT/'WORKING_PENDING'/model_id;pending.mkdir(parents=True,exist_ok=True)
    if not (pending/'MODEL.mps.gz').exists():master.write(str(pending/'MODEL.mps.gz'))
    write(pending/'IDENTITY.json',{'model_id':model_id,'rows':master.NumConstrs,'columns':master.NumVars,'binary':master.NumBinVars,'pending_unsolved':True,'Pi_CBais_not_fabricated':True},True)
    if state.get('native_optimal') and not added and state.get('native_model_id')==model_id and state['processed'][state['point_id']]['q_interval_max']<=.001:status='NO_PROGRESS_DIAGNOSTIC';break
    continue
   if phase!='SOLVE':raise ValueError('UNKNOWN_PHASE '+phase)
   checked();parameters(master,folder/f'MIP_{time.time_ns()}.log',deadline-time.monotonic());publish('SOLVE_INTEGER_MASTER',cuts=len(state['cuts']));model_id=d.digest([r['fingerprint'] for r in state['cuts']]);call=native(master,'ARO_INTEGER_MASTER');native_status=master.Status;end={'call':call,'status':native_status,'model_id':model_id,'SolCount':master.SolCount,'rows':master.NumConstrs,'columns':master.NumVars,'binary':master.NumBinVars,'runtime':master.Runtime,'Work':master.Work,'MemUsed_GB':master.MemUsed,'MaxMemUsed_GB':master.MaxMemUsed};bound=None
   try:bound=float(master.ObjBoundC)
   except gp.GurobiError:pass
   end['ObjBoundC']=bound;write(folder/f'MIP_END_{call}.json',end,True)
   if native_status==gp.GRB.MEM_LIMIT:raise ResourceRecoveryNeeded({'kind':'MASTER_MEM_LIMIT','native_status':17})
   if native_status in (gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED,gp.GRB.NUMERIC):raise ValueError('MASTER_NATIVE_CORRECTNESS_'+str(native_status))
   new=copy.deepcopy(state);new['best_LB']=bound_update(-np.inf if state['best_LB'] is None else state['best_LB'],bound);new['best_LB']=None if not np.isfinite(new['best_LB']) else new['best_LB'];new['native_optimal']=native_status==2;new['native_model_id']=model_id;state=commit_state(new);gap_action=check_external_gap(state)
   if gap_action=='FINAL_CERTIFY':state=commit_state(dict(state,phase='FINAL_CERTIFY'));continue
   if gap_action=='BOUND_INCONSISTENCY_REVIEW':raise ValueError('NEGATIVE_GAP_AFTER_MASTER_BOUND')
   if master.SolCount:
    pointdir=folder/f'POINT_{call}';x,pid=canonical(np.asarray(xvar.X),pointdir,deadline);current_eta=float(eta.X);processed=state['processed'].get(pid);cached=(ROOT/'evaluations'/pid/'COMPLETE.json').exists();already=bool(processed and processed['model_id']==model_id and processed['eta']==current_eta);action=check_external_gap(state,has_incumbent=True,cached_evaluation=cached,separated=already,native_optimal=native_status==2)
    if action=='DIAGNOSE_UNCLOSED_GAP':status='EVIDENCE_REFINEMENT_REQUIRED' if processed['q_interval_max']>.001 else 'NO_PROGRESS_DIAGNOSTIC';break
    if action in ('REPLAY_CACHED_EVALUATION','EVALUATE400'):
     new=candidate_transition(state,point_id=pid,point_ref=ref(pointdir/'POINT.npz'),eta=current_eta);new['master_source']=ref(folder/f'MIP_END_{call}.json');new['master_model_id']=model_id;state=commit_state(new)
   write(ROOT/'LAST_SOLVED_REPAIR.json',{'native':ref(folder/f'MIP_END_{call}.json'),'raw':ref(ROOT/'CALLS'/call/'RAW_STATE.npz'),'model_id':model_id,'no_MIP_Pi_or_basis_claimed':True})
   if state['outer'] and state['outer']%5==0 and not (ROOT/f'REPAIR_SNAPSHOT_{state["outer"]}.mps.gz').exists():master.write(str(ROOT/f'REPAIR_SNAPSHOT_{state["outer"]}.mps.gz'))
  if status=='RUNNING':status='GLOBAL_WALL_LIMIT'
 except (ResourceRecoveryNeeded,ResourceRetryExhausted,SchedulerStopped) as exc:status='RESOURCE_RECOVERY_NEEDED' if isinstance(exc,ResourceRecoveryNeeded) else exc.event['kind'];write(attempt/'RESOURCE_EVENT.json',exc.event,True)
 except BaseException as exc:status='IMPLEMENTATION_OR_CORRECTNESS_FAILED';write(attempt/'FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'phase':state['phase']},True)
 finally:
  if master is not None:
   try:master.write(str(attempt/'EXIT_WORKING.mps.gz'))
   except BaseException as exc:cleanup.append({'phase':'master.write','error':str(exc)})
   try:master.dispose()
   except BaseException as exc:cleanup.append({'phase':'master.dispose','error':str(exc)})
  write(attempt/'CLEANUP.json',{'errors':cleanup,'original_status_retained':status},True);write(attempt/'RESULT.json',{'status':status,'state':state,'original_deadline_utc':run['deadline_utc'],'wall_since_original_RUN':time.time()-run['start_utc'],'cleanup_errors':cleanup,'production_signals_sent':[]},True);publish(status)
 return exit_code(status)
def launcher():
 run,deadline=context();lease=None;result=None
 with rt.cc.lock(ROOT/'OWNER.lock',nonblock=True):
  try:
   for index in range(1,5):
    attempt=ROOT/f'ATTEMPT_{index:03d}'
    if (attempt/'WAIT.json').exists():continue
    if attempt.exists():raise ValueError('UNWAITED_NONEMPTY_ATTEMPT')
    start=time.monotonic()
    while lease is None:
     checked();lease,inv=rt.acquire(ROOT,run,read(ROOT/'PROTECTION_BEFORE.json'))
     if lease:break
     if time.monotonic()-start>=900:write(REV/'RESOURCE_NOT_AVAILABLE.json',inv);return
     time.sleep(5)
    attempt.mkdir();cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(ROOT),'--config',args.config,'--role','child','--attempt',str(attempt)]
    with (attempt/'STDOUT.log').open('xb') as so,(attempt/'STDERR.log').open('xb') as se:p=subprocess.Popen(cmd,stdout=so,stderr=se)
    owner=rt.wait_process(p.pid);write(attempt/'PROCESS.json',{'child':owner,'parent':rt.proc(os.getpid()),'modules':code_check()},True);renew=time.monotonic()+6900
    while p.poll() is None:
     if time.monotonic()>=renew:
      new,inv=rt.acquire(ROOT,run,read(ROOT/'PROTECTION_BEFORE.json'),renew=lease)
      if new:lease=new;renew=time.monotonic()+6900
      else:write(ROOT/'PAUSE_RESOURCE.json',{'reason':'renewal admission unavailable'})
     if time.monotonic()>rt.deadline(run)+30:
      if rt.same(owner,rt.proc(p.pid)):p.terminate()
      break
     time.sleep(2)
    rc=p.wait();write(attempt/'WAIT.json',{'actual_parent_wait':True,'returncode':rc,'child':owner,'epoch':time.time()},True);rt.release(ROOT,lease);lease=None;result=read(attempt/'RESULT.json');write(REV/'STATUS.json',{'attempt':index,'status':result['status'],'returncode':rc,'epoch':time.time()})
    if not(rc==75 and result['status']=='RESOURCE_RECOVERY_NEEDED' and time.monotonic()<deadline-120):break
   final=ROOT/'FINAL_REVISIONS/REPAIR_01_SOURCE_AND_CONTROL'/attempt.name;final.mkdir(parents=True,exist_ok=False);write(final/'REPORT.json',result,True);write(ROOT/'LATEST_FINAL_REPAIR.json',ref(final/'REPORT.json'));write(final/'SEALED_MANIFEST.json',{'files':[ref(p) for p in REV.rglob('*') if p.is_file() and not p.name.endswith('.log')],'real_WAIT':ref(attempt/'WAIT.json'),'lease_released':True},True)
  finally:
   if lease:rt.release(ROOT,lease)
def main():
 run,deadline=context();code_check()
 if args.role=='child':return child(Path(args.attempt))
 if args.role=='launcher':return launcher()
 if args.role=='preflight':return preflight()
 rt.verify_ref(read(REV/'RECOVERY_RESOLUTION.json')['old_FAILURE']);rt.verify_ref(read(REV/'RECOVERY_RESOLUTION.json')['old_FINAL'])
 if time.monotonic()>=deadline:write(REV/'EXPIRED.json',{'scientific_dispatch':False});return
 if (REV/'LAUNCHER.json').exists() and rt.same(read(REV/'LAUNCHER.json')['owner'],rt.proc(read(REV/'LAUNCHER.json')['owner']['pid'])):print('ATTACHED_EXISTING');return
 with (REV/'launcher.log').open('xb') as log:p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(ROOT),'--config',args.config,'--role','launcher'],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
 write(REV/'LAUNCHER.json',{'owner':rt.wait_process(p.pid),'modules':code_check(),'remaining':deadline-time.monotonic()},True);print(json.dumps(read(REV/'LAUNCHER.json')))
if __name__=='__main__':raise SystemExit(main())
