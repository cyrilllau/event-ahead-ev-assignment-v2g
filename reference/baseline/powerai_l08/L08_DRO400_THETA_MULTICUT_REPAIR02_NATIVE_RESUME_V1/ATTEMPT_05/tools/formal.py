from aggregate_core import *
import traceback,gc,shutil
from csr_reader import read_csr

def historical_bound():
 cached=OUT/'HISTORICAL_LB_REPLAY.json'
 if cached.exists():
  old=read(cached);rt.verify_ref(old['certificate']);assert old['passed'];return old['LB']
 root=BASE/'L08_DRO_R44_POOLED_ROOT_CONTINUATION/AUTONOMOUS_CONVERGENCE';head=read(root/'LAST_COMMITTED.json');tx=read(rt.verify_ref(head['transaction']));payload=read(rt.verify_ref(tx['payload']));meta=payload['metadata'];proof=read(rt.verify_ref(meta['best_LB_source']));sol=dict(np.load(rt.verify_ref(proof['solution'])));wanted=proof['mapping']['active'];needed=set(wanted);blocks=[];rhs=[];sense=[];order=[]
 arr=dict(np.load(root/'BASE/ARRAYS.npz'));bank=read_csr(root/'BASE/ROW_BANK.npz');ids=['base/'+str(i) for i in range(bank.shape[0])];ii=[i for i,v in enumerate(ids) if v in needed];blocks.append(bank[ii]);rhs.extend(arr['rhs'][ii]);sense.extend(arr['sense'][ii]);order.extend(ids[i] for i in ii);del bank
 for rr in payload['row_blocks']:
  z=np.load(rt.verify_ref(rr));ids=z['ids'].tolist();ii=[i for i,v in enumerate(ids) if v in needed]
  if not ii:continue
  bank=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']));blocks.append(bank[ii]);rhs.extend(z['rhs'][ii]);sense.extend(z['sense'][ii]);order.extend(ids[i] for i in ii)
 A=sp.vstack(blocks,format='csr');lookup={v:i for i,v in enumerate(order)};perm=[lookup[v] for v in wanted];A=A[perm].tocsr();rh=np.asarray(rhs)[perm];se=np.asarray(sense)[perm]
 assert b.matrix_sha(A)==proof['identity']['matrix_sha'] and len(sol['Pi'])==len(wanted)
 assert np.array_equal(arr['cost'][:60780],first()[1]['cost']) and np.array_equal(arr['lo'][:60780],first()[1]['lo']) and np.array_equal(arr['hi'][:60780],first()[1]['hi'])
 lb,_=b.lagrangian(A,rh,se,arr['cost'],arr['lo'],arr['hi'],sol['Pi']);assert abs(lb-proof['LB'])<1e-8
 write(OUT/'HISTORICAL_LB_REPLAY.json',dict(passed=True,LB=lb,certificate=meta['best_LB_source'],upstream=proof['upstream_cut_proofs'],source_scope='93-scenario continuous relaxation, remaining costs lower-bounded by zero; lower bound only',integer_UB_inherited=False,objective_floor_added=False));return lb

def imported_cache(x,reps):
 key=hashlib.sha256(x.tobytes()).hexdigest();roots=[ARO/'evaluations',BASE/'L08_ARO_DRO_GAP1_ACCELERATION_V1/MEMORY_EXIT_REPAIR_V1/CONTINUE_01/ARO/evaluations'];hits=0
 for root in roots:
  p=root/key/'COMPLETE.json'
  if not p.exists():continue
  z=read(p)
  if z.get('scientific_id')!=SCIENCE or z.get('point_sha')!=key or not z.get('full400'):continue
  for rec in z['records']:
   sid=int(rec['sid'])
   if sid not in reps:continue
   task=e.task_id(SCIENCE,sid,x,'RECOURSE');target=OUT/'EVALUATION_CACHE'/task/'RESULT.json'
   if target.exists():continue
   oracle=Oracle(sid,OUT/'CACHE_REPLAY')
   try:
    source=Path(rec['primal']).parent/'RESULT.json';validated=oracle.validate_result(x,source);write(target,validated,True);hits+=1
   finally:oracle.close(discard=True)
  break
 return hits

def evaluate(x,workers,eq,prob,label):
 folder=OUT/'evaluations'/hashlib.sha256(x.tobytes()).hexdigest();complete=folder/'COMPLETE.json'
 if complete.exists():
  c=read(complete);verify_complete(c,x);return c,True
 folder.mkdir(parents=True,exist_ok=True);reps=[g[0] for g in eq['groups']];hits=imported_cache(x,reps);last=[0]
 def progress(n):
  if time.monotonic()-last[0]>3:write(OUT/'STATUS.json',dict(phase='FULL400_GLOBAL_SUPPORT',point_sha=hashlib.sha256(x.tobytes()).hexdigest(),completed=n,total=len(reps),zero=len(eq['zeros']),epoch=time.time()));last[0]=time.monotonic()
 raw=workers.evaluate(x,reps,SCIENCE,progress);by={r['sid']:r for r in raw};records=[]
 for sid in eq['zeros']:records.append(dict(sid=sid,universal_zero=True,qL=0.,qU_audited=0.,global_cut_valid=True,zero_proof=ref(OUT/'SCENARIO_EQUIVALENCE.json')))
 for group in eq['groups']:
  r=by[group[0]]
  for sid in group:records.append(dict(r,sid=sid,equivalence_source_sid=group[0]))
 records.sort(key=lambda r:r['sid']);qL=full_vector(records,'qL');qU=full_vector(records,'qU_audited');save(folder/'FULL400_Q.npz',x=x,qL=qL,qU=qU);write(folder/'GLOBAL_SUPPORT_RECORDS.json',records)
 ps,ri=prob.solve_full(qL,qU,np.load(OUT/'QBAR.npz')['values'],label+'_'+uuid.uuid4().hex);al,be,proof=aggregate(ps,records,first()[1]['lo'],first()[1]['hi'],ri['probability_check']['passed']);save(folder/'AGGREGATE.npz',alpha=al,beta=be,pstar=ps);write(folder/'AGGREGATE_PROOF.json',proof)
 F=float(b.enclosure(sp.csr_matrix(first()[1]['cost'][None,:]),x)[1][0]);UB=float(np.nextafter(F+ri['risk_U'],np.inf));act=float(np.longdouble(al)+np.asarray(be,np.longdouble)@np.asarray(x,np.longdouble))
 if act>ri['risk_U']+1e-5:raise ValueError('AGGREGATE_EXCEEDS_SAME_POINT_UPPER')
 c=dict(P_identity=CTX.config['P_identity'],proof_version=CTX.config['proof_version'],science=SCIENCE,point_sha=hashlib.sha256(x.tobytes()).hexdigest(),F=F,UB_audited=UB,strict_UB=None,risk=ri,activity=act,tightness=ri['risk_U']-act,aggregate=ref(folder/'AGGREGATE.npz'),records=ref(folder/'GLOBAL_SUPPORT_RECORDS.json'),Q=ref(folder/'FULL400_Q.npz'),full400=True,imported_global_support=hits,universal_zero=len(eq['zeros']),equivalence_reuse=400-len(reps)-len(eq['zeros']))
 write(complete,c,True);return c,False


from control_policy import (finite_bound,json_safe,bound_state,route_after_bound_update,
 candidate_route,pending_candidate,request_handover,cut_identity,commit_phase)

def verify_complete(c,x):
 assert c['science']==SCIENCE and c['P_identity']==CTX.config['P_identity'] and c['proof_version']==CTX.config['proof_version']
 assert c['point_sha']==hashlib.sha256(x.tobytes()).hexdigest() and c['full400']
 q=np.load(rt.verify_ref(c['Q']));assert np.array_equal(x,q['x'])
 records=read(rt.verify_ref(c['records']));assert np.array_equal(full_vector(records,'qU_audited'),q['qU'])
 for r in records:
  if r.get('universal_zero'):rt.verify_ref(r['zero_proof']);continue
  o=Oracle(r['sid'],OUT/'CACHE_REPLAY')
  try:o.validate_result(x,Path(r['primal']).parent/'RESULT.json')
  finally:o.close(discard=True)
 pc=np.load(rt.verify_ref(c['risk']['certificate']));assert b.probability_check(pc['pstar'])['passed']
 risk=b.risk_upper(q['qU'],float(pc['beta0']),pc['plus'],pc['minus']);assert risk['risk_upper']<=c['risk']['risk_U']+1e-8
 rt.verify_ref(c['aggregate'])

def attr(m,name):
 try:return getattr(m,name)
 except (gp.GurobiError,AttributeError):return None

def native_master(m,x,rho,folder,known,revision,on_bound):
 folder.mkdir(parents=True,exist_ok=False);remaining=CTX.deadline-time.monotonic()
 if remaining<=0:raise TimeoutError('GLOBAL_WALL_LIMIT')
 for k,v in dict(Threads=4,Method=1,NodeMethod=1,PreSparsify=1,MIPFocus=0,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,MIPGap=.001,MIPGapAbs=1,SoftMemLimit=8,TimeLimit=remaining,OutputFlag=1,LogFile=str(folder/'NATIVE.log')).items():setattr(m.Params,k,v)
 effective={k:getattr(m.Params,k) for k in ['Threads','Method','NodeMethod','PreSparsify','MIPFocus','Seed','FeasibilityTol','OptimalityTol','IntFeasTol','MIPGap','MIPGapAbs','SoftMemLimit','TimeLimit','Cutoff','SolutionLimit']}
 assert abs(m.Params.Cutoff)>=1e90 and m.Params.SolutionLimit>1
 effective['Cutoff']=None;effective['Cutoff_classification']='UNLIMITED';write(folder/'PARAMETERS.json',json_safe(effective))
 root_end=None;last=0.;reason=None;queue={};cb_error=None
 cid=uuid.uuid4().hex;begin=dict(id=cid,phase='BEGIN',kind='AGGREGATED_FULL_INTEGER_MASTER',epoch=time.time(),remaining=remaining,owner=rt.proc(os.getpid()),math_revision=revision)
 write(folder/'BEGIN.json',begin,True);rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',begin)
 def callback(model,where):
  nonlocal root_end,last,reason,cb_error
  try:
   now=time.monotonic()
   if now>=CTX.deadline or (OUT/'PAUSE_RESOURCE.json').exists() or (OUT/'USER_STOP.json').exists():reason='GLOBAL_OR_RESOURCE_STOP';model.terminate();return
   if where==gp.GRB.Callback.MIPSOL:
    xx=np.asarray(model.cbGetSolution(x.tolist()),float);rr=float(model.cbGetSolution(rho));key=hashlib.sha256(xx.tobytes()).hexdigest()
    if pending_candidate(key,rr,known):
     prev=queue.get(key)
     if prev is None or rr<prev['rho']:queue[key]=dict(x=xx,rho=rr,raw_x_sha=key,native_objective=float(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)),math_revision=revision,source='MIPSOL')
   if where==gp.GRB.Callback.MIP:
    nodes=model.cbGet(gp.GRB.Callback.MIP_NODCNT)
    if nodes>0 and root_end is None:root_end=now
    if now-last>=5:
     raw_bound=finite_bound(model.cbGet(gp.GRB.Callback.MIP_OBJBND));guarded=None if raw_bound is None else raw_bound-1e-6
     on_bound(guarded,dict(source=str(folder/'NATIVE.log'),raw=raw_bound,guard=1e-6,level='solver-supported callback',math_revision=revision))
     write(OUT/'NATIVE_PROGRESS.json',dict(epoch=time.time(),root_completed=root_end is not None,nodes=nodes,bound=guarded,runtime=model.cbGet(gp.GRB.Callback.RUNTIME),pending_candidates=len(queue)));last=now
    if request_handover(root_completed=root_end is not None,seconds_after_root=0 if root_end is None else now-root_end,pending_evaluation_or_separation=bool(queue)):reason='ROOT_COMPLETED_PENDING_CANDIDATE_HANDOVER';model.terminate()
  except BaseException:
   cb_error=traceback.format_exc();reason='CALLBACK_ERROR';model.terminate()
 try:m.optimize(callback)
 finally:
  raw_bound=finite_bound(attr(m,'ObjBoundC'));bound=None if raw_bound is None else raw_bound-1e-6
  rec=json_safe(dict(id=cid,phase='END',kind=begin['kind'],epoch=time.time(),status=attr(m,'Status'),runtime=attr(m,'Runtime'),work=attr(m,'Work'),max_mem_GB=attr(m,'MaxMemUsed'),nodes=attr(m,'NodeCount'),raw_bound=raw_bound,bound_guard=1e-6,bound=bound,bound_kind='solver-supported' if bound is not None else 'UNKNOWN',root_completed=root_end is not None,reason=reason,SolCount=attr(m,'SolCount'),math_revision=revision,callback_error=cb_error))
  raw={}
  if (attr(m,'SolCount') or 0)>0:
   raw=dict(x=x.X.copy(),rho=np.array(rho.X),objective=np.array(m.ObjVal));key=hashlib.sha256(raw['x'].tobytes()).hexdigest()
   if key not in queue:queue[key]=dict(x=raw['x'],rho=float(raw['rho']),raw_x_sha=key,native_objective=float(raw['objective']),math_revision=revision,source='FINAL_NATIVE')
  candidates=[]
  for i,v in enumerate(queue.values()):
   v=dict(v);xx=v.pop('x');v['point']=save(folder/f'CANDIDATE_{i:04d}.npz',x=xx);candidates.append(v)
  save(folder/'RAW_NATIVE.npz',**raw);write(folder/'CANDIDATES.json',candidates,True);write(folder/'END.json',rec,True);rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',rec)
  write(OUT/'LAST_SOLVED.json',dict(math_revision=revision,rows=m.NumConstrs,columns=m.NumVars,binaries=m.NumBinVars,bound_record=ref(folder/'END.json'),raw=ref(folder/'RAW_NATIVE.npz'),candidates=ref(folder/'CANDIDATES.json'),LP_Pi_basis_available=False))
  on_bound(bound,dict(source=ref(folder/'END.json'),raw=raw_bound,guard=1e-6,level='solver-supported',math_revision=revision))
 if cb_error:raise RuntimeError('CALLBACK_ERROR '+cb_error)
 return candidates,rec

def final_certify(best,folder,remaining_cert):
 import strict_upper as su
 start=time.monotonic();deadline=min(CTX.deadline,start+remaining_cert);folder.mkdir(parents=True,exist_ok=False)
 z=np.load(rt.verify_ref(best['Q']));point=z['x'];A,a,_=first();binary=a['vtype']=='B'
 assert np.array_equal(point[binary],np.rint(point[binary]))
 fc=su.check(A,a['rhs'],a['sense'],a['lo'],a['hi'],point,deadline);write(folder/'FIRST_STAGE.json',fc)
 if not fc['passed']:return dict(status='CERTIFICATION_PENDING',first_stage=fc,elapsed=time.monotonic()-start)
 records=read(rt.verify_ref(best['records']));qu=np.zeros(400)
 for r in records:
  sid=r['sid']
  if r.get('universal_zero'):continue
  o=Oracle(sid,folder/'INPUT')
  try:
   raw=np.load(r['primal']);check=su.check(o.W,o.a['h'],o.a['sense'],o.a['lo'],o.a['hi'],raw['y'],deadline,T=o.T,first=point);write(folder/f'{sid:04d}.json',check)
   if not check['passed']:return dict(status='CERTIFICATION_PENDING',sid=sid,failed=check,elapsed=time.monotonic()-start)
   qu[sid]=su.cost_upper(o.a['c'],raw['y'])
  finally:o.close(discard=True)
 cert=np.load(rt.verify_ref(best['risk']['certificate']));risk=b.risk_upper(qu,float(cert['beta0']),cert['plus'],cert['minus']);ub=float(np.nextafter(su.cost_upper(a['cost'],point)+risk['risk_upper'],np.inf))
 save(folder/'STRICT_UPPER.npz',qU=qu,x=point,beta0=risk['beta0'],plus=cert['plus'],minus=cert['minus'])
 return dict(status='STRICT_PRIMAL_AND_PROBABILITY_UPPER',strict_UB=ub,point_sha=best['point_sha'],elapsed=time.monotonic()-start,certificate=ref(folder/'STRICT_UPPER.npz'))

def execute():
 start=time.monotonic();write(OUT/'FORMAL_EXECUTE_BEGIN.json',dict(epoch=time.time(),owner=rt.proc(os.getpid()),formal_execute_called=True),True)
 checkpoint=OUT/'CHECKPOINT.json';session=uuid.uuid4().hex;eq=read(OUT/'SCENARIO_EQUIVALENCE.json')
 if checkpoint.exists():
  state=read(checkpoint);assert state['execution_id']==CTX.run['execution_id'];cold=True
 else:
  try:historicalLB=historical_bound()
  except (FileNotFoundError,AssertionError,KeyError) as exc:
   historicalLB=None;write(OUT/'HISTORICAL_LB_NOT_INHERITED.json',dict(error=str(exc),traceback=traceback.format_exc()))
  state=dict(execution_id=CTX.run['execution_id'],phase='EVALUATE',next_global_outer=0,eval_version=0,x_eval=ref(OUT/'INITIAL_POINT.npz'),x_eval_key='values',x_master=None,x_best=None,cuts=[],best=None,LB_historical=historicalLB,LB_native=None,LB_evidence=None,strict_UB=None,strict_point=None,certified_points={},cert_seconds=0.,queue=[],native_calls=0,completed_evaluations=0,completed_cycles=0,last_native=None,aliases={},status='RUNNING');cold=False
 m=x=rho=prob=workers=None;status='RUNNING';error=None;known={};last_snapshot=time.monotonic()
 def publish(event):
  vals=[v for v in (state['LB_historical'],state['LB_native']) if v is not None];lb=max(vals) if vals else None
  bounds=bound_state(lb,None if state['best'] is None else state['best']['UB_audited'],state['strict_UB']);state['bounds']=bounds
  z=dict(bounds,epoch=time.time(),event=event,execution_id=state['execution_id'],phase=state['phase'],outer=state['next_global_outer'],LB_native=state['LB_native'],LB_historical=state['LB_historical'],LB_evidence=state['LB_evidence'],best_point=None if state['best'] is None else state['best']['point_sha'],strict_point=state['strict_point'])
  write(OUT/'BOUND_STATE.json',z);write(OUT/'GAP.json',z);write(OUT/'PROGRESS.json',z);write(OUT/'BEST.json',dict(bound_state=z,plan=state['best']))
  return bounds
 def commit(phase):
  state['phase']=phase;state['remaining_seconds']=max(0,CTX.deadline-time.monotonic());state['branch_tree_restored']=False if cold else None
  commit_phase(checkpoint,state,write,rt.verify_ref);rt.append_event(OUT/'PHASE_HISTORY.jsonl',dict(epoch=time.time(),phase=phase,outer=state['next_global_outer'],checkpoint_sha=sha(checkpoint)))
 def update_lb(value,evidence):
  if value is not None and (state['LB_native'] is None or value>state['LB_native']):state['LB_native']=value;state['LB_evidence']=evidence
  publish('NATIVE_BOUND_UPDATE')
 def target_check(event):
  bounds=publish(event)
  if state['best'] is None:return False
  key=state['best']['point_sha'];done=key in state['certified_points']
  route=route_after_bound_update(bounds['LB'],bounds['UB_audited'],bounds['strict_UB'],certification_done_for_audited_point=done)
  if route=='FINAL_CERTIFY':
   previous=state['phase'];commit('FINAL_CERTIFY');cert=final_certify(state['best'],OUT/'FINAL_CERTIFICATION'/key,1800-state['cert_seconds']);state['cert_seconds']+=cert['elapsed'];state['certified_points'][key]=cert
   if cert.get('strict_UB') is not None and (state['strict_UB'] is None or cert['strict_UB']<state['strict_UB']):state['strict_UB']=cert['strict_UB'];state['strict_point']=key
   commit(previous);bounds=publish('STRICT_CERTIFICATE_UPDATE');route=route_after_bound_update(bounds['LB'],bounds['UB_audited'],bounds['strict_UB'],certification_done_for_audited_point=True)
  if route=='TARGET_WITH_CERTIFIED_PRIMAL_UB_CHECK_LB_LEVEL':state['status']='EXTERNAL_TARGET_WITH_CERTIFIED_PRIMAL_UB_SOLVER_SUPPORTED_LB';return True
  if route=='NUMERICAL_TARGET_CERTIFICATION_PENDING':state['status']=route;return True
  return False
 try:
  publish('RESTORED_BOUNDS');commit(state['phase']);m,x,rho=master();m.write(str(OUT/'BASE_MASTER.mps.gz'))
  for r in state['cuts']:
   z=np.load(rt.verify_ref(r['aggregate']));assert cut_identity(z['alpha'],z['beta'])==r['id'];add_cut(m,x,rho,float(z['alpha']),z['beta'],'DRO_AGGREGATE/'+r['id'])
  m.update();assert m.NumVars==60781 and m.NumBinVars==25770 and len(CTX.config['support'])==400
  write(OUT/'FORMAL_DOMAIN_GATE.json',dict(passed=True,columns=m.NumVars,binary=m.NumBinVars,support=400,common_rows=90768,unrestricted_routes=20100),True)
  policy=read(ARO/'PATCH_POLICY.json');policy.update(frozen_pool=[g[0] for g in eq['groups']],run_id=CTX.run['run_id'],original_deadline_utc=CTX.run['deadline_epoch'],max_local_retries=2,max_controller_attempts_including_old=3,legacy_task_ids=[])
  write(OUT/'PATCH_POLICY.json',policy,True)
  workers=e.Workers(OUT,dict(CTX.run,source=str(R41),execution_root=str(OUT),source_root=str(CTX.source_root),config=CTX.config),CTX.deadline,OUT/'workers'/session,oracle_cls=Oracle,policy=policy)
  (OUT/'probability').mkdir(exist_ok=True);prob=Probability(OUT/'probability'/session);need_start=True
  while True:
   CTX.check()
   if target_check('LOOP_BOUND_CHECK'):status=state['status'];break
   phase=state['phase']
   if phase in ('EVALUATE','FINAL_CERTIFY'):
    point=np.load(rt.verify_ref(state['x_eval']))[state.get('x_eval_key','x')];key=hashlib.sha256(point.tobytes()).hexdigest()
    point=canonical(point,OUT/'points'/('eval_'+key));c,cached=evaluate(point,workers,eq,prob,'EVAL_'+str(state['eval_version']))
    state['eval_version']+=1;state['completed_evaluations']+=int(not cached);state['current']=c;known[key]={'activity':c['activity'],'tightness':c['tightness']}
    if state['best'] is None or c['UB_audited']<state['best']['UB_audited']:state['best']=c;state['x_best']=c['Q']
    commit('CUT_READY')
    if target_check('CACHE_HIT' if cached else 'FULL400_UB_UPDATE'):status=state['status'];break
   if state['phase']=='CUT_READY':
    c=state['current'];z=np.load(rt.verify_ref(c['aggregate']));key=cut_identity(z['alpha'],z['beta']);existing={r['id']:r for r in state['cuts']}
    if key not in existing:
     state['cuts'].append(dict(id=key,aggregate=c['aggregate'],evaluation=ref(Path(c['Q']['path']).parent/'COMPLETE.json')));commit('CUT_PENDING_INSTALL')
     add_cut(m,x,rho,float(z['alpha']),z['beta'],'DRO_AGGREGATE/'+key);m.update();need_start=True
    commit('MASTER_PENDING');write(OUT/'WORKING_PENDING.json',dict(math_revision=len(state['cuts']),cuts=state['cuts'],rows=m.NumConstrs,cols=m.NumVars,unsolved=True))
   if state['phase']=='CUT_PENDING_INSTALL':
    # Cold restore already replayed every committed cut.
    commit('MASTER_PENDING');need_start=True
   if state['phase']=='MASTER_PENDING':
    if need_start and state['best']:
     best=state['best'];point=np.load(rt.verify_ref(best['Q']))['x'];risk=best['risk']['risk_U']
     for r in state['cuts']:
      z=np.load(rt.verify_ref(r['aggregate']));activity=float(np.longdouble(z['alpha'])+np.asarray(z['beta'],np.longdouble)@np.asarray(point,np.longdouble))
      if activity>risk+1e-5:raise ValueError('SUPPORT_ABOVE_TRUSTED_SAME_POINT_RISK')
     m.NumStart=1;m.Params.StartNumber=0;m.setAttr('Start',m.getVars(),[gp.GRB.UNDEFINED]*m.NumVars);x.Start=point;rho.Start=risk;m.update();need_start=False
     known[best['point_sha']]={'activity':best['activity'],'tightness':best['tightness']}
    candidates,native=native_master(m,x,rho,OUT/'native'/f'{state["native_calls"]:06d}',known,len(state['cuts']),update_lb)
    state['native_calls']+=1;state['last_native']=native;state['queue']=candidates;state['next_global_outer']+=1;state['completed_cycles']+=int(bool(state.get('current')) and native['bound'] is not None)
    commit('MASTER_SOLVED_PENDING_EVALUATION');write(OUT/'WORKING_PENDING.json',dict(math_revision=len(state['cuts']),unsolved=False))
    if not (OUT/'FIRST_REAL_CYCLE_ACCEPTANCE.json').exists():
     c=state['current'];write(OUT/'FIRST_REAL_CYCLE_ACCEPTANCE.json',dict(status='COMPLETE' if native['bound'] is not None and c['full400'] else 'PARTIAL',integer_point=state['x_eval'],full400=c['full400'],initial_audited_UB=c['UB_audited'],not_new_historical_discovery=c['point_sha']=='951cc92fb17df719e6a1d224095b39d0e51a23d47a2b56289b2a84cbf1eb2f91',imported=c['imported_global_support'],zero=c['universal_zero'],aggregate=c['aggregate'],probability=c['risk']['certificate'],native=native,columns=m.NumVars,binary=m.NumBinVars,checkpoint=ref(checkpoint),checkpoint_readback=True,bounds=state['bounds']),True)
    if target_check('POST_NATIVE'):status=state['status'];break
    if native['status'] in (gp.GRB.MEM_LIMIT,gp.GRB.NUMERIC):status='PAUSED_RESOURCE' if native['status']==gp.GRB.MEM_LIMIT else 'NUMERIC';break
    if state['native_calls']%5==0 or time.monotonic()-last_snapshot>1800:
     (OUT/'snapshots').mkdir(exist_ok=True);m.write(str(OUT/'snapshots'/f'MASTER_{state["native_calls"]:06d}.mps.gz'));last_snapshot=time.monotonic()
   if state['phase']=='MASTER_SOLVED_PENDING_EVALUATION':
    selected=None;diagnoses=[]
    for candidate in state['queue']:
     raw=np.load(rt.verify_ref(candidate['point']))['x'];accepted=canonical(raw,OUT/'points'/('candidate_'+candidate['raw_x_sha']));key=hashlib.sha256(accepted.tobytes()).hexdigest();state['aliases'][candidate['raw_x_sha']]=key
     complete=OUT/'evaluations'/key/'COMPLETE.json'
     if not complete.exists():selected=(accepted,candidate);break
     c,_=evaluate(accepted,workers,eq,prob,'CACHE_RESEPARATE');known[key]={'activity':c['activity'],'tightness':c['tightness']}
     if state['best'] is None or c['UB_audited']<state['best']['UB_audited']:state['best']=c;state['x_best']=c['Q']
     if target_check('CACHE_RESEPARATION_BOUND_UPDATE'):status=state['status'];break
     route=candidate_route(complete_cached=True,certified_cut_violates=c['activity']-candidate['rho']>1e-6,oracle_tight=c['tightness']<=1e-4,master_proof_complete=state['last_native']['status']==gp.GRB.OPTIMAL)
     cut=np.load(rt.verify_ref(c['aggregate']));cutid=cut_identity(cut['alpha'],cut['beta']);in_model=cutid in {r['id'] for r in state['cuts']}
     if route=='SEPARATE_CACHED_SUPPORT' and not in_model:selected=(accepted,candidate);break
     diagnoses.append(dict(point=key,rho=candidate['rho'],activity=c['activity'],tightness=c['tightness'],route=route,cut_already_installed=in_model,math_revision=len(state['cuts'])))
    if status!='RUNNING':break
    if selected:
     accepted,candidate=selected;state['x_master']=candidate['point'];state['x_eval']=save(OUT/'points'/('next_'+hashlib.sha256(accepted.tobytes()).hexdigest()+'.npz'),x=accepted);state['x_eval_key']='x';commit('EVALUATE');continue
    write(OUT/'TIGHTNESS_DIAGNOSTIC.json',dict(diagnoses=diagnoses,native=state['last_native'],bounds=state['bounds']))
    if state['last_native']['status']==gp.GRB.OPTIMAL:
     status='ORACLE_TIGHTNESS_OR_MASTER_GAP_DIAGNOSTIC';break
    if state['last_native']['status'] in (gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED):status='NATIVE_INCONSISTENCY_REQUIRES_CERTIFICATE';break
    commit('MASTER_PENDING')
 except TimeoutError as exc:status='GLOBAL_WALL_LIMIT';error=str(exc)
 except BaseException as exc:
  status='BLOCKED_COMPUTATION';error=traceback.format_exc();write(OUT/'FAILURE.json',dict(type=type(exc).__name__,error=str(exc),traceback=error));raise
 finally:
  state['status']=status
  cleanup=[]
  for fn in ([workers.close] if workers else [])+([prob.model.dispose] if prob else []):
   try:fn()
   except BaseException:cleanup.append(traceback.format_exc())
  if m is not None:
   try:m.write(str(OUT/'FINAL_MASTER.mps.gz'));m.dispose()
   except BaseException:cleanup.append(traceback.format_exc())
  publish('FINAL_EXPORT');commit(state['phase']);write(OUT/'FINAL_RESULT.json',dict(status=status,bounds=state['bounds'],strict_UB=state['strict_UB'],strict_point=state['strict_point'],LB_native=state['LB_native'],LB_historical=state['LB_historical'],best=state['best'],complete_integer_outer=state['completed_cycles'],complete_full400_evaluations=state['completed_evaluations'],native_master_calls=state['native_calls'],formal_execute_called=True,formal_wall=time.monotonic()-start,total_wall=time.monotonic()-CTX.run['start_monotonic'],certification_seconds=state['cert_seconds'],error=error,cleanup_errors=cleanup,production_signals_sent=[],epoch=time.time()))
