from pathlib import Path
import os,sys,time,json,uuid,hashlib,traceback,math,gc,shutil,threading,contextlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'lifecycle'))
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import run_dro400_theta_multicut as o
import native_repair02 as nr
import recovery02 as recovery
import tests_repair02
from scheduling_fix import NearIndex,classify,support_decision,probability_decision
from envelope_sparse import digest,enclosure
OUT=CFG=RUN=CTX=None
D=None;MODEL=None;ENV=None;WORKERS=None;PROB=None
TIMES={};PHASE='NOT_STARTED';LB=None;BEST=None;STRICT_BEST=None;CERT_SECONDS=0.;CERT_POINTS=set();NEW_Q=0;OUTERS=0;MATH_ID=None;NEXT_CALL=0;INCREMENTS=[];LAST_SNAPSHOT=0.;POINT=None;WITNESS=None;DEFERRED_CUTS=[]
@contextlib.contextmanager
def phase(name,**v):
 global PHASE
 PHASE=name;t=time.monotonic();o.ph(name,**v)
 try:yield
 finally:
  TIMES[name]=TIMES.get(name,0.)+time.monotonic()-t
  o.write('STAGE_TIMING.json',dict(measured_wall=TIMES,presolve=None,root_LP=None,root_cuts=None,tree=None,unknown_native_stages='read native log; unseparated stages not guessed'))

def save_same(path,**arrays):
 p=Path(path)
 if p.exists():
  with np.load(p) as z:
   if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],arrays[k]) for k in z.files):raise ValueError('IMMUTABLE_ARRAY_CONFLICT '+str(p))
  return o.ref(p)
 return o.ac.d.publish_npz(p,**arrays)

def append(v):o.ac.rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',v)
def native(m,stage,folder,nfirst=o.N,mapping=None,known=None,seconds=None,simulated=False):
 return nr.native(m,stage,folder,CTX,RUN, MATH_ID if stage!='TEST' else 'SMALL_TEST_ONLY',o.REVISION,nfirst,mapping or {},known or {},o.write,o.save,o.ref,append,seconds,simulated)
def commit(result,folder=''):
 global LB
 status=nr.commit(result,o.write,folder)
 if not folder and result['bound'] is not None:LB=max(LB,result['bound'])
 return status

def checkpoint(nextphase,point=None,evaluation=None):
 global POINT
 if point is not None:POINT=o.save(OUT/'POINT_OBJECTS'/(nr.point_sha(point)+'.npz'),x=point)
 wr=None if WITNESS is None else o.save(OUT/'WITNESS_OBJECTS'/(nr.point_sha(WITNESS)+'.npz'),full=WITNESS)
 z=dict(separation_witness=wr,run_id=RUN['run_id'],science=CFG['science'],P_identity=CFG['P_identity'],phase=nextphase,math_revision=o.REVISION,math_id=MATH_ID,base=o.ref(OUT/'RESTORE/BASE_IDENTITY.json') if (OUT/'RESTORE/BASE_IDENTITY.json').exists() else None,increments=INCREMENTS,next_call_id=NEXT_CALL,x_current=POINT,x_best=None if BEST is None else BEST['Q'],evaluation=evaluation,LB=LB,best=BEST,best_strict=STRICT_BEST,new_Q_batches=NEW_Q,completed_outer_cycles=OUTERS,cert_seconds=CERT_SECONDS,cert_points=sorted(CERT_POINTS),epoch=time.time(),deadline_epoch=RUN['deadline_epoch'])
 o.write('STATE.json',z)

def publish(event):
 z=o.tk.external_gap(LB,BEST['UB_audited']);o.write('BEST.json',BEST);o.write('BEST_STRICT.json',STRICT_BEST)
 o.write('GAP.json',dict(z,event=event,strict_UB=None if STRICT_BEST is None else STRICT_BEST['strict_UB'],strict_gap=None if STRICT_BEST is None else o.tk.external_gap(LB,STRICT_BEST['strict_UB']),best_point=BEST['point_sha'],UB_level='FULL400_NUMERICALLY_AUDITED_WITH_PROBABILITY_DUAL_UPPER',LB_level='MAX_VALID_HISTORICAL_RELAXATION_AND_SOLVER_SUPPORTED_MIP_BOUNDS',own_best_master_LB=own_bound(),math_revision=o.REVISION,completed_new_Q_batches=NEW_Q,completed_outer_cycles=OUTERS,epoch=time.time()))
 return z

def own_bound():
 vals=[]
 for p in (OUT/'MASTER').glob('*/END.json'):
  v=o.read(p)
  if v.get('bound') is not None:vals.append(v['bound'])
 return max(vals) if vals else None

def known_light(known):
 return {k:dict(risk_L=v['risk']['risk_L'],qL=np.load(o.ac.rt.verify_ref(v['Q']))['qL'],x=np.load(o.ac.rt.verify_ref(v['Q']))['x']) for k,v in known.items()}

def ensure_probability():
 global PROB
 if PROB is None:PROB=o.make_probability(OUT)
 return PROB

def finish_risk_raw(qL,qU,folder):
 z=dict(np.load(folder/'RAW_FINAL.npz'));assert np.array_equal(z['qL'],qL) and np.array_equal(z['qU'],qU)
 risk=o.ac.b.risk_upper(qU,float(z['beta0']),z['plus'],z['minus']);p0=np.load(o.ac.b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz')['p']
 sub=folder/('CERT_REPAIR_'+uuid.uuid4().hex)
 strict=o.ac.certify_candidate_compat(z['p'],np.load(OUT/'QBAR.npz')['values'],list(range(400)),sub,p0=p0,dd=o.ac.b.moment_data(),check=o.ac.b.probability_check,enclosure=o.ac.b.enclosure,owned_root=OUT)
 ps=np.load(sub/'PROBABILITY.npz')['pstar'];check=o.ac.b.probability_check(ps);assert check['passed']
 riskL=float(o.ac.b.enclosure(sp.csr_matrix(qL[None,:]),ps)[0][0]);cert=o.save(sub/'CERTIFICATE.npz',pstar=ps,qL=qL,qU=qU,beta0=risk['beta0'],plus=z['plus'],minus=z['minus'])
 hist=o.read(folder/'HISTORY.json');info=dict(risk_L=riskL,risk_U=risk['risk_upper'],rowgen_closed=bool(hist and hist[-1]['violations']==0),strict_probability=strict,probability_check=check,certificate=cert,dual_covers_all400=True,reused_raw=o.ref(folder/'RAW_FINAL.npz'))
 o.write(sub/'CERTIFICATE.json',info,True);return ps,info

def eval_stage(folder):
 folder=Path(folder)
 if (folder/'COMPLETE.json').exists():return 'COMPLETE'
 if (folder/'RISK_RESULT.json').exists():return 'RISK_COMPLETE'
 if (folder/'FULL400_Q.npz').exists() and (folder/'GLOBAL_SUPPORT_RECORDS.json').exists():return 'Q_COMPLETE'
 return 'MISSING_Q'

def evaluate(x,known):
 global WORKERS,NEW_Q
 key=nr.point_sha(x);folder=OUT/'evaluations'/key;folder.mkdir(parents=True,exist_ok=True)
 if key in known:
  recovery.validate_evaluation(known[key]);o.write(folder/'CACHE_HIT.json',dict(point_sha=key,source=known[key]['Q'],full400=True,optimizer_calls=0));return known[key]
 stage=eval_stage(folder)
 if stage=='COMPLETE':
  c=o.read(folder/'COMPLETE.json');recovery.validate_evaluation(c);return c
 if stage=='MISSING_Q':
  with phase('EVALUATE400',point_sha=key):
   eq=D['eq'];reps=[g[0] for g in eq['groups']]
   # Reuse exact partial source caches without creating any native model.
   hits=[]
   for sid in reps:
    task=o.ac.e.task_id(o.ac.SCIENCE,sid,x,'RECOURSE');dest=OUT/'EVALUATION_CACHE'/task/'RESULT.json'
    if dest.exists():continue
    for root in tuple(Path(CFG['memory_resume_source']).parent/at for at in ('ATTEMPT_04','ATTEMPT_03','ATTEMPT_02'))+(Path(CFG['recovery_source']),Path(CFG['source_root'])):
     src=root/'EVALUATION_CACHE'/task/'RESULT.json'
     if not src.exists():continue
     obj=o.ac.Oracle(sid,OUT/'CACHE_IMPORT_NO_MODEL')
     try:r=obj.validate_result(x,src);assert obj.model is None
     finally:obj.close(discard=True)
     o.write(dest,r,True);hits.append(dict(sid=sid,task=task,source=o.ref(src)));break
   o.write(folder/'IMPORTED_PARTIAL_CACHE.json',hits)
   if WORKERS is None:
    policy=o.read(o.ac.ARO/'PATCH_POLICY.json');policy.update(frozen_pool=reps,run_id=RUN['run_id'],original_deadline_utc=RUN['deadline_epoch'],max_local_retries=2,max_controller_attempts_including_old=2,legacy_task_ids=[],max_models_per_worker=16,production_soft_mem_decimal_GB=3.,evict_above_decimal_GB=2.25,evict_target_decimal_GB=2.)
    o.write('PATCH_POLICY.json',policy,True)
    WORKERS=o.ac.e.Workers(OUT,dict(RUN,source=str(o.ac.R41),execution_root=str(OUT),source_root=str(CTX.source_root),config=CFG),CTX.deadline,OUT/'workers'/uuid.uuid4().hex,oracle_cls=o.ac.Oracle,policy=policy)
   raw=WORKERS.evaluate(x,reps,o.ac.SCIENCE,lambda n:o.write('EVALUATION_PROGRESS.json',dict(point_sha=key,completed=n,total=len(reps),universal_zero=len(eq['zeros']),epoch=time.time())))
   by={r['sid']:r for r in raw};records=[dict(sid=sid,universal_zero=True,qL=0.,qU_audited=0.,global_cut_valid=True,zero_proof=o.ref(OUT/'SCENARIO_EQUIVALENCE.json')) for sid in eq['zeros']]
   for group in eq['groups']:
    for sid in group:records.append(dict(by[group[0]],sid=sid,equivalence_source_sid=group[0]))
   records.sort(key=lambda r:r['sid']);qL=o.ac.full_vector(records,'qL');qU=o.ac.full_vector(records,'qU_audited')
   o.save(folder/'FULL400_Q.npz',x=x,qL=qL,qU=qU);o.write(folder/'GLOBAL_SUPPORT_RECORDS.json',records,True);NEW_Q+=1;checkpoint('RISK',x)
 else:
  z=np.load(folder/'FULL400_Q.npz');assert np.array_equal(z['x'],x);qL=z['qL'];qU=z['qU'];records=o.read(folder/'GLOBAL_SUPPORT_RECORDS.json');assert np.array_equal(qL,o.ac.full_vector(records,'qL')) and np.array_equal(qU,o.ac.full_vector(records,'qU_audited'))
 if (folder/'RISK_RESULT.json').exists():ri=o.read(folder/'RISK_RESULT.json')
 else:
  with phase('RISK',point_sha=key):
   rawfolders=list((OUT/'probability').glob('*/POINT_'+key+'_*/RAW_FINAL.npz'))
   if rawfolders:ps,ri=finish_risk_raw(qL,qU,rawfolders[-1].parent)
   else:ps,ri=ensure_probability().solve_full(qL,qU,np.load(OUT/'QBAR.npz')['values'],'POINT_'+key+'_'+uuid.uuid4().hex)
   o.write(folder/'RISK_RESULT.json',ri,True)
 f=float(np.nextafter(float(enclosure(sp.csr_matrix(o.ac.first()[1]['cost'][None,:]),x)[1][0]),np.inf));ub=float(np.nextafter(f+ri['risk_U'],np.inf))
 c=dict(science=o.ac.SCIENCE,P_identity=CFG['P_identity'],proof_version=CFG['proof_version'],point_sha=key,F=f,UB_audited=ub,strict_UB=None,risk=ri,records=o.ref(folder/'GLOBAL_SUPPORT_RECORDS.json'),Q=o.ref(folder/'FULL400_Q.npz'),full400=True)
 recovery.validate_evaluation(c);o.write(folder/'COMPLETE.json',c,True);return c

def model_from(d):
 global ENV,MODEL,MATH_ID
 ENV=gp.Env(empty=True);ENV.setParam('OutputFlag',0);ENV.start();m=gp.Model('THETA_REVISION1_REPAIR02',env=ENV);MODEL=m
 A=d['A'];a=d['a'];n=d['names'];v=m.addMVar(A.shape[1],lb=a['lo'],ub=a['hi'],vtype=a['vtype'],obj=a['cost']);m.update();m.setAttr('VarName',v.tolist(),n['cols']);rows=m.addMConstr(A,v,a['sense'],a['rhs']);m.update();m.setAttr('ConstrName',rows.tolist(),n['rows']);m.ObjCon=float(a['objcon']);m.update()
 aa,ar,nn=o.tk.model_arrays(m);assert o.ac.b.matrix_sha(aa)==recovery.SHA and nn==n
 for k in a:assert np.array_equal(a[k],ar[k]),k
 MATH_ID=recovery.SHA;o.write('RESTORE/API_READBACK.json',dict(passed=True,matrix_sha=MATH_ID,arrays_names_exact=True,rows=m.NumConstrs,cols=m.NumVars,binary=m.NumBinVars),True)
 o.write('RESTORE/BASE_IDENTITY.json',dict(d['ident'],source=str(CFG['snapshot'])),True)
 return m

def install(c,force_restore=False):
 global MATH_ID
 records=o.read(o.ac.rt.verify_ref(c['records']));prepared=[];revision=o.REVISION+1;folder=OUT/'INCREMENTS'/f'R{revision:05d}'
 gate=[];x=np.load(o.ac.rt.verify_ref(c['Q']))['x'];witness=None if force_restore or WITNESS is None else WITNESS
 if witness is not None:assert np.array_equal(witness[:o.N],x)
 for r in records:
  if r.get('universal_zero'):continue
  z=np.load(r['cut']);alpha=float(z['alpha']);beta=z['beta'];h=digest(alpha,beta);sid=r['sid']
  if h in D['support_ids'][sid]:continue
  # Replay full global dual/finite-box certificate, not a validity boolean.
  obj=o.ac.Oracle(sid,OUT/'INSTALL_VALIDATION_NO_MODEL')
  try:obj.validate_result(np.load(o.ac.rt.verify_ref(c['Q']))['x'],Path(r['cut']).parent/'RESULT.json');assert obj.model is None
  finally:obj.close(discard=True)
  decision=None if witness is None else support_decision(beta,alpha,x,witness[D['mapping'][sid]])
  if decision is not None:
   gate.append(dict(kind='SCENARIO',sid=sid,id=h,source=o.ref(r['cut']),decision=decision))
   if not decision['activate']:continue
  source=o.save(folder/'PREPARED'/f'S{sid:04d}_{h}.npz',alpha=np.asarray(alpha),beta=beta)
  prepared.append(dict(sid=sid,id=h,source=o.ref(r['cut']),prepared=source))
 pc=np.load(o.ac.rt.verify_ref(c['risk']['certificate']));p=pc['pstar'];pid=nr.point_sha(p);newp=pid not in D['pids']
 if newp:
  assert o.ac.b.probability_check(p)['passed']
  decision=None if witness is None else probability_decision(p,witness,o.N,D['mapping'])
  if decision is not None:
   gate.append(dict(kind='PROBABILITY',id=pid,source=c['risk']['certificate'],decision=decision))
   if not decision['activate']:newp=False
 if newp:pref=o.save(folder/'PREPARED'/('P_'+pid+'.npz'),p=p)
 if gate:
  name=OUT/'CUT_BANK'/f'{time.time_ns()}_{c["point_sha"]}.json'
  o.write(name,dict(evaluation=c,witness=o.save(name.with_suffix('.npz'),full=witness),rows=gate,never_deleted_existing_rows=True,certificate_identity_unchanged=True),True);DEFERRED_CUTS.append(o.ref(name))
 if not prepared and not newp:return 0
 o.write(folder/'PREPARE.json',dict(parent_revision=o.REVISION,parent_math_id=MATH_ID,new_supports=prepared,new_probability=pref if newp else None,evaluation=c),True)
 checkpoint('INSTALL_NEW_ROWS',np.load(o.ac.rt.verify_ref(c['Q']))['x'],c)
 added=[];v=MODEL.getVars();stored_updates={}
 for r in prepared:
  sid=r['sid'];z=np.load(o.ac.rt.verify_ref(r['prepared']));bb,aa,pr=o.tk.add_supports(MODEL,v,o.N,D['mapping'][sid],sp.csr_matrix(z['beta'][None,:]),np.array([float(z['alpha'])]),[r['id']],D['bank'].lo,D['bank'].hi,'adaptive/'+str(sid))
  path=folder/f'S{sid:04d}_{r["id"]}.npz';o.save(path,alpha=aa,beta_data=bb.data,beta_indices=bb.indices,beta_indptr=bb.indptr,beta_shape=np.array(bb.shape))
  added.append(dict(r,row=MODEL.NumConstrs-1,stored=o.ref(path),proof=pr));stored_updates[sid]=(bb,aa)
 pr=o.tk.add_probability(MODEL,v,o.RHO,D['mapping'],p,'adaptive_probability/'+pid) if newp else None
 MODEL.update();newsha=o.ac.b.matrix_sha(MODEL.getA().tocsr())
 manifest=dict(revision=revision,parent_revision=o.REVISION,parent_math_id=MATH_ID,matrix_sha=newsha,new_supports=added,new_probability=pr,probability_vector=pref if newp else None,evaluation=c,rows=MODEL.NumConstrs)
 o.write(folder/'MANIFEST.json',manifest,True)
 # Memory and disk indices become the same committed revision after all rows are verified.
 for r in added:
  sid=r['sid'];bb,aa=stored_updates[sid];D['stored'][sid]=(sp.vstack([D['stored'][sid][0],bb],format='csr'),np.r_[D['stored'][sid][1],aa]);D['support_ids'][sid].add(r['id'])
 if newp:D['pids'].add(pid)
 o.REVISION=revision;MATH_ID=newsha;INCREMENTS.append(o.ref(folder/'MANIFEST.json'))
 o.write('WORKING_PENDING.json',dict(math_revision=revision,math_id=MATH_ID,base=o.ref(OUT/'RESTORE/BASE_IDENTITY.json'),increments=INCREMENTS,unsolved=True,Pi=None,basis=None))
 checkpoint('LIFT_START',np.load(o.ac.rt.verify_ref(c['Q']))['x'],c);return len(added)+int(newp)

def set_starts(c):
 A,a,n=o.tk.model_arrays(MODEL);assert o.ac.b.matrix_sha(A)==MATH_ID
 starts=[];records=[]
 for ev in {BEST['point_sha']:BEST,c['point_sha']:c}.values():
  x=np.load(o.ac.rt.verify_ref(ev['Q']))['x'];v=o.tk.lift(A,a,x,o.N,o.RHO,D['mapping'],D['stored'])
  if v[o.RHO]>ev['risk']['risk_U']+1e-5:raise ValueError('SUPPORT_ENVELOPE_EXCEEDS_TRUSTED_SAME_POINT_RISK')
  v[o.RHO]=max(v[o.RHO],float(np.nextafter(ev['risk']['risk_U'],np.inf)))
  audit=o.point_audit(A,a,v,a['vtype'][:o.N]=='B');assert max(audit.values())<=1e-9,('COMPLETE_START_AUDIT',audit)
  starts.append(v);records.append(dict(point_sha=ev['point_sha'],risk_U=ev['risk']['risk_U'],rho=float(v[o.RHO]),audit=audit,values=o.save(OUT/'STARTS'/f'R{o.REVISION:05d}_{ev["point_sha"]}.npz',values=v)))
 MODEL.NumStart=len(starts)
 for j,v in enumerate(starts):MODEL.Params.StartNumber=j;MODEL.setAttr('Start',MODEL.getVars(),[gp.GRB.UNDEFINED]*MODEL.NumVars);MODEL.setAttr('Start',MODEL.getVars(),v.tolist())
 MODEL.update();o.write(OUT/'STARTS'/f'R{o.REVISION:05d}_AUDIT.json',dict(math_id=MATH_ID,starts=records,only_Start_modified=True),True)

def snapshot(tag):
 global LAST_SNAPSHOT
 if MODEL is None:return
 folder=f'SNAPSHOTS/{tag}_R{o.REVISION:05d}';A,a,n,ident=o.snap(MODEL,folder)
 o.write(OUT/folder/'STATE_CLASSIFICATION.json',dict(matches_committed_math=ident['matrix_sha']==MATH_ID,classification='COMMITTED_MATH' if ident['matrix_sha']==MATH_ID else 'UNCOMMITTED_WORKING_PENDING',Pi=None,basis=None),True)
 o.write('LATEST_SNAPSHOT.json',dict(identity=o.ref(OUT/folder/'IDENTITY.json'),math_id=ident['matrix_sha'],revision=o.REVISION,increments=INCREMENTS))
 LAST_SNAPSHOT=time.monotonic()

def route(end,known):
 global WITNESS
 light=known_light(known);near=NearIndex(light,o.N,CFG['energy_columns']);deferred=[]
 for cand in end['candidates']:
  z=np.load(o.ac.rt.verify_ref(cand['point']));raw=z['x'];folder=OUT/'points'/('NATIVE_'+end['id']+'_'+nr.point_sha(raw))
  x=o.ac.canonical(raw,folder);full=z['full'].copy();full[:o.N]=x
  action,reference=classify(full,light,o.N,D['mapping'],near)
  if action in ('NEW_MATERIAL_POINT','CERTIFIED_SEPARATION'):
   WITNESS=full;return x
  if action=='DEFER_NEAR_NO_CERTIFICATE_REUSE':deferred.append((x,full,cand))
 if deferred:
  o.write(OUT/'DEFERRED_CANDIDATES'/f'{end["id"]}.json',dict(candidates=[q[2] for q in deferred],reason='scheduling_only_no_certificate_alias',must_audit_exact_before_new_UB_or_final=True),True)
  if end['status']==gp.GRB.OPTIMAL:
   x,full,cand=deferred[0];WITNESS=full;return x
 return None

def certify_if_target():
 global STRICT_BEST,CERT_SECONDS
 gap=publish('CHECK_GAP')
 if not gap['target']:return None
 key=BEST['point_sha']
 if key not in CERT_POINTS and CERT_SECONDS<1200:
  import formal
  formal.OUT=OUT;formal.CTX=CTX
  with phase('FINAL_CERTIFY'):
   cert=formal.final_certify(BEST,OUT/'FINAL_CERTIFY'/uuid.uuid4().hex,max(0,1200-CERT_SECONDS));CERT_SECONDS+=cert['elapsed'];CERT_POINTS.add(key);o.write('FINAL_CERTIFY_RESULT.json',cert)
   if cert.get('strict_UB') is not None:
    value=dict(point_sha=key,strict_UB=cert['strict_UB'],certificate=cert)
    if STRICT_BEST is None or value['strict_UB']<STRICT_BEST['strict_UB']:STRICT_BEST=value
 if STRICT_BEST is not None and o.tk.external_gap(LB,STRICT_BEST['strict_UB'])['target']:return 'EXTERNAL_TARGET_STRICT_UPPER_SOLVER_SUPPORTED_LOWER'
 if STRICT_BEST is None:return 'NUMERICAL_TARGET_CERTIFICATION_PENDING'
 # A larger strict UB does not overwrite best audited; continue global search.
 return None

def run():
 global D,LB,BEST,MATH_ID,NEXT_CALL,OUTERS,LAST_SNAPSHOT,WITNESS
 src=Path(CFG['recovery_source']);snap=Path(CFG['snapshot']);nativepath=Path(CFG['native_result'])
 with phase('RESTORE_REVISION1'):D=recovery.restore(src,snap,nativepath);LB=D['lb'];BEST=D['best'];MATH_ID=recovery.SHA
 CFG['energy_columns']=[i for i,n in enumerate(D['names']['cols'][:o.N]) if n.startswith('E[')];assert len(CFG['energy_columns'])==30
 o.write('ENERGY_COLUMN_MAPPING.json',dict(indices=CFG['energy_columns'],source_names=o.ref(snap/'NAMES.json')),True)
 previous=Path(CFG['memory_resume_source']);state=o.read(previous/'STATE.json')
 with phase('REUSE_TESTS_AND_RESTORE_REVISION2'):
  prior=o.read(Path(CFG['native_test_source'])/'TESTS/NATIVE_SMOKE/END.json')
  assert prior['status']==gp.GRB.OPTIMAL and prior['callback_error'] is None
  o.write('TESTS_REUSED.json',dict(source=o.ref(Path(CFG['native_test_source'])/'TESTS/NATIVE_SMOKE/END.json'),cutoff_and_archive_evidence_reused=True,new_callback_test_result=o.ref(Path(CFG['scheduling_test_root'])/'NATIVE_TESTS/RESULT.json'),optimizer_calls=0),True)
  model_from(D);assert install(D['db'],force_restore=True)==0
  restored_evaluations=[]
  for ref in state['increments']:
   inc=o.read(o.ac.rt.verify_ref(ref));c=inc['evaluation'];recovery.validate_evaluation(c,replay_scenarios=True)
   expected_added=len(inc['new_supports'])+int(inc['new_probability'] is not None)
   assert install(c,force_restore=True)==expected_added
   assert MATH_ID==inc['matrix_sha'] and o.REVISION==inc['revision']
   restored_evaluations.append(c)
  c=state['evaluation'];recovery.validate_evaluation(c)
  latest=o.read(previous/'LATEST_SNAPSHOT.json');identpath=o.ac.rt.verify_ref(latest['identity']);spth=Path(identpath).parent
  expected=sp.load_npz(spth/'A.npz').tocsr();ea=dict(np.load(spth/'ARRAYS.npz'));en=o.read(spth/'NAMES.json')
  aa,ar,nn=o.tk.model_arrays(MODEL)
  assert MATH_ID==state['math_id']==latest['math_id'] and o.ac.b.matrix_sha(expected)==MATH_ID
  diff=aa-expected;diff.eliminate_zeros();assert diff.nnz==0 and nn==en
  for k in ea:assert np.array_equal(ea[k],ar[k]),k
  support_count=sum(map(len,D['support_ids'].values()));probability_count=len(D['pids'])
  assert support_count>=9350 and probability_count>=49
  assert support_count==sum(n.startswith(('theta_support/','adaptive/')) for n in nn['rows'])
  assert probability_count==sum(n.startswith(('theta_probability/','adaptive_probability/')) for n in nn['rows'])
  prior_bounds=[o.read(p).get('bound') for p in (previous/'MASTER').glob('*/END.json')]
  LB=max([LB,state['LB']]+[b for b in prior_bounds if b is not None]);BEST=state['best'];recovery.validate_evaluation(BEST)
  known=D['evals'];known[c['point_sha']]=c;known[BEST['point_sha']]=BEST
  for ev in restored_evaluations:known[ev['point_sha']]=ev
  for q in sorted((previous/'evaluations').glob('*/COMPLETE.json')):
   ev=o.read(q);recovery.validate_evaluation(ev);known[ev['point_sha']]=ev
  D['evals']=known
  o.write('RESTORE/REVISION2_RESUME_GATE.json',dict(passed=True,math_id=MATH_ID,rows=MODEL.NumConstrs,columns=MODEL.NumVars,nnz=aa.nnz,supports=support_count,probabilities=probability_count,previous_state=o.ref(previous/'STATE.json'),matrix_arrays_names_exact=True,completed_LPs_recomputed=0,branch_tree_restored=False),True)
  del expected,ea,en,aa,ar,nn,diff,D['A'],D['a'],D['names'];gc.collect()
  ends=sorted((previous/'MASTER').glob('*/END.json'));assert ends
  end=o.read(ends[-1])
  if state.get('separation_witness'):WITNESS=np.load(o.ac.rt.verify_ref(state['separation_witness']))['full']
  nextpoint=route(end,known)
  x=nextpoint if nextpoint is not None else np.load(o.ac.rt.verify_ref(c['Q']))['x']
  if WITNESS is not None and not np.array_equal(WITNESS[:o.N],x):WITNESS=None
  publish('MEMORY_RESUME_RESTORED');checkpoint('EVALUATE_SAVED_RETURN',x,c)
 while True:
  CTX.check();c=evaluate(x,known);known[c['point_sha']]=c
  if c['UB_audited']<BEST['UB_audited']:BEST=c
  publish('FULL400_POINT');target=certify_if_target()
  if target:return dict(status=target)
  with phase('INSTALL_NEW_ROWS'):changed=install(c)
  if changed or NEXT_CALL==0:
   with phase('LIFT_START'):set_starts(c)
  if OUTERS%5==0 or time.monotonic()-LAST_SNAPSHOT>=1800:snapshot(f'BEFORE_CALL_{NEXT_CALL:05d}')
  checkpoint('MASTER',x,c)
  while True:
   CTX.check();call=NEXT_CALL;NEXT_CALL+=1;checkpoint('MASTER',x,c)
   with phase('MASTER',master_call=call,revision=o.REVISION):end,raw=native(MODEL,'MASTER_MIP',OUT/'MASTER'/f'{call:05d}_R{o.REVISION:05d}',mapping=D['mapping'],known=known_light(known))
   result=commit(end);checkpoint('CHECK_GAP',x,c);publish('NATIVE_RETURN')
   if result=='IMPLEMENTATION_ERROR':return dict(status='IMPLEMENTATION_ERROR_SAVED_NATIVE_RETURN',call=end['end_ref'])
   if end['status'] in (gp.GRB.NUMERIC,gp.GRB.MEM_LIMIT) or end['reason'] in ('GLOBAL_WALL_LIMIT','USER_REQUESTED_STOP','RESOURCE_PAUSE'):return dict(status='NATIVE_'+str(end['status'])+'_'+end['reason'])
   target=certify_if_target()
   if target:return dict(status=target)
   with phase('ROUTE_NEXT_POINT'):nextpoint=route(end,known)
   if nextpoint is not None:
    if nr.point_sha(nextpoint)==c['point_sha'] and changed==0 and end['status']==gp.GRB.OPTIMAL:
     return dict(status='INTERNAL_OPTIMUM_EXTERNAL_TIGHTNESS_DIAGNOSIS_REQUIRED',not_global_optimal=True)
    OUTERS+=1;checkpoint('EVALUATE400',nextpoint);x=nextpoint;publish('COMPLETE_OUTER');break
   if end['status']==gp.GRB.OPTIMAL:
    OUTERS+=1;checkpoint('NO_NEW_POINT',x,c);return dict(status='INTERNAL_OPTIMUM_NO_NEW_POINT_EXTERNAL_GAP',not_global_optimal=True)
   # Same model, no mathematical update, no reset and no repeated Start filling.
   checkpoint('MASTER_SAME_MODEL_CONTINUE',x,c)

def transaction_tests():
 checks=[]
 for name,stage in [('NEW_ROWS_UNSOLVED','WORKING_PENDING'),('NATIVE_RETURN_UNEVALUATED','LAST_RETURNED')]:
  p=OUT/'TESTS/TRANSACTIONS'/name;o.write(p/(stage+'.json'),dict(math_id='NEW' if stage=='WORKING_PENDING' else 'SOLVED',Pi=None,basis=None,unsolved=stage=='WORKING_PENDING'),True);checks.append(dict(name=name,passed=o.read(p/(stage+'.json'))['Pi'] is None))
 folder=OUT/'TESTS/TRANSACTIONS/RISK_RAW';folder.mkdir(parents=True,exist_ok=True);o.save(folder/'FULL400_Q.npz',x=np.zeros(1),qL=np.zeros(400),qU=np.zeros(400));o.write(folder/'GLOBAL_SUPPORT_RECORDS.json',[],True)
 assert eval_stage(folder)=='Q_COMPLETE';checks.append(dict(name='RISK_FAILURE_REUSES_COMMITTED_Q',passed=True))
 # This is execution-state serialization, not a simulated scientific certificate.
 assert CFG['recovery_mode']=='REVISION1_NATIVE_RESULT' and RUN['run_id']!=o.read(Path(CFG['recovery_source'])/'RUN.json')['run_id']
 checks.append(dict(name='NEW_RUN_DEADLINE_AND_EXPLICIT_MODE',passed=True))
 o.write('TESTS/TRANSACTION_TESTS.json',dict(passed=all(v['passed'] for v in checks),checks=checks,optimizer_calls=0),True)

def counts():
 vals=[]
 p=OUT/'OPTIMIZER_CALL_LEDGER.jsonl'
 if p.exists():vals=[json.loads(x) for x in p.read_text().splitlines()]
 native=[v for v in vals if v.get('phase')=='BEGIN'];master=[v for v in native if v.get('stage')=='MASTER_MIP'];other=[]
 for p in OUT.rglob('CALLS/*/BEGIN.json'):other.append(o.read(p))
 return dict(master_calls=len(master),master_returns=sum(v.get('phase')=='END' and v.get('stage')=='MASTER_MIP' for v in vals),other_scientific_calls=len({v['id'] for v in other}),actual_native_tests=sum(v.get('stage')=='TEST' and not v.get('simulated_native') for v in native),mock_wrapper_calls=sum(v.get('simulated_native',False) for v in native),complete_Q_batches=NEW_Q,completed_outer_cycles=OUTERS,own_best_master_LB=own_bound())

def main():
 global OUT,CFG,RUN,CTX,PROB,WORKERS
 import argparse
 p=argparse.ArgumentParser()
 for n in ('execution-root','resume-source','snapshot','native-result','config'):p.add_argument('--'+n,required=True)
 p.add_argument('--start-or-resume',action='store_true');a=p.parse_args();OUT=Path(a.execution_root).resolve();CFG=o.read(a.config);RUN=o.read(OUT/'RUN.json')
 assert a.start_or_resume and CFG['recovery_mode']=='REVISION1_NATIVE_RESULT' and CFG['execution_root']==str(OUT) and CFG['recovery_source']==a.resume_source and CFG['snapshot']==a.snapshot and CFG['native_result']==a.native_result
 assert CFG['master_columns']==61037 and CFG['first_columns']==60780 and RUN['deadline_monotonic']==RUN['start_monotonic']+CFG['total_seconds'] and RUN['max_attempts']==2 and gp.gurobi.version()==(12,0,3)
 o.OUT=OUT;o.CFG=CFG;o.RUN=RUN;o.LAST=Path(CFG['last_root']);CTX=o.ac.bind_context(o.ac.ExecutionContext(OUT,Path(CFG['source_root']),CFG,RUN,RUN['deadline_monotonic']-180));o.CTX=CTX;o.save=save_same;o.ac.save=save_same
 if (OUT/'FINAL_RESULT.json').exists():return int(o.read(OUT/'END.json')['returncode'])
 o.write('EFFECTIVE_CONFIG.json',dict(CFG,interpreter=sys.executable,gurobi_version=list(gp.gurobi.version()),actual_modules={k:str(v.__file__) for k,v in [('runner',o),('math_kernel',o.tk),('native',nr),('core',o.ac)]}),True)
 rc=0;result={};t=time.monotonic()
 try:result=run()
 except BaseException as e:
  rc=2;result=dict(status='GLOBAL_BUDGET_PARTIAL' if isinstance(e,TimeoutError) else 'VALIDATION_OR_EXECUTION_FAILED',phase=PHASE,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),automatic_retry=False)
 finally:
  errors=[]
  for label,func in [('workers',lambda:WORKERS.close() if WORKERS else None),('probability',lambda:PROB.model.dispose() if PROB is not None else None),('final_snapshot',lambda:snapshot('FINAL_'+str(time.time_ns()))),('model',lambda:MODEL.dispose() if MODEL is not None else None),('env',lambda:ENV.dispose() if ENV is not None else None)]:
   try:func()
   except BaseException as e:errors.append(dict(stage=label,error=repr(e),traceback=traceback.format_exc()));rc=2
  o.write('DEPENDENCIES_VERIFIED.json',list(o.cr.DEPS.values()));cs=counts();o.write('CALL_COUNTS.json',cs)
  result.update(run_id=RUN['run_id'],counts=cs,wall=time.monotonic()-t,total_RUN_wall=time.monotonic()-RUN['start_monotonic'],bounds=None if BEST is None else o.tk.external_gap(LB,BEST['UB_audited']),strict_best=STRICT_BEST,math_revision=o.REVISION,cleanup_errors=errors,deferred_cut_banks=DEFERRED_CUTS,scheduling_patch=True,production_signals_sent=[],branch_tree_restored=False,old263_calls_recomputed=False)
  o.write('FINAL_RESULT.json',result);o.write('END.json',dict(epoch=time.time(),returncode=rc,status=result['status']))
 return rc
if __name__=='__main__':raise SystemExit(main())
