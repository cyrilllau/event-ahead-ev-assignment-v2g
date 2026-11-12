#!/usr/bin/env python3
"""One authorized, persistent 12-hour frozen93 continuous-root run."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,uuid,argparse,subprocess,traceback,threading,copy,zipfile
sys.dont_write_bytecode=True
import numpy as np
from scipy import sparse as sp
import durable as d
import runtime as rt
import engine as e
b=e.b
from memory_policy import ResourceRecoveryNeeded,ResourceRetryExhausted,SchedulerStopped,controller_exit
INITIAL_LB=11846.515277934659
EXPECTED={'rows':91241,'columns':60874,'nnz':1469323,'matrix_sha':'60d5469f253c1f2e6bdba793c69647c0ab154c48b819e42d94cee88da225de59','rhs_sense_sha':'9a829a1225ffb2abe83e5f9bf712f1eb91061e32412ef1dbecc24ab67eb65d95','objective_bounds_sha':'f67fd9c59397e94bac8a396d8f6d46a33819c660137f24ee6462d38499fff2c9','rows_sha':'98c82f528148578e065fb2bac055656b5705e691f4e16f0a114dd847a7c40868','columns_sha':'5bb4e0bbb977b45bcb4f77ac49c7a5efb8681797c9b60a72246c621ed0ca801f','objective_constant':0.0}

def immutable_npz(path,**a):
 p=Path(path)
 if Path(p).exists():
  z=np.load(p)
  if set(z.files)!=set(a) or any(not np.array_equal(z[k],v) for k,v in a.items()):raise ValueError('NPZ_IMMUTABLE_CONTENT_CONFLICT')
  return dict(rt.ref(p),encoding='uncompressed_npz')
 return d.publish_npz(p,**a)
def solution_save(out,store,state,z,lb,tag,kind='NATIVE',native=None):
 ident=d.identity(state)
 if len(z['Pi'])!=ident['rows'] or len(z['CBasis'])!=ident['rows'] or any(len(z[k])!=ident['columns'] for k in ('x','RC','VBasis')):raise ValueError('SOLVED_ARRAY_IDENTITY_MISMATCH')
 lb2,cert=b.lagrangian(state['A'],state['rhs'],state['sense'],state['cost'],state['lo'],state['hi'],z['Pi'])
 if abs(lb2-lb)>1e-5:raise ValueError('SOLUTION_LB_REPLAY_MISMATCH')
 store.register(state,[]);folder=out/'SOLVE_RECORDS'/tag
 arrays=immutable_npz(folder/'SOLUTION_BASIS.npz',**z,certificate_Pi=cert)
 proof={'identity':ident,'mapping':store.mapping(state),'solution':arrays,'LB':lb2,'kind':kind,'native':native,'basis_is_seed_only':kind=='TRANSFERRED','upstream_cut_proofs':rt.ref(out/'SOURCES.json')}
 d.atomic(folder/'CERTIFICATE.json',proof,True);return rt.ref(folder/'CERTIFICATE.json')
def solution_load(store,ref):
 proof=d.read(rt.verify_ref(ref));s=store.materialize(proof['mapping']);z=dict(np.load(rt.verify_ref(proof['solution'])));z.pop('certificate_Pi',None)
 lb,_=b.lagrangian(s['A'],s['rhs'],s['sense'],s['cost'],s['lo'],s['hi'],z['Pi'])
 if abs(lb-proof['LB'])>1e-8:raise ValueError('LAST_SOLVED_LB_INTEGRITY_FAILURE')
 return s,z,proof
def seed_attach(state,seed):
 z=np.load(rt.verify_ref(seed));state['VBasis']=z['VBasis'];state['CBasis']=z['CBasis']
 if len(state['CBasis'])!=state['A'].shape[0] or len(state['VBasis'])!=state['A'].shape[1]:raise ValueError('PENDING_BASIS_SEED_MISMATCH')
def basis_seed(out,state,model,tag):
 vb=np.asarray(model.getAttr('VBasis'),int);cb=np.asarray(model.getAttr('CBasis'),int)
 return immutable_npz(out/'BASIS_SEEDS'/(tag+'.npz'),VBasis=vb,CBasis=cb)
def initial_import(out,run,store):
 src=Path(run['source']);s=e.core.load_state(src/'LAST_SOLVED');p=e.core.load_state(src/'WORKING_PENDING')
 if d.identity(s)!=EXPECTED or d.identity(p)!=EXPECTED:raise ValueError('INITIAL_SCIENCE_IDENTITY_CONFLICT')
 if (s['dA']!=p['dA']).nnz or s['dnames']!=p['dnames'] or not np.array_equal(s['drhs'],p['drhs']):raise ValueError('INITIAL_PENDING_DIFFERENT')
 pool=list(map(int,b.read(b.R41/'LOCK.json')['pool']));b.GLOBAL_POOL=pool;b.GLOBAL_QBARS=[b.read(b.R41/'oracles'/f'{sid:04d}'/'QBAR_PROOF.json')['Qbar'] for sid in pool]
 if len(pool)!=93 or len(set(pool))!=93 or len(s['dnames'])!=5058 or s['dA'].nnz!=20246176:raise ValueError('INITIAL_POOL_CONFLICT')
 if sum(b.scenario_name(n) for n in s['names']['rows'])!=214 or sum(b.probability_name(n) for n in s['names']['rows'])!=259:raise ValueError('INITIAL_ROW_CLASSIFICATION')
 proofs={int(r['sid']):r for r in b.read(b.R42/'GLOBAL_BOUND_PROOFS.json')}
 if any(proofs[sid]['Qbar']!=q for sid,q in zip(pool,b.GLOBAL_QBARS)) or np.any(s['lo'][b.FIRST_N:b.FIRST_N+93]<0) or np.any(s['hi'][b.FIRST_N:b.FIRST_N+93]>b.GLOBAL_QBARS):raise ValueError('THETA_BOUND_PROOF_FAILURE')
 deps=[src/'LAST_SOLVED'/f for f in ('A.npz','ARRAYS.npz','NAMES.json','SOLUTION.npz','BASIS.npz','DORMANT_A.npz','DORMANT_ARRAYS.npz','DORMANT_NAMES.json','LB_CERTIFICATE.json')]
 deps += [b.R41/'LOCK.json',b.R41/'identity/FIRST_A.npz',b.R41/'identity/FIRST_ARRAYS.npz',b.R42/'GLOBAL_BOUND_PROOFS.json',b.R42/'RECOURSE_CUT_AUDIT.json',b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz',b.R29/'05_DRO_MOMENTS/EFFECTIVE_INTERVALS.npz',src/'probability/OUTER_027/RAW_FINAL.npz']
 deps += list((src/'RECOVERY').glob('*.json'))
 for sid in pool:deps += [b.R41/'oracles'/f'{sid:04d}'/f for f in ('W.npz','T.npz','ARRAYS.npz','QBAR_PROOF.json')]
 source={'initial':EXPECTED,'dependencies':[rt.ref(f) for f in deps],'pool':pool,'scope':'frozen93 continuous root, full400 probability, other307 Q lower bound0','old_evidence_read_only':True,'source_cut_validity':'inherits hash-bound CONTINUE_01, R42 cut audit and R41 global-bound proof chain; no re-solve'}
 d.atomic(out/'SOURCES.json',source,True);d.atomic(out/'RECOVERY/THETA_BOUND_AUDIT.json',{'passed':True,'ids':pool,'Qbar':b.GLOBAL_QBARS,'proof':rt.ref(b.R42/'GLOBAL_BOUND_PROOFS.json')},True)
 store.initialize(s);store.load_bank([])
 z=dict(np.load(src/'LAST_SOLVED/SOLUTION.npz'));z.update(dict(np.load(src/'LAST_SOLVED/BASIS.npz')))
 lb,_=b.lagrangian(s['A'],s['rhs'],s['sense'],s['cost'],s['lo'],s['hi'],z['Pi'])
 if abs(lb-INITIAL_LB)>1e-7:raise ValueError('INITIAL_LB_REPLAY_FAILURE')
 last=solution_save(out,store,s,z,lb,'INITIAL','INHERITED',rt.ref(src/'LAST_SOLVED/LB_CERTIFICATE.json'))
 # Validate the actual best accepted point, all93 payloads and independent risk upper.
 ev=np.load(src/'outer_021/EVALUATION.npz');er=d.read(src/'outer_021/EVALUATION.json');x=ev['accepted_x'];q=ev['q'];xh=__import__('hashlib').sha256(x.tobytes()).hexdigest()
 if xh!=er['accepted_x_sha'] or xh!='0c72ac6dfb2d8a5feedd45afeaa1145e1b171b16c4080a52bc9c85f5188cc39e':raise ValueError('HISTORICAL_BEST_POINT_CONFLICT')
 results=[]
 for sid in pool:
  f=src/'oracles'/f'{sid:04d}'/'OUTER_021';r=d.read(f/'RESULT.json');pt=Path(r.get('primal',str(f/'POINT.npz')));zz=np.load(pt);oracle=b.Oracle(sid,out/'INHERITANCE_CHECK')
  if not np.array_equal(zz['x'],x) or r['sid']!=sid or r['point_sha']!=xh:raise ValueError('HISTORICAL_Q_POINT_CONFLICT')
  check,*_,qu=oracle.validate_point(x,zz['y'],zz['Pi']);oracle.close()
  if qu!=q[sid]:raise ValueError('HISTORICAL_Q_REPLAY_FAILURE')
  results.append({'sid':sid,'primal':str(pt),'cut':r['cut'],'qU_audited':qu,'point_sha':xh,'sources':[rt.ref(pt),rt.ref(r['cut']),rt.ref(f/'RESULT.json')]})
 if any(q[i]!=0 for i in range(400) if i not in pool):raise ValueError('NONPOOL_Q_NOT_ZERO')
 raw=np.load(src/'probability/OUTER_021/RAW_FINAL.npz')
 if not np.array_equal(raw['q'],q):raise ValueError('HISTORICAL_RISK_Q_CONFLICT')
 risk=b.risk_upper(q,float(raw['beta0']),raw['plus'],raw['minus']);fz=np.load(b.R41/'identity/FIRST_ARRAYS.npz');_,fup=b.enclosure(sp.csr_matrix(fz['cost'][None,:]),x)
 if er['audited_UB']!=13131.310333674706 or abs(risk['risk_upper']-er['risk_upper'])>1e-8 or float(fup[0])!=er['F_upper']:raise ValueError('HISTORICAL_UPPER_REPLAY_FAILURE')
 inherited=out/'INHERITED_BEST';immutable_npz(inherited/'EVALUATION.npz',accepted_x=x,q=q,p=ev['p']);d.atomic(inherited/'RESULTS.json',results,True);d.atomic(inherited/'EVALUATION.json',dict(er,source=rt.ref(src/'outer_021/EVALUATION.json'),primal_status='AUDITED_NOT_STRICT'),True)
 seed=immutable_npz(out/'BASIS_SEEDS/INITIAL.npz',VBasis=z['VBasis'],CBasis=z['CBasis'])
 meta={'phase':'NEW_Q','outer':28,'last_solved':last,'basis_seed':seed,'best_LB':INITIAL_LB,'current_LB':lb,'best_LB_source':rt.ref(src/'LAST_SOLVED/LB_CERTIFICATE.json'),'audited_UB':er['audited_UB'],'strict_UB':None,'best_evaluation':str(inherited),'pool_closed':True,'side_source':rt.ref(src/'probability/OUTER_027/RAW_FINAL.npz'),'complete_outer':0,'complete_Q_batches':0,'dispatched_Q_batches':0,'attempted_outer':0,'master_index':0,'readded':0,'recompressed':0,'history':[],'timing':{},'last_snapshot_wall':time.monotonic(),'final_certification_used_s':0.0,'science_id':d.digest(source),'status':'RUNNING'}
 store.commit(s,meta);store.snapshot(s,'INITIAL');d.atomic(out/'RECOVERY/INITIAL_REPLAY.json',{'identity':EXPECTED,'LB':lb,'historical_retained_LB':INITIAL_LB,'new_gain':0,'initial_phase':'NEW_Q','all93_audited_best_inherited':True},True)
 return s,meta

def summary(out,run,s,meta,attempt,completed_sid=None):
 remaining=max(0,rt.deadline(run)-time.monotonic());ub=meta['audited_UB'];lb=meta['best_LB'];su=meta.get('strict_UB');num=ub-lb if ub is not None else None;strict=su-lb if su is not None else None
 return {'run_id':run['run_id'],'attempt':attempt.name,'phase':meta['phase'],'global_outer':meta['outer'],'complete_outer':meta['complete_outer'],'completed_sid':completed_sid,'complete_Q_batches':meta['complete_Q_batches'],'dispatched_Q_batches':meta['dispatched_Q_batches'],'remaining_global_s':remaining,'current_LB':meta['current_LB'],'LB_certified':lb,'LB_gain':lb-INITIAL_LB,'UB_audited':ub,'UB_certified':su,'best_evaluation':meta.get('best_evaluation'),'numerical_gap':num,'numerical_relative_gap':None if num is None else num/max(1,abs(ub)),'strict_gap':strict,'active_rows':s['A'].shape[0],'active_nnz':s['A'].nnz,'active_scenario_rows':sum(b.scenario_name(n) for n in s['names']['rows']),'dormant_rows':len(s['dnames']),'dormant_nnz':s['dA'].nnz,'pool_closed':meta['pool_closed'],'epoch':time.time(),'historical_optimizer_calls':2821,'production_signals_sent':[],'scope':'frozen93 continuous; other307 Q=0; not full400 integer'}
def threshold(lb,ub,cfg):return ub is not None and ub>=lb and (ub-lb<=cfg['certified_abs_gap'] or (ub-lb)/max(1,abs(ub))<=cfg['certified_rel_gap'])

def final_certify(out,run,meta,cfg,deadline):
 import strict_upper as strict
 start=time.monotonic();left=cfg['final_certification_total_limit_s']-meta['final_certification_used_s'];end=min(deadline,start+max(0,left));ev=Path(meta['best_evaluation']);z=np.load(ev/'EVALUATION.npz');x=z['accepted_x'];fd=out/'FINAL_CERTIFICATIONS'/__import__('hashlib').sha256(x.tobytes()).hexdigest();fd.mkdir(parents=True,exist_ok=True)
 if (fd/'RESULT.json').exists():return d.read(fd/'RESULT.json')
 first=dict(np.load(b.R41/'identity/FIRST_ARRAYS.npz'));A=sp.load_npz(b.R41/'identity/FIRST_A.npz').tocsr();rec={'candidate':str(ev),'scope':'frozen93 continuous, other307 Q lower0','passed':False,'links':[],'method':'independent exact rational verification; no invented exact repair solver'}
 try:
  check=strict.check(A,first['rhs'],first['sense'],first['lo'],first['hi'],x,end);rec['links'].append({'stage':'FIRST_STAGE','result':check})
  if not check['passed']:rec['missing_link']='a: supplied first-stage binary64 primal is not exactly feasible';return rec
  q=np.zeros(400);entries=d.read(ev/'RESULTS.json')
  for r in entries:
   for ref in r['sources']:rt.verify_ref(ref)
   sid=r['sid'];p=np.load(r['primal'])
   if not np.array_equal(p['x'],x):raise ValueError('STRICT_POINT_MISMATCH')
   src=b.R41/'oracles'/f'{sid:04d}';a=dict(np.load(src/'ARRAYS.npz'));W=sp.load_npz(src/'W.npz').tocsr();T=sp.load_npz(src/'T.npz').tocsr()
   check=strict.check(W,a['h'],a['sense'],a['lo'],a['hi'],p['y'],end,T=T,first=x);rec['links'].append({'stage':'RECOURSE','sid':sid,'result':check})
   if not check['passed']:rec['missing_link']='b: original h-T*x exact recourse primal not certified for sid '+str(sid);return rec
   q[sid]=strict.cost_upper(a['c'],p['y'])
  # Reuse existing dual multipliers and independently cover the stricter q_safe.
  raw=np.load(rt.verify_ref(d.read(ev/'EVALUATION.json')['risk_raw'])) if (ev/'EVALUATION.json').exists() and 'risk_raw' in d.read(ev/'EVALUATION.json') else np.load(Path(run['source'])/'probability/OUTER_021/RAW_FINAL.npz')
  risk=b.risk_upper(q,float(raw['beta0']),raw['plus'],raw['minus']);F=strict.cost_upper(first['cost'],x);UB=strict.upper(strict.fraction(F)+strict.fraction(risk['risk_upper']));immutable_npz(fd/'Q_SAFE_AND_DUAL.npz',q_safe=q,plus=raw['plus'],minus=raw['minus'],beta0=risk['beta0'],accepted_x=x)
  rec.update(passed=True,UB_certified=UB,F_upper=F,risk_upper=risk['risk_upper'],missing_link=None)
  return rec
 except TimeoutError:rec['missing_link']='exact verification budget exhausted';return rec
 finally:
  elapsed=time.monotonic()-start;meta['final_certification_used_s']+=elapsed;rec['wall_s']=elapsed;d.atomic(fd/'RESULT.json',rec,True)

def scientific(out,cfg,run,attempt):
 deadline=rt.deadline(run)-120;e.configure(out,run,deadline);store=d.Store(out)
 if not (out/'LAST_COMMITTED.json').exists():s,meta=initial_import(out,run,store)
 else:s,meta=store.replay()
 source=d.read(out/'SOURCES.json');b.GLOBAL_POOL=source['pool'];b.GLOBAL_QBARS=d.read(out/'RECOVERY/THETA_BOUND_AUDIT.json')['Qbar']
 # Every restart checks immutable sources; no successful old LP is rerun.
 for ref in source['dependencies']:rt.verify_ref(ref)
 if not (out/'TEST_REPORT.json').exists():
  from preflight import run_tests
  run_tests(out,run,store,s,meta)
 code={p.name:d.sha(p) for p in Path(__file__).resolve().parent.glob('*.py')};frozen=d.read(Path(__file__).resolve().parent/'CODE_SHA.json')
 if code!=frozen:raise ValueError('FROZEN_CODE_CHANGED_AFTER_PREFLIGHT')
 model=v=None;workers=e.Workers(out,run,deadline,attempt);prob=None;last_hb=0;last_progress=0;stop=threading.Event();status='RUNNING'
 def publish(completed=None,force=False):
  nonlocal last_hb,last_progress
  now=time.monotonic()
  if force or now-last_hb>=cfg['heartbeat_interval_s']:
   d.atomic(out/'HEARTBEAT.json',{'run_id':run['run_id'],'attempt':attempt.name,'phase':meta['phase'],'outer':meta['outer'],'epoch':time.time(),'remaining':deadline+120-now});last_hb=now
  if force or now-last_progress>=cfg['progress_interval_s']:
   d.atomic(out/'PROGRESS.json',summary(out,run,s,meta,attempt,completed));last_progress=now
 def heartbeat():
  while not stop.wait(5):
   try:publish()
   except BaseException as exc:rt.append_event(attempt/'HEARTBEAT_ERRORS.jsonl',{'type':type(exc).__name__,'message':str(exc),'epoch':time.time()})
 th=threading.Thread(target=heartbeat,daemon=True);th.start()
 def commit(phase=None,sources=None):
  if phase:meta['phase']=phase
  t=time.monotonic();store.commit(s,meta,sources);meta['timing']['journal']=meta['timing'].get('journal',0)+time.monotonic()-t;d.atomic(out/'CONTROLLER_STATE.json',{'phase':meta['phase'],'outer':meta['outer'],'last_committed':rt.ref(out/'LAST_COMMITTED.json')});publish(force=True)
 def need_model():
  nonlocal model,v
  if model is None:seed_attach(s,meta['basis_seed']);model,v=b.build_master(s,attempt/'MASTER_BUILD.log')
 def snapshot(tag):
  t=time.monotonic();store.snapshot(s,tag);meta['timing']['full_snapshot']=meta['timing'].get('full_snapshot',0)+time.monotonic()-t;meta['last_snapshot_wall']=time.monotonic()
 try:
  while time.monotonic()<deadline:
   if (out/'USER_STOP.json').exists():status='USER_REQUESTED_STOP';break
   if (out/'PAUSE_RESOURCE.json').exists():status='WAIT_RESOURCE';break
   phase=meta['phase'];outer=meta['outer'];od=out/f'outer_{outer:03d}';od.mkdir(exist_ok=True)
   if phase=='NEW_Q':
    disk=d.disk_check(out);d.atomic(od/'DISK_CHECK.json',disk)
    if not disk['passed']:status='WAIT_RESOURCE';break
    meta.update(attempted_outer=meta['attempted_outer']+1,master_index=0,pool_closed=False);commit('POINT_REPAIR')
   elif phase=='POINT_REPAIR':
    _,last,proof=solution_load(store,meta['last_solved']);pd=od/'POINT_REPAIR'
    if (pd/'POINTS.npz').exists():
     z=np.load(pd/'POINTS.npz');accepted=z['accepted'];audit=d.read(pd/'AUDIT.json')
     if not np.array_equal(z['raw'],last['x'][:b.FIRST_N]):raise ValueError('POINT_REPAIR_SOURCE_CONFLICT')
    else:accepted,audit=b.point_repair(last['x'][:b.FIRST_N],pd,deadline)
    meta['accepted']=rt.ref(pd/'POINTS.npz');meta['accepted_sha']=__import__('hashlib').sha256(accepted.tobytes()).hexdigest();meta['dispatched_Q_batches']+=1;commit('EVALUATE_93')
   elif phase=='EVALUATE_93':
    x=np.load(rt.verify_ref(meta['accepted']))['accepted'];t=time.monotonic()
    try:records=workers.evaluate(x,b.GLOBAL_POOL,meta['science_id'],lambda n:publish(n))
    finally:
     meta['timing']['parallel_recourse_wall']=meta['timing'].get('parallel_recourse_wall',0)+time.monotonic()-t
     meta['timing']['last_evaluation_components']=dict(workers.last_accounting)
    d.atomic(od/'RESULTS.json',records,True);q=np.zeros(400)
    for r in records:q[r['sid']]=r['qU_audited']
    immutable_npz(od/'Q.npz',accepted_x=x,q=q);meta['complete_Q_batches']+=1;meta['results']=rt.ref(od/'RESULTS.json');commit('RISK_SOLVE')
   elif phase in ('RISK_SOLVE','RISK_CERTIFY'):
    z=np.load(od/'Q.npz');q=z['q'];t=time.monotonic();folder=Path(meta.get('risk_folder',str(out/'PROBABILITY_CACHE'/f'OUTER_{outer:03d}_A{attempt.name}')))
    if (folder/'RAW_FINAL.npz').exists():p,risk,strict,info=e.replay_risk(folder,q)
    else:
     if folder.exists():
      # Preserve partial native evidence; restore active sides into a new episode.
      raws=sorted(folder.glob('CALL_*/RAW_NATIVE.npz'))
      if raws:meta['side_source']=rt.ref(raws[-1])
      folder=out/'PROBABILITY_CACHE'/(f'OUTER_{outer:03d}_'+uuid.uuid4().hex)
     if prob is not None:prob.close()
     episode_name=folder.name;prob=e.Probability(out/'PROBABILITY_CACHE'/('OBJECT_'+uuid.uuid4().hex),meta['side_source']);meta['risk_folder']=str(prob.out/episode_name);commit('RISK_SOLVE')
     e.CONTEXT.clear();e.CONTEXT.update(task=d.digest(['RISK',meta['accepted_sha'],outer]));p,risk,strict,info=prob.solve(q,episode_name,deadline)
     # core stores episodes beneath the object's private root.
     actual=prob.out/episode_name
     meta['risk_folder']=str(actual);folder=actual
    if not info.get('rowgen_closed'):raise RuntimeError('RISK_ROWGEN_NOT_CLOSED')
    meta['timing']['risk_wall']=meta['timing'].get('risk_wall',0)+time.monotonic()-t
    meta['side_source']=rt.ref(folder/'RAW_FINAL.npz');meta['risk_certificate']={'folder':str(folder),'info':info};x=z['accepted_x'];first=np.load(b.R41/'identity/FIRST_ARRAYS.npz');_,fu=b.enclosure(sp.csr_matrix(first['cost'][None,:]),x);ub=float(np.nextafter(float(fu[0])+risk['risk_upper'],np.inf))
    immutable_npz(od/'EVALUATION.npz',accepted_x=x,q=q,p=p);d.atomic(od/'EVALUATION.json',{'accepted_x_sha':meta['accepted_sha'],'F_upper':float(fu[0]),'risk_upper':risk['risk_upper'],'audited_UB':ub,'strict_UB':None,'risk_raw':rt.ref(folder/'RAW_FINAL.npz'),'q_scope':'frozen93; other307 zero'},True)
    if ub<meta['audited_UB']:meta['audited_UB']=ub;meta['best_evaluation']=str(od)
    commit('ADD_CUTS')
   elif phase=='ADD_CUTS':
    need_model();_,last,_=solution_load(store,meta['last_solved']);raw=last['x'];records=d.read(rt.verify_ref(meta['results']));fps={b.row_fingerprint(s['A'].getrow(i),s['rhs'][i],s['sense'][i]) for i,n in enumerate(s['names']['rows']) if b.scenario_name(n)};fps.update(b.row_fingerprint(s['dA'].getrow(i),s['drhs'][i],s['dsense'][i]) for i in range(len(s['dnames'])));pfps={b.row_fingerprint(s['A'].getrow(i),s['rhs'][i],s['sense'][i]) for i,n in enumerate(s['names']['rows']) if b.probability_name(n)};added=[]
    for r in records:
     cut=np.load(r['cut']);alpha=float(cut['alpha']);beta=cut['beta'];sid=r['sid'];violation=float(np.longdouble(alpha)+np.asarray(beta,np.longdouble)@np.asarray(raw[:b.FIRST_N],np.longdouble)-raw[b.FIRST_N+b.GLOBAL_POOL.index(sid)])
     if violation>1e-6:
      name=f'r44_benders/{sid}/{outer}';yes,fp=b.add_scenario_cut(s,model,v,sid,alpha,beta,name,fps)
      if yes:added.append({'name':name,'stable_id':fp,'source':rt.ref(r['cut'])})
    p,risk,strict,info=e.replay_risk(meta['risk_folder'],np.load(od/'Q.npz')['q']);yes,fp=b.add_probability_cut(s,model,v,p,strict['epsilon'],f'r44_probability/{outer}',pfps)
    if yes:added.append({'name':f'r44_probability/{outer}','stable_id':fp,'source':rt.ref(Path(meta['risk_folder'])/'RAW_FINAL.npz'),'strict':strict})
    meta['added_this_outer']=len(added);meta['basis_seed']=basis_seed(out,s,model,f'{outer}_ADD');d.atomic(od/'ADD_TRANSACTION.json',{'transaction':d.digest([meta['science_id'],outer,'ADD']),'added':added},True);commit('MASTER_SOLVE',added)
   elif phase in ('MASTER_SOLVE','READD_AND_RESOLVE'):
    need_model();j=meta['master_index'];folder=od/f'MASTER_{j:03d}';task=d.digest(['MASTER',outer,j,d.identity(s)]);t=time.monotonic();z,lb,native=e.solve_master(s,model,v,folder,task,deadline)
    meta['timing']['master_wall']=meta['timing'].get('master_wall',0)+time.monotonic()-t;last=solution_save(out,store,s,z,lb,f'{outer}_{j}',native=native);meta.update(last_solved=last,current_LB=lb)
    if lb>meta['best_LB']:meta.update(best_LB=lb,best_LB_source=last)
    meta['basis_seed']=immutable_npz(out/'BASIS_SEEDS'/f'{outer}_{j}_SOLVED.npz',VBasis=z['VBasis'],CBasis=z['CBasis']);commit('SCAN_DORMANT')
   elif phase=='SCAN_DORMANT':
    _,z,_=solution_load(store,meta['last_solved']);t=time.monotonic();local,viol=b.scan_dormant(s['dA'],s['drhs'],s['dsense'],z['x']);meta['timing']['scan']=meta['timing'].get('scan',0)+time.monotonic()-t;j=meta['master_index'];immutable_npz(od/f'SCAN_{j:03d}.npz',readd_local=local,upper_violation=viol)
    if not len(local):meta['pool_closed']=True;commit('RECOMPRESS')
    else:
     need_model();b.append_rows(s,model,v,s['dA'][local],s['drhs'][local],s['dsense'][local],[s['dnames'][i] for i in local]);keep=np.ones(len(s['dnames']),bool);keep[local]=False;s['dA']=s['dA'][keep].tocsr();s['drhs']=s['drhs'][keep];s['dsense']=s['dsense'][keep];s['dnames']=[n for i,n in enumerate(s['dnames']) if keep[i]];s['dsource']=[None]*len(s['dnames']);meta['readded']+=len(local);meta['master_index']+=1;meta['basis_seed']=basis_seed(out,s,model,f'{outer}_{j}_READD');commit('READD_AND_RESOLVE')
   elif phase=='RECOMPRESS':
    need_model();_,z,_=solution_load(store,meta['last_solved']);t=time.monotonic();model,v,z,rec=b.maybe_recompress(s,model,v,z,outer);meta['timing']['compression']=meta['timing'].get('compression',0)+time.monotonic()-t
    if rec['executed']:
     _,cert=b.lagrangian(s['A'],s['rhs'],s['sense'],s['cost'],s['lo'],s['hi'],z['Pi']);meta['recompressed']+=rec['candidate_rows'];meta['last_solved']=solution_save(out,store,s,z,rec['after_LB'],f'{outer}_COMPRESSED','TRANSFERRED',None);meta['current_LB']=rec['after_LB'];meta['basis_seed']=immutable_npz(out/'BASIS_SEEDS'/f'{outer}_COMPRESSED.npz',VBasis=z['VBasis'],CBasis=z['CBasis'])
    commit('COMMIT')
   elif phase=='COMMIT':
    # A durable commit is independently replayed before accepting the outer.
    check,cm=d.Store(out).replay()
    if d.identity(check)!=d.identity(s):raise ValueError('OUTER_COMMIT_REPLAY_FAILURE')
    meta['complete_outer']+=1;row={'outer':outer,'best_LB':meta['best_LB'],'audited_UB':meta['audited_UB'],'point':meta['accepted_sha'],'added':meta['added_this_outer'],'matrix':d.identity(s),'epoch':time.time()};meta['history'].append(row);d.atomic(od/'OUTER_ACCEPTANCE.json',dict(row,pool_closed=True,commit_readback=True),True)
    if meta['complete_outer']%cfg['full_snapshot_every_outer']==0 or time.monotonic()-meta['last_snapshot_wall']>=cfg['full_snapshot_max_interval_s']:snapshot(f'OUTER_{outer:03d}')
    commit('CHECK_CONVERGENCE')
   elif phase=='CHECK_CONVERGENCE':
    if threshold(meta['best_LB'],meta.get('strict_UB'),cfg):status='CERTIFIED_CONVERGED';break
    if threshold(meta['best_LB'],meta['audited_UB'],cfg):commit('FINAL_CERTIFY');continue
    if meta['complete_outer']%cfg['diagnostic_window_outer']==0:
     h=meta['history'][-5:];repeated=len(h)==5 and len({(x['point'],x['matrix']['matrix_sha'],x['best_LB'],x['audited_UB']) for x in h})==1 and all(x['added']==0 for x in h)
     diag={'history':h,'same_exact_candidate_model_no_new_cut_no_bound_gain':repeated,'cache_exact_hash_only':True,'whole_dormant_closed':meta['pool_closed'],'phase_replay':d.Store(out).replay()[1]['phase'],'stage_timing':meta['timing']};d.atomic(out/'DIAGNOSTICS'/f'{outer}.json',diag,True)
     if repeated:status='NO_PROGRESS_DIAGNOSTIC';break
    meta['outer']+=1;meta.pop('risk_folder',None);commit('NEW_Q')
   elif phase=='FINAL_CERTIFY':
    result=final_certify(out,run,meta,cfg,deadline)
    if result['passed']:
     meta['strict_UB']=result['UB_certified'];meta['strict_UB_source']=str(out/'FINAL_CERTIFICATIONS')
     if meta['strict_UB']<meta['best_LB']-1e-5:raise ValueError('STRICT_LB_UB_CONTRADICTION')
     if threshold(meta['best_LB'],meta['strict_UB'],cfg):status='CERTIFIED_CONVERGED';break
     meta['outer']+=1;meta.pop('risk_folder',None);commit('NEW_Q')
    else:status='NUMERICAL_TARGET_REACHED_CERTIFICATION_PENDING';break
   else:raise ValueError('UNKNOWN_PERSISTED_PHASE '+phase)
  if status=='RUNNING':status='GLOBAL_WALL_LIMIT'
 except ResourceRetryExhausted as exc:
  status=exc.event['kind'];d.atomic(attempt/'RESOURCE_TERMINAL.json',{'event':exc.event,'phase':meta['phase'],'traceback':traceback.format_exc()},True)
 except ResourceRecoveryNeeded as exc:
  status='RESOURCE_RECOVERY_NEEDED';d.atomic(attempt/'RECOVERABLE_RESOURCE.json',{'event':exc.event,'phase':meta['phase'],'traceback':traceback.format_exc()},True)
 except SchedulerStopped as exc:
  status=exc.event['kind'] if exc.event['kind'] in ('GLOBAL_WALL_LIMIT','USER_REQUESTED_STOP','WAIT_RESOURCE') else 'RETRYABLE_NATIVE_OR_INFRASTRUCTURE';d.atomic(attempt/'SCHEDULER_STOP.json',{'event':exc.event,'phase':meta['phase']},True)
 except TimeoutError as exc:
  status='WAIT_RESOURCE' if (out/'PAUSE_RESOURCE.json').exists() else 'USER_REQUESTED_STOP' if (out/'USER_STOP.json').exists() else 'GLOBAL_WALL_LIMIT' if time.monotonic()>=deadline else 'RETRYABLE_NATIVE_OR_INFRASTRUCTURE';d.atomic(attempt/'RECOVERABLE_EVENT.json',{'phase':meta['phase'],'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
 except BaseException as exc:
  message=str(exc);status='DATA_INTEGRITY_FAILURE' if any(x in message for x in ('SHA','INTEGRITY','CONFLICT','IDENTITY','REPLAY')) else 'SCIENTIFIC_VALIDITY_FAILURE' if any(x in message for x in ('CERTIFICATE','PROOF','FIRST_STAGE','STRICT','ROWGEN')) else 'UNHANDLED_IMPLEMENTATION_ERROR';d.atomic(attempt/'FAILURE.json',{'type':type(exc).__name__,'message':message,'phase':meta['phase'],'traceback':traceback.format_exc()},True)
 finally:
  stop.set();th.join(timeout=15)
  cleanup_errors=[]
  for label,fn in [('workers',workers.close),('probability',None if prob is None else prob.close),('master',None if model is None else model.dispose)]:
   if fn is None:continue
   try:fn()
   except BaseException as cleanup_exc:cleanup_errors.append({'component':label,'type':type(cleanup_exc).__name__,'message':str(cleanup_exc)})
  d.atomic(attempt/'CLEANUP_ERRORS.json',{'errors':cleanup_errors,'original_status_retained':status,'worker_timing':workers.last_accounting},True)
  if cleanup_errors and status=='RUNNING':status='RESOURCE_RECOVERY_NEEDED'
  # Persist only states with verified identities; no old Pi is attached to pending.
  meta['status']=status;commit();snapshot(attempt.name+'_EXIT');d.atomic(attempt/'RESULT.json',dict(summary(out,run,s,meta,attempt),status=status,timing=meta['timing'],cache_hits=workers.cache_hits),True)
 return controller_exit(status)

def protect(out):
 owner=d.read(b.DET/'PROCESS.json')['owner'];actual=rt.proc(owner['pid'])
 if actual is None or str(actual['start_ticks'])!=str(owner['start_ticks']):raise ValueError('DET_OWNER_NOT_MATCHED')
 return {'process':actual,'model':rt.ref(b.DET/'MODEL.mps.gz'),'source_owner':owner,'epoch':time.time(),'production_signals_sent':[]}
def finalize(out,cfg,run):
 if (out/'FINAL/SEALED_MANIFEST.json').exists():return
 final=out/'FINAL';final.mkdir(exist_ok=True);s,meta=d.Store(out).replay();result=summary(out,run,s,meta,Path(meta.get('last_attempt','ATTEMPT_UNKNOWN')));result.update(status=meta['status'],timing=meta['timing'],global_wall=time.time()-run['start_utc'],scientific_scope='frozen93 continuous root only; not integer/full400',new_dispatch_stopped=True)
 d.atomic(final/'REPORT.json',result,True);d.atomic(final/'README.md.json',{'replay_command':f'{sys.executable} {out}/tools/replay.py --out {out}','all_old_sources_retained_server':True,'preparation_wall_separate':cfg.get('preparation_wall_s'),'restart_after_boot':'replay supported, automatic user boot service not installed','note':'audited upper is not strict upper; progress does not imply certified convergence'},True)
 files=[rt.ref(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('SEALED_MANIFEST.json','PACKAGE_RESULT.json') and not p.name.startswith('LAUNCHER_') and not p.name.endswith(('.tmp','.lock')) and 'delivery' not in p.parts]
 d.atomic(final/'SEALED_MANIFEST.json',{'sealed':True,'files':files,'exclusions':['manifest itself','*.tmp','*.lock','delivery'],'sealed_after_parent_WAIT_and_lease_release':True},True)
 delivery=out/'delivery';delivery.mkdir(exist_ok=True);analysis=delivery/'R44_AUTONOMOUS_CONVERGENCE_PRO_ANALYSIS.zip'
 with zipfile.ZipFile(analysis,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for folder in ('tools','FINAL','ATTEMPT_000','RESOURCE','RECOVERY','DIAGNOSTICS'):
   for p in (out/folder).rglob('*'):
    if p.is_file() and p.suffix in ('.py','.json','.jsonl','.log','.md') and p.stat().st_size<20*1024**2:z.write(p,str(p.relative_to(out)))
  for name in ('CONFIG.json','AUTHORIZATION.json','RUN.json','CODE_SHA.json','TEST_REPORT.json','SOURCES.json','LAST_COMMITTED.json','PROGRESS.json'):z.write(out/name,name)
 # Complete replay payloads are split by size, with source dependencies explicit.
 parts=[];part=None;bytes_in=0
 for p in sorted(out.rglob('*')):
  if not p.is_file() or 'delivery' in p.parts or p.name.startswith('LAUNCHER_') or p.name.endswith(('.tmp','.lock')):continue
  if part is None or bytes_in+p.stat().st_size>900*1024**2:
   if part:part.close()
   target=delivery/f'R44_AUTONOMOUS_DATA_{len(parts)+1:02d}.zip';part=zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=1);parts.append(target);bytes_in=0
  part.write(p,str(p.relative_to(out)));bytes_in+=p.stat().st_size
 if part:part.close()
 d.atomic(delivery/'PACKAGE_RESULT.json',{'analysis':rt.ref(analysis),'data':[rt.ref(p) for p in parts],'restore':'extract all DATA zip files to one directory; absolute source paths are indexed in SOURCES.json; replay.py supports --relocate-root','external_source_dependencies':rt.ref(out/'SOURCES.json')},True)

def launcher(out,cfg,run):
 try:lock=rt.cc.lock(out/'RUN.lock',nonblock=True);lock.__enter__()
 except BlockingIOError:return
 lease=None
 try:
  if (out/'FINAL/REPORT.json').exists():return
  d.atomic(out/'LAUNCHER_OWNER.json',{'process':rt.proc(os.getpid()),'run_id':run['run_id'],'epoch':time.time()})
  protection=d.read(out/'PRODUCTION_PROTECTION.json') if (out/'PRODUCTION_PROTECTION.json').exists() else protect(out)
  if not (out/'PRODUCTION_PROTECTION.json').exists():d.atomic(out/'PRODUCTION_PROTECTION.json',protection,True)
  for index in range(cfg['max_process_attempts']):
   attempt=out/f'ATTEMPT_{index:03d}'
   if (attempt/'WAIT.json').exists():continue
   if attempt.exists() and (attempt/'PROCESS.json').exists():
    old=d.read(attempt/'PROCESS.json')['child']
    if rt.same(old,rt.proc(old['pid'])):raise ValueError('ORPHAN_LIVE_CONTROLLER_NO_SECOND_WRITER')
    d.atomic(attempt/'WAIT_NOT_OBTAINED.json',{'real_wait':None,'reason':'previous parent disappeared; do not fabricate WAIT'},True);continue
   wait_start=time.monotonic()
   while lease is None and time.monotonic()<rt.deadline(run)-120:
    disk=d.disk_check(out)
    if disk['passed']:lease,inventory=rt.acquire(out,run,protection)
    else:lease=None;inventory={'disk':disk,'admitted':False}
    if lease:break
    d.atomic(out/'RESOURCE_WAIT.json',{'phase':'WAIT_RESOURCE','inventory':inventory,'wait_s':time.monotonic()-wait_start,'epoch':time.time()})
    if time.monotonic()-wait_start>=cfg['resource_wait_limit_s']:break
    time.sleep(min(15,max(0,rt.deadline(run)-120-time.monotonic())))
   if lease is None:
    d.atomic(out/'EARLY_TERMINAL.json',{'status':'RESOURCE_NOT_AVAILABLE','epoch':time.time()});return
   attempt.mkdir(exist_ok=True);d.atomic(attempt/'BEGIN.json',{'epoch':time.time(),'run_id':run['run_id'],'attempt':index,'remaining':rt.deadline(run)-time.monotonic(),'lease':lease},True)
   # Source preparation and all preflight runtime lie after the run BEGIN.
   command=[sys.executable,'-B',str(out/'tools/autonomous_controller.py'),'--config',str(out/'CONFIG.json'),'--child',attempt.name]
   stdout=(attempt/'STDOUT.log').open('xb');stderr=(attempt/'STDERR.log').open('xb');child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True);stdout.close();stderr.close();child_id=rt.wait_process(child.pid,expected=str(out/'tools/autonomous_controller.py'))
   d.atomic(attempt/'PROCESS.json',{'child':child_id,'parent':rt.proc(os.getpid()),'command':command},True);renew=time.monotonic()+cfg['resource_lease_s']-cfg['lease_renew_before_s'];pause=False
   while child.poll() is None:
    now=time.monotonic()
    if now>=renew and not pause:
     new,inv=rt.acquire(out,run,protection,renew=lease)
     if new:lease=new;renew=now+cfg['resource_lease_s']-cfg['lease_renew_before_s']
     else:d.atomic(out/'PAUSE_RESOURCE.json',{'epoch':time.time(),'reason':'renewal readmission failed','inventory':inv});pause=True
    if now>=rt.deadline(run)-90:
     d.atomic(out/'PAUSE_RESOURCE.json',{'epoch':time.time(),'reason':'global deadline cleanup reserve'});pause=True
    if now>=rt.deadline(run)+5:
     if rt.same(child_id,rt.proc(child.pid)):child.terminate()
     break
    time.sleep(3)
   rc=child.wait(timeout=90);d.atomic(attempt/'WAIT.json',{'actual_parent_wait':True,'returncode':rc,'child':child_id,'epoch':time.time()},True);rt.release(out,lease);lease=None
   d.atomic(attempt/'END.json',{'epoch':time.time(),'returncode':rc,'true_WAIT':rt.ref(attempt/'WAIT.json'),'lease_released':True},True)
   if rc!=75:break
   if time.monotonic()>=rt.deadline(run)-120:break
   if (out/'PAUSE_RESOURCE.json').exists():(out/'PAUSE_RESOURCE.json').unlink()
  if (out/'LAST_COMMITTED.json').exists():
   store=d.Store(out);s,meta=store.replay();last_result=d.read(attempt/'RESULT.json') if (attempt/'RESULT.json').exists() else {};meta['status']=last_result.get('status','UNHANDLED_IMPLEMENTATION_ERROR');meta['last_attempt']=attempt.name
   if meta['status'] in ('WAIT_RESOURCE','RETRYABLE_NATIVE_OR_INFRASTRUCTURE'):meta['status']='GLOBAL_WALL_LIMIT' if time.monotonic()>=rt.deadline(run)-120 else 'RETRY_LIMIT'
   store.commit(s,meta);d.atomic(out/'PRODUCTION_PROTECTION_AFTER.json',{'current_process':rt.proc(protection['process']['pid']),'same_owner_or_natural_exit':rt.proc(protection['process']['pid']) is None or rt.same(protection['process'],rt.proc(protection['process']['pid'])),'model_sha':d.sha(protection['model']['path']),'expected_model_sha':protection['model']['sha256'],'production_signals_sent':[]},True);finalize(out,cfg,run)
 except BaseException:
  d.atomic(out/f'LAUNCHER_FAILURE_{os.getpid()}.json',{'traceback':traceback.format_exc(),'epoch':time.time()})
  raise
 finally:
  if lease:rt.release(out,lease)
  lock.__exit__(None,None,None)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--start-or-resume',action='store_true');parser.add_argument('--launcher',action='store_true');parser.add_argument('--child');parser.add_argument('--preflight-child',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();cfg=d.read(args.config);out=Path(cfg['out']);run=rt.create_run(out,cfg);run['source']=cfg['source'];e.configure(out,run,rt.deadline(run)-120)
 if args.preflight_child:
  st=d.Store(out)
  if (out/'LAST_COMMITTED.json').exists():s,m=st.replay()
  else:s,m=initial_import(out,run,st)
  from preflight import run_tests
  run_tests(out,run,st,s,m)
  d.atomic(out/'CODE_SHA.json',{p.name:d.sha(p) for p in (out/'tools').glob('*.py')},True)
  return
 if args.prepare_only:
  protection=protect(out)
  if not (out/'PRODUCTION_PROTECTION.json').exists():d.atomic(out/'PRODUCTION_PROTECTION.json',protection,True)
  lease,inv=rt.acquire(out,run,protection)
  if lease is None:raise RuntimeError('PREFLIGHT_RESOURCE_NOT_AVAILABLE '+json.dumps(inv))
  pre=out/'PREFLIGHT_EXECUTION';pre.mkdir(exist_ok=True);tag=str(time.time_ns());command=[sys.executable,'-B',str(Path(__file__).resolve()),'--config',args.config,'--preflight-child'];p=None
  try:
   with (pre/(tag+'_STDOUT.log')).open('xb') as so,(pre/(tag+'_STDERR.log')).open('xb') as se:
    p=subprocess.Popen(command,stdout=so,stderr=se);owner=rt.wait_process(p.pid,expected=str(Path(__file__).resolve()));d.atomic(pre/(tag+'_PROCESS.json'),{'child':owner,'parent':rt.proc(os.getpid()),'command':command},True);rc=p.wait();d.atomic(pre/(tag+'_WAIT.json'),{'returncode':rc,'actual_parent_wait':True,'child':owner},True)
   if rc:raise RuntimeError('PREFLIGHT_CHILD_FAILED '+str(rc))
  finally:rt.release(out,lease)
  return
 if args.child:
  attempt=out/args.child
  try:raise SystemExit(scientific(out,cfg,run,attempt))
  except SystemExit:raise
  except BaseException as exc:d.atomic(attempt/'ENTRY_FAILURE.json',{'traceback':traceback.format_exc(),'type':type(exc).__name__},True);raise
 if args.launcher:return launcher(out,cfg,run)
 if args.start_or_resume:
  if (out/'LAUNCHER_OWNER.json').exists():
   owner=d.read(out/'LAUNCHER_OWNER.json')['process']
   if rt.same(owner,rt.proc(owner['pid'])):print(json.dumps({'status':'ATTACHED_EXISTING','run_id':run['run_id'],'owner':owner}));return
  if (out/'FINAL/REPORT.json').exists():print(json.dumps({'status':'TERMINAL_RUN_NO_RESTART','report':d.read(out/'FINAL/REPORT.json')}));return
  log=(out/f'LAUNCHER_{time.time_ns()}.log').open('xb');p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--config',args.config,'--launcher'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();print(json.dumps({'status':'LAUNCHER_STARTED','pid':p.pid,'run_id':run['run_id'],'deadline_utc':run['deadline_utc']}))

if __name__=='__main__':main()
