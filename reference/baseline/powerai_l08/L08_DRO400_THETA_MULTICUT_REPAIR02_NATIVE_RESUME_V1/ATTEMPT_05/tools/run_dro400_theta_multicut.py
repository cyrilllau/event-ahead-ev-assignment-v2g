from pathlib import Path
import sys,os,time,json,hashlib,uuid,traceback,gc,math,zipfile,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'lifecycle'))
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import aggregate_core as ac
import cache_restore as cr
import theta_kernel as tk
from repair_guard_helpers import safe_native_observations
from envelope_sparse import enclosure,digest
OUT=CFG=RUN=CTX=LAST=None
STATE={};CURRENT=None;CALLS=0;TESTCALLS=0;REVISION=0
N=60780;RHO=N
write=ac.write;read=ac.read;ref=ac.ref;save=ac.save

def ph(phase,**v):
 elapsed=time.monotonic()-RUN['start_monotonic']
 STATE.update(phase=phase,epoch=time.time(),elapsed=elapsed,preparation_1800_warning_only=bool(elapsed>1800 and phase in ('RESTORE','TEST','BUILD_THETA','TEST_LIFTS')),**v)
 write('STATUS.json',STATE);write('CHECKPOINT.json',dict(STATE,run_id=RUN['run_id'],deadline_epoch=RUN['deadline_epoch'],math_revision=REVISION))
 CTX.check()

def snap(m,folder):
 folder=OUT/folder;folder.mkdir(parents=True,exist_ok=False)
 A,a,n=tk.model_arrays(m);sp.save_npz(folder/'A.npz',A);save(folder/'ARRAYS.npz',**a);write(folder/'NAMES.json',n)
 ident=dict(rows=A.shape[0],cols=A.shape[1],nnz=A.nnz,binaries=int(sum(a['vtype']=='B')),matrix_sha=ac.b.matrix_sha(A),revision=REVISION,files=[ref(folder/v) for v in ('A.npz','ARRAYS.npz','NAMES.json')])
 write(folder/'IDENTITY.json',ident);return A,a,n,ident

def native(m,stage,folder,seconds=None,known=None):
 global CALLS,TESTCALLS
 assert stage in ('TEST','CACHE_LP','CACHE_MIP','MASTER_MIP')
 CTX.check();folder=OUT/folder;folder.mkdir(parents=True,exist_ok=False)
 mip=bool(m.IsMIP);limit=CTX.deadline-time.monotonic()
 if seconds is not None:limit=min(limit,seconds)
 pars=dict(Threads=4 if mip and stage!='TEST' else 1,Method=1,NodeMethod=1,PreSparsify=1,MIPFocus=0,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,MIPGap=.001,MIPGapAbs=1,SoftMemLimit=8,TimeLimit=limit,OutputFlag=1,LogFile=str(folder/'NATIVE.log'))
 if stage=='TEST':pars.update(MIPGap=0,MIPGapAbs=0)
 for k,v in pars.items():setattr(m.Params,k,v)
 assert m.Params.SolutionLimit>1 and abs(m.Params.Cutoff)>1e90
 write(folder/'PARAMETERS.json',dict(pars,Cutoff=None,SolutionLimit=m.Params.SolutionLimit))
 cid=uuid.uuid4().hex;begin=dict(id=cid,phase='BEGIN',stage=stage,kind=stage,epoch=time.time(),owner=ac.rt.proc(os.getpid()),revision=REVISION)
 write(folder/'BEGIN.json',begin,True);ac.rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',begin)
 CALLS+=int(stage!='TEST');TESTCALLS+=int(stage=='TEST')
 queue={};root_end=None;last=0.;cb_error=None;reason=None;known=known or set()
 def callback(model,where):
  nonlocal root_end,last,cb_error,reason
  try:
   now=time.monotonic()
   if now>=CTX.deadline or (OUT/'USER_STOP.json').exists() or (OUT/'PAUSE_RESOURCE.json').exists():reason='GLOBAL_OR_RESOURCE_STOP';model.terminate();return
   if mip and stage!='TEST' and where==gp.GRB.Callback.MIPSOL:
    v=np.asarray(model.cbGetSolution(model.getVars()),float);key=hashlib.sha256(v[:N].tobytes()).hexdigest()
    if key not in known:queue[key]=(v,float(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)),'MIPSOL')
   if mip and where==gp.GRB.Callback.MIP:
    nodes=model.cbGet(gp.GRB.Callback.MIP_NODCNT)
    if nodes>0 and root_end is None:root_end=now
    if stage=='MASTER_MIP' and root_end is not None and now-root_end>=120 and queue:reason='POST_ROOT_NEW_POINT';model.terminate()
   if now-last>=10 and stage!='TEST':
    r=dict(stage=stage,epoch=time.time(),runtime=model.cbGet(gp.GRB.Callback.RUNTIME),root_completed=root_end is not None,pending=len(queue),revision=REVISION)
    if mip and where==gp.GRB.Callback.MIP:
     bd=model.cbGet(gp.GRB.Callback.MIP_OBJBND);r['native_bound']=bd if math.isfinite(bd) and abs(bd)<1e90 else None;r['nodes']=nodes
    write('NATIVE_PROGRESS.json',r);last=now
  except BaseException:
   cb_error=traceback.format_exc();reason='CALLBACK_ERROR';model.terminate()
 error=None
 try:m.optimize(callback)
 except BaseException as exc:error=exc
 ob=safe_native_observations(m,infinity=gp.GRB.INFINITY);val=lambda k:ob.get(k,{}).get('value')
 end=dict(begin,phase='END',epoch=time.time(),status=val('Status'),runtime=val('Runtime'),work=val('Work'),max_mem_GB=val('MaxMemUsed'),observations=ob,root_completed=root_end is not None,reason=reason,callback_error=cb_error,exception=None if error is None else repr(error))
 write(folder/'NATIVE_END.json',end,True)
 raw={}
 for key in ('X','Pi','RC','VBasis','CBasis'):
  try:
   ar=np.asarray(m.getAttr(key))
   if not np.isfinite(ar).all():raise ValueError('NONFINITE_RAW_'+key)
   raw[key]=ar
  except (gp.GurobiError,AttributeError):pass
 end['raw']=save(folder/'RAW_NATIVE.npz',**raw)
 if mip:
  try:bd=float(m.ObjBoundC)
  except (gp.GurobiError,AttributeError):bd=float('nan')
  end['raw_bound']=bd if math.isfinite(bd) and abs(bd)<1e90 else None
  end['bound']=None if end['raw_bound'] is None or end['status'] in (gp.GRB.NUMERIC,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED) else bd-1e-6
  if 'X'in raw and stage!='TEST':
   key=hashlib.sha256(raw['X'][:N].tobytes()).hexdigest();queue.pop(key,None)
   queue={key:(raw['X'],float(m.ObjVal),'FINAL_NATIVE'),**queue}
 candidates=[]
 if stage!='TEST':
  ordered=sorted(queue.items(),key=lambda kv:(0 if kv[1][2]=='FINAL_NATIVE' else 1,kv[1][1],kv[0]))
  for i,(key,(v,obj,source)) in enumerate(ordered):candidates.append(dict(point=save(folder/f'CANDIDATE_{i:04d}.npz',x=v[:N],full=v),point_sha=key,objective=obj,source=source))
 write(folder/'CANDIDATES.json',candidates);end['candidates']=candidates
 write(folder/'END.json',end,True);ac.rt.append_event(OUT/'OPTIMIZER_CALL_LEDGER.jsonl',end)
 if error:raise error
 if cb_error:raise RuntimeError('CALLBACK_ERROR')
 return end,raw

def source_restore():
 cr.OUT=OUT;cr.SRC=Path(CFG['source_root']);cr.CFG=dict(CFG,reuse_preparation=CFG['bank_root'],committed_preparation=CFG['probability_bank_root']);cr.RUN=RUN;cr.B=ac.b
 cr.check=lambda prep=False:CTX.check()
 bank,evs,cp,Af,af,names=cr.restore_committed_bank()
 # Bind LAST to the sealed delivery manifest, not fresh self-generated references.
 archive=LAST.parent/'delivery/DRO_CACHE_ENVELOPE_REPAIR01_PRO_ANALYSIS_01.zip'
 assert cr.sha(archive)=='15b11ce888728e62710536ee95ed0cd22cf27e4536cead8bdc18fd5fa483b9a2'
 with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('PAYLOAD_MANIFEST.json'))
 refs={r['source_path']:dict(path=r['source_path'],sha256=r['sha256'],bytes=r['bytes']) for r in manifest['members']}
 for f in ('B_FROZEN/A.npz','B_FROZEN/ARRAYS.npz','B_FROZEN/NAMES.json','B_FROZEN/IDENTITY.json','B_MIP/RAW_NATIVE.npz','HISTORICAL_LB_SOURCE.json','INHERITED_UB.json'):
  cr.verify(refs[str(LAST/f)])
 A=sp.load_npz(LAST/'B_FROZEN/A.npz').tocsr();a=dict(np.load(LAST/'B_FROZEN/ARRAYS.npz'));nm=read(LAST/'B_FROZEN/NAMES.json');ident=read(LAST/'B_FROZEN/IDENTITY.json')
 assert A.shape==(90976,60781) and ac.b.matrix_sha(A)==ident['matrix_sha'] and sum(a['vtype']=='B')==25770
 assert nm['cols'][:N]==names and np.array_equal(a['cost'][:N],af['cost']) and np.array_equal(a['lo'][:N],af['lo']) and np.array_equal(a['hi'][:N],af['hi']) and np.array_equal(a['vtype'][:N],af['vtype'])
 diff=A[:90768,:N]-Af;diff.eliminate_zeros();assert diff.nnz==0 and A[:90768,N:].nnz==0
 assert np.array_equal(a['rhs'][:90768],af['rhs']) and np.array_equal(a['sense'][:90768],af['sense'])
 historical=read(LAST/'HISTORICAL_LB_SOURCE.json');cr.verify(historical['prior_replay']);cr.verify(historical['upstream'])
 lb=cr.historical();best=min(evs,key=lambda z:z['ce']['UB_audited'])['ce']
 assert abs(best['UB_audited']-34938.625632205454)<1e-7
 eq=read(cr.SRC/'SCENARIO_EQUIVALENCE.json');write('SCENARIO_EQUIVALENCE.json',eq,True)
 # Qbar is already an original-network certified upper bound, not a theta bound.
 qrefs=[r for r in read(Path(CFG['probability_bank_root'])/'SOURCE_BANK_MANIFEST.json')['dependencies'] if r['path'].endswith('/QBAR.npz')]
 assert len(qrefs)==1
 qfile=cr.verify(qrefs[0]);qbar=np.load(qfile)['values'];assert qbar.shape==(400,) and np.isfinite(qbar).all();assert np.array_equal(qbar,np.load(cr.SRC/'QBAR.npz')['values']);save(OUT/'QBAR.npz',values=qbar)
 write('RESTORED_IDENTITY.json',dict(science=ac.SCIENCE,P_identity=CFG['P_identity'],legacy_rows=208,first_rows=90768,first_columns=N,binary=25770,last_snapshot=ident,archive=ref(archive),same_FIRST_as_oracle=ac.b.matrix_sha(ac.first()[0])==ac.b.matrix_sha(Af)))
 assert ac.b.matrix_sha(ac.first()[0])==ac.b.matrix_sha(Af)
 return bank,evs,A,a,nm,eq,lb,best

def build(bank,A,a,nm):
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=gp.Model('DRO400_THETA_MULTICUT',env=env)
 x=m.addMVar(len(a['cost']),lb=a['lo'],ub=a['hi'],vtype=a['vtype'],obj=a['cost']);m.update();m.setAttr('VarName',x.tolist(),nm['cols']);rows=m.addMConstr(A,x,a['sense'],a['rhs']);m.update();m.setAttr('ConstrName',rows.tolist(),nm['rows']);m.ObjCon=float(a['objcon'])
 sids=[s for s,b in enumerate(bank.blocks) if len(b['ids'])];assert len(sids)==256
 theta=m.addMVar(len(sids),lb=0.,ub=gp.GRB.INFINITY,obj=0.,vtype='C');m.update();mapping={s:60781+i for i,s in enumerate(sids)};m.setAttr('VarName',theta.tolist(),[f'THETA_SCENARIO_{s}' for s in sids]);m.update()
 assert m.NumVars==61037 and m.NumBinVars==25770
 stored={};proofs=[];v=m.getVars()
 for sid in sids:
  CTX.check();b=bank.blocks[sid];bb,aa,p=tk.add_supports(m,v,N,mapping[sid],b['B'],b['alpha'],b['ids'],a['lo'][:N],a['hi'][:N],f'theta_support/{sid}')
  stored[sid]=(bb,aa);proofs.append(dict(sid=sid,rows=p))
 pr=[]
 for p,pid in zip(bank.p,bank.pids):pr.append(tk.add_probability(m,v,RHO,mapping,p,'theta_probability/'+pid))
 assert (m.NumVars,m.NumConstrs,m.NumBinVars)==(61037,100039,25770)
 write('THETA_MAP.json',mapping);write('SUPPORT_STORAGE_PROOFS.json',proofs);write('PROBABILITY_STORAGE_PROOFS.json',pr)
 write('REPRESENTATION_PROOF.json',dict(passed=True,original_first_unchanged=True,legacy208_retained=True,nonnegative_theta_by_bounds_only=True,unique_supports=9016,probability_rows=47,finite_cache_is_lower_relaxation=True,full400_probability_coordinates=True,theta_upper_bounds='INFINITY',expected_shape=[100039,61037],binary=25770))
 return env,m,mapping,stored

def tests():
 records=[]
 def ok(name,condition=True):
  if not condition:raise AssertionError(name)
  records.append(dict(name=name,passed=True))
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
 try:
  # Same production encoder; compare explicit combinations with theta LP and MIP.
  import itertools
  vals={}
  B=[sp.csr_matrix([[2.,-1.],[-1.,2.]]),sp.csr_matrix([[-2.,1.],[1.,-2.]])];alpha=[np.array([.2,.1]),np.array([.7,.6])];ps=[np.array([.25,.5,.25]),np.array([.5,.25,.25])]
  for integer in (False,True):
   for theta_form in (False,True):
    m=gp.Model(env=env);x=m.addMVar(2,lb=0,ub=1,vtype='B' if integer else 'C',obj=[.3,.2]);rho=m.addVar(obj=1);m.update()
    if theta_form:
     t=m.addMVar(2);m.update();mapping={0:3,1:4}
     for sid in (0,1):tk.add_supports(m,m.getVars(),2,mapping[sid],B[sid],alpha[sid],['a','b'],np.zeros(2),np.ones(2),f's{sid}')
     for i,p in enumerate(ps):tk.add_probability(m,m.getVars(),2,mapping,p,f'p{i}')
    else:
     for ip,p in enumerate(ps):
      for i,j in itertools.product(range(-1,2),repeat=2):
       beta=np.zeros(2);al=0.
       for sid,k in ((0,i),(1,j)):
        if k>=0:beta+=p[sid]*B[sid].getrow(k).toarray().ravel();al+=p[sid]*alpha[sid][k]
       m.addConstr(rho>=beta@x+al)
    m.update();end,raw=native(m,'TEST',f'TESTS/NATIVE_{int(integer)}_{int(theta_form)}',30)
    assert end['status']==gp.GRB.OPTIMAL;vals[integer,theta_form]=m.ObjVal
    if theta_form and not integer:
     A,a,_=tk.model_arrays(m);cert,pi=tk.lagrangian(A,a,raw['Pi'],2,2);ok('multi_infinite_certificate',cert['LB']<=m.ObjVal+1e-9 and cert['lower_certificate_verified'])
    m.dispose()
   ok('theta_vs_all_combinations_'+str(integer),abs(vals[integer,0]-vals[integer,1])<1e-8)
  m=gp.Model(env=env);v=m.addMVar(4,lb=0,ub=[1,1,gp.GRB.INFINITY,gp.GRB.INFINITY]);m.update()
  p=np.array([2.**-50,1.-2.**-50]);proof=tk.add_probability(m,m.getVars(),2,{0:3},p,'tiny');ok('tiny_probability_actual_api',proof['faithful'] and proof['power2']>0)
  _,_,pr=tk.add_supports(m,m.getVars(),2,3,sp.csr_matrix([[1e-15,-2.]]),np.array([1.]),['tiny'],np.zeros(2),np.ones(2),'tiny_support');ok('dropped_coefficient_compensated',pr[0]['alpha_loss']>=0)
  A,a,_=tk.model_arrays(m);cert,_=tk.lagrangian(A,a,None,2,2);ok('no_PI_not_certified',cert['LB'] is None)
  try:
   bad=dict(a);bad['cost']=a['cost'].copy();bad['cost'][2]=-1;tk.lagrangian(A,bad,np.zeros(A.shape[0]),2,2);raise AssertionError('negative direction accepted')
  except ValueError:ok('negative_unbounded_rejected')
  m.dispose()
  try:ac.full_vector([dict(sid=0,q=0.)],'q');raise AssertionError('partial accepted')
  except ValueError:ok('partial400_cannot_be_UB')
  try:ac.d.encoded(dict(x=float('nan')));raise AssertionError('NaN accepted')
  except ValueError:ok('NaN_json_rejected')
  ok('strict_UB_raise_rechecks_gap',tk.external_gap(99,100)['target'] and not tk.external_gap(99,200)['target'])
  from control_policy import pending_candidate,request_handover
  ok('start_only_not_new',not pending_candidate('same',10,{'same':dict(activity=10.,tightness=0.)}))
  ok('no_root_soft_handover',not request_handover(root_completed=False,seconds_after_root=10000,pending_evaluation_or_separation=True))
  ok('new_point_not_cut_identity',pending_candidate('new',10,{'same':dict(activity=10.,tightness=0.)}))
  write('TESTS/RESULT.json',dict(passed=True,tests=records,native_tests=TESTCALLS,not_bundle_reference=True))
 finally:env.dispose()

def point_audit(A,a,v,first_binary=None):
 r=ac.b.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],v)
 if first_binary is not None:r['integrality']=float(np.max(abs(v[:len(first_binary)][first_binary]-np.rint(v[:len(first_binary)][first_binary])),initial=0))
 return r

def lift_all(m,mapping,stored,evs,bank):
 A,a,_=tk.model_arrays(m);records=[]
 for i,ev in enumerate(evs):
  CTX.check();x=ev['q']['x'];v=tk.lift(A,a,x,N,RHO,mapping,stored);audit=point_audit(A,a,v,a['vtype'][:N]=='B')
  if max(audit.values())>1e-9:raise ValueError(('HISTORICAL_LIFT_INFEASIBLE',i,audit))
  if v[RHO]>ev['ce']['risk']['risk_U']+1e-5:raise ValueError('CACHE_ABOVE_KNOWN_RISK')
  records.append(dict(point_sha=ev['ce']['point_sha'],rho=float(v[RHO]),risk_U=ev['ce']['risk']['risk_U'],audit=audit))
  save(OUT/'LIFTS'/f'HIST_{i:03d}.npz',values=v)
 raw=np.load(LAST/'B_MIP/RAW_NATIVE.npz')['X'];x=ac.canonical(raw[:N],OUT/'points/DIAGNOSTIC');v=tk.lift(A,a,x,N,RHO,mapping,stored);audit=point_audit(A,a,v,a['vtype'][:N]=='B')
 assert max(audit.values())<=1e-9
 value=float(np.longdouble(a['cost'])@np.asarray(v,np.longdouble)+np.longdouble(a['objcon']))
 save(OUT/'LIFTS/DIAGNOSTIC.npz',values=v,x=x)
 rec=dict(full47=records,diagnostic=dict(point_sha=hashlib.sha256(x.tobytes()).hexdigest(),numerical_cache_integer_cost=value,rho=float(v[RHO]),audit=audit,full400_risk_NOT_computed=True,strict_integer_cache_upper_NOT_claimed=True),cache_integer_point_weakness='numerically validated incumbent below historical LB; not inferred from LP lower bound alone')
 write('LIFT_AUDIT.json',rec);return x,v,value

def evaluate(x,workers,eq,prob,number):
 key=hashlib.sha256(x.tobytes()).hexdigest();folder=OUT/'evaluations'/key;complete=folder/'COMPLETE.json'
 if complete.exists():return read(complete)
 folder.mkdir(parents=True,exist_ok=True);reps=[g[0] for g in eq['groups']]
 def progress(n):write('EVALUATION_PROGRESS.json',dict(point_sha=key,completed=n,total=len(reps),universal_zero=len(eq['zeros']),epoch=time.time()))
 raw=workers.evaluate(x,reps,ac.SCIENCE,progress);by={r['sid']:r for r in raw};records=[]
 for sid in eq['zeros']:records.append(dict(sid=sid,universal_zero=True,qL=0.,qU_audited=0.,global_cut_valid=True,zero_proof=ref(OUT/'SCENARIO_EQUIVALENCE.json')))
 for group in eq['groups']:
  for sid in group:records.append(dict(by[group[0]],sid=sid,equivalence_source_sid=group[0]))
 records.sort(key=lambda r:r['sid']);qL=ac.full_vector(records,'qL');qU=ac.full_vector(records,'qU_audited')
 qref=save(folder/'FULL400_Q.npz',x=x,qL=qL,qU=qU);write(folder/'GLOBAL_SUPPORT_RECORDS.json',records,True)
 ph('RISK',point_sha=key)
 ps,ri=prob.solve_full(qL,qU,np.load(OUT/'QBAR.npz')['values'],f'POINT_{number:04d}_{key}')
 f=float(np.nextafter(float(enclosure(sp.csr_matrix(ac.first()[1]['cost'][None,:]),x)[1][0]),np.inf));ub=float(np.nextafter(f+ri['risk_U'],np.inf))
 c=dict(science=ac.SCIENCE,P_identity=CFG['P_identity'],proof_version=CFG['proof_version'],point_sha=key,F=f,UB_audited=ub,strict_UB=None,risk=ri,records=ref(folder/'GLOBAL_SUPPORT_RECORDS.json'),Q=qref,full400=True)
 write(complete,c,True);return c

def make_probability(root):
 root=Path(root);parent=root/'probability';parent.mkdir(parents=True,exist_ok=True)
 return ac.Probability(parent/uuid.uuid4().hex)

def adaptive(m,env,mapping,stored,bank,eq,x,evs,history_lb,best):
 global REVISION,CURRENT
 native_lb=None;lp=read(OUT/'CACHE_LP/SOLVED/LOWER_CERTIFICATE.json');lb=max(history_lb,lp['LB'] if lp['LB'] is not None else -math.inf)
 known={ev['ce']['point_sha']:ev['ce'] for ev in evs};support_ids={s:set(bank.blocks[s]['ids']) for s in mapping};pids=set(bank.pids);strict_ub=None;cert_seconds=0.;evaluated=0;cycles=0;master_calls=0;rows_added=[];lastsnap=time.monotonic();workers=None;prob=None;queue=[]
 def publish(event):
  z=tk.external_gap(lb,best['UB_audited']);write('GAP.json',dict(z,UB_type='full400_audited',strict_UB=strict_ub,best_point=best['point_sha'],event=event,epoch=time.time(),completed_new_evaluations=evaluated,completed_outer_cycles=cycles,math_revision=REVISION));write('BEST.json',best)
  return z
 def install(c):
  nonlocal rows_added
  global REVISION
  recs=read(ac.rt.verify_ref(c['records']));added=[];v=m.getVars()
  for r in recs:
   sid=r['sid']
   if r.get('universal_zero'):continue
   if not r.get('global_cut_valid'):raise ValueError('NON_GLOBAL_CUT')
   z=np.load(r['cut']);alpha=float(z['alpha']);beta=z['beta'];h=digest(alpha,beta)
   if h in support_ids[sid]:continue
   bb,aa,p=tk.add_supports(m,v,N,mapping[sid],sp.csr_matrix(beta[None,:]),np.array([alpha]),[h],bank.lo,bank.hi,f'adaptive/{sid}')
   stored[sid]=(sp.vstack([stored[sid][0],bb],format='csr'),np.r_[stored[sid][1],aa]);support_ids[sid].add(h)
   path=OUT/'INCREMENTS'/f'R{REVISION+1:05d}'/f'S{sid:04d}_{h}.npz';save(path,alpha=aa,beta_data=bb.data,beta_indices=bb.indices,beta_indptr=bb.indptr,beta_shape=np.asarray(bb.shape))
   added.append(dict(sid=sid,id=h,row=m.NumConstrs-1,source=ref(r['cut']),stored=ref(path),proof=p))
  pc=np.load(ac.rt.verify_ref(c['risk']['certificate']));p=pc['pstar'];h=hashlib.sha256(p.tobytes()).hexdigest();pr=None
  if h not in pids:
   if not c['risk']['probability_check']['passed']:raise ValueError('PROBABILITY_NOT_VERIFIED')
   pr=tk.add_probability(m,v,RHO,mapping,p,'adaptive_probability/'+h);pids.add(h)
  if added or pr:
   REVISION+=1;m.update();rowrec=dict(revision=REVISION,new_supports=added,new_probability=pr,evaluation=ref(Path(c['Q']['path']).parent/'COMPLETE.json'),rows=m.NumConstrs)
   write(OUT/'INCREMENTS'/f'R{REVISION:05d}'/'MANIFEST.json',rowrec,True);rows_added.append(rowrec)
  write('WORKING_PENDING.json',dict(revision=REVISION,rows=m.NumConstrs,cols=m.NumVars,binary=m.NumBinVars,matrix_sha=ac.b.matrix_sha(m.getA().tocsr()),increments=rows_added,base=ref(OUT/'THETA_BASE/IDENTITY.json'),unsolved=True,Pi=None,basis=None))
  return len(added)+int(pr is not None)
 try:
  policy=read(ac.ARO/'PATCH_POLICY.json');policy.update(frozen_pool=[g[0] for g in eq['groups']],run_id=RUN['run_id'],original_deadline_utc=RUN['deadline_epoch'],max_local_retries=2,max_controller_attempts_including_old=3,legacy_task_ids=[],max_models_per_worker=16,production_soft_mem_decimal_GB=3.,evict_above_decimal_GB=2.25,evict_target_decimal_GB=2.)
  write('PATCH_POLICY.json',policy,True)
  workers=ac.e.Workers(OUT,dict(RUN,source=str(ac.R41),execution_root=str(OUT),source_root=str(CTX.source_root),config=CFG),CTX.deadline,OUT/'workers'/uuid.uuid4().hex,oracle_cls=ac.Oracle,policy=policy)
  prob=make_probability(OUT)
  while True:
   CTX.check();ph('VERIFY_POINT',outer=cycles)
   key=hashlib.sha256(x.tobytes()).hexdigest()
   if key in known:c=known[key]
   else:
    ph('EVALUATE400',point_sha=key,outer=cycles);c=evaluate(x,workers,eq,prob,evaluated);known[key]=c;evaluated+=1
   if c['UB_audited']<best['UB_audited']:best=c
   ph('ADD_SCENARIO_AND_RISK_ROWS');newrows=install(c);publish('EVALUATION_AND_ROWS')
   # Both native starts are complete lifts in the CURRENT matrix.
   A,a,_=tk.model_arrays(m);startx=np.load(ac.rt.verify_ref(best['Q']))['x'];starts=[tk.lift(A,a,startx,N,RHO,mapping,stored),tk.lift(A,a,x,N,RHO,mapping,stored)]
   m.NumStart=len(starts)
   for j,v in enumerate(starts):
    assert max(point_audit(A,a,v,a['vtype'][:N]=='B').values())<=1e-9
    m.Params.StartNumber=j;m.setAttr('Start',m.getVars(),[gp.GRB.UNDEFINED]*m.NumVars);m.setAttr('Start',m.getVars(),v.tolist())
   m.update();del A,a,starts;gc.collect()
   if cycles%5==0 or time.monotonic()-lastsnap>=1800:
    snap(m,f'SNAPSHOTS/BEFORE_{master_calls:04d}_R{REVISION:05d}');lastsnap=time.monotonic()
   while True:
    ph('MASTER_MIP',outer=cycles,master_call=master_calls)
    folder=f'MASTER/{master_calls:04d}_R{REVISION:05d}';end,raw=native(m,'MASTER_MIP',folder,known=set(known));master_calls+=1
    write('LAST_SOLVED.json',dict(revision=REVISION,rows=m.NumConstrs,cols=m.NumVars,matrix_sha=ac.b.matrix_sha(m.getA().tocsr()),base=ref(OUT/'THETA_BASE/IDENTITY.json'),increments=rows_added,raw=end['raw'],end=ref(OUT/folder/'END.json'),Pi_basis_available=False))
    if end['bound'] is not None:lb=max(lb,end['bound']);native_lb=end['bound']
    cycles+=1;gap=publish('MASTER_RETURN')
    if end['status'] in (gp.GRB.MEM_LIMIT,gp.GRB.NUMERIC):return dict(status='NATIVE_MEMORY_OR_NUMERIC_STOP',bounds=gap,completed_outer_cycles=cycles)
    if gap['target']:
     import formal
     formal.OUT=OUT;formal.CTX=CTX
     ph('FINAL_CERTIFY');cert=formal.final_certify(best,OUT/'FINAL_CERTIFY'/str(time.time_ns()),max(0,1200-cert_seconds));cert_seconds+=cert['elapsed'];write('FINAL_CERTIFY_RESULT.json',cert)
     if cert.get('strict_UB') is None:return dict(status='NUMERICAL_TARGET_STRICT_UPPER_PENDING',bounds=gap,certificate=cert)
     strict_ub=cert['strict_UB'];sg=tk.external_gap(lb,strict_ub)
     if sg['target']:return dict(status='TARGET_WITH_STRICT_PRIMAL_PROBABILITY_UPPER_SOLVER_SUPPORTED_LB',bounds=sg,certificate=cert)
     if strict_ub>best['UB_audited']:best=dict(best,UB_audited=strict_ub)
    candidates=end['candidates'];nextpoint=None
    for cand in candidates:
     rawx=np.load(ac.rt.verify_ref(cand['point']))['x'];h=hashlib.sha256(rawx.tobytes()).hexdigest()
     if h in known:continue
     x0=ac.canonical(rawx,OUT/'points'/f'M{master_calls:04d}_{h}');h0=hashlib.sha256(x0.tobytes()).hexdigest()
     if h0 not in known:nextpoint=x0;break
    if nextpoint is not None:x=nextpoint;break
    if end['status']==gp.GRB.OPTIMAL:return dict(status='CACHE_MASTER_INTERNAL_OPTIMUM_NO_NEW_POINT_EXTERNAL_GAP',bounds=gap,completed_outer_cycles=cycles,not_global_optimal=True)
    CTX.check()
    # No new point/row: resume SAME live object, preserving its native search state.
  
 finally:
  if workers is not None:workers.close()
  if prob is not None and getattr(prob,'model',None) is not None:prob.model.dispose()
  write('ADAPTIVE_COUNTERS.json',dict(completed_new_evaluations=evaluated,completed_outer_cycles=cycles,master_calls=master_calls,math_revision=REVISION,cert_seconds=cert_seconds,new_probability_rows=len(pids)-47,new_scenario_supports=sum(len(support_ids[s])-len(bank.blocks[s]['ids']) for s in mapping)))

def scientific():
 global CURRENT
 ph('RESTORE');bank,evs,A,a,nm,eq,lb,best=source_restore();ph('TEST');tests()
 ph('BUILD_THETA');env,m,mapping,stored=build(bank,A,a,nm);CURRENT=m
 try:
  AA,aa,nn,ident=snap(m,'THETA_BASE');assert np.array_equal(aa['vtype'][:N],a['vtype'][:N]);diff=AA[:90976,:60781]-A;diff.eliminate_zeros();assert diff.nnz==0
  assert np.array_equal(aa['rhs'][:90976],a['rhs']) and np.array_equal(aa['sense'][:90976],a['sense'])
  write('FULL_API_READBACK_TEST.json',dict(passed=True,identity=ident,original_columns_binary=25770,theta_continuous=256,legacy208_exact=True))
  del A,a,AA,aa;gc.collect();ph('TEST_LIFTS');x,v,value=lift_all(m,mapping,stored,evs,bank)
  ph('CACHE_LP');m.update();relaxed=m.relax();relaxed.update();lp_env=gp.Env(empty=True);lp_env.setParam('OutputFlag',0);lp_env.start();lp=relaxed.copy(env=lp_env);relaxed.dispose();lp.update();assert lp.NumBinVars==0 and m.NumBinVars==25770 and lp.NumVars==61037
  try:
   end,raw=native(lp,'CACHE_LP','CACHE_LP',600);LA,la,ln,li=snap(lp,'CACHE_LP/SOLVED');cert,pi=tk.lagrangian(LA,la,raw.get('Pi'),N,RHO)
   cert.update(identity=li,native_status=end['status'],raw=end['raw'],strict_primal_UB=None)
   if pi is not None:cert['corrected_pi']=save(OUT/'CACHE_LP/SOLVED/CORRECTED_PI.npz',Pi=pi)
   if 'X'in raw:cert['primal_audit']=point_audit(LA,la,raw['X'])
   write('CACHE_LP/SOLVED/LOWER_CERTIFICATE.json',cert)
   write('LAST_SOLVED.json',dict(scope='CACHE_LP_RELAXATION',identity=li,certificate=ref(OUT/'CACHE_LP/SOLVED/LOWER_CERTIFICATE.json'),raw=end['raw']))
  finally:lp.dispose();lp_env.dispose()
  del LA,la,ln,raw;gc.collect()
  if not cert['lower_certificate_verified']:return dict(status='CACHE_LP_NOT_CERTIFIED',certificate=cert,remaining='NOT_RUN')
  if value<lb:
   write('CACHE_MIP_GATE.json',dict(run=False,reason='VALIDATED_LOW_CACHE_INTEGER_POINT',cache_integer_cost=value,historical_LB=lb,proof_level='original first-stage and actual cache rows numerical 1e-9; not a strict primal certificate',proceed_full400=True))
  else:
   write('CACHE_MIP_GATE.json',dict(run=True,reason='NO_LOW_CACHE_INTEGER_WITNESS'))
   m.NumStart=1;m.Params.StartNumber=0;m.setAttr('Start',m.getVars(),v.tolist());m.update()
   end,raw=native(m,'CACHE_MIP','CACHE_MIP',900,known={ev['ce']['point_sha'] for ev in evs})
   if end['candidates']:x=ac.canonical(np.load(end['candidates'][0]['point']['path'])['x'],OUT/'points/CACHE_MIP')
  ph('ADAPTIVE_BEGIN');return adaptive(m,env,mapping,stored,bank,eq,x,evs,lb,best)
 finally:
  m.dispose();env.dispose();CURRENT=None

def resume_scientific():
 global CURRENT
 parent=Path(CFG['resume_from']);ph('INFRASTRUCTURE_PREFLIGHT')
 # Same deployed constructor from an absent parent. Any optimize is forbidden.
 old=gp.Model.optimize;count=[0];pobj=None
 def forbidden(*args,**kwargs):count[0]+=1;raise RuntimeError('PREFLIGHT_OPTIMIZE_FORBIDDEN')
 gp.Model.optimize=forbidden
 try:
  pobj=make_probability(OUT/'PREFLIGHT/ABSENT_PARENT');assert pobj.model.NumVars==400 and count[0]==0
  write('PREFLIGHT/RESULT.json',dict(passed=True,same_deployed_constructor=True,parent_initially_absent=True,original400_probability_columns=pobj.model.NumVars,optimizer_calls=0))
 finally:
  gp.Model.optimize=old
  if pobj is not None:pobj.model.dispose()
 ph('RESTORE_COMMITTED_NO_REOPTIMIZE');bank,evs,oldA,olda,nm,eq,lb,best=source_restore();del oldA,olda
 seal=read(OUT/'PREVIOUS_ATTEMPT_SEAL.json')
 for r in seal['files']:ac.rt.verify_ref(r)
 for relative in ('TESTS/RESULT.json','CACHE_LP/SOLVED/LOWER_CERTIFICATE.json','CACHE_MIP_GATE.json','LIFT_AUDIT.json'):
  dest=OUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/relative,dest)
 proof=read(parent/'CACHE_LP/SOLVED/LOWER_CERTIFICATE.json');assert proof['lower_certificate_verified']
 for r in proof['identity']['files']:ac.rt.verify_ref(r)
 ac.rt.verify_ref(proof['corrected_pi']);ac.rt.verify_ref(proof['raw'])
 A=sp.load_npz(parent/'THETA_BASE/A.npz').tocsr();a=dict(np.load(parent/'THETA_BASE/ARRAYS.npz'));names=read(parent/'THETA_BASE/NAMES.json');identity=read(parent/'THETA_BASE/IDENTITY.json')
 assert ac.b.matrix_sha(A)==identity['matrix_sha'] and A.shape==(100039,61037) and sum(a['vtype']=='B')==25770
 mapping={int(s):j for s,j in read(parent/'THETA_MAP.json').items()};stored={};row=90976
 for sid in sorted(mapping):
  n=len(bank.blocks[sid]['ids']);stored[sid]=(-A[row:row+n,:N].tocsr(),a['rhs'][row:row+n].copy());row+=n
 assert row==99992
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=gp.Model('THETA_RESUME_SAME_MATRIX',env=env);CURRENT=m
 try:
  v=m.addMVar(A.shape[1],lb=a['lo'],ub=a['hi'],vtype=a['vtype'],obj=a['cost']);m.update();m.setAttr('VarName',v.tolist(),names['cols']);rows=m.addMConstr(A,v,a['sense'],a['rhs']);m.update();m.setAttr('ConstrName',rows.tolist(),names['rows']);m.ObjCon=float(a['objcon']);m.update()
  aa,ar,nn=tk.model_arrays(m);delta=aa-A;delta.eliminate_zeros();assert delta.nnz==0 and ac.b.matrix_sha(aa)==identity['matrix_sha']
  for k in a:assert np.array_equal(ar[k],a[k]),('RESUME_ARRAY_IDENTITY',k)
  assert nn==names
  x=np.load(parent/'LIFTS/DIAGNOSTIC.npz')['x'];assert hashlib.sha256(x.tobytes()).hexdigest()==read(parent/'LIFT_AUDIT.json')['diagnostic']['point_sha']
  snap(m,'THETA_BASE');write('RESUME_IDENTITY_GATE.json',dict(passed=True,matrix_sha=identity['matrix_sha'],rows=100039,columns=61037,binary=25770,original_RUN=RUN,old_tests_reused=True,old_CACHE_LP_reused=True,old_lifts_reused=True,old_scientific_optimizer_calls=1,new_optimizer_calls_before_adaptive=0,LP_LB=proof['LB']))
  del A,a,aa,ar;gc.collect();ph('ADAPTIVE_BEGIN');return adaptive(m,env,mapping,stored,bank,eq,x,evs,lb,best)
 finally:m.dispose();env.dispose();CURRENT=None

def main():
 global OUT,CFG,RUN,CTX,LAST
 import argparse
 p=argparse.ArgumentParser()
 for arg in ('execution-root','last-envelope-root','bank-root','probability-bank-root','science-root','config'):p.add_argument('--'+arg,required=True)
 p.add_argument('--start-or-resume',action='store_true');args=p.parse_args()
 OUT=Path(args.execution_root).resolve();CFG=read(args.config);RUN=read(OUT/'RUN.json');LAST=Path(args.last_envelope_root).resolve()
 assert args.start_or_resume and CFG['last_root']==str(LAST) and CFG['bank_root']==args.bank_root and CFG['probability_bank_root']==args.probability_bank_root and CFG['source_root']==args.science_root
 CTX=ac.bind_context(ac.ExecutionContext(OUT,Path(args.science_root),CFG,RUN,RUN['deadline_monotonic']-180))
 assert RUN['deadline_monotonic']==RUN['start_monotonic']+14400 and gp.gurobi.version()==(12,0,3)
 if (OUT/'FINAL_RESULT.json').exists():return 0
 rc=0;result={};start=time.monotonic()
 try:result=resume_scientific() if CFG.get('resume_from') else scientific()
 except BaseException as exc:
  rc=2;result=dict(status='GLOBAL_WALL_LIMIT' if isinstance(exc,TimeoutError) else 'UNKNOWN_VALIDATION_OR_EXECUTION_FAILURE',phase=STATE.get('phase'),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False,remaining_stages='NOT_RUN')
 finally:
  write('DEPENDENCIES_VERIFIED.json',list(cr.DEPS.values()))
  result.update(run_id=RUN['run_id'],child_wall=time.monotonic()-start,total_wall=time.monotonic()-RUN['start_monotonic'],scientific_native_calls_main=CALLS,native_tests=TESTCALLS,production_signals_sent=[],branch_tree_restored=False,existing47_batches_recomputed=False)
  write('FINAL_RESULT.json',result);write('END.json',dict(epoch=time.time(),returncode=rc,status=result['status']))
 return rc
if __name__=='__main__':raise SystemExit(main())
