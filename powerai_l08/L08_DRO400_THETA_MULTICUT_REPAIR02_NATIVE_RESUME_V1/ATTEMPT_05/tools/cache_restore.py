from pathlib import Path
import os,sys,time,json,hashlib,uuid,traceback,gc,math
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
from fractions import Fraction as F
from envelope_sparse import Bank,protect,enclosure,digest
BASE=Path('/home/Shixin/ev-assignment')
SCIENCE='c1fe0c77564e0b350782b0375447b2f637f2e5241871b28601a68cf6aa543765'
PID='cadf362cde9b380d196b8c33b34943f62183818374aac49ab0682b49e40a3201'
OUT=SRC=CFG=RUN=None
B=None;PHASE='NOT_STARTED';DEPS={};REPLAY_COUNT=0
native_optimize=gp.Model.optimize
def cv(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(r):
 p=Path(r['path'])
 if str(p) not in DEPS:
  assert sha(p)==r['sha256'],('DEPENDENCY_SHA',str(p))
  DEPS[str(p)]=r
 else:assert DEPS[str(p)]['sha256']==r['sha256']
 return p
def write(p,v):
 p=Path(p);p=p if p.is_absolute() else OUT/p
 assert OUT in p.resolve().parents
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,default=cv,allow_nan=False,sort_keys=True,indent=2)+'\n');os.replace(tmp,p)
def save(path,**v):
 p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),str(p)
 with p.with_suffix(p.suffix+'.tmp').open('xb') as f:np.savez_compressed(f,**v)
 os.replace(p.with_suffix(p.suffix+'.tmp'),p)
 return ref(p)
def check(prep=False):
 deadline=RUN['start_monotonic']+(CFG['preparation_seconds'] if prep else CFG['total_seconds']-120)
 if time.monotonic()>=deadline:raise TimeoutError('PREPARATION_INCOMPLETE' if prep else 'GLOBAL_WALL_LIMIT')
 if (OUT/'USER_STOP.json').exists():raise TimeoutError('USER_STOP')
def status(phase,**v):
 global PHASE
 PHASE=phase;write('STATUS.json',dict(phase=phase,epoch=time.time(),elapsed=time.monotonic()-RUN['start_monotonic'],**v))
def event(v):
 with (OUT/'OPTIMIZER_CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(v,default=cv,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def blocked_optimize(*a,**kw):raise RuntimeError('UNREGISTERED_OR_FORBIDDEN_OPTIMIZER_CALL')
def optimize(m,kind,folder,seconds,wall_end,test=False):
 assert kind in ['TEST_SYNTHETIC','A_LP','B_LP','A_MIP','B_MIP']
 check();folder=OUT/folder;folder.mkdir(parents=True,exist_ok=True)
 rem=min(seconds,wall_end-time.monotonic(),RUN['start_monotonic']+CFG['total_seconds']-120-time.monotonic())
 if rem<=0:raise TimeoutError('STAGE_BUDGET')
 m.Params.TimeLimit=rem;m.Params.OutputFlag=1;m.Params.LogFile=str(folder/'NATIVE.log')
 params={k:getattr(m.Params,k) for k in ['Threads','Method','NodeMethod','PreSparsify','MIPFocus','Seed','FeasibilityTol','OptimalityTol','IntFeasTol','SoftMemLimit','MIPGap','MIPGapAbs','TimeLimit','Cutoff','SolutionLimit']}
 params={k:(None if isinstance(v,float) and abs(v)>=1e90 else v) for k,v in params.items()}
 write(folder/'PARAMETERS.json',params)
 cid=uuid.uuid4().hex;start=time.monotonic();entry=dict(id=cid,kind=kind,scope='TEST' if test else 'SCIENCE',phase='BEGIN',epoch=time.time(),monotonic=start,folder=str(folder))
 write(folder/'BEGIN.json',entry);event(entry);queue={};last=0
 def callback(mm,where):
  nonlocal last
  now=time.monotonic()
  if now>=wall_end or now>=RUN['start_monotonic']+CFG['total_seconds']-120 or (OUT/'USER_STOP.json').exists():mm.terminate();return
  if where==gp.GRB.Callback.MIPSOL:
   v=np.asarray(mm.cbGetSolution(mm.getVars()));h=hashlib.sha256(v.tobytes()).hexdigest()
   queue[h]=(v,float(mm.cbGet(gp.GRB.Callback.MIPSOL_OBJ)))
  if now-last>10:
   rec=dict(kind=kind,epoch=time.time(),runtime=mm.cbGet(gp.GRB.Callback.RUNTIME))
   if where==gp.GRB.Callback.MIP:
    bd=mm.cbGet(gp.GRB.Callback.MIP_OBJBND);rec.update(bound=None if abs(bd)>=1e90 else bd,nodes=mm.cbGet(gp.GRB.Callback.MIP_NODCNT))
   write('NATIVE_PROGRESS.json',rec);last=now
 try:native_optimize(m,callback)
 finally:
  info=dict(entry,phase='END',epoch=time.time(),wall=time.monotonic()-start,status=m.Status,runtime=m.Runtime,work=m.Work,max_mem_GB=m.MaxMemUsed,sol_count=m.SolCount)
  if m.IsMIP:
   info.update(nodes=m.NodeCount,bound=float(m.ObjBoundC)-1e-6 if abs(m.ObjBoundC)<1e90 and np.isfinite(m.ObjBoundC) else None)
  raw={}
  for a in ['X','Pi','RC','VBasis','CBasis']:
   try:raw[a]=np.asarray(m.getAttr(a))
   except gp.GurobiError:pass
  if raw:save(folder.relative_to(OUT)/'RAW_NATIVE.npz',**raw)
  if m.SolCount:info['native_objective']=m.ObjVal
  write(folder/'END.json',info);event(info)
 if m.IsMIP:
  candidates=[]
  if m.SolCount:
   v=np.asarray(m.X);h=hashlib.sha256(v.tobytes()).hexdigest();queue.pop(h,None);candidates.append(('FINAL_NATIVE',h,v,m.ObjVal))
  for h,(v,obj) in queue.items():candidates.append(('CALLBACK',h,v,obj))
  for i,(kind0,h,v,obj) in enumerate(candidates):save(folder.relative_to(OUT)/f'CANDIDATE_{i:04d}.npz',x=v[:-1],rho=v[-1],objective=obj)
  write(folder/'CANDIDATES.json',[dict(source=k,sha=h,objective=o,file=f'CANDIDATE_{i:04d}.npz') for i,(k,h,v,o) in enumerate(candidates)])
 return info,raw
def params(m,mip=False):
 for k,v in dict(Threads=4 if mip else 1,Method=1,NodeMethod=1,PreSparsify=1 if mip else -1,MIPFocus=0,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,SoftMemLimit=8,MIPGap=.001,MIPGapAbs=1).items():setattr(m.Params,k,v)
def snap(m,folder):
 A=m.getA().tocsr();v=m.getVars();r=m.getConstrs()
 a=dict(cost=np.asarray(m.getAttr('Obj',v)),lo=np.asarray(m.getAttr('LB',v)),hi=np.asarray(m.getAttr('UB',v)),rhs=np.asarray(m.getAttr('RHS',r)),sense=np.asarray(m.getAttr('Sense',r)),vtype=np.asarray(m.getAttr('VType',v)),objcon=np.array(m.ObjCon))
 folder=Path(folder);sp.save_npz(OUT/folder/'A.npz',A) if (OUT/folder).exists() else None
 (OUT/folder).mkdir(parents=True,exist_ok=True)
 if not (OUT/folder/'A.npz').exists():sp.save_npz(OUT/folder/'A.npz',A)
 save(folder/'ARRAYS.npz',**a);write(folder/'NAMES.json',dict(rows=m.getAttr('ConstrName',r),cols=m.getAttr('VarName',v)))
 id0=dict(matrix_sha=B.matrix_sha(A),rows=m.NumConstrs,cols=m.NumVars,binaries=m.NumBinVars,nnz=A.nnz,objcon=m.ObjCon,arrays=ref(OUT/folder/'ARRAYS.npz'))
 write(folder/'IDENTITY.json',id0);return A,a,id0
def lp_certificate(m,raw,folder):
 A,a,ident=snap(m,folder)
 if 'Pi' not in raw:return dict(status='NO_PI',LB=None)
 # rho has an infinite upper bound. Scale only valid aggregate multipliers
 # inward until c_rho - A_rho^T pi is certifiably nonnegative.
 pi=np.where(a['sense']=='<',np.minimum(raw['Pi'],0),np.where(a['sense']=='>',np.maximum(raw['Pi'],0),raw['Pi'])).copy()
 for attempt in range(8):
  l,u=enclosure(A.T,pi);rcL=np.asarray(a['cost'],np.longdouble)-u;rcU=np.asarray(a['cost'],np.longdouble)-l
  err=4*np.finfo(np.longdouble).eps*(abs(a['cost'])+np.maximum(abs(l),abs(u))+1);rcL-=err;rcU+=err
  if rcL[-1]>=0:break
  idx=np.flatnonzero(A[:,-1].toarray().ravel()!=0);factor=float(np.nextafter(min(1.,1./float(u[-1]+1e-12)),-np.inf));pi[idx]*=factor
 else:raise ValueError('RHO_INFINITE_BOX_DUAL_SIGN_UNRESOLVED')
 assert a['lo'][-1]==0 and a['hi'][-1]>=gp.GRB.INFINITY and np.isfinite(a['lo'][:-1]).all() and np.isfinite(a['hi'][:-1]).all()
 ll=a['lo'][:-1];hh=a['hi'][:-1]
 terms=np.minimum.reduce([rcL[:-1]*ll,rcL[:-1]*hh,rcU[:-1]*ll,rcU[:-1]*hh])
 pbl,pbu=enclosure(sp.csr_matrix(a['rhs'][None,:]),pi)
 guard=(8*len(terms)+32)*np.finfo(np.longdouble).eps*np.sum(abs(terms),dtype=np.longdouble)
 lower=np.longdouble(m.ObjCon)+pbl[0]+np.sum(terms,dtype=np.longdouble)-guard
 LB=float(np.nextafter(float(lower),-np.inf));save(Path(folder)/'CERTIFIED_PI.npz',Pi=pi)
 audit=B.row_audit(A,a['rhs'],a['sense'],a['lo'],a['hi'],raw['X']) if 'X'in raw else None
 numerical=bool(m.Status==2 and audit and max(audit.values())<=1e-9)
 rec=dict(LB=LB,identity=ident,certificate='directed full box Lagrangian; unbounded rho has certified nonnegative residual',rho_rc_lower=float(rcL[-1]),numerical_gate=numerical,primal_audit=audit,native_objective=float(m.ObjVal) if m.SolCount else None,strict_primal_upper=None)
 write(Path(folder)/'LOWER_CERTIFICATE.json',rec);write('LAST_SOLVED.json',dict(directory=str(OUT/folder),identity=ident,LB=LB));return rec
def validate_bounds(W,T,a,names,proof):
 lo=a['lo'].copy();hi=a['hi'].copy();ix={n:i for i,n in enumerate(names['recourse'])}
 for r in proof['records']:
  j=ix[r['variable']];i=int(r['row'])-90768;sg=int(r['direction']);row=W.getrow(i);tr=T.getrow(i);coef=sg*float(row[0,j])
  assert i>=0 and coef!=0
  # direction changes original row to <=
  assert (a['sense'][i]=='<' and sg==1) or (a['sense'][i]=='>' and sg==-1) or a['sense'][i]=='='
  rest=F(0)
  for jj,c in zip(row.indices,row.data):
   if jj==j:continue
   c=sg*float(c);bound=lo[jj] if c>=0 else hi[jj]
   assert np.isfinite(bound),('BOUND_DEPENDENCY',j,int(jj))
   rest+=F(c)*F(float(bound))
  for jj,c in zip(tr.indices,tr.data):
   c=sg*float(c);rest+=F(c)*F(float(a['xlo'][jj] if c>=0 else a['xhi'][jj]))
  bound=(F(sg*float(a['h'][i]))-rest)/F(coef)
  old=F(*r['exact']);assert old==bound,('BOUND_PROOF_EXACT',r,bound)
  if coef>0:
   assert r['kind']=='upper' and F(r['outward'])>=bound;hi[j]=min(hi[j],r['outward'])
  else:
   assert r['kind']=='lower' and F(r['outward'])<=bound;lo[j]=max(lo[j],r['outward'])
 assert np.array_equal(lo,a['finite_lo']) and np.array_equal(hi,a['finite_hi'])
 return True
def store_row(bank,sep,j,tag,x,rho):
 a,be,cert=bank.row(sep,j);low,up=bank.activity(a,be,x)
 cert.update(activity_lower=low,activity_upper=up,rho=rho,violation_lower=low-rho,tau=sep['tau'],point_sha=hashlib.sha256(x.tobytes()).hexdigest())
 return dict(alpha=a,beta=be,certificate=cert,sep=sep,j=j,tag=tag)
def add_row(m,bank,row,folder):
 a=row['alpha'];be=row['beta'];v=m.getVars();nz=np.flatnonzero(be)
 co=m.addConstr(v[-1]>=gp.LinExpr(be[nz].tolist(),[v[int(k)] for k in nz])+a,name='cache/'+row['certificate']['row_id']);m.update()
 rr=m.getRow(co);actual=np.zeros(len(be));rho=0
 for k in range(rr.size()):
  ix=rr.getVar(k).index;value=rr.getCoeff(k)
  if ix==len(be):rho=value
  else:actual[ix]=-value
 assert rho==1 and co.Sense=='>'
 aa,bb,proof=bank.row(row['sep'],row['j'],stored=actual)
 co.RHS=min(co.RHS,aa);m.update();assert co.RHS<=aa
 save(Path(folder)/'ROW.npz',alpha=co.RHS,beta=actual,pstar=bank.p[row['j']])
 write(Path(folder)/'PROOF.json',dict(row['certificate'],stored_guard=proof,actual_rhs=co.RHS,actual_coefficients=ref(OUT/folder/'ROW.npz'),getRow_verified=True))
 return co
def baseline(env,bank,evs,cp,Af,af,names):
 check(True);m=gp.read(str(SRC/'FINAL_MASTER.mps.gz'),env=env);m.Params.OutputFlag=0
 assert (m.NumConstrs,m.NumVars,m.NumBinVars,m.NumNZs)==(90815,60781,25770,750930)
 A=m.getA().tocsr();vs=m.getVars();rs=m.getConstrs();n=m.getAttr('VarName',vs)
 assert n[:-1]==names and n[-1]=='DRO_RHO' and m.ObjCon==0
 assert np.array_equal(m.getAttr('VType',vs)[:-1],af['vtype'])
 # MPS roundtrip must match the common first block exactly, including original bounds and objective.
 assert np.array_equal(m.getAttr('LB',vs)[:-1],af['lo']) and np.array_equal(m.getAttr('UB',vs)[:-1],af['hi'])
 assert np.array_equal(m.getAttr('Obj',vs)[:-1],af['cost'])
 common=A[:90768,:60780];diff=(common-Af).tocsr();diff.eliminate_zeros()
 assert diff.nnz==0 and A[:90768,-1].nnz==0
 assert np.array_equal(m.getAttr('RHS',rs)[:90768],af['rhs']) and np.array_equal(m.getAttr('Sense',rs)[:90768],af['sense'])
 checks=[]
 for k,ev in enumerate(evs):
  z=dict(np.load(verify(ev['ce']['aggregate'])));r=rs[90768+k];actual=-A.getrow(90768+k).toarray().ravel()[:-1]
  assert A[90768+k,-1]==1 and r.Sense=='>'
  # Reconstruct the source combination from exactly the corresponding bank.
  selected=[];alpha=[]
  for sid in range(400):
   rec=ev['records'][sid]
   if rec.get('universal_zero'):alpha.append(0.);selected.append(sp.csr_matrix((1,60780)))
   else:
    block=bank.blocks[sid];ix=block['ids'].index(SUPPORT_BY_BANK[(ev['ce']['point_sha'],sid)]);alpha.append(float(block['alpha'][ix]));selected.append(block['B'].getrow(ix))
  safe,_,proof=protect(alpha,sp.vstack(selected,format='csr'),ev['p'],af['lo'],af['hi'],stored=actual)
  old=r.RHS;r.RHS=min(old,safe)
  checks.append(dict(row=r.ConstrName,old_rhs=old,new_rhs=r.RHS,changed=r.RHS!=old,proof=proof,source=ev['ce']['aggregate']))
 m.update();m.write(str(OUT/'A_BASE.mps.gz'));write('ORIGINAL_47_ROWS.json',checks)
 snap(m,'BASELINE');m.dispose()
 return checks
def witness(bank,evs):
 x=np.load(SRC/'native/000046/RAW_NATIVE.npz')['x'];first='951cc92fb17df719e6a1d224095b39d0e51a23d47a2b56289b2a84cbf1eb2f91';last='382f9baee6f53953560cad1b9f3f924835f7a4102d0f6f2cb43e6d888a2df324'
 wanted=[next(e for e in evs if e['ce']['point_sha']==k) for k in [first,last]]
 results=[]
 for subset in [wanted[:1],wanted]:
  blocks=[]
  for sid in range(400):
   rr=[];aa=[];ids=[]
   for ev in subset:
    r=ev['records'][sid]
    if r.get('universal_zero'):continue
    z=np.load(r['cut']);a=float(z['alpha']);bb=z['beta'];rr.append(sp.csr_matrix(bb[None,:]));aa.append(a);ids.append(digest(a,bb))
   blocks.append(dict(B=sp.vstack(rr,format='csr') if rr else sp.csr_matrix((0,60780)),alpha=np.asarray(aa),ids=ids))
  b0=Bank(blocks,[e['p'] for e in (wanted[:1] if len(subset)==1 else evs)],[hashlib.sha256(e['p'].tobytes()).hexdigest() for e in (wanted[:1] if len(subset)==1 else evs)],bank.lo,bank.hi)
  sp0=b0.separate(x,0);results.append(sp0)
 assert abs(results[0]['H_upper']-15705.577346)<1e-5
 assert abs(results[1]['H_upper']-21192.449161)<1e-5
 full=bank.separate(x,0);assert full['H_upper']+1e-5>=results[-1]['H_lower']
 write('WITNESS_REGRESSION.json',dict(two_subsets=results,full47=full,point_only_not_global_LB=True));return full
def historical():
 # Reuse the exact prior immutable replay and explicitly reference the real source certificate.
 rec=read(SRC/'HISTORICAL_LB_REPLAY.json');verify(rec['certificate'])
 proof=read(verify(rec['certificate']));verify(proof['solution'])
 assert abs(rec['LB']-12536.584767617334)<1e-8 and abs(proof['LB']-rec['LB'])<1e-8
 write('HISTORICAL_LB_SOURCE.json',dict(rec,prior_replay=ref(SRC/'HISTORICAL_LB_REPLAY.json'),not_native_676_evidence=True))
 return rec['LB']
def set_variable_types(model, values):
 model.update()
 variables=model.getVars()
 assert len(variables)==len(values)
 model.setAttr('VType',variables,values)
 model.update()
 assert model.getAttr('VType',variables)==list(values)

def scientific():
 status('RESTORE_COMMITTED_RUN_STAGE')
 committed=Path(CFG['committed_run_stage'])
 expected=read(OUT/'COMMITTED_STAGE_REFERENCES.json')
 for item in expected:verify(item)
 assert read(committed/'PREPARATION_RESULT.json')['passed']
 bank,evs,cp,Af,af,names=restore_committed_bank();check(True)
 write('PREPARATION_REUSED.json',dict(source=ref(committed/'PREPARATION_RESULT.json'),tests_not_repeated=True,witness_not_repeated=True,baseline_not_rebuilt=True))
 import shutil
 shutil.copyfile(committed/'A_BASE.mps.gz',OUT/'A_BASE.mps.gz')
 historical_lb=read(committed/'HISTORICAL_LB_SOURCE.json')['LB']
 write('HISTORICAL_LB_SOURCE.json',read(committed/'HISTORICAL_LB_SOURCE.json'))
 best=next(e for e in evs if e['ce']['point_sha']==cp['best']['point_sha'])
 cert=np.load(verify(best['ce']['risk']['certificate']))
 risk=B.risk_upper(best['q']['qU'],float(cert['beta0']),cert['plus'],cert['minus'])
 assert abs(risk['risk_upper']-best['ce']['risk']['risk_U'])<1e-8
 fupper=float(np.nextafter(float(enclosure(sp.csr_matrix(af['cost'][None,:]),best['q']['x'])[1][0]),np.inf))
 ub=float(np.nextafter(fupper+risk['risk_upper'],np.inf));write('INHERITED_UB.json',dict(audited_UB=ub,strict_UB=None,point=best['ce']['Q'],risk=best['pc'],risk_replay={k:v for k,v in risk.items() if k not in ['plus','minus']},F=fupper))
 write('PREPARATION_RESULT.json',dict(passed=True,elapsed=time.monotonic()-RUN['start_monotonic'],source_bank=ref(OUT/'SOURCE_BANK_MANIFEST.json')))
 status('A_LP')
 with gp.Env(empty=True) as env:
  env.setParam('OutputFlag',0);env.start()
  ma=gp.read(str(OUT/'A_BASE.mps.gz'),env=env);set_variable_types(ma,['C']*ma.NumVars);ma.update();params(ma)
  info,raw=optimize(ma,'A_LP','A_LP',300,min(time.monotonic()+330,RUN['start_monotonic']+5280));ac=lp_certificate(ma,raw,'A_LP/SOLVED');ma.dispose()
 if 'X' not in raw:raise RuntimeError('LP_COMPARISON_INCOMPLETE_NO_BASE_POINT')
 separation_start=time.monotonic();sep_end=min(separation_start+1500,RUN['start_monotonic']+5280)
 candidates={};seq=[]
 witness_raw=np.load(SRC/'native/000046/RAW_NATIVE.npz');seq.append(('FIXED_COUNTEREXAMPLE',witness_raw['x'],0.))
 for p in sorted((SRC/'native').glob('*/RAW_NATIVE.npz')):
  z=np.load(p)
  if 'x'in z:seq.append((str(p),z['x'],float(z['rho'])))
 for ev in evs:seq.append((ev['ce']['point_sha'],ev['q']['x'],ev['ce']['risk']['risk_U']))
 seq.append(('A_LP',raw['X'][:-1],raw['X'][-1]))
 status('INITIAL_CACHE_SEPARATION',points=len(seq));initial_scanned=0
 for tag,x,rho in seq:
  check()
  if time.monotonic()>=min(sep_end,separation_start+300):break
  ss=bank.separate(x,rho);initial_scanned+=1
  row=store_row(bank,ss,ss['order'][0],tag,x,rho)
  if row['certificate']['violation_lower']>ss['tau']:
   h=row['certificate']['row_id']
   if h not in candidates or candidates[h]['certificate']['violation_lower']<row['certificate']['violation_lower']:candidates[h]=row
 write('INITIAL_POINT_SCAN.json',dict(total=len(seq),scanned=initial_scanned,not_scanned=len(seq)-initial_scanned,wall=time.monotonic()-separation_start))
 initial=sorted(candidates.values(),key=lambda r:(-r['certificate']['violation_lower'],r['certificate']['row_id']))[:48]
 added=[];used=set();actual_used=set();trace=[];bc=None;last_raw=None;cache_state='PARTIAL'
 with gp.Env(empty=True) as env:
  env.setParam('OutputFlag',0);env.start();mb=gp.read(str(OUT/'A_BASE.mps.gz'),env=env);set_variable_types(mb,['C']*mb.NumVars);mb.update();params(mb)
  for row in initial:add_row(mb,bank,row,Path('ROWS')/f'{len(added):04d}');added.append(row);used.add(row['certificate']['row_id']);actual_used.add(row['actual_id'])
  for iteration in range(40):
   check()
   if time.monotonic()>=sep_end-10:break
   folder=Path('B_LP')/f'{iteration:03d}';status('B_LP',iteration=iteration,added=len(added))
   info,rr=optimize(mb,'B_LP',folder,max(.01,sep_end-time.monotonic()),sep_end);certb=lp_certificate(mb,rr,folder/'SOLVED')
   if certb.get('lower_certificate_verified') and certb.get('LB') is not None and (bc is None or certb['LB']>bc['LB']):bc=certb
   if 'X' not in rr:break
   last_raw=rr
   ss=bank.separate(rr['X'][:-1],rr['X'][-1]);trace.append(dict(iteration=iteration,LB=certb.get('LB'),best_LB=None if bc is None else bc['LB'],model_id=certb['model_id'],H_lower=ss['H_lower'],H_upper=ss['H_upper'],rho=float(rr['X'][-1]),closed=ss['closed'],rows=mb.NumConstrs,nnz=mb.NumNZs,native_status=info['status'],native_objective=info['native_objective'],stage_wall=time.monotonic()-separation_start))
   write('LP_TRAJECTORY.json',trace)
   if ss['closed']:cache_state='CACHE_CLOSED';break
   new=0
   for j in ss['order']:
    if new>=3 or len(added)>=256 or time.monotonic()>=sep_end:break
    row=store_row(bank,ss,j,'LP_'+str(iteration),rr['X'][:-1],rr['X'][-1]);rid=row['certificate']['row_id']
    if rid in used or rid in actual_used or row['certificate']['violation_lower']<=ss['tau']:continue
    add_row(mb,bank,row,Path('ROWS')/f'{len(added):04d}');added.append(row);used.add(rid);actual_used.add(row['actual_id']);new+=1
   if not new:
    cache_state='SEPARATION_NUMERIC_UNRESOLVED' if len(added)<256 and time.monotonic()<sep_end else 'PARTIAL';break
   write('WORKING_PENDING.json',dict(rows=mb.NumConstrs,new_rows=len(added),last_solved=read(OUT/'LAST_SOLVED.json') if (OUT/'LAST_SOLVED.json').exists() else None,basis_not_attached_to_pending=True))
  set_variable_types(mb,list(af['vtype'])+['C']);mb.update();assert mb.NumBinVars==25770
  mb.write(str(OUT/'B_FROZEN.mps.gz'));pending_identity=snap(mb,'B_FROZEN')[2];write('WORKING_PENDING.json',dict(identity=pending_identity,matrix=str(OUT/'B_FROZEN/A.npz'),integer_vtype_restored=True,has_solution_or_basis_attached=False));mb.dispose()
 # Every new row regression over all prior evaluated points (no new Q calls).
 for i in range(len(added)):
  z=np.load(OUT/'ROWS'/f'{i:04d}'/'ROW.npz')
  violations=[]
  for ev in evs:
   _,high=bank.activity(float(z['alpha']),z['beta'],ev['q']['x'])
   if high>ev['ce']['risk']['risk_U']+1e-5:violations.append(ev['ce']['point_sha'])
  assert not violations,('NEW_ROW_ABOVE_AUDITED_RISK',i,violations)
 write('NEW_ROW_POINT_REGRESSION.json',dict(passed=True,rows=len(added),points=47,not_substitute_for_global_proof=True))
 gate_record=checked_lp_gate(ac,bc);gate=gate_record['run_mip']
 new_lp=max([c['LB'] for c in [ac,bc] if c and c.get('lower_certificate_verified') and c.get('LB') is not None],default=None)
 comparison=dict(A_LP=ac,B_LP=bc,cache_state=cache_state,new_rows=len(added),LP_gate=gate_record['status'],gate_record=gate_record,historical_LB=historical_lb,audited_UB=ub,strict_UB=None,strongest_new_LB=new_lp,effective_LB=max(historical_lb,new_lp) if new_lp is not None else historical_lb,preparation_and_LP_wall=time.monotonic()-RUN['start_monotonic'])
 write('REAL_LP_CYCLE_ACCEPTANCE.json',dict(A_executed=True,B_optimizations=len(trace),full_cache_scans_after_solve=len(trace),closed=cache_state=='CACHE_CLOSED',returned_X_evaluated=bool(trace),reoptimization_after_added_row=len(trace)>=2,cache_state=cache_state,remaining_stages_conditional=True))
 write('COMPARISON.json',comparison)
 if not gate:return dict(comparison,status=comparison['LP_gate'],new_full400_UB=False,MIP='NOT_RUN')
 if RUN['deadline_monotonic']-time.monotonic()<2040:return dict(comparison,status='INSUFFICIENT_COMPLETE_PAIR_BUDGET',MIP='NOT_RUN')
 outcomes=[]
 for label,file in [('A_MIP','A_BASE.mps.gz'),('B_MIP','B_FROZEN.mps.gz')]:
  start=time.monotonic();wall_end=start+960;status(label)
  with gp.Env(empty=True) as env:
   env.setParam('OutputFlag',0);env.start();m=gp.read(str(OUT/file),env=env);params(m,True);assert m.NumVars==60781 and m.NumBinVars==25770
   v=m.getVars();m.NumStart=0;m.update();m.NumStart=1;m.Params.StartNumber=0
   x=best['q']['x'];start_rho=risk['risk_upper'];A=m.getA();vals=np.r_[x,start_rho];audit=B.row_audit(A,np.asarray(m.RHS),np.asarray(m.Sense),np.asarray(m.LB),np.asarray(m.UB),vals)
   assert max(audit.values())<=1e-9,('COMPLETE_START_FAILED',audit)
   m.setAttr('Start',v,vals.tolist());m.update();save(Path(label)/'START.npz',X=vals);write(Path(label)/'START_AUDIT.json',audit)
   info,rr=optimize(m,label,label,900,wall_end);outcomes.append(info);m.dispose()
  queue=read(OUT/label/'CANDIDATES.json');auditq=[]
  for i,c in enumerate(queue):
   check()
   if time.monotonic()>=wall_end-2:break
   z=np.load(OUT/label/c['file']);x=z['x'];rho=float(z['rho']);ss=bank.separate(x,rho);row=store_row(bank,ss,ss['order'][0],label,x,rho)
   pa=B.row_audit(Af,af['rhs'],af['sense'],af['lo'],af['hi'],x);integ=float(np.max(abs(x[af['vtype']=='B']-np.rint(x[af['vtype']=='B']))))
   auditq.append(dict(c,first_stage=pa,integrality=integ,H_upper=ss['H_upper'],rho=rho,cache_status='CACHE_CLOSED' if ss['closed'] else 'PENDING_CACHE_CUT',full400_evaluation='EXACT_HASH_HIT' if hashlib.sha256(x.tobytes()).hexdigest() in [e['ce']['point_sha'] for e in evs] else 'NOT_RUN'))
   if i==0:save(Path(label)/'FINAL_PENDING_ROW.npz',alpha=row['alpha'],beta=row['beta']);write(Path(label)/'FINAL_PENDING_ROW_PROOF.json',row['certificate'])
   if time.monotonic()>=wall_end-2:break
  auditq.sort(key=lambda r:(r['source']!='FINAL_NATIVE',-(r['H_upper']-r['rho']),r['objective'],r['sha']));write(Path(label)/'CANDIDATE_AUDIT.json',auditq)
  info['branch_wall']=time.monotonic()-start;write(Path(label)/'BRANCH_RESULT.json',info)
  if info['branch_wall']>960:write(Path(label)/'BRANCH_WALL_OVERAGE.json',dict(actual=info['branch_wall'],classification='BUDGET_OR_RESOURCE_NOT_COMPARABLE'))
 lbs=[v['bound'] for v in outcomes if v.get('bound') is not None];newLB=max(lbs+[bc['LB']]);valid=max(historical_lb,newLB)
 ok=all(v['branch_wall']<=960 for v in outcomes)
 conclusion='BUDGET_OR_RESOURCE_NOT_COMPARABLE' if not ok else 'SINGLE_PAIR_LOWER_APPROXIMATION_IMPROVED' if all(v.get('bound') is not None for v in outcomes) and outcomes[1]['bound']>=outcomes[0]['bound']+1 else 'NO_FIXED_MODEL_BOUND_GAIN'
 return dict(comparison,status=conclusion,MIP=outcomes,strongest_new_LB=newLB,effective_LB=valid,new_external_LB_progress=newLB>historical_lb+1,external_audited_absolute_gap=ub-valid,external_audited_relative_gap=(ub-valid)/max(1,abs(ub)),new_full400_UB=False,new_full400_calls=0)
def restore_committed_bank():
 root=Path(CFG['reuse_preparation']);prev=Path(CFG['committed_preparation'])
 manifest=read(prev/'SOURCE_BANK_MANIFEST.json');assert manifest['full'] and manifest['source_records']==12032
 # Certificate replay already committed. Verify the actual derived payload used now,
 # not all unused historic primal files a second time.
 refs={r['path']:r for r in manifest['dependencies']}
 for r in read(root/'CODE_SHA.json')+read(prev/'CODE_SHA.json'):verify(r)
 write('REUSED_PROOF_CLOSURE.json',dict(source=ref(prev/'SOURCE_BANK_MANIFEST.json'),previous_wait=read(prev/'WAIT.json'),complete_proof_replay_preserved=True,old_primal_files_not_rehashed=True))
 f=read(SRC/'FIRST_IDENTITY.json');Af=sp.load_npz(verify(f['files'][0])).tocsr();af=dict(np.load(verify(f['files'][1])));names=read(verify(f['files'][2]))
 zeros=set(read(SRC/'SCENARIO_EQUIVALENCE.json')['zeros']);blocks=[]
 for sid in range(400):
  check(True)
  if sid in zeros:blocks.append(dict(B=sp.csr_matrix((0,60780)),alpha=np.array([]),ids=[]));continue
  d=root/'BANK'/f'{sid:04d}'
  for fn in ['B.npz','ALPHA.npz','IDS.json','SOURCE_RECORDS.json','DOMAIN_PROOF.json']:verify(refs[str(d/fn)])
  blocks.append(dict(B=sp.load_npz(d/'B.npz').tocsr(),alpha=np.load(d/'ALPHA.npz')['alpha'],ids=read(d/'IDS.json')))
 cp=read(SRC/'CHECKPOINT.json');evs=[]
 for item in cp['cuts']:
  ce=read(verify(item['evaluation']));recs=read(verify(ce['records']));q=dict(np.load(verify(ce['Q'])));pc=np.load(verify(ce['risk']['certificate']));evs.append(dict(ce=ce,records=recs,q=q,p=pc['pstar'],pc=ce['risk']['certificate']))
 checks=read(prev/'PROBABILITY_REPLAY.json');assert len(checks)==47 and all(c['check']['passed'] for c in checks)
 probs=np.load(prev/'BANK/PROBABILITIES.npz')['p'];pids=read(prev/'BANK/PIDS.json')
 assert all(hashlib.sha256(p.tobytes()).hexdigest()==h for p,h in zip(probs,pids))
 save('BANK/PROBABILITIES.npz',p=probs);write('BANK/PIDS.json',pids);write('PROBABILITY_REPLAY.json',checks)
 for fn in ['SUPPORT_MANIFEST.json','NONNEGATIVITY_AND_BOUNDS.json']:write(fn,read(prev/fn))
 manifest['reused_proof_closure']=ref(prev/'SOURCE_BANK_MANIFEST.json');write('SOURCE_BANK_MANIFEST.json',manifest)
 return Bank(blocks,probs,pids,af['lo'],af['hi']),evs,cp,Af,af,names

# Integrated execution repairs. This file is appended to the isolated source runner.
from repair_guard_helpers import (save_npz_immutable, safe_native_observations,
                                 read_optional_numeric, certificate_record, checked_lp_gate)
PREP_WARNING=False
SIMULATED_NATIVE=False
def save(path, /, **arrays):
 return ref(save_npz_immutable(OUT,path,**arrays))
def check(prep=False):
 global PREP_WARNING
 now=time.monotonic()
 if prep and now-RUN['start_monotonic']>=900 and not PREP_WARNING:
  write('PREPARATION_TIME_WARNING.json',dict(elapsed=now-RUN['start_monotonic'],fatal=False));PREP_WARNING=True
 if now>=RUN['deadline_monotonic']-120:raise TimeoutError('GLOBAL_CLEANUP_RESERVE')
 if (OUT/'USER_STOP.json').exists():raise TimeoutError('USER_STOP')
def captured_error(exc):return dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
def safe_cleanup(obj,folder):
 try:obj.dispose();return None
 except Exception as exc:
  err=captured_error(exc);write(Path(folder)/'CLEANUP_ERROR.json',err);return err
def optimize(m,kind,folder,seconds,wall_end,test=False):
 assert kind in ['TEST_SYNTHETIC','A_LP','B_LP','A_MIP','B_MIP']
 check();folder=Path(folder);(OUT/folder).mkdir(parents=True,exist_ok=True)
 rem=min(seconds,wall_end-time.monotonic(),RUN['deadline_monotonic']-120-time.monotonic())
 if rem<=0:raise TimeoutError('STAGE_BUDGET')
 m.Params.TimeLimit=rem;m.Params.OutputFlag=1;m.Params.LogFile=str(OUT/folder/'NATIVE.log')
 ps={k:read_optional_numeric(lambda n=k:getattr(m.Params,n),infinity=gp.GRB.INFINITY) for k in ['Threads','Method','NodeMethod','PreSparsify','MIPFocus','Seed','FeasibilityTol','OptimalityTol','IntFeasTol','SoftMemLimit','MIPGap','MIPGapAbs','TimeLimit','Cutoff','SolutionLimit']}
 write(folder/'PARAMETERS.json',ps)
 cid=uuid.uuid4().hex;start=time.monotonic();entry=dict(id=cid,kind=kind,scope='TEST' if test else 'SCIENCE',simulated_native=SIMULATED_NATIVE,phase='BEGIN',epoch=time.time(),monotonic=start,folder=str(OUT/folder))
 write(folder/'BEGIN.json',entry);event(entry);queue={};last=0;cb_errors=[];native_error=None
 def callback(mm,where):
  nonlocal last
  try:
   now=time.monotonic()
   if now>=wall_end or now>=RUN['deadline_monotonic']-120 or (OUT/'USER_STOP.json').exists():mm.terminate();return
   if where==gp.GRB.Callback.MIPSOL:
    v=np.asarray(mm.cbGetSolution(mm.getVars()));h=hashlib.sha256(v.tobytes()).hexdigest();queue[h]=(v,float(mm.cbGet(gp.GRB.Callback.MIPSOL_OBJ)))
   if now-last>10:
    rec=dict(kind=kind,epoch=time.time(),runtime=read_optional_numeric(lambda:mm.cbGet(gp.GRB.Callback.RUNTIME),infinity=gp.GRB.INFINITY))
    if where==gp.GRB.Callback.MIP:rec['bound']=read_optional_numeric(lambda:mm.cbGet(gp.GRB.Callback.MIP_OBJBND),infinity=gp.GRB.INFINITY)
    write('NATIVE_PROGRESS.json',rec);last=now
  except Exception as exc:
   cb_errors.append(captured_error(exc));mm.terminate()
 try:native_optimize(m,callback)
 except Exception as exc:native_error=captured_error(exc)
 obs=safe_native_observations(m,infinity=gp.GRB.INFINITY)
 value=lambda k:obs.get(k,{}).get('value')
 bound=value('ObjBoundC')
 info=dict(entry,phase='END',epoch=time.time(),wall=time.monotonic()-start,status=value('Status'),runtime=value('Runtime'),work=value('Work'),max_mem_GB=value('MaxMemUsed'),sol_count=value('SolCount'),observations=obs,native_error=native_error,callback_errors=cb_errors,bound=None if bound is None else bound-1e-6,native_objective=value('ObjVal'))
 # Small END survives unavailable attributes or subsequent large-payload failure.
 write(folder/'END.json',info);event(info)
 raw={};unavailable={}
 try:
  attrs=['X'] if value('IsMIP')==1 else ['X','Pi','RC','VBasis','CBasis']
  for a in attrs:
   try:arr=np.asarray(m.getAttr(a))
   except (gp.GurobiError,AttributeError) as exc:unavailable[a]=dict(type=type(exc).__name__,message=str(exc));continue
   expected=m.NumConstrs if a in ['Pi','CBasis'] else m.NumVars
   if arr.shape!=(expected,) or not np.isfinite(arr).all():raise ValueError('INVALID_NATIVE_ARRAY_'+a)
   raw[a]=arr
  write(folder/'ARRAY_AVAILABILITY.json',unavailable)
  if raw:save(folder/'RAW_NATIVE.npz',**raw)
  if value('IsMIP')==1:
   candidates=[]
   if 'X'in raw:
    v=raw['X'];h=hashlib.sha256(v.tobytes()).hexdigest();queue.pop(h,None);candidates.append(('FINAL_NATIVE',h,v,value('ObjVal')))
   for h,(v,obj) in sorted(queue.items(),key=lambda kv:(kv[1][1],kv[0])):candidates.append(('CALLBACK',h,v,obj))
   for i,(k,h,v,obj) in enumerate(candidates):save(folder/f'CANDIDATE_{i:04d}.npz',x=v[:-1],rho=v[-1],objective=obj)
   write(folder/'CANDIDATES.json',[dict(source=k,sha=h,objective=o,file=f'CANDIDATE_{i:04d}.npz') for i,(k,h,v,o) in enumerate(candidates)])
 except Exception as exc:
  write(folder/'EVIDENCE_WRITE_FAILED.json',captured_error(exc));raise
 if native_error or cb_errors:raise RuntimeError('NATIVE_OR_CALLBACK_FAILED: see '+str(folder/'END.json'))
 return info,raw
_original_lp_certificate=lp_certificate
def lp_certificate(m,raw,folder):
 folder=Path(folder);ob=safe_native_observations(m,infinity=gp.GRB.INFINITY);ns=ob['Status']['value']
 rec=None
 try:
  old=_original_lp_certificate(m,raw,folder)
  identity=read(OUT/folder/'IDENTITY.json');lb=old.get('LB')
  rec=certificate_record(native_status=ns,model_id=identity['matrix_sha'],status='NO_PI' if 'Pi'not in raw else 'CERTIFIED' if ns==gp.GRB.OPTIMAL else 'PARTIAL',lower=lb,lower_verified=lb is not None and math.isfinite(lb),native_objective=ob['ObjVal']['value'],numerical_gate=bool(old.get('numerical_gate',False) and 'Pi'in raw),primal_audit=old.get('primal_audit'),reason=None if 'Pi'in raw else 'Pi unavailable; no certified lower')
  rec.update(identity=identity,matrix_ref=ref(OUT/folder/'A.npz'),raw_solution_ref=ref(OUT/folder.parent/'RAW_NATIVE.npz') if (OUT/folder.parent/'RAW_NATIVE.npz').exists() else None,dual_ref=ref(OUT/folder/'CERTIFIED_PI.npz') if (OUT/folder/'CERTIFIED_PI.npz').exists() else None,rho_rc_lower=old.get('rho_rc_lower'))
 except Exception as exc:
  ident=read(OUT/folder/'IDENTITY.json') if (OUT/folder/'IDENTITY.json').exists() else {'matrix_sha':'SNAPSHOT_FAILED'}
  rec=certificate_record(native_status=ns,model_id=ident['matrix_sha'],status='CERTIFICATE_ERROR',reason=str(exc));rec['error_details']=[captured_error(exc)];write(folder/'LOWER_CERTIFICATE.json',rec);raise
 write(folder/'LOWER_CERTIFICATE.json',rec)
 if rec['lower_certificate_verified']:write('LAST_SOLVED.json',dict(directory=str(OUT/folder),identity=rec['identity'],LB=rec['LB'],certificate=ref(OUT/folder/'LOWER_CERTIFICATE.json')))
 return rec
def restore_committed_bank():
 root=Path(CFG['reuse_preparation']);prev=Path(CFG['committed_preparation'])
 # The previous sealed archive is an external immutable anchor, not a freshly made ref.
 anchor=root/'delivery/DRO_CACHE_ENVELOPE_PRO_ANALYSIS_01.zip'
 assert sha(anchor)=='3aa0bf607eb64bd26ff9e5215ab630ccc35d86cd558b88bb16e843fe13194697'
 import zipfile
 with zipfile.ZipFile(anchor) as z:sealed=json.loads(z.read('PAYLOAD_MANIFEST.json'))
 historical_refs={e['source_path']:dict(path=e['source_path'],bytes=e['bytes'],sha256=e['sha256']) for e in sealed['members']}
 for p in [prev/'SOURCE_BANK_MANIFEST.json',prev/'PROBABILITY_REPLAY.json',prev/'BANK/PROBABILITIES.npz',prev/'BANK/PIDS.json',root/'SUPPORT_MANIFEST.json',root/'NONNEGATIVITY_AND_BOUNDS.json']:verify(historical_refs[str(p)])
 manifest=read(prev/'SOURCE_BANK_MANIFEST.json');assert (manifest['source_records'],manifest['unique_supports'],manifest['evaluation_points'],manifest['unique_probabilities'],manifest['universal_zero'])==(12032,9016,47,47,144)
 assert manifest['source_science']==SCIENCE and manifest['P_identity']==PID
 refs={e['path']:e for e in manifest['dependencies']}
 for fn in ['FIRST_IDENTITY.json','CHECKPOINT.json','FINAL_MASTER.mps.gz','CONFIG.json','SCENARIO_EQUIVALENCE.json']:verify(refs[str(SRC/fn)])
 f=read(SRC/'FIRST_IDENTITY.json');Af=sp.load_npz(verify(f['files'][0])).tocsr();af=dict(np.load(verify(f['files'][1])));names=read(verify(f['files'][2]))
 assert Af.shape==(90768,60780) and sum(af['vtype']=='B')==25770
 zeros=set(read(SRC/'SCENARIO_EQUIVALENCE.json')['zeros']);assert len(zeros)==144
 support=read(root/'SUPPORT_MANIFEST.json');assert len(support)==12032
 bysid={s:[] for s in range(400)}
 for e in support:bysid[e['sid']].append(e)
 bounds=read(root/'NONNEGATIVITY_AND_BOUNDS.json');assert len(bounds)==400
 blocks=[];closure=[]
 for sid in range(400):
  check(True);b=bounds[sid];assert b['sid']==sid and b['nonnegative_objective'] and b['finite_box_replayed']
  verify(b['proof']);verify(b['arrays'])
  if sid in zeros:
   assert not bysid[sid];blocks.append(dict(B=sp.csr_matrix((0,60780)),alpha=np.array([]),ids=[]));continue
  d=root/'BANK'/f'{sid:04d}'
  for fn in ['B.npz','ALPHA.npz','IDS.json','SOURCE_RECORDS.json','DOMAIN_PROOF.json']:verify(refs[str(d/fn)])
  bb=sp.load_npz(d/'B.npz').tocsr();al=np.load(d/'ALPHA.npz')['alpha'];ids=read(d/'IDS.json');rr=read(d/'SOURCE_RECORDS.json')
  assert rr==bysid[sid] and len(rr)==47 and all(e['replay_exact_match'] for e in rr)
  assert set(ids)==set(e['support_id'] for e in rr) and len(ids)==len(set(ids))
  for i,h in enumerate(ids):assert digest(float(al[i]),bb.getrow(i).toarray().ravel())==h
  blocks.append(dict(B=bb,alpha=al,ids=ids));closure.append(dict(sid=sid,source_records=47,unique=len(ids),array_support_id_match=True))
 assert sum(len(b['ids']) for b in blocks)==9016
 cp=read(SRC/'CHECKPOINT.json');evs=[]
 for item in cp['cuts']:
  check(True);ce=read(verify(item['evaluation']));recs=read(verify(ce['records']));q=dict(np.load(verify(ce['Q'])));pc=np.load(verify(ce['risk']['certificate']));p=pc['pstar']
  assert ce['science']==SCIENCE and ce['P_identity']==PID and ce['full400'] and hashlib.sha256(q['x'].tobytes()).hexdigest()==ce['point_sha']
  assert [e['sid'] for e in recs]==list(range(400))
  evs.append(dict(ce=ce,records=recs,q=q,p=p,pc=ce['risk']['certificate']))
 assert len(evs)==47 and len({e['ce']['point_sha'] for e in evs})==47
 checks=read(prev/'PROBABILITY_REPLAY.json');probs=np.load(prev/'BANK/PROBABILITIES.npz')['p'];pids=read(prev/'BANK/PIDS.json');assert probs.shape==(47,400) and len(set(pids))==47
 for p,h,ck in zip(probs,pids,checks):
  assert hashlib.sha256(p.tobytes()).hexdigest()==h==ck['p_id'] and ck['check']['passed'];verify(ck['source'])
  assert np.array_equal(np.load(ck['source']['path'])['pstar'],p)
 assert {hashlib.sha256(e['p'].tobytes()).hexdigest() for e in evs}==set(pids)
 # Original zero proof and original P input dependencies were in the committed closure.
 for path,r in refs.items():
  if any(k in path for k in ['UNIVERSAL_ZERO_PROOFS','CALIBRATED_RAW_BOUNDS','FEATURE_REGISTRY','EXACT_COMPRESSION']):verify(r)
 save('BANK/PROBABILITIES.npz',p=probs);write('BANK/PIDS.json',pids);write('PROBABILITY_REPLAY.json',checks)
 write('SOURCE_BANK_MANIFEST.json',dict(manifest,reused_proof_closure=historical_refs[str(prev/'SOURCE_BANK_MANIFEST.json')]))
 write('RESTORE_COMMITTED_BANK.json',dict(passed=True,source_records=12032,unique_supports=9016,per_sid=closure,old_Q_P_optimizer_calls=0,old_support_math_replays=0,proof_closure_anchor_sha=sha(anchor),verification='committed historical SHA plus full per-derived-row support digest/source mapping'))
 global SUPPORT_BY_BANK
 SUPPORT_BY_BANK={(e['source_bank_id'],e['sid']):e['support_id'] for e in support}
 return Bank(blocks,probs,pids,af['lo'],af['hi']),evs,cp,Af,af,names
def subset_bank(bank,support_banks,probability_evs):
 blocks=[]
 for sid,b in enumerate(bank.blocks):
  wanted={SUPPORT_BY_BANK[(h,sid)] for h in support_banks if (h,sid) in SUPPORT_BY_BANK}
  idx=[i for i,h in enumerate(b['ids']) if h in wanted]
  blocks.append(dict(B=b['B'][idx].copy(),alpha=b['alpha'][idx].copy(),ids=[b['ids'][i] for i in idx]))
 return Bank(blocks,[e['p'].copy() for e in probability_evs],[hashlib.sha256(e['p'].tobytes()).hexdigest() for e in probability_evs],bank.lo,bank.hi)
def witness(bank,evs):
 x=np.load(SRC/'native/000046/RAW_NATIVE.npz')['x'];ids=['951cc92fb17df719e6a1d224095b39d0e51a23d47a2b56289b2a84cbf1eb2f91','382f9baee6f53953560cad1b9f3f924835f7a4102d0f6f2cb43e6d888a2df324'];chosen=[next(e for e in evs if e['ce']['point_sha']==h) for h in ids]
 results=[]
 for label,support_ids,ps,target in [('A',ids[:1],chosen[:1],15705.57734620686),('B',ids,chosen,20910.256125811873),('C',ids,evs,21192.44916106056)]:
  small=subset_bank(bank,support_ids,ps);sep=small.separate(x,0);assert abs(sep['H_upper']-target)<=1e-5
  winner=small.pids[sep['order'][0]]
  if label=='C':assert winner=='72d5599c7014ed8335d098a47c05b89c1cc70a65fd157a0bd315f3df960528d7'
  results.append(dict(label=label,support_bank_count=len(support_ids),probability_count=len(ps),target=target,winner=winner,**sep))
 full=bank.separate(x,0);assert full['H_upper']>=results[-1]['H_lower']
 write('WITNESS_REGRESSION.json',dict(passed=True,fixtures=results,full47=full,point_values_not_global_bounds=True))
 return full
_old_add_row=add_row
def add_row(m,bank,row,folder):
 co=_old_add_row(m,bank,row,folder);stored=np.load(OUT/folder/'ROW.npz');actual=digest(float(stored['alpha']),stored['beta'])
 selection=hashlib.sha256(json.dumps([row['certificate']['p_id'],row['certificate']['support_ids']],separators=(',',':')).encode()).hexdigest()
 proof=read(OUT/folder/'PROOF.json');proof.update(selection_id=selection,generated_id=row['certificate']['row_id'],actual_stored_row_id=actual)
 write(folder/'PROOF.json',proof);row['actual_id']=actual;return co

def integration_tests():
 global native_optimize,SIMULATED_NATIVE
 from types import SimpleNamespace
 root=Path('TESTS')/('SESSION_'+uuid.uuid4().hex);records=[]
 def ok(n,v=True):
  assert v,n
  records.append(dict(test=n,passed=True))
 # Same deployed save and filesystem publication.
 for i in range(2):
  path=root/str(i)/'arrays.npz';save(path,p=np.array([.25,.75]),path=np.array([7]));z=np.load(OUT/path);ok('session_'+str(i),np.array_equal(z['p'],[.25,.75]) and z['path'][0]==7)
 try:save(root/'0/arrays.npz',p=np.array([9]));raise AssertionError('overwrite accepted')
 except FileExistsError:ok('immutable_conflict')
 try:save('../forbidden.npz',p=np.array([1]));raise AssertionError('escape accepted')
 except ValueError:ok('path_escape')
 link=OUT/root/'old_link';link.symlink_to(Path(CFG['reuse_preparation']),target_is_directory=True)
 try:save(root/'old_link/forbidden.npz',p=np.array([1]));raise AssertionError('symlink accepted')
 except ValueError:ok('symlink_escape')
 class Mock:
  def __init__(self,status=9):
   self.Params=SimpleNamespace(**{k:v for k,v in dict(Threads=1,Method=1,NodeMethod=1,PreSparsify=1,MIPFocus=0,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,IntFeasTol=1e-9,SoftMemLimit=8,MIPGap=.001,MIPGapAbs=1,Cutoff=gp.GRB.INFINITY,SolutionLimit=2000000000).items()})
   self.Status=status;self.IsMIP=1;self.SolCount=0;self.Runtime=.01;self.Work=0;self.MaxMemUsed=0;self.NumVars=2;self.NumConstrs=1;self.NodeCount=0
  def getAttr(self,a):raise AttributeError(a+' unavailable')
  @property
  def ObjBoundC(self):raise AttributeError('bound unavailable')
  def dispose(self):raise RuntimeError('injected cleanup error')
 real_native=native_optimize;SIMULATED_NATIVE=True
 try:
  for st in [gp.GRB.TIME_LIMIT,gp.GRB.MEM_LIMIT]:
   native_optimize=lambda m,cb:None
   info,raw=optimize(Mock(st),'TEST_SYNTHETIC',root/('MOCK_'+str(st)),1,time.monotonic()+10,True)
   ok('optional_native_attrs_'+str(st),info['status']==st and info['bound'] is None)
  def throw(m,cb):raise RuntimeError('injected native failure')
  native_optimize=throw
  try:optimize(Mock(),'TEST_SYNTHETIC',root/'MOCK_THROW',1,time.monotonic()+10,True);raise AssertionError('native exception swallowed')
  except RuntimeError:ok('native_exception_END_preserved',(OUT/root/'MOCK_THROW/END.json').exists())
 finally:native_optimize=real_native;SIMULATED_NATIVE=False
 ok('cleanup_error_recorded',safe_cleanup(Mock(),root/'MOCK_CLEANUP') is not None)
 ok('nan_classification',read_optional_numeric(lambda:float('nan'),infinity=gp.GRB.INFINITY)['classification']=='NAN')
 # Actual matrix snapshot/certificate function, no optimize. Pi supplied as a synthetic fixture.
 with gp.Env(empty=True) as env:
  env.setParam('OutputFlag',0);env.start();m=gp.Model(env=env);x=m.addVar(lb=0,ub=1,obj=1);rho=m.addVar(lb=0,ub=gp.GRB.INFINITY,obj=1);m.addConstr(x>=0);m.update()
  no=lp_certificate(m,{},root/'NO_PI/SOLVED');ok('NO_PI_total_schema',no['LB'] is None and not no['numerical_gate'] and no['matrix_ref'])
  part=lp_certificate(m,{'Pi':np.array([0.])},root/'NO_X/SOLVED');ok('NO_X_partial_dual',part['lower_certificate_verified'] and part['status']=='PARTIAL' and not part['numerical_gate'])
  ok('no_PI_no_MIP',not checked_lp_gate(no,part)['run_mip'])
  m.dispose()
 write(root/'RESULT.json',dict(passed=True,tests=records,real_optimizer_calls=0,mock_optimize_wrapper_calls=3,scope='same deployed I/O, native finishing, LP certificate, cleanup and gate functions'))
 write('TESTS/INTEGRATION_RESULT.json',dict(passed=True,session=str(root),tests=len(records),old_17_tests_reused=True,old_math_kernel_unchanged=True,source=ref(Path(CFG['reuse_preparation'])/'TESTS/RESULT.json')))


def main():
 global OUT,SRC,CFG,RUN,B
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--science-source',required=True);p.add_argument('--bank-source',required=True);p.add_argument('--committed-probability-source',required=True);p.add_argument('--execution-root',required=True);p.add_argument('--config',required=True);p.add_argument('--start-or-resume',action='store_true');a=p.parse_args()
 OUT=Path(a.execution_root).resolve();SRC=Path(a.science_source).resolve();CFG=read(a.config);RUN=read(OUT/'RUN.json')
 assert a.start_or_resume and CFG['science']==SCIENCE and CFG['P_identity']==PID and CFG['source_root']==str(SRC)
 assert str(Path(a.bank_source).resolve())==CFG['reuse_preparation'] and str(Path(a.committed_probability_source).resolve())==CFG['committed_preparation']
 assert RUN['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and RUN['deadline_monotonic']==RUN['start_monotonic']+5400
 assert RUN['run_id']!=read(Path(a.bank_source)/'RUN.json')['run_id']
 assert gp.gurobi.version()==(12,0,3);write('EFFECTIVE_CONFIG.json',CFG);write('INPUT_PATHS.json',vars(a))
 sys.path.insert(0,str(SRC/'tools/lifecycle'));import base_r44;B=base_r44
 for name in ['OUT','ROOT','WORK','OUTPUT']: 
  if hasattr(B,name):setattr(B,name,OUT/'ISOLATED_IMPORTED_GLOBALS')
 gp.Model.optimize=blocked_optimize
 start=time.monotonic();rc=0;result={'status':'INITIALIZING'}
 try:result=scientific()
 except BaseException as exc:
  rc=2;result=dict(status=str(exc) if isinstance(exc,TimeoutError) else 'VALIDATION_OR_EXECUTION_BLOCKED',phase=PHASE,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),remaining_stages='NOT_RUN')
 finally:
  write('DEPENDENCIES_VERIFIED.json',list(DEPS.values()))
  result.update(wall=time.monotonic()-RUN['start_monotonic'],child_wall=time.monotonic()-start,production_signals_sent=[],no_recourse_or_probability_calls=True,branch_tree_restored=False)
  write('FINAL_RESULT.json',result);write('END.json',dict(epoch=time.time(),returncode=rc,status=result['status']))
 return rc
if __name__=='__main__':raise SystemExit(main())
