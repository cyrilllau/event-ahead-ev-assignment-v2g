from aro_core import *
import copy,csv,math
from aro_state_policy import next_action,candidate_transition
def exit_code(status):
 if status in ('RESOURCE_RECOVERY_NEEDED','WAIT_RESOURCE'):return 75
 if status in ('GLOBAL_WALL_LIMIT','USER_REQUESTED_STOP','NUMERICAL_TARGET_REACHED_CERTIFICATION_PENDING','ARO400_SOLVER_GAP_TARGET_REACHED','NO_PROGRESS_DIAGNOSTIC','EVIDENCE_REFINEMENT_REQUIRED'):return 0
 return 70
def check_external_gap(state,**kw):
 return next_action(lower=state.get('best_LB'),upper_audited=state.get('best_UB_audited'),has_incumbent=kw.get('has_incumbent',False),cached_evaluation=kw.get('cached_evaluation',False),cache_separated_for_current_master=kw.get('separated',False),native_optimal=kw.get('native_optimal',False))
def commit_state(state,fail_at=None):
 value=copy.deepcopy(state);folder=ROOT/'REPAIR_JOURNAL';folder.mkdir(exist_ok=True);key=d.digest(value);p=folder/(key+'.json');write(p,value,True)
 if fail_at=='before':raise RuntimeError('TEST_BEFORE_HEAD')
 write(ROOT/'REPAIR_STATE.json',{'state':ref(p)});actual=read(rt.verify_ref(read(ROOT/'REPAIR_STATE.json')['state']))
 if actual!=value:raise ValueError('STATE_COMMIT_READBACK')
 if fail_at=='after':raise RuntimeError('TEST_AFTER_HEAD')
 return actual
def read_state():
 return read(rt.verify_ref(read(ROOT/'REPAIR_STATE.json')['state'])) if (ROOT/'REPAIR_STATE.json').exists() else {'phase':'PREFLIGHT','outer':0,'best_LB':None,'best_UB_audited':None,'best_UB_certified':None,'best_plan':None,'cuts':[],'history':[],'processed':{}}
def domain_check(m,A,a,names,rows,objective=False):
 checks={'matrix':(m.getA()-A).nnz==0,'shape':m.getA().shape==A.shape,'rows':m.getAttr('ConstrName')==rows,'columns':m.getAttr('VarName')==names}
 for k,attr in [('rhs','RHS'),('sense','Sense'),('lo','LB'),('hi','UB'),('vtype','VType')]:checks[k]=np.array_equal(m.getAttr(attr),a[k])
 if objective:checks.update(cost=np.array_equal(m.getAttr('Obj'),a['cost']),ObjCon=m.ObjCon==0,ModelSense=m.ModelSense==1)
 if not all(checks.values()):raise ValueError('SOURCE_GATE_FAILED '+repr(checks))
 return checks
def source_gates():
 gates=REV/'gates';gates.mkdir(exist_ok=True);A,a,names=first();rows=read(R41/'identity/FIRST_ROWS.json');assert A.shape==(90768,60780) and A.nnz==368760
 if not (gates/'DOMAIN.json').exists():
  m=gp.read(str(R36/'COMMON_FIRST_STAGE.mps.gz'))
  try:checks=domain_check(m,A,a,names,rows);assert not np.any(m.getAttr('Obj'));write(gates/'DOMAIN.json',{'passed':True,'checks':checks,'source':ref(R36/'COMMON_FIRST_STAGE.mps.gz'),'zero_objective_template_domain_only':True},True)
  finally:m.dispose()
 if not (gates/'OBJECTIVE.json').exists():
  m=gp.Model();m.Params.OutputFlag=0
  try:
   fs=orig.add_first(m,orig.routes(),30,8,orig.full_bounds());orig.add_strengthening(m,fs);m.setObjective(fs['objective']);m.update();checks=domain_check(m,A,a,names,rows,True);m.getVars()[0].Obj=1;m.update()
   try:domain_check(m,A,a,names,rows,True);raise AssertionError('COST_CORRUPTION_ACCEPTED')
   except ValueError:pass
  finally:m.dispose()
  m=gp.read(str(R41/'identity/FIRST_INTEGER_BLOCK.mps.gz'))
  try:checks2=domain_check(m,A,a,names,rows,True)
  finally:m.dispose()
  groups={}
  for n,c in zip(names,a['cost']):
   if c!=0:groups.setdefault(n.split('[')[0],[]).append(float(c))
  assert {k:len(v) for k,v in groups.items()}==dict(route=19860,selected=30,K=30,J=30,E=30)
  for key,value in [('selected',25),('K',-1),('J',31),('E',.02)]:assert all(v==value for v in groups[key])
  write(gates/'OBJECTIVE.json',{'passed':True,'rebuilt_checks':checks,'integer_block_checks':checks2,'cost_groups':{k:{'count':len(v),'min':min(v),'max':max(v)} for k,v in groups.items()},'cost_tamper_rejected':True,'optimizer_calls':0},True)
 refs=[];support=[];selected=list(csv.DictReader((R36/'02_SOURCE_AND_CONFIG/SELECTED400.csv').open()));mapping=list(csv.DictReader((R36/'02_SOURCE_AND_CONFIG/POOL_TO_SELECTED.csv').open()));assert len(selected)==400 and len(mapping)==2000
 for sid,row in enumerate(selected):
  assert int(row['local_sid'])==sid;scene=orig.scene_path('TRAIN',sid);meta=read(scene/'PRIMITIVES_AND_IDENTITY.json');assert meta['source_sid']==int(row['source_sid']) and meta['bank']==row['bank']=='R27_POOL2000_K4L4_V1'
  refs.extend(ref(p) for p in (scene/'physical.npz',scene/'recovery.npz',scene/'PRIMITIVES_AND_IDENTITY.json',orig.verify_replay('TRAIN',sid)));support.append({'sid':sid,'source_sid':meta['source_sid'],'bank':meta['bank']})
 for row in mapping:assert int(selected[int(row['local_sid'])]['source_sid'])==int(row['selected_source_sid'])
 for name in ('SELECTED400.csv','POOL_TO_SELECTED.csv'):refs.append(ref(R36/'02_SOURCE_AND_CONFIG'/name))
 write(gates/'SUPPORT.json',{'passed':True,'support':support,'physical_recovery_replay_refs':refs},True)
 old=read(R36/'evaluations/7cc27e720e13d3cbe46a1a25a9e704f9495bbcf045cf2df212cb702bd24fdf27/COMPLETE.json');archived=read(R41/'LOCK.json')['best'];assert archived['plan_hash']==old['plan_hash'];q=np.array(archived['qU']);assert len(q)==400 and q.argmax()==385 and abs(old['F']+q.max()-693189.0272067727)<1e-7;write(gates/'ARCHIVED_ARITHMETIC.json',{'passed':True,'F':old['F'],'max_qU':float(q.max()),'F_plus_max':old['F']+float(q.max()),'not_inherited_as_current_UB':True,'qU_source':ref(R41/'LOCK.json')},True)
 refs.extend(ref(p) for p in (R41/'identity').glob('FIRST*') if p.is_file());refs.extend(ref(p) for p in (R36/'tools').glob('*.py'));lock={'files':refs,'gates':[ref(p) for p in gates.glob('*.json')],'science':'ARO400_FINITE_SUPPORT_FULL_INTEGER','route_policy':'FORECAST_SELECTED_FIXED_ROUTE_REPLAY'};write(ROOT/'SOURCE_LOCK.json',lock,True)
def verify_source_lock():
 lock=read(ROOT/'SOURCE_LOCK.json')
 for r in lock['files']+lock['gates']:rt.verify_ref(r)
 return sha(ROOT/'SOURCE_LOCK.json')
def evaluate(x,point_id,attempt,science,run,deadline,publish):
 folder=ROOT/'evaluations'/point_id;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETE.json').exists():records=read(folder/'COMPLETE.json')['records']
 else:
  session=attempt/'worker_sessions'/point_id/uuid.uuid4().hex;session.mkdir(parents=True);workers=e.Workers(ROOT,run,deadline,session,oracle_cls=AROOracle)
  try:records=workers.evaluate(x,list(range(400)),science,lambda n:publish('EVALUATE400',completed_sid=n,point_id=point_id))
  finally:workers.close()
 validated=[]
 for r in records:
  task=e.task_id(science,r['sid'],x,'RECOURSE');assert task==r['task_id'];o=AROOracle(r['sid'],ROOT/'VALIDATION',None)
  try:rr=o.validate_result(x,ROOT/'EVALUATION_CACHE'/task/'RESULT.json')
  finally:o.close(discard=True)
  validated.append(dict(rr,scientific_id=science))
 full_bound(0,validated,list(range(400)),science,point_id)
 if not (folder/'COMPLETE.json').exists():write(folder/'COMPLETE.json',{'records':validated,'point_sha':point_id,'scientific_id':science,'full400':True},True)
 return validated
def separate(master,xvar,eta,records,x,eta_value,cuts):
 new=[];seen={c['fingerprint'] for c in cuts}
 for r in sorted(records,key=lambda q:q['sid']):
  z=np.load(rt.verify_ref(ref(r['cut'])));alpha=float(z['alpha']);beta=z['beta'];fp=hashlib.sha256(str(r['sid']).encode()+np.array([alpha]).tobytes()+beta.tobytes()).hexdigest();value=float(np.longdouble(alpha)+np.asarray(beta,np.longdouble)@np.asarray(x,np.longdouble))
  if value>eta_value+1e-6 and fp not in seen:
   name=f'aro_support/{r["sid"]}/{fp}';add_support(master,xvar,eta,alpha,beta,name);new.append({'sid':r['sid'],'cut':ref(r['cut']),'name':name,'fingerprint':fp});seen.add(fp)
 master.update();return new
def strict_certify(state,deadline):
 import strict_upper as strict
 A,a,names=first();x=np.load(rt.verify_ref(state['best_plan']))['accepted'];point=hashlib.sha256(x.tobytes()).hexdigest();out=ROOT/'STRICT_CERTIFICATION'/point;out.mkdir(parents=True,exist_ok=True);firstcheck=strict.check(A,a['rhs'],a['sense'],a['lo'],a['hi'],x,deadline);write(out/'FIRST.json',firstcheck,True)
 if not firstcheck['passed']:return {'passed':False,'reason':'EXACT_FIRST_PRIMAL','evidence':ref(out/'FIRST.json')}
 records=read(ROOT/'evaluations'/point/'COMPLETE.json')['records'];upper=[]
 for r in records:
  sid=r['sid'];p=out/f'{sid:04d}.json'
  if p.exists():check=read(p)
  else:
   o=AROOracle(sid,ROOT/'STRICT_VALIDATION',None)
   try:
    y=np.load(r['primal'])['y'];check=strict.check(o.W,o.a['h'],o.a['sense'],o.a['lo'],o.a['hi'],y,deadline,T=o.T,first=x)
    if check['passed']:check['qU_certified']=strict.cost_upper(o.a['c'],y)
    write(p,check,True)
   finally:o.close(discard=True)
  if not check['passed']:return {'passed':False,'reason':'EXACT_RECOURSE_PRIMAL','sid':sid,'evidence':ref(p)}
  upper.append(check['qU_certified'])
 return {'passed':True,'UB_certified':float(np.nextafter(np.longdouble(strict.cost_upper(a['cost'],x))+max(map(np.longdouble,upper)),np.inf)),'scope':'exact binary64 primal upper + solver-based MIP lower, not rational global branch proof'}
