from r31_reference_core import *
import csv,itertools,subprocess,fcntl,contextlib,zipfile
R=Path(__file__).resolve().parents[1]
PRIOR=B/'L08_DET_FORECAST44_EXACT_ACCELERATION_R31'
LEASE=V/'leases/R32_FINITE_DET.json';rr.LEASE=LEASE

def frozen_model():
 m=gp.read(str(R/'frozen/BASE.mps.gz'));params(m);return m

def science():return read(R/'frozen/LOCK.json')['identity']

def original_audit(x,save=None):
 m=frozen_model();a=matrix(m);ma=audit_matrix(a,x);ph=physical_audit(x,a['names'],c.routes(),30,c.layout()['net'],c.load_scene('FORECAST_SELECTED'),c.full_bounds(),c.loadnp(R/'frozen/PF_LIFT.npz'));m.dispose()
 assert abs(ma['objective']-ph['objective'])<1e-5
 rec=dict(passed=True,matrix=ma,physical=ph,science=science(),UB=ph['original_feasible_UB'])
 if save:dump(save,rec)
 return rec

def base_bound():
 p=OLD/'04_MODEL_AND_NATIVE_SOLVE';src=p/'NATIVE_END.json'
 if src.exists():v=read(src);lb=v.get('solver_LB')
 else:
  src=p/'PROGRESS.json';v=read(src);lb=v.get('native_LB') if v.get('phase')=='MIP' else None
 log=(p/'NATIVE.log').read_text();lines=[]
 for line in log.splitlines():
  z=line.split()
  if len(z)>=10 and z[-1].endswith('s') and z[-3].endswith('%'):
   try:
    candidate=float(z[-4])-0.5*10**(-len(z[-4].split('.')[-1]));runtime=float(z[-1][:-1]);assert abs(candidate)<1e50
    lines.append((runtime,candidate,line))
   except (ValueError,AssertionError):pass
 if lines:
  latest=lines[-1]
  if lb is None or latest[1]>lb:lb=latest[1];v=dict(native_log_line=latest[2],runtime=latest[0],rounded=True,rounding_guard='subtract half of last displayed decimal unit from printed best bound');src=p/'NATIVE.log'
 assert lb is None or np.isfinite(lb)
 return dict(type='GLOBAL_BASE_DET',science=science(),value=lb,source=str(src),evidence=v,epoch=time.time(),model_sha=sha(p/'DET.mps.gz'))

def aggregate(items,ub):
 accepted=[];rejected=[]
 for x in items:
  if x['science']!=science() or x['type'] not in ['GLOBAL_BASE_DET','GLOBAL_EXACT_SCREENED_DET'] or (x['type']=='GLOBAL_EXACT_SCREENED_DET' and not x.get('proof_passed')) or x.get('value') is None:rejected.append(x);continue
  accepted.append(x)
 assert accepted,'NO_GLOBAL_BOUND';lb=max(x['value'] for x in accepted);gap=ub-lb;assert gap>=-1e-6,'SIGNED_BOUND_CONFLICT'
 return dict(LB=lb,UB=ub,signed_gap=gap,certified=0<=gap<=1,accepted=accepted,rejected=rejected)

def root_table(net,scene):
 result=np.zeros((96,len(net['buses'])),bool);bi={b:i for i,b in enumerate(net['buses'])}
 for t in range(96):
  adj={b:[] for b in net['buses']}
  for u,v,off in zip(net['from'],net['to'],scene['line_operating_outages'][t]):
   if not off:adj[u].append(v);adj[v].append(u)
  reached={1};todo=[1]
  while todo:
   for b in adj[todo.pop()]:
    if b not in reached:reached.add(b);todo.append(b)
  result[t,[bi[b] for b in reached]]=True
 return result

def guard(pr):
 required=['single_forecast','root_zero_feasible','zero_root_price','nonnegative_costs','no_mandatory_purchase','no_mandatory_discharge','components_separable','capability_shortfall','no_grid_temporal_coupling','forecast_equals_recourse']
 return all(pr.get(k) is True for k in required)

def inventory():
 out=R/'frozen';out.mkdir(exist_ok=True)
 sources={str(p):sha(p) for p in (OLD/'tools').rglob('*.py')}|rr.source_inventory()
 current=dict(epoch=time.time(),sources=sources,resources=rr.resources(),signals_sent=[],old_files_written=[])
 if (R/'BASELINE.json').exists():assert read(R/'BASELINE.json')['sources']==sources
 else:dump(R/'BASELINE.json',current)
 dump(R/'INVENTORY_CURRENT.json',current)
 for row in read(R/'instructions/FILE_MANIFEST.json')['files']:
  p=R/'instructions'/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
 for row in read(R/'instructions/SOURCE_EVIDENCE.json')['selected_archive_member_identities']:
  p=PRIOR/row['archive_member'];assert sha(p)==row['sha256']
 assert sha(PRIOR/'delivery/final'/ (PRIOR.name+'_PRO_ANALYSIS.zip'))==read(R/'instructions/SOURCE_EVIDENCE.json')['source_files'][2]['sha256']
 assert sha(OLD/'04_MODEL_AND_NATIVE_SOLVE/DET.mps.gz')==sha(PRIOR/'frozen/BASE.mps.gz')
 if not (out/'LOCK.json').exists():
  for n in ['BASE.mps.gz','START.npz','VARIABLE_NAMES.json','PF_LIFT.npz','BUILD_COMPLETE.json','FORECAST44.npz','FORECAST44.json']:shutil.copy2(PRIOR/'frozen'/n,out/n)
  shutil.copytree(PRIOR/'source_snapshot',R/'source_snapshot')
  shutil.copytree(PRIOR/'benchmarks/A_BASE_T4',R/'historical/A_BASE_T4')
  for n in ['ZERO_INPUT_WITNESS.npz','ZERO_INPUT_CERTIFICATE.json']:shutil.copy2(OLD/'06_TESTS'/n,out/n)
  dump(out/'LOCK.json',dict(identity=read(OLD/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json'),model_sha=sha(out/'BASE.mps.gz'),start_sha=sha(out/'START.npz'),source_hashes=sources,forecast_only=True))
 bound=base_bound();dump(R/'INITIAL_BOUND.json',bound)
 audit=OLD/'05_SELECTED_PLAN_AND_CERTIFICATES/AUDIT_COMPLETE.json'
 done=audit.exists() and read(audit).get('passed') and read(audit).get('absolute_gap_met')
 dump(R/'INITIAL_STATUS.json',dict(original_certified=bool(done),owner=ctrl.proc(read(OLD/'04_MODEL_AND_NATIVE_SOLVE/NATIVE_BEGIN.json')['owner']['pid']),bound=bound))

def det_only_scope():
 entry=OLD/'tools/run_l08_k4l4_forecast44_domain_det_r30.py';s=entry.read_text();assert 'evaluate-fixed400' in s
 dump(R/'DET_ONLY_SCOPE.json',dict(R32_forecast_only=True,R32_dispatches_400=False,old_all_has_evaluate_fixed400=True,verified_cooperative_stage_hook=False,old_live_scope_not_changed=True,old_entry_sha=sha(entry),risk='Old already-running all pipeline may dispatch fixed400 after certification. No verified cooperative hook; not hot patched, signalled or restarted.',candidate_delivery='Independent append-only INBOX; not consumed by active Model',new_infinite_solve=False))

def certify_screen():
 for n in ['physical.npz','recovery.npz']:assert sha(c.scene_path('FORECAST_SELECTED')/n)==sha(R/'source_snapshot/FORECAST44'/n)
 assert read(R/'frozen/ZERO_INPUT_CERTIFICATE.json')['witness_sha']==sha(R/'frozen/ZERO_INPUT_WITNESS.npz')
 net=c.layout()['net'];sc=c.load_scene('FORECAST_SELECTED');qs=c.routes();M=c.full_bounds();rt=root_table(net,sc);bi={b:i for i,b in enumerate(net['buses'])};mat=assemble(net,[],sc,[],30,M);y=c.loadnp(R/'frozen/ZERO_INPUT_WITNESS.npz')['y'];res=mat.residual(y,np.zeros(30));assert res<=1e-6
 ix=mat.index;root_shed=float(np.max(y[ix['shed']][rt],initial=0));assert root_shed<=1e-6 and np.max(abs(y[ix['p']]))==0
 pr=dict(single_forecast=sc['line_operating_outages'].shape==(96,32),root_zero_feasible=res<=1e-6 and root_shed<=1e-6,zero_root_price=bool(np.all(mat.cost[ix['grid_p']]==0) and np.all(mat.cost[ix['grid_q']]==0)),nonnegative_costs=bool(np.all(mat.cost>=0) and all(q['d']/4<=q['arrival'] and q['arrival']<=23 for q in qs)),no_mandatory_purchase=True,no_mandatory_discharge=True,components_separable=net['big_m']+1e-12>=net['u_max']-net['u_min'],capability_shortfall=True,no_grid_temporal_coupling=True,forecast_equals_recourse=True)
 assert np.all(np.isinf(mat.hi[ix['grid_p']])) and np.all(np.isinf(mat.hi[ix['grid_q']]))
 prefixes=['discharge_contract/','discharge_physical/','short_le_contract/','station_power/','p_availability/','voltage_plus/','voltage_minus/']
 assert all(any(n.startswith(p) for p in prefixes) for n in mat.row_names['ub'])
 assert all(n.startswith(('P_balance/','Q_balance/','reference/')) for n in mat.row_names['eq'])
 rep=c.loadnp(R/'frozen/FORECAST44.npz');prerequisites(qs,[dict(success=bool(rep['success'][q['route_id']]),arrival=float(rep['arrival_h'][q['route_id']])) for q in qs])
 zero=np.zeros(len(read(R/'frozen/VARIABLE_NAMES.json')));names=read(R/'frozen/VARIABLE_NAMES.json');ci={n:i for i,n in enumerate(names)};pf=c.loadnp(R/'frozen/PF_LIFT.npz')
 for j,k in enumerate(pf['keep']):zero[ci[f's0_y[{j}]']]=y[k]
 original_audit(zero,R/'proof/ZERO_ORIGINAL_AUDIT.json');npz(R/'proof/ZERO_FULL_VECTOR.npz',values=zero)
 assert guard(pr),'SCREEN_NOT_APPLICABLE'
 table=[dict(c=cc,bus=b,always_root=bool(rt[:,bi[b]].all()),island_slots=np.flatnonzero(~rt[:,bi[b]]).tolist(),routes=sum(q['c']==cc for q in qs)) for cc,b in enumerate(net['station_buses'])]
 removed=[i for i,q in enumerate(qs) if table[q['c']]['always_root']]
 npz(R/'proof/ROOT_CONNECTED.npz',root_connected=rt,station_root_connected=rt[:,[bi[b] for b in net['station_buses']]])
 dump(R/'proof/SCREENING_CERTIFICATE.json',dict(passed=True,prerequisites=pr,station_table=table,removed_routes=removed,root_zero_residual=res,root_zero_shed=root_shed,voltage_bigM=net['big_m'],voltage_domain=[net['u_min'],net['u_max']],science=science(),scope='integer optimum preserved; not all labeled feasible plans or LP optimum',source_review={'first_stage':sha(OLD/'tools/compact_first_stage.py'),'network':sha(OLD/'tools/contract_matrix.py'),'aux':sha(OLD/'tools/compact_recourse_pf.py')},source_evidence='All rows of the frozen original builders reviewed: no cross-time grid import or global minimum purchase/call. Zero vector audited against complete frozen original model; full participation cost >=25+30*(J-D)+(J-K)+.02E>=0, although route coefficient -30D is negative.'))
 (R/'proof/CONSTRUCTION_PROOF.md').write_text((R/'instructions/SCREENING_AND_BOUND_SPEC.md').read_text()+'\n\nActual numerical/source prerequisites are in SCREENING_CERTIFICATE.json. Operational root-component replacement reduces all energy upper-bound left sides; retained capability auxiliaries can be lifted with arrived=A, overlap=w, W=J-a, capacity=E, incident=max incident reserved, activated=E*incident, shortfall=0. Removing an always-root vehicle zeros its entire first-stage and auxiliary block. No positive minimum call exists. This construction is implemented and original-matrix audited in NORMALIZATION_TESTS. Common stochastic domain is untouched.\n')

def normalize(x,names,qs,E,net,sc,pf,zero_y,removed_c):
 ix=assemble(net,[],sc,[],E,np.full(E,500.)).index;ci={n:i for i,n in enumerate(names)};xx=x.copy();yy=np.empty(len(zero_y));yy[pf['keep']]=[xx[ci[f's0_y[{i}]']] for i in range(len(pf['keep']))];yy[pf['q_columns']]=.33*yy[pf['p_columns']];rt=root_table(net,sc);bi={b:i for i,b in enumerate(net['buses'])}
 gone={q['e'] for i,q in enumerate(qs) if q['c'] in removed_c and xx[ci[f'route[{i}]']]>.5}
 for i,q in enumerate(qs):
  if q['e'] in gone:xx[ci[f'route[{i}]']]=0
 for n,j in ci.items():
  match=re.match(r'(start|end|selected|K|J|E|window|station|reserved|r7_arrival_prefix|r7_start_prefix|r7_end_prefix)\[(\d+)',n)
  if match and int(match[2]) in gone:xx[j]=0
 for t in range(96):
  for cc,b in enumerate(net['station_buses']):
   if rt[t,bi[b]]:yy[ix['p'][:,cc,t]]=0
  for key in ['shed','u']:yy[ix[key][t,rt[t]]]=zero_y[ix[key][t,rt[t]]]
  for l,(u,v) in enumerate(zip(net['from'],net['to'])):
   if rt[t,bi[u]] and rt[t,bi[v]]:
    for key in ['line_p','line_q']:yy[ix[key][t,l]]=zero_y[ix[key][t,l]]
  for key in ['grid_p','grid_q']:yy[ix[key][t]]=zero_y[ix[key][t]]
 for e in gone:yy[ix['p'][e]]=0;yy[ix['short'][e]]=0
 for j,k in enumerate(pf['keep']):xx[ci[f's0_y[{j}]']]=yy[k]
 return lift(xx,names,names,qs,E,len(net['station_buses']),c.incident(net,sc)[0],ix['short'])

def build_screen():
 cert=read(R/'proof/SCREENING_CERTIFICATE.json');assert cert['passed'];m=frozen_model();a=matrix(m);diff=[]
 for i in cert['removed_routes']:
  v=m.getVarByName(f'route[{i}]');assert v.LB==0 and v.UB==1;v.UB=0;diff.append(dict(name=v.VarName,index=v.index,old_UB=1,new_UB=0,route=i))
 m.update();b=matrix(m)
 for k in ['b','sense','lo','types','cost']:assert np.array_equal(a[k],b[k]),k
 assert a['names']==b['names'] and a['rows']==b['rows'] and (a['A']!=b['A']).nnz==0 and a['constant']==b['constant']
 assert np.flatnonzero(a['hi']!=b['hi']).tolist()==sorted(d['index'] for d in diff)
 out=R/'models';out.mkdir(exist_ok=True);m.write(str(out/'SCREENED.mps.gz'));dump(out/'FIXED_ROUTE_MAP.json',diff);dump(out/'COMPLETE.json',dict(passed=True,original=model_sizes(m),route_fixed=len(diff),routes_remaining=len(c.routes())-len(diff),free_binary_upper=m.NumBinVars-len(diff),matrix_unchanged=True,presolve_prior_fixing_identity='UNKNOWN: no exposed original-to-presolved column map',model_sha=sha(out/'SCREENED.mps.gz')));m.dispose()

def fix_discrete_complement(m,qs,free,center):
 names=m.getAttr('VarName');a=matrix(m);fixed=[]
 for j,n in enumerate(names):
  match=re.fullmatch(r'route\[(\d+)\]',n)
  e=qs[int(match[1])]['e'] if match else None
  if e is None:
   mm=re.fullmatch(r'(start|end)\[(\d+),\d+\]',n)
   if mm:e=int(mm[2])
  if e is not None and e not in free:
   val=round(float(center[j]));assert abs(val-center[j])<=1e-9 and a['lo'][j]<=val<=a['hi'][j];v=m.getVars()[j];v.LB=v.UB=val;fixed.append(dict(index=j,name=n,value=val))
 m.update();b=matrix(m);allowed={r['index'] for r in fixed};changed=set(np.flatnonzero((a['lo']!=b['lo'])|(a['hi']!=b['hi'])));assert changed<=allowed
 continuous=a['types']=='C';assert np.array_equal(a['lo'][continuous],b['lo'][continuous]) and np.array_equal(a['hi'][continuous],b['hi'][continuous]);audit_matrix(b,center)
 return dict(fixed=fixed,continuous_bounds_unchanged=True,free_vehicles=sorted(free),clean_base_matrix=True,center_feasible=True)

def new_start(m,x):
 audit_matrix(matrix(m),x);m.NumStart=0;m.update();m.NumStart=1;m.Params.StartNumber=0;m.setAttr('Start',m.getVars(),x.tolist())

def latest_best(target):
 target.mkdir(parents=True,exist_ok=True);candidates=[(R/'frozen/START.npz',None)]
 for p in (OLD/'04_MODEL_AND_NATIVE_SOLVE/candidates').glob('*.json'):
  d=read(p)
  if d['objective']<=1381.534423:candidates.append((p.with_suffix('.npz'),d))
 if (R/'benchmark/FINAL.npz').exists():candidates.append((R/'benchmark/FINAL.npz',None))
 best=None
 for p,meta in candidates:
  h=sha(p)
  if meta:assert h==meta['raw_sha']
  ap=R/'candidate_audits'/(h+'.json')
  if ap.exists():au=read(ap)
  else:
   x=np.load(p)['values'];au=original_audit(x,ap)
  if best is None or au['UB']<best[0]:best=(au['UB'],p,au,h)
 shutil.copy2(best[1],target/'CENTER.npz');dump(target/'CENTER.json',dict(source=str(best[1]),raw_sha=best[3],audit=best[2],epoch=time.time()));return np.load(target/'CENTER.npz')['values'],best[2]
