from r32_core import *
import traceback,argparse

def load_search_model():
 m=gp.read(str(R/'models/SCREENED.mps.gz'));params(m);return m

def native_run(m,x,out,seconds,focus,kind):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'BEGIN.json').exists(),'INCOMPLETE_OR_COMPLETED_NATIVE_NO_RESTART';new_start(m,x);m.Params.OutputFlag=1;par=params(m,4,focus,seconds,out/'NATIVE.log');par['OutputFlag']=1;dump(out/'PARAMETERS.json',par);dump(out/'BEGIN.json',dict(epoch=time.time(),owner=ctrl.proc(os.getpid()),science=science(),kind=kind,limit=seconds,branch_tree_restored=False,start_sha=c.digest(x),model_variables=m.NumVars,model_rows=m.NumConstrs));npz(out/'SNAPSHOT_0.npz',values=x)
 best=[float(matrix(m)['cost']@x+m.ObjCon),x.copy()];last=[-10.];cross=[False];cbseconds=[0.];errors=[];f=(out/'PROGRESS.jsonl').open('a',buffering=1)
 def cb(mod,where):
  if where not in [GRB.Callback.MIP,GRB.Callback.MIPSOL]:return
  t=time.monotonic()
  try:
   runtime=mod.cbGet(GRB.Callback.RUNTIME)
   if where==GRB.Callback.MIPSOL:
    val=mod.cbGet(GRB.Callback.MIPSOL_OBJ)
    if val<best[0]-1e-8:best[:]=[val,np.array(mod.cbGetSolution(mod.getVars()))]
   if where==GRB.Callback.MIP:
    d=dict(runtime=runtime,LB=float(mod.cbGet(GRB.Callback.MIP_OBJBND)),UB=float(mod.cbGet(GRB.Callback.MIP_OBJBST)),nodes=float(mod.cbGet(GRB.Callback.MIP_NODCNT)))
    if runtime-last[0]>=10:f.write(json.dumps(d)+'\n');last[0]=runtime
    if kind=='GLOBAL_EXACT_SCREENED_DET' and runtime>=180 and not cross[0]:cross[0]=True;dump(out/'SNAPSHOT_180.json',d);npz(out/'SNAPSHOT_180.npz',values=best[1])
  except Exception as exc:errors.append(repr(exc))
  finally:cbseconds[0]+=time.monotonic()-t
 t=time.monotonic();m.optimize(cb);wall=time.monotonic()-t;f.close();rec=dict(status=m.Status,solutions=m.SolCount,LB=float(m.ObjBound) if np.isfinite(m.ObjBound) and abs(m.ObjBound)<1e99 else None,UB=float(m.ObjVal) if m.SolCount else None,kind=kind,science=science(),runtime=m.Runtime,wall=wall,nodes=m.NodeCount,iterations=m.IterCount,work=m.Work,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,solver_max_memory_GB=m.MaxMemUsed,callback_seconds=cbseconds[0],callback_errors=errors,epoch=time.time())
 if m.SolCount:npz(out/'FINAL.npz',values=np.array(m.getAttr('X')))
 dump(out/'END.json',rec);assert not errors,errors
 if m.SolCount:
  t=time.monotonic();au=original_audit(np.load(out/'FINAL.npz')['values'],out/'ORIGINAL_AUDIT.json');dump(out/'AUDIT_TIME.json',dict(seconds=time.monotonic()-t));rec['safe_UB']=au['UB']
 return rec

def benchmark_screen():
 assert read(R/'tests/COMPLETE.json')['passed'];out=R/'benchmark'
 if (out/'END.json').exists():assert (out/'ORIGINAL_AUDIT.json').exists();return
 for n,h in read(R/'tests/COMPLETE.json')['source_hashes'].items():assert sha(R/n)==h,('UNTESTED_SOURCE',n)
 t=time.monotonic();m=load_search_model();x=np.load(R/'frozen/START.npz')['values'];out.mkdir(exist_ok=True);dump(out/'BUILD_TIME.json',dict(seconds=time.monotonic()-t));native_run(m,x,out,600,0,'GLOBAL_EXACT_SCREENED_DET');m.dispose()
 for tag in ['SNAPSHOT_0','SNAPSHOT_180']:
  if (out/(tag+'.npz')).exists():original_audit(np.load(out/(tag+'.npz'))['values'],out/(tag+'_AUDIT.json'))

def current_global(ub):
 items=[base_bound()]
 if (R/'benchmark/END.json').exists():
  en=read(R/'benchmark/END.json');items.append(dict(type='GLOBAL_EXACT_SCREENED_DET',science=science(),value=en['LB'],proof_passed=read(R/'proof/SCREENING_CERTIFICATE.json')['passed'],source='benchmark/END.json'))
 return aggregate(items,ub)

def plan_neighborhoods():
 # Only future finite workers use the logging correction. Active E uses its loaded original code.
 patch=R/'patches/LOGGING_ACCEPTANCE.json'
 if not patch.exists():
  from r32_tests import solve_test
  tm=gp.Model('R32_LOGGING_NATIVE_TEST');params(tm);vv=tm.addVars(2,vtype=GRB.BINARY);tm.addConstr(2*vv[0]+vv[1]>=1);tm.setObjective(vv[0]+2*vv[1]);tm.update();tm.Params.OutputFlag=1;solve_test(tm,R/'patches/LOGGING_TRUE_MIP');tm.dispose();log=(R/'patches/LOGGING_TRUE_MIP.log').read_text();assert 'Optimal solution found' in log
  dump(patch,dict(passed=True,native_test='LOGGING_TRUE_MIP',before_sha=sha(R/'patches/entry_benchmark_original.py'),after_sha=sha(Path(__file__)),scope='OutputFlag=1 for future native runs only; no math or solver-search parameter change',E_not_restarted=True,E_presolve_and_exact_root_time='NOT_OBSERVED',epoch=time.time()))
  # Separate Model.presolve observation, not a rerun of E optimize or an inferred column map.
  pm=load_search_model();pm.Params.OutputFlag=1;pm.Params.Threads=4;pm.Params.LogFile=str(R/'patches/SCREEN_MODEL_PRESOLVE.log');t=time.monotonic();pres=pm.presolve();dump(R/'patches/SCREEN_MODEL_PRESOLVE.json',dict(observation='standalone Model.presolve; not necessarily optimize internal presolve',seconds=time.monotonic()-t,original=model_sizes(pm),presolved=model_sizes(pres),eliminated_route_identity='UNKNOWN'));pres.dispose();pm.dispose()
 out=R/'local';out.mkdir(exist_ok=True)
 if (out/'SCHEDULE.json').exists():return
 x,au=latest_best(out);bound=current_global(au['UB']);dump(out/'PRE_LOCAL_GLOBAL.json',bound)
 if bound['certified']:dump(out/'SCHEDULE.json',dict(skipped=True,reason='GLOBAL_GAP_ALREADY_CERTIFIED',sets=[]));return
 plans=au['physical']['plans'];qs=c.routes();selected=sorted(plans,key=lambda h:(-(c.first_cost(h['d'],h['k'],h['r'])+.02*h['contract']),h['ev']));sel={h['e'] for h in selected};unsel=sorted(set(range(30))-sel);names={q['e']:q['ev'] for q in qs};cert=read(R/'proof/SCREENING_CERTIFICATE.json');kept={v['c'] for v in cert['station_table'] if not v['always_root']}
 def replacement(h,exclude):
  scores=[]
  for e in unsel:
   if e in exclude:continue
   eligible=[]
   for q in qs:
    if q['e']!=e or q['c']!=h['c'] or q['arrival']>h['k']/4:continue
    cap=c.ebar(q['departure_dc'],q['distance'],q['arrival'],h['k'],h['r'])
    if cap>=15:eligible.append((cap<h['contract']-1e-9,c.first_cost(q['d'],h['k'],h['r'])+.02*min(cap,h['contract']),-cap,q['route_id']))
   score=(0,*min(eligible),names[e]) if eligible else (1,-sum(q['e']==e and q['c'] in kept for q in qs),names[e])
   scores.append((score,e))
  return min(scores) if scores else None
 # Round-robin distinct incident time groups for pairs, then fee-ranked disjoint pairs for quads.
 groups={}
 for h in selected:groups.setdefault(tuple(cert['station_table'][h['c']]['island_slots']),[]).append(h)
 ordered=[]
 while any(groups.values()):
  for key in sorted(groups,key=lambda k:(k[0] if k else 999,k)):
   if groups[key]:ordered.append(groups[key].pop(0))
 sets=[];used=set();seen=set()
 for h in ordered[:4]:
  rep=replacement(h,used);free=[h['e']]+([rep[1]] if rep else []);used.update(free);sig=tuple(sorted(free))
  if sig not in seen:seen.add(sig);sets.append(dict(index=len(sets),kind='PAIR',free=sorted(free),selected=[h['e']],replacements=[rep[1]] if rep else [],ranking=[rep[0]] if rep else [],reason='cost-ranked with incident-group round robin'))
 for j in range(4):
  if len(selected)<2:break
  hs=[selected[(2*j)%len(selected)],selected[(2*j+1)%len(selected)]];free=[h['e'] for h in hs];reps=[];scores=[]
  for h in hs:
   rep=replacement(h,set(free))
   if rep:free.append(rep[1]);reps.append(rep[1]);scores.append(rep[0])
  sig=tuple(sorted(free))
  if sig not in seen:seen.add(sig);sets.append(dict(index=len(sets),kind='QUAD',free=list(sig),selected=[h['e'] for h in hs],replacements=reps,ranking=scores,reason='consecutive pairs of fee-ranked selected vehicles; cyclic if fewer than8'))
 assert len(sets)<=8 and all(len(s['free'])<=4 for s in sets)
 lock=dict(center_sha=sha(out/'CENTER.npz'),science=science(),selected_fee_order=[dict(e=h['e'],ev=h['ev'],c=h['c'],fee=c.first_cost(h['d'],h['k'],h['r'])+.02*h['contract']) for h in selected],sets=sets,budget_seconds=720,per_solve=90,deterministic=True,heuristic_only=True,epoch=time.time());lock['schedule_sha']=c.digest(lock);dump(out/'SCHEDULE.json',lock)

def local_search():
 schedule=read(R/'local/SCHEDULE.json')
 if schedule.get('skipped'):return
 qs=c.routes();center=np.load(R/'local/CENTER.npz')['values'];au=read(R/'local/CENTER.json')['audit'];bestub=au['UB'];accepted=[]
 for spec in schedule['sets']:
  out=R/'local'/f"N{spec['index']:02d}";out.mkdir(exist_ok=True)
  if (out/'DECISION.json').exists():
   dec=read(out/'DECISION.json')
   if dec['accepted']:center=np.load(out/'FINAL.npz')['values'];bestub=dec['new_safe_UB'];accepted.append(str(out.relative_to(R)))
   continue
  bounds=current_global(bestub)
  if bounds['certified']:dump(out/'SKIPPED.json',dict(reason='GLOBAL_CERTIFIED',bounds=bounds));break
  used=sum(read(p)['wall'] for p in (R/'local').glob('N*/END.json'));budget=min(90.,720-used)
  if budget<=0:dump(out/'SKIPPED.json',dict(reason='LOCAL_BUDGET_EXHAUSTED',used=used));break
  t=time.monotonic();m=load_search_model();fix=fix_discrete_complement(m,qs,set(spec['free']),center);npz(out/'CENTER.npz',values=center);dump(out/'FIXINGS.json',fix);dump(out/'BUILD.json',dict(seconds=time.monotonic()-t,center_sha=sha(out/'CENTER.npz'),center_safe_UB=bestub,remaining_budget=budget,schedule_sha=schedule['schedule_sha']));rec=native_run(m,center,out,budget,1,'LOCAL_NEIGHBORHOOD');m.dispose()
  newub=rec.get('safe_UB');accept=newub is not None and bestub-newub>1e-5;dump(out/'DECISION.json',dict(accepted=accept,old_safe_UB=bestub,new_safe_UB=newub,improvement=bestub-newub if newub is not None else None,local_bound_never_global=True))
  if accept:
   center=np.load(out/'FINAL.npz')['values'];bestub=newub;accepted.append(str(out.relative_to(R)));h=sha(out/'FINAL.npz');dest=R/'INBOX'/h;dest.mkdir(parents=True,exist_ok=False);shutil.copy2(out/'FINAL.npz',dest/'VALUES.npz');shutil.copy2(out/'ORIGINAL_AUDIT.json',dest/'CERTIFICATE.json');dump(dest/'DELIVERY.json',dict(status='BEST_FOR_NEXT_SAFE_START',consumed_by_active_worker=False,science=science(),source=str(out),values_sha=h))
 npz(R/'local/BEST.npz',values=center);dump(R/'local/COMPLETE.json',dict(passed=True,safe_UB=bestub,accepted=accepted,native_wall_used=sum(read(p)['wall'] for p in (R/'local').glob('N*/END.json')),local_bounds_excluded=True))

def audit_best():
 out=R/'best';out.mkdir(exist_ok=True);x,au=latest_best(out)
 if (R/'local/BEST.npz').exists():
  lx=np.load(R/'local/BEST.npz')['values'];la=original_audit(lx)
  if la['UB']<au['UB']:x=lx;au=la
 npz(out/'VALUES.npz',values=x);dump(out/'ORIGINAL_AUDIT.json',au);m=frozen_model();names=m.getAttr('VarName');m.dispose();v=dict(zip(names,x));pf=c.loadnp(R/'frozen/PF_LIFT.npz');net=c.layout()['net'];sc=c.load_scene('FORECAST_SELECTED');ix=assemble(net,[],sc,[],30,c.full_bounds()).index;p=np.array([v[f's0_y[{j}]'] for j in ix['p'].flat]).reshape(ix['p'].shape);plans={h['e']:h for h in au['physical']['plans']};rows=[];fleet={q['e']:q['ev'] for q in c.routes()}
 for e in range(30):
  h=plans.get(e);rows.append(dict(e=e,ev=fleet[e],selected=h is not None,station=h['c']+1 if h else None,bus=net['station_buses'][h['c']] if h else None,departure=h['d']/4 if h else None,arrival=h['arrival'] if h else None,K=h['k']/4 if h else None,J=(h['r']+1)/4 if h else None,E=v[f'E[{e}]'],actual_V2G_kWh=.25*p[e].sum(),aux_DC_kWh=(h['r']+1)/4-h['arrival'] if h else 0,fee=c.first_cost(h['d'],h['k'],h['r'])+.02*v[f'E[{e}]'] if h else 0))
 for row in rows:
  trace=next((q for q in au['physical']['SOC'] if q['ev']==row['ev']),None)
  row.update(min_SOC_percent=100*trace['min_dc']/500 if trace else None,release_SOC_percent=100*trace['terminal_dc']/500 if trace else None,SOC_note='full dwell plus actual V2G' if trace else 'unselected: no emergency schedule; not filled with zero SOC')
 c.csvout(out/'VEHICLES30.csv',rows);npz(out/'DISCHARGE_30x8x96.npz',p_ac_kw=p);dump(out/'GLOBAL_BOUNDS.json',current_global(au['UB']))

def compare():
 out=R/'report';out.mkdir(exist_ok=True);rows=[]
 for name,d in [('A_BASE_T4',R/'historical/A_BASE_T4'),('E_BASE_SITE_SCREEN_T4',R/'benchmark')]:
  en=read(d/'END.json');audit=read(d/'ORIGINAL_AUDIT.json');ub=audit.get('UB',audit.get('original_UB'));lb=en.get('LB',en.get('native_LB'));log=(d/'NATIVE.log').read_text();root=re.search(r'Root relaxation: objective ([^,]+), (\d+) iterations, ([\d.]+) seconds',log);pres=re.findall(r'Presolved: (\d+) rows, (\d+) columns, (\d+) nonzeros',log);types=re.findall(r'Variable types: (\d+) continuous, (\d+) integer \((\d+) binary\)',log)
  rows.append(dict(profile=name,LB=lb,UB=ub,gap=ub-lb,runtime=en['runtime'],nodes=en['nodes'],root_LP=dict(objective_log_rounded=float(root[1]),seconds=float(root[3])) if root else None,presolved_counts=pres,presolved_types=types,presolve_fixing_identity='UNKNOWN no column mapping',work=en['work'],peak_RSS_bytes=en['peak_RSS_bytes']))
 dump(out/'BENCHMARK_COMPARISON.json',dict(profiles=rows,same_immutable_start=True,start_sha=sha(R/'frozen/START.npz'),statistical_speedup_not_claimed=True,historical_A_callback_error_and_background_load_caveat=True,gap_improvement=rows[0]['gap']-rows[1]['gap'],E_log_limitation='OutputFlag inherited0: exact rootLP/rootcut time and optimize presolve size NOT_OBSERVED. Structured native bounds/nodes/work/runtime are available. No rerun. Separate Model.presolve observation is not optimize internal mapping.'))

def report():
 sources=read(R/'BASELINE.json')['sources'];assert all(sha(Path(p))==h for p,h in sources.items());base=read(R/'BASELINE.json');protect=dict(sources_unchanged=True,signals_sent=[],old_owners=[dict(source=v['source'],same_owner_live=ctrl.same(v.get('owner'))) for v in base['resources']['entries'] if v.get('owner')],R32_production_deployed=False,R32_400_calls=0)
 dump(R/'report/PROTECTION.json',protect);a=read(R/'best/ORIGINAL_AUDIT.json');bounds=read(R/'best/GLOBAL_BOUNDS.json');cmp=read(R/'report/BENCHMARK_COMPARISON.json');screen=read(R/'proof/SCREENING_CERTIFICATE.json');decisions={str(p.relative_to(R)):read(p) for p in (R/'local').glob('N*/DECISION.json')}
 dump(R/'report/REPORT.json',dict(status='FINITE_DIAGNOSTICS_COMPLETE_NOT_DEPLOYED',screen=screen,benchmark=cmp,local=decisions,best=a,bounds=bounds,protection=protect,scope=read(R/'DET_ONLY_SCOPE.json')))
 lines=['# R32 实际执行结果','', '仅forecast44 DET；原模型辅助结构保留；共同8站域未修改。','', '|档位|LB|原式安全UB|gap|根LP秒|节点|','|---|---:|---:|---:|---:|---:|']
 for p in cmp['profiles']:lines.append(f"|{p['profile']}|{p['LB']}|{p['UB']}|{p['gap']}|{p['root_LP']['seconds'] if p['root_LP'] else None}|{p['nodes']}|")
 lines+=['', '筛除站点：'+str([v['bus'] for v in screen['station_table'] if v['always_root']]),'固定route数：'+str(len(screen['removed_routes']))+'；原列不删除。原presolve已包含哪些同样固定：未知，无列映射。','局部结果：'+json.dumps(decisions,ensure_ascii=False),'合并合法全局界：'+json.dumps(bounds,ensure_ascii=False),'局部LB全部排除；新候选未由活worker消费，保存在INBOX供安全接续。','原R30 all仍可能在认证后执行fixed400，无已验证合作钩子，本轮没有修改或停止旧进程。R32自身不调用400阶段。','A为已完成历史基准，监测代码和背景负载有差异，不能宣称可靠统计加速倍数。','有限诊断完成后停止新派发。']
 (R/'report/README_ZH.md').write_text('\n'.join(lines)+'\n')

def package():
 assert (R/'report/REPORT.json').exists();out=R/'delivery';out.mkdir(exist_ok=True);z=out/(R.name+'_PRO_ANALYSIS.zip');assert not z.exists()
 files=[p for p in R.rglob('*') if p.is_file() and out not in p.parents and '__pycache__' not in p.parts and p.suffix not in ['.lock','.tmp']];manifest={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zz:
  for p in files:zz.write(p,str(p.relative_to(R)))
  zz.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(z) as zz:assert zz.testzip() is None
 dump(out/'PACKAGE_INDEX.json',dict(file=z.name,bytes=z.stat().st_size,sha256=sha(z),members=len(manifest)))

def dispatch(stage):
 if stage=='tests':
  from r32_tests import tests;tests()
 else:globals()[stage.replace('-','_')]()

def all_stages():
 stages=['inventory','det-only-scope','certify-screen','build-screen','tests','benchmark-screen','plan-neighborhoods','local-search','audit-best','compare','report','package']
 with (R/'RUN.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with ctrl.lock(V/'DISPATCH.lock'):
   assert not ctrl.same(ctrl.read(LEASE).get('owner'));res=rr.resources();dump(R/'RESOURCE_ADMISSION.json',res);assert not res['unknown_scheduler_limits'] and res['signed_headroom_bytes']>=12*1024**3,'RETRYABLE_RESOURCE_WAIT';ctrl.dump(LEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=12*1024**3,threads=4,method='R32_FINITE_DET',production=False))
  try:
   for stage in stages:
    if stage!='inventory' and read(R/'INITIAL_STATUS.json')['original_certified']:
     dump(R/'ORIGINAL_ALREADY_CERTIFIED.json',dict(no_new_trials=True,source=read(OLD/'05_SELECTED_PLAN_AND_CERTIFICATES/AUDIT_COMPLETE.json')));break
    d=R/'execution'/stage;d.mkdir(parents=True,exist_ok=True)
    if (d/'WAIT.json').exists():assert read(d/'WAIT.json')['returncode']==0,('FAILED_STAGE_REQUIRES_REVIEW',stage);print('REUSE',stage,flush=True);continue
    assert not (d/'PROCESS.json').exists(),'UNKNOWN_PARENT_WAIT'
    with (d/'stdout.log').open('wb') as so,(d/'stderr.log').open('wb') as se:
     cmd=[sys.executable,'-B',str(Path(__file__)),stage,'--det-only'];t=time.monotonic();p=subprocess.Popen(cmd,stdout=so,stderr=se,cwd=R);dump(d/'PROCESS.json',dict(pid=p.pid,parent=os.getpid(),argv=cmd,epoch=time.time()));print('BEGIN',stage,p.pid,flush=True);rc=p.wait();dump(d/'WAIT.json',dict(actual_parent_wait=True,returncode=rc,elapsed_s=time.monotonic()-t,epoch=time.time()));print('WAIT',stage,rc,flush=True)
    assert rc==0,('STAGE_FAILED',stage)
  finally:
   with ctrl.lock(V/'DISPATCH.lock'):
    if (ctrl.read(LEASE).get('owner') or {}).get('pid')==os.getpid():ctrl.dump(LEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time()))

if __name__=='__main__':
 stage=sys.argv[1] if len(sys.argv)>1 else 'all'
 if stage in ['all','--resume','resume']:all_stages()
 elif stage=='status':print(json.dumps({str(p.relative_to(R)):read(p) for p in (R/'execution').glob('*/WAIT.json')}))
 else:resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,resource.getrlimit(resource.RLIMIT_AS)[1]));dispatch(stage)
