from r36_recourse import *
from r36_probability import moments
from scipy import sparse as sp
def parameters(m,log,threads=4,soft_gb=None):
 m.Params.Threads=threads;m.Params.Method=1;m.Params.NodeMethod=1;m.Params.PreSparsify=1;m.Params.MIPGap=0;m.Params.MIPGapAbs=1;m.Params.FeasibilityTol=1e-9;m.Params.IntFeasTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.TimeLimit=GRB.INFINITY
 if soft_gb is not None:m.Params.SoftMemLimit=soft_gb
 m.Params.OutputFlag=1;m.Params.LogFile=str(log)
def add_risk(m,a):
 n=len(a['lower']);risk=dict(beta0=m.addVar(lb=-GRB.INFINITY,name='r36_beta0'),plus=m.addMVar(n,name='r36_betaPlus'),minus=m.addMVar(n,name='r36_betaMinus'))
 risk['plane_expr']=risk['beta0']+a['phi'].T@(risk['plus']-risk['minus']);m.addConstr(risk['plane_expr']>=np.zeros(a['phi'].shape[1]),name='R36_ALL_SUPPORT_ZERO_LOWER_PLANE');risk['objective']=risk['beta0']+a['upper']@risk['plus']-a['lower']@risk['minus'];return risk
def validate_model(m,fs,pool):
 m.update();vs=m.getVars();names=[v.VarName for v in vs];assert len(names)==len(set(names));assert len(fs['v'])==20100 and fs['Ccount']==8 and fs['Ecount']==30
 assert sum(v.VType==GRB.BINARY for v in vs)==25770
 assert all(v.LB==0 and v.UB==1 for v in fs['v'].values()),'DET_ROUTE_FIXING_IN_DRO'
 for row in m.getConstrs():assert not row.ConstrName.lower().startswith(('r32_','r33_','r34_','r35_','r35_count/','count_branch/','window_admission/')),'DET_PRIVATE_ROW'
 assert sum(x.startswith('route[') for x in names)==20100 and sum(x.startswith('E[') for x in names)==30
 import re
 assert not any(re.match(r'^s\d+_(route|start|end|E)\[',n) for n in names),'SCENARIO_SPECIFIC_FIRST_STAGE'
 for sid in pool:assert any(n.startswith(f's{sid}_y[') for n in names)
 return dict(passed=True,one_shared_first_stage=True,binaries=25770,stations=8,scenario_count=len(pool),routes=20100)
def make_model(pool,log,with_risk=True):
 m=gp.Model(SCIENCE);parameters(m,log);fs=add_first(m,routes(),30,8,full_bounds());add_strengthening(m,fs);risk=add_risk(m,moments()) if with_risk else None;copies={}
 for sid in pool:copies[sid]=add_scene(m,fs,risk,sid)
 m.setObjective(fs['objective']+(risk['objective'] if risk else sum(rc['objective'] for rc in copies.values())));m.update();validate_model(m,fs,pool);return m,fs,risk,copies
def add_scene(m,fs,risk,sid):
 rc=add_recourse(m,fs,layout()['net'],load_scene('TRAIN',sid),route_states('L08',fs['qs'],'TRAIN',sid),sid)
 if risk:m.addConstr(risk['plane_expr'][sid]>=rc['objective'],name=f'R36_RUNNING_PLANE/{sid}')
 return rc
def bind(m):
 qs=routes();E=30;C=8;starts=list(range(93));ends=list(range(1,97));allvars={v.VarName:v for v in m.getVars()}
 def table(name,keys):return gp.tupledict({k:allvars[name+'['+(','.join(map(str,k)) if isinstance(k,tuple) else str(k))+']'] for k in keys})
 fs=dict(qs=qs,Ecount=E,Ccount=C,M=full_bounds(),starts=starts,ends=ends,by_e=[[i for i,q in enumerate(qs) if q['e']==e] for e in range(E)])
 for key,name,indices in [('v','route',range(len(qs))),('s','start',[(e,k) for e in range(E) for k in starts]),('f','end',[(e,j) for e in range(E) for j in ends]),('u','selected',range(E)),('K','K',range(E)),('J','J',range(E)),('E','E',range(E)),('w','window',[(e,t) for e in range(E) for t in range(96)]),('b','station',[(e,c) for e in range(E) for c in range(C)]),('wc','reserved',[(e,c,t) for e in range(E) for c in range(C) for t in range(96)])]:fs[key]=table(name,indices)
 fs['objective']=25*gp.quicksum(fs['u'].values())+30*gp.quicksum(fs['J'].values())-gp.LinExpr([30*q['d']/4 for q in qs],list(fs['v'].values()))+gp.quicksum(fs['J'][e]-fs['K'][e] for e in range(E))+.02*gp.quicksum(fs['E'].values())
 a=moments();risk=dict(beta0=allvars['r36_beta0'],plus=gp.MVar.fromlist([allvars[f'r36_betaPlus[{i}]'] for i in range(45139)]),minus=gp.MVar.fromlist([allvars[f'r36_betaMinus[{i}]'] for i in range(45139)]));risk['plane_expr']=risk['beta0']+a['phi'].T@(risk['plus']-risk['minus']);risk['objective']=risk['beta0']+a['upper']@risk['plus']-a['lower']@risk['minus'];return fs,risk
def build_rmp():
 out=R/'prepared';out.mkdir(exist_ok=True);pool=read(R/'INITIAL_POOL.json')['pool'];m,fs,risk,_=make_model(pool,out/'BUILD.log');m.write(str(out/'RMP.mps.gz'));rec=dict(passed=True,pool=pool,sha256=sha(out/'RMP.mps.gz'),variables=m.NumVars,rows=m.NumConstrs,nonzeros=m.NumNZs,binaries=m.NumBinVars,all_support_planes=400,no_optimize_called=True);m.dispose()
 m=gp.read(str(out/'RMP.mps.gz'));fs,risk=bind(m);validate_model(m,fs,pool)
 from r36_control import set_start
 m.NumStart=2;m.update()
 for slot in range(2):
  m.Params.StartNumber=slot;m.setAttr('Start',m.getVars(),[1.]*m.NumVars)
 m.Params.StartNumber=0
 for v in fs['v'].values():v.Start=0
 m.update();assert fs['u'][13].Start==1 and all(v.Start==0 for v in fs['v'].values());conflict=1.
 cert=loadnp(R/'probability/ZERO/CERTIFICATE.npz');start=set_start(m,fs,[],risk,cert);assert m.NumStart==1 and fs['u'][13].Start==0 and all(v.Start==0 for v in fs['v'].values())
 known=set(v.index for key in ['v','s','f','u','K','J','E','w','b','wc'] for v in fs[key].values())|{risk['beta0'].index}|{v.index for v in risk['plus'].tolist()+risk['minus'].tolist()}
 assert all(v.Start==GRB.UNDEFINED for v in m.getVars() if v.index not in known)
 dump(out/'WARM_START_REPAIR.json',dict(passed=True,old_one_route_13_violation=conflict,new_selected13=fs['u'][13].Start,slots=m.NumStart,unknown_count=m.NumVars-len(known),**start))
 m.dispose();rec['reload_bindings_verified']=True;rec['full_warm_start_repair_test']=True;return rec
def universal_zero_and_pool():
 q=loadnp(R/'evaluations/ZERO/Q.npz');metrics=read(R/'tests/REAL_SCENE_SELECTION.json')['metrics'];zeros=[];proofs=[]
 assert CFG['short']>=0 and CFG['degradation']>=0 and np.all(tou()>=0)
 for sid in range(400):
  a=load_scene('TRAIN',sid);zeta=incident(layout()['net'],a)[0];v=read(R/'evaluations/ZERO'/f'{sid:04d}'/'COMPLETE.json')
  proof=not a['line_operating_outages'].any() and not zeta.any() and abs(v['Q'])<1e-8 and v['raw_residual']<=1e-7
  if proof:zeros.append(sid)
  proofs.append(dict(sid=sid,proved=bool(proof),physical_sha=v['physical_sha'],recovery_sha=v['recovery_sha'],zero_primal_sha=sha(R/'evaluations/ZERO'/f'{sid:04d}'/'PRIMAL.npz'),reason='Original ZERO network supplies all load; no incident for any station/window; zero power and shortfall feasible for any forecast-legal plan; nonnegative operating costs' if proof else 'NOT_PROVEN_NOT_ELIMINATED'))
 categories={}
 for x in metrics:
  if x['sid'] in zeros:continue
  # Exogenous incident duration categories, declared before selecting by q.
  cat='no_station_island' if x['incident_slots']==0 else 'short_1h' if x['incident_slots']<=4 else 'medium_6h' if x['incident_slots']<=24 else 'long_over6h'
  categories.setdefault(cat,[]).append(x['sid'])
 pool=[]
 for cat,ids in sorted(categories.items()):pool.append(min(ids))
 for sid in sorted((s for s in range(400) if s not in zeros),key=lambda s:(-q['qU'][s],s)):
  if sid not in pool:pool.append(sid)
  if len(pool)>=12:break
 rec=dict(passed=True,pool=pool,universal_zero=sorted(zeros),categories=categories,rule='First ID in each predeclared island-duration class, then descending ZERO qU and ID; at most12',all_support_planes=400)
 dump(R/'UNIVERSAL_ZERO_PROOFS.json',proofs);dump(R/'INITIAL_POOL.json',rec);return rec
def resource_check():
 pool=universal_zero_and_pool();extra=[s for s in range(400) if s not in pool['universal_zero'] and s not in pool['pool']];ids=(pool['pool']+extra)[:20];out=R/'resources';out.mkdir(exist_ok=True)
 m=gp.Model('R36_RESOURCE_ONLY');parameters(m,out/'BUILD.log',1);start=time.time();fs=add_first(m,routes(),30,8,full_bounds());add_strengthening(m,fs);m.update();base_rss=rss();counts=[]
 for n,sid in enumerate(ids,1):
  rc=add_scene(m,fs,None,sid);m.update()
  if n in [1,5,20]:
   rec=dict(copies=n,variables=m.NumVars,binaries=m.NumBinVars,rows=m.NumConstrs,nonzeros=m.NumNZs,cumulative_build_s=time.time()-start,peak_rss_bytes=rss());counts.append(rec);dump(out/f'COPIES_{n}.json',rec)
 risk=add_risk(m,moments());m.update();risk_rss=rss();validate_model(m,fs,ids)
 # Reject accidental DET and duplicate-first-stage modifications using the actual built model.
 v=fs['v'][0];v.UB=0;m.update()
 from r36_tests import negative
 assert negative(lambda:validate_model(m,fs,ids));v.UB=1;m.update();bad=m.addConstr(fs['u'][0]<=0,name='r34_det_window_admission/0');m.update();assert negative(lambda:validate_model(m,fs,ids));m.remove(bad);m.update()
 bad=m.addVar(name='s1_E[0]');m.update();assert negative(lambda:validate_model(m,fs,ids));m.remove(bad);m.update()
 peak=max(risk_rss,rss());slope=max(1.,(counts[-1]['peak_rss_bytes']-counts[0]['peak_rss_bytes'])/19);initial_est=base_rss+slope*(len(pool['pool'])+8)+(risk_rss-counts[-1]['peak_rss_bytes']);audit_budget=2*1024**3;request=int(np.ceil((5*max(peak,initial_est)+audit_budget+2*1024**3)/1024**3));m.dispose()
 resources=resource_reader.resources();ok=resources['signed_headroom_bytes']>=request*1024**3 and not resources['unknown_scheduler_limits'];rec=dict(passed=True,admitted=ok,builds=counts,base_rss=base_rss,with_probability_peak=peak,slope_bytes_per_scene=slope,request_GiB=request,soft_mem_GB=(request*1024**3-audit_budget-2*1024**3)/1e9,audit_workers=1,audit_threads=1,MIP_threads=4,initial_pool_plus8_estimate=initial_est,resources=resources,measured_only=True)
 dump(R/'RESOURCE_PROFILE.json',rec);return rec
