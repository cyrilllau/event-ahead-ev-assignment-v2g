from r36_model import *
from r36_tests import negative
from r36_queue import lifecycle_test
def semantic_tests():
 out=R/'tests/semantics';out.mkdir(parents=True,exist_ok=True);checks={}
 paper=Path(read(R/'ROUTE_POLICY_EVIDENCE.json')['source']);text=paper.read_text();assert 'selected route remains unchanged after the disaster scenario is realized' in text
 dump(R/'ROUTE_POLICY_ACCEPTED.json',dict(passed=True,path=str(paper),sha256=sha(paper),quote='selected route remains unchanged after the disaster scenario is realized',note='Snapshot still contains old pre-R15 fee text; current R15 fee follows explicit user revisions and frozen current builders, no rerouting conflict'))
 checks['route_policy']=True
 assert first_cost(38,39,45)==86.75 and first_cost(38,39,83)==381.25
 assert actual(150,23,92,95)['qualified'] and not actual(150,23.25,92,95)['qualified'];assert actual(np.nan,np.inf,0,3,False)['available_ac']==0
 x=forecast(300,9.2,40,47);y=forecast(300,9.2,42,47);assert x['battery_ac']==y['battery_ac'] and x['power_ac']!=y['power_ac'];checks['R10_R15_boundary']=True
 from r30_replay import replay
 # Actual replay implementation: right endpoint enters next state; no path replacement.
 rows=[dict(route_id=0,d=1,departure_dc=500,route=dict(distance_km=1.))];geo=dict(arc_index=np.array([[0]]),arc_count=np.array([1]),remaining_fraction=np.array([[1.]]))
 a=dict(times_h=np.arange(97)/4,road_movable=np.ones((97,1),bool),road_blocked=np.zeros((97,1),bool),road_travel_min=np.ones((97,1))*15)
 rr=replay(rows,geo,a);assert rr['entry_state_index'][0,0]==1 and rr['arrival_h'][0]==.5
 a['road_movable'][1]=False;temp=replay(rows,geo,a);assert not temp['success'][0] and temp['failure_code'][0]==4 and np.isinf(temp['arrival_h'][0]);a['road_blocked'][1]=True;perm=replay(rows,geo,a);assert perm['failure_code'][0]==3;checks['entry_boundary_permanent_temporary_no_reroute']=True
 checks['wrong_forecast_handle']=negative(lambda:scene_path('FORECAST_SELECTED',44));checks['missing_scene']=negative(lambda:scene_path('TRAIN',400));checks['L13_rejected']=negative(lambda:layout('L13'))
 # Native shared-first-stage models, two and five real physical scenes, all original network rows.
 ids=read(R/'tests/REAL_SCENE_SELECTION.json')['ids'];qs0=routes();chosen=[]
 for e,c in [(0,1),(0,5),(1,1),(1,5)]:
  chosen.append(next(q for q in qs0 if q['e']==e and q['c']==c and q['d']==0))
 shared=[]
 for count in [2,5]:
  dest=out/f'SHARED_{count}';dest.mkdir(exist_ok=True)
  if (dest/'COMPLETE.json').exists():shared.append(read(dest/'COMPLETE.json'));continue
  m=gp.Model('R36_SHARED_REAL_'+str(count));parameters(m,dest/'NATIVE.log',1);m.Params.MIPGapAbs=1e-6
  fs=add_first(m,chosen,30,8,full_bounds(),starts=[0,55,92],ends=[4,59,96]);theta=m.addVar(name='TEST_WORST_RISK');rcs=[]
  for sid in ids[:count]:rc=add_recourse(m,fs,layout()['net'],load_scene('TRAIN',sid),route_states('L08',chosen,'TRAIN',sid),sid);m.addConstr(theta>=rc['objective']);rcs.append(rc)
  m.setObjective(fs['objective']+theta);m.update();assert sum(v.VarName.startswith('route[') for v in m.getVars())==4 and sum(v.VarName.startswith('E[') for v in m.getVars())==30
  for rc in rcs:assert all(v.VType==GRB.CONTINUOUS for v in rc['y'].tolist())
  dump(dest/'BEGIN.json',dict(epoch=time.time(),variables=m.NumVars,rows=m.NumConstrs,first_stage_shared=True));m.optimize();dump(dest/'END.json',dict(status=m.Status,runtime=m.Runtime,objective=m.ObjVal if m.SolCount else None,bound=m.ObjBound,epoch=time.time()));assert m.Status==GRB.OPTIMAL
  from compact_first_stage import decode
  pp=decode(fs);verify_plan(pp);pfix=np.zeros(30)
  for h in pp:pfix[h['e']]=h['contract']
  for sid,rc in zip(ids[:count],rcs):
   ts=[task(next(q for q in chosen if q['route_id']==h['route_id']),h['k'],h['r']) for h in pp];mat=assemble(layout()['net'],ts,load_scene('TRAIN',sid),route_states('L08',ts,'TRAIN',sid),30,full_bounds());assert mat.residual(lift(rc['y'].X,rc['pfmap']),np.r_[np.ones(len(ts)),pfix])<=1e-6
  rec=dict(passed=True,scenes=count,objective=m.ObjVal,selected=len(pp),shared_route_binaries=4,shared_E_variables=30,first_stage_per_scene_copies=0);dump(dest/'COMPLETE.json',rec);shared.append(rec);m.dispose()
 checks['native_shared2_shared5']=all(x['passed'] for x in shared)
 checks['required_FINAL_survives_local_failure']=lifecycle_test()['passed']
 # Real duplicate owner is rejected without changing the current lease.
 before=sha(LEASE);cmd=[sys.executable,'-B','-c',"import sys;sys.path.insert(0,"+repr(str(R/'tools'))+");from r36_common import *\nwith admission('duplicate',1):raise RuntimeError('BAD_DUPLICATE_ADMITTED')"]
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE);so,se=p.communicate();dump(out/'DUPLICATE_WAIT.json',dict(actual_parent_wait=True,returncode=p.returncode,stderr=se.decode()));assert p.returncode!=0 and b'DUPLICATE_R36_OWNER' in se and sha(LEASE)==before;checks['unique_owner_actual_reject']=True
 # Pure observer paths must run with optimizer construction and subprocess dispatch forbidden.
 import run_l08_k4l4_fullst_dro_parallel_r36 as entry
 oldmodel,oldread,oldpopen=gp.Model,gp.read,subprocess.Popen
 def forbidden(*a,**k):raise AssertionError('OBSERVER_DISPATCH_FORBIDDEN')
 argv=sys.argv.copy()
 try:
  gp.Model=gp.read=subprocess.Popen=forbidden
  for name in ['status','report']:sys.argv=['entry',name];entry.main()
 finally:gp.Model,gp.read,subprocess.Popen=oldmodel,oldread,oldpopen;sys.argv=argv
 checks['pure_observers']=True
 return dict(passed=all(checks.values()),checks=checks,shared_native_models=shared,protection=protected())
