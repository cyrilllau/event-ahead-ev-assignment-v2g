from r36_recourse import *
from r36_control import *
from r36_probability import dual_certificate
from scipy import sparse as sp
from scipy.optimize import linprog
def negative(fn):
 try:fn()
 except (AssertionError,ValueError,KeyError):return True
 raise AssertionError('NEGATIVE_NOT_REJECTED')
def control_tests():
 out=R/'tests/control';out.mkdir(parents=True,exist_ok=True);checks={}
 now=time.time();rr=dict(verified=True,science=SCIENCE,request='old_certified',first_seen=now-601,violations=[(i,float(i+1)) for i in range(15)])
 ask=batch_add([rr,dict(rr,first_seen=now)],{14});assert len(ask)==8 and ask[0]['sid']==13 and ask[0]['first_seen']==rr['first_seen'];checks['batch_age_dedup']=True
 assert not return_ready([],True,now-1000,now) and not return_ready(ask,False,now-200,now) and return_ready(ask,False,now-301,now)
 assert next_action(0,ask,GRB.INTERRUPTED)=='ADD_AFTER_RETURN';checks['no_incumbent_add']=True
 # Native simplex stage, with a historically completed request; no time-limit interrupts.
 m=gp.Model('R36_ROOT_CONTROL');m.Params.Threads=1;m.Params.Method=1;m.Params.Presolve=0;m.Params.LogFile=str(out/'ROOT_NATIVE.log');v=m.addMVar(120,ub=1);rng=np.random.default_rng(3601);A=rng.random((80,120));m.addConstr(A@v>=A.sum(1)*.35);m.setObjective(rng.random(120)@v);m.update()
 ctl=Control(lambda:ask);ctl.started=now-301;dump(out/'ROOT_BEGIN.json',dict(epoch=time.time(),historical_request_age_s=601,test_clock_offset_s=301,kind='TRUE_LP_NATIVE_CALLBACK'))
 def root_cb(mm,where):
  if where==GRB.Callback.SIMPLEX:ctl.callback(mm,where)
 m.optimize(root_cb);dump(out/'ROOT_END.json',dict(status=m.Status,observed=sorted(ctl.observed),SolCount=m.SolCount,reason=ctl.reason));assert GRB.Callback.SIMPLEX in ctl.observed and ctl.reason=='ADD_CERTIFIED_BATCH';m.dispose();checks['native_simplex_control']=True
 # True mixed-integer CCG versus full expansion, same shared binary decision.
 demands=np.array([1.,3.,5.,8.]);cost=3.;Q=lambda z:np.maximum(demands-4*z,0)
 def optimize(pool,name):
  mm=gp.Model(name);mm.Params.OutputFlag=1;mm.Params.Threads=1;mm.Params.MIPGap=0;mm.Params.MIPGapAbs=1e-8;mm.Params.LogFile=str(out/(name+'.log'));x=mm.addVar(vtype=GRB.BINARY,name='shared_x');theta=mm.addVar(name='risk')
  for sid in pool:yy=mm.addVar(name='q_'+str(sid));mm.addConstr(yy>=demands[sid]-4*x);mm.addConstr(theta>=yy)
  mm.setObjective(cost*x+theta);mm.optimize();assert mm.Status==GRB.OPTIMAL;rec=dict(value=mm.ObjVal,x=round(x.X),theta=theta.X,bound=mm.ObjBound,pool=pool);mm.dispose();return rec
 full=optimize(list(range(4)),'FULL_SMALL');pool=[0];history=[]
 while True:
  sol=optimize(pool.copy(),'CCG_'+str(len(pool)));q=Q(sol['x']);pr=linprog(-q,A_eq=np.ones((1,4)),b_eq=[1],bounds=(0,None),method='highs',options={'threads':1});assert pr.success
  sid=max(range(4),key=lambda s:(q[s]-sol['theta'],-s));history.append(dict(sol,risk=-pr.fun))
  if q[sid]-sol['theta']<=1e-8:break
  assert sid not in pool;pool.append(sid)
 assert abs(sol['value']-full['value'])<1e-8;checks['true_MIP_true_probability_LP_CCG']=True
 # Signed and split dual on negative moments, with genuinely free signed coordinates.
 phi=np.array([[-2.,0.,1.,3.],[1.,0.,1.,0.]]);lo=np.array([-.4,.2]);hi=np.array([.8,.8]);q=np.array([2.,7.,3.,5.]);pr=linprog(-q,A_ub=np.r_[phi,-phi],b_ub=np.r_[hi,-lo],A_eq=np.ones((1,4)),b_eq=[1],bounds=(0,None),method='highs');assert pr.success
 objectives=[]
 for signed in [False,True]:
  mm=gp.Model('R36_DUAL_TEST');mm.Params.OutputFlag=0;mm.Params.Threads=1;b0=mm.addVar(lb=-GRB.INFINITY)
  if signed:
   g=mm.addMVar(2,lb=-GRB.INFINITY);t=mm.addMVar(2,lb=-GRB.INFINITY);mm.addConstr(t>=lo*g);mm.addConstr(t>=hi*g);mm.addConstr(b0+phi.T@g>=q);mm.setObjective(b0+t.sum())
  else:
   bp=mm.addMVar(2);bm=mm.addMVar(2);mm.addConstr(b0+phi.T@(bp-bm)>=q);mm.setObjective(b0+hi@bp-lo@bm)
  mm.optimize();assert mm.Status==GRB.OPTIMAL;objectives.append(mm.ObjVal);mm.NumStart=2;mm.update()
  for slot in range(2):mm.Params.StartNumber=slot;mm.setAttr('Start',mm.getVars(),[1.]*mm.NumVars)
  clear_starts(mm);assert mm.NumStart==0;mm.dispose()
 assert max(abs(x+pr.fun) for x in objectives)<1e-7;checks['negative_load_split_signed_and_clear_all_starts']=True
 checks['wrong_risk_identity']=negative(lambda:batch_add([dict(rr,science='OLD_K1')],set()))
 checks['source_local_cache_conflict']=negative(lambda:validate_replay_metadata(dict(cache_key='OLD_K1',sha256='x'),'R36_forecast44_source44','x'))
 for bad in [dict(route_id=-1,e=0),dict(plans()[0],c=7),dict(plans()[0],contract=1e6)]:checks['bad_plan_'+str(len(checks))]=negative(lambda b=bad:verify_plan([b]))
 dump(out/'SMALL_CCG.json',dict(full=full,history=history))
 return dict(passed=all(checks.values()),checks=checks,native_root_observed=sorted(ctl.observed),native_phase_not_observed=[n for n in ['BARRIER','MIPSOL','MIPNODE'] if getattr(GRB.Callback,n) not in ctl.observed])
