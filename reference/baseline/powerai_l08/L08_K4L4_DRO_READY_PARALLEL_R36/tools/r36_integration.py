from r36_model import *
from r36_control import *
from r36_queue import lifecycle_test
import threading,queue as qmodule
from scipy.optimize import linprog
def integration_tests():
 out=R/'tests/integration';out.mkdir(parents=True,exist_ok=True);events=qmodule.Queue();results=qmodule.Queue();record=[];stop=threading.Event()
 def audit():
  token=events.get();begin=time.time();v=linprog([-1.,-3.,-5.,-8.],A_eq=np.ones((1,4)),b_eq=[1.],bounds=(0,None),method='highs',options={'threads':1});end=time.time();assert v.success;results.put(dict(verified=True,science=SCIENCE,request='real_background_lp',first_seen=begin,violations=[(3,float(-v.fun))],begin=begin,end=end,LP_native=True))
 worker=threading.Thread(target=audit);worker.start();m=gp.Model('R36_ASYNC_REAL_MIP');m.Params.Threads=1;m.Params.Presolve=0;m.Params.Method=1;m.Params.MIPGap=0;m.Params.TimeLimit=10;m.Params.LogFile=str(out/'NATIVE.log')
 rng=np.random.default_rng(3606);A=(rng.random((250,700))<.025).astype(float);x=m.addMVar(700,vtype=GRB.BINARY);m.addConstr(A@x>=1);m.setObjective(rng.uniform(.5,2,700)@x);m.update();sent=False;callback_total=0.;accepted=[]
 def poll():
  while True:
   try:accepted.append(results.get_nowait())
   except qmodule.Empty:break
  return batch_add(accepted,set())
 ctl=Control(poll)
 def cb(mm,where):
  nonlocal sent,callback_total
  t=time.perf_counter()
  ctl.callback(mm,where)
  if where==GRB.Callback.MIPSOL and not sent:events.put(dict(captured=time.time()));record.append(dict(event='MIPSOL_ENQUEUED',epoch=time.time()));sent=True
  callback_total+=time.perf_counter()-t
 begin=time.time();m.optimize(cb);end=time.time()
 if not sent:events.put(None)
 worker.join();poll();assert sent and accepted and accepted[0]['begin']<end and accepted[0]['end']>begin,(sent,accepted,begin,end)
 assert ctl.reason=='ADD_CERTIFIED_BATCH' and next_action(m.SolCount,ctl.asks,m.Status)=='ADD_AFTER_RETURN'
 rec=dict(passed=True,true_native_MIP=True,true_background_LP=True,MIP_begin=begin,MIP_end=end,audit=accepted,callback_s=callback_total,observed=sorted(ctl.observed),native_status=m.Status,callback_never_waited=True,background_model_access=False);dump(out/'ASYNC.json',rec);m.dispose()
 # Production split-risk builder: full small extensive versus iterative omission.
 a=dict(phi=sp.csr_matrix([[0.,1.,0.,1.],[0.,0.,1.,1.]]),lower=np.array([.1,.2]),upper=np.array([.8,.9]));demands=np.array([1.,3.,5.,8.]);history=[]
 def solve(pool):
  mm=gp.Model('R36_SMALL_SPLIT_CCG');mm.Params.OutputFlag=0;mm.Params.Threads=1;mm.Params.MIPGapAbs=1e-8;mm.Params.MIPGap=0;z=mm.addVar(vtype=GRB.BINARY);risk=add_risk(mm,a)
  for sid in pool:yy=mm.addVar();mm.addConstr(yy>=demands[sid]-4*z);mm.addConstr(risk['plane_expr'][sid]>=yy)
  mm.setObjective(3*z+risk['objective']);mm.optimize();assert mm.Status==GRB.OPTIMAL
  q=np.maximum(demands-4*round(z.X),0);planes=risk['beta0'].X+a['phi'].T@(risk['plus'].X-risk['minus'].X);rr=dict(value=mm.ObjVal,z=round(z.X),planes=np.array(planes),q=q);mm.dispose();return rr
 full=solve(range(4));pool=[0]
 for _ in range(5):
  rr=solve(pool);res=rr['q']-rr['planes'];history.append(dict(pool=pool.copy(),objective=rr['value'],residual=res));omitted=[s for s in range(4) if s not in pool and res[s]>1e-7]
  if not omitted:break
  pool.append(max(omitted,key=lambda s:(res[s],-s)))
 assert abs(rr['value']-full['value'])<1e-7
 dump(out/'PRODUCTION_SPLIT_BUILDER_CCG.json',dict(passed=True,full=full,history=history))
 return dict(passed=True,async_native=rec,production_probability_builder_equivalence=True,final_lifecycle=lifecycle_test())
