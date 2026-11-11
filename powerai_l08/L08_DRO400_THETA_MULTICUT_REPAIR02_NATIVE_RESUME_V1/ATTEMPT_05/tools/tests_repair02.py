from pathlib import Path
from types import SimpleNamespace
import time,math,hashlib,json
import numpy as np
import gurobipy as gp
import native_repair02 as nr
import recovery02 as rc

def run(o,d,native,commit):
 records=[]
 def ok(name,cond=True):
  if not cond:raise AssertionError(name)
  records.append(dict(name=name,passed=True))
 def reject(name,f):
  try:f()
  except (ValueError,AssertionError):ok(name);return
  raise AssertionError('NOT_REJECTED_'+name)
 C=gp.GRB.Callback
 class Clock:
  now=100.
  def __call__(self):return self.now
 class Fake:
  def __init__(self):self.where=None;self.queries=[];self.terminated=False;self.bad=False;self.nodes=0;self.v=np.array([1.,2.,5.,3.])
  def cbGet(self,what):
   self.queries.append(what)
   if self.bad:raise RuntimeError('injected query error')
   legal={C.MIP:{C.MIP_NODCNT:self.nodes,C.MIP_OBJBND:1.,C.MIP_PHASE:1},C.MIPSOL:{C.MIPSOL_OBJ:6.}}
   if what not in legal.get(self.where,{}):raise RuntimeError('illegal callback query')
   return legal[self.where][what]
  def cbGetSolution(self,v):
   self.queries.append('SOLUTION');assert self.where==C.MIPSOL;return self.v.copy()
  def cbGetNodeRel(self,v):raise AssertionError('UNUSED_NODE_RELAXATION_QUERIED')
  def terminate(self):self.terminated=True
 def factory(known=None,stop=None,emit=None):
  clock=Clock();m=Fake();cb=nr.Callback(list(range(4)),2,{0:3},known or {},1000,stop or (lambda:None),emit or (lambda v:None),'INJECT',1,'MASTER_MIP',clock)
  def invoke(w):m.where=w;cb(m,w)
  return m,cb,clock,invoke
 for stop in (None,'USER_REQUESTED_STOP','RESOURCE_PAUSE','DEADLINE'):
  m,cb,clock,call=factory(stop=lambda:None if stop=='DEADLINE' else stop)
  if stop=='DEADLINE':cb.deadline=0
  call(C.POLLING);ok('POLLING_ZERO_QUERIES_'+str(stop),not m.queries and m.terminated==bool(stop))
 for w,n in nr.WHERE.items():
  m,cb,clock,call=factory();call(w);ok('EVENT_QUERY_CONTRACT_'+n,cb.error is None)
  if w not in (C.MIP,C.MIPSOL):ok('UNUSED_EVENT_ZERO_QUERIES_'+n,not m.queries)
 m,cb,clock,call=factory();call(98765);ok('UNKNOWN_NO_QUERY',not m.queries)
 call(C.MIPNODE);ok('ROOT_NODE_OPTIMAL_NOT_ROOT_COMPLETE',cb.root_end is None and not m.queries)
 key=nr.point_sha(np.array([1.,2.]));known={key:dict(risk_L=5.,qL=np.array([3.]))}
 m,cb,clock,call=factory(known);call(C.MIPSOL);m.nodes=1;call(C.MIP);clock.now+=121;call(C.MIP);ok('START_ONLY_NO_HANDOVER',not cb.queue and not m.terminated)
 m,cb,clock,call=factory(known);m.v[2]=4.;call(C.MIPSOL);m.nodes=1;call(C.MIP);clock.now+=121;call(C.MIP);ok('KNOWN_POINT_RISK_UNDERESTIMATE_ROUTED',bool(cb.queue) and m.terminated)
 m,cb,clock,call=factory(known);m.v[3]=2.;call(C.MIPSOL);ok('KNOWN_POINT_THETA_UNDERESTIMATE_ROUTED',bool(cb.queue))
 m,cb,clock,call=factory();call(C.MIPSOL);clock.now+=200;call(C.MIP);ok('NO_ROOT_HANDOVER',not m.terminated)
 m.bad=True;call(C.MIP);first=dict(cb.error);call(C.MIPSOL);ok('FIRST_ERROR_PRESERVED',cb.error==first and all(k in first for k in ('call_id','math_revision','where','where_name','query_name','exception')))
 def ioerror(v):raise OSError('temporary telemetry failure')
 m,cb,clock,call=factory(emit=ioerror);call(C.MIP);ok('TELEMETRY_FAILURE_NOT_SCIENCE_FAILURE',cb.error is None and not m.terminated and len(cb.telemetry_errors)==1)
 reject('REJECT_OLD_9016_INDEX',lambda:rc.assert_counts({s:set(oops) for s,oops in {0:range(9016)}.items()},set(range(48))))
 reject('REJECT_WRONG_RAW_MATRIX',lambda:rc.bind_raw(d['A'],d['a'],dict(d['ident'],matrix_sha='WRONG'),dict(revision=1)))
 best=np.load(o.ac.rt.verify_ref(d['best']['Q']))['x'];binary=d['a']['vtype'][:o.N]=='B';ok('SAME_BINARY_DIFFERENT_E_CACHE_MISS',np.max(abs(d['raw'][:o.N][binary]-best[binary]))<=1e-9 and nr.point_sha(d['raw'][:o.N])!=nr.point_sha(best))
 ok('REV1_9253_48',sum(map(len,d['support_ids'].values()))==9253 and len(d['pids'])==48)
 ok('STRICT_UB_RECHECK',o.tk.external_gap(99,100)['target'] and not o.tk.external_gap(99,200)['target'])
 # Same structured native/commit chain under unavailable/NaN/exception attributes.
 class Model:
  def __init__(self,mode):
   self.mode=mode;self.Params=SimpleNamespace(Cutoff=gp.GRB.INFINITY,SolutionLimit=2000000000);self.IsMIP=1;self.NumVars=3;self.NumConstrs=1;self.Status=11;self.SolCount=1 if mode=='callback' else 0;self.Runtime=.01;self.Work=0.;self.MaxMemUsed=0.;self.NodeCount=0.;self.ObjVal=2.
  def update(self):pass
  def getVars(self):return [0,1,2]
  def getParamInfo(self,n):return (n,int,2000000000,1,2000000000,2000000000)
  @property
  def ObjBoundC(self):
   if self.mode=='nan':return float('nan')
   if self.mode=='callback':return 1.
   raise AttributeError('no bound')
  def getAttr(self,n):
   if n=='X' and self.SolCount:return [1.,0.,1.]
   raise AttributeError(n)
  def optimize(self,cb):
   if self.mode=='callback':cb(self,C.MIP)
  def cbGet(self,k):raise RuntimeError('actual callback test query exception')
  def terminate(self):pass
 for mode in ('missing','nan','callback'):
  end,raw=native(Model(mode),'TEST',o.OUT/'TESTS'/('MOCK_'+mode),nfirst=2,mapping={},known={},seconds=20,simulated=True)
  status=commit(end,folder='TESTS/MOCK_'+mode)
  ok('SAFE_END_'+mode,(o.OUT/'TESTS'/('MOCK_'+mode)/'END.json').exists() and 'Pi'not in raw)
  if mode!='callback':ok('NO_BOUND_IS_NULL_'+mode,end['bound'] is None and not end['has_X'])
  else:ok('CALLBACK_ERROR_COMMITTED_BEFORE_STOP',status=='IMPLEMENTATION_ERROR' and (o.OUT/'TESTS/MOCK_callback/LAST_RETURNED.json').exists() and end['has_X'])
 # Exactly one genuine MILP, full Start, same native and result-commit functions.
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=gp.Model('REPAIR02_NATIVE_SMOKE',env=env)
 try:
  x=m.addMVar(2,lb=0,ub=1,vtype='B',obj=[1.,2.]);rho=m.addVar(lb=0,obj=1);m.addConstr(x.sum()>=1);m.addConstr(rho>=3-2*x[0]-x[1]);m.update();m.NumStart=1;m.Params.StartNumber=0;m.setAttr('Start',m.getVars(),[1.,1.,0.]);m.update()
  end,raw=native(m,'TEST',o.OUT/'TESTS/NATIVE_SMOKE',nfirst=2,mapping={},known={},seconds=60)
  status=commit(end,folder='TESTS/NATIVE_SMOKE');ok('REAL_NATIVE_SMOKE',status=='RETURNED' and end['status']==2 and 'X'in raw and end['bound'] is not None)
  ok('REAL_MIP_NO_FAKE_PI_BASIS',all(k not in raw for k in ('Pi','RC','VBasis','CBasis')))
  o.write('TESTS/NATIVE_SMOKE_RESULT.json',dict(passed=True,actual_native_calls=1,observed=end['observed_where'],not_observed=end['unobserved_where'],result=end),True)
 finally:m.dispose();env.dispose()
 o.write('CALLBACK_TEST_REPORT.json',dict(passed=True,tests=records,actual_small_MILP_calls=1,simulated_wrapper_calls=3,old_tests_repeated=False,same_production_functions=True,scope='callback/native/commit and actual revision1 restore; not a production optimization result'),True)
 return records
