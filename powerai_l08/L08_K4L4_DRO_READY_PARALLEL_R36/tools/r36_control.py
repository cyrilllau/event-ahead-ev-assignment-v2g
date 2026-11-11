"""One optimizer writer. Files contain immutable audited requests, never model handles."""
from r36_common import *
import gurobipy as gp
from gurobipy import GRB
def batch_add(requests,embedded,limit=8):
 merged={}
 for r in requests:
  assert r['verified'] and r['science']==SCIENCE
  for sid,res in r['violations']:
   assert isinstance(sid,int) and 0<=sid<400 and np.isfinite(res)
   if sid in embedded or res<=1e-4:continue
   if sid not in merged:merged[sid]=dict(sid=sid,residual=res,first_seen=r['first_seen'],request=r['request'])
   else:
    merged[sid]['first_seen']=min(merged[sid]['first_seen'],r['first_seen']);merged[sid]['residual']=max(merged[sid]['residual'],res)
 return sorted(merged.values(),key=lambda r:(-r['residual'],r['sid']))[:limit]
def return_ready(asks,root_ready,epoch_started,now,root_started=None,max_wait_s=3600.,root_grace_s=1200.):
 # Wall clock alone is not root progress. Keep request ages immutable.
 # This is an ADD latency guard, NOT a solver TimeLimit.
 if not asks:return False
 if root_ready:return True
 elapsed=now-epoch_started
 # Give the first actual root iterations a sustained interval. Presolve does not count.
 if root_started is not None and now-root_started>=root_grace_s:return True
 # Preserve bounded handling of certified ADDs even if presolve/root stalls.
 return elapsed>=max_wait_s and now-min(a['first_seen'] for a in asks)>=600
def next_action(solcount,asks,status):
 if asks:return 'ADD_AFTER_RETURN'
 if status==GRB.MEM_LIMIT:return 'PAUSED_RESOURCE'
 if status in [GRB.NUMERIC,GRB.INF_OR_UNBD,GRB.INFEASIBLE,GRB.UNBOUNDED]:return 'BLOCKED_NATIVE'
 if solcount==0:return 'PAUSED_NO_INTEGER'
 return 'AUDIT_FINAL'
def clear_starts(m):
 m.update()
 for i in range(m.NumStart):
  m.Params.StartNumber=i;m.setAttr('Start',m.getVars(),[GRB.UNDEFINED]*m.NumVars)
 m.NumStart=0;m.update()
def set_start(m,fs,plan,risk,certificate):
 clear_starts(m);m.NumStart=1;m.Params.StartNumber=0;m.setAttr('Start',m.getVars(),[GRB.UNDEFINED]*m.NumVars)
 by={h['e']:h for h in plan}
 for i,q in enumerate(fs['qs']):fs['v'][i].Start=float(q['e'] in by and q['route_id']==by[q['e']]['route_id'])
 for e in range(fs['Ecount']):
  h=by.get(e)
  for k in fs['starts']:fs['s'][e,k].Start=float(h is not None and k==h['k'])
  for j in fs['ends']:fs['f'][e,j].Start=float(h is not None and j==h['r']+1)
  fs['u'][e].Start=float(h is not None);fs['E'][e].Start=h['contract'] if h else 0;fs['K'][e].Start=h['k']/4 if h else 0;fs['J'][e].Start=(h['r']+1)/4 if h else 0
  for t in range(96):fs['w'][e,t].Start=float(h is not None and h['k']<=t<=h['r'])
  for c in range(fs['Ccount']):
   fs['b'][e,c].Start=float(h is not None and h['c']==c)
   for t in range(96):fs['wc'][e,c,t].Start=float(h is not None and h['c']==c and h['k']<=t<=h['r'])
 risk['beta0'].Start=float(certificate['beta0']);risk['plus'].Start=certificate['plus'];risk['minus'].Start=certificate['minus'];m.update()
 return dict(partial_start=True,unknown_auxiliaries='GRB.UNDEFINED',all_previous_slots_cleared=True,probability_scaled_coordinates=True)
class Control:
 def __init__(self,poll):self.poll=poll;self.started=time.time();self.root_ready=False;self.root_started=None;self.reason=None;self.bound=None;self.asks=[];self.observed=set();self.last_poll=0
 def callback(self,m,where):
  self.observed.add(int(where));now=time.time()
  if where==GRB.Callback.MIP:
   v=m.cbGet(GRB.Callback.MIP_OBJBND)
   if np.isfinite(v) and abs(v)<1e90:self.bound=max(self.bound if self.bound is not None else -np.inf,float(v)-1e-6)
   if m.cbGet(GRB.Callback.MIP_NODCNT)>0:self.root_ready=True
  elif where==GRB.Callback.MIPNODE:
   if m.cbGet(GRB.Callback.MIPNODE_STATUS)==GRB.OPTIMAL:self.root_ready=True
  if where in [GRB.Callback.SIMPLEX,GRB.Callback.BARRIER,GRB.Callback.MIPNODE]:
   if self.root_started is None:self.root_started=now
  # No SPX objective is ever treated as an integer bound.
  if now-self.last_poll>=1:
   self.asks=self.poll();self.last_poll=now
  if return_ready(self.asks,self.root_ready,self.started,now,self.root_started):self.reason='ADD_CERTIFIED_BATCH';m.terminate()
