"""One deadline and explicit allowed stages, including worker native calls."""
import time,uuid,os,traceback
import numpy as np
import gurobipy as gp
from repair_guard_helpers import safe_native_observations

def optimize(model,kind,context,extra=None):
 import engine as e
 d=e.d;rt=e.rt;root=e.OUT
 stages={'FULL_DOMAIN_GLOBAL_SUPPORT':'RECOURSE','FULL400_PROBABILITY':'RISK','FIXED_INTEGER_REPAIR':'POINT_REPAIR'}
 if kind not in stages:raise ValueError('UNREGISTERED_NATIVE_STAGE '+kind)
 if e.PREFLIGHT:raise RuntimeError('PREFLIGHT_OPTIMIZER_FORBIDDEN')
 if time.monotonic()>=e.DEADLINE or (root/'USER_STOP.json').exists() or (root/'PAUSE_RESOURCE.json').exists():raise TimeoutError('COOPERATIVE_STOP_BEFORE_OPTIMIZE')
 cid=uuid.uuid4().hex;call=root/'CALLS'/cid;call.mkdir(parents=True)
 ctx=dict(e.CONTEXT,**context);begin=dict(id=cid,phase='BEGIN',stage=stages[kind],kind=kind,context=ctx,epoch=time.time(),owner=rt.proc(os.getpid()),remaining=e.DEADLINE-time.monotonic())
 d.atomic(call/'BEGIN.json',begin,True);rt.append_event(root/'OPTIMIZER_CALL_LEDGER.jsonl',begin)
 model.Params.TimeLimit=min(float(model.Params.TimeLimit),max(.001,e.DEADLINE-time.monotonic()))
 error=None;last=[0.]
 def callback(m,where):
  now=time.monotonic()
  if now>=e.DEADLINE:m.terminate();return
  if now-last[0]>1:
   last[0]=now
   if (root/'USER_STOP.json').exists() or (root/'PAUSE_RESOURCE.json').exists():m.terminate()
 try:model.optimize(callback)
 except BaseException as exc:error=exc
 ob=safe_native_observations(model,infinity=gp.GRB.INFINITY)
 val=lambda key:ob.get(key,{}).get('value')
 end=dict(begin,phase='END',epoch=time.time(),status=val('Status'),runtime=val('Runtime'),work=val('Work'),max_mem_GB=val('MaxMemUsed'),observations=ob,exception=None if error is None else repr(error))
 d.atomic(call/'NATIVE_END.json',end,True)
 try:
  raw={}
  for name in ('X','Pi','RC','VBasis','CBasis'):
   try:
    v=np.asarray(model.getAttr(name))
    if not np.isfinite(v).all():raise ValueError('NONFINITE_RAW_'+name)
    raw[name]=v
   except (gp.GurobiError,AttributeError):pass
  end['raw']=d.publish_npz(call/'RAW_STATE.npz',**raw)
 except BaseException as exc:
  end['payload_exception']=repr(exc)
  if error is None:error=exc
 d.atomic(call/'END.json',end,True);rt.append_event(root/'OPTIMIZER_CALL_LEDGER.jsonl',end)
 if error is not None:raise error
 return cid
