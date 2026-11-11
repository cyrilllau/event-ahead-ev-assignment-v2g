from r30_model import *
from lp_certificate import certify

def gate():
 t=read(R/'06_TESTS/COMPLETE.json');assert t['passed']
 for n,h in t['source_hashes'].items():assert sha(R/n)==h,('TESTED_SOURCE_CHANGED',n)
 assert read(R/'05_SUPPORT400_ROUTE_REPLAY/COMPLETE.json')['support_count']==400

def build_det():
 gate();out=R/'04_MODEL_AND_NATIVE_SOLVE';out.mkdir(exist_ok=True)
 qs=routes();net=layout()['net'];a=load_scene('FORECAST_SELECTED');M=full_bounds();st=route_states('L08',qs)
 t=time.monotonic();m,fs,rc=construct(qs,30,net,a,st,M,'L08_R30_FORECAST44_DET')
 original=rc['original'];cert=read(R/'06_TESTS/ZERO_INPUT_CERTIFICATE.json');assert cert['witness_sha']==sha(R/'06_TESTS/ZERO_INPUT_WITNESS.npz');zero=loadnp(R/'06_TESTS/ZERO_INPUT_WITNESS.npz')['y'];assert original.residual(zero,np.zeros(30))<1e-6
 assert np.max(abs(zero[original.index['p']]))<1e-8
 # A fresh model contains only this start. Every retained variable is initialized coherently.
 for v in m.getVars():v.Start=GRB.UNDEFINED
 for v in m.getVars():v.Start=0.
 rc['y'].Start=zero[rc['pfmap']['keep']];m.update()
 m.write(str(out/'DET.mps.gz'));m.write(str(out/'ZERO.mst'));dump(out/'VARIABLE_NAMES.json',m.getAttr('VarName'));dump(out/'VARIABLE_TYPES.json',m.getAttr('VType'))
 mp=rc['pfmap'];assert mp['applicable'];npz(out/'PF_LIFT.npz',keep=mp['keep'],q_columns=mp['q_columns'],p_columns=mp['p_columns']);npz(out/'ZERO_ORIGINAL.npz',y=zero)
 dump(out/'ZERO_CERTIFICATE.json',cert);dump(out/'BUILD_COMPLETE.json',dict(passed=True,model_sha=sha(out/'DET.mps.gz'),start_sha=sha(out/'ZERO.mst'),variables=m.NumVars,binaries=m.NumBinVars,rows=m.NumConstrs,nonzeros=m.NumNZs,route_binary_candidates=len(qs),logical_tasks=read(R/'04_R10_COMPACT_DOMAIN/COMPLETE.json')['logical_tasks'],copies=1,risk_planes=0,threads=1,elapsed_s=time.monotonic()-t,PF_proof=mp['reason'],ZERO_qU=cert['qU'],warmstart='fresh complete ZERO',identity=read(R/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json')));m.dispose();print('BUILD_DET_COMPLETE',flush=True)

def solve_det():
 gate();out=R/'04_MODEL_AND_NATIVE_SOLVE';bc=read(out/'BUILD_COMPLETE.json');assert sha(out/'DET.mps.gz')==bc['model_sha']
 assert not (out/'SOLVE_COMPLETE.json').exists(),'Native completion already exists; no automatic cold restart'
 prior=out/'NATIVE_BEGIN.json'
 if prior.exists():raise RuntimeError('EXISTING_NATIVE_ATTEMPT_REQUIRES_IDENTITY_AND_PARENT_WAIT_REVIEW')
 m=gp.read(str(out/'DET.mps.gz'));parameters(m,out/'NATIVE.log');m.read(str(out/'ZERO.mst'))
 assert m.getAttr('VarName')==read(out/'VARIABLE_NAMES.json')
 from r30_control import ctrl
 dump(prior,dict(epoch=time.time(),owner=ctrl.proc(os.getpid()),model_sha=bc['model_sha'],threads=1,TimeLimit='Infinity',MIPGapAbs=1.,SoftMemLimit=m.Params.SoftMemLimit,solver_version=gp.gurobi.version(),branch_tree_restored=False))
 last=[0.];best=[float('inf')];candidates=out/'candidates';candidates.mkdir(exist_ok=True)
 def cb(mod,where):
  try:
   now=time.time();d=dict(epoch=now,where=where)
   if where==GRB.Callback.MIPSOL:
    val=float(mod.cbGet(GRB.Callback.MIPSOL_OBJ))
    if val<best[0]-1e-8:
     best[0]=val;name=str(time.time_ns());npz(candidates/(name+'.npz'),values=np.array(mod.cbGetSolution(mod.getVars())));dump(candidates/(name+'.json'),dict(epoch=now,objective=val,phase='MIPSOL',raw_sha=sha(candidates/(name+'.npz')),certified=False));print('CANDIDATE',val,flush=True)
   if now-last[0]<30:return
   last[0]=now
   if where==GRB.Callback.MIP:d.update(phase='MIP',native_LB=float(mod.cbGet(GRB.Callback.MIP_OBJBND)),native_UB=float(mod.cbGet(GRB.Callback.MIP_OBJBST)),nodes=float(mod.cbGet(GRB.Callback.MIP_NODCNT)))
   elif where==GRB.Callback.SIMPLEX:d.update(phase='SIMPLEX',iterations=float(mod.cbGet(GRB.Callback.SPX_ITRCNT)),simplex_objective_not_MIP_LB=float(mod.cbGet(GRB.Callback.SPX_OBJVAL)))
   elif where==GRB.Callback.PRESOLVE:d.update(phase='PRESOLVE',rows_removed=int(mod.cbGet(GRB.Callback.PRE_ROWDEL)),columns_removed=int(mod.cbGet(GRB.Callback.PRE_COLDEL)))
   else:return
   for k,v in list(d.items()):
    if isinstance(v,float) and (not np.isfinite(v) or abs(v)>=1e99):d[k]=None
   dump(out/'PROGRESS.json',d)
  except Exception as exc:
   # Reporting cannot mutate or terminate the live scientific model.
   with (out/'CALLBACK_REPORT_ERRORS.log').open('a') as f:f.write(repr(exc)+'\n')
 started=time.monotonic();m.optimize(cb);rec=dict(status=m.Status,solutions=m.SolCount,elapsed_s=time.monotonic()-started,epoch=time.time(),solver_LB=float(m.ObjBound) if np.isfinite(m.ObjBound) and abs(m.ObjBound)<1e99 else None,solver_UB=float(m.ObjVal) if m.SolCount else None,runtime=m.Runtime,NodeCount=m.NodeCount,iterations=m.IterCount)
 dump(out/'NATIVE_END.json',rec)
 if m.SolCount:
  npz(out/'NATIVE_VALUES.npz',values=np.array(m.getAttr('X')));m.write(str(out/'NATIVE.sol'));rec.update(values_sha=sha(out/'NATIVE_VALUES.npz'),integer_residual=max(abs(v.X-round(v.X)) for v in m.getVars() if v.VType in [GRB.BINARY,GRB.INTEGER]))
 rec['passed']=bool(m.SolCount and rec['solver_LB'] is not None);dump(out/'SOLVE_COMPLETE.json',rec);print('NATIVE_END',json.dumps(rec),flush=True);m.dispose()
 if not rec['passed']:raise RuntimeError('DET_NO_CERTIFIABLE_NATIVE_CANDIDATE')
