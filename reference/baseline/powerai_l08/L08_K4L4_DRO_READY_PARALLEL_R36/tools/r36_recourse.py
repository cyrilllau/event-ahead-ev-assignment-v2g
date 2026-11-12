from r36_common import *
from r30_audit import evaluate_one
from compact_first_stage import add_first,fix_first
from compact_recourse_pf import add_recourse
from exact_transform import lift,add_strengthening
from contract_matrix import assemble,first_stage
import gurobipy as gp
from gurobipy import GRB
def plans():
 p=R/'candidates/DET/PLAN.json'
 if not p.exists():
  src=B/'L08_DET_COUNT_DECOMPOSITION_AND_RESERVATION_LINK_R35/start/CENTER.json';rec=read(src);pp=rec['audit']['physical']['plans'];dump(p,pp);dump(p.with_name('SOURCE.json'),dict(path=str(src),sha256=sha(src),DET_LB_inherited=False,forecast_cost_not_DRO_UB=True))
 return read(p)
def verify_plan(pp):
 qs={q['route_id']:q for q in routes()};seen=set();tasks=[];contracts=np.zeros(30)
 for h in pp:
  q=qs[h['route_id']];assert h['e']==q['e'] and h['e'] not in seen;seen.add(h['e'])
  for k in ['c','d','arrival','distance','departure_dc']:assert h[k]==q[k],k
  hh=task(q,h['k'],h['r']);assert 15-1e-7<=h['contract']<=hh['Ebar']+1e-7;tasks.append(hh);contracts[h['e']]=h['contract']
 A,b,parts,_,_=first_stage(tasks,30,8,full_bounds());z=np.r_[np.ones(len(tasks)),contracts];assert np.max(A@z-b,initial=0)<=1e-6
 return float(parts.sum(0)@z)
def evaluate(pp,sid,out):
 verify_plan(pp);out=Path(out);ph=plan_digest(pp)
 if (out/'COMPLETE.json').exists():
  row=read(out/'COMPLETE.json');ce=read(out/'CERTIFICATE.json');assert row['plan_sha']==ph and row['replay_sha']==sha(verify_replay('TRAIN',sid))
  assert ce['primal_sha']==sha(out/'PRIMAL.npz') and ce['dual_sha']==sha(out/'DUAL_AND_BOUNDS.npz');return row
 if (out/'CERTIFICATE.json').exists():
  ce=read(out/'CERTIFICATE.json');assert ce['plan_sha']==ph and ce['physical_sha']==sha(scene_path('TRAIN',sid)/'physical.npz') and ce['recovery_sha']==sha(scene_path('TRAIN',sid)/'recovery.npz')
  assert ce['primal_sha']==sha(out/'PRIMAL.npz') and ce['dual_sha']==sha(out/'DUAL_AND_BOUNDS.npz')
  pr=loadnp(out/'PRIMAL.npz');du=loadnp(out/'DUAL_AND_BOUNDS.npz');tasks=[task(next(q for q in routes() if q['route_id']==h['route_id']),h['k'],h['r']) for h in pp];n=layout()['net'];a=load_scene('TRAIN',sid);mat=assemble(n,tasks,a,route_states('L08',tasks,'TRAIN',sid),30,full_bounds());y,z=pr['y'],pr['z'];res=mat.residual(y,z);assert res<=1e-7
  lam=du['lambda_ub'];mu=du['mu_eq'];assert np.max(lam,initial=0)<=0;red=mat.cost-mat.A.T@lam-mat.H.T@mu;dual=float(lam@(mat.b-mat.B@z)+mu@(mat.d-mat.G@z)+np.minimum(red*du['lower'],red*du['upper']).sum());assert ce['qL']<=dual+1e-8 and ce['qU']>=mat.cost@y and np.max(y-du['upper'],initial=0)<=1e-7
  _,_,parts,_,_=first_stage(tasks,30,8,full_bounds());F=float(parts.sum(0)@z);crit=np.array(n['shed_cost'])==50;shed=y[mat.index['shed']];pd=y[mat.index['p']];short=y[mat.index['short']]
  row=dict(passed=True,objective_policy=OBJECTIVE_POLICY,bank='TRAIN',scenario=sid,F=F,Q=float(mat.cost@y),qL=ce['qL'],qU=ce['qU'],critical_ENS=float(.25*shed[:,crit].sum()),other_ENS=float(.25*shed[:,~crit].sum()),actual_V2G_ac=float(.25*pd.sum()),shortfall_ac=float(short.sum()),raw_residual=res,cost_parts=mat.parts@y,first_cost_parts=parts@z,plan_sha=ph,physical_sha=ce['physical_sha'],recovery_sha=ce['recovery_sha'],replay_sha=sha(verify_replay('TRAIN',sid)),M_sha=sha(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json'))
  dump(out/'METADATA_REPAIR.json',dict(passed=True,complete_payload_reused=True,LP_repeated=False,epoch=time.time()));dump(out/'COMPLETE.json',row);return row
 return evaluate_one(pp,'TRAIN',sid,out)
def choose_real_scenes():
 metrics=[];qs=routes()
 for sid in range(400):
  a=load_scene('TRAIN',sid);z=loadnp(verify_replay('TRAIN',sid));rid=np.array([q['route_id'] for q in qs]);arrival=z['arrival_h'][rid];forecast_arr=np.array([q['arrival'] for q in qs]);n=float(a['line_operating_outages'].sum()/4)
  metrics.append(dict(sid=sid,line_hours=n,zero=not a['line_operating_outages'].any(),failed=int((~z['success'][rid]).sum()),late=int((z['success'][rid]&(arrival>forecast_arr+1e-9)).sum()),incident_slots=int(incident(layout()['net'],a)[0].any(axis=0).sum())))
 choices={
 'no_fault':next(x['sid'] for x in metrics if x['zero']),
 'short_recovery':min((x for x in metrics if x['line_hours']>0),key=lambda x:(x['line_hours'],x['sid']))['sid'],
 'long_island':max(metrics,key=lambda x:(x['incident_slots'],x['line_hours'],-x['sid']))['sid'],
 'road_failure':max(metrics,key=lambda x:(x['failed'],-x['sid']))['sid'],
 'late_route':max(metrics,key=lambda x:(x['late'],-x['sid']))['sid']}
 ids=list(dict.fromkeys(choices.values()))
 for row in sorted(metrics,key=lambda x:(-x['line_hours'],x['sid'])):
  if len(ids)>=5:break
  if row['sid'] not in ids:ids.append(row['sid'])
 dump(R/'tests/REAL_SCENE_SELECTION.json',dict(choices=choices,ids=ids,metrics=metrics,selection='Fixed physical categories; no rerouting or changing scenarios'))
 return ids
def recourse_tests():
 pp=plans();verify_plan(pp);ids=choose_real_scenes();results=[]
 for sid in ids:
  out=R/'tests/recourse'/f'{sid:04d}';out.mkdir(parents=True,exist_ok=True)
  if (out/'COMPARISON.json').exists():results.append(read(out/'COMPARISON.json'));continue
  explicit=evaluate(pp,sid,out/'EXPLICIT');a=load_scene('TRAIN',sid);qs=routes();st=route_states('L08',qs,'TRAIN',sid)
  m=gp.Model('R36_REAL_RECOURSE_'+str(sid));m.Params.OutputFlag=1;m.Params.LogFile=str(out/'NATIVE.log');m.Params.Threads=1;m.Params.Method=1;m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9
  fs=add_first(m,qs,30,8,full_bounds());add_strengthening(m,fs);fix_first(fs,pp);rc=add_recourse(m,fs,layout()['net'],a,st,sid);m.setObjective(rc['objective']);m.update()
  for v in m.getVars():
   if v.VType==GRB.BINARY:v.VType=GRB.CONTINUOUS
  m.update();dump(out/'BEGIN.json',dict(epoch=time.time(),variables=m.NumVars,rows=m.NumConstrs,owner=ctrl.proc(os.getpid()),kind='FIXED_PLAN_CONTINUOUS_LP'));m.optimize()
  dump(out/'END.json',dict(status=m.Status,runtime=m.Runtime,objective=m.ObjVal if m.SolCount else None,epoch=time.time()))
  assert m.Status==GRB.OPTIMAL
  y=lift(rc['y'].X,rc['pfmap']);tasks=[task(next(q for q in qs if q['route_id']==h['route_id']),h['k'],h['r']) for h in pp];contracts=np.zeros(30)
  for h in pp:contracts[h['e']]=h['contract']
  mat=assemble(layout()['net'],tasks,a,route_states('L08',tasks,'TRAIN',sid),30,full_bounds());zz=np.r_[np.ones(len(tasks)),contracts];res=mat.residual(y,zz);assert res<=1e-6 and abs(m.ObjVal-explicit['Q'])<=1e-4,(sid,res,m.ObjVal,explicit['Q'])
  # Save the full continuous matrix's primal/row dual/bound reduced costs.
  npz(out/'COMPACT_LP.npz',X=np.array(m.getAttr('X')),Pi=np.array(m.getAttr('Pi')),RC=np.array(m.getAttr('RC')),original_y=y)
  m.write(str(out/'COMPACT.mps.gz'));row=dict(passed=True,sid=sid,explicit_qL=explicit['qL'],explicit_qU=explicit['qU'],compact_objective=m.ObjVal,original_residual=res,V2G=explicit['actual_V2G_ac'],ENS=explicit['critical_ENS']+explicit['other_ENS'],shortfall=explicit['shortfall_ac']);dump(out/'COMPARISON.json',row);results.append(row);m.dispose();print('REAL_RECOURSE',sid,flush=True)
 return dict(passed=True,real_scenes=len(results),results=results,original_independent_lp='HiGHS full original P/Q and finite-bound dual certificate',compact_lp='Gurobi single thread fixed first-stage LP')
def zero400():
 rows=[]
 for sid in range(400):
  row=evaluate([],sid,R/'evaluations/ZERO'/f'{sid:04d}');rows.append(row)
  dump(R/'evaluations/ZERO/PROGRESS.json',dict(completed=sid+1,epoch=time.time()))
  if sid%10==9:print('ZERO400',sid+1,flush=True)
 npz(R/'evaluations/ZERO/Q.npz',qL=np.array([x['qL'] for x in rows]),qU=np.array([x['qU'] for x in rows]),Q=np.array([x['Q'] for x in rows]));dump(R/'evaluations/ZERO/PLAN.json',[])
 return dict(passed=True,scenarios=400,plan_hash=plan_digest([]),F=0,Q_sha=sha(R/'evaluations/ZERO/Q.npz'),risk_certificate_pending=True)
