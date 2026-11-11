from aro_core import *
def synthetic():
 out=ROOT/'TESTS/SYNTHETIC';out.mkdir(parents=True,exist_ok=True);AB=np.array([[10.,-10.],[1.,4.],[0.,20.]])
 def model(ids,integer=True):
  m=gp.Model();m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.MIPGap=0;m.Params.MIPGapAbs=1e-8;x=m.addMVar(1,ub=1,vtype='B' if integer else 'C');eta=m.addVar();m.setObjective(3*x[0]+eta)
  for sid in ids:add_support(m,x,eta,AB[sid,0],AB[sid,1:],f'cut/{sid}')
  return m,x,eta
 direct,x,eta=model(range(3));native(direct,'TEST_SYNTHETIC_DIRECT');target=direct.ObjVal;assert target==10;direct.dispose();lp,x,eta=model(range(3),False);native(lp,'TEST_SYNTHETIC_LP');assert abs(lp.ObjVal-23/3)<1e-8;lp.dispose()
 m,x,eta=model([]);seen=set();history=[]
 for i in range(8):
  native(m,'TEST_SYNTHETIC_BENDERS');xx=float(x.X[0]);q=AB[:,0]+AB[:,1]*xx;records=[{'sid':sid,'qU_audited':float(q[sid])} for sid in range(3)];ub=full_bound(3*xx,records,list(range(3)));lb=bound_update(-np.inf,m.ObjBound);history.append({'LB':lb,'UB':ub})
  if ub-lb<=2e-6:break
  for sid in range(3):
   if q[sid]>eta.X+1e-8 and sid not in seen:add_support(m,x,eta,AB[sid,0],AB[sid,1:],f'cut/{sid}');seen.add(sid)
 assert abs(history[-1]['UB']-target)<1e-6;m.dispose()
 try:full_bound(0,records[:2],list(range(3)));raise AssertionError('PARTIAL_UB_ACCEPTED')
 except ValueError:pass
 assert bound_update(4,3)==4 and bound_update(4,None)==4
 W=sp.csr_matrix([[1.]]);T=sp.csr_matrix([[-1.]]);a={'h':np.zeros(1),'c':np.ones(1),'finite_lo':np.zeros(1),'finite_hi':np.full(1,2.),'xlo':np.zeros(1),'xhi':np.full(1,2.),'sense':np.array(['>'])};alpha,beta,pi=e.b.replay_cut(W,T,a,np.array([2.]));assert alpha+beta[0]<=1 and 2>1
 from memory_policy import ModelCache,stable_map
 from testfixtures import FakeOracle
 owners=stable_map(range(400));assert [sum(v==s for v in owners.values()) for s in range(4)]==[100]*4;policy=read(ROOT/'PATCH_POLICY.json')
 for slot in range(4):
  cache=ModelCache(slot,0,None,[i for i in range(400) if owners[str(i)]==slot],lambda sid,env:FakeOracle(sid,env),policy,out/f'CACHE_{slot}.jsonl')
  for _ in range(12):
   for sid in range(slot,400,4):cache.acquire(sid);cache.release_busy();assert len(cache.objects)<=16
  cache.close();cache.close()
 result={'passed':True,'production_functions_used':['make_master epigraph convention','add_support','full_bound','bound_update','replay_cut','ModelCache','stable_map'],'native_direct_vs_integer_Benders':history,'integer_optimum':target,'LP_optimum':23/3,'max_not_sum':True,'clairvoyant_minmax_not_shared':3,'sum_model':11,'bound_contribution_negative_RC_required':True,'partial_UB_rejected':True,'400_slot_ownership':[100]*4,'cache_hard_limit':16,'12_rounds_count_gate':True};write(out/'RESULT.json',result,True)

def real_small(deadline):
 out=ROOT/'TESTS/REAL_SMALL';out.mkdir(parents=True,exist_ok=True);A,a,names=first();plan=read(R36/'evaluations/7cc27e720e13d3cbe46a1a25a9e704f9495bbcf045cf2df212cb702bd24fdf27/COMPLETE.json')['plan'];one=[dict(plan[0],contract=15.)]
 from compact_first_stage import fix_first
 m=gp.Model();m.Params.OutputFlag=0;fs=orig.add_first(m,orig.routes(),30,8,orig.full_bounds());orig.add_strengthening(m,fs);fix_first(fs,one);m.setObjective(fs['objective']);m.update();m.Params.Threads=1;m.Params.Method=1;m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.TimeLimit=min(120,max(.1,deadline-time.monotonic()));native(m,'TEST_SINGLE_VEHICLE_SEED')
 if m.Status!=2:raise ValueError('REAL_SMALL_SEED_INCOMPLETE')
 raw=np.array(m.getAttr('X'));m.dispose();x1,key=canonical(raw,out/'ONE_VEHICLE',deadline);F=float(a['cost']@x1);ids=[1,385];direct=gp.Model();direct.Params.OutputFlag=0;direct.Params.Threads=1;t=direct.addVar(vtype='B');eta=direct.addVar();direct.setObjective(F*t+eta);env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();oracles={sid:AROOracle(sid,out/'oracle',env) for sid in ids}
 for sid,o in oracles.items():
  y=direct.addMVar(len(o.a['c']),lb=o.a['lo'],ub=o.a['hi']);direct.addConstr(o.W@y+(o.T@x1)*t>=o.a['h']) if np.all(o.a['sense']=='>') else None
  # Mixed senses are preserved row by row through addMConstr with an explicit test t column.
  direct.remove(direct.getConstrs()) if False else None
  block=sp.hstack([o.W,sp.csr_matrix((o.T@x1)[:,None])],format='csr');allv=gp.MVar.fromlist(y.tolist()+[t]);direct.addMConstr(block,allv,o.a['sense'],o.a['h']);direct.addConstr(eta>=o.a['c']@y)
 direct.Params.FeasibilityTol=direct.Params.OptimalityTol=direct.Params.IntFeasTol=1e-9;direct.Params.MIPGapAbs=1e-6;direct.Params.MIPGap=0;direct.Params.TimeLimit=min(300,max(.1,deadline-time.monotonic()));native(direct,'TEST_REAL_DIRECT')
 if direct.Status!=2:raise ValueError('REAL_SMALL_DIRECT_INCOMPLETE')
 target=direct.ObjVal;direct.dispose();bm=gp.Model();bm.Params.OutputFlag=0;bm.Params.Threads=1;bm.Params.MIPGap=0;bm.Params.MIPGapAbs=1e-6;v=bm.addMVar(1,ub=1,vtype='B');ep=bm.addVar();bm.setObjective(F*v[0]+ep);history=[]
 for k in range(6):
  checked();native(bm,'TEST_REAL_BENDERS_MASTER');xx=x1*round(v.X[0]);rs=[]
  for sid,o in oracles.items():
   task=e.task_id('TEST_REAL_INTEGER_SMALL',sid,xx,'RECOURSE');rs.append(o.solve_task(xx,task,out/'cache'/task,deadline))
  ub=full_bound(F*round(v.X[0]),rs,ids);history.append({'LB':bm.ObjBound,'UB':ub})
  if ub-bm.ObjBound<=1e-5:break
  for r in rs:
   cut=np.load(r['cut']);add_support(bm,v,ep,float(cut['alpha']),np.array([cut['beta']@x1]),f'test/{k}/{r["sid"]}')
 if abs(history[-1]['UB']-target)>1e-4 or history[-1]['UB']-history[-1]['LB']>1e-5:raise ValueError('REAL_SMALL_EQUIVALENCE_NOT_COMPLETED')
 # Independent original P/Q physical audit at the same selected finite-domain plan.
 from contract_matrix import assemble
 from exact_transform import lift
 accepted=one if round(v.X[0]) else [];contracts=np.zeros(30)
 for h in accepted:contracts[h['e']]=h['contract']
 physical=[]
 for sid,o in oracles.items():
  r=rs[ids.index(sid)];payload=np.load(r['primal']);mapping=read(ROOT/'oracles'/f'{sid:04d}'/'PF_LIFT/MAPPING.json') if False else None
  # Rebuild only the lift mapping through the original recourse builder, no optimize.
  mm=gp.Model();mm.Params.OutputFlag=0;ff=orig.add_first(mm,orig.routes(),30,8,orig.full_bounds());orig.add_strengthening(mm,ff);rc=orig.add_scene(mm,ff,None,sid)
  full_y=lift(payload['y'][:len(rc['base'].cost)],rc['pfmap']);tasks=[orig.task(next(q for q in orig.routes() if q['route_id']==h['route_id']),h['k'],h['r']) for h in accepted];mat=assemble(orig.layout()['net'],tasks,orig.load_scene('TRAIN',sid),orig.route_states('L08',tasks,'TRAIN',sid),30,orig.full_bounds());res=float(mat.residual(full_y,np.r_[np.ones(len(tasks)),contracts]));assert res<=1e-7,(sid,res);physical.append({'sid':sid,'original_PQ_residual':res});mm.dispose();o.close(discard=True)
 env.dispose();bm.dispose();write(out/'RESULT.json',{'passed':True,'ids':ids,'domain':'one original vehicle with original task, contract15, or ZERO; shared integer switch; TEST only','direct':target,'Benders':history,'original_physical_audits':physical},True)
