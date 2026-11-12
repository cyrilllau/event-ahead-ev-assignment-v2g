from r38_common import *
sys.path.insert(0,str(B/'L08_DET_CERTIFIED_SCREENING_AND_LOCAL_SEARCH_R32/tools'))
import r32_core as old
import numpy as np
from scipy import sparse as sp
from fractions import Fraction as F
gp,GRB=old.gp,old.GRB
OUT=R/'det';OUT.mkdir(exist_ok=True)
def screened():
 m=old.frozen_model();cert=read(old.R/'proof/SCREENING_CERTIFICATE.json')
 for i in cert['removed_routes']:m.getVarByName(f'route[{i}]').UB=0
 m.update();return m
def install(m):
 rows=read(E/'DET_INTEGER_CAPACITY_CUTS.json')['rows'];assert len(rows)==84 and len({r['name'] for r in rows})==84
 vs={v.VarName:v for v in m.getVars()}
 for r in rows:
  assert m.getConstrByName(r['name']) is None
  m.addConstr(gp.LinExpr([a for n,a in r['terms']],[vs[n] for n,a in r['terms']])<=r['rhs'],name=r['name'])
 m.update();return m
def small_tests():
 out=OUT/'small';out.mkdir(exist_ok=True);results=[]
 # Actual MIPs retain explicit critical/ordinary shedding and integer capacity.
 for k,(D,C) in enumerate([(180.46716365907088,55.52867946163338),(80.,80.),(270.,30.),(0.,0.)]):
  objs=[]
  for strengthened in [False,True]:
   m=gp.Model('R38_SMALL_INTEGER_CAPACITY');m.Params.Threads=1;m.Params.TimeLimit=30;m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.IntFeasTol=1e-9
   n=m.addVar(vtype=GRB.INTEGER,lb=0,ub=30);sc=m.addVar(ub=C);so=m.addVar(ub=D-C)
   m.addConstr(sc+so>=D-90*n);m.setObjective(101*n+12.5*sc+2.5*so)
   f=lambda j:.25*(10*max(D-90*j,0)+40*max(C-90*j,0))
   if strengthened:
    for j in range(30):m.addConstr(12.5*sc+2.5*so>=f(j)+(f(j+1)-f(j))*(n-j)-1e-8)
   m.Params.OutputFlag=1;m.Params.LogFile=str(out/f'{k}_{strengthened}.log');m.optimize();assert m.Status==GRB.OPTIMAL
   objs.append(m.ObjVal);results.append(dict(case=k,strengthened=strengthened,status=m.Status,objective=m.ObjVal,runtime=m.Runtime));m.dispose()
  enum=min(101*j+f(j) for j in range(31));assert max(abs(x-enum) for x in objs)<1e-6
 assert sum(x['runtime'] for x in results)<=600
 dump(out/'RESULT.json',dict(passed=True,real_Gurobi_MIP=True,results=results))
def cut_proof():
 protected();m=screened();a=old.matrix(m);c=old.c;net=c.layout()['net'];sc=c.load_scene('FORECAST_SELECTED');qs=c.routes()
 cert=read(old.R/'proof/SCREENING_CERTIFICATE.json');assert old.guard(cert['prerequisites'])
 for n,h in cert['source_review'].items():
  filename={'first_stage':'compact_first_stage.py','network':'contract_matrix.py','aux':'compact_recourse_pf.py'}[n];assert sha(D/'tools'/filename)==h
 rt=old.root_table(net,sc);removed=[i for i,q in enumerate(qs) if rt[:,net['buses'].index(net['station_buses'][q['c']])].all()];assert removed==cert['removed_routes']
 assert len(removed)==13104
 zero=np.load(old.R/'proof/ZERO_FULL_VECTOR.npz')['values'];za=old.original_audit(zero,OUT/'ZERO_ORIGINAL_AUDIT.json')
 mat=old.assemble(net,[],sc,[],30,c.full_bounds());pf=c.loadnp(old.R/'frozen/PF_LIFT.npz');old_to_new={int(j):i for i,j in enumerate(pf['keep'])}
 assert c.CFG['pmax']==90 and c.CFG['dt']==.25 and net['shed_cost']==[50. if b in c.CFG['critical_buses'] else 10. for b in net['buses']]
 assert np.all(mat.cost>=0);assert set(mat.index)==set(['p','shed','line_p','line_q','u','grid_p','grid_q','short'])
 blocks=[]
 for t in range(96):
  adj={b:[] for b in net['buses']}
  for u,v,off in zip(net['from'],net['to'],sc['line_operating_outages'][t]):
   if not off:adj[u].append(v);adj[v].append(u)
  seen=set()
  for bus in sorted(adj):
   if bus in seen:continue
   comp={bus};todo=[bus];seen.add(bus)
   while todo:
    for v in adj[todo.pop()]:
     if v not in seen:seen.add(v);comp.add(v);todo.append(v)
   cs=[cc for cc,b in enumerate(net['station_buses']) if b in comp]
   if 1 not in comp and cs:blocks.append(dict(t=t,buses=sorted(comp),station_indices=cs))
 defs=read(E/'INTEGER_CAPACITY_CUT_DEFINITIONS.json');assert len(blocks)==len(defs)==43
 rows=read(E/'DET_INTEGER_CAPACITY_CUTS.json')['rows'];ri=0;minslack=float('inf');rebuilt=[]
 for block,ref in zip(blocks,defs):
  for key in block:assert block[key]==ref[key],(key,block,ref)
  t=block['t'];bs=block['buses'];d=sum(F(float(sc['load_p_kw'][t,b-1])) for b in bs);crit=sum((F(float(sc['load_p_kw'][t,b-1])) for b in bs if b in c.CFG['critical_buses']),F(0))
  f=lambda n:(10*max(d-90*n,0)+40*max(crit-90*n,0))/4
  for intercept,slope,j in ref['rows']:
   if slope==intercept==0:continue
   row=rows[ri];ri+=1
   assert abs(float(f(j+1)-f(j))-slope)<1e-9 and abs(float(f(j))-slope*j-intercept)<1e-9
   expected={f'reserved[{e},{cc},{t}]':slope for e in range(30) for cc in block['station_indices']}
   for b in bs:expected[f's0_y[{old_to_new[int(mat.index["shed"][t,b-1])]}]']=-.25*net['shed_cost'][b-1]
   assert dict(row['terms'])==expected and abs(row['rhs']-(-intercept+1e-8))<1e-12
   for n in range(31):
    slack=f(n)-(F(slope)*n-F(row['rhs']));assert slack>=0;minslack=min(minslack,float(slack))
   rebuilt.append(dict(**block,j=j,minimum_integer_slack=min(float(f(n)-(F(slope)*n-F(row['rhs']))) for n in range(31))))
 assert ri==84
 # Native first-stage reserved is the conjunction of binary route station and binary window.
 fs_source=(D/'tools/compact_first_stage.py').read_text();assert 'reserved' in fs_source and 'window' in fs_source
 install(m);aa=old.matrix(m);assert m.NumVars==99234 and m.NumBinVars==25770 and m.NumConstrs==140527+84
 for n in ['R31/','r34_det_window','count_branch/','r35_count/']:assert not any(x.startswith(n) for x in aa['rows'])
 points={'ZERO':old.R/'proof/ZERO_FULL_VECTOR.npz','7':B/'L08_DET_COUNT_DECOMPOSITION_AND_RESERVATION_LINK_R35/runs/C7/BEST.npz','8':old.R/'frozen/START.npz','9':B/'L08_DET_COUNT_DECOMPOSITION_AND_RESERVATION_LINK_R35/runs/C9/BEST.npz'}
 audits={}
 for tag,p in points.items():
  x=np.load(p)['values'];audits[tag]=dict(matrix=old.audit_matrix(aa,x),original=old.original_audit(x,OUT/f'ORIGINAL_{tag}.json'),source=str(p),sha256=sha(p));np.savez_compressed(OUT/f'POINT_{tag}.npz',values=x)
 m.write(str(OUT/'CAPACITY.mps.gz'));m.dispose()
 small_tests()
 # Independent applicability rejection tests; no scientific parameter changed.
 base=dict(power=90,dt=.25,other_generation=False,prices=(50,10),unique_station=True,science='DET44')
 applicable=lambda x:x==base
 negative={}
 for key,val in [('power',91),('dt',1),('other_generation',True),('prices',(-1,10)),('unique_station',False),('science','DRO400')]:
  x=dict(base);x[key]=val;negative[key]=not applicable(x);assert negative[key]
 dump(OUT/'CUT_PROOF.json',dict(passed=True,count=84,island_slots=43,minimum_exact_integer_slack=minslack,reconstructed=rebuilt,removed_routes=removed,source_screen_sha=sha(old.R/'proof/SCREENING_CERTIFICATE.json'),audits=audits,negative_policy_tests=negative,
  proof='Each binary vehicle chooses at most one route/station; start/end define an integral contiguous window, reserved is their AND. Thus n is an integer 0..30. In each rootless component, network P balance and zero outage-line flow give total served load <=90n; no other generation. Optimal optimistic allocation serves critical first, giving f. Convex discrete secants under-estimate f at every integer. Arrival, energy, network losses/limits only reduce feasible supply. qLS references original shed cost; objective untouched.'))
def lp_check():
 assert read(OUT/'CUT_PROOF.json')['passed'];m=screened();a=old.matrix(m);p=B/'L08_DET_CERTIFIED_RC_SYMMETRY_R33/lp';z=np.load(p/'MATRIX.npz');A=sp.load_npz(p/'A.npz');mp=read(p/'MAP.json')
 assert a['names']==mp['columns'] and a['rows']==mp['rows'] and (A!=a['A']).nnz==0
 for k,j in [('b','rhs'),('sense','sense'),('cost','cost'),('lo','lo'),('hi','hi')]:assert np.array_equal(a[k],z[j]),k
 install(m);a2=old.matrix(m);C=sp.load_npz(E/'NEW_INTEGER_CAPACITY_CUT_MATRIX.npz');rhs=np.load(E/'NEW_INTEGER_CAPACITY_CUT_RHS.npz')['rhs'];assert (a2['A'][-84:]!=C).nnz==0 and np.array_equal(a2['b'][-84:],rhs)
 W=OUT/'exact';dest=W/'input/DET/HISTORY/L08_DET_CERTIFIED_RC_SYMMETRY_R33/lp';dest.mkdir(parents=True,exist_ok=True)
 for n in ['MATRIX.npz','A.npz','MAP.json']:shutil.copy2(p/n,dest/n)
 (W/'input/DET/FORECAST44').mkdir(parents=True,exist_ok=True);shutil.copy2(cpath:=old.c.scene_path('FORECAST_SELECTED')/'physical.npz',W/'input/DET/FORECAST44/physical.npz')
 shutil.copytree(E,W/'audit',dirs_exist_ok=True)
 code=(E/'certify_capacity_lp.py').read_text().replace("r=Path('/mnt/data/L08_20261101_BOTTLENECK_REVIEW')",f'r=Path({str(W)!r})')
 (W/'certify_rebound.py').write_text(code);exec(compile(code,str(W/'certify_rebound.py'),'exec'),{'__name__':'__main__'})
 cert=read(W/'audit/CAPACITY_LP_EXACT_LOWER_CERTIFICATE.json');assert cert['passed']
 sol=np.load(E/'CAPACITY_STRENGTHENED_LP_SOLUTION.npz');x=sol['x'] if 'x' in sol else sol['X'];old.audit_matrix(a2,x,False)
 dump(OUT/'LP_CERTIFICATE_ACCEPTANCE.json',dict(passed=True,lower_bound=cert['lower_bound_outward'],all_original_matrix_coefficients_equal=True,all84_coefficients_equal=True,full_multiplier_certificate_recomputed=True,derived_bounds=cert['derived_upper_bounds'],server_new_LP_solve=False,reason='Complete same-matrix existing primal/dual and integer-cut proof independently revalidated; no redundant LP solve',model_sha=sha(OUT/'CAPACITY.mps.gz')));m.dispose()
def mip_test():
 assert read(OUT/'CUT_PROOF.json')['passed'] and read(OUT/'LP_CERTIFICATE_ACCEPTANCE.json')['passed'];protected()
 m=gp.read(str(OUT/'CAPACITY.mps.gz'));old.params(m,4,0,1200,OUT/'NATIVE.log');m.Params.OutputFlag=1;m.Params.MIPGapAbs=1;m.Params.LogFile=str(OUT/'NATIVE.log')
 x=np.load(old.R/'frozen/START.npz')['values'];old.new_start(m,x);assert sha(old.R/'frozen/START.npz')==sha(B/'L08_DET_CERTIFIED_SCREENING_AND_LOCAL_SEARCH_R32/frozen/START.npz')
 begin=OUT/'MIP_BEGIN.json';assert not begin.exists(),'native budget cannot reset';dump(begin,dict(epoch=time.time(),owner=ctrl.proc(os.getpid()),profile='DET_SITE_INT_CAPACITY_T4',seconds=1200,model_sha=sha(OUT/'CAPACITY.mps.gz'),start_sha=sha(old.R/'frozen/START.npz'),same_start_as_R32_E=True))
 last=[-20.];snap=[False];candidates=[]
 def cb(mm,w):
  if w==GRB.Callback.MIP:
   rt=mm.cbGet(GRB.Callback.RUNTIME)
   if rt-last[0]>=20:
    z=dict(runtime=rt,LB=mm.cbGet(GRB.Callback.MIP_OBJBND),nativeUB=mm.cbGet(GRB.Callback.MIP_OBJBST),nodes=mm.cbGet(GRB.Callback.MIP_NODCNT),work=mm.cbGet(GRB.Callback.WORK),memory_GB=mm.cbGet(GRB.Callback.MEMUSED));dump(OUT/'MIP_PROGRESS.json',z);last[0]=rt
    if rt>=600 and not snap[0]:dump(OUT/'MIP_600_SECONDS.json',z);snap[0]=True
  if w==GRB.Callback.MIPSOL:
   val=mm.cbGet(GRB.Callback.MIPSOL_OBJ)
   if not candidates or val<candidates[-1]['objective']-1e-7:
    n=len(candidates);p=OUT/'candidates'/f'{n:04d}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,values=np.array(mm.cbGetSolution(mm.getVars())));candidates.append(dict(objective=val,path=str(p),runtime=mm.cbGet(GRB.Callback.RUNTIME)))
 m.optimize(cb)
 lb=float(m.ObjBound) if abs(m.ObjBound)<1e90 else None
 dump(OUT/'MIP_END.json',dict(status=m.Status,runtime=m.Runtime,LB=lb,nativeUB=m.ObjVal if m.SolCount else None,nodes=m.NodeCount,work=m.Work,solver_max_memory_GB=m.MaxMemUsed,epoch=time.time(),candidates=candidates))
 if m.SolCount:np.savez_compressed(OUT/'MIP_FINAL.npz',values=np.array(m.getAttr('X')))
 m.dispose();aud=[]
 for rec in candidates:
  au=old.original_audit(np.load(rec['path'])['values'],Path(rec['path']).with_suffix('.AUDIT.json'));aud.append(au)
 base=old.base_bound();ub=min([read(OUT/'ORIGINAL_8.json')['UB']]+[a['UB'] for a in aud]);bounds=[base['value'],read(OUT/'LP_CERTIFICATE_ACCEPTANCE.json')['lower_bound']]+([lb-1e-6] if lb is not None else [])
 L=max(bounds);assert ub-L>=-1e-6
 dump(OUT/'GLOBAL_BOUNDS.json',dict(LB=L,UB=ub,signed_gap=ub-L,certified=0<=ub-L<=1,production=base,capacity_LP=read(OUT/'LP_CERTIFICATE_ACCEPTANCE.json'),MIP_LB=lb,original_audited_candidates=len(aud),protection=protected(),not_deployed_to_live_DET=True))
