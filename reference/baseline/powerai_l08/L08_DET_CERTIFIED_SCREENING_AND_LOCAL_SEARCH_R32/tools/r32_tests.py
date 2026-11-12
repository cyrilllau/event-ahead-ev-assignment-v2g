from r32_core import *
from scipy.optimize import linprog

def solve_test(m,out):
 a=matrix(m);identity=c.digest(dict(names=a['names'],lo=a['lo'],hi=a['hi'],cost=a['cost'],rhs=a['b'],A_data=a['A'].data,A_indices=a['A'].indices,A_indptr=a['A'].indptr))
 if out.with_suffix('.json').exists():
  assert read(out.with_suffix('.json'))['identity']==identity;xx=np.load(out.with_suffix('.npz'))['values'];audit_matrix(a,xx);return xx
 ledger=R/'tests/NATIVE_BUDGET.json';old=read(ledger) if ledger.exists() else dict(used=0,calls=[]);remaining=300-old['used'];assert remaining>0
 params(m,1,0,remaining,out.with_suffix('.log'));m.Params.MIPGapAbs=1e-7;t=time.monotonic();m.optimize();elapsed=time.monotonic()-t;old['used']+=elapsed;old['calls'].append(dict(label=str(out),wall=elapsed,native=m.Runtime,status=m.Status));dump(ledger,old);assert m.Status==GRB.OPTIMAL,(str(out),m.Status)
 x=np.array(m.getAttr('X'));audit_matrix(matrix(m),x);npz(out.with_suffix('.npz'),values=x);dump(out.with_suffix('.json'),dict(identity=identity,status=m.Status,objective=m.ObjVal,wall=elapsed,passed=True));return x

def tiny(case,late):
 out=R/'tests'/case;out.mkdir(parents=True,exist_ok=True)
 net=dict(buses=[1,2,3],line_ids=[1,2],line_from=[1,2],line_to=[2,3],to=[2,3],station_buses=[2,3],p_limit=[500,500],q_limit=[500,500],mp=[.00001,.00001],mq=[.00001,.00001],big_m=.2,u_min=.9025,u_max=1.1025,u_ref=1,shed_cost=[10,50,10]);net['from']=[1,2]
 scene=dict(line_operating_outages=np.zeros((96,2),bool),load_p_kw=np.tile([0.,10.,120.],(96,1)));scene['load_q_kvar']=.33*scene['load_p_kw'];scene['line_operating_outages'][92:96 if late else 44,1]=True
 if not late:scene['line_operating_outages'][39:44,1]=True
 qs=[dict(route_id=2*e+cc,e=e,c=cc,d=(91+cc if late else 38+cc),departure_dc=170.+10*e,distance=1.+cc,arrival=(22.9+.1*cc if late else 9.5+cc*.25),B=70.+10*e-1.2*(1+cc),ev='SYNTHETIC'+str(e)) for e in range(2) for cc in range(2)]
 ks=[92] if late else [39,40];js=[96] if late else [43,44];M=np.array([90.,90.]);oldports=c.CFG['ports'];c.CFG['ports']=1
 try:
  m,fs,rc=native_fixture(qs,2,net,scene,M,ks,js);aa=matrix(m);m.write(str(out/'BASE.mps.gz'));x=solve_test(m,out/'ORIGINAL');pa=physical_audit(x,aa['names'],qs,2,net,scene,M,rc['pfmap']);assert pa['V2G_ac']>0 and pa['shed_ac']>0
  sm=m.copy();removed=[i for i,q in enumerate(qs) if q['c']==0]
  for i in removed:sm.getVarByName(f'route[{i}]').UB=0
  sm.update();sx=solve_test(sm,out/'SCREENED');sa=physical_audit(sx,aa['names'],qs,2,net,scene,M,rc['pfmap']);assert abs(pa['objective']-sa['objective'])<1e-6;audit_matrix(aa,sx)
  # Independent explicit tasks, continuous contract/operation and exhaustive discrete choices.
  tasks=[c.task(q,k,j-1) for q in qs for k in ks for j in js if j>=k+4 and q['arrival']<=k/4];mt=assemble(net,tasks,scene,[dict(success=True,arrival=h['arrival']) for h in tasks],2,M);f,b,parts,_,_=first_stage(tasks,2,2,M)
  AA=sp.vstack([sp.hstack([f,sp.csr_matrix((len(b),len(mt.cost)))]),sp.hstack([mt.B,mt.A])],format='csr');HH=sp.hstack([mt.G,mt.H],format='csr');cost=np.r_[parts.sum(0),mt.cost];enum=[]
  for choice in itertools.product(*[[None]+[i for i,h in enumerate(tasks) if h['e']==e] for e in range(2)]):
   xx=np.zeros(len(tasks));xx[[i for i in choice if i is not None]]=1;lo=np.r_[xx,np.zeros(2),mt.lo];hi=np.r_[xx,M,mt.hi];ledger=R/'tests/NATIVE_BUDGET.json';budget=read(ledger);remaining=300-budget['used'];assert remaining>0;t=time.monotonic()
   sol=linprog(cost,A_ub=AA,b_ub=np.r_[b,mt.b],A_eq=HH,b_eq=mt.d,bounds=np.c_[lo,hi],method='highs',options={'threads':1,'time_limit':remaining,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});elapsed=time.monotonic()-t;budget['used']+=elapsed;budget['calls'].append(dict(label=case+'_enum_'+str(choice),wall=elapsed,status=sol.status));dump(ledger,budget);assert sol.status in [0,2]
   enum.append(dict(choice=choice,status=sol.status,objective=float(sol.fun) if sol.success else None,retained=all(i is None or tasks[i]['c']!=0 for i in choice)))
  best=min(a['objective'] for a in enum if a['objective'] is not None);screenbest=min(a['objective'] for a in enum if a['objective'] is not None and a['retained']);assert abs(best-pa['objective'])<1e-6 and abs(best-screenbest)<1e-6
  # Actual native complement construction and free-action tests.
  results=[]
  selected=[h['e'] for h in pa['plans']];unselected=[e for e in range(2) if e not in selected];assert selected and unselected
  for label,free,force in [('JOIN',unselected,1),('EXIT',[selected[0]],0),('CHANGE_STATION_DEPARTURE_WINDOW',[selected[0]],1)]:
   mm=m.copy();fix=fix_discrete_complement(mm,qs,set(free),x);e=free[0]
   if force==0:mm.addConstr(mm.getVarByName(f'selected[{e}]')==0)
   else:
    qi=next(i for i,q in enumerate(qs) if q['e']==e and q['c']==0);mm.addConstr(mm.getVarByName(f'route[{qi}]')==1);mm.addConstr(mm.getVarByName(f'start[{e},{ks[-1]}]')==1);mm.addConstr(mm.getVarByName(f'end[{e},{js[-1]}]')==1)
   mm.update();xx=solve_test(mm,out/label);audit_matrix(aa,xx);ph=physical_audit(xx,aa['names'],qs,2,net,scene,M,rc['pfmap']);results.append(dict(label=label,fix=fix,audit=ph));mm.dispose()
  # All discrete choices fixed, all E/operation still free. Raise selected E in feasible center then reoptimize it down.
  inflated=x.copy();ci={n:i for i,n in enumerate(aa['names'])};h=pa['plans'][0];e=h['e'];available=c.ebar(h['departure_dc'],h['distance'],h['arrival'],h['k'],h['r']);before=float(inflated[ci[f'E[{e}]']]);continuous_test=dict(bounds_free=True,at_capacity=abs(before-available)<1e-6)
  if available-before>1e-4:
   inflated[ci[f'E[{e}]']]=available;inflated=lift(inflated,aa['names'],aa['names'],qs,2,2,c.incident(net,scene)[0],rc['original'].index['short']);audit_matrix(aa,inflated);mm=m.copy();fix=fix_discrete_complement(mm,qs,set(),inflated);xx=solve_test(mm,out/'E_CONTINUOUS');assert xx[ci[f'E[{e}]']]<available-1e-5;continuous_test.update(observed_change=True,initial=available,optimized=xx[ci[f'E[{e}]']]);mm.dispose()
  mm=m.copy();fix_discrete_complement(mm,qs,set(),x);mm.addConstr(mm.getVarByName(f'E[{e}]')<=15);mm.update();xx=solve_test(mm,out/'E_CONTINUOUS_LOWER');assert abs(xx[ci[f'E[{e}]']]-15)<1e-6;audit_matrix(aa,xx);continuous_test.update(actual_reoptimization_to_15=True);mm.dispose()
  dump(out/'ENUMERATION.json',enum);dump(out/'COMPLETE.json',dict(passed=True,original=pa,screened=sa,enumerated=len(enum),enumeration_best=best,neighborhood_tests=results,E_continuous=continuous_test,ports=1,late_23_24=late,partial_shed=True,nonzero_discharge=True));m.dispose();sm.dispose()
 finally:c.CFG['ports']=oldports

def full_checks():
 out=R/'tests';qs=c.routes();net=c.layout()['net'];sc=c.load_scene('FORECAST_SELECTED');cert=read(R/'proof/SCREENING_CERTIFICATE.json');removed={r['c'] for r in cert['station_table'] if r['always_root']};m=frozen_model();aa=matrix(m);pf=c.loadnp(R/'frozen/PF_LIFT.npz');zy=c.loadnp(R/'frozen/ZERO_INPUT_WITNESS.npz')['y'];samples=[R/'frozen/START.npz']+sorted((PRIOR/'tests').glob('LEGAL_SAMPLE_*.npz'));records=[]
 ci={n:j for j,n in enumerate(aa['names'])}
 for i,p in enumerate(samples):
  x=np.load(p)['values'];orig=original_audit(x);xx=normalize(x,aa['names'],qs,30,net,sc,pf,zy,removed);new=original_audit(xx);assert new['physical']['objective']<=orig['physical']['objective']+1e-6;assert all(xx[ci[f'route[{j}]']]==0 for j in cert['removed_routes']);npz(out/f'NORMALIZED_{i}.npz',values=xx);records.append(dict(source=str(p),source_sha=sha(p),before=orig,after=new))
 dump(out/'NORMALIZATION_TESTS.json',dict(passed=True,samples=records));sm=gp.read(str(R/'models/SCREENED.mps.gz'));params(sm);x=np.load(R/'frozen/START.npz')['values'];audit_matrix(matrix(sm),x);new_start(sm,x)
 fixed=fix_discrete_complement(sm,qs,{0,1},x);dump(out/'FULL_COMPLEMENT_FIXING.json',fixed);fresh=gp.read(str(R/'models/SCREENED.mps.gz'));params(fresh);assert np.array_equal(matrix(fresh)['lo'],matrix(m)['lo']);assert len([n for n in fresh.getAttr('VarName') if n.startswith('E[')])==30
 negatives=[]
 pr=cert['prerequisites']
 for k in pr:
  bad=dict(pr);bad[k]=False;assert not guard(bad);negatives.append(dict(condition=k,rejected=True,type='MOCK_GUARD'))
 # Numerical source witnesses: nonzero grid cost, broken line cap, voltage bound, root shed, inadequate disconnected M.
 ix=assemble(net,[],sc,[],30,c.full_bounds()).index
 for key in ['grid_price','root_shed','line_limit','voltage','disconnected_M']:
  if key=='grid_price':bad_cost=np.zeros(len(zy));bad_cost[ix['grid_p']]=1;detected=bool(np.any(bad_cost[ix['grid_p']]!=0))
  elif key=='root_shed':by=zy.copy();by[ix['shed'][0,0]]=1;detected=assemble(net,[],sc,[],30,c.full_bounds()).residual(by,np.zeros(30))>1e-6
  elif key=='line_limit':bn=dict(net,p_limit=[0.]*len(net['from']));detected=assemble(bn,[],sc,[],30,c.full_bounds()).residual(zy,np.zeros(30))>1e-6
  elif key=='voltage':by=zy.copy();by[ix['u'][0,0]]=2;detected=assemble(net,[],sc,[],30,c.full_bounds()).residual(by,np.zeros(30))>1e-6
  else:detected=.1<net['u_max']-net['u_min']
  assert detected;negatives.append(dict(condition=key,rejected=True,type='FULL_NETWORK_NUMERIC_NEGATIVE'))
 valid=dict(type='GLOBAL_BASE_DET',science=science(),value=100);bad=[dict(type='LOCAL_NEIGHBORHOOD',science=science(),value=199),dict(type='GLOBAL_BASE_DET',science={'wrong':'forecast'},value=500),dict(type='GLOBAL_EXACT_SCREENED_DET',science=science(),value=198,proof_passed=False)];assert aggregate([valid]+bad,200)['LB']==100
 dump(out/'NEGATIVES_AND_BOUND_TYPES.json',dict(passed=True,negative=negatives,local_and_foreign_bounds_rejected=True));m.dispose();sm.dispose();fresh.dispose()

def tests():
 out=R/'tests';out.mkdir(exist_ok=True);full_checks()
 for name,late in [('MULTI_WINDOW',False),('LATE_23_24',True)]:
  if not (out/name/'COMPLETE.json').exists():tiny(name,late)
 dump(out/'COMPLETE.json',dict(passed=True,epoch=time.time(),native_budget=read(out/'NATIVE_BUDGET.json'),source_hashes={str(p.relative_to(R)):sha(p) for p in (R/'tools').glob('*.py')},evidence_types=['PROOF: component substitution and full integer optimum preservation','FULL_DOMAIN_CHECK: source and topology, ZERO, normalization, matrix bounds-only diff','TRUE_MIP: two original/screened fixture pairs; free join/exit/change tests','TRUE_LP: exhaustive continuous-E task enumeration','MOCK_GUARD plus real numeric negative network witnesses','No shared stochastic domain pruning; no400 solve']))
