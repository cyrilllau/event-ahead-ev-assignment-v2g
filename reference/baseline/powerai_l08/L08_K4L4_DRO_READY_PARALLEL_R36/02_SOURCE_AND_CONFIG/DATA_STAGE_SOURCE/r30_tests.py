from r30_model import *
from lp_certificate import certify
import itertools
from scipy.optimize import linprog

def tests():
 out=R/'06_TESTS';out.mkdir(exist_ok=True);records=[]
 provenance=[]
 for name in ['energy_ledger.py','compact_first_stage.py','compact_recourse_pf.py','contract_matrix.py','lp_certificate.py','exact_transform.py','r4_common.py','r10_common.py']:
  old=B/'L08_DET_R10_OCCUPANCY_COST_R15/tools'/name;new=R/'tools'/name;assert sha(old)==sha(new)
  provenance.append(dict(source=str(old),source_sha=sha(old),copy=str(new),copy_sha=sha(new),byte_identical=True))
 dump(R/'02_SOURCE_AND_CONFIG/MATH_COMPONENT_PROVENANCE.json',provenance)
 import importlib
 for module in ['r3_common','r4_common','r10_common','compact_first_stage','compact_recourse_pf','contract_matrix','energy_ledger','exact_transform']:
  assert (R/'tools') in Path(importlib.import_module(module).__file__).resolve().parents
 def ok(name,detail):
  records.append(dict(name=name,passed=True,detail=detail));dump(out/'PROGRESS.json',dict(complete=False,tests=records));print('TEST_PASS',name,flush=True)
 def reject(fn):
  try:fn()
  except (AssertionError,ValueError,FileNotFoundError):return
  raise AssertionError('Negative control was accepted')
 archive=R29/'delivery'/('L08_K4L4_CALIBRATION_AND_FORECAST_R29_PRO_ANALYSIS.zip');assert sha(archive)==read(R/'instructions/SOURCE_EVIDENCE.json')['source_archive_sha256']['R29_PRO_ANALYSIS']
 ii=read(R/'02_SOURCE_AND_CONFIG/SCENE_IDENTITIES.json');assert ii[0]['source_sid']==44 and ii[0]['source_bank']!=ii[45]['source_bank']
 reject(lambda:scene_path('FORECAST_SELECTED',44));reject(lambda:scene_path('TRAIN',400));reject(lambda:layout('L13'))
 from r12_common import cache_key
 assert cache_key(20,'p','n',0,500,1,2)!=cache_key(44,'p','n',0,500,1,2)
 reject(lambda: loadnp(R/'missing_physical.npz'))
 ok('identity_schema_negative',dict(forecast=ii[0],support_local44=ii[45],missing_is_error=True))
 assert first_cost(38,39,45)==86.75 and first_cost(38,39,83)==381.25 and first_cost(81,82,85)==63.5
 assert np.all(tou()[:32]==.14296) and np.all(tou()[32:64]==.17297) and np.all(tou()[64:84]==.2154) and np.all(tou()[84:]==.17297)
 assert abs(90*.25*.05/.95-1.1842105263157896)<1e-12
 assert actual(150,23,92,95)['qualified'];assert not actual(150,23.25,92,95)['qualified'];assert actual(200,22.75,88,91)['qualified'];assert not actual(200,22.75001,88,91)['qualified']
 assert actual(np.nan,np.inf,0,3,False)['available_ac']==0
 reject(lambda:forecast(150,23,92,94));reject(lambda:forecast(150,23,92,96))
 x=forecast(300,9.2,40,47);y=forecast(300,9.2,42,47);assert x['battery_ac']==y['battery_ac'] and x['power_ac']!=y['power_ac']
 assert forecast(300,9.2,40,47,aux=0)['battery_ac']==190
 p=np.zeros(96);p[92:]=forecast(150,23,92,95)['cap_ac'];tr=battery_trace(150,23,24,p);assert abs(tr['terminal_dc']-100)<1e-8
 ok('time_energy_cost_units','23h/last-full-slot/failure-inf; full dwell once; three cost regressions; ToU/DC-AC')
 from r12_common import routes_load,normalized_roads,layout as roadlayout
 from r30_domain import bounds_for,encode,decode as decode_index
 from r30_replay import geometry,replay
 rows=routes_load();dom=loadnp(R/'04_R10_COMPACT_DOMAIN/DOMAIN.npz');M=np.zeros(30)
 assert len(rows)==23040 and len({(q['e'],q['c'],q['d']) for q in rows})==23040
 for q in rows:
  valid,cap=bounds_for(q);rid=q['route_id'];assert np.array_equal(valid,(np.arange(1,97)[None,:]>=np.arange(93)[:,None]+4)&(np.arange(1,97)[None,:]<=dom['Jmax'][rid,:,None]))
  # Independent integer-window counting from scalar inequalities, not the stored Jmax table.
  if q['route']:
   arrival=q['route']['arrival_h'];A=q['departure_dc']-1.2*q['route']['distance_km'];limit=min(96,int(np.floor(4*(A-100+arrival-15/.95))))
   expected_count=np.array([max(0,limit-k-3) if k/4>=arrival and A>=150 else 0 for k in range(93)])
  else:expected_count=np.zeros(93,int)
  assert np.array_equal(expected_count,dom['kcount'][rid]);assert np.array_equal(dom['koffset'][rid],np.r_[0,np.cumsum(expected_count)]);assert dom['offset'][rid+1]-dom['offset'][rid]==expected_count.sum()
  M[q['e']]=max(M[q['e']],float(np.max(np.where(valid,cap,0))))
  for k in np.flatnonzero(dom['kcount'][rid]):
   for J in [k+4,int(dom['Jmax'][rid,k])]:assert decode_index(dom,encode(dom,rid,int(k),J))==(rid,int(k),J)
 assert np.array_equal(M,full_bounds());wit=read(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json')['witness']
 for e,w in enumerate(wit):assert abs(ebar(rows[w['route_id']]['departure_dc'],rows[w['route_id']]['route']['distance_km'],w['arrival'],w['k'],w['r'])-M[e])<1e-10
 assert not np.array_equal(np.zeros(30),full_bounds())
 ok('full_domain_and_bigM',dict(queries=23040,logical_tasks=int(dom['offset'][-1]),M=M.tolist(),all_route_start_end_boundaries_checked=True))
 g=loadnp(R/'05_SUPPORT400_ROUTE_REPLAY/FIXED_ROUTE_GEOMETRY.npz');f=loadnp(R/'05_SUPPORT400_ROUTE_REPLAY/FORECAST44.npz')
 assert any(q.get('prefix') for q in rows) and any(q['route'] and not q['route']['arc_ids'] for q in rows)
 # Independently replay all forecast routes by scalar frozen replay routine, including virtual arcs.
 from r12_route_adapter import network_for_state,replay_route
 lay=roadlayout();ar=normalized_roads(scene_path('FORECAST_SELECTED'),lay);nets={};count=0
 for q in rows:
  if not q['route']:continue
  st=q['source_state'];key=(st['node'],st['kind'],st['arc'],st['fraction'])
  if key not in nets:nets[key]=network_for_state(lay,ar,st)
  net,origin,_=nets[key];rr=replay_route(net,q['route']['arc_ids'],q['d']/4,q['departure_dc']/500,500,1.2,0,origin=origin,destination=q['destination'])
  assert rr and abs(rr.arrival_h-f['arrival_h'][q['route_id']])<1e-10;count+=1
 # Close every road at all epochs: all nonempty paths fail; empty paths remain physical successes.
 closed={k:v.copy() for k,v in ar.items()};closed['road_movable'][:]=False;closed['road_blocked'][:]=True
 zz=replay(rows,g,closed);nonempty=g['arc_count']>0;assert not zz['success'][nonempty].any() and np.isinf(zz['arrival_h'][nonempty]).all()
 ok('route_scalar_vector_and_closed_roads',dict(all_qualified_scalar_replayed=count,virtual_and_zero_distance=True,no_reroute=True))
 a=load_scene('FORECAST_SELECTED');n=layout()['net'];assert n['station_buses']==[21,15,24,3,10,14,32,27];assert CFG['station_kw']==1800 and CFG['ports']==20 and CFG['pmax']==90 and CFG['fixed_q_over_p']==.33;assert n['shed_cost']==[50. if b in CFG['critical_buses'] else 10. for b in n['buses']];off=a['line_operating_outages'];expected={12:(55,81),13:(63,76),11:(90,91),19:(93,96)}
 for l,(i,j) in expected.items():assert np.array_equal(np.flatnonzero(off[:,l]),np.arange(i,j))
 assert off.sum()==26+13+1+3 and a['line_terminal_outages'][19]
 ok('forecast44_recovery_and_terminal',expected)
 zero=loadnp(scene_path('FORECAST_SELECTED')/'ZERO_WITNESS.npz');mat=assemble(n,[],a,[],30,full_bounds());zy=np.zeros(len(mat.cost));ix=mat.index
 for name,source in [('line_p','P_line'),('line_q','Q_line'),('grid_p','P_root'),('grid_q','Q_root'),('u','voltage_squared'),('shed','shed_p')]:zy[ix[name]]=zero[source]
 zr=mat.residual(zy,np.zeros(30));assert zr<=1e-6
 crit=np.array(n['shed_cost'])==50;ens=.25*zy[ix['shed']].sum();critical=.25*zy[ix['shed']][:,crit].sum();cost=float(mat.cost@zy)
 assert abs(ens-1407.497149064476)<1e-7 and abs(critical-439.3333127105663)<1e-7 and abs(cost-31648.303999067415)<1e-7
 # Network component balance proves mandatory island shedding for the zero-procurement plan.
 for t in range(96):
  root=next(g for g in components(n,a['line_operating_outages'][t]) if 1 in g)
  expected_shed=np.array([0. if bus in root else a['load_p_kw'][t,i] for i,bus in enumerate(n['buses'])]);assert np.max(abs(expected_shed-zy[ix['shed']][t]))<1e-9
 guard=max(1e-7,10*zr*float(abs(mat.cost).sum()),128*np.finfo(float).eps*max(1,2*abs(cost)))
 npz(out/'ZERO_INPUT_WITNESS.npz',y=zy);dump(out/'ZERO_INPUT_CERTIFICATE.json',dict(passed=True,qL=float(np.nextafter(cost-guard,-np.inf)),qU=float(np.nextafter(cost+guard,np.inf)),guard=guard,primal=cost,original_residual=zr,proof='Feasible original PQ point and component balance forcing full non-root shedding with no EV; nonnegative objective; frozen numerical guard max(1e-7, residual amplification, rounding)',physical_sha=sha(scene_path('FORECAST_SELECTED')/'physical.npz'),source_witness_sha=sha(scene_path('FORECAST_SELECTED')/'ZERO_WITNESS.npz'),witness_sha=sha(out/'ZERO_INPUT_WITNESS.npz'),solver_called=False))
 ok('ZERO_input_original_witness',dict(total_ENS=ens,critical_ENS=critical,cost=cost,original_residual=zr,source_witness_reused=True))
 # Independent explicit-task MILP and exhaustive first-stage enumeration with continuous E.
 nn=dict(buses=[1,2,3],line_ids=[1,2],line_from=[1,2],line_to=[2,3],to=[2,3],station_buses=[2,3],p_limit=[500,500],q_limit=[500,500],mp=[.00001,.00001],mq=[.00001,.00001],big_m=.2,u_min=.9025,u_max=1.1025,u_ref=1,shed_cost=[10,50,10]);nn['from']=[1,2]
 aa=dict(line_operating_outages=np.zeros((96,2),bool),load_p_kw=np.tile([0.,70.,25.],(96,1)));aa['load_q_kvar']=.33*aa['load_p_kw'];aa['line_operating_outages'][39:44,0]=True
 qq=[dict(route_id=2*e+c,e=e,c=c,d=38+c,departure_dc=300.,distance=1.+c,arrival=9.5+c*.25,B=200.-1.2*(1+c),ev='TEST'+str(e)) for e in range(2) for c in range(2)]
 oldports=CFG['ports'];CFG['ports']=1  # Synthetic port competition fixture only; restored before real-input tests.
 MM=np.array([190.,190.]);ks=[39,40];js=[43,44];hh=[task(q,k,j-1) for q in qq for k in ks for j in js if j-k>=4 and q['arrival']<=k/4];st=[dict(success=True,arrival=q['arrival']) for q in qq];mts=assemble(nn,hh,aa,[dict(success=True,arrival=h['arrival']) for h in hh],2,MM);fm,bb,fp,_,_=first_stage(hh,2,2,MM)
 ref=gp.Model('R30_SMALL_EXPLICIT');parameters(ref,out/'SMALL_EXPLICIT.log',1e-7);zz=ref.addMVar(len(hh)+2,lb=0,ub=np.r_[np.ones(len(hh)),MM],vtype=np.array([GRB.BINARY]*len(hh)+[GRB.CONTINUOUS]*2));yy=ref.addMVar(len(mts.cost),lb=mts.lo,ub=mts.hi);ref.addConstr(fm@zz<=bb);ref.addConstr(mts.A@yy+mts.B@zz<=mts.b);ref.addConstr(mts.H@yy+mts.G@zz==mts.d);ref.setObjective(fp.sum(0)@zz+mts.cost@yy);ref.optimize();assert ref.SolCount
 m,fs,rc=construct(qq,2,nn,aa,st,MM,'SMALL_COMPACT',ks,js,True);m.optimize();assert m.SolCount and abs(m.ObjVal-ref.ObjVal)<1e-6;save_raw(m,fs,rc,out/'SMALL_COMPACT_RAW',nn,aa,st)
 # Enumerate route/window choices only; each feasible combination optimizes continuous E and dispatch.
 import scipy.sparse as sp
 AA=sp.vstack([sp.hstack([fm,sp.csr_matrix((len(bb),len(mts.cost)))]),sp.hstack([mts.B,mts.A])],format='csr');HH=sp.hstack([mts.G,mts.H],format='csr');cc=np.r_[fp.sum(0),mts.cost];enum=[]
 by=[[None]+[i for i,h in enumerate(hh) if h['e']==e] for e in range(2)]
 for choice in itertools.product(*by):
  x=np.zeros(len(hh));x[[i for i in choice if i is not None]]=1
  lo=np.r_[x,np.zeros(2),mts.lo];hi=np.r_[x,MM,mts.hi]
  sol=linprog(cc,A_ub=AA,b_ub=np.r_[bb,mts.b],A_eq=HH,b_eq=mts.d,bounds=np.c_[lo,hi],method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  if sol.success:enum.append(float(sol.fun))
 assert enum and abs(min(enum)-m.ObjVal)<1e-6
 ok('independent_enumeration_explicit_compact',dict(enumerated=len(list(itertools.product(*by))),feasible=len(enum),enum_objective=min(enum),explicit=ref.ObjVal,compact=m.ObjVal,continuous_contracts=True))
 m.dispose();ref.dispose();CFG['ports']=oldports
 # Actual failed route branch in native LP: force a selected plan and ensure zero injection/finite shortfall.
 q=qq[0];h=task(q,39,43);h['contract']=30.;m,fs,rc=construct([q],1,nn,aa,[dict(success=False,arrival=np.inf)],MM[:1],'FAILED_ROUTE',ks,js,True);fix_first(fs,[h]);m.optimize();assert m.SolCount
 pp,yy=save_raw(m,fs,rc,out/'FAILED_ROUTE_RAW',nn,aa,[dict(success=False,arrival=np.inf)]);assert np.max(abs(yy[rc['original'].index['p']]))<1e-8 and abs(yy[rc['original'].index['short']][0]-30)<1e-6;m.dispose();ok('native_failed_route','Failed path selected: zero injection and activated shortfall30, inf handled first')
 # Real forecast44 positive injection: three vehicles, three electrically relevant stations; no pruning in production.
 net=layout()['net'];plans=[];qs=routes()
 for e,bus,k,j in [(0,14,55,81),(1,15,55,81),(2,21,92,96)]:
  c=net['station_buses'].index(bus);eligible=[q for q in qs if q['e']==e and q['c']==c and q['arrival']<=k/4]
  assert eligible;q=max(eligible,key=lambda q:q['d']);h=task(q,k,j-1);h['contract']=min(60.,h['Ebar']);plans.append(h)
 chosen=[next(q for q in qs if q['route_id']==h['route_id']) for h in plans];states=[dict(success=True,arrival=q['arrival']) for q in chosen]
 m,fs,rc=construct(chosen,30,net,a,states,full_bounds(),'REAL_NONZERO_FIXED',test=True);fix_first(fs,plans);m.optimize();assert m.SolCount
 pp,yy=save_raw(m,fs,rc,out/'REAL_NONZERO_RAW',net,a,states);assert .25*yy[rc['original'].index['p']].sum()>1
 for h in plans:
  trace=battery_trace(h['B']+100,h['arrival'],(h['r']+1)/4,yy[rc['original'].index['p']][h['e']].sum(0));assert trace['min_dc']>=100-1e-6
 ok('real44_nonzero_compact_raw_original_PQ',dict(actual_V2G=.25*yy[rc['original'].index['p']].sum(),raw_audit=read(out/'REAL_NONZERO_RAW/RAW_AUDIT.json')));m.dispose()
 from r30_fixed400 import weighted_cvar
 assert abs(weighted_cvar(np.array([10.,20.]),np.array([.97,.03]))-16.)<1e-12
 reject(lambda:weighted_cvar(np.array([10.,20.]),np.array([.5,.6])))
 ww=csvrows(R/'02_SOURCE_AND_CONFIG/WEIGHTS.csv');w=np.array([float(x['weight']) for x in ww]);assert len(w)==400 and abs(w.sum()-1)<1e-12 and np.ptp(w)>0
 assert all(int(x['local_sid'])==i for i,x in enumerate(ww))
 ok('nominal_weights_fractional_CVaR',dict(sum=float(w.sum()),maximum=float(w.max()),fractional_tail_example=16.,nonuniform=True))
 # Real competing admission attempt must fail without modifying this owner's lease.
 from r30_control import LEASE,ctrl
 import subprocess
 before=sha(LEASE);probe=out/'duplicate_owner_probe.py'
 probe.write_text("import sys;sys.path.insert(0,"+repr(str(R/'tools'))+");from r30_control import *\nwith admission('duplicate_test',8): raise RuntimeError('DUPLICATE_ADMITTED')\n")
 p=subprocess.Popen([sys.executable,'-B',str(probe)],stdout=subprocess.PIPE,stderr=subprocess.PIPE);stdout,stderr=p.communicate()
 dump(out/'DUPLICATE_PARENT_WAIT.json',dict(returncode=p.returncode,actual_parent_wait=True,stdout=stdout.decode(),stderr=stderr.decode()))
 assert p.returncode!=0 and 'DUPLICATE_R30_OWNER' in stderr.decode() and sha(LEASE)==before
 ok('unique_owner_actual_child_rejected','Dispatch lock checked; existing lease byte-identical; true subprocess wait recorded')
 dump(out/'COMPLETE.json',dict(passed=True,tests=records,epoch=time.time(),solver_version=gp.gurobi.version(),source_hashes={str(p.relative_to(R)):sha(p) for p in (R/'tools').glob('*.py') if p.name not in ['run_l08_k4l4_forecast44_domain_det_r30.py','run_parent.py']}))
