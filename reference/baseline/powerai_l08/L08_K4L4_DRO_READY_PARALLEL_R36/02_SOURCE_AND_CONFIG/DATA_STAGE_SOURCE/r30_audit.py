from r3_common import *
from contract_matrix import assemble,first_stage
from lp_certificate import certify

def decode_plan():
 out=R/'04_MODEL_AND_NATIVE_SOLVE';names=read(out/'VARIABLE_NAMES.json');vals=loadnp(out/'NATIVE_VALUES.npz')['values'];v=dict(zip(names,vals));qs=routes();plans=[];domain=loadnp(R/'04_R10_COMPACT_DOMAIN/DOMAIN.npz')
 for e in range(30):
  ids=[i for i,q in enumerate(qs) if q['e']==e and v[f'route[{i}]']>.5]
  if not ids:assert abs(v[f'E[{e}]'])<1e-6;continue
  assert len(ids)==1;q=qs[ids[0]];ks=[k for k in range(93) if v[f'start[{e},{k}]']>.5];js=[j for j in range(1,97) if v[f'end[{e},{j}]']>.5];assert len(ks)==len(js)==1;k=ks[0];J=js[0];rid=q['route_id'];assert k+4<=J<=domain['Jmax'][rid,k];h=task(q,k,J-1);h['contract']=float(v[f'E[{e}]']);h['global_h_index']=int(domain['offset'][rid]+domain['koffset'][rid,k]+J-k-4);assert 15-1e-6<=h['contract']<=h['Ebar']+1e-6;plans.append(h)
 return plans,v

def evaluate_one(plans,bank,sid,out,compact_y=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);n=layout()['net'];M=full_bounds();qs={q['route_id']:q for q in routes()};contracts=np.zeros(30);tasks=[];ports=np.zeros((8,96),int)
 for h in plans:
  q=qs[h['route_id']];assert h['e']==q['e'];hh=task(q,h['k'],h['r']);assert 15-1e-6<=h['contract']<=hh['Ebar']+1e-6;contracts[h['e']]=h['contract'];tasks.append(hh);ports[q['c'],h['k']:h['r']+1]+=1
 assert ports.max(initial=0)<=20;st=route_states('L08',tasks,bank,sid);a=load_scene(bank,sid);mat=assemble(n,tasks,a,st,30,M);z=np.r_[np.ones(len(tasks)),contracts];Fmat,b,parts,_,_=first_stage(tasks,30,8,M);assert max(Fmat@z-b,default=0)<=1e-6;F=float(parts.sum(0)@z)
 if compact_y is not None:assert mat.residual(compact_y,z)<=1e-6,('NATIVE_ORIGINAL_RESIDUAL',mat.residual(compact_y,z))
 sol=certify(mat,z,a);cert=sol['certificate'];y=sol['y'];ix=mat.index;pd=y[ix['p']];shed=y[ix['shed']];short=y[ix['short']];traces=[];vehicles=[]
 for h,s in zip(tasks,st):
  e=h['e'];A=h['B']+100;k=h['k'];r=h['r'];p=pd[e].sum(0);nz=np.flatnonzero(p>1e-6);j=(r+1)/4;trace=None
  if s['success'] and s['arrival']<j:
   trace=battery_trace(A,s['arrival'],j,p,extra=[k/4]);assert trace['min_dc']>=100-1e-6;traces.append(dict(e=e,ev=h['ev'],**trace))
  vehicles.append(dict(e=e,ev=h['ev'],route_id=h['route_id'],station='CS'+str(h['c']+1),bus=n['station_buses'][h['c']],d_h=h['d']/4,forecast_arrival=h['arrival'],actual_arrival=s['arrival'] if s['success'] else None,route_success=s['success'],K_h=k/4,J_h=j,reserve_duration_h=j-k/4,arrival_soc_pct=A/5 if s['success'] else None,aux_dc_kwh=trace['aux_dc'] if trace else None,contract_ac=contracts[e],actual_ac=float(.25*p.sum()),shortfall_ac=float(short[e]),soc_J_pct=trace['terminal_dc']/5 if trace else None,first_injection_h=float(nz[0]/4) if len(nz) else None,last_injection_slot_h=float(nz[-1]/4) if len(nz) else None,last_injection_end_h=float((nz[-1]+1)/4) if len(nz) else None,actual_departure_h=h['d']/4))
 # Connected-component regression: no incident gating is added; positive-cost injection should be zero in grid-connected components.
 connected_p=0.
 for t in range(96):
  root=next(g for g in components(n,a['line_operating_outages'][t]) if 1 in g)
  for c,bus in enumerate(n['station_buses']):
   if bus in root:connected_p=max(connected_p,float(pd[:,c,t].sum()))
 assert connected_p<=1e-6,('CONNECTED_V2G',bank,sid,connected_p)
 crit=np.array(n['shed_cost'])==50;rec=dict(passed=True,objective_policy=OBJECTIVE_POLICY,bank=bank,scenario=sid,F=F,Q=cert['primal'],qL=cert['qL'],qU=cert['qU'],critical_ENS=float(.25*shed[:,crit].sum()),other_ENS=float(.25*shed[:,~crit].sum()),actual_V2G_ac=float(.25*pd.sum()),shortfall_ac=float(short.sum()),raw_residual=mat.residual(y,z),native_original_residual=mat.residual(compact_y,z) if compact_y is not None else None,cost_parts=mat.parts@y,first_cost_parts=parts@z,plan_sha=plan_digest(plans),physical_sha=sha(scene_path(bank,sid)/'physical.npz'),recovery_sha=sha(scene_path(bank,sid)/'recovery.npz'),replay_sha=sha(R/'05_SUPPORT400_ROUTE_REPLAY/FORECAST44.npz') if bank=='FORECAST_SELECTED' else sha(R/'05_SUPPORT400_ROUTE_REPLAY'/f'{sid:04d}.npz'),M_sha=sha(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json'),max_grid_connected_V2G_kw=connected_p)
 npz(out/'PRIMAL.npz',y=y,z=z,p_dis=pd,shed=shed,short=short,grid_p=y[ix['grid_p']],grid_q=y[ix['grid_q']],line_p=y[ix['line_p']],line_q=y[ix['line_q']],u=y[ix['u']],outage_current=a['line_operating_outages'],zeta=np.array(mat.metadata['zeta']),port_reservation=ports)
 npz(out/'DUAL_AND_BOUNDS.npz',**{k:v for k,v in sol.items() if k not in ['y','certificate']});cert.update(plan_sha=rec['plan_sha'],physical_sha=rec['physical_sha'],recovery_sha=rec['recovery_sha'],primal_sha=sha(out/'PRIMAL.npz'),dual_sha=sha(out/'DUAL_AND_BOUNDS.npz'));dump(out/'CERTIFICATE.json',cert);dump(out/'SOC.json',traces);dump(out/'VEHICLES.json',vehicles);dump(out/'COMPLETE.json',rec);return rec

def audit():
 out=R/'05_SELECTED_PLAN_AND_CERTIFICATES'
 if '--resume' in sys.argv and (out/'AUDIT_COMPLETE.json').exists():
  rec=read(out/'AUDIT_COMPLETE.json');cert=read(out/'FORECAST44/CERTIFICATE.json');assert rec['passed'] and rec['plan_sha']==plan_digest(read(out/'PLAN.json'));assert cert['primal_sha']==sha(out/'FORECAST44/PRIMAL.npz') and cert['dual_sha']==sha(out/'FORECAST44/DUAL_AND_BOUNDS.npz');print('AUDIT_ALREADY_COMMITTED_AND_HASH_VERIFIED',json.dumps(rec),flush=True);return
 native=read(R/'04_MODEL_AND_NATIVE_SOLVE/SOLVE_COMPLETE.json');assert native['solutions']>0 and native['integer_residual']<=CFG['integer_tol'];assert sha(R/'04_MODEL_AND_NATIVE_SOLVE/NATIVE_VALUES.npz')==native['values_sha'];plans,vals=decode_plan();out=R/'05_SELECTED_PLAN_AND_CERTIFICATES';dump(out/'PLAN.json',plans)
 n=layout()['net'];a=load_scene('FORECAST_SELECTED');mat=assemble(n,[],a,[],30,full_bounds());mapping=loadnp(R/'04_MODEL_AND_NATIVE_SOLVE/PF_LIFT.npz');yy=np.empty(len(mat.cost));reduced=np.array([vals[f's0_y[{i}]'] for i in range(len(mapping['keep']))]);yy[mapping['keep']]=reduced;yy[mapping['q_columns']]=.33*yy[mapping['p_columns']]
 rec=evaluate_one(plans,'FORECAST_SELECTED',0,out/'FORECAST44',yy);rec.update(solver_LB=native['solver_LB'],audited_UB=rec['F']+rec['qU']);rec['signed_gap']=rec['audited_UB']-rec['solver_LB'];assert rec['signed_gap']>=0;rec['absolute_gap_met']=rec['signed_gap']<=1;rec['selected_count']=len(plans);rec['solver_UB']=native['solver_UB'];rec['native_raw_recourse_cost']=float(mat.cost@yy);rec['fixed_recourse_improvement']=rec['native_raw_recourse_cost']-rec['Q'];assert abs(rec['F']+rec['native_raw_recourse_cost']-native['solver_UB'])<=1e-5;assert rec['Q']<=rec['native_raw_recourse_cost']+1e-5
 vv=read(out/'FORECAST44/VEHICLES.json');roster=read(R/'03_FORECAST44_ROUTES/ROSTER.json');allv=[]
 for e,ev in enumerate(roster):
  row=next((x for x in vv if x['e']==e),None)
  if row:row=dict(selected=True,**row);h=next(h for h in plans if h['e']==e);row.update(fixed_fee=25.,occupancy_fee=30*(h['r']+1-h['d'])/4,reserve_fee=(h['r']-h['k']+1)/4,contract_fee=.02*h['contract'])
  else:row=dict.fromkeys(['selected']+(['e','ev','route_id','station','bus','d_h','forecast_arrival','actual_arrival','route_success','K_h','J_h','reserve_duration_h','arrival_soc_pct','aux_dc_kwh','contract_ac','actual_ac','shortfall_ac','soc_J_pct','first_injection_h','last_injection_slot_h','last_injection_end_h','actual_departure_h'])+['fixed_fee','occupancy_fee','reserve_fee','contract_fee']);row.update(selected=False,e=e,ev=ev,contract_ac=0.,actual_ac=0.,shortfall_ac=0.,fixed_fee=0.,occupancy_fee=0.,reserve_fee=0.,contract_fee=0.)
  row.update(total_first_stage_cost=sum(row[x] for x in ['fixed_fee','occupancy_fee','reserve_fee','contract_fee']),business_occupancy_h=(row['J_h']-row['d_h']) if row['selected'] else 0.,objective_policy=OBJECTIVE_POLICY)
  if not row['selected']:row.update(d_h=None,K_h=None,J_h=None)
  allv.append(row)
 csvout(out/'ALL_30_EV_DECISIONS.csv',allv);dump(out/'AUDIT_COMPLETE.json',rec);print('AUDIT_COMPLETE',json.dumps(read(out/'AUDIT_COMPLETE.json')),flush=True)

