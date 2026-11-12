from r30_audit import *

def weighted_cvar(x,w,tail=.05):
 assert len(x)==len(w) and np.all(np.isfinite(x)) and np.all(w>=0) and abs(sum(w)-1)<1e-12
 remaining=tail;total=0.
 for i in np.argsort(-np.asarray(x),kind='stable'):
  mass=min(remaining,float(w[i]));total+=mass*x[i];remaining-=mass
  if remaining<=1e-15:break
 assert remaining<1e-12;return total/tail

def fixed400():
 assert read(R/'05_SELECTED_PLAN_AND_CERTIFICATES/AUDIT_COMPLETE.json')['absolute_gap_met'];plans=read(R/'05_SELECTED_PLAN_AND_CERTIFICATES/PLAN.json');ph=plan_digest(plans);out=R/'07_FIXED400';wrows=csvrows(R/'02_SOURCE_AND_CONFIG/WEIGHTS.csv');ids=read(R/'02_SOURCE_AND_CONFIG/SCENE_IDENTITIES.json')[1:];rows=[]
 for sid in range(400):
  d=out/f'{sid:04d}';src=ids[sid];wrow=wrows[sid];assert int(wrow['local_sid'])==sid and int(wrow['source_sid'])==src['source_sid']
  key=digest(dict(plan=ph,input=read(R/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json'),scene=src,replay=sha(R/'05_SUPPORT400_ROUTE_REPLAY'/f'{sid:04d}.npz'),M=sha(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json'),matrix=sha(R/'tools/contract_matrix.py'),cert=sha(R/'tools/lp_certificate.py')))
  if (d/'ACCEPTED.json').exists():
   ac=read(d/'ACCEPTED.json');assert ac['cache_key']==key
   for name,h in ac['files'].items():assert sha(d/name)==h
   row=read(d/'SUMMARY.json')
  else:
   if (d/'COMPLETE.json').exists():
    row=read(d/'COMPLETE.json');ce=read(d/'CERTIFICATE.json');assert row['plan_sha']==ph and row['physical_sha']==src['files']['physical.npz'] and row['recovery_sha']==src['files']['recovery.npz']
    assert ce['primal_sha']==sha(d/'PRIMAL.npz') and ce['dual_sha']==sha(d/'DUAL_AND_BOUNDS.npz')
   else:row=evaluate_one(plans,'TRAIN',sid,d)
   a=load_scene('TRAIN',sid);n=layout()['net'];incident_mask=incident(n,a)[0].any(axis=0)
   # Physical island anywhere, not only stations.
   incident_mask=np.array([any(1 not in g for g in components(n,off)) for off in a['line_operating_outages']])
   ix=np.searchsorted(a['times_h'],np.arange(96)/4);blocked=a['parent_road_blocked'][ix].any(axis=1)
   vv=read(d/'VEHICLES.json');row.update(local_sid=sid,source_bank=src['source_bank'],source_sid=src['source_sid'],w=float(wrow['weight']),incident=bool(incident_mask.any()),joint_event=bool((incident_mask&blocked).any()),route_success_count=sum(v['route_success'] for v in vv),qualified_count=sum(actual(h['B']+100,st['arrival'],h['k'],h['r'],st['success'])['qualified'] for h,st in zip(plans,route_states('L08',plans,'TRAIN',sid))),called_vehicles=sum(v['actual_ac']>1e-6 for v in vv))
   dump(d/'SUMMARY.json',row);dump(d/'ACCEPTED.json',dict(cache_key=key,files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='ACCEPTED.json'}))
  rows.append(row)
  if sid%10==9:dump(out/'PROGRESS.json',dict(completed=sid+1,plan_hash=ph,epoch=time.time()));print('FIXED400',sid+1,flush=True)
 w=np.array([x['w'] for x in rows]);assert abs(w.sum()-1)<1e-12;L=np.array([x['qL'] for x in rows]);U=np.array([x['qU'] for x in rows]);Q=np.array([x['Q'] for x in rows]);worst=int(np.argmax(Q));F=rows[0]['F']
 result=dict(passed=True,scenarios=400,plan_hash=ph,F=F,meanQ=float(w@Q),mean_interval=[float(w@L),float(w@U)],CVaR95_interval=[weighted_cvar(L,w),weighted_cvar(U,w)],maxQ_interval=[float(L.max()),float(U.max())],worst_local_sid=worst,worst_source_sid=rows[worst]['source_sid'],weighted={k:float(w@np.array([r[k] for r in rows])) for k in ['critical_ENS','other_ENS','actual_V2G_ac','shortfall_ac']},V2G_call_scenarios=sum(r['actual_V2G_ac']>1e-6 for r in rows),V2G_call_mass=sum(r['w'] for r in rows if r['actual_V2G_ac']>1e-6),first_stage_reoptimized=False,nominal_support_only=True,OOS=False,DRO_risk=False)
 result['subsets']={}
 for key in ['incident','joint_event']:
  mask=np.array([r[key] for r in rows]);mass=float(w[mask].sum());result['subsets'][key]=dict(count=int(mask.sum()),mass=mass,conditional_meanQ=float(w[mask]@Q[mask]/mass) if mass else None,conditional_total_cost=F+float(w[mask]@Q[mask]/mass) if mass else None)
 result['subset_definitions']=dict(Natural='all original R28 frozen support',Incident='any non-root electrical component during [0,24)',Joint='at least one 15min slot with electrical island and physical road blockage')
 csvout(out/'SCENARIO_SUMMARY.csv',rows);dump(out/'COMPLETE.json',result);print('FIXED400_COMPLETE',json.dumps(result),flush=True)
