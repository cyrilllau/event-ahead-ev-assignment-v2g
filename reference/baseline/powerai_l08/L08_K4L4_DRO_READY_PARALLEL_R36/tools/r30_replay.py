from r12_common import *
CODES={0:'SUCCESS',1:'FORECAST_ROUTE_NOT_DECLARED',2:'ENTRY_OUTSIDE_HORIZON',3:'PERMANENT_ROAD_BLOCK',4:'WIND_TEMPORARY_CLOSURE',5:'NONFINITE_TRAVEL'}
def geometry(rows,lay):
 maxlen=max(len(q['route']['arc_ids']) if q['route'] else 0 for q in rows);n=len(rows);idx=np.full((n,maxlen),-1,np.int16);aid=np.full((n,maxlen),-1,np.int32);frac=np.ones((n,maxlen));count=np.zeros(n,np.int16);parent=np.full((n,maxlen),-1,np.int16)
 byid={int(x['a']):(j,x) for j,x in enumerate(lay['arcs'])}
 for q in rows:
  if not q['route']:continue
  i=q['route_id'];count[i]=len(q['route']['arc_ids'])
  for j,a in enumerate(q['route']['arc_ids']):
   aid[i,j]=a;p=q.get('prefix');orig=p['original_arc_id'] if p and a==p['virtual_arc_id'] else a
   idx[i,j],rr=byid[orig];parent[i,j]=int(rr['parent_arc_id'])
   if orig!=a:frac[i,j]=p['remaining_fraction']
 return dict(arc_index=idx,arc_id=aid,parent_arc_id=parent,remaining_fraction=frac,arc_count=count)
def replay(rows,g,a):
 n=len(rows);shape=g['arc_index'].shape;declared=np.array([q['route'] is not None for q in rows]);cur=np.array([q['d']/4 for q in rows],float);success=declared.copy();code=np.where(declared,0,1).astype(np.int8);failed=np.full(n,-1,np.int16)
 entry=np.full(shape,np.nan);slot=np.full(shape,-1,np.int16);mov=np.full(shape,-1,np.int8);block=np.full(shape,-1,np.int8);duration=np.full(shape,np.nan);states=a['times_h'][:-1]
 for j in range(shape[1]):
  ids=np.flatnonzero(success&(g['arc_count']>j));entry[ids,j]=cur[ids];ts=np.searchsorted(states,cur[ids],side='right')-1;inside=(ts>=0)&(cur[ids]<24)
  bad=ids[~inside];success[bad]=False;code[bad]=2;failed[bad]=j;ids=ids[inside];ts=ts[inside];arc=g['arc_index'][ids,j];slot[ids,j]=ts
  mm=a['road_movable'][ts,arc];bb=a['road_blocked'][ts,arc];tt=a['road_travel_min'][ts,arc]/60*g['remaining_fraction'][ids,j];mov[ids,j]=mm;block[ids,j]=bb;duration[ids,j]=tt
  reason=np.where(~mm,np.where(bb,3,4),np.where(np.isfinite(tt),0,5));bad=ids[reason!=0];code[bad]=reason[reason!=0];failed[bad]=j;success[bad]=False;ok=ids[reason==0];cur[ok]+=tt[reason==0]
 arrival=np.where(success,cur,np.inf);A=np.array([q['departure_dc']-1.2*q['route']['distance_km'] if q['route'] else np.nan for q in rows]);first=np.full(n,-1,np.int16);first[success]=np.ceil(arrival[success]*4).astype(np.int16)
 # Arrival energy is route distance based. On failure it is not an attained endpoint.
 return dict(route_id=np.arange(n,dtype=np.int32),success=success,arrival_h=arrival,arrival_dc=A,failure_code=code,failed_step=failed,entry_times_h=entry,entry_state_index=slot,observed_movable=mov,observed_blocked=block,arc_travel_h=duration,first_service_slot=first,arrival_qualified=success&(A>=150)&(arrival<=23))

def build():
 scope();rows=routes_load();lay=layout();g=geometry(rows,lay);out=R/'05_SUPPORT400_ROUTE_REPLAY';npz(out/'FIXED_ROUTE_GEOMETRY.npz',**g);ident=read(R/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json')['identity'];inputs=read(R/'02_SOURCE_AND_CONFIG/SCENE_IDENTITIES.json');stats=[];start=time.time()
 from r3_common import scene_path as path_for
 for bank,sid in [('FORECAST_SELECTED',0)]+[('TRAIN',i) for i in range(400)]:
  d=path_for(bank,sid);target=out/('FORECAST44.npz' if bank=='FORECAST_SELECTED' else f'{sid:04d}.npz');meta=target.with_suffix('.json')
  key=digest(dict(input_identity=ident,route_sha=sha(R/'03_FORECAST44_ROUTES/ROUTES.jsonl'),source=read(d/'PRIMITIVES_AND_IDENTITY.json'),physical=sha(d/'physical.npz'),recovery=sha(d/'recovery.npz'),geometry=sha(out/'FIXED_ROUTE_GEOMETRY.npz'),energy=sha(R/'tools/energy_ledger.py')))
  if meta.exists():
   rec=read(meta);assert rec['cache_key']==key and sha(target)==rec['sha256']
  else:
   assert not target.exists(),'Uncommitted payload preserved for metadata verification'
   z=replay(rows,g,normalized_roads(d,lay));assert np.all(np.isinf(z['arrival_h'][~z['success']]))
   if bank=='FORECAST_SELECTED':
    for q in rows:
     if q['route']:assert z['success'][q['route_id']] and abs(z['arrival_h'][q['route_id']]-q['route']['arrival_h'])<1e-10
   npz(target,**z);rec=dict(bank=bank,local_sid=sid,source_bank=read(d/'PRIMITIVES_AND_IDENTITY.json')['bank'],source_sid=read(d/'PRIMITIVES_AND_IDENTITY.json')['source_sid'],cache_key=key,route_rows=len(rows),success=int(z['success'].sum()),qualified=int(z['arrival_qualified'].sum()),sha256=sha(target),failure_counts={str(c):int((z['failure_code']==c).sum()) for c in CODES});dump(meta,rec)
  stats.append(rec)
  if sid%25==24:dump(out/'PROGRESS.json',dict(completed=sid+1,elapsed_s=time.time()-start));print('REPLAY',sid+1,flush=True)
 dump(out/'FILE_INDEX.json',stats);dump(out/'SCHEMA.json',dict(failure_codes=CODES,arrival='infinity on failure; no reroute',source='R28 local to source mapping preserved',energy='R10 full dwell to J; no rounding auxiliary time',cache='full input,route,physical,recovery,geometry,energy identity'))
 dump(out/'COMPLETE.json',dict(passed=True,support_count=400,forecast_replay_verified=True,rows_per_scenario=len(rows),epoch=time.time()));print('REPLAY400_COMPLETE',flush=True)
