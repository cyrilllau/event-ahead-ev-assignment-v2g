from r12_common import *
from r12_route_adapter import source_states,network_for_state,search_route,replay_route

def build():
 scope();assert read(R/'01_REPORTS/PREFLIGHT_COMPLETE.json')['passed'];out=R/'03_FORECAST44_ROUTES';out.mkdir(parents=True,exist_ok=True)
 lay=layout();a=normalized_roads(B/'L08_K4L4_CALIBRATION_AND_FORECAST_R29/07_FORECAST512/scenes/0044',lay);states=source_states();roster=sorted({x['ev'] for x in states});dump(out/'SOURCE_STATES.json',states);dump(out/'ROSTER.json',roster);identity=read(R/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json')['identity'];ph=sha(B/'L08_K4L4_CALIBRATION_AND_FORECAST_R29/07_FORECAST512/scenes/0044/physical.npz');cache={};nets={};rows=[];searches=0;hits=0;start=time.time()
 with (out/'ROUTES.jsonl.inprogress').open('w') as f:
  for si,state in enumerate(states):
   e=roster.index(state['ev']);d=state['d'];net=origin=prefix=None
   saved=out/'queries'/f'{si:04d}.json'
   if saved.exists():
    prior=read(saved);assert prior['identity']==identity
    assert digest(prior['rows'])==prior['rows_sha']
    for row in prior['rows']:assert row['route_id']==len(rows);rows.append(row);f.write(json.dumps(row,allow_nan=False)+'\n')
    continue
   before=len(rows)
   if d<=92:
    nk=(state['node'],state['kind'],state['arc'],state['fraction'])
    if nk not in nets:nets[nk]=network_for_state(lay,a,state)
    net,origin,prefix=nets[nk]
   for ci,station in enumerate(lay['stations']):
    qid=len(rows);row=dict(route_id=qid,e=e,ev=state['ev'],depot=state['ev'].split('-')[0],c=ci,station=station['c'],station_bus=int(station['n_c']),destination=int(station['road_node_c']),d=d,departure_h=d/4,departure_dc=state['dc'],source_state=state,origin=origin,prefix=prefix,forecast_candidate=44,input_identity=identity,cache_hit=False,route=None,reason=None)
    if d>92:row.update(status='COMPLETE_INELIGIBLE_DEPARTURE',reason='DEPARTURE_AFTER_23')
    elif state['dc']<150:row.update(status='COMPLETE_INELIGIBLE_DEPARTURE',reason='DEPARTURE_SOC_LT30')
    else:
     key=cache_key(44,ph,net.network_hash,d,state['dc'],origin,row['destination']);row['query_cache_key']=key
     if key in cache:result=cache[key];row['cache_hit']=True;hits+=1
     else:
      obj=search_route(net,origin,row['destination'],d/4,state['dc']/500,500,1.2,0,.30,latest_arrival_h=23,**CFG['search_limits']);result=obj.as_dict();cache[key]=result;searches+=1
      if obj.route is not None:assert replay_route(net,obj.route.arc_ids,d/4,state['dc']/500,500,1.2,0,origin=origin,destination=row['destination']).as_dict()==obj.route.as_dict()
     row.update(result)
     if result['status']!='COMPLETE':
      f.write(json.dumps(row)+'\n');f.flush();dump(out/'FAILURE.json',row);raise RuntimeError('INCOMPLETE_FORECAST_ROUTE_SEARCH')
     if row['route'] is None:row['reason']='NO_QUALIFIED_FORECAST_ROUTE'
    rr=row['route'];row['forecast_success']=rr is not None
    if rr is not None:
     row['arrival_dc']=state['dc']-1.2*rr['distance_km'];row['arrival_soc']=row['arrival_dc']/500;assert row['arrival_dc']>=150 and rr['arrival_h']<=23
     if not rr['arc_ids']:assert origin==row['destination'] and rr['distance_km']==0
    else:row['arrival_dc']=None;row['arrival_soc']=None
    row['route_input_hash']=digest({k:v for k,v in row.items() if k not in ['certificate','cache_hit','status','reason']});f.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
   dump(saved,dict(identity=identity,rows=rows[before:],rows_sha=digest(rows[before:])))
   if (si+1)%96==0:
    f.flush();dump(out/'PROGRESS.json',dict(vehicles_done=e+1,rows=len(rows),searches=searches,cache_hits=hits,elapsed_s=time.time()-start));print('ROUTES',e+1,len(rows),flush=True)
 os.replace(out/'ROUTES.jsonl.inprogress',out/'ROUTES.jsonl')
 assert len(rows)==23040
 summary=[dict(route_id=q['route_id'],ev=q['ev'],depot=q['depot'],station=q['station'],bus=q['station_bus'],d_h=q['d']/4,departure_dc=q['departure_dc'],source_kind=q['source_state']['kind'],source_node=q['source_state']['node'],source_arc=q['source_state']['arc'],source_fraction=q['source_state']['fraction'],success=q['forecast_success'],arrival_h=q['route']['arrival_h'] if q['route'] else None,distance_km=q['route']['distance_km'] if q['route'] else None,arrival_soc=q['arrival_soc'],failure_reason=q['reason'],route_hash=q['route_input_hash']) for q in rows];csvout(out/'ROUTE_QUALIFICATION_SUMMARY.csv',summary)
 dump(out/'COMPLETE.json',dict(passed=True,queries=len(rows),qualified_routes=sum(x['forecast_success'] for x in rows),failed_or_ineligible_routes=sum(not x['forecast_success'] for x in rows),searches=searches,exact_query_cache_hits=hits,cache_scope='R29_FORECAST512_K4L4_V1/source44 only, full network/physical/input identity',route_sha=sha(out/'ROUTES.jsonl'),input_identity=identity,elapsed_s=time.time()-start));print('FORECAST_ROUTES_COMPLETE',flush=True)
if __name__=='__main__':build()
