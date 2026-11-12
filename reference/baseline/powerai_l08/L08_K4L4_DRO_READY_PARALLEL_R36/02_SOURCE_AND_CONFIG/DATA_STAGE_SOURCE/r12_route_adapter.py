from r12_common import *
from frozen_routing.forecast_routes import Arc,ForecastNetwork,search_route,replay_route

def source_states():
 rows=csvrows(SNAP/'inputs/FLEET/NORMAL_BUSINESS_DAY_LEDGER.csv');group={}
 for row in rows:group.setdefault(row['ev_id'],[]).append(row)
 result=[]
 for ev,evrows in sorted(group.items()):
  for d in range(96):
   t=d/4
   matches=[r for r in evrows if float(r['start_relative_dispatch_h'])<=t<float(r['end_relative_dispatch_h'])]
   assert len(matches)==1,(ev,d,len(matches))
   r=matches[0];start=float(r['start_relative_dispatch_h']);end=float(r['end_relative_dispatch_h']);f=(t-start)/(end-start)
   dc=float(r['battery_start_dc_kwh'])+f*(float(r['battery_end_dc_kwh'])-float(r['battery_start_dc_kwh']))
   result.append(dict(ev=ev,d=d,dc=dc,node=int(r['start_node']),kind=r['kind'],arc=int(r['road_arc_id']) if r['kind']=='TRAVEL' else None,fraction=f,event=r['source_event_id']))
 assert len(group)==30 and len(result)==2880
 return result

def network_for_state(lay,a,state):
 rows=lay['arcs'];arcs=tuple(Arc(int(r['a']),int(r['m_road']),int(r['n_road']),float(r['D_a_km']),float(r['T0_a_min'])/60) for r in rows)
 travel=a['road_travel_min'][:-1]/60
 travel=np.where(a['road_movable'][:-1],travel,np.inf)
 origin=state['node'];prefix=None
 if state['kind']=='TRAVEL':
  source=csvrows(SNAP/'inputs/LAYOUT_INPUTS/L08/ROAD_ARCS.csv')[state['arc']-1]
  parent=int(source['parent_arc_id']);f0=float(source['parent_start_fraction']);f1=float(source['parent_end_fraction']);f=f0+state['fraction']*(f1-f0)
  candidates=[r for r in rows if int(r['parent_arc_id'])==parent and float(r['parent_start_fraction'])<=f<float(r['parent_end_fraction'])]
  assert len(candidates)==1,(lay['label'],state,parent,f)
  row=candidates[0];aid=int(row['a']);ar=arcs[aid-1]
  local=(f-float(row['parent_start_fraction']))/(float(row['parent_end_fraction'])-float(row['parent_start_fraction']))
  if local==0:origin=ar.source
  else:
   residual=1-local;origin=-1;vid=1000000+aid;va=Arc(vid,origin,ar.target,ar.distance_km*residual,ar.freeflow_h*residual)
   arcs=arcs+(va,);travel=np.column_stack([travel,travel[:,aid-1]*residual])
   prefix=dict(virtual_arc_id=vid,original_arc_id=aid,remaining_fraction=residual,source=origin,target=ar.target,distance_km=va.distance_km)
 return ForecastNetwork(arcs,a['times_h'][:-1],travel,24.),origin,prefix
