from r3_common import *

def reachability():
 qs=routes();n=layout()['net'];a=load_scene('FORECAST_SELECTED');zeta,refs=incident(n,a);windows=[(13.75,20.25),(22.5,22.75),(23.25,24.)];rows=[];topology=[]
 for t in range(96):topology.append(dict(t=t,time_h=t/4,groups=components(n,a['line_operating_outages'][t]),zeta=zeta[:,t].tolist()))
 for wi,(left,right) in enumerate(windows):
  k=int(min(left,23)*4);J=max(int(right*4),k+4)
  assert J<=96
  for e in range(30):
   for c,bus in enumerate(n['station_buses']):
    cand=[q for q in qs if q['e']==e and q['c']==c and q['arrival']<=k/4]
    caps=[ebar(q['departure_dc'],q['distance'],q['arrival'],k,J-1) for q in cand];valid=[(q,cap) for q,cap in zip(cand,caps) if cap>=15]
    best=max(valid,key=lambda x:(x[0]['d'],x[1])) if valid else None
    rows.append(dict(event=wi,event_start=left,event_end=right,e=e,station=c,bus=bus,electrical_coverage_slots=int(zeta[c,int(left*4):int(right*4)].sum()),reachable=bool(valid),valid_route_queries=len(valid),latest_departure=best[0]['d']/4 if best else None,arrival=best[0]['arrival'] if best else None,arrival_SOC=best[0]['B']/5+20 if best else None,K=k/4,J=J/4,available_AC=best[1] if best else None,route_id=best[0]['route_id'] if best else None,ports=20,station_kw=1800,individual_upper_bound_not_joint_supply=True))
 csvout(R/'06_REACHABILITY/EV_STATION_EVENTS.csv',rows);dump(R/'06_REACHABILITY/TOPOLOGY96.json',topology);dump(R/'06_REACHABILITY/COMPLETE.json',dict(passed=True,rows=len(rows),windows=windows,qualification_deadline=23,no_station_pruning=True,individual_capacities_not_additive=True))
