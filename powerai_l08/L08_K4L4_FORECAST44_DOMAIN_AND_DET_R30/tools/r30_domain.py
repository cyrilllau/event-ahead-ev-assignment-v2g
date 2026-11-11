from r12_common import *
from energy_ledger import forecast,actual

def bounds_for(q):
 k=np.arange(93)[:,None];J=np.arange(1,97)[None,:]
 if not q['route']:return np.zeros((93,96),bool),np.zeros((93,96))
 a=q['route']['arrival_h'];A=q['departure_dc']-1.2*q['route']['distance_km']
 cap=np.minimum(.95*(A-100-(J/4-a)),22.5*(J-k))
 valid=(A>=150)&(a<=k/4)&(J-k>=4)&(cap>=15)
 return valid,cap

def encode(domain,q,k,J):
 mx=int(domain['Jmax'][q,k]);assert k+4<=J<=mx
 return int(domain['offset'][q]+domain['koffset'][q,k]+J-(k+4))
def decode(domain,h):
 assert 0<=h<int(domain['offset'][-1]);q=int(np.searchsorted(domain['offset'],h,side='right')-1);v=h-int(domain['offset'][q]);k=int(np.searchsorted(domain['koffset'][q],v,side='right')-1)
 return q,k,k+4+v-int(domain['koffset'][q,k])
def frozen_M(route_subset=None):
 # Subsets deliberately never recompute this whole-domain bound.
 return np.asarray(read(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json')['M'])
def scenario_window(record,q,k,r):
 return actual(float(record['arrival_dc'][q]),float(record['arrival_h'][q]),k,r,success=bool(record['success'][q]))
def build():
 scope();assert read(R/'03_FORECAST44_ROUTES/COMPLETE.json')['passed'];rows=routes_load();out=R/'04_R10_COMPACT_DOMAIN';N=len(rows);jm=np.full((N,93),-1,np.int16);kc=np.zeros((N,93),np.int64);M=np.zeros(30);wit=[None]*30;summary=[]
 for q in rows:
  i=q['route_id'];v,cap=bounds_for(q);kc[i]=v.sum(axis=1);jm[i]=np.where(kc[i]>0,np.max(np.where(v,np.arange(1,97),-1),axis=1),-1)
  reconstructed=(np.arange(1,97)[None,:]>=np.arange(93)[:,None]+4)&(np.arange(1,97)[None,:]<=jm[i,:,None]);assert np.array_equal(v,reconstructed)
  best=float(np.max(np.where(v,cap,0)))
  if best>M[q['e']]:
   k,j=np.unravel_index(np.argmax(np.where(v,cap,0)),cap.shape);M[q['e']]=best;wit[q['e']]=dict(route_id=i,ev=q['ev'],k=int(k),r=int(j),exclusive_J_slot=int(j+1),Ebar=best,A=q['arrival_dc'],arrival=q['route']['arrival_h'],aux_dc=(j+1)/4-q['route']['arrival_h'])
  summary.append(dict(route_id=i,ev=q['ev'],station=q['station'],d=q['d'],logical_tasks=int(kc[i].sum()),max_Ebar_ac=best))
 ko=np.column_stack([np.zeros(N,np.int64),np.cumsum(kc,axis=1)]);off=np.r_[0,np.cumsum(kc.sum(axis=1))];npz(out/'DOMAIN.npz',route_id=np.arange(N),e=np.array([q['e'] for q in rows]),c=np.array([q['c'] for q in rows]),d=np.array([q['d'] for q in rows]),Jmax=jm,kcount=kc,koffset=ko,offset=off,arrival_hat=np.array([q['route']['arrival_h'] if q['route'] else np.inf for q in rows]),arrival_dc=np.array([q['arrival_dc'] if q['route'] else np.nan for q in rows]))
 dump(out/'GLOBAL_ENERGY_BOUNDS.json',dict(M=M,witness=wit,domain='ALL forecast44 qualified e,c,d,k,r; never subset-recomputed',routes_sha=sha(R/'03_FORECAST44_ROUTES/ROUTES.jsonl'),energy_module_sha=sha(R/'tools/energy_ledger.py'),logical_task_count=int(off[-1])))
 csvout(out/'ROUTE_DOMAIN_COUNTS.csv',summary);csvout(out/'VEHICLE_GLOBAL_M.csv',[dict(ev=wit[e]['ev'],M_ac=M[e],witness_route=wit[e]['route_id'],k=wit[e]['k'],r=wit[e]['r']) for e in range(30)])
 nvalid=int(np.count_nonzero(kc.sum(axis=1)));scale=dict(qualified_route_departure_binary_candidates=nvalid,start_onehot_candidates=30*93,exclusive_end_onehot_candidates=30*93,window_binary_candidates=30*96,station_reservation_binary_candidates=30*8*96,vehicle_continuous_contract_E=30,total_logical_tasks=int(off[-1]),materialized_h_binary_variables=0,estimate_note='Structural variable counts for a one-route-per-EV compact input; no model/MIP assembled. Auxiliary linearization variables are implementation dependent.')
 dump(out/'COMPLETE.json',dict(passed=True,route_rows=N,qualified_route_rows=nvalid,logical_tasks=int(off[-1]),scale_estimate=scale,domain_sha=sha(out/'DOMAIN.npz')))
 dump(out/'SCHEMA.json',dict(J='exclusive integer boundary 1..96; r=J-1',valid='for q,k: k+4 <= J <= Jmax[q,k], else no task',ordering='route_id, k, J; zero-based global h index',energy='min(0.95*(A-100-(J/4-arrival)),22.5*(J-k))',representation='Lossless full logical domain; arrays do not prune scenarios or valid windows',actual_window_adapter='tools/r12_domain.py:scenario_window calls byte-identical frozen R10 actual()',compact_components=['route/departure selection','start','exclusive end','window','station reservation','E_e'],scope='No optimizer invocation'))
 print('DOMAIN_COMPLETE',int(off[-1]),flush=True)
if __name__=='__main__':build()
