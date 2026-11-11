from pathlib import Path
import os,sys,json,hashlib,csv,time,resource,functools,types
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.dont_write_bytecode=True
import numpy as np
ROOT=R=Path(__file__).resolve().parents[1];B=R.parent;R12=B/'L08_CANDIDATE20_R10_DOMAIN_R12';R11=B/'L08_MEDIUM_FORECAST_SELECTION_R11';R10=B/'SECTIONIII_FULL_DWELL_AUX_R10'
paths=json.loads((R12/'02_SOURCE_AND_CONFIG/RESOLVED_CONFIG.json').read_text());BASE=Path(paths['baseline']);SNAP=Path(paths['snapshot']);EXP=Path(paths['expansion'])
OBJECTIVE_POLICY='BUSINESS_OCCUPANCY_TO_RESERVED_END_V1'
R13=B/'L08_DET_R10_CANDIDATE20_R13'
CFG=dict(objective_policy=OBJECTIVE_POLICY,version=R.name,aux_policy='ONBOARD_FULL_DWELL_TO_RESERVED_END_V1',aux_dc_kw=1.,dt=.25,periods=96,capacity_dc=500.,eta=.95,arrival_soc=.30,departure_soc=.20,distance_dc=1.2,wait_kw=1.,travel_aux_kw=0.,minimum_contract=15.,pmax=90.,ports=20,station_kw=1800.,fixed=25.,occupancy=30.,reservation=1.,contract=.02,short=1.,degradation=.05,epsilon_ls=1e-6,feasibility_tol=1e-9,optimality_tol=1e-9,integer_tol=1e-9,physical_residual_tol=1e-6,absolute_gap_usd=1.,threads=1,train=400,forecast_id=44,fixed_q_over_p=.33,critical_buses=[2,4,8,10,14,21,24,27,30,32],critical_price=50.,ordinary_price=10.)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)).encode()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n');os.replace(tmp,p)
def npz(p,**x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**x)
 os.replace(tmp,p)
def loadnp(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvrows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rows):
 rows=list(rows);Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scope(label='L08'):
 if label!='L08':raise ValueError('L13_RETIRED')
 assert read(B/'ACTIVE_PROJECT_SCOPE_R11.json')['active_layouts']==['L08']
def tou():
 t=np.arange(96)/4;return np.select([t<8,t<16,t<21],[.14296,.17297,.21540],default=.17297)
from energy_ledger import forecast,actual,battery_trace
def qualify(departure_dc,distance,arrival,k,r,success=True):
 v=actual(departure_dc-1.2*distance,arrival,k,r,success,1.);return v['qualified'],v['mu'],v['use_dc'],v['available_ac']
def ebar(departure_dc,distance,arrival,k,r):return forecast(departure_dc-1.2*distance,arrival,k,r)['cap_ac']
def first_cost(d,k,r):return 25+30*((r+1-d)/4)+(r-k+1)/4
@functools.lru_cache(None)
def layout(label='L08'):
 scope(label);d=R/'02_SOURCE_AND_CONFIG';st=csvrows(d/'C__evcs_bus_road_mapping.csv');n=read(d/'NETWORK_NEW_CLOCK.json');n['from']=n['line_from'];n['to']=n['line_to'];n['station_buses']=[int(x['n_c']) for x in st];n['shed_cost']=[50. if b in CFG['critical_buses'] else 10. for b in n['buses']];return dict(label=label,stations=st,net=n)
R29=B/'L08_K4L4_CALIBRATION_AND_FORECAST_R29'
R28=B/'L08_SAME_POOL_BALANCED_SELECT400_R28'
def scene_path(bank,sid=0):
 if bank=='FORECAST_SELECTED':
  if sid!=0:raise ValueError('Forecast handle must be 0; physical bank R29_FORECAST512_K4L4_V1/source44')
  return R29/'07_FORECAST512/scenes/0044'
 if bank!='TRAIN' or not isinstance(sid,int) or not 0<=sid<400:raise ValueError('support local_sid out of range')
 return R28/'04_SELECTED400/scenes'/f'{sid:04d}'
def load_scene(bank,sid=0,layout='L08'):
 scope(layout);d=scene_path(bank,sid);a=loadnp(d/'physical.npz');r=loadnp(d/'recovery.npz');ix=np.searchsorted(r['times_h'],np.arange(97)/4);assert np.array_equal(r['times_h'][ix],np.arange(97)/4)
 a['legacy_cumulative_operation']=a['line_operating_outages'].copy();a['line_operating_outages']=r['outage_current'][ix[:-1]].copy();a['line_terminal_outages']=r['outage_current'][ix[-1]].copy();assert np.array_equal(a['load_q_kvar'],.33*a['load_p_kw']);return a
def components(n,off):
 adj={b:[] for b in n['buses']}
 for u,v,o in zip(n['from'],n['to'],off):
  if not o:adj[u].append(v);adj[v].append(u)
 unseen=set(adj);groups=[]
 while unseen:
  todo=[min(unseen)];g=set()
  while todo:
   b=todo.pop()
   if b not in g:g.add(b);todo+=adj[b]
  unseen-=g;groups.append(sorted(g))
 return groups
def incident(n,a):
 z=np.zeros((len(n['station_buses']),96),bool);refs=[];bi={b:i for i,b in enumerate(n['buses'])}
 for t in range(96):
  rr=[]
  for group in components(n,a['line_operating_outages'][t]):
   stations=set(group)&set(n['station_buses']);rr.append(1 if 1 in group else min(stations or group))
   if 1 not in group and sum(a['load_p_kw'][t,bi[b]] for b in group)>CFG['epsilon_ls']:
    for c,b in enumerate(n['station_buses']):z[c,t]=b in group or z[c,t]
  refs.append(rr)
 return z,refs
C=types.SimpleNamespace(load_scene=load_scene,components=components)
@functools.lru_cache(None)
def routes(label='L08'):
 scope(label);qs=[];domain=loadnp(R/'04_R10_COMPACT_DOMAIN/DOMAIN.npz')
 for l in (R/'03_FORECAST44_ROUTES/ROUTES.jsonl').read_text().splitlines():
  q=json.loads(l)
  if q['route'] and domain['offset'][q['route_id']+1]>domain['offset'][q['route_id']]:qs.append(dict(route_id=q['route_id'],e=q['e'],c=q['c'],d=q['d'],departure_dc=q['departure_dc'],distance=q['route']['distance_km'],arrival=q['route']['arrival_h'],B=q['arrival_dc']-100,ev=q['ev']))
 assert len(qs)==read(R/'04_R10_COMPACT_DOMAIN/COMPLETE.json')['qualified_route_rows'];return qs
def full_bounds(label='L08'):
 scope(label);return np.array(read(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json')['M'])
def route_states(label,qs,bank='FORECAST_SELECTED',sid=0):
 scope(label)
 if bank=='FORECAST_SELECTED':return [dict(success=True,arrival=q['arrival']) for q in qs]
 assert bank=='TRAIN' and 0<=sid<400;z=loadnp(R/'05_SUPPORT400_ROUTE_REPLAY'/f'{sid:04d}.npz');return [dict(success=bool(z['success'][q['route_id']]),arrival=float(z['arrival_h'][q['route_id']])) for q in qs]
def task(q,k,r):return dict(q,k=int(k),r=int(r),Ebar=ebar(q['departure_dc'],q['distance'],q['arrival'],k,r))
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def plan_digest(plans):return digest(dict(objective_policy=OBJECTIVE_POLICY,input_identity=read(R/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json')['identity'],M_sha=sha(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json'),plans=plans))
