"""R12 data-only adapters. No procurement solver or physical generator imports."""
from pathlib import Path
import os,sys,json,csv,hashlib,time,shutil
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.dont_write_bytecode=True
import numpy as np
R=Path(__file__).resolve().parents[1];B=R.parent;CFG=json.loads((R/'02_SOURCE_AND_CONFIG/RESOLVED_CONFIG.json').read_text());BASE=Path(CFG['baseline']);SNAP=Path(CFG['snapshot']);R11=Path(CFG['r11']);R10=Path(CFG['r10']);R6=Path(CFG['r6']);R3=Path(CFG['r3']);EXP=Path(CFG['expansion']);R2=Path(CFG['r2'])
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item() if isinstance(a,np.generic) else str(a))+'\n');os.replace(t,p)
def npz(p,**a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**a)
 os.replace(t,p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvrows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rows,fields=None):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
  for x in rows:w.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in x.items()})
def scope(label='L08'):
 if label!='L08':raise ValueError('L13_RETIRED_L08_ONLY')
 assert read(B/'ACTIVE_PROJECT_SCOPE_R11.json')['active_layouts']==['L08']
def scene_path(sid):
 assert isinstance(sid,int) and 0<=sid<400
 return B/'L08_SAME_POOL_BALANCED_SELECT400_R28/04_SELECTED400/scenes'/f'{sid:04d}'
def layout(label='L08'):
 scope(label);p=SNAP/'inputs/LAYOUT_INPUTS/L08';return dict(label=label,arcs=csvrows(p/'ROAD_ARCS.csv'),stations=csvrows(p/'C__evcs_bus_road_mapping.csv'),net=read(p/'NETWORK_NEW_CLOCK.json'))
def normalized_roads(d,lay=None):
 if lay is None:lay=layout()
 a=load(Path(d)/'physical.npz');rows=lay['arcs'];j=np.array([int(x['parent_arc_id'])-1 for x in rows]);fraction=np.array([float(x['parent_fraction']) for x in rows]);assert np.allclose(np.bincount(j,weights=fraction,minlength=88),1,rtol=0,atol=1e-12)
 result=dict(times_h=a['times_h'],parent_road_movable=a['parent_road_movable'],parent_road_blocked=a['parent_road_blocked'])
 for k in ['movable','blocked','travel_min']:
  result['road_'+k]=a['parent_road_'+k][:,j].copy()
  if k=='travel_min':result['road_'+k]*=fraction
 return result

def routes_load():return [json.loads(l) for l in (R/'03_FORECAST44_ROUTES/ROUTES.jsonl').read_text().splitlines()]
def cache_key(candidate,physical_hash,network_hash,d,dc,origin,destination):return digest(dict(candidate=candidate,physical=physical_hash,network=network_hash,d=d,dc=dc,origin=origin,destination=destination,policy=CFG['energy_policy']))
