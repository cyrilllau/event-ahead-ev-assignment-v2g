from r3_common import *
import contextlib,fcntl,shutil,traceback
V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
sys.path.insert(0,str(V/'tools'));import control_core as ctrl
sys.path.insert(0,str(B/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'));import r23_common as resource_reader
LEASE=V/'leases/R30_FORECAST44_DET.json';resource_reader.LEASE=LEASE

def inventory(label):
 x=dict(epoch=time.time(),signals_sent=[],old_files_written=[],resources=resource_reader.resources(),sources=resource_reader.source_inventory())
 baseline=R/'08_SCOPE/BASELINE.json'
 if not baseline.exists():dump(baseline,x)
 prior=read(baseline);changed=[p for p,h in prior['sources'].items() if x['sources'].get(p)!=h];x['source_changes']=changed
 dump(R/'08_SCOPE'/f'{label}.json',x);assert not changed,changed
 return x
@contextlib.contextmanager
def admission(stage,gib):
 with ctrl.lock(V/'DISPATCH.lock'):
  old=ctrl.read(LEASE);assert not ctrl.same(old.get('owner')),'DUPLICATE_R30_OWNER'
  a=resource_reader.resources();ok=not a['unknown_scheduler_limits'] and a['signed_headroom_bytes']>=gib*1024**3
  rec=dict(stage=stage,epoch=time.time(),request_GiB=gib,admitted=ok,resources=a)
  dump(R/'08_SCOPE'/f'ADMISSION_{time.time_ns()}.json',rec)
  if not ok:
   dump(R/'01_REPORTS/RESOURCE_WAIT.json',rec);raise RuntimeError('RETRYABLE_RESOURCE_WAIT')
  ctrl.dump(LEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=gib*1024**3,threads=1,layout='L08',method='R30_FORECAST44_DET',stage=stage,root=str(R)))
 # Hard process cap is task-local; solver retains a lower SoftMemLimit.
 oldlimit=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(gib*1024**3,oldlimit[1]))
 dump(R/'02_SOURCE_AND_CONFIG/RESOURCE_ADMISSION.json',dict(soft_mem_limit_GB=min(gib-2,10),request_GiB=gib,stage=stage))
 try:yield
 finally:
  resource.setrlimit(resource.RLIMIT_AS,oldlimit)
  with ctrl.lock(V/'DISPATCH.lock'):
   q=ctrl.read(LEASE)
   if ctrl.same(q.get('owner')) and q['owner']['pid']==os.getpid():ctrl.dump(LEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time(),stage=stage))

def inputs():
 scope();inventory('INPUT_START');files=[]
 for row in read(R/'instructions/MANIFEST.json')['files']:
  p=R/'instructions'/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
 for row in read(R/'instructions/SOURCE_EVIDENCE.json')['basis']:
  if 'archive' in row:
   root=R29 if row['archive'].startswith('L08_K4L4_CALIBRATION') else B/'L08_DET_R10_OCCUPANCY_COST_R15'
   p=root/row['member'];assert sha(p)==row['sha256'],str(p);files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
  else:assert sha(R/'instructions'/row['bundled_reference'])==row['sha256']
 cfg=dict(paths);cfg.update(task=R.name,root=str(R),forecast_candidate=44,forecast_bank='R29_FORECAST512_K4L4_V1',support_bank='R27_POOL2000_K4L4_V1',energy_policy=CFG['aux_policy'],search_limits=paths['search_limits'])
 dump(R/'02_SOURCE_AND_CONFIG/RESOLVED_CONFIG.json',cfg);dump(R/'02_SOURCE_AND_CONFIG/MODEL_POLICY.json',CFG)
 for name in ['NORMAL_BUSINESS_DAY_LEDGER.csv','ROAD_ARCS.csv','C__evcs_bus_road_mapping.csv','NETWORK_NEW_CLOCK.json']:
  p=SNAP/'inputs'/('FLEET' if name.startswith('NORMAL') else 'LAYOUT_INPUTS/L08')/name
  dest=R/'02_SOURCE_AND_CONFIG'/name;shutil.copyfile(p,dest);files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 for name in ['SELECTED400.csv','WEIGHTS.csv','SELECTION_LOCK.json','POOL_TO_SELECTED.csv']:
  shutil.copyfile(R29/'03_SUPPORT400'/name,R/'02_SOURCE_AND_CONFIG'/name)
 selected=csvrows(R29/'03_SUPPORT400/SELECTED400.csv');weights=csvrows(R29/'03_SUPPORT400/WEIGHTS.csv');assert len(selected)==len(weights)==400
 print('SUPPORT_SCHEMA',selected[0],weights[0],flush=True)
 checks=[]
 for bank,sid in [('FORECAST_SELECTED',0)]+[('TRAIN',i) for i in range(400)]:
  d=scene_path(bank,sid);co=read(d/'COMPLETE.json');meta=read(d/'PRIMITIVES_AND_IDENTITY.json')
  expected=44 if bank=='FORECAST_SELECTED' else int(selected[sid]['source_sid']);expected_bank='R29_FORECAST512_K4L4_V1' if bank=='FORECAST_SELECTED' else 'R27_POOL2000_K4L4_V1'
  assert meta['source_sid']==expected and meta['bank']==expected_bank and meta['profile']=='K4_L4';assert meta['calendar']=='2020-10-28T00:00:00Z'
  for name,h in co['files'].items():
   p=d/name;assert sha(p)==h,(bank,sid,name);files.append(dict(path=str(p),sha256=h,bytes=p.stat().st_size))
  a=loadnp(d/'physical.npz');rec=loadnp(d/'recovery.npz');assert a['load_p_kw'].shape==(96,33);assert np.array_equal(a['load_q_kvar'],.33*a['load_p_kw']);assert rec['outage_current'].shape[1]==32
  checks.append(dict(bank=bank,local_sid=sid,source_bank=expected_bank,source_sid=expected,directory=str(d),files=co['files']))
  if sid%50==49:print('INPUT_VERIFIED',bank,sid+1,flush=True)
 dump(R/'02_SOURCE_AND_CONFIG/SCENE_IDENTITIES.json',checks)
 identity=digest(dict(scenes=checks,assets=files[-4:],physical_files=[x for x in files if '/scenes/' in x['path']],fleet=sha(R/'02_SOURCE_AND_CONFIG/NORMAL_BUSINESS_DAY_LEDGER.csv'),roads=sha(R/'02_SOURCE_AND_CONFIG/ROAD_ARCS.csv'),stations=sha(R/'02_SOURCE_AND_CONFIG/C__evcs_bus_road_mapping.csv'),energy=sha(R/'tools/energy_ledger.py'),routing=sha(R/'tools/frozen_routing/forecast_routes.py'),weights=sha(R/'02_SOURCE_AND_CONFIG/WEIGHTS.csv'),policy=CFG))
 dump(R/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json',dict(identity=identity,forecast_bank='R29_FORECAST512_K4L4_V1',forecast_source_sid=44,support_bank='R27_POOL2000_K4L4_V1',support_size=400))
 dump(R/'08_SCOPE/INPUT_MANIFEST.json',files);dump(R/'01_REPORTS/PREFLIGHT_COMPLETE.json',dict(passed=True,identity=identity,files_verified=len(files),scenes=401,epoch=time.time()));print('INPUTS_VERIFIED',identity,flush=True)
