from r36_common import *
from compact_first_stage import add_first
from exact_transform import add_strengthening
import gurobipy as gp
def inventory():
 for row in read(R/'instructions/FILE_MANIFEST.json'):
  p=R/'instructions'/row['file'];assert sha(p)==row['sha256']
 src={str(p):sha(p) for p in (OLD/'tools').rglob('*.py')}
 dump(R/'SOURCE_LOCK.json',src)
 procs=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    q=ctrl.proc(int(p.name))
    if q and str(B) in q.get('argv',''):procs.append(q)
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 paper=B/'L08_EXTREME_HAZARD_AND_FAILURE_AUDIT_R25/02_SOURCE_AND_PROVENANCE/paper_33_preserved_v4.tex'
 lines=paper.read_text().splitlines();hits=[dict(line=i+1,text=s) for i,s in enumerate(lines) if any(w in s.lower() for w in ['rerout','fixed-route','fixed route'])]
 active=[x for x in hits if not x['text'].lstrip().startswith('%')]
 dump(R/'ROUTE_POLICY_EVIDENCE.json',dict(source=str(paper),sha256=sha(paper),all_hits=hits,uncommented_hits=active,code_fixed_route=True,policy='FORECAST_SELECTED_FIXED_ROUTE_REPLAY',formal_policy_review_required=True))
 dump(R/'INVENTORY.json',dict(passed=True,protection=protected(),processes=procs,resources=resource_reader.resources(),epoch=time.time(),scientific_identity=SCIENCE))
 return read(R/'INVENTORY.json')
def payload():
 checks=[]
 for row in csvrows(R29/'03_SUPPORT400/PAYLOAD_FILE_MANIFEST.csv'):
  p=R28/row['path'];assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],str(p)
  checks.append(dict(path=str(p),sha256=row['sha256']))
 selected=csvrows(R/'02_SOURCE_AND_CONFIG/SELECTED400.csv');assert len(selected)==400
 for i,row in enumerate(selected):
  meta=read(scene_path('TRAIN',i)/'PRIMITIVES_AND_IDENTITY.json');assert meta['bank']=='R27_POOL2000_K4L4_V1' and meta['source_sid']==int(row['source_sid'])
 for name in ['SELECTED400.csv','WEIGHTS.csv','POOL_TO_SELECTED.csv','SELECTION_LOCK.json']:
  assert sha(R/'02_SOURCE_AND_CONFIG'/name)==sha(R29/'03_SUPPORT400'/name)
 manifest=read(R29/'delivery/RESTORE_MANIFEST.json');expected={x['path']:(x['full_sha256'],x['full_bytes']) for x in manifest}
 for rel,(h,size) in expected.items():
  if not rel.startswith(('04_CAL2000/','05_DRO_MOMENTS/','06_NONEMPTY/','07_FORECAST512/','02_SOURCE_CONFIG/')):continue
  p=R29/rel;assert p.stat().st_size==size and sha(p)==h,rel;checks.append(dict(path=str(p),sha256=h))
 for root,record in [(R28,read(R29/'03_SUPPORT400/RECEIPT_INDEX.json')),(R29,read(R29/'delivery/RECEIPT_SHA.json'))]:
  p=root/'delivery'/record['name'];assert sha(p)==record['sha256'];assert zipfile.ZipFile(p).testzip() is None
 dump(R/'INPUT_FILES.json',checks)
 return dict(passed=True,files=len(checks),input_manifest_sha=sha(R/'INPUT_FILES.json'),support=400,calibration=2000,forecast=512,old_files_modified=False)
def common_domain():
 qs=routes();assert len(qs)==20100 and set(q['c'] for q in qs)==set(range(8))
 a=loadnp(R/'04_R10_COMPACT_DOMAIN/DOMAIN.npz');M=full_bounds();w=read(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json')
 for e,ww in enumerate(w['witness']):assert abs(ww['Ebar']-M[e])<1e-10
 m=gp.Model('R36_COMMON');m.Params.OutputFlag=0;fs=add_first(m,qs,30,8,M);add_strengthening(m,fs);m.update()
 rv=[v for v in m.getVars() if v.VarName.startswith('route[')];assert len(rv)==20100 and all(v.UB==1 and v.LB==0 for v in rv)
 forbidden=['r32_','r33_','r34_','r35_','count_branch/','window_admission/'];assert not any(c.ConstrName.lower().startswith(tuple(forbidden)) for c in m.getConstrs())
 m.write(str(R/'COMMON_FIRST_STAGE.mps.gz'));rec=dict(passed=True,stations=8,vehicles=30,queries=len(a['e']),routes=len(qs),variables=m.NumVars,binaries=m.NumBinVars,constraints=m.NumConstrs,model_sha=sha(R/'COMMON_FIRST_STAGE.mps.gz'),global_M=M,global_M_witness_sha=sha(R/'04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json'),inherited_DET_fixings=0,domain_diff='Byte-identical unpruned R30 common inputs; no DET private models loaded')
 m.dispose();return rec
def replay400():
 from r30_replay import replay
 import r12_common as r12
 allrows=[json.loads(s) for s in (R/'03_FORECAST44_ROUTES/ROUTES.jsonl').read_text().splitlines()];geometry=loadnp(R/'05_SUPPORT400_ROUTE_REPLAY/FIXED_ROUTE_GEOMETRY.npz');records=[]
 for sid in range(400):
  p=verify_replay('TRAIN',sid);z=loadnp(p);assert len(z['success'])==23040 and np.isinf(z['arrival_h'][~z['success']]).all()
  meta=read(p.with_suffix('.json'));records.append(meta)
 # Independent regeneration for deterministic identities, including arc-entry observations.
 for sid in [0,1,399]:
  actual_replay=replay(allrows,geometry,r12.normalized_roads(scene_path('TRAIN',sid),r12.layout()));old=loadnp(verify_replay('TRAIN',sid))
  for k in actual_replay:assert np.array_equal(actual_replay[k],old[k],equal_nan=True),(sid,k)
 dump(R/'REPLAY_IDENTITIES.json',records)
 return dict(passed=True,support=400,queries_per_scene=23040,qualified_routes=20100,reused_completed_payload=True,independently_regenerated_ids=[0,1,399],manifest_sha=sha(R/'REPLAY_IDENTITIES.json'),no_rerouting=True)
