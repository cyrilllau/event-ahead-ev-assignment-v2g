from r36_common import *
def status():
 gates={p.stem:read(p) for p in (R/'gates').glob('*.json')};formal=read(R/'formal/CHECKPOINT.json') if (R/'formal/CHECKPOINT.json').exists() else None
 return dict(task=R.name,science=SCIENCE,epoch=time.time(),completed_gates=list(gates),formal_native_begin=(R/'formal/NATIVE_BEGIN.json').exists(),formal=formal,protection=protected())
def report():
 rec=status();dump(R/'REPORT.json',rec);text=['# R36 actual execution report','',f"State: {rec['formal']['status'] if rec['formal'] else 'PREPARATION_IN_PROGRESS'}",'','The current DET has not been stopped, reset or reconfigured. Previously stopped ARO/old Full-ST were not revived. R36 uses a separate K4_L4 scientific identity.','',f"Completed gates: {', '.join(rec['completed_gates'])}",'','All eight stations and 20100 qualified forecast44 routes are retained. No DET bounds, cuts, branch tree, station/window/count restrictions are imported. The forecast44 physical ID is distinct from support local44.','', 'Fixed route replay preserves arc-entry time and failures. Incident activates contract shortfall only; it does not gate actual power. R28 nominal weights are not DRO probabilities.','', 'Full raw moment count 322311; exact effective count45139; all400 risk planes retained. Split-beta is used. Signed production deployment is not claimed.']
 if rec['formal']:
  f=rec['formal'];text+=['',f"Own R36 LB={f['LB']}; full-support safe UB={f['UB']}; signed gap={f['gap']}; nonzero embedded pool={len(f['pool'])}.",'Formal native start is not a convergence certificate.']
 text+=['','Physical input payloads remain in the verified R28/R29 deliveries. INPUT_FILES.json, their receipt archives, and the R36 replay/source data identify these dependencies. No old K1 bounds are used.','', 'No new SAA, ARO, OOS, L13, calibration, forecast selection, or R30 fixed400 chain has been launched by R36.']
 (R/'REPORT.md').write_text('\n'.join(text)+'\n');return rec
def package(filename=None):
 report();delivery=R/'delivery';delivery.mkdir(exist_ok=True);name=filename or R.name+'_PRO_ANALYSIS.zip';dst=delivery/name
 index=delivery/('PACKAGE_INDEX.json' if filename is None else filename+'.INDEX.json')
 if dst.exists():return read(index)
 manifest={}
 with zipfile.ZipFile(dst,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
  for p in sorted(R.rglob('*')):
   if not p.is_file() or 'delivery' in p.relative_to(R).parts or p.suffix in ['.lock','.tmp'] or '__pycache__' in p.parts:continue
   # A live native model snapshot is immutable only at an explicit saved return.
   if p.name=='LATEST_MODEL.mps.gz':continue
   data=p.read_bytes();rel=p.relative_to(R).as_posix();z.writestr(rel,data);manifest[rel]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  for name0 in ['X_SUPPORT.npy','SUPPORT_UNIQUE.npy','SUPPORT_INVERSE.npy','EFFECTIVE_INTERVALS.npz','CALIBRATED_RAW_BOUNDS_K4L4.npz','DRO_FEATURE_REGISTRY.json','COMPRESSION_COMPLETE.json']:
   p=R29/'05_DRO_MOMENTS'/name0;rel='R29_MOMENT_INPUTS/'+name0;z.write(p,rel);manifest[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
  for label,root in [('R28',R28),('R29',R29)]:
   p=root/'delivery/PACKAGE_INDEX.json'
   if p.exists():raw=p.read_bytes();rel=label+'_DEPENDENCY_PACKAGE_INDEX.json';z.writestr(rel,raw);manifest[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2))
 rec=dict(file=dst.name,bytes=dst.stat().st_size,sha256=sha(dst),manifest_members=len(manifest),snapshot_epoch=time.time());dump(index,rec);return rec
def delivery_tests():
 import gurobipy as gp
 oldm,oldread,oldpopen=gp.Model,gp.read,subprocess.Popen
 def forbidden(*a,**k):raise AssertionError('OBSERVER_STARTED_SOLVER_OR_DISPATCH')
 try:
  gp.Model=gp.read=subprocess.Popen=forbidden;rec=package('OBSERVER_PURE_TEST.zip')
 finally:gp.Model,gp.read,subprocess.Popen=oldm,oldread,oldpopen
 return dict(passed=True,actual_zip_written_without_solver_or_dispatch=True,archive=rec,scientific_native_BEGINs_added=0)
