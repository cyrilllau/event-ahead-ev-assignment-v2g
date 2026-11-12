"""R23 isolated preparation; shared controller is used read-only except own lease."""
from pathlib import Path
import os,sys,json,time,hashlib,contextlib,shutil,math
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];B=R.parent
OLD=B/'L08_MARGINAL_ONLY_DRO400_PARALLEL_R22'
V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
sys.path.insert(0,str(V/'tools'))
import control_core as ctrl
read,dump,sha,digest=ctrl.read,ctrl.dump,ctrl.sha,ctrl.digest
LEASE=V/'leases/R23_EXACT_CONTINUOUS_DIAGNOSTIC.json'
INSTR=R/'instructions/L08_MARGINAL_DRO_EXACT_ACCELERATION_R23'
def record(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def ownwrite(p,data):
 p=Path(p).resolve();assert R==p or R in p.parents
 dump(p,data)
def resources():
 ps=ctrl.allproc();seen=set();entries=[]
 for f in sorted((V/'leases').glob('*.json')):
  z=read(f);o=z.get('owner');sup=z.get('supervisor')
  if f==LEASE:continue
  root=o if ctrl.same(o) else sup if ctrl.same(sup) else None
  if root is None:
   # Never reclaim a nonzero stale reservation merely because PID lookup failed.
   if z.get('reserved_bytes',0)>0:
    entries.append(dict(source=str(f),owner=o or sup,tree=[],reserved_bytes=z['reserved_bytes'],pss_credit=0,unused_growth=z['reserved_bytes'],status='STALE_RESERVATION_WAIT_IDENTITY_AND_PARENT_WAIT_REQUIRED'))
   continue
  fam=ctrl.descendants(root,ps);fresh=[p for p in fam if p['pid'] not in seen];seen.update(p['pid'] for p in fresh)
  pss=sum(p['pss'] or 0 for p in fresh)
  entries.append(dict(source=str(f),owner=root,tree=fresh,reserved_bytes=z['reserved_bytes'],pss_credit=pss,unused_growth=max(0,z['reserved_bytes']-pss)))
 for p in ps:
  if p['pid'] in seen:continue
  argv=' '.join(p['argv'])
  if any(n in argv for n in ['run_l08_dro_moment_ablation_parallel_r20.py','run_l08_dro_moment_ablation_rowgen_r21.py']):
   # Only the executed entry script, not parent wrappers.
   scripts=[x for x in p['argv'] if x.endswith('.py')]
   if not scripts or Path(scripts[0]).name not in ['run_l08_dro_moment_ablation_parallel_r20.py','run_l08_dro_moment_ablation_rowgen_r21.py']:continue
   fam=[q for q in ctrl.descendants(p,ps) if q['pid'] not in seen];seen.update(q['pid'] for q in fam)
   reserve=12*1024**3;pss=sum(q['pss'] or 0 for q in fam)
   entries.append(dict(source='existing external diagnostic 12 GiB recorded limit/reservation',owner=p,tree=fam,reserved_bytes=reserve,pss_credit=pss,unused_growth=max(0,reserve-pss)))
 mem=ctrl.memory_base();growth=sum(e['unused_growth'] for e in entries)
 return dict(epoch=time.time(),**mem,entries=entries,unused_growth=growth,available_for_new_bytes=max(0,mem['half_available_bytes']-growth),signed_headroom_bytes=mem['half_available_bytes']-growth,formula='new reservation + sum(max(0, old reservation - unique verified tree PSS)) <= half effective available; no old reservation reduced')
@contextlib.contextmanager
def admission(label,need):
 with ctrl.lock(V/'DISPATCH.lock'):
  z=read(LEASE)
  if ctrl.same(z.get('owner')) and z['owner']['pid']!=os.getpid():raise RuntimeError('DUPLICATE_R23_OWNER')
  res=resources();ok=not res['unknown_scheduler_limits'] and res['available_for_new_bytes']>=need
  rec=dict(label=label,requested_bytes=need,admitted=ok,resource=res,epoch=time.time())
  dump(R/'08_SCOPE'/('ADMISSION_'+str(time.time_ns())+'.json'),rec)
  if ok:
   dump(LEASE,dict(layout='L08',method='R23_CONTINUOUS_DIAGNOSTIC_ONLY',owner=ctrl.proc(os.getpid()),supervisor=None,reserved_bytes=need,threads=1,procurement_MIP=False))
 if not ok:
  dump(R/'01_REPORTS/RESOURCE_WAIT.json',rec);yield False;return
 try:yield True
 finally:
  with ctrl.lock(V/'DISPATCH.lock'):
   z=read(LEASE)
   if (z.get('owner') or {}).get('pid')==os.getpid():
    dump(LEASE,dict(layout='L08',method='R23_CONTINUOUS_DIAGNOSTIC_ONLY',owner=None,supervisor=None,reserved_bytes=0,released=time.time()))
def source_inventory():
 roots=[OLD/'tools',B/'L08_ARO400_R15_OCCUPANCY_COST_R16/tools',B/'L08_ARO400_R15_OCCUPANCY_COST_R16/continuations/RESUME_20261029_MEMORY48_V1/tools',B/'L08_DRO_BATCH_ROOT_PROGRESS_R20/tools',B/'L08_DRO400_R15_PARALLEL_R18/tools']
 return {str(p):sha(p) for d in roots for p in sorted(d.glob('*.py'))}
def inventory():
 out=R/'08_SCOPE/BASELINE.json'
 if not out.exists():dump(out,dict(epoch=time.time(),sources=source_inventory(),resources=resources()))
 base=read(out);current=source_inventory();changes=[p for p,h in base['sources'].items() if current.get(p)!=h]
 res=resources();observed=[]
 for e in res['entries']:
  if any(n in e['source'] for n in ['R16_RESUME','R18_','R22_']):
   o=ctrl.proc(e['owner']['pid']) if e.get('owner') else None
   if o and ctrl.same(e['owner']):observed.append(dict(owner=o,source=e['source']))
 x=dict(epoch=time.time(),source_changes=changes,protected_owners=observed,resources=res,signals_sent=0,old_case_writes=0)
 dump(R/'08_SCOPE/CURRENT.json',x);assert not changes,changes
 return x
def freeze():
 target=R/'04_FORMULATION/S0.json'
 if target.exists():
  x=read(target);assert sha(x['split_model']['path'])==x['split_model']['sha256'];return x
 inventory();z=read(OLD/'checkpoint/INITIAL_CONTEXT.json');p=Path(z['model']['path']);assert sha(p)==z['model']['sha256']
 dest=R/'04_FORMULATION/S0_SPLIT.mps.gz';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 assert sha(dest)==z['model']['sha256']
 for name in ['IDENTITIES.json','RESOURCE_PROFILE.json']:
  src=OLD/'02_SOURCE_AND_CONFIG'/name;shutil.copy2(src,dest.parent/name)
 cp=OLD/'04_ORIGINAL_P400/COMPRESSION.npz';shutil.copy2(cp,dest.parent/'COMPRESSION.npz')
 for src in sorted((OLD/'tools').glob('*.py')):
  d=R/'02_SOURCE/original_r22'/src.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,d)
 out=dict(epoch=time.time(),context=z,split_model=record(dest),scientific_identity=read(dest.parent/'IDENTITIES.json'),pool=z['pool'],zero=z['zero'],source_model=record(p),S0_frozen=True,branch_tree_restored=False,completed_same_identity_split_LP_baseline=None)
 dump(target,out);return out
def moments():
 import numpy as np
 z=np.load(R/'04_FORMULATION/COMPRESSION.npz')
 p=B/'DRO_ORIGINAL_BOUNDS_SUPPORT_EXPANSION_V1/20261027T043033Z/05_FEATURES'
 x=np.load(p/'TRAIN400_FEATURES.npy',mmap_mode='r')
 phi=np.array(x[:,z['representative_raw_ids']].T,order='C')
 lo,hi=z['lower'],z['upper'];scale=np.maximum.reduce([np.max(abs(phi),axis=1),abs(lo),abs(hi),np.full(len(lo),1e-3)])
 return phi,lo,hi,scale,z
