from r3_common import *
import contextlib,fcntl,shutil,subprocess,traceback,zipfile
V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
sys.path.insert(0,str(V/'tools'));import control_core as ctrl
sys.path.insert(0,str(B/'L08_MARGINAL_DRO_EXACT_ACCELERATION_R23/tools'));import r23_common as resource_reader
sys.path.insert(0,str(R/'tools'))
LEASE=V/'leases/R36_K4L4_FULL_ST.json';resource_reader.LEASE=LEASE
SCIENCE='L08_K4L4_FULL_ST_DRO400_R36'
OLD=B/'L08_K4L4_FORECAST44_DOMAIN_AND_DET_R30'
def npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 os.replace(tmp,path)
def protected():
 req=read(B/'USER_STOP_EXCEPT_CURRENT_DET_20261101/REQUEST.json')
 for p,h in req['det_source_hashes'].items():assert sha(Path(p))==h,p
 return dict(epoch=time.time(),det=ctrl.proc(req['det']['pid']),det_identity_matches=ctrl.same(req['det']),signals_sent=[],old_sources_unchanged=True)
@contextlib.contextmanager
def admission(stage,gib=8,threads=1):
 with ctrl.lock(V/'DISPATCH.lock'):
  prev=ctrl.read(LEASE);assert not ctrl.same(prev.get('owner')),'DUPLICATE_R36_OWNER'
  rs=resource_reader.resources();ok=not rs['unknown_scheduler_limits'] and rs['signed_headroom_bytes']>=gib*1024**3
  rec=dict(stage=stage,epoch=time.time(),request_GiB=gib,resources=rs,admitted=ok)
  dump(R/'operations'/f'ADMISSION_{time.time_ns()}.json',rec)
  if not ok:dump(R/'RESOURCE_BLOCKED.json',rec);raise RuntimeError('PREPARED_RESOURCE_BLOCKED')
  ctrl.dump(LEASE,dict(owner=ctrl.proc(os.getpid()),reserved_bytes=gib*1024**3,threads=threads,method=SCIENCE,stage=stage,root=str(R)))
 try:yield
 finally:
  with ctrl.lock(V/'DISPATCH.lock'):
   q=ctrl.read(LEASE)
   if q.get('owner',{}).get('pid')==os.getpid():ctrl.dump(LEASE,dict(owner=None,reserved_bytes=0,released_epoch=time.time(),stage=stage))
def committed(path,fn):
 p=Path(path)
 if p.exists():
  rec=read(p);assert rec['passed'];return rec
 rec=fn();assert rec['passed'];dump(p,rec);return rec
