"""Clock, ownership, resource admission, and persisted bounded recovery."""
from pathlib import Path
import os,sys,time,json,uuid,fcntl,subprocess,hashlib
import durable as d
CONTROL=Path('/home/Shixin/ev-assignment/R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2')
sys.path.insert(0,str(CONTROL/'tools'))
import control_core as cc

def proc(pid):return cc.proc(int(pid))
def wait_process(pid,expected=None,seconds=3):
 end=time.monotonic()+seconds;last=None
 while time.monotonic()<end:
  current=proc(pid)
  if current and current.get('argv') and (expected is None or expected in current['argv']):
   if last and all(last.get(k)==current.get(k) for k in ('pid','start_ticks','boot_id','argv')):return current
   last=current
  time.sleep(.02)
 return last
def same(a,b):
 return bool(a and b and a.get('pid')==b.get('pid') and a.get('start_ticks')==b.get('start_ticks') and a.get('boot_id')==b.get('boot_id') and a.get('argv')==b.get('argv') and b.get('state')!='Z')
def deadline(run,now_mono=None,now_utc=None,boot=None):
 now_mono=time.monotonic() if now_mono is None else now_mono;now_utc=time.time() if now_utc is None else now_utc
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if boot is None else boot
 if boot==run['boot_id']:return run['deadline_monotonic']
 if now_utc<run['start_utc'] or now_utc+2<run.get('last_observed_utc',run['start_utc']):raise RuntimeError('CLOCK_TRUST_UNCERTAIN')
 return now_mono+max(0,run['deadline_utc']-now_utc)
def create_run(out,cfg):
 p=out/'RUN.json'
 if p.exists():
  run=d.read(p)
  if run['config_sha']!=d.sha(out/'CONFIG.json') or run['authorization_sha']!=d.sha(out/'AUTHORIZATION.json'):raise ValueError('OUTPUT_IDENTITY_CONFLICT')
  return run
 if any(out.glob('ATTEMPT_*')) or (out/'LAST_COMMITTED.json').exists():raise ValueError('OUTPUT_IDENTITY_CONFLICT')
 t=time.time();m=time.monotonic();run={'run_id':'R44_AUTO_'+uuid.uuid4().hex,'start_utc':t,'deadline_utc':t+cfg['global_wall_limit_s'],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_monotonic':m,'deadline_monotonic':m+cfg['global_wall_limit_s'],'config_sha':d.sha(out/'CONFIG.json'),'authorization_sha':d.sha(out/'AUTHORIZATION.json'),'automatic_boot_service':'not installed; restart replay supported, automatic boot launch not guaranteed'}
 d.atomic(p,run,True);d.atomic(out/'BEGIN.json',dict(run,stage='BEFORE_RECOVERY_AND_PREFLIGHT'),True);return run
def retry_allowed(out,task,signature,limit,whitelisted):
 p=out/'RETRIES.json';counts=d.read(p) if p.exists() else {};key=d.digest([task,signature]);n=counts.get(key,0)
 if not whitelisted or n>=limit:return False,n
 counts[key]=n+1;d.atomic(p,counts);return True,n+1
def append_event(path,event):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('ab') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(d.encoded(event));f.flush();os.fsync(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)
def resource_inventory():
 mem=cc.memory_base();reservations=[];seen={}
 for p in (CONTROL/'leases').glob('*.json'):
  try:
   r=d.read(p);owner=r.get('owner') or r.get('supervisor');live=proc(owner['pid']) if owner else None
   if not same(owner,live):continue
   amount=int(r.get('reserved_bytes',r.get('reservation_bytes',r.get('memory_reserved_bytes',0))))
   key=(owner['pid'],str(owner.get('start_ticks')));seen[key]=max(seen.get(key,0),amount);reservations.append({'file':str(p),'bytes':amount,'owner':owner})
  except (OSError,ValueError,KeyError,TypeError):continue
 reserved=sum(seen.values());avail=int(mem.get('half_available_bytes',0));return {'memory':mem,'reservations':reservations,'full_reserved_bytes':reserved,'conservative_remaining_bytes':avail-reserved,'policy':'existing half-effective-available rule, conservatively subtract full live reservations without PSS credit'}
def acquire(out,run,production,renew=None):
 # Use the existing sole dispatcher lock, without dispatching another case.
 with cc.lock(CONTROL/'DISPATCH.lock'):
  current=proc(production['process']['pid'])
  if current is not None and not same(production['process'],current):raise ValueError('PRODUCTION_IDENTITY_CHANGED')
  if d.sha(production['model']['path'])!=production['model']['sha256']:raise ValueError('PRODUCTION_MODEL_CHANGED')
  inv=resource_inventory();need=24*1024**3
  if renew:
   actual=d.read(renew['path'])
   if actual['lease_id']!=renew['lease_id'] or not same(actual['owner'],proc(os.getpid())):raise ValueError('OWN_LEASE_IDENTITY_CONFLICT')
   inv['conservative_remaining_bytes']+=need
  if inv['conservative_remaining_bytes']<need:return None,inv
  lease={'lease_id':renew['lease_id'] if renew else uuid.uuid4().hex,'run_id':run['run_id'],'owner':proc(os.getpid()),'layout':'L08','method':'DRO_ROOT_R44_AUTONOMOUS','reserved_bytes':need,'acquired_epoch':time.time(),'expires_epoch':time.time()+7200,'production_signals_sent':[]}
  path=Path(renew['path']) if renew else CONTROL/'leases'/('R44_AUTO_'+lease['lease_id']+'.json')
  d.atomic(path,lease);ref=dict(path=str(path),lease_id=lease['lease_id']);d.atomic(out/'RESOURCE'/f'{time.time_ns()}_ADMISSION.json',{'lease':lease,'inventory':inv,'renewal':renew is not None},True);return ref,inv
def release(out,ref):
 if not ref:return
 with cc.lock(CONTROL/'DISPATCH.lock'):
  p=Path(ref['path']);r=d.read(p)
  if r['lease_id']!=ref['lease_id'] or not same(r['owner'],proc(os.getpid())):raise ValueError('LEASE_RELEASE_OWNER_MISMATCH')
  d.atomic(out/'RESOURCE'/f'{time.time_ns()}_LEASE_RELEASED.json',{'lease':r,'actual_release':True,'epoch':time.time()},True);p.unlink();d.syncdir(p.parent)
def verify_ref(ref):
 if d.sha(ref['path'])!=ref['sha256']:raise ValueError('REFERENCED_OBJECT_SHA_FAILURE '+ref['path'])
 return Path(ref['path'])
def ref(path):return {'path':str(path),'sha256':d.sha(path),'bytes':Path(path).stat().st_size}
