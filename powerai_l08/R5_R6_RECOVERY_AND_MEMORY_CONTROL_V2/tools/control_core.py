"""V2 control only. No solver signals, no reset, no mathematical edits."""
from pathlib import Path
import os, sys, json, time, hashlib, fcntl, contextlib, subprocess, math, re, uuid
B=Path('/home/Shixin/ev-assignment')
V=B/'R5_R6_RECOVERY_AND_MEMORY_CONTROL_V2'
R6=B/'SECTIONIII_TRAIN400_ORIGINAL_DRO_R6'
R5=B/'SECTIONIII_MULTISCENARIO_R5'
S=R5/'staging/UNLIMITED_V1'
ORDER=[(l,m) for m in ['ARO','Proposed','SAA'] for l in ['L08','L13']]
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def read(p,default=None):
 p=Path(p)
 if not p.exists():return {} if default is None else default
 return json.loads(p.read_text())
def dump(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
 with t.open('x') as f:json.dump(a,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(4*1024*1024),b''):h.update(a)
 return h.hexdigest()
def digest(a):return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def finite(x):
 try:x=float(x);return x if math.isfinite(x) and abs(x)<1e99 else None
 except (TypeError,ValueError):return None
def proc(pid):
 try:
  p=Path('/proc')/str(pid);raw=(p/'stat').read_text();st=[str(pid),raw[raw.index('('):raw.rindex(')')+1]]+raw[raw.rindex(')')+2:].split();a=(p/'cmdline').read_bytes().split(b'\0');argv=[x.decode(errors='replace') for x in a if x]
  status={l.split(':',1)[0]:l.split(':',1)[1].strip() for l in (p/'status').read_text().splitlines() if ':' in l}
  rss=int(status.get('VmRSS','0 kB').split()[0])*1024;pss=None;error=None
  try:pss=int(next(l.split()[1] for l in (p/'smaps_rollup').read_text().splitlines() if l.startswith('Pss:')))*1024
  except (OSError,StopIteration) as ex:error=type(ex).__name__
  return dict(pid=int(pid),start_ticks=st[21],boot_id=BOOT,ppid=int(st[3]),state=st[2],argv=argv,cwd=os.readlink(p/'cwd'),rss=rss,pss=pss,pss_error=error,peak_rss=int(status.get('VmHWM','0 kB').split()[0])*1024,cpu_ticks=int(st[13])+int(st[14]))
 except (OSError,ProcessLookupError):return None
def same(a):
 if not a:return False
 p=proc(a['pid'])
 return bool(p and p['state']!='Z' and all(p[k]==a[k] for k in ['start_ticks','boot_id','argv']))
@contextlib.contextmanager
def lock(p,nonblock=False):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblock else 0));yield f
def olddir(l,m):return R6/'06_SAA_ARO_PROPOSED_PLANS_AND_CERTIFICATES'/l/m
def instance_key(l,m):return read(V/'BASELINE.json')['r6_identities'][l+'_'+m]
def allproc():
 return [q for p in Path('/proc').iterdir() if p.name.isdigit() for q in [proc(int(p.name))] if q and q['state']!='Z']
def worker_pair(p):
 a=p['argv']
 # Only the executed script is an entry point. run_logged.py carries the optimizer
 # filename as an argument; interpreting that argument would count parent and child twice.
 script_index=next((i for i,x in enumerate(a) if x.endswith('.py')),None)
 if script_index is None:return None
 for name,run in [('r5_resume_unlimited.py','R5'),('r5_optimize.py','R6'),('r6_continuous_entry.py','R6'),('v2_worker.py','R6')]:
  for i,arg in [(script_index,a[script_index])]:
   if Path(arg).name!=name:continue
   tail=a[i+1:]
   if name=='r5_resume_unlimited.py':
    if not tail or tail[0]!='run':return None
    tail=tail[1:]
   if len(tail)>=2 and tail[0] in ['L08','L13']:
    return run,tail[0],tail[1]
 return None
def descendants(p,ps):
 ids={p['pid']}
 while True:
  nxt=ids|{x['pid'] for x in ps if x['ppid'] in ids}
  if nxt==ids:return [x for x in ps if x['pid'] in ids]
  ids=nxt
def memory_base():
 h=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 limits=[];cg=Path('/sys/fs/cgroup');rel=Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/');p=cg/rel
 while p==cg or cg in p.parents:
  if (p/'memory.max').exists():
   t=(p/'memory.max').read_text().strip();cur=int((p/'memory.current').read_text())
   limits.append(dict(path=str(p),max_bytes=None if t=='max' else int(t),current_bytes=cur,remaining=None if t=='max' else max(0,int(t)-cur)))
  if p==cg:break
  p=p.parent
 scheduler={k:v for k,v in os.environ.items() if k.startswith(('SLURM_MEM','PBS_RESOURCE','LSB_JOB_MEM'))}
 hard=[];unknown=[]
 for k,val in scheduler.items():
  if k=='SLURM_MEM_PER_NODE' and val.isdigit():hard.append(int(val)*1024**2)
  elif k=='SLURM_MEM_PER_CPU' and val.isdigit():hard.append(int(val)*1024**2*len(os.sched_getaffinity(0)))
  else:unknown.append(k)
 # Scheduler hard limit is conservatively reduced by RSS of all accessible same-UID processes.
 uid_rss=sum(p['rss'] for p in allproc()) if hard else 0
 effective=min([h]+[x['remaining'] for x in limits if x['remaining'] is not None]+[max(0,x-uid_rss) for x in hard])
 return dict(host_available_bytes=h,cgroup=limits,scheduler_environment=scheduler,scheduler_limit_bytes=hard,scheduler_accounted_rss=uid_rss,unknown_scheduler_limits=unknown,effective_available_bytes=effective,half_available_bytes=effective//2)
def peak_reservation(run,l,m,pool):
 if run=='R5':
  hb=read(S/'HEARTBEAT.json');a=hb.get('active',{}).get(l+'_'+m)
  if a:return int(a['reserved_bytes'])
 e=read((R5 if run=='R5' else R6)/'01_REPORTS/RESOURCE_ASSESSMENT.json')['layouts'][l]
 return int(4*(e['fixed_estimate']+e['slope_bytes_per_copy']*max(1,pool))+e.get('extra_margin_bytes',6*1024**3))
def ledger(ps=None):
 ps=allproc() if ps is None else ps;mem=memory_base();entries=[];seen=set()
 for p in ps:
  pair=worker_pair(p)
  if not pair:continue
  run,l,m=pair;lease=read(V/'leases'/(instance_key(l,m)+'.json')) if run=='R6' else {}
  cp=read((S/'instances'/l/m if run=='R5' else olddir(l,m))/'CHECKPOINT.json')
  if lease.get('owner') and same(lease['owner']) and lease['owner']['pid']==p['pid']:reserve=lease['reserved_bytes']
  else:reserve=peak_reservation(run,l,m,len(cp.get('pool',[])))
  family=[q for q in descendants(p,ps) if q['pid'] not in seen];seen.update(q['pid'] for q in family)
  # PSS removes shared-page double count. Missing PSS=>0 credit, conservative growth.
  pss=sum(q['pss'] or 0 for q in family);rss=sum(q['rss'] for q in family)
  entries.append(dict(run=run,layout=l,method=m,owner=p,process_tree_pids=[q['pid'] for q in family],tree_rss=rss,tree_pss=pss,reserved_peak_bytes=reserve,unused_growth_bytes=max(0,reserve-pss),python_matrix_bytes=None,solver_memory_bytes=None,unavailable_reason='Live legacy process exposes no Gurobi/Python memory API; RSS/PSS are not solver allocation'))
 # A pending supervisor/lease is counted even before worker registration.
 for f in (V/'leases').glob('*.json'):
  a=read(f);sup=a.get('supervisor');o=a.get('owner')
  if (same(sup) or same(o)) and not any(e['owner']['pid']==(o or {}).get('pid') for e in entries):
   entries.append(dict(run='R6',layout=a['layout'],method=a['method'],owner=sup or o,process_tree_pids=[],tree_rss=0,tree_pss=0,reserved_peak_bytes=a['reserved_bytes'],unused_growth_bytes=a['reserved_bytes'],pending_registration=True))
 growth=sum(e['unused_growth_bytes'] for e in entries)
 return dict(**mem,entries=entries,unused_growth_total=growth,available_for_new_bytes=max(0,mem['half_available_bytes']-growth),formula='new_peak + sum(max(0, live_reserved_peak - unique_tree_PSS)) <= 0.5 * min(host,cgroup,scheduler available); used memory already deducted by OS; unavailable PSS gets no credit',old_R6_resource_claim=read(R6/'logs/RESOURCE_CLAIM.json'),duplicate_claim_policy='R6 parent claim is provenance only; matched child reservation counted once; no guessed next-job reservation')
def classify(*,alive=False,identity_ok=True,status=None,certificate=False,pending_audit=False,native_return=None):
 if not identity_ok:return 'BLOCKED_IDENTITY'
 if alive:return 'RUNNING_OBSERVED_LEGACY'
 if certificate:return 'CERTIFIED_GAP_LE_1_USD'
 if pending_audit:return 'NATIVE_FINISHED_PENDING_AUDIT'
 if native_return==17:return 'RETRYABLE_MEM_LIMIT'
 if status in ['DEFERRED_RESOURCE','RESOURCE_BLOCKED']:return 'DEFERRED_RESOURCE'
 if status in ['NUMERICAL_BLOCKED','BLOCKED_DATA','RESOURCE_BLOCKED_NEEDS_CAPACITY','BLOCKED_IDENTITY']:return status
 if status:return 'STALE_STATUS_NEEDS_AUDIT'
 return 'NOT_STARTED'
def retry_allowed(history,old,new):
 # A clock tick is not a changed condition. Both algorithm and memory changes are explicit.
 return len(history)<2 and (new.get('threads',1)<old.get('threads',1) or new.get('root_method')!=old.get('root_method') or new.get('solver_bytes',0)>old.get('solver_bytes',0)) and new!=old
def admission(resources,need,coordinated=True,new_live=False):
 if not coordinated:return False,'DEFERRED_UNCOORDINATED_CONTROL'
 if new_live:return False,'ONE_NEW_LARGE_MIP_ACTIVE'
 if resources.get('unknown_scheduler_limits'):return False,'UNPARSED_SCHEDULER_MEMORY_LIMIT'
 if need>resources['available_for_new_bytes']:return False,'DEFERRED_RESOURCE'
 return True,'ADMITTED'
def latest_dir(l,m):
 dirs=sorted((V/'attempts'/l/m).glob('*')) if (V/'attempts'/l/m).exists() else []
 return dirs[-1] if dirs else olddir(l,m)
def tail(p,size=18000):
 try:
  with Path(p).open('rb') as f:f.seek(max(0,Path(p).stat().st_size-size));return f.read().decode(errors='replace')
 except OSError:return ''
def cert_check(l,m,cp,ident):
 """Conservative certificate classification; never accepts a status string alone."""
 errors=[];b=cp.get('best') or {};lb=finite(cp.get('LB'));ub=finite(b.get('UB'));gap=None if lb is None or ub is None else ub-lb
 if cp.get('joint_identity')!=ident:errors.append('identity')
 if gap is None or not 0<=gap<=1:errors.append('signed_gap')
 e=read(Path(b.get('evaluation','/nonexistent'))/'COMPLETE.json');rows=e.get('rows',[])
 if len(rows)!=400 or sorted(x['sid'] for x in rows)!=list(range(400)) or not e.get('passed'):errors.append('full400_evaluation')
 if e.get('plan_hash')!=b.get('plan_hash'):errors.append('plan_binding')
 if not errors:
  # Verify each archived numerical certificate's bound and payload identity.
  for row in rows:
   d=Path(row['folder']);c=read(d/'CERTIFICATE.json')
   if c.get('qL')!=row['qL'] or c.get('qU')!=row['qU'] or c.get('primal_residual',1)>1e-7:errors.append('row_certificate');break
   if any(not (d/f).exists() or sha(d/f)!=c.get(k) for f,k in [('PRIMAL.npz','primal_sha'),('DUAL_AND_BOUNDS.npz','dual_sha')]):errors.append('row_payload_sha');break
  if m=='SAA':safe=e['F']+sum(e['qU'])/400
  elif m=='ARO':safe=e['F']+max(e['qU'])
  else:
   risk=read(Path(e['folder'])/'RISKS.json');p=risk.get('Proposed',{});safe=e['F']+p.get('upper',float('inf'))
   if risk.get('plan_hash')!=b['plan_hash'] or risk.get('ambiguity_hash')!=sha(R6/'05_ORIGINAL_MOMENTS_N400/EFFECTIVE.npz') or p.get('kind')!='corrected_full_support_dual' or p.get('residual_after_correction',1)>0 or not (Path(e['folder'])/'PROBABILITY_CERTIFICATE.npz').exists():errors.append('probability_dual')
  if not math.isfinite(safe) or abs(safe-ub)>1e-6:errors.append('safe_UB')
  if m!='SAA':
   hs=[h for h in cp.get('history',[]) if h.get('plan_hash')==b['plan_hash'] and (h.get('separation') or {}).get('plan_hash')==b['plan_hash']]
   if not hs or hs[-1]['separation'].get('max_positive_upper',1)>1e-4:errors.append('same_plan_separation')
  if not any(h.get('LB')==lb and (h.get('native') or {}).get('status') not in [3,4,5] for h in cp.get('history',[])):errors.append('LB_provenance')
 return dict(scientific_certified=not errors,missing=errors,signed_gap=gap,evaluated_count=len(rows),evidence=e.get('folder'))
def snapshot():
 ps=allproc();base=read(V/'BASELINE.json');rows=[];r5=[]
 for run in ['R6','R5']:
  pairs=ORDER if run=='R6' else [(l,m) for l in ['L08','L13'] for m in ['SAA','ARO','DRO-A-EXPERIMENTAL']]
  for l,m in pairs:
   d=latest_dir(l,m) if run=='R6' else S/'instances'/l/m;cp=read(d/'CHECKPOINT.json');status=read(d/'STATUS.json');b=cp.get('best') or status.get('best') or {};owners=[p for p in ps if worker_pair(p)==(run,l,m)]
   if run=='R6':
    ident=base['r6_identities'][l+'_'+m];ok=all(x.get('joint_identity',ident)==ident for x in [cp,status]);cert=cert_check(l,m,cp,ident) if cp.get('status')=='CERTIFIED_GAP_LE_1_USD' or status.get('status')=='CERTIFIED_GAP_LE_1_USD' else dict(scientific_certified=False,missing=['not_certified'],evaluated_count=0)
   else:ident=cp.get('joint_identity');ok=True;cert=dict(scientific_certified=False,missing=['R5_observation_only'],evaluated_count=0)
   ep=Path(b.get('evaluation','/nonexistent'));e=read(ep/'COMPLETE.json');evaluated=e.get('scenario_count',len(list(ep.glob('*/ROW.json'))) if ep.exists() else 0)
   pr=read(d/'NATIVE_PROGRESS.json');calls=sorted((d/'CONTINUOUS_NATIVE').glob('*/BEGIN.json'));begin=read(calls[-1]) if calls else {};end=read(calls[-1].parent/'END.json') if calls else {};history=cp.get('history',[]);nat=(history[-1].get('native') or {}) if history else {}
   log=tail(d/'solver.log');memlines=[x.strip() for x in log.splitlines() if 'memory limit' in x.lower() or 'barrier performed' in x.lower() or 'Root relaxation' in x][-8:]
   pend=read(d/'PENDING_AUDIT.json').get('pending',False);st=classify(alive=bool(owners),identity_ok=ok,status=status.get('status'),certificate=cert['scientific_certified'],pending_audit=pend,native_return=end.get('actual_status'))
   if len(owners)>1:st='BLOCKED_DUPLICATE_OWNER'
   if owners and any(Path(a).name=='v2_worker.py' for a in owners[0]['argv']):st='RUNNING_AUDIT' if read(d/'PHASE.json').get('phase')=='AUDIT' else 'RUNNING_SOLVER'
   lb=finite(cp.get('LB',status.get('LB')));ub=finite(b.get('UB'));gap=None if lb is None or ub is None else ub-lb
   params=read(d/'PARAMETERS.json').get('actual',begin.get('protected_parameters',{}));lastbound=None;prevlb=None;prevub=None
   for h in history:
    if (h.get('LB') is not None and (prevlb is None or h['LB']>prevlb)) or (h.get('best_UB') is not None and (prevub is None or h['best_UB']<prevub)):lastbound=dict(iteration=h.get('iteration'),LB=h.get('LB'),UB=h.get('best_UB'),epoch=h.get('epoch'),epoch_unknown=h.get('epoch') is None)
    prevlb=h.get('LB',prevlb);prevub=h.get('best_UB',prevub)
   row=dict(run=run,layout=l,method=m,N=400 if run=='R6' else 200,bounds_policy='ORIGINAL_UNMODIFIED' if run=='R6' else 'R5_METHOD_SPECIFIC',directory=str(d),scientific_identity=ident,process_alive=bool(owners),owner=owners,status=st,legacy_status=status.get('status'),phase=read(d/'PHASE.json').get('phase',pr.get('phase','LEGACY_LOG_ONLY' if owners else None)),native_LB=pr.get('native_bound'),native_UB=pr.get('native_incumbent'),native_latest_return=nat,native_call_begin=begin,native_call_end=end,audited_LB=lb,audited_UB=ub,signed_gap=gap,relative_gap=None if gap is None or ub is None else gap/max(1,abs(ub)),selected_EV=len(b.get('plan',[])) if b else None,contract_energy=sum(x['contract'] for x in b.get('plan',[])) if b else None,plan_hash=b.get('plan_hash'),pool_size=len(cp.get('pool',[])),evaluated_count=evaluated,threads=params.get('Threads'),root_method=params.get('Method'),parameters=params,parameters_provenance='new actual getter' if (d/'PARAMETERS.json').exists() else 'legacy BEGIN or unknown; no live parameter introspection',RSS=sum(p['rss'] for p in owners),peak_RSS=sum(p['peak_rss'] for p in owners),solver_memory=pr.get('solver_memory_GB'),SoftMemLimit_GB=params.get('SoftMemLimit'),last_progress=pr.get('observed_epoch',cp.get('epoch')),last_bound_improvement=lastbound,memory_event=memlines,resume_attempt=len(list((V/'attempts'/l/m).glob('*'))) if run=='R6' else None,certificate=cert,scientific_certified=cert['scientific_certified'],next_action='observe original owner; no intervention' if owners else 'priority admission or certificate audit',blocking_reason=None if owners else st,log_tail=log[-6000:])
   (rows if run=='R6' else r5).append(row)
 resources=ledger(ps);result=dict(epoch=time.time(),boot_id=BOOT,instances=rows,R5_observation=r5,resources=resources,controllers=[p for p in ps if any(Path(a).name in ['continuous_scheduler.py','run_sectioniii_train400_original_dro_r6.py','run_r5_r6_control_v2.py'] for a in p['argv'])],actual_all_completed=all(r['scientific_certified'] for r in rows),all_certified=all(r['scientific_certified'] for r in rows),all_attempts_accounted=all(not r['process_alive'] and r['status']!='NOT_STARTED' for r in rows),old_phase_markers={str(p):read(p) for p in (R6/'logs/phases').glob('optimize.json')},old_runner_status=read(R6/'logs/STATUS.json'))
 dump(V/'01_REPORTS/CURRENT.json',result);dump(V/'04_PROCESS_AND_RESOURCES/RESOURCE_LEDGER.json',resources)
 for row in rows:dump(V/'05_INSTANCE_PROGRESS'/row['layout']/(row['method']+'.json'),row)
 return result
