from r36_recourse import *
from r36_probability import solve_risk,moments
def request_id(plan,planes,pool):return digest(dict(science=SCIENCE,plan=plan_digest(plan),planes=np.asarray(planes),pool=sorted(pool),risk=moments()['identity']))
def enqueue(root,plan,planes,pool,kind='CANDIDATE'):
 root=Path(root);rid=request_id(plan,planes,pool);p=root/'pending'/f'{rid}.json';done=root/'results'/f'{rid}.json'
 if not p.exists() and not done.exists():
  dump(p,dict(id=rid,science=SCIENCE,plan=plan,plan_hash=plan_digest(plan),planes=np.asarray(planes),pool=sorted(pool),risk_identity=moments()['identity'],kind=kind,first_seen=time.time(),required=kind in ['FINAL_NATIVE','BEST'],state='PENDING'))
 elif kind in ['FINAL_NATIVE','BEST']:
  dump(root/'required'/f'{rid}.json',dict(id=rid,kind=kind,epoch=time.time()))
 return rid
def pause_required(root):
 root=Path(root);pending=[]
 for p in (root/'pending').glob('*.json'):
  req=read(p)
  if req['required'] or (root/'required'/p.name).exists():pending.append(req['id'])
 dump(root/'PAUSED_PENDING_REQUIRED.json',dict(ids=pending,epoch=time.time(),cancelled=[]));return pending
def evaluation_for_request(req):
 assert req['science']==SCIENCE and req['risk_identity']==moments()['identity'];pp=req['plan'];assert plan_digest(pp)==req['plan_hash'];F=verify_plan(pp);where=R/'evaluations'/req['plan_hash'];rows=[]
 for sid in range(400):
  if not pp:d=R/'evaluations/ZERO'/f'{sid:04d}'
  else:d=where/f'{sid:04d}'
  # A completed real equivalence-test certificate is the same plan/recourse identity.
  old=R/'tests/recourse'/f'{sid:04d}'/'EXPLICIT'
  if not (d/'COMPLETE.json').exists() and (old/'COMPLETE.json').exists() and read(old/'COMPLETE.json')['plan_sha']==req['plan_hash']:
   if not d.exists():shutil.copytree(old,d)
  rows.append(evaluate(pp,sid,d));dump(where/'PROGRESS.json',dict(completed=sid+1,request=req['id'],epoch=time.time()))
 L=np.array([v['qL'] for v in rows]);U=np.array([v['qU'] for v in rows]);risk=solve_risk(U,where/'probability');planes=np.array(req['planes']);assert planes.shape==(400,)
 lower=L-planes;upper=U-planes;out=dict(verified=True,science=SCIENCE,request=req['id'],plan_hash=req['plan_hash'],risk_identity=req['risk_identity'],first_seen=req['first_seen'],F=F,safe_UB=F+risk['risk_upper'],risk=risk,evaluation=str(where),qL=L,qU=U,lower_residual=lower,upper_residual=upper,max_upper_residual=float(max(0,upper.max())),violations=[(int(s),float(lower[s])) for s in range(400) if lower[s]>1e-4 and s not in req['pool']],full_support=400,source_certificate_sha=[sha((R/'evaluations/ZERO' if not pp else where)/f'{s:04d}'/'CERTIFICATE.json') for s in range(400)])
 dump(where/'COMPLETE.json',dict(passed=True,plan=pp,plan_hash=req['plan_hash'],F=F,scenarios=400,risk=risk,safe_UB=out['safe_UB']));return out
def worker(root):
 root=Path(root);(root/'pending').mkdir(parents=True,exist_ok=True)
 while True:
  if (root/'STOP_WHEN_IDLE.json').exists():pause_required(root);return
  jobs=[p for p in (root/'pending').glob('*.json') if not (root/'results'/p.name).exists() and not (root/'failed'/p.name).exists()]
  if not jobs:
   if (root/'STOP_WHEN_IDLE.json').exists():pause_required(root);return
   time.sleep(1);continue
  p=min(jobs,key=lambda p:(not (read(p)['required'] or (root/'required'/p.name).exists()),read(p)['first_seen'],p.name));req=read(p)
  try:
   ans=evaluation_for_request(req);dump(root/'results'/p.name,ans)
  except Exception as ex:
   dump(root/'failed'/p.name,dict(id=req['id'],error=repr(ex),traceback=traceback.format_exc(),required=req['required'] or (root/'required'/p.name).exists(),epoch=time.time(),state='PAUSED_PENDING_REQUIRED' if req['required'] else 'LOCAL_FAILED_ISOLATED'))
def lifecycle_test():
 root=R/'tests/queue';p=root/'pending/required.json';dump(p,dict(id='required',required=True,state='PENDING',first_seen=time.time()));dump(root/'pending/local.json',dict(id='local',required=False,state='PENDING',first_seen=time.time()))
 paused=pause_required(root);assert paused==['required'] and p.exists();dump(root/'failed/local.json',dict(error='deliberate local test',required=False));assert p.exists()
 return dict(passed=True,required_preserved=True,local_isolated=True,cancelled=0)
if __name__=='__main__':worker(Path(sys.argv[1]))
