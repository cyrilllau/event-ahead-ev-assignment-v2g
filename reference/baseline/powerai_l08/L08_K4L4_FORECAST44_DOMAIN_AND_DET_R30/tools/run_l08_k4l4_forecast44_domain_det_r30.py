from r30_control import *
import argparse,importlib
STAGES=['inputs','routes','domain','replay400','reachability','tests','build-det','solve-det','audit-det','evaluate-fixed400','report','package']
DONE={'inputs':'01_REPORTS/PREFLIGHT_COMPLETE.json','routes':'03_FORECAST44_ROUTES/COMPLETE.json','domain':'04_R10_COMPACT_DOMAIN/COMPLETE.json','replay400':'05_SUPPORT400_ROUTE_REPLAY/COMPLETE.json','reachability':'06_REACHABILITY/COMPLETE.json','tests':'06_TESTS/COMPLETE.json','build-det':'04_MODEL_AND_NATIVE_SOLVE/BUILD_COMPLETE.json','solve-det':'04_MODEL_AND_NATIVE_SOLVE/SOLVE_COMPLETE.json','audit-det':'05_SELECTED_PLAN_AND_CERTIFICATES/AUDIT_COMPLETE.json','evaluate-fixed400':'07_FIXED400/COMPLETE.json','report':'09_DELIVERY_REPORT/COMPLETE.json','package':'delivery/PACKAGE_INDEX.json'}
FUN={'routes':('r30_routes','build'),'domain':('r30_domain','build'),'replay400':('r30_replay','build'),'reachability':('r30_reachability','reachability'),'tests':('r30_tests','tests'),'build-det':('r30_native','build_det'),'solve-det':('r30_native','solve_det'),'audit-det':('r30_audit','audit'),'evaluate-fixed400':('r30_fixed400','fixed400'),'report':('r30_report','report'),'package':('r30_report','package')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=STAGES+['all','data','preflight','status']);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 if args.stage=='status':
  print(json.dumps({s:read(R/p) if (R/p).exists() else None for s,p in DONE.items()},default=str));return
 if args.stage=='preflight':inventory('PREFLIGHT_READONLY');return
 stages=STAGES if args.stage=='all' else STAGES[:4] if args.stage=='data' else [args.stage]
 with (R/'RUN.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for stage in stages:
   target=R/DONE[stage]
   if target.exists() and stage not in ['report','package']:
    assert args.resume and read(target).get('passed',False),('Prior incomplete or unaccepted result',stage)
    if stage=='audit-det':assert read(target)['absolute_gap_met'],'DET_NOT_CERTIFIED'
    print('ALREADY_COMPLETE',stage,flush=True);continue
   dump(R/'01_REPORTS/STATUS.json',dict(stage=stage,state='STARTING',epoch=time.time()))
   if stage in ['report','package']:
    mod,fn=FUN[stage];getattr(importlib.import_module(mod),fn)()
   else:
    with admission(stage,12 if stage in ['tests','build-det','solve-det','audit-det','evaluate-fixed400'] else 8):
     if stage=='inputs':inputs()
     else:
      mod,fn=FUN[stage];getattr(importlib.import_module(mod),fn)()
   if stage=='audit-det':assert read(target)['absolute_gap_met'],'DET_NOT_CERTIFIED_GAP'
   dump(R/'01_REPORTS/STATUS.json',dict(stage=stage,state='COMPLETE',epoch=time.time()))
  inventory('RUN_END_'+str(time.time_ns()))
if __name__=='__main__':
 try:main()
 except Exception:
  p=R/'01_REPORTS'/('FAILURE_'+str(time.time_ns())+'.json');dump(p,dict(error=traceback.format_exc(),epoch=time.time()));raise
