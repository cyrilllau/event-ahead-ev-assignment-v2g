from r30_control import *
import argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 stages=['inputs','routes','domain','replay400'] if args.stage=='data' else [args.stage]
 with (R/'RUN.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for stage in stages:
   done={'inputs':'01_REPORTS/PREFLIGHT_COMPLETE.json','routes':'03_FORECAST44_ROUTES/COMPLETE.json','domain':'04_R10_COMPACT_DOMAIN/COMPLETE.json','replay400':'05_SUPPORT400_ROUTE_REPLAY/COMPLETE.json'}[stage]
   if (R/done).exists():
    assert args.resume and read(R/done)['passed'];print('ALREADY_COMPLETE',stage,flush=True);continue
   dump(R/'01_REPORTS/STATUS.json',dict(stage=stage,state='STARTING',epoch=time.time()))
   with admission(stage,8):
    if stage=='inputs':inputs()
    else:
     import importlib
     importlib.import_module({'routes':'r30_routes','domain':'r30_domain','replay400':'r30_replay'}[stage]).build()
   dump(R/'01_REPORTS/STATUS.json',dict(stage=stage,state='COMPLETE',epoch=time.time()))
  inventory('DATA_END')
if __name__=='__main__':
 try:main()
 except Exception:
  dump(R/'01_REPORTS/FAILURE_'+str(time.time_ns())+'.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
