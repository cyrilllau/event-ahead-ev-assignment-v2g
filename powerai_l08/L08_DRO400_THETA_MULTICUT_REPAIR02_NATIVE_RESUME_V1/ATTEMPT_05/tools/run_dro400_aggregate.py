"""Unified isolated entry; old launchers are never called."""
import argparse,os
from aggregate_core import *
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--start-or-resume',action='store_true');p.add_argument('--status',action='store_true');args=p.parse_args(argv)
 cfg=read(args.config);assert cfg['science']==SCIENCE and cfg['run_id']==CTX.run['run_id'];assert cfg['deadline_monotonic']==CTX.deadline
 if args.status:
  print(json.dumps(read(OUT/'STATUS.json') if (OUT/'STATUS.json').exists() else read(OUT/'RUN.json'),indent=2));return
 if not args.start_or_resume:raise ValueError('EXPLICIT_START_REQUIRED')
 if os.environ.get('AGG_ATTEMPT') not in ('ATTEMPT_003',):raise ValueError('USE_VERIFIED_SINGLE_PARENT; NO_COMPETING_OWNER')
 from formal import execute
 execute()
if __name__=='__main__':main()
