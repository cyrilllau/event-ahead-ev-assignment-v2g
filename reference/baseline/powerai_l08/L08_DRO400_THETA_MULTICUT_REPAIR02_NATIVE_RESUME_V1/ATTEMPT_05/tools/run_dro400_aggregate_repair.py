from pathlib import Path
import argparse,sys,os,json,time,subprocess,hashlib,traceback,shutil

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-root',required=True);p.add_argument('--source-root',required=True);p.add_argument('--config',required=True);p.add_argument('--start-or-resume',action='store_true');p.add_argument('--mode',choices=['prepare','test','execute','status']);p.add_argument('--test-child',action='store_true');a=p.parse_args()
 root=Path(a.execution_root).resolve();source=Path(a.source_root).resolve();config=json.loads(Path(a.config).read_text());run=json.loads((root/'RUN.json').read_text())
 if a.mode=='status':
  for n in ['STATUS.json','GAP.json','FINAL_RESULT.json']:
   if (root/n).exists():print(n,(root/n).read_text())
  return
 tools=Path(config['execution_root'])/'tools';sys.path[:0]=[str(tools),str(tools/'lifecycle')]
 import aggregate_core as core
 ctx=core.ExecutionContext(root,source,config,run,deadline=min(run['deadline_monotonic']-120,time.monotonic()+1200) if a.test_child else None);core.bind_context(ctx)
 from control_policy import drive
 from source_prepare import prepare
 def tests():
  from resume_validation import validate_resume
  validate_resume()
 def execute():
  r=core.read(root/'TEST_ACCEPTANCE.json');assert r['passed'];core.rt.verify_ref(r['result'])
  import formal
  return formal.execute()
 try:
  if a.test_child:
   import actual_tests;actual_tests.tests();return
  if a.start_or_resume:drive(prepare,tests,execute)
  elif a.mode=='prepare':prepare()
  elif a.mode=='test':tests()
  elif a.mode=='execute':execute()
  else:p.error('select --start-or-resume or --mode')
 except BaseException as exc:
  core.write(root/'ENTRY_FAILURE.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),epoch=time.time()))
  if not (root/'FINAL_RESULT.json').exists():core.write(root/'FINAL_RESULT.json',dict(status='PREPARATION_BLOCKED',formal_execute_called=(root/'FORMAL_EXECUTE_BEGIN.json').exists(),error=str(exc),epoch=time.time()))
  raise

if __name__=='__main__':main()
