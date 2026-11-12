from pathlib import Path
import os,sys,json,time,shutil,hashlib,zipfile,ast,importlib.util,difflib,copy
ROOT=Path(os.environ['ARO_ROOT']).resolve();OLD=ROOT/'CODE_REVISIONS/REPAIR_01_SOURCE_AND_CONTROL/EXECUTION_REVISION_02';REV=Path(__file__).resolve().parent
if (REV/'CODE_SHA.json').exists():raise SystemExit('ALREADY_FROZEN_NO_REPREPARE')
for name in ('aro_core.py','control.py','aro_tests.py','aro_state_policy.py','repair_tests.py','run_aro400.py'):shutil.copyfile(OLD/name,REV/name)
shutil.copytree(OLD/'lifecycle',REV/'lifecycle',dirs_exist_ok=True);shutil.copyfile(OLD/'PREFLIGHT_RESULT.json',REV/'PREFLIGHT_RESULT.json');sys.path.insert(0,str(REV));from control import *
run=read(ROOT/'RUN.json');assert run['run_id']=='R44_AUTO_6e708d658e074a2988bea6e49319ba2d' and run['deadline_utc']==1793715496.6286445
for p in ROOT.glob('ATTEMPT_*/PROCESS.json'):
 owner=read(p)['child'];assert not rt.same(owner,rt.proc(owner['pid']))
owner=read(OLD/'LAUNCHER.json')['owner'];assert not rt.same(owner,rt.proc(owner['pid']))
assert read(ROOT/'ATTEMPT_002/WAIT.json')['returncode']==70
for name,h in read(OLD/'CODE_SHA.json').items():assert sha(OLD/name)==h,('UNREGISTERED_CHANGE',name)
with zipfile.ZipFile(REV/'instructions/ARO400_PREFLIGHT_REUSE_REVIEW_AND_COMMAND.zip') as z:
 for item in json.loads(z.read('REVIEW_PACKAGE_MANIFEST.json')):assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
 z.extractall(REV/'instructions/extracted')
guard="""def preflight():
 if (REV/'PREFLIGHT_RESULT.json').exists():
  saved=read(REV/'PREFLIGHT_RESULT.json')
  if saved.get('passed') is not True:raise ValueError('STORED_PREFLIGHT_NOT_PASSED')
  for field in ('control','synthetic','real_small','frozen_source'):
   reference=saved[field]
   if Path(reference['path']).stat().st_size!=reference['bytes']:raise ValueError('STORED_PREFLIGHT_REFERENCE_BYTES')
   rt.verify_ref(reference)
  if 'exit_protocol_test' in saved:
   reference=saved['exit_protocol_test']
   if Path(reference['path']).stat().st_size!=reference['bytes']:raise ValueError('STORED_EXIT_TEST_REFERENCE_BYTES')
   rt.verify_ref(reference)
  verify_source_lock()
  lock=read(ROOT/'SOURCE_LOCK.json')
  for reference in lock['files']+lock['gates']:
   if Path(reference['path']).stat().st_size!=reference['bytes']:raise ValueError('SOURCE_LOCK_DEPENDENCY_BYTES')
  return
"""
source=(REV/'run_aro400.py').read_text();assert source.count('def preflight():\n')==1;patched=source.replace('def preflight():\n',guard,1).replace("ROOT/'FINAL_REVISIONS/REPAIR_01_SOURCE_AND_CONTROL'","ROOT/'FINAL_REVISIONS/REPAIR_02_PREFLIGHT_REUSE'");compile(patched,'new_preflight','exec');(REV/'run_aro400.py').write_text(patched);(REV/'PATCH_DIFF.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),patched.splitlines(True),fromfile='OLD/run_aro400.py',tofile='REV/run_aro400.py')))
session=REV/'REUSE_CHECKS'/str(time.time_ns());session.mkdir(parents=True);lease,inv=rt.acquire(ROOT,run,read(ROOT/'PROTECTION_BEFORE.json'))
if lease is None:write(session/'RESOURCE_NOT_AVAILABLE.json',inv,True);raise SystemExit('RESOURCE_NOT_AVAILABLE')
try:
 sys.argv=['new_entry','--root',str(ROOT),'--config',str(ROOT/'CONFIG.json'),'--role','preflight'];spec=importlib.util.spec_from_file_location('actual_new_entry',REV/'run_aro400.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);old_opt=gp.Model.optimize;calls=[]
 def forbidden(*a,**k):calls.append(True);raise AssertionError('OPTIMIZER_FORBIDDEN_DURING_REUSE_TEST')
 gp.Model.optimize=forbidden
 try:
  original_bytes=(REV/'PREFLIGHT_RESULT.json').read_bytes();module.preflight();module.preflight();assert (REV/'PREFLIGHT_RESULT.json').read_bytes()==original_bytes
  saved=read(REV/'PREFLIGHT_RESULT.json');sandbox=session/'COPIED_REVISION';sandbox.mkdir();shutil.copyfile(REV/'PREFLIGHT_RESULT.json',sandbox/'PREFLIGHT_RESULT.json');module.REV=sandbox;module.preflight();assert (sandbox/'PREFLIGHT_RESULT.json').read_bytes()==original_bytes
  # The old final publication expression must still reproduce the original failure.
  tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight');captured=[];ns={'REV':OLD,'ROOT':ROOT,'ref':ref,'write':lambda p,v,*a:captured.append(v)};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn.body[-1]],type_ignores=[])),'old_final_write','exec'),ns);testfile=session/'OLD_IMMUTABLE_REPRO.json';write(testfile,saved,True)
  try:write(testfile,captured[0],True);raise AssertionError('OLD_CONFLICT_NOT_REPRODUCED')
  except ValueError as exc:assert 'IMMUTABLE_OBJECT_CONFLICT' in str(exc)
  negative=[]
  for name in ('sha','bytes','missing','false','truthy','exit_sha'):
   altered=copy.deepcopy(saved)
   if name=='sha':altered['control']['sha256']='0'*64
   if name=='bytes':altered['synthetic']['bytes']=0
   if name=='missing':del altered['real_small']
   if name=='false':altered['passed']=False
   if name=='truthy':altered['passed']='true'
   if name=='exit_sha':altered['exit_protocol_test']['sha256']='f'*64
   d.atomic(sandbox/'PREFLIGHT_RESULT.json',altered)
   try:module.preflight();raise AssertionError('INVALID_REPORT_ACCEPTED '+name)
   except (ValueError,KeyError):negative.append(name)
  d.atomic(sandbox/'PREFLIGHT_RESULT.json',saved);checker=module.verify_source_lock
  def bad_source():raise ValueError('INJECTED_NESTED_SOURCE_FAILURE')
  module.verify_source_lock=bad_source
  try:module.preflight();raise AssertionError('NESTED_SOURCE_FAILURE_IGNORED')
  except ValueError as exc:assert str(exc)=='INJECTED_NESTED_SOURCE_FAILURE'
  finally:module.verify_source_lock=checker;module.REV=REV
  ready=sorted(int(p.parent.name) for p in (ROOT/'oracles').glob('*/CACHE_COMPLETE.json'));expected=set(range(351))|{385};assert expected<=set(ready)
  cache_before={str(p):sha(p) for sid in ready for p in (ROOT/'oracles'/f'{sid:04d}').rglob('*') if p.is_file()};builder_before=len(list((ROOT/'CALLS').glob('*/BEGIN.json')));build_oracles(ready);cache_after={p:sha(p) for p in cache_before};assert cache_before==cache_after;assert builder_before==len(list((ROOT/'CALLS').glob('*/BEGIN.json')))
  write(session/'CACHE_AUDIT.json',{'ready_ids':ready,'missing_ids':sorted(set(range(400))-set(ready)),'original352_verified_unchanged':True,'payload_SHA_before':cache_before,'cache_hit_calls':len(ready),'new_models_built':0,'new_native_calls':0},True)
  report={'passed':True,'original_conflict_reproduced':True,'formal_preflight_called_twice_no_write':True,'copied_revision_reuse_preserves_original_bytes':True,'negative_cases_rejected':negative,'nested_SOURCE_LOCK_failure_propagates':True,'actual_nested_sources_revalidated':True,'strict_passed_is_True':True,'all_saved_reference_SHA_and_bytes_checked':True,'new_native_calls':len(calls),'old14_test_calls_not_rerun':True,'matrices_cache_hits':len(ready),'missing_matrices':sorted(set(range(400))-set(ready))};assert not calls;write(session/'RESULT.json',report,True)
 finally:gp.Model.optimize=old_opt
finally:rt.release(ROOT,lease)
unchanged={name:sha(REV/name)==sha(OLD/name) for name in ('aro_core.py','control.py','aro_tests.py','aro_state_policy.py','repair_tests.py')};assert all(unchanged.values());write(REV/'UNCHANGED_SCIENTIFIC_MODULES.json',{'modules':unchanged,'lifecycle':all(sha(p)==sha(OLD/'lifecycle'/p.name) for p in (REV/'lifecycle').glob('*.py')),'only_preflight_and_final_registration_path_changed':True},True)
frozen={str(p.relative_to(REV)):sha(p) for p in REV.rglob('*.py') if 'instructions' not in p.parts};write(REV/'CODE_SHA.json',frozen,True)
resolution={'run_id':run['run_id'],'original_RUN':ref(ROOT/'RUN.json'),'old_FAILURE':ref(ROOT/'ATTEMPT_002/FAILURE.json'),'old_WAIT':ref(ROOT/'ATTEMPT_002/WAIT.json'),'old_FINAL':ref(ROOT/'FINAL_REVISIONS/REPAIR_01_SOURCE_AND_CONTROL/ATTEMPT_002/REPORT.json'),'old_CODE_SHA':ref(OLD/'CODE_SHA.json'),'new_CODE_SHA':ref(REV/'CODE_SHA.json'),'authority':ref(REV/'instructions/REQUEST.txt'),'test':ref(session/'RESULT.json'),'cache_audit':ref(session/'CACHE_AUDIT.json'),'original_deadline_utc':run['deadline_utc'],'remaining_s':rt.deadline(run)-time.monotonic(),'old_attempts_used':3,'next_attempt':3,'max_attempts':5,'no_budget_reset':True,'reopen_only_ATTEMPT002_metadata_failure':True};write(REV/'RECOVERY_RESOLUTION.json',resolution,True);write(ROOT/'ARO_ACTIVE_REVISION_REUSE.json',{'path':str(REV),'code':ref(REV/'CODE_SHA.json'),'resolution':ref(REV/'RECOVERY_RESOLUTION.json')},True);print(json.dumps({'ready':True,'tests':report,'resolution':ref(REV/'RECOVERY_RESOLUTION.json'),'remaining_s':resolution['remaining_s']}))
