from aggregate_core import *
import subprocess,traceback
results={};folder=OUT/'tests';folder.mkdir()
try:
 CTX.check(True)
 with (folder/'SYNTHETIC.stdout.log').open('xb') as so,(folder/'SYNTHETIC.stderr.log').open('xb') as se:
  p=subprocess.Popen([sys.executable,'-B',str(OUT/'tools/math_preflight.py'),'--output',str(folder/'SYNTHETIC_OUTPUT.json')],stdout=so,stderr=se,cwd=folder);t=time.monotonic();rc=p.wait(timeout=max(.1,CTX.prep-time.monotonic()))
 write(folder/'SYNTHETIC_WAIT.json',dict(actual_parent_wait=True,returncode=rc,elapsed=time.monotonic()-t));assert rc==0;results['synthetic']='PASS'
 from actual_tests import function_tests,reference_regression,real_small_test
 CTX.check(True);function_tests();results['actual_function_tests']='PASS'
 CTX.check(True);reference_regression();results['reference_regression']='PASS'
 CTX.check(True);e.DEADLINE=min(CTX.deadline,CTX.prep);real_small_test();results['real_small_test']='PASS'
 # A complete certified full400-to-integer-master closed loop has not been
 # established before the original preparation deadline. No budget reset.
 write(OUT/'TERMINAL.json',dict(status='PREPARATION_INCOMPLETE',completed=results,formal_MIP='NOT_RUN',reason='Correct closed loop not established within retained preparation budget',epoch=time.time()))
except BaseException as exc:
 write(OUT/'TERMINAL.json',dict(status='PREPARATION_INCOMPLETE' if time.monotonic()>=CTX.prep-15 else 'PREFLIGHT_BLOCKED',completed=results,error=repr(exc),traceback=traceback.format_exc(),formal_MIP='NOT_RUN',epoch=time.time()));raise
