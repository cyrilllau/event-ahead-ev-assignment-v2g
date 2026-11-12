from pathlib import Path
import os,sys,json,subprocess,time
R=Path(__file__).resolve().parents[1];d=R/'operations'/('PARENT_'+str(time.time_ns()));d.mkdir(parents=True);stages=sys.argv[1:]
for stage in stages:
 cmd=[sys.executable,'-B',str(R/'tools/run_l08_k4l4_fullst_dro_parallel_r36.py'),stage,'--resume'];start=time.time()
 with (d/(stage+'.stdout')).open('wb') as out,(d/(stage+'.stderr')).open('wb') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err);(d/(stage+'.PROCESS.json')).write_text(json.dumps(dict(pid=p.pid,parent=os.getpid(),cmd=cmd,epoch=start)));rc=p.wait()
 (d/(stage+'.WAIT.json')).write_text(json.dumps(dict(returncode=rc,actual_parent_wait=True,start=start,end=time.time(),elapsed=time.time()-start,parent=os.getpid(),child=p.pid)))
 print(stage,rc,flush=True)
 if rc:sys.exit(rc)
