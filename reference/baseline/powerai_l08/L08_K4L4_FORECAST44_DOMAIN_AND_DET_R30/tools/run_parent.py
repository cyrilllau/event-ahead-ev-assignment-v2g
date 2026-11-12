from pathlib import Path
import subprocess,sys,json,time,os
R=Path(__file__).resolve().parents[1];d=R/'logs'/('PARENT_'+str(time.time_ns()));d.mkdir(parents=True)
def save(n,x):
 p=d/n;p.write_text(json.dumps(x,indent=2)+'\n')
with (d/'stdout.log').open('wb') as out,(d/'stderr.log').open('wb') as err:
 p=subprocess.Popen([sys.executable,'-B',str(R/'tools/run_l08_k4l4_forecast44_domain_det_r30.py'),*sys.argv[1:]],stdout=out,stderr=err,cwd=R)
 save('PROCESS.json',dict(pid=p.pid,parent=os.getpid(),argv=p.args,epoch=time.time()));print(str(d),flush=True);s=time.monotonic();rc=p.wait();save('WAIT.json',dict(returncode=rc,elapsed_s=time.monotonic()-s,epoch=time.time(),actual_parent_wait=True))
sys.exit(rc)
