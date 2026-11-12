from pathlib import Path
import json,subprocess,time,hashlib,zipfile,shutil
work=Path.cwd()/'.r36_work';out=Path.cwd()/'outputs/L08_K4L4_DRO_READY_PARALLEL_R36';out.mkdir(parents=True,exist_ok=True)
remote='/home/Shixin/ev-assignment/L08_K4L4_DRO_READY_PARALLEL_R36/delivery/'
def transfer(name,dest):
 assert not dest.exists()
 t=time.monotonic();p=subprocess.Popen(['scp','Shixin@powerai:'+remote+name,str(dest)],stdout=subprocess.PIPE,stderr=subprocess.PIPE);so,se=p.communicate()
 r=dict(returncode=p.returncode,actual_parent_wait=True,elapsed_s=time.monotonic()-t,stderr=se.decode(errors='replace'));(work/(name.replace('.','_')+'_WAIT.json')).write_text(json.dumps(r,indent=2));assert p.returncode==0;return r
transfer('PACKAGE_INDEX.json',work/'PACKAGE_INDEX.json');ix=json.loads((work/'PACKAGE_INDEX.json').read_text());assert shutil.disk_usage(out).free>ix['bytes']+300_000_000
p=out/ix['file'];wait=transfer(ix['file'],p);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()==ix['sha256'] and p.stat().st_size==ix['bytes']
with zipfile.ZipFile(p) as z:
 manifest=json.loads(z.read('FILE_MANIFEST.json'));assert len(manifest)==ix['manifest_members'];assert set(z.namelist())==set(manifest)|{'FILE_MANIFEST.json'}
 for name,item in manifest.items():
  mh=hashlib.sha256();count=0
  with z.open(name) as f:
   for b in iter(lambda:f.read(8*1024**2),b''):mh.update(b);count+=len(b)
  assert count==item['bytes'] and mh.hexdigest()==item['sha256'],name
result=dict(passed=True,path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size,manifest_members=len(manifest),CRC_pass=True,CRC_method='Every member read to EOF; ZipFile checks its stored CRC',transfer=wait)
(work/'LOCAL_PACKAGE_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
