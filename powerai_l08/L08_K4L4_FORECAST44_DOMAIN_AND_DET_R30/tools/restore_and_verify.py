import json,hashlib,zipfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archives',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();idx=json.loads((a.archives/'PACKAGE_INDEX.json').read_text());a.destination.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
for package in idx['packages']:
 src=a.archives/package['name'];assert sha(src)==package['sha256']
 with zipfile.ZipFile(src) as z:
  assert z.testzip() is None
  for r in json.loads(z.read('FILE_MANIFEST.json')):
   dst=(a.destination/r['path']).resolve();assert a.destination.resolve() in dst.parents
   dst.parent.mkdir(parents=True,exist_ok=True)
   with z.open(r['path']) as source,dst.open('wb') as dest:
    while True:
     b=source.read(4*1024**2)
     if not b:break
     dest.write(b)
   assert dst.stat().st_size==r['bytes'] and sha(dst)==r['sha256']
 print('VERIFIED',package['name'])
