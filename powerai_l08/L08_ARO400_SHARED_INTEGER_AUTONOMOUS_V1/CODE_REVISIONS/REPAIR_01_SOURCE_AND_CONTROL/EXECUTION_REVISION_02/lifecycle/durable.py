"""Single-writer immutable objects, hash-linked journal, and exact row replay."""
from pathlib import Path
import os,json,hashlib,uuid,time,shutil
import numpy as np
from scipy import sparse as sp

def cv(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def encoded(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),default=cv,allow_nan=False)+'\n').encode()
def digest(x):return hashlib.sha256(encoded(x)).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def syncdir(p):
 fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic(p,x,immutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=encoded(x)
 if immutable and p.exists():
  if p.read_bytes()!=data:raise ValueError('IMMUTABLE_OBJECT_CONFLICT '+str(p))
  return
 temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 if immutable and p.exists():raise ValueError('IMMUTABLE_PUBLISH_RACE')
 os.replace(temp,p);syncdir(p.parent)
 if p.read_bytes()!=data:raise ValueError('PUBLISHED_OBJECT_READBACK_FAILED')
def read(p):return json.loads(Path(p).read_text())
def publish_npz(path,**arrays):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(str(p))
 temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 with temp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
 with np.load(temp,allow_pickle=False) as z:
  for k,v in arrays.items():
   if not np.array_equal(z[k],v):raise ValueError('NPZ_READBACK_FAILED '+k)
 h=sha(temp);os.replace(temp,p);syncdir(p.parent)
 if sha(p)!=h:raise ValueError('NPZ_PUBLISH_SHA_FAILED')
 return {'path':str(p),'sha256':h,'bytes':p.stat().st_size,'encoding':'uncompressed_npz'}
def publish_csr(p,A):
 A=A.tocsr();return publish_npz(p,data=A.data,indices=A.indices,indptr=A.indptr,shape=np.array(A.shape))
def load_csr(p):
 z=np.load(p);return sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
def identity(s):
 A=s['A'];return {'rows':A.shape[0],'columns':A.shape[1],'nnz':A.nnz,'matrix_sha':hashlib.sha256(A.data.tobytes()+A.indices.tobytes()+A.indptr.tobytes()).hexdigest(),'rhs_sense_sha':hashlib.sha256(s['rhs'].tobytes()+s['sense'].tobytes()).hexdigest(),'rows_sha':hashlib.sha256(json.dumps(s['names']['rows']).encode()).hexdigest(),'columns_sha':hashlib.sha256(json.dumps(s['names']['columns']).encode()).hexdigest(),'objective_bounds_sha':hashlib.sha256(b''.join(s[k].tobytes() for k in ('cost','lo','hi'))).hexdigest(),'objective_constant':0.0}
def disk_check(root,estimated=400*1024**2,limit=100*1024**3,free_override=None):
 root=Path(root);used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());free=shutil.disk_usage(root).free if free_override is None else free_override
 reserve=max(5*1024**3,4*estimated)
 return {'passed':used<limit and free>=reserve,'OUT_bytes':used,'OUT_limit':limit,'free':free,'minimum_free':reserve,'quota':'unknown: no readable per-user quota interface assumed'}

class Store:
 def __init__(self,root,base=None):
  self.root=Path(root);self.base=Path(base) if base else self.root/'BASE';self.root.mkdir(parents=True,exist_ok=True)
  self.bank=None;self.rhs=None;self.sense=None;self.ids=[];self.names=[];self.lookup={};self.blocks=[]
 def initialize(self,s):
  if self.base.exists():return
  self.base.mkdir();A=sp.vstack([s['A'],s['dA']],format='csr');rhs=np.r_[s['rhs'],s['drhs']];sense=np.concatenate([s['sense'],s['dsense']]);names=s['names']['rows']+s['dnames']
  if len(set(names))!=len(names):raise ValueError('DUPLICATE_INITIAL_ROW_NAMES')
  refs=[publish_csr(self.base/'ROW_BANK.npz',A),publish_npz(self.base/'ARRAYS.npz',rhs=rhs,sense=sense,cost=s['cost'],lo=s['lo'],hi=s['hi'])]
  atomic(self.base/'NAMES.json',{'rows':names,'columns':s['names']['columns'],'initial_active':s['names']['rows'],'initial_dormant':s['dnames']},True)
  atomic(self.base/'IDENTITY.json',{'files':refs,'initial_identity':identity(s),'names_sha':sha(self.base/'NAMES.json')},True)
 def load_bank(self,blocks):
  proof=read(self.base/'IDENTITY.json')
  for ref in proof['files']:
   if sha(ref['path'])!=ref['sha256']:raise ValueError('BASE_INTEGRITY_FAILURE')
  if sha(self.base/'NAMES.json')!=proof['names_sha']:raise ValueError('BASE_NAMES_INTEGRITY_FAILURE')
  self.bank=load_csr(self.base/'ROW_BANK.npz');self.arr=dict(np.load(self.base/'ARRAYS.npz'));self.rhs=self.arr['rhs'];self.sense=self.arr['sense'];self.meta=read(self.base/'NAMES.json');self.names=list(self.meta['rows']);self.ids=['base/'+str(i) for i in range(len(self.names))]
  self.lookup={n:i for i,n in enumerate(self.names)};self.blocks=[]
  for ref in blocks:self.add_block(ref)
 def add_block(self,ref):
  if sha(ref['path'])!=ref['sha256']:raise ValueError('ROW_BLOCK_INTEGRITY_FAILURE')
  z=np.load(ref['path']);A=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']));names=z['names'].tolist();ids=z['ids'].tolist()
  if any(n in self.lookup for n in names):raise ValueError('ROW_NAME_CONTENT_CONFLICT')
  self.bank=sp.vstack([self.bank,A],format='csr');self.rhs=np.r_[self.rhs,z['rhs']];self.sense=np.concatenate([self.sense,z['sense']]);start=len(self.names)
  self.names+=names;self.ids+=ids;self.lookup.update({n:start+i for i,n in enumerate(names)});self.blocks.append(ref)
 def register(self,s,sources):
  missing=[i for i,n in enumerate(s['names']['rows']) if n not in self.lookup]
  if not missing:return
  A=s['A'][missing].tocsr();names=[s['names']['rows'][i] for i in missing];rhs=s['rhs'][missing];sense=s['sense'][missing]
  ids=[hashlib.sha256(A.getrow(j).indices.tobytes()+A.getrow(j).data.tobytes()+np.array([rhs[j]]).tobytes()+str(sense[j]).encode()).hexdigest() for j in range(len(missing))]
  key=digest({'ids':ids,'names':names});p=self.root/'ROW_STORE'/(key+'.npz')
  if not p.exists():ref=publish_npz(p,data=A.data,indices=A.indices,indptr=A.indptr,shape=np.array(A.shape),rhs=rhs,sense=sense,names=np.array(names),ids=np.array(ids))
  else:ref={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'encoding':'uncompressed_npz'}
  atomic(p.with_suffix('.sources.json'),{'sources':sources,'ids':ids,'names':names},True);self.add_block(ref)
 def mapping(self,s):
  A=s['dA'];dh=hashlib.sha256(A.data.tobytes()+A.indices.tobytes()+A.indptr.tobytes()+s['drhs'].tobytes()+s['dsense'].tobytes()+encoded(s['dnames'])).hexdigest()
  return {'active':[self.ids[self.lookup[n]] for n in s['names']['rows']],'dormant':[self.ids[self.lookup[n]] for n in s['dnames']],'identity':identity(s),'dormant_sha':dh}
 def materialize(self,mapping):
  index={n:i for i,n in enumerate(self.ids)};aa=[index[i] for i in mapping['active']];dd=[index[i] for i in mapping['dormant']]
  s={'A':self.bank[aa].tocsr(),'rhs':self.rhs[aa],'sense':self.sense[aa],'cost':self.arr['cost'],'lo':self.arr['lo'],'hi':self.arr['hi'],'names':{'rows':[self.names[i] for i in aa],'columns':self.meta['columns']},'dA':self.bank[dd].tocsr(),'drhs':self.rhs[dd],'dsense':self.sense[dd],'dnames':[self.names[i] for i in dd],'dsource':[None]*len(dd)}
  if identity(s)!=mapping['identity']:raise ValueError('REPLAY_IDENTITY_FAILURE')
  if 'dormant_sha' in mapping and self.mapping(s)['dormant_sha']!=mapping['dormant_sha']:raise ValueError('DORMANT_REPLAY_IDENTITY_FAILURE')
  return s
 def head(self,validate=True):
  p=self.root/'LAST_COMMITTED.json'
  if not p.exists():return None
  head=read(p);ref=head['transaction']
  if sha(ref['path'])!=ref['sha256']:raise ValueError('COMMITTED_JOURNAL_SHA_FAILURE')
  tx=read(ref['path'])
  if validate:
   current=tx
   while current['previous']:
    prev=current['previous']
    if sha(prev['path'])!=prev['sha256']:raise ValueError('JOURNAL_CHAIN_BROKEN')
    parent=read(prev['path'])
    if parent['sequence']+1!=current['sequence']:raise ValueError('JOURNAL_SEQUENCE_BROKEN')
    current=parent
  return tx
 def commit(self,s,metadata,sources=None,fail_at=None):
  self.register(s,sources or [])
  old=self.head(False);seq=0 if old is None else old['sequence']+1
  payload={'working':self.mapping(s),'row_blocks':self.blocks,'metadata':metadata}
  # Detect altered coefficients behind an existing row name before publishing.
  self.materialize(payload['working'])
  payload_path=self.root/'OBJECTS'/(digest(payload)+'.json');atomic(payload_path,payload,True)
  if fail_at=='payload':raise RuntimeError('INJECTED_AFTER_PAYLOAD')
  prev=read(self.root/'LAST_COMMITTED.json')['transaction'] if old else None
  tx={'sequence':seq,'previous':prev,'payload':{'path':str(payload_path),'sha256':sha(payload_path)},'epoch':time.time()}
  path=self.root/'JOURNAL'/f'{seq:08d}_{digest(tx)}.json';atomic(path,tx,True)
  if fail_at=='journal':raise RuntimeError('INJECTED_AFTER_JOURNAL')
  ref={'path':str(path),'sha256':sha(path)}
  if fail_at=='reference':raise RuntimeError('INJECTED_BEFORE_REFERENCE')
  atomic(self.root/'LAST_COMMITTED.json',{'transaction':ref,'sequence':seq})
  if self.head(False)!=tx:raise ValueError('HEAD_READBACK_FAILURE')
  return payload
 def replay(self):
  tx=self.head();ref=tx['payload']
  if sha(ref['path'])!=ref['sha256']:raise ValueError('COMMITTED_PAYLOAD_SHA_FAILURE')
  payload=read(ref['path']);self.load_bank(payload['row_blocks']);return self.materialize(payload['working']),payload['metadata']
 def snapshot(self,s,tag):
  d=self.root/'SNAPSHOTS'/tag
  if d.exists():
   saved=read(d/'IDENTITY.json')
   if saved['identity']!=identity(s):raise ValueError('SNAPSHOT_IDENTITY_CONFLICT')
   for ref in saved['files']:
    if sha(ref['path'])!=ref['sha256']:raise ValueError('SNAPSHOT_SHA_FAILURE')
   return {'path':str(d),'identity':identity(s)}
  d.mkdir(parents=True)
  refs=[publish_csr(d/'A.npz',s['A']),publish_csr(d/'DORMANT_A.npz',s['dA']),publish_npz(d/'ARRAYS.npz',**{k:s[k] for k in ('rhs','sense','cost','lo','hi','drhs','dsense')})]
  atomic(d/'NAMES.json',{'rows':s['names']['rows'],'columns':s['names']['columns'],'dormant':s['dnames']},True);atomic(d/'IDENTITY.json',{'identity':identity(s),'files':refs},True);return {'path':str(d),'identity':identity(s)}
