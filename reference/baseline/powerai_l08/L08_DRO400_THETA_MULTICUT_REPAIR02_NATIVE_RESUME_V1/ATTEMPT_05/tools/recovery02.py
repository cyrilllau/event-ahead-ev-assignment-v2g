from pathlib import Path
import json,hashlib,zipfile,shutil,time,collections,gc
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import run_dro400_theta_multicut as o
from envelope_sparse import digest,enclosure
from native_repair02 import point_sha
ANCHORS=['aea926953d50564b106f850fb0026d2f8497fd71f0e186a36b4d6d513bf05a8a','402d6168c57bc15102c42e694a993a2bb09705b53b5e52e58e3a5e4211f87cc8','18cafaef61fadae7be7b9b616bb0bfa58f99311c744ade43cc95c4180beecc5c']
SHA='f87c0f2ec5eb4d106ac4979fa1bb1adffc881cc01d220f5ef316ecf393b36ebd'
FILESHAS=['7c8b84cb0c1d24156e59de66fdfc2849398cec38c6b705033812823200e1e2a4','d5acd20faf25e18d2641a5d808c89eb1c9bbf2b17875d79b0b4dc0578b7b6fc6','ad79137b243d831af96211c437225e61c737f937f40fb6c646cf1912d3b3ccc0']
class Sealed:
 def __init__(self,src):
  self.refs={};self.used={};self.src=src
  for i,sha in enumerate(ANCHORS,1):
   p=src.parent/'DELIVERY_POSTPROCESS'/f'DRO_THETA_MULTICUT_PRO_ANALYSIS_{i:02d}.zip'
   assert o.cr.sha(p)==sha,('ARCHIVE_SHA',p)
   with zipfile.ZipFile(p) as z:v=json.loads(z.read('PAYLOAD_MANIFEST.json'))
   o.write(f'SOURCE_MANIFESTS/PART_{i:02d}.json',v,True)
   for e in v['members']:
    r=dict(path=e['source_path'],bytes=e['bytes'],sha256=e['sha256'])
    if r['path'] in self.refs:assert self.refs[r['path']]==r
    self.refs[r['path']]=r
  p=src.parent/'DELIVERY_POSTPROCESS/FULL_SERVER_MANIFEST.json';self.verify(p);v=o.read(p)
  rows=v if isinstance(v,list) else v['all_files']
  for r in rows:
   path=r.get('path',r.get('source_path'));rr=dict(path=path,bytes=r['bytes'],sha256=r['sha256'])
   if path in self.refs:assert self.refs[path]==rr
   self.refs[path]=rr
 def verify(self,p):
  p=str(p);r=self.refs[p];o.cr.verify(r);self.used[p]=r;return Path(p)
 def read(self,p):return o.read(self.verify(p))
 def finish(self):o.write('SOURCE_MANIFEST.json',dict(verified_sealed_references=list(self.used.values()),archive_sha256=ANCHORS,upstream_dependencies=list(o.cr.DEPS.values()),no_optimizer=True))

def assert_counts(ids,pids):
 if sum(map(len,ids.values()))!=9253 or len(pids)!=48:raise ValueError('REVISION1_INSTALLED_SETS_INCOMPLETE')
def bind_raw(A,a,ident,end):
 if ident['matrix_sha']!=SHA or o.ac.b.matrix_sha(A)!=SHA or A.shape!=(100277,61037) or A.nnz!=9604659 or end['revision']!=1:raise ValueError('RAW_MATRIX_BINDING_FAILURE')
 if sum(a['vtype']=='B')!=25770:raise ValueError('WRONG_INTEGER_DOMAIN')
def validate_evaluation(c,write_copy=None,replay_scenarios=False):
 assert c['science']==o.ac.SCIENCE and c['P_identity']==o.CFG['P_identity'] and c['full400']
 q=dict(np.load(o.ac.rt.verify_ref(c['Q'])));records=o.read(o.ac.rt.verify_ref(c['records']));pc=dict(np.load(o.ac.rt.verify_ref(c['risk']['certificate'])))
 assert point_sha(q['x'])==c['point_sha'] and np.array_equal(q['qL'],o.ac.full_vector(records,'qL')) and np.array_equal(q['qU'],o.ac.full_vector(records,'qU_audited'))
 assert np.array_equal(pc['qL'],q['qL']) and np.array_equal(pc['qU'],q['qU'])
 check=o.ac.b.probability_check(pc['pstar']);assert check['passed']
 risk=o.ac.b.risk_upper(q['qU'],float(pc['beta0']),pc['plus'],pc['minus']);assert abs(risk['risk_upper']-c['risk']['risk_U'])<1e-7
 f=float(np.nextafter(float(enclosure(sp.csr_matrix(o.ac.first()[1]['cost'][None,:]),q['x'])[1][0]),np.inf));ub=float(np.nextafter(f+risk['risk_upper'],np.inf));assert abs(f-c['F'])<1e-7 and abs(ub-c['UB_audited'])<1e-7
 if replay_scenarios:
  for r in records:
   o.CTX.check()
   if r.get('universal_zero'):continue
   oracle=o.ac.Oracle(r['sid'],o.OUT/'CACHE_VERIFICATION_NO_MODEL')
   try:
    result=Path(r['cut']).parent/'RESULT.json';rr=oracle.validate_result(q['x'],result)
    assert rr['qL']==r['qL'] and rr['qU_audited']==r['qU_audited'] and oracle.model is None
   finally:oracle.close(discard=True)
 if write_copy:o.write(write_copy,dict(passed=True,point_sha=c['point_sha'],replayed_F=f,replayed_UB=ub,probability_check=check,source_Q=c['Q'],source_risk=c['risk']['certificate'],recourse_replay_no_optimizer=replay_scenarios),True)
 return q,records,pc

def restore(src,snap,native):
 sealed=Sealed(src)
 for f,h in zip(('A.npz','ARRAYS.npz','NAMES.json'),FILESHAS):assert o.cr.sha(sealed.verify(snap/f))==h
 A=sp.load_npz(snap/'A.npz').tocsr();a=dict(np.load(snap/'ARRAYS.npz'));names=o.read(snap/'NAMES.json');ident=sealed.read(snap/'IDENTITY.json');end=sealed.read(native/'END.json')
 bind_raw(A,a,ident,end);assert end['status']==11 and end['reason']=='CALLBACK_ERROR'
 rawpath=sealed.verify(native/'RAW_NATIVE.npz');assert o.cr.sha(rawpath)=='617d4e63f164e36a6ba71f3473fb44882c4150786d0b60461218a482e9ff70a1'
 raw=np.load(rawpath)['X'];cand=np.load(sealed.verify(native/'CANDIDATE_0000.npz'));assert np.array_equal(raw,cand['full']) and np.array_equal(raw[:o.N],cand['x']) and point_sha(raw[:o.N])=='40c0c93690795ccadefbb91bb04b92bce418d9f535d2b0cc8b735925dc86cbdb'
 sealed.verify(native/'BEGIN.json');sealed.verify(native/'CANDIDATES.json');sealed.verify(native/'NATIVE.log')
 # Reads committed old arrays and their existing proof closure. No optimizer/build(bank).
 bank,evs,legacy,la,ln,eq,lb,best=o.source_restore()
 d=A[:90976,:60781]-legacy;d.eliminate_zeros();assert d.nnz==0 and A[:90976,60781:].nnz==0
 for k in ('cost','lo','hi','vtype'):assert np.array_equal(a[k][:60781],la[k])
 for k in ('rhs','sense'):assert np.array_equal(a[k][:90976],la[k])
 assert float(a['objcon'])==float(la['objcon']) and names['cols'][:60781]==ln['cols'] and names['rows'][:90976]==ln['rows']
 del legacy,la,ln;gc.collect()
 mapping={int(k):v for k,v in sealed.read(src.parent/'THETA_MAP.json').items()};assert len(mapping)==256 and set(mapping.values())==set(range(60781,61037))
 assert names['cols'][60780]=='DRO_RHO' and all(names['cols'][j]=='THETA_SCENARIO_'+str(s) for s,j in mapping.items())
 assert np.all(a['cost'][60781:]==0) and np.all(a['lo'][60781:]==0) and np.all(a['hi'][60781:]>=gp.GRB.INFINITY) and np.all(a['vtype'][60781:]=='C')
 oldproof=sealed.read(src.parent/'SUPPORT_STORAGE_PROOFS.json');oldproof={int(v['sid']):{e['id']:e for e in v['rows']} for v in oldproof}
 inc=sealed.read(src/'INCREMENTS/R00001/MANIFEST.json');newproof={(r['sid'],r['id']):r for r in inc['new_supports']};assert len(newproof)==237
 db=o.read(o.ac.rt.verify_ref(inc['evaluation']));assert db['point_sha']=='db7366543bf6982e29a9201ec0089c046c258b2ca2b088fd36ffe5ebc9466f97'
 dbq,dbrecords,dbpc=validate_evaluation(db,'RESTORE/DB_POINT_REPLAY.json',True)
 evals={e['ce']['point_sha']:e['ce'] for e in evs};evals[db['point_sha']]=db
 qbest,_,_=validate_evaluation(best,'RESTORE/BEST_REPLAY.json');assert point_sha(qbest['x'])=='951cc92fb17df719e6a1d224095b39d0e51a23d47a2b56289b2a84cbf1eb2f91'
 # Validate old rows against committed source coefficients and their exact storage guard.
 rowgroups={s:[] for s in mapping};support_ids={s:set() for s in mapping};prooflinks=[];p_rows=[]
 for i,name in enumerate(names['rows'][90976:],90976):
  o.CTX.check()
  if name.startswith(('theta_support/','adaptive/')):
   _,sid,h=name.split('/');sid=int(sid);assert sid in mapping and h not in support_ids[sid]
   row=A.getrow(i);aux=row[:,o.N:];assert aux.nnz==1 and float(A[i,mapping[sid]])==1 and a['sense'][i]=='>'
   actual=-row[:,:o.N].tocsr()
   if name.startswith('theta_support/'):
    j=bank.blocks[sid]['ids'].index(h);source=bank.blocks[sid]['B'].getrow(j);alpha=float(bank.blocks[sid]['alpha'][j]);pr=oldproof[sid][h];assert pr['row']==i
   else:
    nr=newproof[sid,h];assert nr['row']==i
    z=np.load(o.cr.verify(nr['source']));alpha=float(z['alpha']);source=sp.csr_matrix(z['beta'][None,:]);assert digest(alpha,z['beta'])==h
    st=np.load(o.cr.verify(nr['stored']));sb=sp.csr_matrix((st['beta_data'],st['beta_indices'],st['beta_indptr']),shape=tuple(st['beta_shape']));dd=sb-actual;dd.eliminate_zeros();assert dd.nnz==0 and float(st['alpha'][0])==a['rhs'][i]
    pr=nr['proof'][0]
   rhs,pr2=o.tk.compensate(alpha,source,actual,bank.lo,bank.hi);assert rhs==a['rhs'][i] and pr2['changed_coefficients']==pr['changed_coefficients'] and pr2['alpha_loss']==pr['alpha_loss']
   support_ids[sid].add(h);rowgroups[sid].append(i);prooflinks.append(dict(row=i,name=name,source_id=h,sid=sid,storage_guard=pr2))
  elif name.startswith(('theta_probability/','adaptive_probability/')):p_rows.append((i,name.rsplit('/',1)[1]))
  else:raise ValueError(('UNKNOWN_NONPHYSICAL_ROW',i,name))
 pmap={h:p for h,p in zip(bank.pids,bank.p)};pmap[point_sha(dbpc['pstar'])]=dbpc['pstar'];assert len(pmap)==48
 pproof=sealed.read(src.parent/'PROBABILITY_STORAGE_PROOFS.json')+[inc['new_probability']];pproof={p['full400_p_sha']:p for p in pproof}
 pids=set()
 for i,h in p_rows:
  p=pmap[h];pr=pproof[h];assert pr['row']==i and pr['faithful'] and len(p)==400
  expected=sp.csr_matrix((list([-np.ldexp(float(p[s]),pr['power2']) for s,j in sorted(mapping.items()) if p[s]!=0])+[pr['scale']],([0]*(sum(p[s]!=0 for s in mapping)+1),[j for s,j in sorted(mapping.items()) if p[s]!=0]+[o.RHO])),shape=(1,61037))
  diff=A.getrow(i)-expected;diff.eliminate_zeros();assert diff.nnz==0 and a['rhs'][i]==0 and a['sense'][i]=='>'
  pids.add(h)
 assert_counts(support_ids,pids);stored={s:(-A[rows,:o.N].tocsr(),a['rhs'][rows].copy()) for s,rows in rowgroups.items()}
 # Recover actual historical ledger; do not trust old controller counters.
 counts=collections.Counter();test=0
 for root in (src.parent,src):
  for p in root.glob('OPTIMIZER_CALL_LEDGER.jsonl'):
   for line in p.read_text().splitlines():
    v=json.loads(line)
    if v.get('phase')=='BEGIN':
     if v.get('stage')=='TEST':test+=1
     else:counts[v.get('kind',v.get('stage'))]+=1
  for p in (root/'CALLS').glob('*/BEGIN.json'):
   v=o.read(p);counts[v.get('kind','UNKNOWN')]+=1
 # Worker calls have their own locations; enumerate native BEGIN only, deduplicate call IDs.
 ledger={}
 for root in (src.parent,src):
  for p in root.rglob('BEGIN.json'):
   if root==src.parent and src in p.parents:continue
   if 'DELIVERY_POSTPROCESS' in p.parts:continue
   v=o.read(p)
   if v.get('id') and v.get('kind'):ledger[v['id']]=v
 hist=collections.Counter(v['kind'] for v in ledger.values());scientific=sum(n for k,n in hist.items() if k!='TEST')
 assert scientific==263 and hist.get('TEST',0)==4,('HISTORICAL_CALL_LEDGER',dict(hist),scientific)
 o.write('HISTORICAL_CALL_COUNTS.json',dict(by_kind=dict(hist),scientific=scientific,tests=4,old_controller_master_calls=0,correct_master_calls=1,source='unique actual BEGIN IDs'),True)
 # Preserve source partial result, no Pi/basis or tree invented.
 o.write('RECOVERED_PARTIAL_NATIVE_RESULT.json',dict(end=end,snapshot=ident,raw=sealed.refs[str(rawpath)],raw_bound=end['raw_bound'],bound=end['bound'],partial=True,controller_stale_unsolved=True,branch_tree_restored=False,old_failure_where='UNKNOWN_NOT_RECORDED'),True)
 o.write('RESTORE/ROW_PROOF_MAPPING.json',prooflinks,True)
 o.write('RESTORE/INSTALLED.json',dict(support_ids={s:sorted(v) for s,v in support_ids.items()},pids=sorted(pids),revision=1),True)
 duplicate=sum(1 for r in inc['new_supports'] if r['id'] not in support_ids[r['sid']])+int(inc['new_probability']['full400_p_sha'] not in pids);assert duplicate==0
 o.write('RESTORE/REVISION1_GATE.json',dict(passed=True,rows=100277,columns=61037,nnz=9604659,binary=25770,physical=90768,legacy=208,supports=9253,probabilities=48,matrix_sha=SHA,replayed_db_install_new_rows=duplicate,stored_source_verified=True,old_scientific_recomputed=0),True)
 cachelp=sealed.read(src/'CACHE_LP/SOLVED/LOWER_CERTIFICATE.json');assert cachelp['lower_certificate_verified'];o.ac.rt.verify_ref(cachelp['corrected_pi']);o.ac.rt.verify_ref(cachelp['raw'])
 o.write('BOUND_SOURCES.json',dict(history=o.read(o.OUT/'HISTORICAL_LB_SOURCE.json'),cache_lp=cachelp,partial_native=dict(raw=end['raw_bound'],guarded=end['bound']),merged=max(lb,cachelp['LB'],end['bound']),no_objective_floor=True),True)
 sealed.finish();o.REVISION=1
 return dict(A=A,a=a,names=names,ident=ident,mapping=mapping,stored=stored,support_ids=support_ids,pids=pids,bank=bank,evals=evals,eq=eq,lb=max(lb,cachelp['LB'],end['bound']),best=best,raw=raw,db=db,sealed=sealed)
