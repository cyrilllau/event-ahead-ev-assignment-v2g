from runtime import *
from fractions import Fraction as F
from proof_core import prove,row_hash,decode,encode,normalized,validate_dependencies
from replay_certificates import replay
import argparse,traceback,subprocess,csv,gc,shutil
BASE=Path('/home/Shixin/ev-assignment')
SCIENCE='c1fe0c77564e0b350782b0375447b2f637f2e5241871b28601a68cf6aa543765'
EXPECTED='f87c0f2ec5eb4d106ac4979fa1bb1adffc881cc01d220f5ef316ecf393b36ebd'
def pure_tests():
 tests=[]
 def ok(n,v):assert v,n;tests.append(dict(name=n,passed=True))
 def fixture(old=0.,q=1.,scale=1.):
  A=sp.csr_matrix([[0.,0.,1.,0.],[0.,0.,scale,-q*scale],[0.,0.,0.,1.],[1.,-1.,0.,0.]])
  a=dict(rhs=np.array([old,0.,0.,0.]),sense=np.array(['>','>','>','=']),lo=np.zeros(4),hi=np.array([1.,1.,np.inf,np.inf]));return A,a
 def p(A,a,**kw):return prove(A,a,0,1,{0:2},{1,2,3},{0:3},nfirst=2,rho=2,**kw)
 A,a=fixture();ok('identical_actual_rows',p(A,a)['status']=='PROVEN')
 A,a=fixture(-1);ok('weaker_old_rhs',p(A,a)['status']=='PROVEN')
 A,a=fixture(1e-20);ok('tiny_stronger_old_rejected',p(A,a)['status']=='NOT_PROVEN_NEGATIVE_MARGIN')
 A,a=fixture(scale=512);cert=p(A,a);ok('scaled512',cert['status']=='PROVEN' and replay(A,a,cert,{1,2,3},2,2)['passed'])
 A,a=fixture(q=-1)
 try:p(A,a);raise AssertionError('negative q accepted')
 except ValueError:ok('negative_q',True)
 A,a=fixture()
 for label,choices,K in [('missing_support',{}, {1,2,3}),('circular_support',{0:0},{1,2,3}),('circular_risk',{0:2},{2,3})]:
  try:prove(A,a,0,1,choices,K,{0:3},nfirst=2,rho=2);raise AssertionError(label)
  except ValueError:ok(label,True)
 z=prove(A,a,0,1,{0:None},{1,2,3},{0:3},nfirst=2,rho=2);ok('zero_original_theta_bound',z['status']=='PROVEN')
 a['lo'][3]=-1
 try:prove(A,a,0,1,{0:None},{1,2,3},{0:3},nfirst=2,rho=2);raise AssertionError('bad zero')
 except ValueError:ok('negative_theta_lb',True)
 A,a=fixture();A=A.tolil();A[2,0]=-1;A[2,1]=1;A=A.tocsr()
 ok('point_feasible_not_global',p(A,a)['status']=='NOT_PROVEN_NEGATIVE_MARGIN')
 z=p(A,a,equality_pivots=[(0,3)]);ok('original_equality_closes_gap',z['status']=='PROVEN' and replay(A,a,z,{1,2,3},2,2)['passed'])
 a['sense'][3]='>'
 try:p(A,a,equality_pivots=[(0,3)]);raise AssertionError('non equality')
 except ValueError:ok('nonoriginal_equality_rejected',True)
 A,a=fixture();A=A.tolil();A[2,0]=1;A=A.tocsr();a['hi'][0]=np.inf;ok('unbounded_direction',p(A,a)['status']=='BOUND_INCOMPLETE')
 A,a=fixture();cert=p(A,a);A=A.copy();A.data[0]=2
 try:replay(A,a,cert,{1,2,3},2,2);raise AssertionError('changed coefficient')
 except AssertionError as e:
  if str(e)=='changed coefficient':raise
  ok('changed_stored_coefficient_rejected',True)
 A,a=fixture();A=A.tolil();A[2,0]=1;A=A.tocsr();a['rhs'][2]=.4;ok('no_integer_only_or_point_shortcut',p(A,a)['status']=='NOT_PROVEN_NEGATIVE_MARGIN')
 keep=np.array([1,2,3]);ok('stable_row_mapping',np.array_equal(A[keep].toarray(),np.delete(A.toarray(),0,axis=0)))
 dump('TESTS/NO_OPTIMIZER.json',dict(passed=True,tests=tests,optimizer_calls=0,formal_core_used=True))

def native_test():
 status('TEST_SMALL');results=[]
 for trim in [False,True]:
  env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=gp.Model(env=env)
  x=m.addVar(lb=0,ub=1,vtype='B',obj=.2,name='x');y=m.addVar(lb=0,obj=1,name='y');old=m.addConstr(y>=1-x,name='old');m.addConstr(y>=2-2*x,name='retained');m.update()
  m.setAttr('Start',m.getVars(),[1.,0.]);m.update()
  if trim:m.remove(old);m.update()
  assert m.NumBinVars==1 and m.getAttr('VarName')==['x','y'] and m.getAttr('Start')==[1.,0.]
  obs,err=attrs(m);assert obs['ObjVal'] is None
  lp=m.relax();lp.update();assert m.NumBinVars==1 and lp.NumBinVars==0
  for label,model in [('LP',lp),('MIP',m)]:
   folder='TESTS/NATIVE/'+('B_' if trim else 'A_')+label;params(model,label=='MIP');e,raw=optimize(model,'TEST_SMALL',folder,10,time.monotonic()+30)
   assert e['attributes']['Status']==2 and abs(e['attributes']['ObjVal']-.2)<1e-9
   results.append(dict(folder=folder,objective=e['attributes']['ObjVal'],original_binary=m.NumBinVars,callback=False))
  lp.dispose();m.dispose();env.dispose()
 dump('TESTS/NATIVE_RESULT.json',dict(passed=True,results=results,real_native_calls=4));return 0

def input_and_lineage(cfg):
 src=Path(cfg['source_root']);snap=Path(cfg['snapshot']);root=src.parent
 status('LOAD_FROZEN_REVISION_1')
 identity=js(snap/'IDENTITY.json')
 for x in identity['files']:verify(x)
 A=sm(snap/'A.npz');a=nz(snap/'ARRAYS.npz');names=js(snap/'NAMES.json')
 assert A.shape==(100277,61037) and A.nnz==9604659 and matrix_sha(A)==EXPECTED
 assert sum(a['vtype']=='B')==25770 and len(names['rows'])==100277 and len(names['cols'])==61037
 assert np.isfinite(A.data).all() and len(set(names['rows']))==100277 and len(set(names['cols']))==61037
 assert js(src/'RESTORED_IDENTITY.json')['science']==SCIENCE
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=build(A,a,names,env)
 gate=exact_model(m,A,a,names);dump('ACTUAL_API_IDENTITY.json',gate)
 ident=export('A_FULL',*tk.model_arrays(m));m.dispose();env.dispose();gc.collect()
 tm={int(k):v for k,v in js(root/'THETA_MAP.json').items()};assert len(tm)==256 and set(tm.values())==set(range(60781,61037))
 dump('THETA_MAP.json',tm)
 source_manifest=js(src/'SOURCE_BANK_MANIFEST.json');assert source_manifest['source_science']==SCIENCE
 expected_refs={x['path']:x for x in source_manifest['dependencies']}
 def committed(p):
  p=Path(p)
  if str(p) in expected_refs:verify(expected_refs[str(p)])
  else:source(p)
  return p
 first=BASE/'L08_DRO_R41_NESTED_BENDERS_ROOT/REPAIR_01/identity'
 AF=sp.load_npz(committed(first/'FIRST_A.npz')).tocsr();af=dict(np.load(committed(first/'FIRST_ARRAYS.npz')));fn=read(committed(first/'FIRST_NAMES.json'));fr=read(committed(first/'FIRST_ROWS.json'))
 assert (A[:90768,:60780]-AF).nnz==0 and A[:90768,60780:].nnz==0 and fn==names['cols'][:60780]
 for k in ['rhs','sense']:assert np.array_equal(a[k][:90768],af[k])
 for k in ['cost','lo','hi','vtype']:assert np.array_equal(a[k][:60780],af[k])
 agg=BASE/'L08_DRO400_AGGREGATED_CUTS_INTEGER_REPAIR01_FULL_CYCLE_V1/RESUME_01'
 last=BASE/'L08_DRO400_CACHED_ENVELOPE_REPAIR01_REAL_LP_CYCLE/RESUME_01'
 bank=BASE/'L08_DRO400_CACHED_NONNEGATIVE_ENVELOPE_PILOT_V1'
 cp=read(committed(agg/'CHECKPOINT.json'));smanifest=js(bank/'SUPPORT_MANIFEST.json');support_by_bank={(v['source_bank_id'],v['sid']):v['support_id'] for v in smanifest}
 oldA=sm(last/'B_FROZEN/A.npz');olda=nz(last/'B_FROZEN/ARRAYS.npz');oldn=js(last/'B_FROZEN/NAMES.json')
 assert oldA.shape==(90976,60781) and matrix_sha(oldA)==js(src/'RESTORED_IDENTITY.json')['last_snapshot']['matrix_sha']
 assert (A[:90976,:60781]-oldA).nnz==0 and names['rows'][:90976]==oldn['rows']
 for k in ['rhs','sense']:assert np.array_equal(a[k][:90976],olda[k])
 for k in ['cost','lo','hi','vtype']:assert np.array_equal(a[k][:60781],olda[k])
 support_meta=js(root/'SUPPORT_STORAGE_PROOFS.json');prob_meta=js(root/'PROBABILITY_STORAGE_PROOFS.json');inc=js(src/'INCREMENTS/R00001/MANIFEST.json')
 row_by_support={};row_by_p={};categories={'physical':list(range(90768)),'old_aggregate':[],'old_recombined':[],'theta_original':[],'theta_added':[],'probability':[]};provenance={}
 for block in support_meta:
  sid=block['sid'];d=bank/'BANK'/f'{sid:04d}';bb=sp.load_npz(committed(d/'B.npz')).tocsr();alpha=np.load(committed(d/'ALPHA.npz'))['alpha'];ids=read(committed(d/'IDS.json'));idx={h:i for i,h in enumerate(ids)}
  for rec in block['rows']:
   i=rec['row'];h=rec['id'];ii=idx[h];r=A.getrow(i);actual=-r[:,:60780]
   assert r[0,tm[sid]]==1 and r[:,60780:].nnz==1 and a['sense'][i]=='>' and names['rows'][i]==f'theta_support/{sid}/{h}'
   rhs,pr=tk.compensate(float(alpha[ii]),bb.getrow(ii),actual,a['lo'][:60780],a['hi'][:60780]);assert rhs==a['rhs'][i]
   row_by_support[(sid,h)]=i;categories['theta_original'].append(i)
 for rec in inc['new_supports']:
  st=dict(np.load(verify(rec['stored'])));i=rec['row'];sid=rec['sid'];h=rec['id'];r=A.getrow(i)
  # Stored increment schema is recovered from the actual archive.
  if 'beta' in st:sb=sp.csr_matrix(st['beta'].reshape(1,-1))
  else:sb=sp.csr_matrix((st['beta_data'],st['beta_indices'],st['beta_indptr']),shape=tuple(st['beta_shape']))
  assert (r[:,:60780]+sb).nnz==0 and float(st['alpha'])==a['rhs'][i] and r[0,tm[sid]]==1
  row_by_support[(sid,h)]=i;categories['theta_added'].append(i)
 probs=np.load(committed(src/'BANK/PROBABILITIES.npz'))['p'];pids=read(committed(src/'BANK/PIDS.json'));pbank=dict(zip(pids,probs))
 evnew=js(verify(inc['evaluation']));pbank[inc['new_probability']['full400_p_sha']]=np.load(verify(evnew['risk']['certificate']))['pstar']
 for rec in prob_meta+[inc['new_probability']]:
  i=rec['row'];pid=rec['full400_p_sha'];p=pbank[pid];assert hashlib.sha256(p.tobytes()).hexdigest()==pid
  ra,q,scale,sign=normalized(A,a,i,60780,tm.values());assert ra==0 and sign==1
  assert all(q.get(j,F(0))==F(float(p[sid])) for sid,j in tm.items())
  row_by_p[pid]=i;categories['probability'].append(i)
 aggrows={names['rows'][i]:i for i in range(90768,90815)}
 for item in cp['cuts']:
  i=aggrows['DRO_AGGREGATE/'+item['id']];ev=js(verify(item['evaluation']));ag=dict(np.load(verify(item['aggregate'])));p=ag['pstar'];pid=hashlib.sha256(p.tobytes()).hexdigest()
  choices={sid:row_by_support.get((sid,support_by_bank.get((ev['point_sha'],sid)))) for sid in tm}
  missing=[sid for sid in tm if (ev['point_sha'],sid) not in support_by_bank]
  # Matrix identity above traces actual stored old row to source frozen B and its explicit generated cut ID.
  provenance[i]=dict(kind='old_aggregate',generated_cut_id=item['id'],probability_id=pid,risk=row_by_p.get(pid),choices=choices,missing=missing,source=ref(verify(item['aggregate'])),evaluation=ref(verify(item['evaluation'])),old_frozen_row_sha=row_hash(oldA,olda,i),actual_row_sha=row_hash(A,a,i))
  categories['old_aggregate'].append(i)
 recomb={}
 for p in sorted((last/'ROWS').glob('*/PROOF.json')):
  pr=js(p);recomb['cache/'+pr['row_id']]=(p,pr)
 for i in range(90815,90976):
  path,pr=recomb[names['rows'][i]];z=dict(np.load(verify(pr['actual_coefficients'])));r=A.getrow(i)
  assert r[0,60780]==1 and (r[:,:60780]+sp.csr_matrix(z['beta'][None,:])).nnz==0 and a['rhs'][i]==float(z['alpha'])
  choices={sid:None if pr['support_ids'][sid]=='ZERO' else row_by_support.get((sid,pr['support_ids'][sid])) for sid in tm}
  missing=[sid for sid in tm if pr['support_ids'][sid]!='ZERO' and choices[sid] is None]
  provenance[i]=dict(kind='old_recombined',probability_id=pr['p_id'],risk=row_by_p.get(pr['p_id']),choices=choices,missing=missing,source=ref(path),actual_coefficients=ref(verify(pr['actual_coefficients'])),actual_row_sha=row_hash(A,a,i));categories['old_recombined'].append(i)
 expected={'physical':(90768,368760),'old_aggregate':(47,382170),'old_recombined':(161,2783372),'theta_original':(9016,5835965),'theta_added':(237,222056),'probability':(48,12336)}
 for k,ids in categories.items():assert (len(ids),A[ids].nnz)==expected[k],(k,len(ids),A[ids].nnz)
 allids=sum(categories.values(),[]);assert len(set(allids))==len(allids)==A.shape[0]
 C=sorted(categories['old_aggregate']+categories['old_recombined']);K=sorted(set(range(A.shape[0]))-set(C))
 # Only existing station = sum(routes to station) defining equations are registered, not an arbitrary equality search.
 pivots=[];Ac=AF.tocsc()
 for j,n in enumerate(fn):
  if not n.startswith('station['):continue
  for i in Ac.indices[Ac.indptr[j]:Ac.indptr[j+1]]:
   rr=AF.getrow(int(i))
   if af['sense'][i]=='=' and af['rhs'][i]==0 and AF[i,j]==1 and all(k==j or (fn[k].startswith('route[') and v==-1) for k,v in zip(rr.indices,rr.data)):
    pivots.append((j,int(i)));break
 dump('EQUALITY_REGISTRY.json',dict(pivots=pivots,source_rows=ref(first/'FIRST_ROWS.json'),rule='only exact station-minus-sum-route zero equalities, proof calculation only'))
 dump('PARTITION.json',dict(C=C,K=K,categories=categories,statistics={k:dict(rows=len(v),nnz=A[v].nnz) for k,v in categories.items()}));dump('ROW_LINEAGE.json',provenance)
 dump('INPUT_MANIFEST.json',list(DEPS.values()));dump('SOURCE_IDENTITY.json',dict(snapshot=str(snap),identity=identity,actual_API=gate,science=SCIENCE,source_version=1,python=sys.executable,other_revisions_not_used=True))
 return A,a,names,tm,C,set(K),provenance,pivots,cp

def proof_phase(A,a,names,tm,C,K,lineage,pivots):
 start=time.monotonic();deadline=min(start+1200,read(R/'RUN.json')['deadline_monotonic']-180);dump('PROOF_BUDGET.json',dict(start_monotonic=start,deadline_monotonic=deadline,seconds=1200));rows=[];removed=[]
 for i in C:
  status('EXACT_ROW_PROOF',checked=len(rows),proven=len(removed),candidate=i)
  pr=lineage[i];d=dict(row=i,name=names['rows'][i],kind=pr['kind'],nnz=A.getrow(i).nnz,status='TIME_NOT_CHECKED',margin=None)
  if time.monotonic()<deadline:
   if pr['risk'] is None or pr['missing']:d.update(status='MISSING_LINEAGE',reason=str(pr['missing']))
   else:
    try:
     cert=prove(A,a,i,pr['risk'],pr['choices'],K,tm,pivots);cert.update(lineage=pr,source_math_id=EXPECTED)
     dump('proofs/'+f'{i:06d}.json',cert);d.update(status=cert['status'],margin=cert['margin'],certificate='proofs/'+f'{i:06d}.json')
     if cert['status']=='PROVEN':
      independent=replay(A,a,cert,K);dump('replay/'+f'{i:06d}.json',independent);removed.append(i)
    except (ValueError,KeyError) as exc:d.update(status='MISSING_LINEAGE',reason=str(exc))
  rows.append(d);dump('ROW_PROGRESS.json',dict(checked=len(rows),proven=len(removed),elapsed=time.monotonic()-start))
 dump('ROW_DECISIONS.json',rows);dump('REMOVED_ROWS.json',removed)
 with (R/'ROW_DECISIONS.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['row','name','kind','nnz','status','margin','certificate','reason']);writer.writeheader();writer.writerows(rows)
 dump('PROOF_RESULT.json',dict(checked=sum(v['status']!='TIME_NOT_CHECKED' for v in rows),proven=len(removed),removed_nnz=int(A[removed].nnz) if removed else 0,candidate_count=208,wall=time.monotonic()-start,status_counts={k:sum(v['status']==k for v in rows) for k in sorted({v['status'] for v in rows})},certificates_only_depend_on_K=True))
 return removed

def trim_model(A,a,names,removed):
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=build(A,a,names,env);cons=m.getConstrs();m.remove([cons[i] for i in removed]);m.update();B,ba,bn=tk.model_arrays(m)
 keep=np.array([i for i in range(A.shape[0]) if i not in set(removed)],dtype=int);oldnew=np.full(A.shape[0],-1);oldnew[keep]=np.arange(len(keep));diff=A[keep]-B;diff.eliminate_zeros();assert diff.nnz==0
 for k in a:assert np.array_equal(a[k][keep] if k in ('rhs','sense') else a[k],ba[k]),k
 assert bn['cols']==names['cols'] and bn['rows']==[names['rows'][i] for i in keep] and m.NumBinVars==25770 and A.nnz-B.nnz==A[removed].nnz
 export('B_TRIMMED',B,ba,bn);save('ROW_MAPPING.npz',old_to_new=oldnew,new_to_old=keep);dump('MATRIX_DIFF.json',dict(passed=True,removed=removed,removed_nnz=A.nnz-B.nnz,retained_exact=True,objective_bounds_types_exact=True,stable_row_order=True,new_variables=0,binary=25770))
 m.dispose();env.dispose();gc.collect();return B,ba,bn

def residual_propagation(A,a,x,removed):
 audit,vios=activity(A,a,x);records=[]
 for i in removed:
  cert=read(R/'proofs'/f'{i:06d}.json');sc=decode(cert['old_normalizer']);scale=float(sc);risk=cert['risk']
  def upper(v):return F(float(np.nextafter(float(max(0,v)),np.inf))) if v>0 else F(0)
  allow=upper(vios[risk])/decode(cert['risk_normalizer'])
  for c in cert['choices']:
   if c['row'] is None:allow+=decode(c['weight'])*upper(-float(x[c['theta']]))
   else:allow+=decode(c['weight'])*upper(vios[c['row']])/decode(c['normalizer'])
  for e in cert['equalities']:allow+=abs(decode(e['gamma']))*upper(vios[e['row']])
  for j,v,endpoint,term in cert['box_terms']:
   d=decode(v);allow+=abs(d)*max(F(0),F(float(a['lo'][j]))-F(float(x[j])) if d>0 else F(float(x[j]))-F(float(a['hi'][j])))
  l,u=enclosure(A.getrow(i),x);allow+=upper((u-l)[0])/sc
  allowed=float(np.nextafter(float(allow),np.inf));raw=max(0.,float(vios[i]));passed=F(raw)/sc<=F(allowed)
  records.append(dict(row=i,raw_violation_upper=raw,normalized_violation_upper=raw/scale,propagated_allowance=allowed,passed=passed,strict_feasible_claim=False))
 return dict(audit=audit,rows=records,passed=all(r['passed'] for r in records))

def common_start(A,a,names,tm,cp,removed):
 best=cp['best'];best=js(verify(best)) if isinstance(best,dict) and 'path'in best else best
 if isinstance(best,str):best=js(best)
 assert best['full400'] and best['science']==SCIENCE
 q=np.load(verify(best['Q']));x=q['x'];assert hashlib.sha256(x.tobytes()).hexdigest()==best['point_sha']
 v=np.zeros(A.shape[1]);v[:60780]=x
 for sid,j in tm.items():
  col=A[:,j].tocoo();ids=list(col.row[col.data>0])
  assert all(i>=90976 for i in ids)
  rr=A[ids].copy().tolil();coeff=np.asarray(A[ids,j].toarray()).ravel();rr[:,j]=0;rr=rr.tocsr();lo,_=enclosure(rr,v);need=(np.asarray(a['rhs'][ids],np.longdouble)-lo)/coeff
  v[j]=max(0.,float(np.nextafter(float(np.max(need,initial=0)),np.inf)))
 rcol=A[:,60780].tocoo();ids=rcol.row;rr=A[ids].copy().tolil();rr[:,60780]=0;lo,_=enclosure(rr.tocsr(),v);need=(np.asarray(a['rhs'][ids],np.longdouble)-lo)/rcol.data
 riskupper=best['risk']['risk_U'];required=max(float(np.max(need,initial=0)),riskupper)
 if required>riskupper+1e-4:raise RuntimeError('START_RISK_SUPPORT_CONFLICT')
 v[60780]=float(np.nextafter(required,np.inf));aud,vios=activity(A,a,v);firstmax=float(np.max(vios[:90768],initial=0));integer=float(np.max(abs(v[a['vtype']=='B']-np.rint(v[a['vtype']=='B'])),initial=0))
 assert firstmax<=1e-9 and aud['max_bound_violation']<=1e-9 and integer<=1e-9
 prop=residual_propagation(A,a,v,removed);assert prop['passed'],'START_REMOVED_ROW_PROPAGATION'
 save('COMMON_START.npz',X=v);dump('START_AUDIT.json',dict(source=best,point_sha=best['point_sha'],x_unchanged=True,rows=aud,first_rows_violation=firstmax,integer_violation=integer,removed_rows=prop,rho=v[60780],full400_evaluation_recomputed=False))
 return v,best

def branch(kind,folder,full,removed,wall_limit,native_limit):
 t=time.monotonic();deadline=min(t+wall_limit,read(R/'RUN.json')['deadline_monotonic']-180);dump('ACTIVE_STAGE.json',dict(kind=kind,wall_deadline=deadline,start_monotonic=t));status(kind)
 src=R/folder;A=sp.load_npz(src/'A.npz');a=dict(np.load(src/'ARRAYS.npz'));n=read(src/'NAMES.json')
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();m=build(A,a,n,env);exact_model(m,A,a,n);mip=kind.endswith('MIP');lp=None
 if not mip:lp=m.relax();lp.update();assert m.NumBinVars==25770 and lp.NumBinVars==0;model=lp
 else:model=m;model.setAttr('Start',model.getVars(),np.load(R/'COMMON_START.npz')['X']);model.update()
 assert model.NumSOS==0 and model.NumGenConstrs==0
 params(model,mip);loaded=time.monotonic();end,raw=optimize(model,kind,kind,native_limit,deadline);solved=time.monotonic();cert=dict(LB=None,lower_certificate_verified=False,status='NO_PI',strict_primal_upper=None);prop=None
 if not mip and 'Pi' in raw:
  cert,pi=tk.lagrangian(A,a,raw['Pi'],60780,60780)
  if pi is not None:save(Path(kind)/'CERTIFIED_PI.npz',Pi=pi)
  cert['strict_primal_upper']=None
 if 'X' in raw:
  audit,_=activity(A,a,raw['X']);cert['primal_audit']=audit
  prop=residual_propagation(full[0],full[1],raw['X'],removed);dump(Path(kind)/'REMOVED_ROW_AUDIT.json',prop)
  if mip:
   cert['first_integer_violation']=float(np.max(abs(raw['X'][a['vtype']=='B']-np.rint(raw['X'][a['vtype']=='B'])),initial=0));cert['original_DRO_candidate_status']='UNEVALUATED_IN_ORIGINAL_DRO'
 else:cert['primal_audit']=None
 if mip:
  bd=end['attributes']['ObjBoundC'];cert.update(LB=bd-1e-6 if bd is not None else None,level='SOLVER_SUPPORTED_GUARDED_ONCE',strict_full400_UB=None)
 dump(Path(kind)/'CERTIFICATE.json',cert);certtime=time.monotonic();model.write(str(R/kind/'MODEL.mps.gz'));written=time.monotonic()
 text=(R/kind/'NATIVE.log').read_text();import re
 pres=re.findall(r'Presolve time: ([\d.]+)s',text);root=re.findall(r'Root relaxation:.*?([\d.]+) seconds',text)
 timing=dict(load=loaded-t,optimize=solved-loaded,certificate=certtime-solved,write=written-certtime,total_wall=written-t,presolve_seconds=float(pres[-1]) if pres else None,root_seconds=float(root[-1]) if root else None)
 result=dict(kind=kind,end=end,certificate=cert,removed_rows_passed=prop['passed'] if prop else None,timing=timing,branch_tree_restored=False)
 dump(Path(kind)/'RESULT.json',result)
 if lp is not None:lp.dispose()
 m.dispose();env.dispose();gc.collect();dump('ACTIVE_STAGE.json',dict(kind='BETWEEN_STAGES',wall_deadline=read(R/'RUN.json')['deadline_monotonic']-180));return result

def execute(cfg):
 pure_tests()
 p=R/'TESTS';p.mkdir(exist_ok=True)
 with (p/'NATIVE.stdout').open('xb') as so,(p/'NATIVE.stderr').open('xb') as se:
  child=subprocess.Popen([sys.executable,'-B',__file__,'--test-native'],stdout=so,stderr=se,cwd=R);rc=child.wait()
 dump('TESTS/NATIVE_WAIT.json',dict(actual_parent_wait=True,pid=child.pid,returncode=rc));assert rc==0 and read(p/'NATIVE_RESULT.json')['passed']
 dump('CODE_SHA.json',[ref(p) for p in sorted((R/'tools').glob('*.py'))])
 A,a,n,tm,C,K,lineage,pivots,cp=input_and_lineage(cfg)
 removed=proof_phase(A,a,n,tm,C,K,lineage,pivots)
 if not removed:return dict(status='NO_PROVEN_REDUNDANCY',LP='NOT_RUN_NO_PROVEN_ROWS',MIP='NOT_RUN_NO_PROVEN_ROWS')
 B,ba,bn=trim_model(A,a,n,removed)
 # Independent standalone replay uses actual payloads, not the generator's pass flags.
 p=subprocess.run([sys.executable,'-B',str(R/'tools/replay_certificates.py'),str(R)],capture_output=True,text=True)
 (R/'REPLAY.stdout').write_text(p.stdout);(R/'REPLAY.stderr').write_text(p.stderr);dump('REPLAY_WAIT.json',dict(actual_parent_wait=True,returncode=p.returncode));assert p.returncode==0
 independent=json.loads(p.stdout);dump('INDEPENDENT_REPLAY.json',independent)
 start,best=common_start(A,a,n,tm,cp,removed)
 archived=[]
 for pth in sorted((Path(cfg['source_root'])/'MASTER/0000_R00001').glob('*.npz')):
  z=nz(pth)
  for key in ('X','x'):
   if key in z and z[key].shape==(61037,):
    audit=residual_propagation(A,a,z[key],removed);archived.append(dict(source=ref(pth),key=key,audit=audit));assert audit['passed'],'ARCHIVED_RESIDUAL_PROPAGATION'
 dump('ARCHIVED_POINT_REGRESSION.json',dict(common_start=read(R/'START_AUDIT.json'),archived=archived,missing_archived_LP='No complete LP X in CACHE_LP/SOLVED',note='point activity is not a proof of redundancy'))
 if read(R/'RUN.json')['deadline_monotonic']-time.monotonic()<1440+180:return dict(status='LP_COMPARISON_INCOMPLETE',LP='NOT_RUN_BUDGET',MIP='NOT_RUN_BUDGET')
 la=branch('A_LP','A_FULL',(A,a,n),removed,720,600);lb=branch('B_LP','B_TRIMMED',(A,a,n),removed,720,600)
 def lpok(r):
  at=r['end']['attributes'];ce=r['certificate'];audit=ce.get('primal_audit')
  return at['Status']==2 and ce.get('lower_certificate_verified') and audit is not None and max(audit.values())<=1e-9 and r['removed_rows_passed'] and ce['LB']<=at['ObjVal']+1e-9 and at['ObjVal']-ce['LB']<=1e-5
 gate=lpok(la) and lpok(lb) and abs(la['end']['attributes']['ObjVal']-lb['end']['attributes']['ObjVal'])<=1e-5
 dump('LP_GATE.json',dict(passed=bool(gate),A=la,B=lb,objective_tolerance=1e-5,original_residual_tolerance=1e-9))
 if not gate:return dict(status='LP_COMPARISON_INCOMPLETE_OR_NUMERICAL_DIAGNOSTIC',LP=[la,lb],MIP='NOT_RUN_LP_GATE')
 if read(R/'RUN.json')['deadline_monotonic']-time.monotonic()<3960+180:return dict(status='LP_COMPARISON_COMPLETE',LP=[la,lb],MIP='NOT_RUN_BUDGET')
 ma=branch('A_MIP','A_FULL',(A,a,n),removed,1980,1800);mb=branch('B_MIP','B_TRIMMED',(A,a,n),removed,1980,1800)
 outcome='INCONCLUSIVE';aa=ma['end']['attributes'];bb=mb['end']['attributes']
 if ma['removed_rows_passed'] and mb['removed_rows_passed']:
  if aa['Status']==bb['Status']==2 and abs(aa['ObjVal']-bb['ObjVal'])<=1e-5:outcome='PROMISING_FIXED_MODEL_RUNTIME' if ma['timing']['total_wall']>=5 and mb['timing']['total_wall']<=.8*ma['timing']['total_wall'] else 'NO_OBSERVED_NET_GAIN'
  elif aa['Status']==bb['Status']==9 and ma['certificate']['LB'] is not None and mb['certificate']['LB'] is not None:outcome='PROMISING_FIXED_BUDGET_BOUND' if mb['certificate']['LB']>=ma['certificate']['LB']+1 else 'NO_OBSERVED_NET_GAIN'
 return dict(status=outcome,LP=[la,lb],MIP=[ma,mb],original_DRO_UB=best['UB_audited'],new_full400_UB=None)

def main():
 if '--test-native' in sys.argv:return native_test()
 p=argparse.ArgumentParser();p.add_argument('--source-snapshot',required=True);p.add_argument('--source-root',required=True);p.add_argument('--execution-root',required=True);p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');args=p.parse_args();assert args.execute and Path(args.execution_root).resolve()==R
 cfg=read(args.config);assert cfg['snapshot']==args.source_snapshot and cfg['source_root']==args.source_root and gp.gurobi.version()==(12,0,3)
 rc=0;result={}
 try:result=execute(cfg)
 except BaseException as e:rc=2;result=dict(status='EXECUTION_OR_VALIDATION_FAILED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),remaining_stages='NOT_RUN')
 finally:
  result.update(wall=time.monotonic()-read(R/'RUN.json')['start_monotonic'],production_signals_sent=[],production_modified=False,new_recourse_calls=0,new_probability_calls=0,original_science_unchanged=True)
  dump('INPUT_MANIFEST.json',list(DEPS.values()));dump('FINAL_RESULT.json',result);dump('END.json',dict(returncode=rc,epoch=time.time(),status=result['status']));dump('ACTIVE_STAGE.json',dict(kind='FINISHED',wall_deadline=read(R/'RUN.json')['deadline_monotonic']))
 return rc
if __name__=='__main__':raise SystemExit(main())
