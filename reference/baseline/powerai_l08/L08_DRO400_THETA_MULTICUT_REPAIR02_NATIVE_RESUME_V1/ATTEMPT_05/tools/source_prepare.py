from aggregate_core import *
import shutil,inspect,ast
from control_policy import schema_adapter

def prepare():
 done=OUT/'SOURCE_LEDGER.json'
 if done.exists():
  z=read(done)
  for r in z['files']:rt.verify_ref(r)
  return z
 source=CTX.source_root;records={}
 def verify(r):
  path=rt.verify_ref(r);records[str(path)]=ref(path);return Path(path)
 inherited=source/'ATTEMPT_003/INHERITED_CHECKS.json'
 verify(ref(inherited))
 for r in read(inherited)['files']:
  p=verify(r);target=OUT/('preflight/'+p.name if p.parent.name=='preflight' else p.name);target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha(target)==r['sha256']
  else:shutil.copyfile(p,target)
 for r in read(OUT/'FIRST_IDENTITY.json')['files']:verify(r)
 A,a,names=first();assert A.shape==(90768,60780) and sum(a['vtype']=='B')==25770 and sum(a['vtype']=='C')==35010
 assert read(OUT/'FIRST_IDENTITY.json')['science']==SCIENCE
 ac=read(ARO/'CONFIG.json');assert ac['identity']==SCIENCE;verify(ref(ARO/'CONFIG.json'))
 ident=read(R36/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json');assert SCIENCE in ident.values();verify(ref(R36/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json'))
 # Verify every referenced immutable payload, not just the previous PASS marker.
 for r in read(source/'EXTERNAL_DEPENDENCY_MANIFEST.json')['entries']:verify(r)
 oracles=read(OUT/'ORACLE_REFERENCES.json');assert len(oracles)==400
 for r in oracles:
  CTX.check(True)
  payload=verify(r['source'])
  for f in read(payload)['files']:verify(f)
  if r['sid']%25==0:write(OUT/'STATUS.json',dict(phase='SOURCE_PAYLOAD_SHA',completed=r['sid'],total=400))
 src=inspect.getsource(orig.add_recourse);tree=ast.parse(src)
 # Common constructor proof is bound to its exact source and original tested builder.
 assert 'objective=sum(parts)' in src and 'vtype=' not in src
 proof=dict(d_zero=True,kappa_zero=True,continuous=True,source=ref(Path(inspect.getsourcefile(orig.add_recourse))),basis='objective=sum(parts), continuous default addVars; mandatory two-real-sid matrix split regression before execute')
 verify(proof['source'])
 adapters=[]
 for r in oracles:
  folder=Path(r['cache']['path']).parent;ar=np.load(folder/'ARRAYS.npz');W=sp.load_npz(folder/'W.npz');T=sp.load_npz(folder/'T.npz');ca=read(verify(r['cache']))
  ok=(T.shape==(W.shape[0],60780) and W.shape[1]==len(ar['c']) and np.array_equal(ar['xlo'],a['lo']) and np.array_equal(ar['xhi'],a['hi']) and np.all(ar['c']>=0))
  adapter=schema_adapter(ca,proof,dict(passed=bool(ok),W=ref(folder/'W.npz'),T=ref(folder/'T.npz'),arrays=ref(folder/'ARRAYS.npz')));adapters.append(dict(sid=r['sid'],**adapter));del W,T,ar
 write(OUT/'SID_SCHEMA_ADAPTER_CERTIFICATE.json',dict(status='SOURCE_AND_MATRIX_VERIFIED_REAL_CONSTRUCTOR_TEST_REQUIRED',records=adapters),True)
 eq=read(OUT/'SCENARIO_EQUIVALENCE.json');assert len(eq['zeros'])==144 and len(eq['groups'])==256 and all(len(g)==1 for g in eq['groups']);assert set(eq['zeros']+[g[0] for g in eq['groups']])==set(range(400))
 # Zero witnesses from the already executed inventory are immutable dependencies.
 zero_files=[R36/'UNIVERSAL_ZERO_PROOFS.json',source/'tools/preparation.py']
 for sid in eq['zeros']:
  zr=R36/'evaluations/ZERO'/f'{sid:04d}';ce=read(zr/'CERTIFICATE.json');verify(dict(path=str(zr/'PRIMAL.npz'),sha256=ce['primal_sha']));verify(ref(zr/'CERTIFICATE.json'))
 for p in zero_files:verify(ref(p))
 for p in (source/'preflight').rglob('*.json'):verify(ref(p))
 P=read(OUT/'ORIGINAL_P_IDENTITY.json');assert P['moment_identity']==CTX.config['P_identity'] and P['effective_rows']==45139 and P['raw_features']==322311
 verify(P['source']);dd=b.moment_data();assert dd['a']['phi'].shape==(45139,400)
 p0=b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz';verify(ref(p0));check=b.probability_check(np.load(p0)['p']);assert check['passed']
 x=np.load(OUT/'INITIAL_POINT.npz')['values'];assert hashlib.sha256(x.tobytes()).hexdigest()=='951cc92fb17df719e6a1d224095b39d0e51a23d47a2b56289b2a84cbf1eb2f91'
 assert np.array_equal(plan_point(read(OUT/'INITIAL_PLAN.json')['plan']),x)
 z=dict(passed=True,files=list(records.values()),initial_F=float(a['cost']@x),initial_UB_not_inherited=True,all400_payload_verified=True,original_P_check=check,source_read_only=True,epoch=time.time())
 write(done,z,True);return z
