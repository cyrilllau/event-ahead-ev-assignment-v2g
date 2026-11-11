from aggregate_core import *
import ast
from csr_reader import read_csr

def validate_resume():
 old=Path(CTX.config['resume_from']);oldtest=read(old/'TEST_ACCEPTANCE.json');assert oldtest['passed'];rt.verify_ref(oldtest['result'])
 for r in oldtest['source_frozen']['files']:rt.verify_ref(r)
 # Only the historical reader and its idempotent certificate reuse changed in formal.
 def functions(p):
  tree=ast.parse(p.read_text());return {v.name:ast.dump(v,include_attributes=False) for v in tree.body if isinstance(v,(ast.FunctionDef,ast.ClassDef))}
 before=functions(old/'tools/formal.py');after=functions(OUT/'tools/formal.py')
 assert set(before)==set(after)
 assert all(before[k]==after[k] for k in before if k!='historical_bound')
 for name in ['aggregate_core.py','actual_tests.py','source_prepare.py','control_policy.py','gap_phase_reference.py']:
  assert sha(old/'tools'/name)==sha(OUT/'tools'/name)
 for p in (old/'tools/lifecycle').glob('*.py'):assert sha(p)==sha(OUT/'tools/lifecycle'/p.name)
 folder=OUT/'CSR_PREFLIGHT';folder.mkdir()
 a=sp.csr_matrix([[1.,0.,2.],[0.,3.,0.]])
 np.savez(folder/'CUSTOM.npz',data=a.data,indices=a.indices,indptr=a.indptr,shape=a.shape);sp.save_npz(folder/'SCIPY.npz',a)
 for n in ['CUSTOM.npz','SCIPY.npz']:
  got=read_csr(folder/n);assert np.array_equal(got.data,a.data) and np.array_equal(got.indices,a.indices) and np.array_equal(got.indptr,a.indptr)
 np.savez(folder/'BAD.npz',data=a.data,indices=a.indices,indptr=np.array([0,3,2]),shape=a.shape)
 try:read_csr(folder/'BAD.npz')
 except ValueError:pass
 else:raise AssertionError('MALFORMED_CSR_NOT_REJECTED')
 import formal
 e.PREFLIGHT=True
 try:lb=formal.historical_bound()
 finally:e.PREFLIGHT=False
 result=dict(passed=True,reader_cases=3,actual_historical_certificate=ref(OUT/'HISTORICAL_LB_REPLAY.json'),LB=lb,optimizer_calls=0,prior_native_tests_repeated=False,unchanged_formal_functions=True,previous_test=ref(old/'TEST_ACCEPTANCE.json'))
 write(folder/'RESULT.json',result,True)
 write(OUT/'TEST_ACCEPTANCE.json',dict(passed=True,result=ref(folder/'RESULT.json'),reused_test=ref(old/'TEST_ACCEPTANCE.json'),source_frozen=read(OUT/'SOURCE_FROZEN.json')),True)
