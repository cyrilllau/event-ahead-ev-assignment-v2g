"""Independent original-coefficient audit of the streamed SAA transformation.

Audits exact elimination and expectation coefficients, one original scenario at
 a time. This does not solve the MILP or certify a returned plan's full Q.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import hstack,vstack,csr_matrix
from .evaluation_lock import file_reference
from .random_streams import canonical_hash


def _need(value,message):
    if not bool(value):raise ValueError(message)


def audit_streaming_saa(model,recourses,*,deadline):
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('SAA original audit shared deadline exhausted')
    check_time();manifest=model.manifest
    _need(canonical_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})==manifest['manifest_sha256'],
          'SAA manifest semantic identity changed')
    _need(json.loads((model.root/'manifest.json').read_text())==manifest,'SAA original manifest differs')
    for ref in manifest['numeric_artifacts']:
        _need(file_reference(ref['path'])==ref,'SAA numeric bytes changed')
    first=model.first_stage;n=len(first.c);nr=first.A.shape[0]
    _need(len(recourses)==len(manifest['scenarios'])==len(manifest['scenario_ids']),'SAA support incomplete')
    _need(manifest['first_stage_sha256']==first.matrix_sha256,'SAA first-stage identity differs')
    _need(model.A.shape==tuple(manifest['shape']),'SAA matrix shape differs')
    _need((model.A[:nr,:n]!=first.A).nnz==0 and model.A[:nr,n:].nnz==0,'SAA shared first-stage rows changed')
    for key in ('row_lower','row_upper'):
        _need(np.array_equal(getattr(model,key)[:nr],getattr(first,key)),'SAA first-stage row bounds differ')
    for key in ('lower_bounds','upper_bounds','c','integrality'):
        _need(np.array_equal(getattr(model,key)[:n],getattr(first,key)),'SAA first-stage column data differ')
    pi=np.asarray(manifest['empirical_probabilities']);constants=np.asarray(manifest['scenario_cost_constants_usd'])
    _need(pi.shape==constants.shape==(len(recourses),) and np.isfinite(pi).all() and (pi>=0).all()
        and abs(pi.sum()-1)<1e-8 and np.isfinite(constants).all(),'SAA law/constants differ')
    _need(len(set(manifest['scenario_ids']))==len(recourses),'SAA duplicate support')
    _need(manifest['domain']!='formal' or len(recourses)==160 and np.array_equal(pi,np.full(160,1/160)),
          'Formal SAA audit requires exact uniform160 law')
    _need(model.objective_constant_usd==manifest['objective_constant_usd']==float(pi@constants),'SAA constant differs')
    nextrow,nextcol=nr,n
    for i,item in enumerate(manifest['scenarios']):
        check_time();template=recourses[i]
        _need(item['index']==i and item['scenario_id']==manifest['scenario_ids'][i] and
            item['scenario_sha256']==manifest['scenario_sha256s'][i]==template.scenario_sha256 and
            item['template_sha256']==template.template_sha256 and
            template.first_stage.matrix_sha256==first.matrix_sha256,'SAA original template/support differs')
        for ref in item['artifacts']:_need(file_reference(ref['path'])==ref,'SAA shard bytes changed')
        folder=Path(item['mapping_reference']['path']).parent
        proof=json.loads((folder/'zero_proof.json').read_text())
        ci=np.asarray(proof['retained_columns'],dtype=int);ri=np.asarray(proof['retained_rows'],dtype=int)
        a=vstack([hstack([-template.Bx_ub,-template.Br_ub,template.A_ub],format='csr'),
                  hstack([-template.Bx_eq,-template.Br_eq,template.A_eq],format='csr')],format='csr')
        a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
        rl=np.r_[np.full(len(template.b_ub),-np.inf),template.b_eq];ru=np.r_[template.b_ub,template.b_eq]
        lo=np.r_[first.lower_bounds,template.lower_bounds];up=np.r_[first.upper_bounds,template.upper_bounds]
        zero=set(np.flatnonzero((lo==0)&(up==0))) - set(range(n))
        # Independently derive admissible zeros from the original row system.
        # No producer proof flag is used as evidence of mathematical validity.
        while True:
            upper_zero=set(np.flatnonzero(up==0));lower_zero=set(np.flatnonzero(lo==0))
            for r in range(a.shape[0]):
                if r%2048==0:check_time()
                k0,k1=a.indptr[r:r+2];remaining=[k for k in range(k0,k1) if int(a.indices[k]) not in zero]
                if len(remaining)!=1:continue
                k=remaining[0];j=int(a.indices[k]);coef=a.data[k]
                if (coef>0 and ru[r]==0) or (coef<0 and rl[r]==0):upper_zero.add(j)
                if (coef>0 and rl[r]==0) or (coef<0 and ru[r]==0):lower_zero.add(j)
            more={j for j in upper_zero & lower_zero if j>=n and lo[j]<=0<=up[j]}-zero
            if not more:break
            zero.update(more);check_time()
        expected_cols=np.array([j for j in range(a.shape[1]) if j not in zero])
        _need(np.array_equal(ci,expected_cols),'SAA eliminated column lacks original zero proof')
        b=a[:,ci].tocsr()
        expected_rows=np.flatnonzero(~((np.diff(b.indptr)==0)&(rl<=0)&(ru>=0)))
        _need(np.array_equal(ri,expected_rows),'SAA omitted row is not redundant')
        b=b[ri].tocsr();r0,r1=item['row_slice'];c0,c1=item['column_slice']
        _need(r0==nextrow and c0==nextcol and r1-r0==len(ri) and c1-c0==len(ci)-n,'SAA blocks have gaps/overlap')
        shifted=np.asarray(b.indices,dtype=np.int64).copy();mask=shifted>=n;shifted[mask]+=c0-n
        expected=csr_matrix((b.data,shifted,b.indptr),shape=(len(ri),model.A.shape[1]))
        _need((model.A[r0:r1]!=expected).nnz==0,'SAA original physical block coefficients differ')
        _need(np.array_equal(model.row_lower[r0:r1],rl[ri]) and np.array_equal(model.row_upper[r0:r1],ru[ri]),'SAA physical row bounds differ')
        local=ci[n:]-n
        _need(np.array_equal(np.load(folder/'recourse_columns.npy',allow_pickle=False),local),'SAA witness reconstruction differs')
        for key in ('lower_bounds','upper_bounds'):
            _need(np.array_equal(getattr(model,key)[c0:c1],getattr(template,key)[local]),'SAA physical column bounds differ')
        _need(np.array_equal(model.c[c0:c1],pi[i]*template.c[local]) and np.all(model.integrality[c0:c1]==0),'SAA expectation objective/integrality differs')
        nextrow,nextcol=r1,c1
    _need((nextrow,nextcol)==model.A.shape,'SAA extra rows/columns')
    check_time()
    return dict(status='PASS_INDEPENDENT_SAA_ORIGINAL_ASSEMBLY_AUDIT',scenario_count=len(recourses),
        manifest_reference=file_reference(model.root/'manifest.json'),
        zero_elimination_checked_from_original_rows=True,original_expectation_coefficients_checked=True,
        solver_bound_audited=False,full_support_Q_audited=False,scientific_spec_pass=False,formal_OOS=False)
