"""Independently replay a saved zero-elimination proof against ORIGINAL rows."""
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.random_streams import canonical_hash


def audit_zero_reduction(matrix, row_lower, row_upper, lower, upper, proof):
    def need(ok,label):
        if not ok:raise ValueError(label)
    d=dict(proof);digest=d.pop('sha256')
    need(canonical_hash(d)==digest,'Reduction proof checksum')
    need(d['kind']=='exact_zero_singleton_v1' and d['coefficient_threshold'] is None,'Reduction kind')
    a=csr_matrix(matrix,copy=True);a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
    rl,ru,lo,hi=map(np.asarray,(row_lower,row_upper,lower,upper))
    zero=(lo==0)&(hi==0)
    steps=d['proof_steps'];need(steps[0]['columns']==np.flatnonzero(zero).tolist(),'Initial zero bounds')
    for step in steps[1:]:
        need(step['kind']=='homogeneous_singleton','Proof step kind')
        new=np.asarray(step['columns'],dtype=int)
        new_set=set(new)
        need(len(new)>0 and len(np.unique(new))==len(new) and np.all((new>=0)&(new<a.shape[1])),'New zero axis')
        need(not np.any(zero[new]) and np.all(lo[new]<=0) and np.all(hi[new]>=0),'New zeros feasible under original bounds')
        allowed=np.flatnonzero(~zero);rows=np.asarray(step['witness_rows'],dtype=int)
        need(np.all((rows>=0)&(rows<a.shape[0])),'Witness axis')
        b=a[rows][:,allowed].tocsr()
        need(np.all(np.diff(b.indptr)==1),'Witness not singleton after PREVIOUS steps')
        upper_zero=set(np.flatnonzero(hi==0));lower_zero=set(np.flatnonzero(lo==0))
        for k,row in enumerate(rows):
            j=allowed[b.indices[b.indptr[k]]];v=b.data[b.indptr[k]]
            need(j in new_set,'Witness outside claimed zeros')
            if (v>0 and ru[row]==0) or (v<0 and rl[row]==0):upper_zero.add(j)
            if (v>0 and rl[row]==0) or (v<0 and ru[row]==0):lower_zero.add(j)
        need(new_set<=upper_zero & lower_zero,'Zero not implied by both original directions')
        zero[new]=True
    cols=np.flatnonzero(~zero)
    need(d['zero_columns']==np.flatnonzero(zero).tolist() and d['retained_columns']==cols.tolist(),'Column lift mismatch')
    b=a[:,cols].tocsr();empty=np.diff(b.indptr)==0
    keep=~(empty & (rl<=0)&(ru>=0));rows=np.flatnonzero(keep)
    need(d['retained_rows']==rows.tolist(),'Dropped a nonredundant original row')
    b=b[keep]
    need(d['reduced_shape']==list(b.shape) and d['reduced_nnz']==b.nnz,'Reduced dimensions')
    return dict(status='PASS_EXACT_ZERO_PROOF',zero_columns=int(zero.sum()),retained_rows=len(rows),retained_columns=len(cols),
                original_rows=a.shape[0],original_columns=a.shape[1],proof_sha256=digest)
