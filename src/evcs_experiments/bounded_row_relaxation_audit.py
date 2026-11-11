"""Independent exact-array proof of a bounded tiny-row outer relaxation."""
from fractions import Fraction
import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256


def audit_bounded_row_relaxation(original, relaxed, record, *, allowed_rows,
        allowed_columns, threshold=1e-13, deadline, bound_witnesses=None, witness_rows=None):
    def need(ok, message):
        if not bool(ok): raise ValueError(message)
    def alive():
        if not math.isfinite(deadline) or time.monotonic()>=deadline: raise TimeoutError('Relaxation audit deadline')
    def exact(v): return Fraction.from_float(float(v))
    def fraction(v):
        return None if v is None else dict(numerator=str(v.numerator),denominator=str(v.denominator))
    def number(v): return float(v) if math.isfinite(v) else ('inf' if v>0 else '-inf')
    def span(v,n):
        need(isinstance(v,(tuple,list)) and len(v)==2 and all(type(x) is int for x in v)
             and 0<=v[0]<=v[1]<=n,'Invalid allowed range')
        return tuple(v)
    alive();need(threshold==1e-13,'Threshold changed')
    A,B=csr_matrix(original['matrix']),csr_matrix(relaxed['matrix'])
    need(A.shape==B.shape and A.has_canonical_format and B.has_canonical_format
         and np.isfinite(A.data).all() and np.isfinite(B.data).all(),'Invalid coefficient arrays')
    m,n=A.shape; ar=span(allowed_rows,m);ac=span(allowed_columns,n)
    for key in ('c','lower_bounds','upper_bounds','integrality'):
        need(np.array_equal(original[key],relaxed[key]),'Objective/domain changed: '+key)
        need(np.asarray(original[key]).shape==(n,),'Invalid column axis')
    need(math.isfinite(original['objective_constant_usd']) and original['objective_constant_usd']==relaxed['objective_constant_usd'],'Constant changed')
    need(np.isfinite(original['c']).all(),'Nonfinite objective')
    lo=np.asarray(original['lower_bounds']);hi=np.asarray(original['upper_bounds']).copy()
    L,U=np.asarray(original['row_lower']),np.asarray(original['row_upper'])
    l,u=np.asarray(relaxed['row_lower']),np.asarray(relaxed['row_upper'])
    need(all(v.shape==(m,) for v in (L,U,l,u)),'Invalid row axis')
    for left,right in ((lo,hi),(L,U),(l,u)):
        need(not np.isnan(left).any() and not np.isnan(right).any() and np.all(left<=right)
             and not np.isposinf(left).any() and not np.isneginf(right).any(),'Invalid bounds')
    body=dict(record);digest=body.pop('relaxation_sha256')
    need(canonical_hash(body)==digest and body['kind']=='BOUNDED_TINY_ROW_RELAXATION_V1','Producer record changed')
    oh,rh=coefficient_sha256(original),coefficient_sha256(relaxed)
    need(body['original_coefficient_sha256']==oh and body['relaxed_coefficient_sha256']==rh,'Coefficient identity differs')
    need(body['allowed_rows']==list(ar) and body['allowed_columns']==list(ac) and body['threshold']==threshold,'Whitelist changed')
    witnesses={} if bound_witnesses is None else bound_witnesses
    need(isinstance(witnesses,dict),'Invalid witness map')
    wr=None if witness_rows is None else span(witness_rows,m)
    need(body.get('witness_rows')==(None if wr is None else list(wr)),'Witness range changed')
    saved=body.get('bound_witnesses',[])
    need(len(saved)==len(witnesses),'Witness inventory differs')
    for item,(col,row) in zip(saved,sorted(witnesses.items())):
        alive();need(type(col) is int and type(row) is int and ac[0]<=col<ac[1]
                     and wr is not None and wr[0]<=row<wr[1]<=ar[0],'Invalid original-prefix witness')
        need(math.isfinite(U[row]),'Unbounded witness row')
        target=float(A[row,col]);need(target>0,'Invalid witness target sign')
        other=Fraction(0)
        for j,a in zip(A.getrow(row).indices,A.getrow(row).data):
            if j==col or a==0: continue
            endpoint=lo[j] if a>0 else original['upper_bounds'][j]
            need(math.isfinite(endpoint),'Unbounded witness other contribution')
            other+=exact(a)*exact(endpoint)
        bound=(exact(U[row])-other)/exact(target)
        value=item['upper_bound']
        need(math.isfinite(value) and exact(value)>=bound and value>=lo[col],'Unsafe witness upper bound')
        need(item==dict(column=col,source_row=row,target_coefficient=target,upper_bound_exact=fraction(bound),upper_bound=value),'Witness evidence differs')
        hi[col]=min(hi[col],value)
    expected=[];rows=[]
    for i in range(m):
        alive();old=dict(zip(A.getrow(i).indices,A.getrow(i).data));new=dict(zip(B.getrow(i).indices,B.getrow(i).data))
        removed=[j for j,a in old.items() if 0<abs(a)<threshold]
        minsum=Fraction(0) if math.isfinite(U[i]) else None
        maxsum=Fraction(0) if math.isfinite(L[i]) else None
        for j in removed:
            a=old[j];need(ar[0]<=i<ar[1] and ac[0]<=j<ac[1],'Tiny outside whitelist')
            mn,mx=(lo[j],hi[j]) if a>0 else (hi[j],lo[j])
            need((minsum is None or math.isfinite(mn)) and (maxsum is None or math.isfinite(mx)),'Unbounded deleted contribution')
            p=None if minsum is None else exact(a)*exact(mn)
            q=None if maxsum is None else exact(a)*exact(mx)
            if p is not None:minsum+=p
            if q is not None:maxsum+=q
            expected.append(dict(row=i,column=int(j),coefficient=float(a),min_contribution_exact=fraction(p),max_contribution_exact=fraction(q)))
        desired={j:a for j,a in old.items() if j not in removed}
        need(desired==new,'Missing/extra/changed coefficient')
        if not removed:
            need(l[i]==L[i] and u[i]==U[i],'Unaffected RHS changed');continue
        need((l[i]==L[i] if maxsum is None else math.isfinite(l[i]) and exact(l[i])<=exact(L[i])-maxsum),'Lower RHS strengthens row')
        need((u[i]==U[i] if minsum is None else math.isfinite(u[i]) and exact(u[i])>=exact(U[i])-minsum),'Upper RHS strengthens row')
        rows.append(dict(row=i,original_lower=number(L[i]),original_upper=number(U[i]),relaxed_lower=number(l[i]),relaxed_upper=number(u[i]),min_sum_exact=fraction(minsum),max_sum_exact=fraction(maxsum)))
    need(body['removed_entries']==expected and body['rows']==rows,'Deletion/proof inventory differs')
    need(body['removed_entry_count']==len(expected) and body['changed_row_count']==len(rows),'Counts differ')
    alive()
    out=dict(kind='INDEPENDENT_BOUNDED_ROW_RELAXATION_AUDIT_V1',passed=True,
        original_coefficient_sha256=oh,relaxed_coefficient_sha256=rh,producer_relaxation_sha256=digest,
        allowed_rows=list(ar),allowed_columns=list(ac),threshold=threshold,removed_entry_count=len(expected),
        changed_row_count=len(rows),bound_witness_count=len(witnesses),objective_and_domain_unchanged=True,
        exact_outward_fraction_proof=True,bound_scope='Relaxed auxiliary minimization lower bound only',
        solver_calls=0,scientific_acceptance=False)
    out['audit_sha256']=canonical_hash(out)
    return out
