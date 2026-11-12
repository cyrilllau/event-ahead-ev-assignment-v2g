"""Exact certificates on actual binary64 stored rows; no optimizer calls."""
from fractions import Fraction as F
import hashlib, math
import numpy as np

def frac(x):
    if isinstance(x,F): return x
    return F(float(x))

def encode(x): return [str(x.numerator),str(x.denominator)]
def decode(x): return F(int(x[0]),int(x[1]))

def row_hash(A,a,i):
    r=A.getrow(i)
    return hashlib.sha256(r.indices.astype('<i8').tobytes()+r.data.astype('<f8').tobytes()+np.asarray([a['rhs'][i]],dtype='<f8').tobytes()+str(a['sense'][i]).encode()).hexdigest()

def normalized(A,a,i,pivot,allowed):
    r=A.getrow(i);d={int(j):frac(v) for j,v in zip(r.indices,r.data)}
    if a['sense'][i] not in ('>','<'):raise ValueError('NOT_INEQUALITY')
    sign=1 if a['sense'][i]=='>' else -1
    p=sign*d.get(pivot,F(0))
    if p<=0:raise ValueError('NONPOSITIVE_PIVOT')
    if set(d)-{pivot}-set(allowed):raise ValueError('UNEXPECTED_COLUMN')
    return sign*frac(a['rhs'][i])/p,{j:-sign*v/p for j,v in d.items() if j!=pivot},p,sign

def box_margin(alpha,beta,lo,hi):
    margin=alpha;terms=[]
    for j,v in sorted(beta.items()):
        if not v:continue
        endpoint=float(lo[j] if v>0 else hi[j])
        if not math.isfinite(endpoint) or abs(endpoint)>=1e100:return None,terms
        term=v*frac(endpoint);margin+=term
        terms.append([j,encode(v),endpoint,encode(term)])
    return margin,terms

def validate_dependencies(candidate,risk,choices,K,theta_map,a,equalities=()):
    if candidate in K:raise ValueError('CANDIDATE_NOT_IN_C')
    if risk not in K:raise ValueError('CIRCULAR_RISK')
    for sid,i in choices.items():
        if sid not in theta_map:raise ValueError('MISSING_THETA')
        if i is None:
            if a['lo'][theta_map[sid]]<0:raise ValueError('ZERO_THETA_UNPROVEN')
        elif i not in K:raise ValueError('CIRCULAR_SUPPORT')
    for i,g in equalities:
        if i not in K or i>=90768 or a['sense'][i]!='=':raise ValueError('NOT_ORIGINAL_EQUALITY')

def prove(A,a,candidate,risk,choices,K,theta_map,equality_pivots=(),nfirst=60780,rho=60780):
    validate_dependencies(candidate,risk,choices,K,theta_map,a)
    oa,ob,op,osign=normalized(A,a,candidate,rho,range(nfirst))
    na,q,rp,rsign=normalized(A,a,risk,rho,theta_map.values())
    if any(v<0 for v in q.values()):raise ValueError('NEGATIVE_Q')
    inverse={j:s for s,j in theta_map.items()};nb={};used=[]
    for j,w in q.items():
        sid=inverse[j]
        if sid not in choices:raise ValueError('MISSING_SUPPORT')
        i=choices[sid]
        if i is None:used.append(dict(sid=sid,theta=j,row=None,weight=encode(w),bound=float(a['lo'][j])));continue
        sa,sb,sp,ss=normalized(A,a,i,j,range(nfirst))
        na+=w*sa
        for k,v in sb.items():nb[k]=nb.get(k,F(0))+w*v
        used.append(dict(sid=sid,theta=j,row=i,weight=encode(w),normalizer=encode(sp),sign=ss,sha256=row_hash(A,a,i)))
    da=na-oa;db=dict(nb)
    for j,v in ob.items():db[j]=db.get(j,F(0))-v
    db={j:v for j,v in db.items() if v};initial,terms=box_margin(da,db,a['lo'],a['hi']);eq=[]
    # At most the registered station-definition identities; no search/optimization.
    if initial is None or initial<0:
        for pivot,i in equality_pivots:
            if not db.get(pivot):continue
            validate_dependencies(candidate,risk,choices,K,theta_map,a,[(i,F(1))])
            er=A.getrow(i);ed={int(j):frac(v) for j,v in zip(er.indices,er.data)}
            g=db[pivot]/ed[pivot];da+=g*frac(a['rhs'][i])
            for j,v in ed.items():db[j]=db.get(j,F(0))-g*v
            eq.append(dict(row=i,pivot=pivot,gamma=encode(g),sha256=row_hash(A,a,i)))
        db={j:v for j,v in db.items() if v}
    margin,terms=box_margin(da,db,a['lo'],a['hi'])
    status='BOUND_INCOMPLETE' if margin is None else 'PROVEN' if margin>=0 else 'NOT_PROVEN_NEGATIVE_MARGIN'
    return dict(candidate=candidate,candidate_sha=row_hash(A,a,candidate),risk=risk,risk_sha=row_hash(A,a,risk),old_normalizer=encode(op),old_sign=osign,risk_normalizer=encode(rp),risk_sign=rsign,choices=used,equalities=eq,old_a=encode(oa),new_a=encode(na),new_b=[[j,encode(v)] for j,v in sorted(nb.items()) if v],delta_a_after_equalities=encode(da),box_terms=terms,margin=None if margin is None else encode(margin),box_only_margin=None if initial is None else encode(initial),status=status,continuous_domain=True,no_tolerance_acceptance=True)
