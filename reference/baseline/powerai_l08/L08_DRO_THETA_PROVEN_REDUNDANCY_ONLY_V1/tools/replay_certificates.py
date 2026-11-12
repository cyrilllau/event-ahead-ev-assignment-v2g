"""Independent proof replay by combining original inequality coefficient vectors."""
from pathlib import Path
from fractions import Fraction as F
import json,sys,numpy as np
from scipy import sparse as sp
from proof_core import row_hash,decode

def replay(A,a,cert,K,nfirst=60780,rho=60780):
    c=cert['candidate'];r=cert['risk'];assert c not in K and r in K
    assert row_hash(A,a,c)==cert['candidate_sha'] and row_hash(A,a,r)==cert['risk_sha']
    # Weighted retained LHS >= weighted retained RHS, scaled to rho coefficient 1.
    coeff={};rhs=F(0);dep=[]
    def add(i,w):
        nonlocal rhs
        assert i in K
        row=A.getrow(i)
        for j,v in zip(row.indices,row.data):coeff[int(j)]=coeff.get(int(j),F(0))+w*F(float(v))
        rhs+=w*F(float(a['rhs'][i]));dep.append(i)
    sg=1 if a['sense'][r]=='>' else -1;rp=sg*F(float(A[r,rho]));assert rp>0
    add(r,F(sg)/rp)
    for x in cert['choices']:
        j=x['theta'];w=-coeff.get(j,F(0));assert w==decode(x['weight']) and w>=0
        i=x['row']
        if i is None:
            assert a['lo'][j]>=0;coeff[j]=coeff.get(j,F(0))+w
        else:
            assert row_hash(A,a,i)==x['sha256'];sign=1 if a['sense'][i]=='>' else -1
            pivot=sign*F(float(A[i,j]));assert pivot>0
            add(i,w*sign/pivot)
    assert coeff.get(rho)==1
    assert all(v==0 for j,v in coeff.items() if j>rho)
    old=A.getrow(c);sign=1 if a['sense'][c]=='>' else -1;p=sign*F(float(A[c,rho]));assert p>0
    delta=-sign*F(float(a['rhs'][c]))/p+rhs
    diff={j:-v for j,v in coeff.items() if j<nfirst}
    for j,v in zip(old.indices,old.data):
        if j<nfirst:diff[int(j)]=diff.get(int(j),F(0))+sign*F(float(v))/p
    for e in cert['equalities']:
        i=e['row'];g=decode(e['gamma']);assert i in K and i<90768 and a['sense'][i]=='='
        assert row_hash(A,a,i)==e['sha256'];delta+=g*F(float(a['rhs'][i]))
        er=A.getrow(i)
        for j,v in zip(er.indices,er.data):diff[int(j)]=diff.get(int(j),F(0))-g*F(float(v))
    assert delta==decode(cert['delta_a_after_equalities'])
    expected={int(j):(decode(v),endpoint,decode(term)) for j,v,endpoint,term in cert['box_terms']}
    for j,v in diff.items():
        if not v:continue
        endpoint=float(a['lo'][j] if v>0 else a['hi'][j]);assert np.isfinite(endpoint) and abs(endpoint)<1e100
        term=v*F(endpoint);assert expected.pop(j)==(v,endpoint,term);delta+=term
    assert not expected and delta==decode(cert['margin'])
    assert (delta>=0)==(cert['status']=='PROVEN')
    return dict(candidate=c,passed=True,margin=cert['margin'],dependency_count=len(dep),independent_combination=True)

def main(root):
    root=Path(root);A=sp.load_npz(root/'A_FULL/A.npz');a=dict(np.load(root/'A_FULL/ARRAYS.npz'));part=json.loads((root/'PARTITION.json').read_text());K=set(part['K']);removed=json.loads((root/'REMOVED_ROWS.json').read_text())
    res=[replay(A,a,json.loads((root/'proofs'/f'{i:06d}.json').read_text()),K) for i in removed]
    B=sp.load_npz(root/'B_TRIMMED/A.npz');ba=dict(np.load(root/'B_TRIMMED/ARRAYS.npz'));keep=np.array([i for i in range(A.shape[0]) if i not in set(removed)])
    d=A[keep]-B;d.eliminate_zeros();assert d.nnz==0
    for k in a:assert np.array_equal(a[k][keep] if k in ('rhs','sense') else a[k],ba[k]),k
    print(json.dumps(dict(passed=True,rows=res,matrix_exact=True)))
if __name__=='__main__':main(sys.argv[1])
