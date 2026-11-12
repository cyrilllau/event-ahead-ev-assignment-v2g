"""Bounded saved-array extra-anchor cuts for Case0 SAA/ARO lower bounds only.

New LPs use original recourse arrays and source support. Fractional first-stage
anchors are allowed for valid minorants, never presented as accepted plans/UBs.
"""
import hashlib,time,warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import csr_matrix,hstack,vstack
from .dual_cut_bound import build_dual_cut
from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256

def need(v,m):
    if not bool(v):raise ValueError(m)
def maximum(x):return float(np.max(x,initial=0.))
def close(a,b):need(abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),'Q objective differs')

def extract_template(A,arrays,h,i):
    sid=h['scenario_ids'][i];sl=h['variable_slices'];nf=sl['first_stage'][1];H=sl['x'][1];a,b=sl['recourses'][sid];th=h['recourse_template_sha256s'][i]
    idx={(v['scope'],v['kind']):v for v in h['block_map']};t=dict(scenario_id=sid,template_sha256=th,lower_bounds=arrays['lower_bounds'][a:b],upper_bounds=arrays['upper_bounds'][a:b])
    for k in ('ub','eq'):
        row=idx[sid,'recourse_'+k];need(row['source_sha256']==th,'Original row source differs');lo,hi=row['row_start'],row['row_stop'];block=A[lo:hi];own=block[:,a:b].tocsr();coupling=block[:,:nf].tocsr()
        need(own.nnz+coupling.nnz==block.nnz,'Foreign recourse coupling')
        need(np.all(np.isneginf(arrays['row_lower'][lo:hi])) if k=='ub' else np.array_equal(arrays['row_lower'][lo:hi],arrays['row_upper'][lo:hi]),'Original row sense differs')
        t.update({f'A_{k}':own,f'Bx_{k}':-coupling[:,:H],f'Br_{k}':-coupling[:,H:],f'b_{k}':arrays['row_upper'][lo:hi]})
    need(h['scenario_cost_constants_usd'][i]==0.,'Only locked zero-constant Case0')
    if h['risk_model']=='SAA':
        n=len(h['scenario_ids']);need(h['empirical_probabilities']==[1./n]*n,'Original uniform SAA law differs');t['c']=arrays['c'][a:b]*n
    else:
        need(h['risk_model']=='ARO' and h['empirical_probabilities'] is None,'Unsupported risk law');r=idx[sid,'risk_epigraph'];theta=sl['risk_threshold'][0];row=A[r['row_start']]
        need(r['source_sha256']==th and r['row_stop']==r['row_start']+1 and row[0,theta]==-1 and row[:,:a].nnz==0 and row[:,b:theta].nnz==0,'Original ARO objective row differs');t['c']=row[:,a:b].toarray().ravel()
    return t

def audit_anchor_q(t,z,v,q,duals):
    H=t['Bx_ub'].shape[1];z=np.asarray(z,float);v=np.asarray(v,float)
    need(v.shape==np.asarray(t['c']).shape and np.isfinite(v).all() and np.isfinite(z).all(),'Invalid new Q vector')
    bub=t['b_ub']+t['Bx_ub']@z[:H]+t['Br_ub']@z[H:];beq=t['b_eq']+t['Bx_eq']@z[:H]+t['Br_eq']@z[H:]
    error=max(maximum(t['A_ub']@v-bub),maximum(abs(t['A_eq']@v-beq)),maximum(t['lower_bounds']-v),maximum(v-t['upper_bounds']))
    need(error<=1e-7,'New original Q primal infeasible');close(float(np.asarray(t['c'])@v),q)
    cut,g=build_dual_cut(t,duals,scenario_constant_usd=0.,anchor_first_stage=z,anchor_q_usd=q);need(cut['passed'],'New original Q dual/anchor failed')
    return cut,g,dict(passed=True,primal_error=error,q_usd=q,cut_sha256=cut['cut_sha256'],scope='Original algebraic Q LP/KKT; no new physical plan acceptance')

def solve_anchor_q(t,z,seconds):
    need(seconds>0,'Q budget exhausted');H=t['Bx_ub'].shape[1];bub=t['b_ub']+t['Bx_ub']@z[:H]+t['Br_ub']@z[H:];beq=t['b_eq']+t['Bx_eq']@z[:H]+t['Br_eq']@z[H:]
    opts=dict(time_limit=float(seconds),threads=1,primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8,ipm_optimality_tolerance=1e-8)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
        r=linprog(t['c'],A_ub=t['A_ub'],b_ub=bub,A_eq=t['A_eq'],b_eq=beq,bounds=np.column_stack((t['lower_bounds'],t['upper_bounds'])),method='highs',options=opts)
    need(r.status==0 and r.x is not None,'New Q did not finish optimal within budget')
    d={k:np.asarray(getattr(r,n).marginals,float) for k,n in [('inequality','ineqlin'),('equality','eqlin'),('lower','lower'),('upper','upper')]}
    c,g,a=audit_anchor_q(t,z,r.x,float(r.fun),d)
    return c,g,a,dict(primal=r.x,**d),dict(status=int(r.status),objective_usd=float(r.fun),message=str(r.message),options=opts)

def build_cut_bundle(first,ids,groups,risk):
    """SAA=sum_s max_k cuts_s,k / N; ARO=max_s,k cuts_s,k."""
    n=len(ids);nf=len(first['c']);need(n>0 and risk in ('SAA','ARO') and len(set(ids))==n,'Bad risk/support');nt=n if risk=='SAA' else 1
    rows=[];rhs=[]
    for group in groups:
        need(set(group)==set(range(n)),'Incomplete anchor support')
        for i in range(n):
            c,g=group[i];body=dict(c);sha=body.pop('cut_sha256');g=np.asarray(g,float)
            need(canonical_hash(body)==sha and c['passed'] and c['scenario_id']==ids[i] and c['scenario_constant_usd']==0. and len(g)==nf,'Invalid cut identity')
            need(hashlib.sha256(g.astype('<f8').tobytes()).hexdigest()==c['gradient_sha256'],'Changed cut gradient')
            tail=np.zeros(nt);tail[i if risk=='SAA' else 0]=-1.;rows.append(csr_matrix(np.r_[g,tail].reshape(1,-1)));rhs.append(-c['intercept_usd'])
    need(len(rows)>0,'Empty cuts')
    model=dict(matrix=vstack([hstack([first['matrix'],csr_matrix((first['matrix'].shape[0],nt))],format='csr'),*rows],format='csr'),c=np.r_[first['c'],np.full(nt,1./n if risk=='SAA' else 1.)],integrality=np.r_[first['integrality'],np.zeros(nt,dtype=int)],lower_bounds=np.r_[first['lower_bounds'],np.full(nt,-np.inf)],upper_bounds=np.r_[first['upper_bounds'],np.full(nt,np.inf)],row_lower=np.r_[first['row_lower'],np.full(len(rows),-np.inf)],row_upper=np.r_[first['row_upper'],rhs],objective_constant_usd=float(first['objective_constant_usd']))
    return model,dict(kind='CASE0_MULTIANCHOR_AFFINE_MODEL_V1',risk_model=risk,support_ids=list(ids),anchor_groups=len(groups),coefficient_sha256=coefficient_sha256(model),original_first_stage_unchanged=True,scientific_acceptance=False)
