"""Independent original-row audit of a multi-anchor auxiliary model and bound.

No epigraph-builder import, solver call, or branch-and-bound proof replay. The
caller admits original cut/floor provenance and binds the exact native call.
"""
import hashlib
import math
import time

import numpy as np
from scipy.sparse import csr_matrix

from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256


def _need(ok,message):
    if not bool(ok):raise ValueError(message)


def _alive(deadline):
    if not np.isfinite(deadline) or time.monotonic()>=deadline:raise TimeoutError('Auxiliary bound audit deadline')


def _same(a,b,message):_need(np.array_equal(a,b),message)


def _seal(record):return dict(record,audit_sha256=canonical_hash(record))


def audit_multi_anchor_model(model,first_stage_problem,support_ids,cut_sets,*,probability_set,floors,deadline):
    """Check every original first-stage/cut/floor row on the shared P columns.

    Mathematical admissibility of supplied original cuts/floors is a separate
    already-checked prerequisite. This validates their exact assembly, not an
    alternative cut derivation or a new physical model.
    """
    _alive(deadline)
    first=first_stage_problem;ids=tuple(support_ids);S=len(ids);K=len(probability_set.feature_ids)
    _need(ids==tuple(probability_set.support_ids) and S>0 and len(set(ids))==S,'Original P/support axis differs')
    _need(isinstance(cut_sets,(list,tuple)) and len(cut_sets)>0,'Missing original anchor sets')
    _need(set(floors)==set(range(S)) and all(type(i) is int for i in floors),'Original complete floor axis differs')
    A=csr_matrix(model['matrix']);F=csr_matrix(first['matrix']);n=len(first['c']);m=F.shape[0]
    _need(A.shape==(m+(len(cut_sets)+1)*S,n+1+2*K),'Auxiliary row/column layout differs')
    _need(np.isfinite(A.data).all() and np.isfinite(F.data).all(),'Nonfinite original coefficients')
    _need((A[:m,:n]!=F).nnz==0 and A[:m,n:].nnz==0,'Original first-stage rows changed')
    for key in ('row_lower','row_upper'):_same(np.asarray(model[key])[:m],first[key],'Original first-stage '+key+' changed')
    for key in ('lower_bounds','upper_bounds','integrality'):
        _same(np.asarray(model[key])[:n],first[key],'Original first-stage domain changed: '+key)
    _same(model['c'],np.r_[first['c'],1.,-probability_set.lower,probability_set.upper],'Original objective/P dual signs changed')
    _same(np.asarray(model['lower_bounds'])[n:],np.r_[-np.inf,np.zeros(2*K)],'Shared ambiguity lower bounds changed')
    _same(np.asarray(model['upper_bounds'])[n:],np.full(1+2*K,np.inf),'Shared ambiguity upper bounds changed')
    _same(np.asarray(model['integrality'])[n:],np.zeros(1+2*K),'Shared ambiguity variable types changed')
    _need(model['objective_constant_usd']==first['objective_constant_usd'],'Original objective constant changed')
    rl,ru=np.asarray(model['row_lower']),np.asarray(model['row_upper'])
    anchors=set();cut_hashes=[];floor_hashes=[]
    for group_index,group in enumerate(cut_sets):
        _alive(deadline);cuts=group['cuts'];anchor=np.asarray(group['anchor_first_stage'],dtype='<f8')
        _need(set(cuts)==set(range(S)) and all(type(i) is int for i in cuts),'Original complete cut axis differs')
        _need(anchor.shape==(n,) and np.isfinite(anchor).all(),'Original anchor vector differs')
        ah=hashlib.sha256(anchor.tobytes()).hexdigest();_need(ah not in anchors,'Duplicate anchor');anchors.add(ah)
        for i,sid in enumerate(ids):
            cut,g=cuts[i];g=np.asarray(g,dtype='<f8');floor=floors[i]
            body=dict(cut);ch=body.pop('cut_sha256');_need(canonical_hash(body)==ch,'Original cut hash differs')
            _need(cut['passed'] is True and cut['status']=='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT'
                  and cut['scenario_id']==sid and cut['anchor_first_stage_sha256']==ah
                  and g.shape==(n,) and np.isfinite(g).all()
                  and cut['gradient_sha256']==hashlib.sha256(g.tobytes()).hexdigest(),'Original cut/anchor identity differs')
            _need(cut['dual_tolerance']==1e-8 and all(np.isfinite(cut[k]) and 0<=cut[k]<=1e-8 for k in
                  ('max_dual_stationarity_residual','max_dual_sign_violation','max_infinite_bound_multiplier')),
                  'Original cut numerical contract differs')
            fbody=dict(floor);fh=fbody.pop('floor_sha256');_need(canonical_hash(fbody)==fh,'Original floor hash differs')
            _need(floor['passed'] is True and floor['status']=='PASS_ORIGINAL_COLUMN_BOX_FLOOR'
                  and floor['scenario_id']==sid and floor['template_sha256']==cut['template_sha256']
                  and floor['scenario_constant_usd']==cut['scenario_constant_usd']
                  and math.isfinite(floor['floor_usd']) and math.isfinite(cut['intercept_usd']),
                  'Original floor/template/constant differs')
            tail=np.r_[1.,-probability_set.features[i],probability_set.features[i]]
            j=m+i if group_index==0 else m+(group_index+1)*S+i
            _same(A[j,:n].toarray().ravel(),-g,'Original cut gradient row differs')
            _same(A[j,n:].toarray().ravel(),tail,'Original cut shared probability columns differ')
            _need(rl[j]==cut['intercept_usd'] and np.isposinf(ru[j]),'Original cut row bounds differ')
            if group_index==0:
                j=m+S+i
                _need(A[j,:n].nnz==0,'Original physical floor depends on new plan')
                _same(A[j,n:].toarray().ravel(),tail,'Original floor shared probability columns differ')
                _need(rl[j]==floor['floor_usd'] and np.isposinf(ru[j]),'Original physical floor removed/changed')
                floor_hashes.append(fh)
            cut_hashes.append(ch)
    _alive(deadline)
    return _seal(dict(kind='INDEPENDENT_MULTI_ANCHOR_MODEL_AUDIT_V1',passed=True,
        coefficient_sha256=coefficient_sha256(model),original_first_stage_sha256=coefficient_sha256(first),
        support_ids=list(ids),ambiguity_sha256=probability_set.ambiguity_sha256,anchor_count=len(cut_sets),
        cut_sha256s=cut_hashes,floor_sha256s=floor_hashes,original_domain_preserved=True,
        all_old_cuts_and_floors_preserved=True,shared_probability_columns=True,
        solver_calls=0,scientific_acceptance=False))


def audit_native_auxiliary_bound(model,model_audit,native_record,*,native_status,native_bound,
                                 incumbent,expected_coefficient_sha256,fixed_upper_bound,deadline,relaxation_audit=None):
    """Check a source-bound Gurobi native bound, preserving original limit status.

    expected_coefficient_sha256 must be pinned before the native call. The native
    wrapper does not itself hash its full input, so caller/source admission must
    establish that exact coefficient model was passed to that call. Invalid or
    missing bounds raise and must remain unaccepted; no substitute is invented.
    """
    _alive(deadline);body=dict(model_audit);digest=body.pop('audit_sha256')
    _need(canonical_hash(body)==digest and body['passed'] is True
          and body['kind']=='INDEPENDENT_MULTI_ANCHOR_MODEL_AUDIT_V1','Original model audit missing/changed')
    actual=coefficient_sha256(model)
    _need(actual==expected_coefficient_sha256,'Native-call coefficients changed')
    relaxation_fields={}
    if relaxation_audit is None:
        _need(actual==body['coefficient_sha256'],'Audited native-call coefficients changed')
    else:
        rb=dict(relaxation_audit);rd=rb.pop('audit_sha256')
        _need(canonical_hash(rb)==rd and rb.get('passed') is True
              and rb.get('kind')=='INDEPENDENT_BOUNDED_ROW_RELAXATION_AUDIT_V1'
              and rb.get('original_coefficient_sha256')==body['coefficient_sha256']
              and rb.get('relaxed_coefficient_sha256')==actual
              and rb.get('exact_outward_fraction_proof') is True
              and rb.get('objective_and_domain_unchanged') is True,
              'Independent relaxation audit missing/changed')
        relaxation_fields=dict(original_coefficient_sha256=body['coefficient_sha256'],
            relaxed_coefficient_sha256=actual,relaxation_audit_sha256=rd,
            bound_scope='Relaxed auxiliary minimization lower bound only')
    nr=native_record;A=csr_matrix(model['matrix'])
    _need(nr['backend']=='gurobipy' and nr['gurobi_version']==[12,0,3],'Native backend/version differs')
    _same(nr['original_matrix_shape'],list(A.shape),'Native matrix shape differs')
    _need(nr['original_matrix_nnz']==A.nnz,'Native matrix sparsity differs')
    opts=nr['effective_options']
    _need(opts['Threads']==1 and opts['MIPGap']==.005 and opts['Seed']==0
          and all(opts[k]==1e-8 for k in ('FeasibilityTol','OptimalityTol','IntFeasTol'))
          and type(nr['requested_time_limit_seconds']) in (int,float)
          and 0<nr['requested_time_limit_seconds']<=600,'Native resource/tolerance policy differs')
    names={2:'OPTIMAL',7:'ITERATION_LIMIT',8:'NODE_LIMIT',9:'TIME_LIMIT',10:'SOLUTION_LIMIT',
           11:'INTERRUPTED',15:'USER_OBJ_LIMIT',16:'WORK_LIMIT',17:'MEM_LIMIT'}
    code=nr['raw_model_status']
    _need(type(code) is int and code in names and nr['status']==names[code]
          and type(native_status) is int and native_status==(0 if code==2 else 1)
          and nr['certificate_failures']==[],'Native status/certificate cannot admit a bound')
    _need(nr['is_mip'] is True and nr['bound_source']=='MIP_ObjBound'
          and nr['solver_bound_consistent'] is True,'Native MIP bound provenance differs')
    for value in (native_bound,nr['raw_solver_bound_without_constant_usd'],nr['solver_bound_without_constant_usd'],fixed_upper_bound):
        _need(type(value) in (float,int) and math.isfinite(value),'Missing finite native bound/fixed UB')
    _need(native_bound==nr['solver_bound_without_constant_usd']==nr['raw_solver_bound_without_constant_usd'],
          'Native raw/reported lower bound differs')
    rl,ru=np.asarray(model['row_lower']),np.asarray(model['row_upper'])
    eq=np.isfinite(rl)&np.isfinite(ru)&(rl==ru)
    mapping=dict(eq=np.flatnonzero(eq),le=np.flatnonzero(np.isfinite(ru)&~eq),ge=np.flatnonzero(np.isfinite(rl)&~eq))
    summary=nr['row_map_summary']
    for name,rows in mapping.items():
        _need(summary[name]==dict(count=len(rows),source_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()),
              'Native original-row mapping differs')
    _need(summary['original_rows']==A.shape[0] and summary['translated_rows']==sum(map(len,mapping.values()))
          and nr['model_rows']==summary['translated_rows'] and nr['model_columns']==A.shape[1],
          'Native translated row/column counts differ')
    _need(code!=2 or incumbent is not None,'Native OPTIMAL is missing its incumbent')
    objective=None
    if incumbent is not None:
        z=np.asarray(incumbent,dtype=float);_need(z.shape==(A.shape[1],) and np.isfinite(z).all(),'Invalid native incumbent')
        maxval=lambda x:float(np.max(x,initial=0.))
        act=A@z;row=np.maximum(np.maximum(rl-act,act-ru),0.)
        lo,hi=np.asarray(model['lower_bounds']),np.asarray(model['upper_bounds']);col=np.maximum(np.maximum(lo-z,z-hi),0.)
        integer=np.asarray(model['integrality'])==1
        _need(max(maxval(row),maxval(row/np.maximum(1.,abs(A)@abs(z))),maxval(col),
                  maxval(col/np.maximum(1.,abs(z))),maxval(abs(z[integer]-np.rint(z[integer]))))<=1e-8,
              'Native incumbent violates original auxiliary model')
        objective=float(np.asarray(model['c'])@z)
        _need(nr['valid_incumbent'] is True and nr['objective_matches'] is True and nr['solution_count']>0
              and math.isfinite(nr['incumbent_without_constant_usd'])
              and abs(objective-nr['incumbent_without_constant_usd'])<=1e-4+1e-7*max(1.,abs(objective)),
              'Native incumbent objective/metadata differs')
        _need(native_bound<=objective+1e-4+1e-7*max(1.,abs(objective)),'Native LB exceeds auxiliary incumbent')
    else:
        _need(nr['valid_incumbent'] is False and nr['solution_count']==0
              and nr['incumbent_without_constant_usd'] is None,'Missing retained native incumbent')
    shifted=float(native_bound+model['objective_constant_usd'])
    tol=1e-4+1e-7*max(1.,abs(shifted),abs(fixed_upper_bound))
    _need(shifted<=fixed_upper_bound+tol,'Native auxiliary bound contradicts accepted fixed-plan UB')
    _alive(deadline)
    return _seal(dict(kind='INDEPENDENT_NATIVE_AUXILIARY_BOUND_AUDIT_V1',passed=True,
        coefficient_sha256=actual,native_status=nr['status'],native_status_code=code,
        native_lower_bound_without_constant_usd=float(native_bound),objective_constant_usd=model['objective_constant_usd'],
        lower_bound_usd=shifted,fixed_upper_bound_usd=float(fixed_upper_bound),
        auxiliary_incumbent_objective_usd=None if objective is None else objective+model['objective_constant_usd'],
        native_branch_and_bound_proof_replayed=False,new_candidate_upper_bound_verified=False,
        solver_calls=0,scientific_acceptance=False,requires_terminal_parent_settlement=True,**relaxation_fields))
