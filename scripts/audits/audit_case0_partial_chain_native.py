"""Exact d9c779 native numeric audit, with explicit partial-model input/output kinds.

Only the function name and two kind strings differ. Raw native metadata,
all status/row/domain/feasibility/bound checks and original policy are retained.
"""
import hashlib,math
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.multi_anchor_bound_audit import _need,_same,_alive,_seal
from evcs_experiments.relaxation_bound import coefficient_sha256
from evcs_experiments.random_streams import canonical_hash

def audit_partial_chain_native_bound(model,model_audit,native_record,*,native_status,native_bound,
                                 incumbent,expected_coefficient_sha256,fixed_upper_bound,deadline):
    """Check a source-bound Gurobi native bound, preserving original limit status.

    expected_coefficient_sha256 must be pinned before the native call. The native
    wrapper does not itself hash its full input, so caller/source admission must
    establish that exact coefficient model was passed to that call. Invalid or
    missing bounds raise and must remain unaccepted; no substitute is invented.
    """
    _alive(deadline);body=dict(model_audit);digest=body.pop('audit_sha256')
    _need(canonical_hash(body)==digest and body['passed'] is True
          and body['kind']=='INDEPENDENT_CASE0_PARTIAL_CHAIN_MODEL_AUDIT_V1','Original model audit missing/changed')
    actual=coefficient_sha256(model)
    _need(actual==expected_coefficient_sha256==body['coefficient_sha256'],'Audited native-call coefficients changed')
    nr=native_record;A=csr_matrix(model['matrix'])
    _need(nr['backend']=='gurobipy' and nr['gurobi_version']==[12,0,3],'Native backend/version differs')
    _same(nr['original_matrix_shape'],list(A.shape),'Native matrix shape differs')
    _need(nr['original_matrix_nnz']==A.nnz,'Native matrix sparsity differs')
    opts=nr['effective_options']
    _need(opts['Threads']==1 and opts['MIPGap']==.005 and opts['Seed']==0
          and opts['FeasibilityTol']==1e-9 and all(opts[k]==1e-8 for k in ('OptimalityTol','IntFeasTol'))
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
    return _seal(dict(kind='INDEPENDENT_CASE0_PARTIAL_CHAIN_NATIVE_AUXILIARY_BOUND_AUDIT_V1',passed=True,
        expected_feasibility_tolerance=1e-9,actual_feasibility_tolerance=opts['FeasibilityTol'],
        baseline_auditor_source_sha256='9fd199f0eab88b7b65326875891ebcd415891241b536eb79543f17a5ff9404ea',
        coefficient_sha256=actual,native_status=nr['status'],native_status_code=code,
        native_lower_bound_without_constant_usd=float(native_bound),objective_constant_usd=model['objective_constant_usd'],
        lower_bound_usd=shifted,fixed_upper_bound_usd=float(fixed_upper_bound),
        auxiliary_incumbent_objective_usd=None if objective is None else objective+model['objective_constant_usd'],
        native_branch_and_bound_proof_replayed=False,new_candidate_upper_bound_verified=False,
        solver_calls=0,scientific_acceptance=False,requires_terminal_parent_settlement=True))

