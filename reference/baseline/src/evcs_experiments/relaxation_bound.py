"""Numerically certified LP relaxation of an unchanged saved restricted master.

Only 0/1 integrality is relaxed. This module neither fixes a plan nor evaluates
recourse/probability. A bound is a floating-point NUMERICAL_LP_RELAXATION_BOUND
under the project's existing original-unit certificate tolerances, not an
exact-arithmetic proof. Formal source/model/support binding belongs to the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import scipy
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._highspy import _core as _bundled_highs
from scipy.sparse import csr_matrix, vstack

TOLERANCE=1e-8
OBJECTIVE_ABSOLUTE_TOLERANCE=1e-4
OBJECTIVE_RELATIVE_TOLERANCE=1e-7
SOLVER_TOLERANCE=1e-10
SOLVER_METHOD='highs-ipm'


def _need(ok,message):
    if not bool(ok):raise ValueError(message)


def _maximum(value):return float(np.max(value,initial=0.))


def _problem(problem):
    matrix=csr_matrix(problem['matrix'],copy=False)
    _need(matrix.ndim==2 and matrix.shape[1]>0 and np.isfinite(matrix.data).all(),'Invalid saved RMP matrix')
    rows,cols=matrix.shape
    result={'matrix':matrix}
    for name,size in (('c',cols),('row_lower',rows),('row_upper',rows),('lower_bounds',cols),('upper_bounds',cols)):
        a=np.asarray(problem[name],dtype=float)
        _need(a.shape==(size,) and not np.isnan(a).any(),'Invalid original array: '+name)
        if name=='c':_need(np.isfinite(a).all(),'Nonfinite objective')
        result[name]=a
    l,u=result['lower_bounds'],result['upper_bounds'];rl,ru=result['row_lower'],result['row_upper']
    _need(np.all(l<=u) and np.all(rl<=ru) and not np.isposinf(l).any() and not np.isneginf(u).any()
          and not np.isposinf(rl).any() and not np.isneginf(ru).any(),'Invalid original bounds')
    integers=np.asarray(problem['integrality'])
    _need(integers.shape==(cols,) and np.isin(integers,[0,1]).all(),'Only binary/integer 0/1 integrality can be relaxed')
    result['integrality']=integers.astype(np.int64,copy=False)
    constant=problem['objective_constant_usd']
    _need(np.ndim(constant)==0 and np.isfinite(constant),'Nonfinite objective constant')
    result['objective_constant_usd']=float(constant)
    return result


def coefficient_sha256(problem):
    """Bind every original coefficient, bound, integrality flag and constant."""
    p=_problem(problem);a=p['matrix'];h=hashlib.sha256()
    h.update(json.dumps({'shape':a.shape,'format':'csr'},sort_keys=True).encode())
    for name,value in [('matrix_data',a.data),('matrix_indices',a.indices),('matrix_indptr',a.indptr),
        *[(k,p[k]) for k in ('c','row_lower','row_upper','lower_bounds','upper_bounds','integrality')],
        ('objective_constant_usd',np.array([p['objective_constant_usd']]))]:
        v=np.asarray(value,dtype='<i8' if np.asarray(value).dtype.kind in 'iu' else '<f8')
        h.update(name.encode());h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


def _row_map(p):
    lo,hi=p['row_lower'],p['row_upper']
    equal=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
    return np.flatnonzero(equal),np.flatnonzero(np.isfinite(hi)&~equal),np.flatnonzero(np.isfinite(lo)&~equal)


def check_relaxation_certificate(problem,values,*,solver_status,raw_objective_without_constant_usd=None):
    """Independently recompute original-matrix primal/dual/KKT arithmetic.

Marginal convention: row_lower >=0, row_upper <=0, row_equal free;
variable_lower >=0 and variable_upper <=0. Inactive and infinite-bound
multiplier positions must be zero within the unchanged 1e-8 numerical gate.
"""
    p=_problem(problem);A=p['matrix'];m,n=A.shape
    cert=dict(status='NEED_WITNESSES',passed=False,certified_lower_bound_usd=None,
        bound_kind='NUMERICAL_LP_RELAXATION_BOUND',exact_arithmetic_proof=False,
        coefficient_sha256=coefficient_sha256(p),integrality_relaxed=True,
        primal_tolerance=TOLERANCE,dual_stationarity_tolerance=TOLERANCE,
        sign_and_inactive_bound_tolerance=TOLERANCE,objective_absolute_tolerance_usd=OBJECTIVE_ABSOLUTE_TOLERANCE,
        objective_relative_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
        original_objective_constant_usd=p['objective_constant_usd'],solver_status=int(solver_status))
    expected={'primal':n,'row_lower_marginals':m,'row_upper_marginals':m,'row_equality_marginals':m,
              'variable_lower_marginals':n,'variable_upper_marginals':n}
    if not all(k in values for k in expected):return cert
    v={k:np.asarray(values[k],dtype=float) for k in expected}
    _need(all(v[k].shape==(size,) and np.isfinite(v[k]).all() for k,size in expected.items()),'Invalid saved primal/dual witness shape or values')
    x=v['primal'];lo,up=p['lower_bounds'],p['upper_bounds'];rl,ru=p['row_lower'],p['row_upper']
    eql,upper,lower=_row_map(p)
    yl,yu,ye=(v[k] for k in ('row_lower_marginals','row_upper_marginals','row_equality_marginals'))
    zl,zu=v['variable_lower_marginals'],v['variable_upper_marginals']
    mask_lower=np.zeros(m,dtype=bool);mask_lower[lower]=True
    mask_upper=np.zeros(m,dtype=bool);mask_upper[upper]=True
    mask_equal=np.zeros(m,dtype=bool);mask_equal[eql]=True
    finite_lo,finite_up=np.isfinite(lo),np.isfinite(up)
    inactive=max(_maximum(abs(yl[~mask_lower])),_maximum(abs(yu[~mask_upper])),_maximum(abs(ye[~mask_equal])),
                 _maximum(abs(zl[~finite_lo])),_maximum(abs(zu[~finite_up])))
    sign=max(_maximum(-yl),_maximum(yu),_maximum(-zl),_maximum(zu))
    activity=A@x;row_violation=np.maximum(np.maximum(rl-activity,activity-ru),0.)
    bound_violation=np.maximum(np.maximum(lo-x,x-up),0.)
    stationarity=p['c']-A.T@(yl+yu+ye)-zl-zu
    primal=max(_maximum(row_violation),_maximum(bound_violation))
    row_scale=np.maximum(1.,abs(A)@abs(x));bound_scale=np.maximum(1.,abs(x))
    normalized=max(_maximum(row_violation/row_scale),_maximum(bound_violation/bound_scale))
    dual=(_maximum(abs(stationarity)))
    pobj=float(p['c']@x+p['objective_constant_usd'])
    dobj=float(rl[lower]@yl[lower]+ru[upper]@yu[upper]+rl[eql]@ye[eql]
               +lo[finite_lo]@zl[finite_lo]+up[finite_up]@zu[finite_up]+p['objective_constant_usd'])
    _need(all(math.isfinite(z) for z in (pobj,dobj,primal,normalized,dual,sign,inactive)),'Nonfinite original-unit certificate arithmetic')
    tolerance=OBJECTIVE_ABSOLUTE_TOLERANCE+OBJECTIVE_RELATIVE_TOLERANCE*max(1.,abs(pobj),abs(dobj))
    gap=pobj-dobj
    raw_matches=raw_objective_without_constant_usd is not None and math.isfinite(raw_objective_without_constant_usd) and abs(pobj-(raw_objective_without_constant_usd+p['objective_constant_usd']))<=tolerance
    free=np.isneginf(lo)&np.isposinf(up)
    complements=[yl[lower]*(activity[lower]-rl[lower]),yu[upper]*(ru[upper]-activity[upper]),
                 zl[finite_lo]*(x[finite_lo]-lo[finite_lo]),zu[finite_up]*(up[finite_up]-x[finite_up])]
    passed=bool(solver_status==0 and max(primal,normalized,dual,sign,inactive)<=TOLERANCE and abs(gap)<=tolerance and raw_matches)
    cert.update(status='PASS_NUMERICAL_LP_CERTIFICATE' if passed else 'NUMERICAL_CERTIFICATE_FAILED',passed=passed,
        certified_lower_bound_usd=dobj if passed else None,primal_objective_usd=pobj,dual_objective_usd=dobj,
        primal_dual_gap_usd=gap,objective_tolerance_usd=tolerance,max_primal_violation=primal,
        max_normalized_primal_violation=normalized,max_dual_stationarity_residual=dual,
        max_free_column_stationarity_residual=_maximum(abs(stationarity[free])),free_column_count=int(free.sum()),
        max_dual_sign_violation=sign,max_inactive_or_infinite_bound_multiplier=inactive,
        maximum_complementarity=_maximum(np.array([_maximum(abs(a)) for a in complements])),
        primal_objective_matches_solver=bool(raw_matches),actual_lp_optimal_status=bool(solver_status==0))
    return cert


def solve_relaxation_bound(problem,*,time_limit_seconds,threads=1,log_path=None):
    """Return (JSON-serializable record, raw NumPy witness arrays).

All original rows/bounds/costs are unchanged. Two-sided finite rows become their
mathematically identical <= pair for linprog. log_path is an exclusive JSON
solver summary, not a native log. Time includes validation/transformation/checks;
a parent hard wall remains necessary to bound third-party solver cleanup/export.
"""
    started=time.monotonic()
    _need(type(time_limit_seconds) in (int,float) and math.isfinite(time_limit_seconds) and time_limit_seconds>0,'Positive finite LP relaxation wall required')
    _need(type(threads) is int and 1<=threads<=4,'LP threads must be 1..4')
    if log_path is not None:_need(not Path(log_path).exists(),'Refusing to overwrite relaxation solver summary')
    p=_problem(problem);digest=coefficient_sha256(p);equal,upper,lower=_row_map(p)
    A=p['matrix'];blocks=[];rhs=[]
    if len(upper):blocks.append(A[upper]);rhs.append(p['row_upper'][upper])
    if len(lower):blocks.append(-A[lower]);rhs.append(-p['row_lower'][lower])
    Aub=vstack(blocks,format='csr') if blocks else None
    bub=np.concatenate(rhs) if rhs else None
    Aeq=A[equal] if len(equal) else None;beq=p['row_lower'][equal] if len(equal) else None
    values={};remaining=float(time_limit_seconds)-(time.monotonic()-started)
    record=dict(bound_kind='NUMERICAL_LP_RELAXATION_BOUND',coefficient_sha256=digest,
        solver_method=SOLVER_METHOD,scipy_version=scipy.__version__,highs_version=_bundled_highs._Highs().version(),
        source_model_sha256=problem.get('source_model_sha256'),formal_problem_sha256=problem.get('formal_problem_sha256'),
        original_shape=list(A.shape),original_nnz=int(A.nnz),original_integer_columns=int(np.count_nonzero(p['integrality'])),
        original_integrality_sha256=hashlib.sha256(p['integrality'].astype('<i8').tobytes()).hexdigest(),
        integrality_relaxed_only=True,objective_constant_usd=p['objective_constant_usd'],
        master_milp_calls=0,recourse_calls=0,probability_calls=0,lp_calls=0,
        formal_OOS=False,scientific_acceptance=False,lower_bound_usd=None,certified_lower_bound_usd=None,
        original_master_result_modified=False,exact_arithmetic_proof=False)
    if remaining<=0:
        record.update(status='TIME_LIMIT',solver_status=None,message='Budget exhausted before LP call',certificate=None)
    else:
        options=dict(time_limit=remaining,threads=threads,primal_feasibility_tolerance=SOLVER_TOLERANCE,
                     dual_feasibility_tolerance=SOLVER_TOLERANCE,ipm_optimality_tolerance=SOLVER_TOLERANCE)
        solve_started=time.monotonic()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            raw=linprog(p['c'],A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                bounds=np.column_stack((p['lower_bounds'],p['upper_bounds'])),method=SOLVER_METHOD,options=options)
        solve_elapsed=time.monotonic()-solve_started
        if raw.x is not None and np.asarray(raw.x).shape==p['c'].shape and np.isfinite(raw.x).all():
            values['primal']=np.asarray(raw.x).copy()
            try:
                marg=np.asarray(raw.ineqlin.marginals,dtype=float);eq=np.asarray(raw.eqlin.marginals,dtype=float)
                zl=np.asarray(raw.lower.marginals,dtype=float);zu=np.asarray(raw.upper.marginals,dtype=float)
                if marg.shape==(len(upper)+len(lower),) and eq.shape==(len(equal),) and zl.shape==zu.shape==p['c'].shape:
                    yl=np.zeros(A.shape[0]);yu=np.zeros(A.shape[0]);ye=np.zeros(A.shape[0])
                    yu[upper]=marg[:len(upper)];yl[lower]=-marg[len(upper):];ye[equal]=eq
                    values.update(row_lower_marginals=yl,row_upper_marginals=yu,row_equality_marginals=ye,
                                  variable_lower_marginals=zl.copy(),variable_upper_marginals=zu.copy())
            except (AttributeError,TypeError,ValueError):pass  # retain missing witnesses as an uncertified outcome
        rawfun=None if raw.fun is None or not np.isfinite(raw.fun) else float(raw.fun)
        try:
            cert=check_relaxation_certificate(p,values,solver_status=int(raw.status),raw_objective_without_constant_usd=rawfun)
        except ValueError as exc:
            # Keep malformed solver witnesses for diagnosis, but never attach a
            # numerical bound to them or let an OPTIMAL label override the gate.
            cert=dict(status='NUMERICAL_CERTIFICATE_FAILED',passed=False,certified_lower_bound_usd=None,
                      coefficient_sha256=digest,error_type=type(exc).__name__,error=str(exc),
                      bound_kind='NUMERICAL_LP_RELAXATION_BOUND',exact_arithmetic_proof=False)
        _need(coefficient_sha256(p)==digest,'Original saved coefficients changed during relaxation solve')
        status='OPTIMAL' if cert['passed'] else {0:'NUMERICAL_CERTIFICATE_FAILED',1:'TIME_LIMIT',2:'INFEASIBLE',3:'UNBOUNDED'}.get(raw.status,'SOLVER_ERROR')
        record.update(status=status,solver_status=int(raw.status),message=str(raw.message),lp_calls=1,
            raw_solver=dict(backend='scipy.optimize.linprog/HiGHS',method=SOLVER_METHOD,scipy_version=record['scipy_version'],highs_version=record['highs_version'],success=bool(raw.success),status=int(raw.status),message=str(raw.message),
                objective_without_constant_usd=rawfun,nit=int(raw.nit or 0),crossover_nit=int(getattr(raw,'crossover_nit',0) or 0),options=options),
            certificate=cert,solver_seconds=solve_elapsed,lower_bound_usd=cert['certified_lower_bound_usd'],
            certified_lower_bound_usd=cert['certified_lower_bound_usd'])
    elapsed=time.monotonic()-started;record['wall_seconds']=elapsed
    if elapsed>=time_limit_seconds:
        record.update(status='TIME_LIMIT',lower_bound_usd=None,certified_lower_bound_usd=None,wall_exhausted_after_solver=True)
    if log_path is not None:
        path=Path(log_path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    return record,values
