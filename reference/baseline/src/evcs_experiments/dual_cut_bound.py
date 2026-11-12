"""Numerical global lower-bound objective from fixed-template recourse duals.

For each scenario the dual feasible set is independent of the first-stage RHS.
An accepted dual therefore gives an affine minorant at every first-stage plan.
A feasible probability vector combines these minorants into a lower bound on
the full DRO objective. This module constructs that objective; it solves no LP.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from scipy.sparse import csr_matrix

from .probability import ProbabilitySet
from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256, _problem

TOLERANCE=1e-8


def _need(ok,message):
    if not bool(ok):raise ValueError(message)


def _max(x):return float(np.max(x,initial=0.))


def _array(value,size,label,finite=True):
    a=np.asarray(value,dtype=float)
    _need(a.shape==(size,) and not np.isnan(a).any() and (not finite or np.isfinite(a).all()),'Invalid '+label)
    return a


def _input_digest(matrices,arrays):
    h=hashlib.sha256()
    for name,a in sorted(matrices.items()):
        h.update(name.encode());h.update(str(a.shape).encode())
        for v in (a.data,a.indices,a.indptr):
            h.update(np.asarray(v,dtype='<i8' if v.dtype.kind in 'iu' else '<f8').tobytes())
    for name,a in sorted(arrays.items()):h.update(name.encode());h.update(np.asarray(a,dtype='<f8').tobytes())
    return h.hexdigest()


def build_dual_cut(template,dual,*,scenario_constant_usd=0.,anchor_first_stage,anchor_q_usd):
    """Return (JSON cut record, x-then-flattened-r gradient), no optimization.

template supplies A_ub,A_eq,Bx_ub,Bx_eq,Br_ub,Br_eq,b_ub,b_eq,c,
lower_bounds,upper_bounds and scenario_id. dual supplies inequality,equality,
lower,upper as saved in dual_kkt.npz. Source hashes are bound by the caller.
"""
    mats={k:csr_matrix(template[k],copy=False) for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
    _need(all(np.isfinite(a.data).all() for a in mats.values()),'Nonfinite template/coupling coefficients')
    mu,n=mats['A_ub'].shape;me,ne=mats['A_eq'].shape;H=mats['Bx_ub'].shape[1];R=mats['Br_ub'].shape[1]
    _need(ne==n and mats['Bx_ub'].shape==(mu,H) and mats['Bx_eq'].shape==(me,H)
          and mats['Br_ub'].shape==(mu,R) and mats['Br_eq'].shape==(me,R),'Recourse/first-stage axes differ')
    arrays={k:_array(template[k],size,k,finite=k not in ('lower_bounds','upper_bounds'))
            for k,size in [('b_ub',mu),('b_eq',me),('c',n),('lower_bounds',n),('upper_bounds',n)]}
    lo,up=arrays['lower_bounds'],arrays['upper_bounds']
    _need(np.all(lo<=up) and not np.isposinf(lo).any() and not np.isneginf(up).any(),'Invalid fixed recourse column bounds')
    y=_array(dual['inequality'],mu,'inequality dual');eq=_array(dual['equality'],me,'equality dual')
    zl=_array(dual['lower'],n,'lower-bound dual');zu=_array(dual['upper'],n,'upper-bound dual')
    _need(isinstance(template['scenario_id'],str) and template['scenario_id'],'Missing original scenario identity')
    anchor=_array(anchor_first_stage,H+R,'anchor first-stage vector')
    _need(math.isfinite(scenario_constant_usd) and math.isfinite(anchor_q_usd),'Nonfinite scenario constant/Q anchor')
    fl,fu=np.isfinite(lo),np.isfinite(up)
    stationarity=arrays['c']-mats['A_ub'].T@y-mats['A_eq'].T@eq-zl-zu
    residual=_max(abs(stationarity));sign=max(_max(y),_max(-zl),_max(zu))
    inactive=max(_max(abs(zl[~fl])),_max(abs(zu[~fu])))
    intercept=float(arrays['b_ub']@y+arrays['b_eq']@eq+lo[fl]@zl[fl]+up[fu]@zu[fu]+scenario_constant_usd)
    gradient=np.r_[mats['Bx_ub'].T@y+mats['Bx_eq'].T@eq,mats['Br_ub'].T@y+mats['Br_eq'].T@eq]
    _need(np.isfinite(gradient).all() and math.isfinite(intercept),'Nonfinite dual affine function')
    anchor_value=float(intercept+gradient@anchor);gap=float(anchor_q_usd-anchor_value)
    objective_tol=1e-4+1e-7*max(1.,abs(anchor_q_usd),abs(anchor_value))
    passed=max(residual,sign,inactive)<=TOLERANCE and abs(gap)<=objective_tol
    digest=_input_digest(mats,{**arrays,'dual_inequality':y,'dual_equality':eq,'dual_lower':zl,'dual_upper':zu,
                                      'scenario_constant':np.array([scenario_constant_usd])})
    record=dict(status='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT' if passed else 'NUMERICAL_DUAL_CUT_CERTIFICATE_FAILED',
        passed=bool(passed),scenario_id=template['scenario_id'],template_sha256=template.get('template_sha256'),
        input_coefficient_and_dual_sha256=digest,gradient_sha256=hashlib.sha256(gradient.astype('<f8').tobytes()).hexdigest(),
        intercept_usd=intercept,scenario_constant_usd=float(scenario_constant_usd),first_stage_columns=H+R,x_columns=H,reserve_columns=R,
        max_dual_stationarity_residual=residual,max_dual_sign_violation=sign,max_infinite_bound_multiplier=inactive,
        dual_tolerance=TOLERANCE,anchor_first_stage_sha256=hashlib.sha256(anchor.astype('<f8').tobytes()).hexdigest(),
        anchor_q_usd=float(anchor_q_usd),anchor_affine_value_usd=anchor_value,anchor_gap_usd=gap,anchor_gap_tolerance_usd=objective_tol,
        validity_scope='all first-stage RHS values with this unchanged A,c,column bounds and fixed scenario',
        exact_arithmetic_proof=False,recourse_calls=0,probability_calls=0)
    record['cut_sha256']=canonical_hash(record)
    return record,gradient


def build_weighted_dual_problem(first_stage_problem,support_ids,probabilities,cuts,*,probability_set):
    """Return the unchanged first-stage feasible set with the affine objective.

cuts is an index -> (cut record, gradient) mapping for EXACTLY positive weights.
Every zero weight is retained on the full support axis and skipped only because
it is exactly zero. The probability vector is neither clipped nor normalized.
"""
    p=_problem(first_stage_problem);ids=tuple(support_ids)
    _need(isinstance(probability_set,ProbabilitySet) and ids==probability_set.support_ids,'Original full ambiguity/support axis differs')
    weights=_array(probabilities,len(ids),'full probability vector')
    _need(np.all(weights>=0),'Negative probabilities cannot weight lower minorants')
    mass=float(weights.sum());moments=probability_set.features.T@weights
    low=np.maximum(probability_set.lower-moments,0.);high=np.maximum(moments-probability_set.upper,0.)
    violation=max(abs(mass-1.),_max(low),_max(high))
    _need(violation<=TOLERANCE,'Weight vector is not numerically feasible in the full original ambiguity set')
    positive=np.flatnonzero(weights>0)
    _need(set(cuts)==set(positive.tolist()) and all(type(i) is int for i in cuts),'Exactly all positive-weight scenario cuts are required')
    objective=p['c'].copy();constant=p['objective_constant_usd'];rows=[]
    for i in positive:
        record,g=cuts[int(i)];g=_array(g,len(objective),'affine first-stage gradient')
        body=dict(record);digest=body.pop('cut_sha256')
        _need(canonical_hash(body)==digest,'Saved affine cut record changed')
        _need(record['passed'] is True and record['status']=='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT'
              and record['scenario_id']==ids[i] and record['first_stage_columns']==len(objective), 'Wrong or uncertified scenario affine cut')
        _need(record['dual_tolerance']==TOLERANCE and all(math.isfinite(record[k]) and 0<=record[k]<=TOLERANCE
            for k in ('max_dual_stationarity_residual','max_dual_sign_violation','max_infinite_bound_multiplier')),
            'Stored numerical dual cut gate differs')
        _need(record['gradient_sha256']==hashlib.sha256(g.astype('<f8').tobytes()).hexdigest(),'Saved affine gradient changed')
        _need(math.isfinite(record['intercept_usd']),'Nonfinite affine intercept')
        objective+=weights[i]*g;constant+=weights[i]*record['intercept_usd']
        rows.append(dict(index=int(i),scenario_id=ids[i],weight=float(weights[i]),cut_record=record))
    _need(np.isfinite(objective).all() and math.isfinite(constant),'Nonfinite combined first-stage objective')
    result=dict(first_stage_problem,c=objective,objective_constant_usd=float(constant))
    report=dict(bound_kind='NUMERICAL_FIRST_STAGE_LP_BOUND_FROM_FEASIBLE_PROBABILITY_AND_RECOURSE_DUAL_MINORANTS',
        original_first_stage_coefficient_sha256=coefficient_sha256(p),bound_problem_coefficient_sha256=coefficient_sha256(result),
        ambiguity_sha256=probability_set.ambiguity_sha256,support_ids=list(ids),probabilities=weights.tolist(),
        probabilities_sha256=canonical_hash(dict(support_ids=list(ids),probabilities=weights.tolist())),
        probability_mass=mass,probability_moments=moments.tolist(),max_probability_feasibility_violation=violation,
        probability_tolerance=TOLERANCE,positive_indices=positive.tolist(),exact_zero_weight_count=int((weights==0).sum()),
        cuts=rows,original_first_stage_constant_usd=p['objective_constant_usd'],combined_objective_constant_usd=float(constant),
        combined_objective_sha256=hashlib.sha256(objective.astype('<f8').tobytes()).hexdigest(),
        first_stage_constraints_and_variable_bounds_unchanged=True,first_stage_cost_added_once=True,
        bound_relation='min over first-stage LP relaxation of Fpre + sum p_s affine_minorant_s <= full DRO optimum',
        recourse_calls=0,probability_calls=0,lp_calls=0,scientific_acceptance=False,formal_OOS=False,
        negative_objective_coefficients_and_negative_bounds_not_clipped=True,exact_arithmetic_proof=False)
    return result,report
