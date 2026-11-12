"""Finite-support moment ambiguity LP and its independently solved paper dual.

Rows of features are complete scenario atoms. This module never regenerates
atoms, repairs moment intervals, normalizes returned probabilities, or mutates
an incumbent/RMP dual. Alpha denotes LOWER and beta UPPER moment multipliers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from types import MappingProxyType
from collections.abc import Mapping
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning

from ev_assignment.domain.random_streams import canonical_hash

FEASIBILITY_TOLERANCE = 1e-8
# Solver tolerances apply to the normalized LP. The independent certificate
# below still requires its original 1e-8 USD absolute dual-feasibility gate.
# Tighter simplex tolerances reduce loss when the raw dual is scaled to USD;
# they do not guarantee certification at arbitrary scales or repair a witness.
SOLVER_FEASIBILITY_TOLERANCE = 1e-10
OBJECTIVE_ABSOLUTE_TOLERANCE = 1e-4
OBJECTIVE_RELATIVE_TOLERANCE = 1e-7
SEPARATION_ABSOLUTE_TOLERANCE = 1e-4
SEPARATION_RELATIVE_TOLERANCE = 1e-8


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _frozen(value):
    if isinstance(value,np.ndarray):
        a=np.ascontiguousarray(value)
        return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
    if isinstance(value,Mapping): return MappingProxyType({k:_frozen(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)): return tuple(_frozen(x) for x in value)
    return value


def _json(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,Mapping): return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json(v) for v in value]
    if isinstance(value,np.generic): return value.item()
    return value


def _ids(values,name,nonempty=False):
    _require(not isinstance(values,(str,bytes)),f'{name} must be an explicit sequence')
    result=tuple(values)
    _require((not nonempty or len(result)>0) and all(isinstance(x,str) and x for x in result)
             and len(set(result))==len(result),f'{name} must have unique nonempty string IDs')
    return result


@dataclass(frozen=True)
class ProbabilitySet:
    support_ids: tuple[str,...]
    feature_ids: tuple[str,...]
    features: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ambiguity_sha256: str = field(init=False)

    def __post_init__(self):
        ids=_ids(self.support_ids,'support_ids',True); features_ids=_ids(self.feature_ids,'feature_ids')
        phi=np.asarray(self.features,dtype=np.float64)
        lo=np.asarray(self.lower,dtype=np.float64);hi=np.asarray(self.upper,dtype=np.float64)
        _require(phi.shape==(len(ids),len(features_ids)) and np.isfinite(phi).all(),
                 'features must be finite [scenario,feature], without implicit transpose')
        _require(lo.shape==hi.shape==(len(features_ids),) and np.isfinite(lo).all() and np.isfinite(hi).all(),
                 'Moment bounds must be finite aligned feature vectors')
        _require(np.all(lo<=hi),'Lower moment bound exceeds upper bound')
        payload={'support_ids':list(ids),'feature_ids':list(features_ids),'features':phi.tolist(),
                 'lower':lo.tolist(),'upper':hi.tolist(),'feature_orientation':'scenario_by_feature',
                 'dual_convention':'alpha_lower_beta_upper'}
        for key,value in [('support_ids',ids),('feature_ids',features_ids),('features',_frozen(phi)),
                          ('lower',_frozen(lo)),('upper',_frozen(hi)),('ambiguity_sha256',canonical_hash(payload))]:
            object.__setattr__(self,key,value)

    def as_dict(self):
        return {'support_ids':list(self.support_ids),'feature_ids':list(self.feature_ids),
                'features':self.features.tolist(),'lower':self.lower.tolist(),'upper':self.upper.tolist(),
                'ambiguity_sha256':self.ambiguity_sha256,'feature_orientation':'scenario_by_feature',
                'dual_convention':'alpha_lower_beta_upper'}


@dataclass(frozen=True)
class WorstDistributionResult:
    status: str
    certified_optimal: bool
    probabilities: np.ndarray | None
    theta0: float | None
    alpha: np.ndarray | None
    beta: np.ndarray | None
    primal_objective_usd: float | None
    dual_objective_usd: float | None
    lower_bound_usd: float | None
    upper_bound_usd: float | None
    support_ids: tuple[str,...]
    feature_ids: tuple[str,...]
    ambiguity_sha256: str
    q_sha256: str
    certificate: Mapping
    primal_solver: Mapping
    dual_solver: Mapping
    wall_seconds: float

    def __post_init__(self):
        for key in ('probabilities','alpha','beta','certificate','primal_solver','dual_solver'):
            object.__setattr__(self,key,_frozen(getattr(self,key)))

    def as_dict(self):
        return {key:_json(value) for key,value in vars(self).items()}


def _solver(c,a_ub,b_ub,a_eq,b_eq,bounds,time_limit):
    options={'time_limit':float(time_limit),'threads':1,'primal_feasibility_tolerance':SOLVER_FEASIBILITY_TOLERANCE,
             'dual_feasibility_tolerance':SOLVER_FEASIBILITY_TOLERANCE}
    began=time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
        raw=linprog(c,A_ub=a_ub,b_ub=b_ub,A_eq=a_eq,b_eq=b_eq,bounds=bounds,method='highs-ds',options=options)
    record={'solver':'scipy.optimize.linprog/HiGHS dual simplex','status':int(raw.status),
            'success':bool(raw.success),'message':str(raw.message),'iterations':int(getattr(raw,'nit',0) or 0),
            'options':options,'wall_seconds':time.monotonic()-began,
            'raw_objective_scaled':None if raw.fun is None else float(raw.fun),
            'raw_solution_scaled':None if raw.x is None else np.asarray(raw.x,dtype=float).copy(),
            'infeasibility_certificate_available':False}
    for name in ('ineqlin','eqlin','lower','upper'):
        group=getattr(raw,name,None)
        if group is not None and getattr(group,'marginals',None) is not None:
            record[name+'_marginals_scaled']=np.asarray(group.marginals,dtype=float).copy()
    return raw,record


def solve_worst_distribution(probability_set,q,*,q_support_ids,time_limit_seconds=30):
    """Solve max_p q·p and min(theta-lower·alpha+upper·beta) independently.

    Both LPs share a total wall-clock budget (the first receives at most half).
    q_support_ids must equal the stored row order exactly. Raw p is retained:
    diagnostic support thresholding never changes a distribution or its value.
    The output bounds are numerically verified within the recorded tolerances;
    no exact-arithmetic or Farkas infeasibility certificate is implied.
    Both LPs use (q-min(q))/scale. The simplex equality and free dual intercept
    make this an equivalent numerical formulation; certification uses original
    q, dollars, and the original cost magnitude, not the optimization scale.
    """
    began=time.monotonic()
    _require(isinstance(probability_set,ProbabilitySet),'Expected ProbabilitySet')
    a=probability_set;ids=_ids(q_support_ids,'q_support_ids',True)
    _require(ids==a.support_ids,'q support IDs/order differ from ambiguity support')
    values=np.asarray(q,dtype=np.float64)
    _require(values.shape==(len(ids),) and np.isfinite(values).all(),'q must be a finite aligned scenario vector')
    _require(not isinstance(time_limit_seconds,(bool,np.bool_)) and np.isfinite(time_limit_seconds)
             and time_limit_seconds>0,'Positive finite total time limit required')
    q_hash=canonical_hash({'support_ids':list(ids),'q':values.tolist()})
    offset=float(np.min(values))
    with np.errstate(over='ignore',invalid='ignore'):
        centered=values-offset
    _require(np.isfinite(centered).all(),'Centered q range exceeds finite float64 representation')
    scale=max(1.,float(np.max(np.abs(centered))))
    scaled_values=centered/scale
    certification_scale=max(1.,float(np.max(np.abs(values))))
    sep_tol=SEPARATION_ABSOLUTE_TOLERANCE+SEPARATION_RELATIVE_TOLERANCE*certification_scale
    tol=FEASIBILITY_TOLERANCE;S,F=a.features.shape
    certificate={'q_offset_usd':offset,'q_scale_usd':scale,
        'certification_scale_usd':certification_scale,'feature_orientation':'scenario_by_feature',
        'objective_transform':{
            'name':'subtract_minimum_then_scale',
            'scaled_cost':'(q - q_offset_usd) / q_scale_usd',
            'primal_solver_objective':'-scaled_cost @ p',
            'dual_solver_rhs':'-scaled_cost',
            'primal_objective_recovery':'q_offset_usd * sum(p) - q_scale_usd * raw_objective_scaled',
            'dual_theta0_recovery':'q_offset_usd + q_scale_usd * raw_theta0',
            'dual_multiplier_recovery':'q_scale_usd * raw_multiplier',
            'raw_primal_witness':'primal_solver.raw_solution_scaled',
            'raw_dual_witness':'dual_solver.raw_solution_scaled'},
        'dual_convention':'theta0 - lower@alpha + upper@beta; alpha=lower, beta=upper',
        'primal_problem':'max q@p: p>=0, sum(p)=1, lower<=features.T@p<=upper',
        'dual_problem':'min theta0-lower@alpha+upper@beta: theta0+features@(beta-alpha)>=q; alpha,beta>=0; theta0 free',
        'feasibility_tolerance':tol,'dual_original_feasibility_tolerance_usd':tol,'objective_abs_tolerance_usd':OBJECTIVE_ABSOLUTE_TOLERANCE,
        'objective_rel_tolerance':OBJECTIVE_RELATIVE_TOLERANCE,
        'separation_tolerance_usd':sep_tol,'separation_scale_usd':certification_scale,
        'bounds_kind':'numerically verified within recorded tolerances',
        'probabilities_clipped_or_renormalized':False,'moment_bounds_repaired':False,
        'raw_RMP_dual_replaced':False,'phase_I_initialization_in_scope':False,
        'primal_feasible':False,'dual_feasible':False,'gap_passed':False,
        'complementarity_passed':False,'analytic_infeasibility':None}
    not_run={'status':None,'success':False,'message':'Not run','raw_solution_scaled':None,
             'infeasibility_certificate_available':False}
    # A mathematically inconsistent constant column must not be accepted merely
    # because a solver tolerates a tiny mismatch. This does not widen its bounds.
    constants=np.all(a.features==a.features[:1],axis=0)
    bad=constants & ((a.features[0]<a.lower)|(a.features[0]>a.upper))
    if np.any(bad):
        certificate['analytic_infeasibility']={'kind':'constant_feature_outside_moment_interval',
            'feature_ids':[a.feature_ids[i] for i in np.flatnonzero(bad)],
            'constant_values':a.features[0,bad].tolist(),'lower':a.lower[bad].tolist(),'upper':a.upper[bad].tolist()}
        return WorstDistributionResult('INFEASIBLE',False,None,None,None,None,None,None,None,None,
            ids,a.feature_ids,a.ambiguity_sha256,q_hash,certificate,not_run,not_run,time.monotonic()-began)
    A=np.vstack((a.features.T,-a.features.T));b=np.concatenate((a.upper,-a.lower))
    primal,primal_record=_solver(-scaled_values,A,b,np.ones((1,S)),np.ones(1),[(0.,None)]*S,
                                  float(time_limit_seconds)/2.)
    # Independent paper dual, not a relabeling of the primal solver marginals.
    remaining=float(time_limit_seconds)-(time.monotonic()-began)
    dual=None;dual_record=not_run
    if remaining>0:
        cdual=np.concatenate(([1.],-a.lower,a.upper))
        Adu=np.column_stack((-np.ones(S),a.features,-a.features))
        dual,dual_record=_solver(cdual,Adu,-scaled_values,None,None,
                                [(None,None)]+[(0.,None)]*(2*F),remaining)
    p=None if primal.x is None else np.asarray(primal.x,dtype=float).copy()
    theta=alpha=beta=None
    if dual is not None and dual.x is not None:
        z=np.asarray(dual.x,dtype=float)*scale
        theta=offset+float(z[0]);alpha=z[1:1+F].copy();beta=z[1+F:].copy()
    primal_obj=dual_obj=None
    primal_ok=dual_ok=False
    feature_scale=np.maximum(1.,np.maximum(np.max(np.abs(a.features),axis=0),np.maximum(np.abs(a.lower),np.abs(a.upper))))
    means=None;slack=None
    if p is not None and np.isfinite(p).all():
        means=a.features.T@p
        moment_violation=np.maximum(np.maximum(a.lower-means,means-a.upper),0.)
        normalization=abs(float(np.sum(p))-1.);negative=float(np.max(-p,initial=0.))
        primal_obj=float(values@p)
        primal_ok=bool(np.isfinite(primal_obj) and max(normalization,negative,float(np.max(moment_violation,initial=0.)))<=tol
                       and float(np.max(moment_violation/feature_scale,initial=0.))<=tol)
        certificate['primal']={'sum_probability':float(p.sum()),'normalization_residual':normalization,
            'minimum_probability':float(np.min(p)),'nonnegative_violation':negative,
            'feature_means':means.copy(),'lower_slack':means-a.lower,'upper_slack':a.upper-means,
            'moment_violation':moment_violation,'max_moment_violation':float(np.max(moment_violation,initial=0.)),
            'max_normalized_moment_violation':float(np.max(moment_violation/feature_scale,initial=0.)),
            'feature_scale':feature_scale.copy(),'finite':True}
        certificate['positive_support_ids_thresholded']=[ids[i] for i in np.flatnonzero(p>tol)]
        certificate['positive_support_threshold']=tol
        certificate['raw_positive_probability_count']=int(np.count_nonzero(p>0))
        certificate['reduced_extreme_point_support_computed']=False
    if theta is not None and np.isfinite(theta) and np.isfinite(alpha).all() and np.isfinite(beta).all():
        slack=theta+a.features@(beta-alpha)-values
        sign_violation=max(float(np.max(-alpha,initial=0.)),float(np.max(-beta,initial=0.)))
        support_violation=float(np.max(-slack,initial=0.))
        dual_obj=float(theta-a.lower@alpha+a.upper@beta)
        dual_ok=bool(np.isfinite(dual_obj) and max(sign_violation,support_violation)<=tol
                     and max(sign_violation,support_violation)/certification_scale<=tol)
        certificate['dual']={'theta0_free':True,'alpha_lower':alpha.copy(),'beta_upper':beta.copy(),
            'alpha_minimum':float(np.min(alpha)) if F else None,'beta_minimum':float(np.min(beta)) if F else None,
            'sign_violation_usd':sign_violation,'support_slack_usd':slack.copy(),
            'support_residual_usd':-slack.copy(),'max_support_violation_usd':support_violation,
            'max_normalized_support_violation':support_violation/certification_scale,
            'normalized_sign_violation':sign_violation/certification_scale}
    gap_passed=False;comp_passed=False
    if primal_obj is not None and dual_obj is not None and np.isfinite(primal_obj) and np.isfinite(dual_obj):
        gap=dual_obj-primal_obj
        gap_tol=OBJECTIVE_ABSOLUTE_TOLERANCE+OBJECTIVE_RELATIVE_TOLERANCE*max(1.,abs(primal_obj),abs(dual_obj))
        gap_passed=abs(gap)<=gap_tol
        certificate.update(signed_duality_gap_usd=gap,absolute_duality_gap_usd=abs(gap),duality_gap_tolerance_usd=gap_tol)
        if means is not None and slack is not None:
            products=np.concatenate((p*slack,alpha*(means-a.lower),beta*(a.upper-means)))
            complementarity=float(np.max(np.abs(products),initial=0.))
            comp_passed=bool(np.isfinite(complementarity) and complementarity<=gap_tol)
            certificate.update(complementarity_products_usd=products,max_complementarity_usd=complementarity)
    both_optimal=primal.status==0 and dual is not None and dual.status==0
    certified=bool(both_optimal and primal_ok and dual_ok and gap_passed and comp_passed)
    certificate.update(primal_feasible=primal_ok,dual_feasible=dual_ok,gap_passed=bool(gap_passed),
                       complementarity_passed=comp_passed,certified_optimal=certified,
                       independent_primal_and_dual_solved=dual is not None)
    if certified: status='OPTIMAL'
    elif primal.status==2: status='INFEASIBLE'
    elif primal.status==1 or dual is None or dual.status==1: status='TIME_LIMIT'
    elif primal.status==3: status='NUMERICAL_ERROR'  # A nonempty finite simplex cannot be unbounded.
    else: status='NUMERICAL_CERTIFICATE_FAILED'
    return WorstDistributionResult(status,certified,p,theta,alpha,beta,primal_obj,dual_obj,
        primal_obj if primal_ok else None,dual_obj if dual_ok else None,ids,a.feature_ids,
        a.ambiguity_sha256,q_hash,certificate,primal_record,dual_record,time.monotonic()-began)
