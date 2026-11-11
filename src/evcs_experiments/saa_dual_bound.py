"""Fixed empirical SAA affine lower-bound objective; no ambiguity optimization.

Consumes independently verified fixed-template dual cuts. Production callers must
bind original matrices, raw duals, support order and constants to the SAA audit.
This constructor neither reads those artifacts nor accepts a numerical result.
"""
import hashlib
import math
import numpy as np
from .random_streams import canonical_hash
from .relaxation_bound import _problem, coefficient_sha256


def build_saa_dual_bound(first_stage_problem, support_ids, cuts, *, constants_usd):
    """Keep the exact original first-stage domain, average ALL scenario cuts.

No probability-set argument is accepted. Every scenario has exactly 1/n mass.
The returned integrality is unchanged; a certified LP relaxation or native MILP
bound may subsequently bound this auxiliary objective, not certify a new plan.
"""
    def need(v,m):
        if not bool(v):raise ValueError(m)
    p=_problem(first_stage_problem);ids=tuple(support_ids);n=len(ids)
    need(n>0 and all(isinstance(s,str) and s for s in ids) and len(set(ids))==n,'Invalid full SAA support')
    constants=np.asarray(constants_usd,dtype=float)
    need(constants.shape==(n,) and np.isfinite(constants).all(),'Invalid original constants')
    need(set(cuts)==set(range(n)) and all(type(i) is int for i in cuts),'Exactly full ordered support cuts required')
    c=p['c'].copy();offset=p['objective_constant_usd'];cut_hashes=[]
    for i in range(n):
        r,g=cuts[i];g=np.asarray(g,dtype=float)
        body=dict(r);digest=body.pop('cut_sha256',None)
        need(digest==canonical_hash(body),'Cut record hash changed')
        need(r['passed'] is True and r['status']=='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT','Unverified numerical cut')
        need(r['scenario_id']==ids[i] and r['first_stage_columns']==len(c),'Wrong support/first-stage axis')
        need(g.shape==c.shape and np.isfinite(g).all(),'Invalid gradient')
        need(r['gradient_sha256']==hashlib.sha256(g.astype('<f8').tobytes()).hexdigest(),'Gradient changed')
        need(r['scenario_constant_usd']==constants[i],'Original scenario constant differs')
        need(r['dual_tolerance']==1e-8 and all(math.isfinite(r[k]) and 0<=r[k]<=1e-8 for k in
             ('max_dual_stationarity_residual','max_dual_sign_violation','max_infinite_bound_multiplier')),'Original dual tolerances differ')
        need(math.isfinite(r['intercept_usd']) and math.isfinite(r['anchor_gap_usd'])
             and abs(r['anchor_gap_usd'])<=r['anchor_gap_tolerance_usd'],'Invalid tight anchor')
        c+=g/n;offset+=r['intercept_usd']/n;cut_hashes.append(digest)
    need(np.isfinite(c).all() and math.isfinite(offset),'Nonfinite aggregate')
    model=dict(first_stage_problem,c=c,objective_constant_usd=float(offset))
    report=dict(kind='SAA_UNIFORM_DUAL_AFFINE_BOUND_MODEL_V1',support_ids=list(ids),
        probabilities=[1./n]*n,constants_usd=constants.tolist(),cut_sha256s=cut_hashes,
        original_coefficient_sha256=coefficient_sha256(p),coefficient_sha256=coefficient_sha256(model),
        risk_certificate_kind='fixed_empirical',first_stage_domain_unchanged=True,
        first_stage_cost_added_once=True,scenario_constants_already_in_cuts=True,
        bound_scope='minimum of original first-stage cost plus uniform affine recourse minorants <= original SAA optimum',
        mathematical_model_changed=False,recourse_calls=0,solver_calls=0,scientific_acceptance=False)
    report['audit_sha256']=canonical_hash(report)
    return model,report
