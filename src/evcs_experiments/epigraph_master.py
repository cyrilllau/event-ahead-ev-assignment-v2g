"""Compact full-support DRO lower master from certified affine cuts and floors.

This auxiliary model preserves the original first-stage feasible set. It does
not contain original recourse copies, does not fix a plan and calls no solver.
The caller binds original template/source bytes and nonempty ambiguity evidence.
"""
import hashlib
import math
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from .dual_cut_bound import TOLERANCE, _array, _need
from .probability import ProbabilitySet
from .random_streams import canonical_hash
from .relaxation_bound import _problem, coefficient_sha256


def _digest(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()


def certify_box_floor(template,*,scenario_constant_usd=0.):
    """Minimum of original linear recourse cost over its unchanged column box.

Rows are dropped only for this lower bound. Zero-cost free variables contribute
exactly zero. Required infinite bounds are rejected; no negative value is clipped.
Original byte/source verification remains the caller's responsibility.
"""
    c=np.asarray(template['c'],dtype=float)
    _need(c.ndim==1 and len(c)>0 and np.isfinite(c).all(),'Invalid original cost')
    lo=_array(template['lower_bounds'],len(c),'original lower bounds',finite=False)
    up=_array(template['upper_bounds'],len(c),'original upper bounds',finite=False)
    _need(np.all(lo<=up) and not np.isposinf(lo).any() and not np.isneginf(up).any(),'Invalid original box')
    sid=template['scenario_id'];th=template['template_sha256']
    _need(isinstance(sid,str) and sid and isinstance(th,str) and th,'Original floor identity required')
    _need(np.ndim(scenario_constant_usd)==0 and math.isfinite(scenario_constant_usd),'Invalid scenario constant')
    pos,neg=c>0,c<0
    _need(np.isfinite(lo[pos]).all() and np.isfinite(up[neg]).all(),'No finite box lower bound')
    parts=c[pos]*lo[pos];negative=c[neg]*up[neg]
    value=float(math.fsum(parts.tolist()+negative.tolist())+float(scenario_constant_usd))
    _need(math.isfinite(value),'Nonfinite box floor')
    record=dict(status='PASS_ORIGINAL_COLUMN_BOX_FLOOR',passed=True,scenario_id=sid,template_sha256=th,
        floor_usd=value,scenario_constant_usd=float(scenario_constant_usd),column_count=len(c),
        cost_sha256=_digest(c),lower_bounds_sha256=_digest(lo),upper_bounds_sha256=_digest(up),
        positive_cost_columns=int(pos.sum()),negative_cost_columns=int(neg.sum()),zero_cost_columns=int((c==0).sum()),
        original_bounds_unchanged=True,rows_dropped_for_lower_bound=True,negative_values_not_clipped=True,
        formula='sum(c_j*lower_j for c_j>0)+sum(c_j*upper_j for c_j<0)+scenario_constant',
        exact_arithmetic_proof=False,solver_calls=0)
    record['floor_sha256']=canonical_hash(record)
    return record


def build_epigraph_master(first_stage_problem,support_ids,cuts,*,probability_set,floors):
    """Return full integer auxiliary model and its source-bound construction record.

cuts maps each original index to (build_dual_cut record, gradient); floors maps
that same complete axis to certify_box_floor records. Auxiliary columns follow
[z, theta0, alpha_lower[K], beta_upper[K]]. For each s, theta0+F_s(beta-alpha)
is bounded below by BOTH its affine cut and its fixed physical box floor.
"""
    p=_problem(first_stage_problem);ids=tuple(support_ids);n=len(p['c']);S=len(ids)
    _need(isinstance(probability_set,ProbabilitySet) and ids==probability_set.support_ids,'Original full ambiguity axis differs')
    _need(isinstance(cuts,dict) and isinstance(floors,dict) and set(cuts)==set(floors)==set(range(S))
          and all(type(i) is int for i in cuts) and all(type(i) is int for i in floors),'Complete original cut and floor axes required')
    K=len(probability_set.feature_ids);gradients=[];intercepts=[];floor_values=[];evidence=[];anchors=set()
    for i,sid in enumerate(ids):
        cut,g=cuts[i];floor=floors[i];g=_array(g,n,'cut gradient')
        body=dict(cut);digest=body.pop('cut_sha256',None)
        _need(canonical_hash(body)==digest,'Affine cut record changed')
        _need(cut['passed'] is True and cut['status']=='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT'
              and cut['scenario_id']==sid and cut['first_stage_columns']==n,'Uncertified or wrong cut identity')
        _need(cut['dual_tolerance']==TOLERANCE and all(math.isfinite(cut[k]) and 0<=cut[k]<=TOLERANCE
            for k in ('max_dual_stationarity_residual','max_dual_sign_violation','max_infinite_bound_multiplier')),'Original dual numerical gate differs')
        _need(cut['gradient_sha256']==_digest(g) and math.isfinite(cut['intercept_usd']),'Changed gradient or intercept')
        body=dict(floor);fd=body.pop('floor_sha256',None)
        _need(canonical_hash(body)==fd,'Physical floor record changed')
        _need(floor['passed'] is True and floor['status']=='PASS_ORIGINAL_COLUMN_BOX_FLOOR'
            and floor['scenario_id']==sid and floor['template_sha256']==cut['template_sha256']
            and floor['scenario_constant_usd']==cut['scenario_constant_usd']
            and floor['original_bounds_unchanged'] is True and floor['rows_dropped_for_lower_bound'] is True
            and math.isfinite(floor['floor_usd']),'Uncertified or mismatched floor identity')
        # The saved optimal anchor cannot lie below a valid physical floor.
        _need(floor['floor_usd']<=cut['anchor_q_usd']+cut['anchor_gap_tolerance_usd'],'Physical floor exceeds certified anchor Q')
        gradients.append(csr_matrix(g.reshape(1,-1)));intercepts.append(cut['intercept_usd']);floor_values.append(floor['floor_usd'])
        anchors.add(cut['anchor_first_stage_sha256'])
        evidence.append(dict(index=i,scenario_id=sid,cut=cut,floor=floor))
    G=vstack(gradients,format='csr');F=csr_matrix(probability_set.features)
    tail=hstack([csr_matrix(np.ones((S,1))),-F,F],format='csr')
    cutsA=hstack([-G,tail],format='csr');floorsA=hstack([csr_matrix((S,n)),tail],format='csr')
    A=vstack([hstack([p['matrix'],csr_matrix((p['matrix'].shape[0],1+2*K))]),cutsA,floorsA],format='csr')
    model=dict(matrix=A,c=np.r_[p['c'],1.,-probability_set.lower,probability_set.upper],
        row_lower=np.r_[p['row_lower'],intercepts,floor_values],row_upper=np.r_[p['row_upper'],np.full(2*S,np.inf)],
        lower_bounds=np.r_[p['lower_bounds'],-np.inf,np.zeros(2*K)],upper_bounds=np.r_[p['upper_bounds'],np.full(1+2*K,np.inf)],
        integrality=np.r_[p['integrality'],np.zeros(1+2*K,dtype=np.int64)],objective_constant_usd=p['objective_constant_usd'])
    record=dict(kind='FULL_SUPPORT_DUAL_CUT_PHYSICAL_FLOOR_MASTER_V1',support_ids=list(ids),support_count=S,
        feature_ids=list(probability_set.feature_ids),ambiguity_sha256=probability_set.ambiguity_sha256,
        dual_variable_convention='alpha_lower_beta_upper',
        row_convention='theta0-F_s@alpha_lower+F_s@beta_upper>=affine_s(z), and >=floor_s',
        objective_convention='Fpre(z)+theta0-lower@alpha_lower+upper@beta_upper',
        original_first_stage_coefficient_sha256=coefficient_sha256(p),master_coefficient_sha256=coefficient_sha256(model),
        original_first_stage_columns=n,first_stage_rows=p['matrix'].shape[0],matrix_shape=list(A.shape),matrix_nnz=A.nnz,
        gradient_nnz=G.nnz,variable_slices=dict(first_stage=[0,n],theta0=[n,n+1],alpha_lower=[n+1,n+1+K],beta_upper=[n+1+K,n+1+2*K]),
        original_first_stage_constraints_bounds_integrality_preserved=True,original_first_stage_cost_added_once=True,
        scenario_constants_in_each_cut_and_floor_once=True,original_first_stage_constant_usd=p['objective_constant_usd'],
        nonempty_original_ambiguity_evidence_required_from_caller=True,anchor_first_stage_sha256s=sorted(anchors),
        cuts_and_floors=evidence,bound_scope='Auxiliary optimum <= original full-support DRO optimum; relaxation also supplies a lower bound',
        integer_auxiliary_incumbent_is_only_candidate=True,original_recourse_solver_calls=0,solver_calls=0,
        scientific_acceptance=False,formal_OOS=False,exact_arithmetic_proof=False)
    record['construction_sha256']=canonical_hash(record)
    return model,record
