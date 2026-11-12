"""One deterministic Case0 scenario selection and one truthful partial cut row.

No solver or recourse assembler is imported. Original cut admissibility (raw
A/B/RHS/primal/dual/constant) must already be independently established by the
caller. This component binds that evidence and checks partial-row assembly.
"""
import hashlib,math,time
import numpy as np
from scipy.sparse import csr_matrix,vstack
from evcs_experiments.campaign_batch import verify_reference
from evcs_experiments.native_gurobi import _check_vector
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.relaxation_bound import coefficient_sha256

TOL=1e-8
POLICY='old_p_weighted_two_cut_disagreement_then_p_then_support_index_v1'

def _need(ok,msg):
    if not bool(ok):raise ValueError(msg)
def _alive(deadline):
    if not math.isfinite(deadline) or time.monotonic()>=deadline:raise TimeoutError('Partial-scenario component deadline')
def _hash(v):return hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest()
def _seal(body,key='audit_sha256'):return dict(body,**{key:canonical_hash(body)})
def _unseal(record,key,label):
    body=dict(record);digest=body.pop(key);_need(canonical_hash(body)==digest,label+' hash differs');return body

def _refs(refs):
    _need(isinstance(refs,list) and len(refs)>0,'Explicit admitted source references required')
    _need(len({r['path'] for r in refs})==len(refs),'Duplicate source reference')
    for ref in refs:verify_reference(ref)

def _base(model,audit,P):
    b=_unseal(audit,'audit_sha256','Accepted base audit');A=csr_matrix(model['matrix']);ids=tuple(P.support_ids);K=len(P.feature_ids);n=A.shape[1]-1-2*K
    _need(b['kind']=='INDEPENDENT_MULTI_ANCHOR_MODEL_AUDIT_V1' and b['passed'] is True and b['anchor_count']==2 and
          b['coefficient_sha256']==coefficient_sha256(model) and b['support_ids']==list(ids) and b['ambiguity_sha256']==P.ambiguity_sha256,
          'Accepted two-anchor model/P identity differs')
    _need(n>0 and len(ids)>0 and A.shape[0]>=3*len(ids) and np.array_equal(np.asarray(model['c'])[n:],np.r_[1.,-P.lower,P.upper]) and
          np.array_equal(np.asarray(model['lower_bounds'])[n:],np.r_[-np.inf,np.zeros(2*K)]) and
          np.array_equal(np.asarray(model['upper_bounds'])[n:],np.full(1+2*K,np.inf)) and
          np.array_equal(np.asarray(model['integrality'])[n:],np.zeros(1+2*K)), 'Shared original P dual columns/objective/domain differ')
    return A,ids,n

def _cut(cut,g,n,sid,*,anchor=None,template=None,constant=None):
    _unseal(cut,'cut_sha256','Numerical cut');g=np.asarray(g,dtype='<f8')
    _need(cut['passed'] is True and cut['status']=='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT' and cut['scenario_id']==sid and
          cut['first_stage_columns']==n and g.shape==(n,) and np.isfinite(g).all() and cut['gradient_sha256']==_hash(g) and
          math.isfinite(cut['intercept_usd']) and cut['dual_tolerance']==TOL and
          all(math.isfinite(cut[k]) and 0<=cut[k]<=TOL for k in ('max_dual_stationarity_residual','max_dual_sign_violation','max_infinite_bound_multiplier')),
          'Original verified cut/gradient/numerical identity differs')
    if template is not None:_need(cut['template_sha256']==template,'Selected original template differs')
    if constant is not None:_need(cut['scenario_constant_usd']==constant,'Selected original scenario constant differs')
    if anchor is not None:
        z=np.asarray(anchor,float);value=float(cut['intercept_usd']+g@z);q=cut['anchor_q_usd']
        _need(z.shape==(n,) and cut['anchor_first_stage_sha256']==_hash(z) and type(q) in (int,float) and math.isfinite(q),'Cut anchor differs from selected plan')
        tol=1e-4+1e-7*max(1.,abs(q),abs(value))
        _need(cut['anchor_gap_tolerance_usd']==tol and abs(value-cut['anchor_affine_value_usd'])<=tol and abs(q-value)<=tol and
              abs((q-value)-cut['anchor_gap_usd'])<=tol,'Selected cut is not certified at its exact anchor')
    return g

def _probabilities(P,probability):
    ids=tuple(P.support_ids);p=np.asarray(probability['probabilities'],float)
    _need(probability['support_ids']==list(ids) and probability['feature_ids']==list(P.feature_ids) and
          probability['ambiguity_sha256']==P.ambiguity_sha256 and p.shape==(len(ids),) and np.isfinite(p).all(),'Fixed old probability identity differs')
    means=P.features.T@p
    _need(max(float(np.max(-p,initial=0.)),abs(float(p.sum())-1.),float(np.max(P.lower-means,initial=0.)),
              float(np.max(means-P.upper,initial=0.)))<=TOL,'Fixed selection probability is not feasible in original P')
    return p

def select_scenario(base_model,base_audit,P,cut_sets,old_probability,full_incumbent,*,source_references,deadline):
    """Persist this sealed record before the single selected original Q call."""
    _alive(deadline);_refs(source_references);A,ids,n=_base(base_model,base_audit,P)
    common=dict(kind='CASE0_PARTIAL_SCENARIO_SELECTION_V1',policy=POLICY,base_coefficient_sha256=coefficient_sha256(base_model),
        base_audit_sha256=base_audit['audit_sha256'],support_ids=list(ids),ambiguity_sha256=P.ambiguity_sha256,
        source_references=source_references,scientific_acceptance=False,full_candidate_evaluated=False,new_UB_verified=False,
        full_support_new_anchor=False,formal_OOS=False,selection_is_error_certificate=False)
    if full_incumbent is None:
        return _seal(dict(common,status='SKIP_NO_AUXILIARY_INCUMBENT',selected_index=None,selected_scenario_id=None,master_call_required=False),'selection_sha256')
    v=np.asarray(full_incumbent,float);check=_check_vector(base_model['c'],base_model['integrality'],base_model['lower_bounds'],base_model['upper_bounds'],A,base_model['row_lower'],base_model['row_upper'],v)
    _need(check['passed'],'Saved full auxiliary incumbent violates accepted original arrays')
    p=_probabilities(P,old_probability);z=v[:n];S=len(ids)
    _need(isinstance(cut_sets,(tuple,list)) and len(cut_sets)==2 and all(set(group)==set(range(S)) and all(type(i) is int for i in group) for group in cut_sets),
          'Exactly two complete previously audited cut sets required')
    rows=[];cut_hashes=[];m=A.shape[0]-3*S
    for group_index,group in enumerate(cut_sets):
        for i,sid in enumerate(ids):
            cut,g=group[i];g=_cut(cut,g,n,sid);cut_hashes.append(cut['cut_sha256'])
            row=m+i if group_index==0 else m+2*S+i
            _need(np.array_equal(A[row,:n].toarray().ravel(),-g) and
                  np.array_equal(A[row,n:].toarray().ravel(),np.r_[1.,-P.features[i],P.features[i]]) and
                  base_model['row_lower'][row]==cut['intercept_usd'] and np.isposinf(base_model['row_upper'][row]),
                  'Selection old cut differs from accepted model row')
    _need(cut_hashes==base_audit['cut_sha256s'],'Selection cut records differ from accepted two-anchor audit')
    for i,sid in enumerate(ids):
        c0,g0=cut_sets[0][i];c1,g1=cut_sets[1][i]
        _need(c0['template_sha256']==c1['template_sha256'] and c0['scenario_constant_usd']==c1['scenario_constant_usd'],'Original scenario template/constant differs across anchors')
        a=float(c0['intercept_usd']+np.asarray(g0)@z);b=float(c1['intercept_usd']+np.asarray(g1)@z);score=float(p[i]*abs(a-b))
        _need(all(math.isfinite(x) for x in (a,b,score)),'Nonfinite deterministic selection score')
        rows.append(dict(index=i,scenario_id=sid,probability=float(p[i]),cut_values_usd=[a,b],score_usd=score,
            template_sha256=c0['template_sha256'],scenario_constant_usd=c0['scenario_constant_usd']))
    order=sorted(range(S),key=lambda i:(-rows[i]['score_usd'],-rows[i]['probability'],i));i=order[0]
    _alive(deadline)
    return _seal(dict(common,status='ONE_SCENARIO_PRESELECTED',selected_index=i,selected_scenario_id=ids[i],scores=rows,ranking=order,
        selected_template_sha256=rows[i]['template_sha256'],selected_scenario_constant_usd=rows[i]['scenario_constant_usd'],
        first_stage_values=z.tolist(),first_stage_sha256=_hash(z),full_incumbent_sha256=_hash(v),first_stage_columns=n,
        original_full_incumbent_check=check,probabilities=p.tolist(),old_probability_canonical_sha256=canonical_hash(old_probability),
        original_cut_sha256s=cut_hashes,at_most_one_new_scenario=True),'selection_sha256')

def _selection(selection,model,audit,P,v):
    _unseal(selection,'selection_sha256','Preselection');A,ids,n=_base(model,audit,P);i=selection['selected_index']
    _need(selection['kind']=='CASE0_PARTIAL_SCENARIO_SELECTION_V1' and selection['status']=='ONE_SCENARIO_PRESELECTED' and
          selection['policy']==POLICY and selection['base_coefficient_sha256']==coefficient_sha256(model) and
          selection['base_audit_sha256']==audit['audit_sha256'] and selection['support_ids']==list(ids) and selection['ambiguity_sha256']==P.ambiguity_sha256 and
          type(i) is int and 0<=i<len(ids) and selection['selected_scenario_id']==ids[i], 'Selected original model/P/support differs')
    v=np.asarray(v,float);z=np.asarray(selection['first_stage_values'],float)
    _need(v.shape==(A.shape[1],) and np.isfinite(v).all() and z.shape==(n,) and np.array_equal(v[:n],z) and
          _hash(v)==selection['full_incumbent_sha256'] and _hash(z)==selection['first_stage_sha256'], 'Saved full incumbent/selected first stage changed')
    _need(_check_vector(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],A,model['row_lower'],model['row_upper'],v)['passed'],
          'Selected original incumbent infeasible')
    _refs(selection['source_references'])
    _need(selection['original_cut_sha256s']==audit['cut_sha256s'],'Selection original cut evidence differs')
    p=_probabilities(P,dict(probabilities=selection['probabilities'],support_ids=list(ids),feature_ids=list(P.feature_ids),ambiguity_sha256=P.ambiguity_sha256))
    scores=selection['scores'];_need(len(scores)==len(ids),'Selection score axis differs');m=A.shape[0]-3*len(ids)
    for j,row in enumerate(scores):
        a,b=row['cut_values_usd'];prob=row['probability']
        expected=[float(model['row_lower'][k]+(-A[k,:n].toarray().ravel())@z) for k in (m+j,m+2*len(ids)+j)]
        _need(row['cut_values_usd']==expected and prob==p[j],'Selection values differ from accepted rows or saved probability')
        _need(row['index']==j and row['scenario_id']==ids[j] and math.isfinite(prob) and row['score_usd']==prob*abs(a-b) and
              math.isfinite(row['score_usd']),'Deterministic selection score changed')
    order=sorted(range(len(ids)),key=lambda j:(-scores[j]['score_usd'],-scores[j]['probability'],j))
    _need(selection['ranking']==order and i==order[0] and selection['selected_template_sha256']==scores[i]['template_sha256'] and
          selection['selected_scenario_constant_usd']==scores[i]['scenario_constant_usd'],'Selected scenario does not match fixed ranking')
    return A,ids,n,i,v,z

def audit_partial_model(model,base_model,base_audit,P,selection,cut,gradient,full_incumbent,*,source_references,deadline):
    """Independent original-prefix/one-row check; never relabel as a full anchor."""
    _alive(deadline);_refs(source_references);B,ids,n,i,v,z=_selection(selection,base_model,base_audit,P,full_incumbent)
    g=_cut(cut,gradient,n,ids[i],anchor=z,template=selection['selected_template_sha256'],constant=selection['selected_scenario_constant_usd'])
    A=csr_matrix(model['matrix']);m=B.shape[0]
    _need(A.shape==(m+1,B.shape[1]) and np.isfinite(A.data).all() and
          np.array_equal(A[:m].data,B.data) and np.array_equal(A[:m].indices,B.indices) and np.array_equal(A[:m].indptr,B.indptr),
          'Retained complete auxiliary matrix prefix changed')
    for k in ('row_lower','row_upper'):_need(np.array_equal(np.asarray(model[k])[:m],base_model[k]) and len(model[k])==m+1,'Retained original row bounds changed')
    for k in ('c','integrality','lower_bounds','upper_bounds'):_need(np.array_equal(model[k],base_model[k]),'Original objective/domain changed: '+k)
    _need(model['objective_constant_usd']==base_model['objective_constant_usd'],'Original objective constant changed')
    _need(np.array_equal(A[m,:n].toarray().ravel(),-g) and np.array_equal(A[m,n:].toarray().ravel(),np.r_[1.,-P.features[i],P.features[i]]) and
          model['row_lower'][m]==cut['intercept_usd'] and np.isposinf(model['row_upper'][m]), 'Selected partial row gradient/P/sign/constant differs')
    activity=float((A[m]@v).item());violation=max(0.,float(cut['intercept_usd'])-activity)
    scale=max(1.,float((abs(A[m])@abs(v)).item()));normalized=violation/scale
    excluded=violation>TOL or normalized>TOL
    proof=dict(kind='INDEPENDENT_CASE0_PARTIAL_SCENARIO_MODEL_AUDIT_V1',passed=True,coefficient_sha256=coefficient_sha256(model),
        base_coefficient_sha256=coefficient_sha256(base_model),base_audit_sha256=base_audit['audit_sha256'],selection_sha256=selection['selection_sha256'],
        support_ids=list(ids),ambiguity_sha256=P.ambiguity_sha256,selected_index=i,selected_scenario_id=ids[i],cut_sha256=cut['cut_sha256'],
        source_references=source_references,retained_prefix_rows=m,appended_rows=1,full_support_new_anchor=False,
        newly_evaluated_scenario_indices=[i],newly_evaluated_scenario_count=1,retained_objective_domain_and_P=True,
        full_candidate_evaluated=False,new_UB_verified=False,formal_OOS=False,scientific_acceptance=False,solver_calls=0,
        old_full_incumbent_sha256=_hash(v),new_row_activity_usd=activity,new_row_lower_usd=float(cut['intercept_usd']),
        incumbent_row_violation=violation,incumbent_row_normalized_violation=normalized,incumbent_excluded=excluded,
        master_call_required=excluded,skip_requires_previously_accepted_auxiliary_optimality=True,
        skip_reason=None if excluded else 'Saved full incumbent satisfies the new row. Caller must bind its previously accepted auxiliary optimality before concluding this row cannot improve the optimum.')
    _alive(deadline);return _seal(proof)

def append_partial_cut(base_model,base_audit,P,selection,cut,gradient,full_incumbent,*,source_references,deadline):
    _alive(deadline);B,ids,n,i,v,z=_selection(selection,base_model,base_audit,P,full_incumbent)
    g=_cut(cut,gradient,n,ids[i],anchor=z,template=selection['selected_template_sha256'],constant=selection['selected_scenario_constant_usd'])
    row=csr_matrix(np.r_[-g,1.,-P.features[i],P.features[i]].reshape(1,-1))
    model=dict(matrix=vstack([B,row],format='csr'),objective_constant_usd=base_model['objective_constant_usd'],
        row_lower=np.r_[base_model['row_lower'],cut['intercept_usd']],row_upper=np.r_[base_model['row_upper'],np.inf],
        **{k:np.asarray(base_model[k]).copy() for k in ('c','integrality','lower_bounds','upper_bounds')})
    proof=audit_partial_model(model,base_model,base_audit,P,selection,cut,g,v,source_references=source_references,deadline=deadline)
    return model,proof,dict(master_call_required=proof['master_call_required'],incumbent_excluded=proof['incumbent_excluded'],
        raw_violation=proof['incumbent_row_violation'],normalized_violation=proof['incumbent_row_normalized_violation'],skip_reason=proof['skip_reason'])
