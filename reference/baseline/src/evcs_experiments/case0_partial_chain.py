"""One retained partial-cut continuation; no recourse or native solver calls.

The driver independently establishes each raw original LP/KKT certificate.
This layer verifies its admitted cut/anchor provenance, all retained rows and
one new row. It never replaces a partial anchor by a complete support anchor.
"""
import math
import numpy as np
from scipy.sparse import csr_matrix,vstack
from evcs_experiments import case0_partial_cut as old
from evcs_experiments.relaxation_bound import coefficient_sha256

POLICY='old_p_weighted_all_retained_cut_range_then_p_then_support_index_v1'
AUDIT_KIND='INDEPENDENT_CASE0_PARTIAL_CHAIN_MODEL_AUDIT_V1'


def _prefix(model,base,extra):
    A=csr_matrix(model['matrix']);B=csr_matrix(base['matrix']);m=B.shape[0]
    old._need(A.shape==(m+extra,B.shape[1]) and np.isfinite(A.data).all() and
        np.array_equal(A[:m].data,B.data) and np.array_equal(A[:m].indices,B.indices) and
        np.array_equal(A[:m].indptr,B.indptr),'Retained original matrix prefix changed')
    for k in ('row_lower','row_upper'):
        old._need(len(model[k])==m+extra and np.array_equal(np.asarray(model[k])[:m],base[k]),'Retained original row bounds changed')
    for k in ('c','integrality','lower_bounds','upper_bounds'):
        old._need(np.array_equal(model[k],base[k]),'Retained original objective/domain changed')
    old._need(model['objective_constant_usd']==base['objective_constant_usd'],'Retained original objective constant changed')
    return A,m


def _identity(entry):
    c=entry['cut']
    return dict(index=entry['index'],scenario_id=c['scenario_id'],cut_sha256=c['cut_sha256'],
        anchor_first_stage_sha256=c['anchor_first_stage_sha256'],gradient_sha256=c['gradient_sha256'],
        source_references=entry['source_references'])


def _chain(model,audit,P,original_model,original_audit,retained_cuts):
    B,ids,n=old._base(original_model,original_audit,P)
    old._need(isinstance(retained_cuts,list) and len(retained_cuts)>=1,'Explicit previously accepted partial cuts required')
    A,m=_prefix(model,original_model,len(retained_cuts));seen=set();identities=[]
    for j,e in enumerate(retained_cuts):
        old._need(set(e)=={'index','cut','gradient','anchor','source_references'},'Retained partial record schema differs')
        i=e['index'];old._need(type(i) is int and 0<=i<len(ids),'Retained partial support index differs')
        old._refs(e['source_references']);g=old._cut(e['cut'],e['gradient'],n,ids[i],anchor=e['anchor'])
        key=(e['cut']['anchor_first_stage_sha256'],ids[i]);old._need(key not in seen,'Duplicate retained anchor/scenario');seen.add(key)
        old._need(np.array_equal(A[m+j,:n].toarray().ravel(),-g) and
            np.array_equal(A[m+j,n:].toarray().ravel(),np.r_[1.,-P.features[i],P.features[i]]) and
            model['row_lower'][m+j]==e['cut']['intercept_usd'] and np.isposinf(model['row_upper'][m+j]),
            'Retained partial row/P/constant differs')
        identities.append(_identity(e))
    body=old._unseal(audit,'audit_sha256','Accepted partial chain')
    old._need(body['passed'] is True and body['coefficient_sha256']==coefficient_sha256(model) and
        body['support_ids']==list(ids) and body['ambiguity_sha256']==P.ambiguity_sha256,'Accepted partial model/P differs')
    if body['kind']=='INDEPENDENT_CASE0_PARTIAL_SCENARIO_MODEL_AUDIT_V1':
        e=retained_cuts[0]
        old._need(len(retained_cuts)==1 and body['base_coefficient_sha256']==coefficient_sha256(original_model) and
            body['base_audit_sha256']==original_audit['audit_sha256'] and body['retained_prefix_rows']==m and body['appended_rows']==1 and
            body['cut_sha256']==e['cut']['cut_sha256'] and body['selected_index']==e['index'] and
            body['selected_scenario_id']==e['cut']['scenario_id'] and body['source_references']==e['source_references'] and
            body['full_support_new_anchor'] is False,'Prior single-partial provenance differs')
    else:
        old._need(body['kind']==AUDIT_KIND and body['original_coefficient_sha256']==coefficient_sha256(original_model) and
            body['original_audit_sha256']==original_audit['audit_sha256'] and body['retained_cut_identities']==identities and
            body['total_partial_rows']==len(retained_cuts) and body['full_support_new_anchor'] is False,
            'Prior retained-chain provenance differs')
    return A,ids,n,identities


def select_scenario(chain_model,chain_audit,P,cut_sets,old_probability,full_incumbent,*,
                    original_model,original_audit,retained_cuts,source_references,deadline):
    old._alive(deadline);old._refs(source_references)
    A,ids,n,identities=_chain(chain_model,chain_audit,P,original_model,original_audit,retained_cuts)
    common=dict(kind='CASE0_PARTIAL_CHAIN_SELECTION_V1',policy=POLICY,chain_coefficient_sha256=coefficient_sha256(chain_model),
        chain_audit_sha256=chain_audit['audit_sha256'],original_coefficient_sha256=coefficient_sha256(original_model),
        original_audit_sha256=original_audit['audit_sha256'],support_ids=list(ids),ambiguity_sha256=P.ambiguity_sha256,
        retained_cut_identities=identities,source_references=source_references,full_support_new_anchor=False,
        new_UB_verified=False,formal_OOS=False,scientific_acceptance=False,at_most_one_new_scenario=True)
    if full_incumbent is None:
        return old._seal(dict(common,status='SKIP_NO_AUXILIARY_INCUMBENT',selected_index=None,selected_scenario_id=None,
            new_Q_required=False,master_call_required=False),'selection_sha256')
    v=np.asarray(full_incumbent,float)
    check=old._check_vector(chain_model['c'],chain_model['integrality'],chain_model['lower_bounds'],chain_model['upper_bounds'],A,
        chain_model['row_lower'],chain_model['row_upper'],v)
    old._need(check['passed'],'New saved full chain incumbent infeasible')
    # The old selector verifies every original cut against the original rows,
    # original audit and fixed P. Its ranking is deliberately not reused.
    original_selection=old.select_scenario(original_model,original_audit,P,cut_sets,old_probability,v,
        source_references=source_references,deadline=deadline)
    p=old._probabilities(P,old_probability);z=v[:n];zh=old._hash(z);scores=[];known={};S=len(ids)
    for i,sid in enumerate(ids):
        allcuts=[cut_sets[k][i] for k in range(2)]
        allcuts += [(e['cut'],e['gradient']) for e in retained_cuts if e['index']==i]
        template=allcuts[0][0]['template_sha256'];constant=allcuts[0][0]['scenario_constant_usd']
        for c,g in allcuts:
            old._cut(c,g,n,sid,template=template,constant=constant)
            key=(c['anchor_first_stage_sha256'],sid)
            if key in known:
                old._need(known[key]==c['cut_sha256'],'Conflicting repeated anchor/scenario evidence')
            known[key]=c['cut_sha256']
        values=[float(c['intercept_usd']+np.asarray(g)@z) for c,g in allcuts]
        score=float(p[i]*(max(values)-min(values)))
        old._need(all(math.isfinite(x) for x in values+[score]),'Nonfinite chain score')
        scores.append(dict(index=i,scenario_id=sid,probability=float(p[i]),cut_values_usd=values,score_usd=score,
            cut_sha256s=[c['cut_sha256'] for c,g in allcuts],template_sha256=template,scenario_constant_usd=constant))
    ranking=sorted(range(S),key=lambda i:(-scores[i]['score_usd'],-scores[i]['probability'],i));i=ranking[0]
    reuse=known.get((zh,ids[i]));old._alive(deadline)
    return old._seal(dict(common,status='REUSE_EXACT_ANCHOR_SCENARIO' if reuse else 'NEW_Q_REQUIRED',
        selected_index=i,selected_scenario_id=ids[i],selected_template_sha256=scores[i]['template_sha256'],
        selected_scenario_constant_usd=scores[i]['scenario_constant_usd'],scores=scores,ranking=ranking,
        first_stage_values=z.tolist(),first_stage_sha256=zh,full_incumbent_sha256=old._hash(v),first_stage_columns=n,
        probabilities=p.tolist(),old_probability=old_probability,original_cut_sha256s=original_selection['original_cut_sha256s'],
        original_full_incumbent_check=check,reuse_cut_sha256=reuse,new_Q_required=reuse is None,
        master_call_required=False if reuse else None,
        reuse_reason='Exact anchor/scenario Q already accepted and its cut is retained; no duplicate Q or unchanged master.' if reuse else None),
        'selection_sha256')


def _selection(selection,model,audit,P,v,original_model,original_audit,retained_cuts,cut_sets,deadline):
    old._unseal(selection,'selection_sha256','Chain preselection')
    expected=select_scenario(model,audit,P,cut_sets,selection['old_probability'],v,original_model=original_model,
        original_audit=original_audit,retained_cuts=retained_cuts,source_references=selection['source_references'],deadline=deadline)
    old._need(selection==expected,'Chain preselection differs from retained arrays, evidence or exact dedup')
    old._need(selection['status']=='NEW_Q_REQUIRED' and selection['new_Q_required'] is True,'Exact existing Q must be reused; no new row allowed')
    return selection['first_stage_columns'],selection['selected_index'],np.asarray(selection['first_stage_values'])


def audit_partial_model(model,chain_model,chain_audit,P,selection,cut,gradient,full_incumbent,*,
                        original_model,original_audit,retained_cuts,cut_sets,source_references,deadline):
    old._alive(deadline);old._refs(source_references)
    n,i,z=_selection(selection,chain_model,chain_audit,P,full_incumbent,original_model,original_audit,retained_cuts,cut_sets,deadline)
    g=old._cut(cut,gradient,n,P.support_ids[i],anchor=z,template=selection['selected_template_sha256'],constant=selection['selected_scenario_constant_usd'])
    A,m=_prefix(model,chain_model,1)
    old._need(np.array_equal(A[m,:n].toarray().ravel(),-g) and
        np.array_equal(A[m,n:].toarray().ravel(),np.r_[1.,-P.features[i],P.features[i]]) and
        model['row_lower'][m]==cut['intercept_usd'] and np.isposinf(model['row_upper'][m]),'New partial row/P/constant differs')
    v=np.asarray(full_incumbent,float);activity=float((A[m]@v).item());violation=max(0.,float(cut['intercept_usd'])-activity)
    normalized=violation/max(1.,float((abs(A[m])@abs(v)).item()));excluded=violation>old.TOL or normalized>old.TOL
    entry=dict(index=i,cut=cut,gradient=g,anchor=z,source_references=source_references)
    identities=[_identity(e) for e in retained_cuts]+[_identity(entry)]
    proof=dict(kind=AUDIT_KIND,passed=True,coefficient_sha256=coefficient_sha256(model),
        base_coefficient_sha256=coefficient_sha256(chain_model),base_audit_sha256=chain_audit['audit_sha256'],
        original_coefficient_sha256=coefficient_sha256(original_model),original_audit_sha256=original_audit['audit_sha256'],
        selection_sha256=selection['selection_sha256'],support_ids=list(P.support_ids),ambiguity_sha256=P.ambiguity_sha256,
        selected_index=i,selected_scenario_id=P.support_ids[i],cut_sha256=cut['cut_sha256'],source_references=source_references,
        retained_cut_identities=identities,total_partial_rows=len(identities),retained_prefix_rows=m,appended_rows=1,
        old_full_incumbent_sha256=old._hash(v),new_row_activity_usd=activity,new_row_lower_usd=float(cut['intercept_usd']),
        incumbent_row_violation=violation,incumbent_row_normalized_violation=normalized,incumbent_excluded=excluded,
        master_call_required=excluded,skip_requires_previously_accepted_auxiliary_optimality=True,
        skip_reason=None if excluded else 'Full previously optimal incumbent remains feasible; do not rerun the same master.',
        full_support_new_anchor=False,newly_evaluated_scenario_indices=[i],newly_evaluated_scenario_count=1,
        retained_objective_domain_and_P=True,new_UB_verified=False,full_candidate_evaluated=False,formal_OOS=False,
        scientific_acceptance=False,solver_calls=0)
    old._alive(deadline);return old._seal(proof)


def append_partial_cut(chain_model,chain_audit,P,selection,cut,gradient,full_incumbent,*,
                       original_model,original_audit,retained_cuts,cut_sets,source_references,deadline):
    n,i,z=_selection(selection,chain_model,chain_audit,P,full_incumbent,original_model,original_audit,retained_cuts,cut_sets,deadline)
    g=old._cut(cut,gradient,n,P.support_ids[i],anchor=z,template=selection['selected_template_sha256'],constant=selection['selected_scenario_constant_usd'])
    row=csr_matrix(np.r_[-g,1.,-P.features[i],P.features[i]].reshape(1,-1))
    model=dict(matrix=vstack([chain_model['matrix'],row],format='csr'),objective_constant_usd=chain_model['objective_constant_usd'],
        row_lower=np.r_[chain_model['row_lower'],cut['intercept_usd']],row_upper=np.r_[chain_model['row_upper'],np.inf],
        **{k:np.asarray(chain_model[k]).copy() for k in ('c','integrality','lower_bounds','upper_bounds')})
    proof=audit_partial_model(model,chain_model,chain_audit,P,selection,cut,g,full_incumbent,original_model=original_model,
        original_audit=original_audit,retained_cuts=retained_cuts,cut_sets=cut_sets,source_references=source_references,deadline=deadline)
    return model,proof,dict(master_call_required=proof['master_call_required'],incumbent_excluded=proof['incumbent_excluded'],
        raw_violation=proof['incumbent_row_violation'],normalized_violation=proof['incumbent_row_normalized_violation'],skip_reason=proof['skip_reason'])
