"""Strengthen an audited cut bundle with one original exact recourse block."""
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from .case0_anchor_refinement import extract_template,need
from .relaxation_bound import coefficient_sha256

def augment_exact_scenario(base,A,arrays,h,index):
    ids=h['scenario_ids'];need(type(index) is int and 0<=index<len(ids),'Invalid selected scenario')
    risk=h['risk_model'];need(risk in ('SAA','ARO'),'Unsupported risk')
    nf=h['variable_slices']['first_stage'][1];nt=len(ids) if risk=='SAA' else 1;prefix=h['block_map'][0]['row_stop'];nb=nf+nt
    need(base['matrix'].shape[1]==nb and np.array_equal(base['c'],np.r_[arrays['c'][:nf],np.full(nt,1./len(ids) if risk=='SAA' else 1.)]),'Base risk objective differs')
    need((base['matrix'][:prefix,:nf]!=A[:prefix,:nf]).nnz==0 and base['matrix'][:prefix,nf:].nnz==0 and all(np.array_equal(base[k][:nf],arrays[k][:nf]) for k in ('lower_bounds','upper_bounds','integrality')) and all(np.array_equal(base[k][:prefix],arrays[k][:prefix]) for k in ('row_lower','row_upper')),'Original first-stage domain changed')
    need(float(base['objective_constant_usd'])==0. and h['scenario_cost_constants_usd'][index]==0.,'Only original zero constants')
    t=extract_template(A,arrays,h,index);nc=len(t['c']);rows=[hstack([base['matrix'],csr_matrix((base['matrix'].shape[0],nc))],format='csr')];lower=[base['row_lower']];upper=[base['row_upper']]
    for suffix in ('ub','eq'):
        coupling=-hstack([t['Bx_'+suffix],t['Br_'+suffix]],format='csr');own=t['A_'+suffix];rhs=t['b_'+suffix]
        rows.append(hstack([coupling,csr_matrix((len(rhs),nt)),own],format='csr'));lower.append(np.full(len(rhs),-np.inf) if suffix=='ub' else rhs);upper.append(rhs)
    cost=np.zeros(nb);cost[nf+(index if risk=='SAA' else 0)]=-1.
    rows.append(csr_matrix(np.r_[cost,t['c']].reshape(1,-1)));lower.append([-np.inf]);upper.append([0.])
    model=dict(matrix=vstack(rows,format='csr'),c=np.r_[base['c'],np.zeros(nc)],integrality=np.r_[base['integrality'],np.zeros(nc,dtype=int)],lower_bounds=np.r_[base['lower_bounds'],t['lower_bounds']],upper_bounds=np.r_[base['upper_bounds'],t['upper_bounds']],row_lower=np.concatenate(lower),row_upper=np.concatenate(upper),objective_constant_usd=0.)
    # Construction retains the complete existing cut model and original exact
    # selected recourse feasible region. All other recourses remain minorants.
    return model,dict(kind='CASE0_PARTIAL_EXACT_RECOURSE_MODEL_V1',risk_model=risk,selected_index=index,scenario_id=ids[index],template_sha256=t['template_sha256'],base_coefficient_sha256=coefficient_sha256(base),coefficient_sha256=coefficient_sha256(model),new_Q_calls=0,scientific_acceptance=False,relation='Exact selected original recourse plus retained valid other-scene minorants <= original optimum')
