"""Recover Case0 SAA uniform dual minorants from original saved extensive arrays.

No scenario construction, Q calls or ambiguity optimization. Source-bound saved
primal/dual pairs are checked against original signed rows, bounds and costs.
"""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from evcs_experiments.campaign_batch import verify_reference
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.saa_dual_bound import build_saa_dual_bound
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.native_gurobi import _check_vector
from evcs_experiments.relaxation_bound import coefficient_sha256
ARRAY_KEYS={'c','row_lower','row_upper','lower_bounds','upper_bounds','integrality','objective_constant_usd'}
def need(v,m):
    if not bool(v):raise ValueError(m)
def read(r):
    verify_reference(r);return json.loads(Path(r['path']).read_text())
def close(a,b,label):
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),label+' differs')
def maximum(x):return float(np.max(x,initial=0.))
def left(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Original master allowance exhausted')

def check_model_hash(matrix,arrays,header):
    need(set(arrays)==ARRAY_KEYS,'Saved model array inventory differs')
    constant=float(arrays['objective_constant_usd']);constants=np.asarray(header['scenario_cost_constants_usd'],float)
    need(matrix.shape==tuple(header['shape']) and matrix.nnz==header['nnz'] and
         header['risk_model']=='SAA' and header['first_stage_fixed'] is False and
         header['all_original_tasks_retained'] is True and constant==header['objective_constant_usd'] and
         np.array_equal(constants,np.zeros(len(header['scenario_ids']))),'Original model shape/risk/constant differs')
    metadata=dict(risk_model=header['risk_model'],scenario_ids=header['scenario_ids'],scenario_sha256s=header['scenario_sha256s'],
        template_sha256s=header['recourse_template_sha256s'],first_stage_sha256=header['first_stage_sha256'],
        ambiguity_sha256=header['ambiguity_sha256'],empirical_probabilities=header['empirical_probabilities'],
        slices=header['variable_slices'],objective_constant_usd=constant)
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode())
    for value in (matrix.data,matrix.indices,matrix.indptr,arrays['row_lower'],arrays['row_upper'],arrays['lower_bounds'],
                  arrays['upper_bounds'],arrays['c'],arrays['integrality'],constants):
        a=np.asarray(value,dtype='<i8' if np.asarray(value).dtype.kind in 'iu' else '<f8')
        digest.update(str(a.shape).encode());digest.update(a.tobytes())
    need(digest.hexdigest()==header['model_sha256'],'Original extensive semantic/array hash differs')
    return digest.hexdigest()


def derive_saved_saa_cuts(A,arrays,header,ub,entries,*,deadline):
    check_model_hash(A,arrays,header)
    ids=tuple(header['scenario_ids']);sl=header['variable_slices'];H=sl['x'][1];nf=sl['first_stage'][1]
    need(sl['x']==[0,H] and sl['r']==[H,nf] and sl['first_stage']==[0,nf] and
         header['risk_model']=='SAA' and ub['method_id']=='SAA' and
         ub['model_sha256']==header['model_sha256'] and ub['scenario_ids']==list(ids), 'Original support/model/first-stage axis differs')
    z=np.r_[ub['plan']['x'],np.asarray(ub['plan']['r']).ravel()]
    need(z.shape==(nf,) and np.isfinite(z).all() and ub['plan']['task_sha256']==header['task_sha256'] and
         ub['plan']['config_hash']==header['config_hash'] and canonical_hash(ub['plan'])==ub['plan_sha256'],'Saved anchor identity differs')
    need(ub['numerical_valid'] and ub['full_support_independently_evaluated'],'Full original Q result required')
    need(len(ids)>0 and len(set(ids))==len(ids) and header['empirical_probabilities']==[1./len(ids)]*len(ids),'Original full uniform law required')
    need(np.asarray(ub['q_usd']).shape==(len(ids),),'Q support axis differs')
    close(float(np.mean(ub['q_usd'])),ub['risk_value_usd'],'Uniform SAA risk')
    close(ub['Fpre_usd']+ub['risk_value_usd'],ub['upper_bound_usd'],'Original SAA UB')
    blocks=header['block_map'];need(len(blocks)==1+2*len(ids),'Original extensive block inventory differs')
    cursor=0;index={}
    for b in blocks:
        need(b['row_start']==cursor and b['row_stop']>=cursor,'Original contiguous block rows differ')
        cursor=b['row_stop'];key=(b['scope'],b['kind']);need(key not in index,'Duplicate original block');index[key]=b
    need(cursor==A.shape[0] and blocks[0]['scope']==blocks[0]['kind']=='first_stage' and
         blocks[0]['source_sha256']==header['first_stage_sha256'],'Original first-stage row provenance differs')
    m=blocks[0]['row_stop'];need(A[:m,nf:].nnz==0,'Original first-stage block has foreign columns')
    first=dict(matrix=A[:m,:nf].tocsr(),objective_constant_usd=0.,
        **{k:arrays[k][:nf] for k in ('c','integrality','lower_bounds','upper_bounds')},
        **{k:arrays[k][:m] for k in ('row_lower','row_upper')})
    close(float(first['c']@z),ub['Fpre_usd'],'Saved first-stage objective')
    fc=_check_vector(first['c'],first['integrality'],first['lower_bounds'],first['upper_bounds'],first['matrix'],first['row_lower'],first['row_upper'],z)
    need(fc['passed'],'Original anchor first-stage feasibility failed')
    need(len(entries)==len(ids) and np.all(arrays['integrality'][nf:]==0) and float(arrays['objective_constant_usd'])==0.,'Original recourse domain/constant differs')
    column_cursor=nf
    cuts={};checks=[]
    for i,sid in enumerate(ids):
        left(deadline);a,b=sl['recourses'][sid];th=header['recourse_template_sha256s'][i]
        need(a==column_cursor and a<b<=A.shape[1],'Original recourse column range differs');column_cursor=b
        template=dict(scenario_id=sid,template_sha256=th,lower_bounds=arrays['lower_bounds'][a:b],upper_bounds=arrays['upper_bounds'][a:b])
        for suffix in ('ub','eq'):
            row=index[(sid,'recourse_'+suffix)];need(row['source_sha256']==th,'Original recourse block source differs')
            ra,rb=row['row_start'],row['row_stop'];block=A[ra:rb];own=block[:,a:b].tocsr();coupling=block[:,:nf].tocsr()
            need(own.nnz+coupling.nnz==block.nnz,'Original recourse has foreign columns')
            rl,ru=arrays['row_lower'][ra:rb],arrays['row_upper'][ra:rb]
            need(np.all(np.isneginf(rl)) if suffix=='ub' else np.array_equal(rl,ru),'Original recourse row sense differs')
            template.update({f'A_{suffix}':own,f'Bx_{suffix}':-coupling[:,:H],f'Br_{suffix}':-coupling[:,H:],f'b_{suffix}':ru})
        template['c']=arrays['c'][a:b]*len(ids)
        entry=entries[i];need(entry['index']==i and entry['scenario_id']==sid and entry['template_sha256']==th,'Saved Q row pairing differs')
        files=entry['files'];saved=read(files['fixed_solver']);verify_reference(files['primal']);checks_record=read(files['checks']);need(checks_record['kkt']['passed'] and checks_record['physical']['passed'],'Original recorded Q/physical checks failed')
        need(saved['status']=='OPTIMAL' and saved['optimal'] is True and saved['solver_status']==0 and
             saved['certificate']['template_sha256']==th,'Saved Q optimal dual source differs')
        with np.load(files['primal']['path'],allow_pickle=False) as data:
            need(set(data.files)=={'fixed','witness'},'Saved Q primal array inventory differs');v=data['fixed'].copy()
        dual={k:np.asarray(saved['certificate']['dual_'+k+'_marginals'],float) for k in ('inequality','equality','lower','upper')}
        y,e,lo,hi=(dual[k] for k in ('inequality','equality','lower','upper'))
        bub=template['b_ub']+template['Bx_ub']@z[:H]+template['Br_ub']@z[H:]
        beq=template['b_eq']+template['Bx_eq']@z[:H]+template['Br_eq']@z[H:]
        need(v.shape==template['c'].shape and np.isfinite(v).all(),'Saved Q primal axes/nonfinite')
        primal=max(maximum(template['A_ub']@v-bub),maximum(abs(template['A_eq']@v-beq)),maximum(template['lower_bounds']-v),maximum(v-template['upper_bounds']))
        need(primal<=1e-7,'Saved original Q primal feasibility failed')
        q=float(template['c']@v);close(q,saved['objective_usd'],'Original saved Q objective');close(q,ub['q_usd'][i],'Original Q support objective')
        cut,g=build_dual_cut(template,dual,scenario_constant_usd=0.,anchor_first_stage=z,anchor_q_usd=q)
        need(cut['passed'],'Original saved Q dual/KKT/anchor minorant failed')
        # Independent base-RHS contraction, distinct from builder's cut record.
        fl,fu=np.isfinite(template['lower_bounds']),np.isfinite(template['upper_bounds'])
        intercept=float(template['b_ub']@y+template['b_eq']@e+template['lower_bounds'][fl]@lo[fl]+template['upper_bounds'][fu]@hi[fu])
        grad=-A[index[(sid,'recourse_ub')]['row_start']:index[(sid,'recourse_ub')]['row_stop'],:nf].T@y-A[index[(sid,'recourse_eq')]['row_start']:index[(sid,'recourse_eq')]['row_stop'],:nf].T@e
        need(np.array_equal(g,grad),'Independent original coupling gradient differs');close(intercept,cut['intercept_usd'],'Independent base RHS intercept')
        cuts[i]=(cut,g)
        check=dict(index=i,scenario_id=sid,template_sha256=th,primal_error=primal,q_usd=q,cut=cut,saved_q_references=files)
        checks.append(check)
        del template,block,coupling,own,v,dual,y,e,lo,hi,grad;left(deadline)
    need(column_cursor==A.shape[1],'Unexpected trailing non-SAA columns')
    model,construction=build_saa_dual_bound(first,ids,cuts,constants_usd=header['scenario_cost_constants_usd'])
    cost=first['c'].copy();constant=0.
    for i in range(len(ids)):
        cost+=cuts[i][1]/len(ids);constant+=cuts[i][0]['intercept_usd']/len(ids)
    need(np.array_equal(model['c'],cost) and model['objective_constant_usd']==constant and
         (model['matrix']!=first['matrix']).nnz==0 and all(np.array_equal(model[k],first[k]) for k in
         ('row_lower','row_upper','lower_bounds','upper_bounds','integrality')),'Independent uniform construction differs')
    return model,dict(kind='CASE0_SAA_SAVED_ARRAY_DUAL_RECHECK_V1',passed=True,model_sha256=header['model_sha256'],plan_sha256=ub['plan_sha256'],
        coefficient_sha256=coefficient_sha256(model),first_stage_check=fc,scenario_checks=checks,construction=construction,
        physical_checks_scope='Original producer physical receipts referenced; algebraic primal/KKT independently replayed here',
        all_saved_Q_rechecked=len(ids),new_Q_calls=0,scientific_acceptance=False),cuts
