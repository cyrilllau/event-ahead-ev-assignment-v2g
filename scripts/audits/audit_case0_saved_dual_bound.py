"""One tiny auxiliary master from eight saved original Case0 recourse duals.

No recourse/scenario rebuilding or new Q solve. Original extensive arrays supply
all first-stage, fixed recourse and ambiguity coefficients. All numerical work,
including extraction/assembly, consumes the original remaining master budget.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.sparse import load_npz,save_npz
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.dual_cut_bound import build_dual_cut,build_weighted_dual_problem
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.scenario_features import FEATURE_NAMES
from evcs_experiments.relaxation_bound import coefficient_sha256,solve_relaxation_bound,check_relaxation_certificate
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.case0_planning import _json
from evcs_experiments.runtime.jobctl import atomic_json

HASH_CHECKER_SHA='7333d02ff1668ad04d75bc73f8b139570401093679a10a6160fd411a0c2c3582'

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())

def left(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('Saved-dual master allowance exhausted')
    return value

def close(a,b,label):
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),label+' differs')

def maximum(x):return float(np.max(x,initial=0.))


def derive_saved_cuts(A,arrays,header,ub,P,probability,*,output,deadline,binding):
    """Extract original signed B/A blocks; independently recheck saved Q KKT."""
    out=Path(output);ids=tuple(header['scenario_ids']);sl=header['variable_slices'];H=sl['x'][1];nf=sl['first_stage'][1]
    need(sl['x']==[0,H] and sl['r']==[H,nf] and sl['first_stage']==[0,nf] and
         ids==P.support_ids and P.ambiguity_sha256==header['ambiguity_sha256'] and
         ub['model_sha256']==header['model_sha256'] and ub['scenario_ids']==list(ids), 'Original support/model/first-stage axis differs')
    z=np.r_[ub['plan']['x'],np.asarray(ub['plan']['r']).ravel()]
    need(z.shape==(nf,) and np.isfinite(z).all() and ub['plan']['task_sha256']==header['task_sha256'] and
         ub['plan']['config_hash']==header['config_hash'] and canonical_hash(ub['plan'])==ub['plan_sha256'],'Saved anchor identity differs')
    pcheck=check_probability_solution(P,np.asarray(ub['q_usd']),probability,q_support_ids=ids)
    need(pcheck.passed,'Original probability nonemptiness/witness check failed')
    close(ub['Fpre_usd']+probability['dual_objective_usd'],ub['upper_bound_usd'],'Original audited UB')
    blocks=header['block_map'];need(len(blocks)==1+3*len(ids),'Original extensive block inventory differs')
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
    alpha=slice(*sl['alpha']);beta=slice(*sl['beta']);theta=sl['theta0'][0]
    need(sl['theta0']==[theta,theta+1] and sl['alpha']==[theta+1,theta+1+len(P.feature_ids)] and
         sl['beta']==[sl['alpha'][1],A.shape[1]] and np.array_equal(arrays['c'][alpha],-P.lower) and
         np.array_equal(arrays['c'][beta],P.upper) and arrays['c'][theta]==1. and
         np.all(arrays['c'][nf:theta]==0) and np.all(arrays['integrality'][nf:]==0) and
         float(arrays['objective_constant_usd'])==0.,'Original ambiguity/objective domain differs')
    need(len(ub['completed_scenario_references'])==len(ids),'Incomplete saved Q axis')
    cuts={};checks=[]
    for i,sid in enumerate(ids):
        left(deadline);a,b=sl['recourses'][sid];th=header['recourse_template_sha256s'][i]
        need(nf<=a<b<=theta,'Original recourse column range differs')
        template=dict(scenario_id=sid,template_sha256=th,lower_bounds=arrays['lower_bounds'][a:b],upper_bounds=arrays['upper_bounds'][a:b])
        for suffix in ('ub','eq'):
            row=index[(sid,'recourse_'+suffix)];need(row['source_sha256']==th,'Original recourse block source differs')
            ra,rb=row['row_start'],row['row_stop'];block=A[ra:rb];own=block[:,a:b].tocsr();coupling=block[:,:nf].tocsr()
            need(own.nnz+coupling.nnz==block.nnz,'Original recourse has foreign columns')
            rl,ru=arrays['row_lower'][ra:rb],arrays['row_upper'][ra:rb]
            need(np.all(np.isneginf(rl)) if suffix=='ub' else np.array_equal(rl,ru),'Original recourse row sense differs')
            template.update({f'A_{suffix}':own,f'Bx_{suffix}':-coupling[:,:H],f'Br_{suffix}':-coupling[:,H:],f'b_{suffix}':ru})
        risk=index[(sid,'risk_epigraph')];rr=risk['row_start'];need(risk['row_stop']==rr+1 and risk['source_sha256']==th,'Original risk row differs')
        row=A[rr];tail=np.r_[-1.,P.features[i],-P.features[i]]
        need(row[:,:a].nnz==0 and row[:,b:theta].nnz==0 and np.array_equal(row[:,theta:].toarray().ravel(),tail) and
             np.isneginf(arrays['row_lower'][rr]) and arrays['row_upper'][rr]==0. and header['scenario_cost_constants_usd'][i]==0.,'Original risk coupling or constant differs')
        template['c']=row[:,a:b].toarray().ravel()
        entry=ub['completed_scenario_references'][i];need(entry['index']==i and entry['scenario_id']==sid and entry['template_sha256']==th,'Saved Q row pairing differs')
        files=entry['files'];saved=read(files['fixed_solver']);verify_reference(files['primal']);verify_reference(files['checks'])
        need(saved['status']=='OPTIMAL' and saved['optimal'] is True and saved['solver_status']==0 and
             saved['certificate']['template_sha256']==th,'Saved Q optimal dual source differs')
        with np.load(files['primal']['path'],allow_pickle=False) as data:
            need(data.files==['fixed'],'Saved Q primal array inventory differs');v=data['fixed'].copy()
        dual={k:np.asarray(saved['certificate']['dual_'+k+'_marginals'],float) for k in ('inequality','equality','lower','upper')}
        y,e,lo,hi=(dual[k] for k in ('inequality','equality','lower','upper'))
        bub=template['b_ub']+template['Bx_ub']@z[:H]+template['Br_ub']@z[H:]
        beq=template['b_eq']+template['Bx_eq']@z[:H]+template['Br_eq']@z[H:]
        need(v.shape==template['c'].shape and np.isfinite(v).all(),'Saved Q primal axes/nonfinite')
        primal=max(maximum(template['A_ub']@v-bub),maximum(abs(template['A_eq']@v-beq)),maximum(template['lower_bounds']-v),maximum(v-template['upper_bounds']))
        need(primal<=1e-7,'Saved original Q primal feasibility failed')
        q=float(template['c']@v);close(q,saved['objective_usd'],'Original saved Q objective');close(q,ub['q_usd'][i],'Original Q support objective');close(q,entry['q_usd'],'Saved Q entry')
        cut,g=build_dual_cut(template,dual,scenario_constant_usd=0.,anchor_first_stage=z,anchor_q_usd=q)
        need(cut['passed'],'Original saved Q dual/KKT/anchor minorant failed')
        # Independent base-RHS contraction, distinct from builder's cut record.
        fl,fu=np.isfinite(template['lower_bounds']),np.isfinite(template['upper_bounds'])
        intercept=float(template['b_ub']@y+template['b_eq']@e+template['lower_bounds'][fl]@lo[fl]+template['upper_bounds'][fu]@hi[fu])
        grad=-A[index[(sid,'recourse_ub')]['row_start']:index[(sid,'recourse_ub')]['row_stop'],:nf].T@y-A[index[(sid,'recourse_eq')]['row_start']:index[(sid,'recourse_eq')]['row_stop'],:nf].T@e
        need(np.array_equal(g,grad),'Independent original coupling gradient differs');close(intercept,cut['intercept_usd'],'Independent base RHS intercept')
        cuts[i]=(cut,g)
        check=dict(index=i,scenario_id=sid,template_sha256=th,primal_error=primal,q_usd=q,cut=cut,saved_q_references=files)
        np.save(out/f'gradient_{i:03d}.npy',g,allow_pickle=False);atomic_json(out/f'cut_{i:03d}.json',_json(check));checks.append(check)
        del template,block,coupling,own,v,dual,y,e,lo,hi,grad,row;left(deadline)
    weights=np.asarray(probability['probabilities'],float);positive=np.flatnonzero(weights>0)
    selected={int(i):cuts[int(i)] for i in positive}
    model,construction=build_weighted_dual_problem(first,ids,weights,selected,probability_set=P)
    # Independent coefficient reconstruction, not a relabelled epigraph audit.
    cost=first['c'].copy();constant=first['objective_constant_usd']
    for i in positive:
        cut,g=cuts[int(i)];cost=cost+weights[i]*g;constant+=weights[i]*cut['intercept_usd']
    need((model['matrix']!=first['matrix']).nnz==0 and np.array_equal(model['c'],cost) and
         model['objective_constant_usd']==constant and all(np.array_equal(model[k],first[k]) for k in
         ('row_lower','row_upper','lower_bounds','upper_bounds','integrality')),'Independent weighted first-stage construction differs')
    check=dict(kind='INDEPENDENT_SAVED_Q_WEIGHTED_LP_MODEL_AUDIT_V1',passed=True,model_sha256=header['model_sha256'],
        coefficient_sha256=coefficient_sha256(model),ambiguity_sha256=P.ambiguity_sha256,support_ids=list(ids),
        probabilities=weights.tolist(),positive_indices=positive.tolist(),zero_indices=np.flatnonzero(weights==0).tolist(),
        probabilities_not_clipped_or_normalized=True,all_eight_saved_Q_duals_rechecked=len(cuts),
        original_first_stage_A_bounds_preserved=True,actual_solve_type='continuous_first_stage_LP_relaxation',
        auxiliary_relation='min over first-stage LP relaxation of Fpre + sum(saved feasible p_s * affine Q minorant_s) <= original full-support DRO MILP optimum',
        source_binding=binding,exact_arithmetic_proof=False,scientific_acceptance=False)
    atomic_json(out/'construction.json',_json(construction));atomic_json(out/'model_audit.json',_json(check))
    return model,check


def numeric_run(refs,out,deadline,solver_seconds,hash_checker):
    left(deadline);header=read(refs['model']);verify_reference(refs['matrix']);verify_reference(refs['model_arrays'])
    A=load_npz(refs['matrix']['path']).tocsr()
    with np.load(refs['model_arrays']['path'],allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    hash_checker.check_model_hash(A,arrays,header)
    ub=read(refs['ub_result']);identity=read(refs['input_identity']);calibration=read(refs['calibration_manifest'])
    need(calibration['context_sha256']==canonical_hash(identity),'Original calibration identity differs')
    moments=read(calibration['moments']);verify_reference(calibration['arrays'])
    with np.load(calibration['arrays']['path'],allow_pickle=False) as data:features=data['planning'].copy()
    bounds=moments['moments']['bounds_by_rho']['1.0'];P=ProbabilitySet(tuple(header['scenario_ids']),FEATURE_NAMES,features,bounds['lower'],bounds['upper'])
    probability=read(refs['probability'])
    model,model_audit=derive_saved_cuts(A,arrays,header,ub,P,probability,output=out,deadline=deadline,binding=refs)
    expected=coefficient_sha256(model);save_npz(out/'aux_matrix.npz',model['matrix'])
    np.savez_compressed(out/'aux_arrays.npz',**{k:model[k] for k in ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')})
    del A,arrays;gc.collect();grant=min(solver_seconds,left(deadline)-5.)
    need(grant>0,'No original master time remains after saved-cut preparation')
    atomic_json(out/'solver_started.json',dict(coefficient_sha256=expected,grant_seconds=grant,
        remaining_entire_master_seconds=left(deadline),preparation_already_charged_to_master=True,
        solver='existing scipy.optimize.linprog HiGHS first-stage continuous relaxation',master_milp_calls=0))
    solved,values=solve_relaxation_bound(model,time_limit_seconds=grant,threads=1,log_path=out/'solver_record.json')
    np.savez_compressed(out/'lp_witness.npz',**values)
    result=dict(kind='CASE0_SAVED_DUAL_BOUND_RESULT_V1',status='NO_VALID_BOUND',model_sha256=header['model_sha256'],
        plan_sha256=ub['plan_sha256'],source_binding=refs,upper_bound_usd=ub['upper_bound_usd'],lower_bound_usd=None,gap=None,
        gap_passed=False,planning_acceptance=False,numerical_auxiliary_only=True,formal_OOS=False,
        solver_status=solved['solver_status'],actual_solve_type='continuous_first_stage_LP_relaxation',
        saved_probability_full_axis_preserved=True,master_milp_calls=0,original_recourse_calls=0,original_scenarios_rebuilt=0)
    try:
        need(solved.get('certificate') is not None and solved.get('raw_solver') is not None and
             solved.get('wall_exhausted_after_solver') is not True,'No complete in-budget LP witness')
        # Reopen the saved numeric witness, then audit independently of the
        # solver wrapper's acceptance flags and reported lower-bound field.
        with np.load(out/'lp_witness.npz',allow_pickle=False) as data:reopened={k:data[k].copy() for k in data.files}
        check=check_relaxation_certificate(model,reopened,solver_status=solved['solver_status'],
            raw_objective_without_constant_usd=solved['raw_solver']['objective_without_constant_usd'])
        need(check['passed'] and coefficient_sha256(model)==expected,'Independent original first-stage LP certificate failed')
        lb=check['certified_lower_bound_usd'];need(lb<=ub['upper_bound_usd']+1e-4+1e-7*abs(ub['upper_bound_usd']),'Auxiliary LB exceeds audited UB')
        atomic_json(out/'lp_bound_audit.json',_json(check));gap=max(0.,ub['upper_bound_usd']-lb)/max(1.,abs(ub['upper_bound_usd']))
        result.update(status='AUDITED_NUMERICAL_AUXILIARY_BOUND',lower_bound_usd=lb,gap=gap,diagnostic_gap_within_threshold=gap<=.005,
            lp_bound_audit_reference=file_reference(out/'lp_bound_audit.json'))
    except ValueError as error:result['bound_rejection']=str(error)
    left(deadline);atomic_json(out/'result.json',_json(result));return result


def run_request(request_path,expected_sha256):
    began=time.monotonic();rr=file_reference(request_path);need(rr['sha256']==expected_sha256,'Saved-dual request changed')
    request=read(rr);refs=request.get('references',{})
    need(set(request)=={'kind','references','output_directory','budget_seconds','solver_seconds'} and
         request['kind']=='CASE0_SAVED_DUAL_BOUND_REQUEST_V1' and set(refs)=={'model','matrix','model_arrays','input_identity','calibration_manifest',
         'config','source_manifest','state','settlement','ub_result','probability','model_hash_checker','review'},'Saved-dual request schema differs')
    review=read(refs['review']);without=dict(request,references={k:v for k,v in refs.items() if k!='review'})
    need(review['kind']=='CASE0_SAVED_DUAL_BOUND_REVIEW_V1' and review['passed'] is True and review['request_without_review']==without and
         review['auditor_sha256']==file_reference(__file__)['sha256'],'Exact saved-dual source/request review required')
    state=read(refs['state']);settlement=read(refs['settlement']);key='size0/EA_DRO'
    need(settlement['kind']=='CASE0_SAVED_LP_BOUND_SETTLEMENT_V1' and settlement['scientific_planning_acceptance'] is False and
         settlement['ub_result_reference']==refs['ub_result'] and state['saved_lp_bound_settlement_reference']==refs['settlement'] and
         state['saved_candidate_ub_reference']==refs['ub_result'] and state['active_config_reference']==refs['config'] and
         state['source_release_reference']==refs['source_manifest'] and state['compute_seconds']==settlement['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={key:settlement['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={key:settlement['closing_master_seconds']} and state['active_job_id'] is None and
         state['active_configuration'] is None and state['pending_saved_lp_bound'] is None and state['evaluation_opened'] is False and state['completed']=={},
         'Original cumulative saved-LP settlement/snapshot differs')
    read(settlement['terminal_review_reference'])
    seconds=request['budget_seconds'];solver_seconds=request['solver_seconds'];remaining=1200-settlement['closing_master_seconds']
    close(remaining,settlement['remaining_master_seconds'],'Original master remaining')
    close(1800-settlement['closing_configuration_seconds'],settlement['remaining_configuration_seconds'],'Original configuration remaining')
    need(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (seconds,solver_seconds)) and solver_seconds<=20 and
         solver_seconds<seconds<=remaining and seconds<=1800-settlement['closing_configuration_seconds'] and
         seconds<=21600-state['compute_seconds'],'Saved-dual original allowance exceeded')
    deadline=began+seconds;paths=get_paths();root=paths.campaign_root
    need(root.name=='ieee33-v54-q1-case0-expanded' and Path(refs['state']['path']).is_relative_to(root/'operations') and
         Path(refs['state']['path'])!=root/'state.json','Expanded immutable launch snapshot required')
    jobid=os.environ.get('EVCS_JOB_ID');jobdir=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
    need(jobid and jobdir==root/'jobs'/jobid and (jobdir/'manifest.json').is_file(),'Exact saved-dual jobctl parent required')
    parent=json.loads((jobdir/'manifest.json').read_text());guard=review['guard_reference'];verify_reference(guard)
    need(Path(guard['path']).parent==Path(rr['path']).parent==Path(__file__).resolve().parent and
         parent['argv']==[str(paths.project_root/'.venv/bin/python'),guard['path']] and parent['job_id']==jobid and
         parent['outroot']==str(root/'jobs') and parent['threads']==1 and seconds<parent['timeout_seconds'] and
         parent['timeout_seconds']+parent['grace_seconds']<=1800-settlement['closing_configuration_seconds'],'Actual reviewed parent/guard differs')
    from evcs_experiments.case0_isolation import verify_current_cgroup
    from evcs_experiments.case0_campaign import verify_sources
    from evcs_experiments.case0_profiles import validate_config,EXPANDED
    verify_current_cgroup();source=read(refs['source_manifest']);config=read(refs['config']);verify_sources(config)
    need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json' and validate_config(config,root)==EXPANDED,'Unchanged producer/config required')
    inventory={r['path']:r['sha256'] for r in source['files']}
    need(len(inventory)==len(source['files']) and all(inventory.get(k)==v for k,v in config['source_sha256'].items()),'Producer source inventory differs')
    inputs=root/'planning/size0/EA_DRO/inputs';solve=inputs.parent/'compact-metadata-1/solve'
    for key,name in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]:need(Path(refs[key]['path'])==solve/name,'Original saved array path differs')
    need(Path(refs['input_identity']['path'])==inputs/'identity.json' and Path(refs['calibration_manifest']['path'])==inputs/'calibration_manifest.json','Original input/calibration path differs')
    ub=read(refs['ub_result']);need(ub['kind']=='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' and ub['passed'] is True and
        ub['first_stage_physical_passed'] is True and ub['full_support_fixed_Q_and_KKT_passed'] is True and ub['probability_primal_dual_passed'] is True and
        len(ub['scenario_ids'])==8 and all(ub['source_binding'][k]==refs[k] for k in ('model','input_identity','config','source_manifest')) and
        Path(refs['probability']['path'])==Path(refs['ub_result']['path']).parent/'probability.json','Original audited UB source differs')
    verify_reference(refs['model_hash_checker']);need(refs['model_hash_checker']['sha256']==HASH_CHECKER_SHA,'Original semantic checker source differs')
    spec=importlib.util.spec_from_file_location('case0_saved_dual_original_hash_checker',refs['model_hash_checker']['path'])
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    out=Path(request['output_directory']).resolve();need(out.is_relative_to(root/'operations') and out!=root/'operations','New saved-dual output required')
    out.mkdir(parents=True,exist_ok=False)
    try:return numeric_run({k:v for k,v in refs.items() if k!='review'},out,deadline,solver_seconds,checker)
    except BaseException as error:
        atomic_json(out/'failure.json',dict(type=type(error).__name__,error=str(error),request_reference=rr,planning_acceptance=False));raise
    finally:
        elapsed=time.monotonic()-began;atomic_json(out/'operation_accounting.json',dict(actual_seconds=elapsed,master_seconds=elapsed,
            request_reference=rr,all_preparation_extraction_assembly_solver_audit_charged_to_master=True,
            scope='Whole isolated parent actual elapsed is authoritative conservative master/configuration/global charge; inner time is nested, never added twice.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();r=run_request(a.request,a.expected_sha256);print(json.dumps({k:r[k] for k in ('status','upper_bound_usd','lower_bound_usd','gap','planning_acceptance')}))

if __name__=='__main__':main()
