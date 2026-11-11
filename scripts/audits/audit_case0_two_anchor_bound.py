"""Strengthen an unchanged Case0 plan's bound using two saved original Q anchors.

No new recourse solve, original extensive solve, candidate acceptance or OOS.
All saved-array extraction, cut reconstruction, assembly, native solve and audit
consume the original cumulative master allowance. Source-pinned prior kernels
retain their original numeric checks and raw native records unchanged.
"""
import argparse,gc,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz,save_npz
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model
from evcs_experiments.native_gurobi import solve_milp,_check_vector
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.scenario_features import FEATURE_NAMES
from evcs_experiments.relaxation_bound import coefficient_sha256
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.case0_planning import _json
from evcs_experiments.runtime.jobctl import atomic_json

COMPACT_SHA='d9c779487cce4cb844cfb8bce2cacbeae9fb87c5171f8c78d5dd88e35cdf33da'
HASH_SHA='7333d02ff1668ad04d75bc73f8b139570401093679a10a6160fd411a0c2c3582'
DUAL_SHA='810614b996832454da34f5a3bab425ca9ae058a0f4eb5047b3e53b751303b889'

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())

def left(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('Two-anchor original master allowance exhausted')
    return value

def load(ref,name,expected):
    verify_reference(ref);need(ref['sha256']==expected,'Exact historical helper source differs')
    s=importlib.util.spec_from_file_location(name,ref['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare_two_anchor(A,arrays,header,old,new,P,old_probability,new_probability,weighted,weighted_directory,*,out,deadline,binding,compact,dual):
    """Keep first anchor's rows/floors, append the second complete original axis."""
    need(new['kind']=='CASE0_PROPOSED_CANDIDATE_UB_AUDIT_V1' and new['passed'] is True and
         old['plan_sha256']!=new['plan_sha256'] and new['model_sha256']==old['model_sha256']==header['model_sha256'],
         'Distinct audited anchors on the same original model required')
    for anchor in (old,new):
        need(anchor['first_stage_physical_passed'] is True and anchor['full_support_fixed_Q_and_KKT_passed'] is True and
             anchor['probability_primal_dual_passed'] is True and anchor['planning_acceptance'] is False and
             canonical_hash(anchor['plan'])==anchor['plan_sha256'],'Audited anchor scope/identity differs')
    oldout=Path(out)/'anchor_0';oldout.mkdir()
    base,basecheck,first,start=compact.prepare_epigraph(A,arrays,header,old,P,old_probability,weighted,weighted_directory,
        output=oldout,deadline=deadline,binding=binding,dual_module=dual)
    newout=Path(out)/'anchor_1';newout.mkdir()
    dual.derive_saved_cuts(A,arrays,header,new,P,new_probability,output=newout,deadline=deadline,binding=binding)
    ids=tuple(header['scenario_ids']);groups=[];floors={}
    for index,(anchor,folder,source) in enumerate(((old,oldout/'revalidated_cuts',binding['ub_result']),
                                                  (new,newout,binding['new_ub_result']))):
        cuts={}
        for i,sid in enumerate(ids):
            left(deadline);saved=read(file_reference(folder/f'cut_{i:03d}.json'))
            g=np.load(folder/f'gradient_{i:03d}.npy',allow_pickle=False)
            need(saved['index']==i and saved['scenario_id']==sid and saved['template_sha256']==header['recourse_template_sha256s'][i],
                 'Original two-anchor cut support differs')
            cuts[i]=(saved['cut'],g)
            if index==0:floors[i]=read(file_reference(oldout/f'floor_{i:03d}.json'))['floor']
        groups.append(dict(anchor_first_stage=np.r_[anchor['plan']['x'],np.asarray(anchor['plan']['r']).ravel()],
                           cuts=cuts,source_references=[source]))
    model,construction=build_multi_anchor_epigraph_master(first,ids,groups,probability_set=P,floors=floors)
    count=base['matrix'].shape[0]
    need((model['matrix'][:count]!=base['matrix']).nnz==0 and all(np.array_equal(model[k][:count],base[k]) for k in ('row_lower','row_upper')) and
         all(np.array_equal(model[k],base[k]) for k in ('c','integrality','lower_bounds','upper_bounds')),
         'Original first-anchor rows/floors/domain changed')
    check=audit_multi_anchor_model(model,first,ids,groups,probability_set=P,floors=floors,deadline=deadline)
    need(not np.any((abs(model['matrix'].data)>0)&(abs(model['matrix'].data)<1e-13)),
         'Native tiny-coefficient rejection; no unreviewed dropping/repair')
    # Original best plan remains a nonbinding start. Raise only free theta0 to
    # accommodate extra valid cuts, never fix or round any first-stage variable.
    nf=len(first['c']);m=first['matrix'].shape[0]
    start[nf]+=max(0.,float(np.max(model['row_lower'][m:]-(model['matrix']@start)[m:],initial=0.)))+1e-6
    startcheck=_check_vector(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],start)
    need(startcheck['passed'],'Unchanged best-plan nonbinding start violates two-anchor model')
    atomic_json(Path(out)/'construction.json',_json(construction));atomic_json(Path(out)/'model_audit.json',_json(check))
    atomic_json(Path(out)/'start_check.json',_json(startcheck))
    return model,check,start


def evaluate_bound(model,check,old,new,prior,solved,*,out,deadline,binding,compact):
    """Native bound only; auxiliary primal never replaces the original UB."""
    need(prior['kind']=='CASE0_COMPACT_DRO_CANDIDATE_RESULT_V1' and prior['model_sha256']==old['model_sha256'] and
         prior['prior_plan_sha256']==old['plan_sha256'] and prior['plan_sha256']==new['plan_sha256'] and
         prior['planning_acceptance'] is False and all(type(prior[k]) in (int,float) and math.isfinite(prior[k]) for k in ('lower_bound_usd','retained_original_lower_bound_usd')) and
         prior['retained_original_lower_bound_usd']>=prior['lower_bound_usd'],
         'Previous independently accepted compact bound/anchors differ')
    result=dict(kind='CASE0_TWO_ANCHOR_BOUND_RESULT_V1',status='NO_NEW_BOUND',source_binding=binding,method_id='EA_DRO',
        model_sha256=old['model_sha256'],auxiliary_model_sha256=coefficient_sha256(model),
        plan=old['plan'],plan_sha256=old['plan_sha256'],anchor_plan_sha256s=[old['plan_sha256'],new['plan_sha256']],
        upper_bound_usd=old['upper_bound_usd'],upper_bound_source='previous independently audited best original plan',
        rejected_candidate_upper_bound_usd=new['upper_bound_usd'],prior_lower_bound_usd=prior['retained_original_lower_bound_usd'],
        new_native_lower_bound_usd=None,lower_bound_usd=prior['retained_original_lower_bound_usd'],
        native_status=int(solved.status),auxiliary_objective_usd=None,auxiliary_mip_gap=solved.mip_gap,
        planning_acceptance=False,full_numerical_acceptance=False,gap_passed=False,formal_OOS=False,
        original_extensive_vector_replayed=False,new_Q_solve_count=0,new_plan_accepted=False,
        actual_solve_type='two_anchor_full_ambiguity_integer_epigraph_auxiliary_bound',raw_native_record_unchanged=True)
    if solved.x is not None:
        vector=np.asarray(solved.x,float);np.save(Path(out)/'auxiliary_vector.npy',vector,allow_pickle=False)
        result['auxiliary_objective_usd']=float(model['c']@vector+model['objective_constant_usd'])
        result['auxiliary_vector_reference']=file_reference(Path(out)/'auxiliary_vector.npy')
    try:
        bound=compact.audit_case0_native_bound(model,check,solved.native_record,native_status=solved.status,
            native_bound=solved.mip_dual_bound,incumbent=solved.x,expected_coefficient_sha256=coefficient_sha256(model),
            fixed_upper_bound=old['upper_bound_usd'],deadline=deadline)
        atomic_json(Path(out)/'native_bound_audit.json',_json(bound))
        result.update(status='AUDITED_TWO_ANCHOR_AUXILIARY_BOUND',new_native_lower_bound_usd=bound['lower_bound_usd'],
            lower_bound_usd=max(prior['retained_original_lower_bound_usd'],bound['lower_bound_usd']),native_bound_audit_reference=file_reference(Path(out)/'native_bound_audit.json'))
    except ValueError as error:result['native_bound_rejection']=str(error)
    need(result['lower_bound_usd']<=old['upper_bound_usd']+1e-4+1e-7*max(1.,abs(old['upper_bound_usd'])),'Bound contradicts retained original UB')
    gap=max(0.,old['upper_bound_usd']-result['lower_bound_usd'])/max(1.,abs(old['upper_bound_usd']))
    result.update(gap=gap,bound_gap_within_threshold=gap<=.005,
        full_planning_acceptance_requires_original_vector_closure_and_independent_review=True)
    left(deadline);atomic_json(Path(out)/'result.json',_json(result));return result


def numeric_run(refs,out,deadline,solver_seconds,checker,dual,compact):
    header=read(refs['model']);verify_reference(refs['matrix']);verify_reference(refs['model_arrays'])
    A=load_npz(refs['matrix']['path']).tocsr()
    with np.load(refs['model_arrays']['path'],allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    checker.check_model_hash(A,arrays,header);left(deadline)
    old=read(refs['ub_result']);new=read(refs['new_ub_result']);prior=read(refs['compact_result']);weighted=read(refs['saved_dual_result'])
    identity=read(refs['input_identity']);cal=read(refs['calibration_manifest']);need(cal['context_sha256']==canonical_hash(identity),'Original calibration input differs')
    moments=read(cal['moments']);verify_reference(cal['arrays'])
    with np.load(cal['arrays']['path'],allow_pickle=False) as data:features=data['planning'].copy()
    b=moments['moments']['bounds_by_rho']['1.0'];P=ProbabilitySet(tuple(header['scenario_ids']),FEATURE_NAMES,features,b['lower'],b['upper'])
    model,check,start=prepare_two_anchor(A,arrays,header,old,new,P,read(refs['probability']),read(refs['new_probability']),weighted,
        Path(refs['saved_dual_result']['path']).parent,out=out,deadline=deadline,binding=refs,compact=compact,dual=dual)
    save_npz(Path(out)/'matrix.npz',model['matrix']);np.savez_compressed(Path(out)/'model_arrays.npz',**{k:model[k] for k in
        ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')})
    del A,arrays;gc.collect();grant=min(solver_seconds,left(deadline)-4.);need(grant>0,'No original master allowance after two-anchor preparation')
    atomic_json(Path(out)/'native_started.json',dict(grant_seconds=grant,coefficient_sha256=coefficient_sha256(model),
        full_ambiguity=True,anchor_count=2,all_preparation_charged_to_master=True,unchanged_best_plan_nonbinding_start=True))
    solved=solve_milp(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],
        options=dict(time_limit=grant,mip_rel_gap=.005,threads=1,solver_feasibility_tolerance=1e-9),log_path=Path(out)/'native.log',start=start)
    atomic_json(Path(out)/'native_record.json',_json(solved.native_record))
    return evaluate_bound(model,check,old,new,prior,solved,out=out,deadline=deadline,binding=refs,compact=compact)


def run_request(path,expected):
    began=time.monotonic();rr=file_reference(path);need(rr['sha256']==expected,'Two-anchor request changed')
    req=read(rr);refs=req.get('references',{})
    names={'model','matrix','model_arrays','input_identity','calibration_manifest','config','source_manifest','state','settlement',
           'ub_result','probability','model_hash_checker','review','saved_dual_result','saved_dual_source','compact_source','compact_result','new_ub_result','new_probability'}
    need(set(req)=={'kind','references','output_directory','budget_seconds','solver_seconds'} and
         req['kind']=='CASE0_TWO_ANCHOR_BOUND_REQUEST_V1' and set(refs)==names,'Two-anchor request schema differs')
    review=read(refs['review']);base=dict(req,references={k:v for k,v in refs.items() if k!='review'})
    need(review['kind']=='CASE0_TWO_ANCHOR_BOUND_REVIEW_V1' and review['passed'] is True and
         review['request_without_review']==base and review['auditor_sha256']==file_reference(__file__)['sha256'],
         'Exact two-anchor source/request review required')
    state=read(refs['state']);s=read(refs['settlement']);key='size0/EA_DRO'
    need(s['kind']=='CASE0_PROPOSED_CANDIDATE_AUDIT_SETTLEMENT_V1' and s['passed'] is True and s['scientific_planning_acceptance'] is False and
         s['new_candidate_UB_verified'] is True and s['new_candidate_improves_incumbent'] is False and s['no_OOS'] is True and
         s['result_reference']==refs['new_ub_result'] and s['best_ub_reference']==refs['ub_result'] and s['best_bound_reference']==refs['compact_result'] and
         state['proposed_candidate_audit_settlement_reference']==refs['settlement'] and state['proposed_candidate_ub_reference']==refs['new_ub_result'] and
         state['saved_candidate_ub_reference']==refs['ub_result'] and state['compact_candidate_result_reference']==refs['compact_result'] and
         state['active_config_reference']==refs['config'] and state['source_release_reference']==refs['source_manifest'] and
         state['compute_seconds']==s['closing_compute_seconds'] and state['configuration_spent_seconds']=={key:s['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={key:s['closing_master_seconds']} and not state['active_job_id'] and not state['active_configuration'] and
         not state['planning_parent_pending_settlement'] and not state['pending_proposed_candidate_audit'] and not state.get('pending_two_anchor_bound') and not state['evaluation_opened'] and state['completed']=={},
         'Original settled two-anchor state/best-plan/budget differs')
    terminal=read(s['terminal_review_reference'])
    need(terminal['kind']=='CASE0_PROPOSED_CANDIDATE_TERMINAL_REVIEW_V1' and terminal['passed'] is True and
         terminal['numerical_recheck_passed'] is True and terminal['references']['result']==refs['new_ub_result'],
         'New original8Q independent terminal review required')
    seconds=req['budget_seconds'];solver=req['solver_seconds'];remaining=1200-s['closing_master_seconds']
    need(s['remaining_master_seconds']==remaining and s['remaining_configuration_seconds']==1800-s['closing_configuration_seconds'] and
         all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (seconds,solver)) and solver<=15 and
         solver<seconds<=remaining and seconds<=s['remaining_configuration_seconds'] and seconds<=21600-state['compute_seconds'],
         'Original two-anchor cumulative budget exceeded')
    paths=get_paths();root=paths.campaign_root
    need(root.name=='ieee33-v54-q1-case0-expanded' and Path(refs['state']['path']).is_relative_to(root/'operations') and
         Path(refs['state']['path'])!=root/'state.json','Immutable expanded snapshot required')
    jobid=os.environ.get('EVCS_JOB_ID');jobdir=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
    need(jobid and jobdir==root/'jobs'/jobid,'Exact owned jobctl parent required')
    parent=json.loads((jobdir/'manifest.json').read_text());guard=review['guard_reference'];verify_reference(guard)
    need(Path(guard['path']).parent==Path(rr['path']).parent==Path(__file__).resolve().parent and
         parent['argv']==[str(paths.project_root/'.venv/bin/python'),guard['path']] and parent['job_id']==jobid and
         parent['outroot']==str(root/'jobs') and parent['threads']==1 and seconds<parent['timeout_seconds'] and
         parent['timeout_seconds']+parent['grace_seconds']<=min(remaining,s['remaining_configuration_seconds']),
         'Whole actual parent must fit original master and configuration allowance')
    from evcs_experiments.case0_isolation import verify_current_cgroup
    from evcs_experiments.case0_campaign import verify_sources
    from evcs_experiments.case0_profiles import validate_config,EXPANDED
    verify_current_cgroup();source=read(refs['source_manifest']);config=read(refs['config']);verify_sources(config)
    inventory={r['path']:r['sha256'] for r in source['files']}
    need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json' and validate_config(config,root)==EXPANDED and
         len(inventory)==len(source['files']) and all(inventory.get(k)==v for k,v in config['source_sha256'].items()) and
         inventory.get('src/evcs_experiments/multi_anchor_epigraph.py')=='6de35e00138c602d5772d35b8337868b403072e6b7ce28e4643a382ebfb6c489' and
         inventory.get('src/evcs_experiments/multi_anchor_bound_audit.py')=='9fd199f0eab88b7b65326875891ebcd415891241b536eb79543f17a5ff9404ea',
         'Unchanged original R5 numerical source/config required')
    inputs=root/'planning/size0/EA_DRO/inputs';solve=inputs.parent/'compact-metadata-1/solve'
    for k,n in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]:need(Path(refs[k]['path'])==solve/n,'Original saved array path differs')
    need(Path(refs['input_identity']['path'])==inputs/'identity.json' and Path(refs['calibration_manifest']['path'])==inputs/'calibration_manifest.json',
         'Original input/calibration path differs')
    old=read(refs['ub_result']);new=read(refs['new_ub_result']);prior=read(refs['compact_result']);weighted=read(refs['saved_dual_result'])
    need(old['kind']=='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' and len(old['scenario_ids'])==8 and
         all(old['source_binding'][k]==new['source_binding'][k]==refs[k] for k in ('model','input_identity','config','source_manifest')) and
         new['source_binding']['candidate']==refs['compact_result'] and new['source_binding']['old_ub']==refs['ub_result'] and
         new['upper_bound_usd']>old['upper_bound_usd']==s['best_upper_bound_usd'] and
         new['upper_bound_usd']==s['new_upper_bound_usd'] and prior['retained_original_lower_bound_usd']==s['best_lower_bound_usd'] and
         prior['source_binding']['ub_result']==refs['ub_result'] and prior['source_binding']['saved_dual_result']==refs['saved_dual_result'] and
         all(prior['source_binding'][k]==refs[k] for k in ('model','matrix','model_arrays','input_identity','calibration_manifest','config','source_manifest','probability','model_hash_checker','saved_dual_source')) and
         all(weighted['source_binding'][k]==refs[k] for k in ('model','matrix','model_arrays','input_identity','calibration_manifest','config','source_manifest','ub_result','probability','model_hash_checker')),
         'Both audited anchors/prior bound source chain differs')
    for result,p in [('ub_result','probability'),('new_ub_result','new_probability')]:need(Path(refs[p]['path'])==Path(refs[result]['path']).parent/'probability.json','Original per-anchor probability path differs')
    compact=load(refs['compact_source'],'case0_two_anchor_prior_compact',COMPACT_SHA)
    checker=load(refs['model_hash_checker'],'case0_two_anchor_original_hash',HASH_SHA)
    dual=load(refs['saved_dual_source'],'case0_two_anchor_original_dual',DUAL_SHA)
    out=Path(req['output_directory']).resolve();need(out.is_relative_to(root/'operations') and out!=root/'operations','New operation output required')
    out.mkdir(parents=True,exist_ok=False)
    try:return numeric_run({k:v for k,v in refs.items() if k!='review'},out,began+seconds,solver,checker,dual,compact)
    except BaseException as e:
        atomic_json(out/'failure.json',dict(type=type(e).__name__,error=str(e),request_reference=rr,planning_acceptance=False));raise
    finally:
        elapsed=time.monotonic()-began;atomic_json(out/'operation_accounting.json',dict(actual_seconds=elapsed,master_seconds=elapsed,
            request_reference=rr,scope='Whole actual parent once original master/config/global; all saved-array work, native and checks nested. No Q or new plan accepted.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();r=run_request(a.request,a.expected_sha256);print(json.dumps({k:r[k] for k in ('status','upper_bound_usd','lower_bound_usd','gap','planning_acceptance')}))
