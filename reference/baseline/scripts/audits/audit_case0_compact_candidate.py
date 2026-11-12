"""One original-DRO compact integer candidate from already audited Q cuts.

Recheck saved original cuts, restore full moment ambiguity, certify physical box
floors, and preserve the original first-stage integer domain. The candidate has
no new original-DRO upper bound until a separate eight-scenario audit succeeds.
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
from scipy.sparse import load_npz,save_npz,csr_matrix
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.epigraph_master import build_epigraph_master,certify_box_floor
from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model,_need,_same,_alive,_seal
from evcs_experiments.native_gurobi import solve_milp,_check_vector
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.scenario_features import FEATURE_NAMES
from evcs_experiments.relaxation_bound import coefficient_sha256
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.case0_planning import _json
from evcs_experiments.runtime.jobctl import atomic_json

HASH_CHECKER_SHA='7333d02ff1668ad04d75bc73f8b139570401093679a10a6160fd411a0c2c3582'
SAVED_DUAL_SHA='810614b996832454da34f5a3bab425ca9ae058a0f4eb5047b3e53b751303b889'

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())

def left(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('Compact DRO candidate allowance exhausted')
    return value

def close(a,b,label):
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),label+' differs')

def load_module(ref,name,expected):
    verify_reference(ref);need(ref['sha256']==expected,'Reviewed saved source differs: '+name)
    spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


# The following checker copies the immutable R5 checker verbatim except its
# name, explicit Case0 FeasibilityTol=1e-9 policy and provenance output fields.
# Raw native records are never altered or normalized for audit.
def audit_case0_native_bound(model,model_audit,native_record,*,native_status,native_bound,
                                 incumbent,expected_coefficient_sha256,fixed_upper_bound,deadline):
    """Check a source-bound Gurobi native bound, preserving original limit status.

    expected_coefficient_sha256 must be pinned before the native call. The native
    wrapper does not itself hash its full input, so caller/source admission must
    establish that exact coefficient model was passed to that call. Invalid or
    missing bounds raise and must remain unaccepted; no substitute is invented.
    """
    _alive(deadline);body=dict(model_audit);digest=body.pop('audit_sha256')
    _need(canonical_hash(body)==digest and body['passed'] is True
          and body['kind']=='INDEPENDENT_MULTI_ANCHOR_MODEL_AUDIT_V1','Original model audit missing/changed')
    actual=coefficient_sha256(model)
    _need(actual==expected_coefficient_sha256==body['coefficient_sha256'],'Audited native-call coefficients changed')
    nr=native_record;A=csr_matrix(model['matrix'])
    _need(nr['backend']=='gurobipy' and nr['gurobi_version']==[12,0,3],'Native backend/version differs')
    _same(nr['original_matrix_shape'],list(A.shape),'Native matrix shape differs')
    _need(nr['original_matrix_nnz']==A.nnz,'Native matrix sparsity differs')
    opts=nr['effective_options']
    _need(opts['Threads']==1 and opts['MIPGap']==.005 and opts['Seed']==0
          and opts['FeasibilityTol']==1e-9 and all(opts[k]==1e-8 for k in ('OptimalityTol','IntFeasTol'))
          and type(nr['requested_time_limit_seconds']) in (int,float)
          and 0<nr['requested_time_limit_seconds']<=600,'Native resource/tolerance policy differs')
    names={2:'OPTIMAL',7:'ITERATION_LIMIT',8:'NODE_LIMIT',9:'TIME_LIMIT',10:'SOLUTION_LIMIT',
           11:'INTERRUPTED',15:'USER_OBJ_LIMIT',16:'WORK_LIMIT',17:'MEM_LIMIT'}
    code=nr['raw_model_status']
    _need(type(code) is int and code in names and nr['status']==names[code]
          and type(native_status) is int and native_status==(0 if code==2 else 1)
          and nr['certificate_failures']==[],'Native status/certificate cannot admit a bound')
    _need(nr['is_mip'] is True and nr['bound_source']=='MIP_ObjBound'
          and nr['solver_bound_consistent'] is True,'Native MIP bound provenance differs')
    for value in (native_bound,nr['raw_solver_bound_without_constant_usd'],nr['solver_bound_without_constant_usd'],fixed_upper_bound):
        _need(type(value) in (float,int) and math.isfinite(value),'Missing finite native bound/fixed UB')
    _need(native_bound==nr['solver_bound_without_constant_usd']==nr['raw_solver_bound_without_constant_usd'],
          'Native raw/reported lower bound differs')
    rl,ru=np.asarray(model['row_lower']),np.asarray(model['row_upper'])
    eq=np.isfinite(rl)&np.isfinite(ru)&(rl==ru)
    mapping=dict(eq=np.flatnonzero(eq),le=np.flatnonzero(np.isfinite(ru)&~eq),ge=np.flatnonzero(np.isfinite(rl)&~eq))
    summary=nr['row_map_summary']
    for name,rows in mapping.items():
        _need(summary[name]==dict(count=len(rows),source_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()),
              'Native original-row mapping differs')
    _need(summary['original_rows']==A.shape[0] and summary['translated_rows']==sum(map(len,mapping.values()))
          and nr['model_rows']==summary['translated_rows'] and nr['model_columns']==A.shape[1],
          'Native translated row/column counts differ')
    _need(code!=2 or incumbent is not None,'Native OPTIMAL is missing its incumbent')
    objective=None
    if incumbent is not None:
        z=np.asarray(incumbent,dtype=float);_need(z.shape==(A.shape[1],) and np.isfinite(z).all(),'Invalid native incumbent')
        maxval=lambda x:float(np.max(x,initial=0.))
        act=A@z;row=np.maximum(np.maximum(rl-act,act-ru),0.)
        lo,hi=np.asarray(model['lower_bounds']),np.asarray(model['upper_bounds']);col=np.maximum(np.maximum(lo-z,z-hi),0.)
        integer=np.asarray(model['integrality'])==1
        _need(max(maxval(row),maxval(row/np.maximum(1.,abs(A)@abs(z))),maxval(col),
                  maxval(col/np.maximum(1.,abs(z))),maxval(abs(z[integer]-np.rint(z[integer]))))<=1e-8,
              'Native incumbent violates original auxiliary model')
        objective=float(np.asarray(model['c'])@z)
        _need(nr['valid_incumbent'] is True and nr['objective_matches'] is True and nr['solution_count']>0
              and math.isfinite(nr['incumbent_without_constant_usd'])
              and abs(objective-nr['incumbent_without_constant_usd'])<=1e-4+1e-7*max(1.,abs(objective)),
              'Native incumbent objective/metadata differs')
        _need(native_bound<=objective+1e-4+1e-7*max(1.,abs(objective)),'Native LB exceeds auxiliary incumbent')
    else:
        _need(nr['valid_incumbent'] is False and nr['solution_count']==0
              and nr['incumbent_without_constant_usd'] is None,'Missing retained native incumbent')
    shifted=float(native_bound+model['objective_constant_usd'])
    tol=1e-4+1e-7*max(1.,abs(shifted),abs(fixed_upper_bound))
    _need(shifted<=fixed_upper_bound+tol,'Native auxiliary bound contradicts accepted fixed-plan UB')
    _alive(deadline)
    return _seal(dict(kind='INDEPENDENT_CASE0_REFINED_NATIVE_AUXILIARY_BOUND_AUDIT_V1',passed=True,
        expected_feasibility_tolerance=1e-9,actual_feasibility_tolerance=opts['FeasibilityTol'],
        baseline_auditor_source_sha256='9fd199f0eab88b7b65326875891ebcd415891241b536eb79543f17a5ff9404ea',
        coefficient_sha256=actual,native_status=nr['status'],native_status_code=code,
        native_lower_bound_without_constant_usd=float(native_bound),objective_constant_usd=model['objective_constant_usd'],
        lower_bound_usd=shifted,fixed_upper_bound_usd=float(fixed_upper_bound),
        auxiliary_incumbent_objective_usd=None if objective is None else objective+model['objective_constant_usd'],
        native_branch_and_bound_proof_replayed=False,new_candidate_upper_bound_verified=False,
        solver_calls=0,scientific_acceptance=False,requires_terminal_parent_settlement=True))


def prepare_epigraph(A,arrays,header,ub,P,probability,weighted_result,weighted_directory,*,output,deadline,binding,dual_module):
    """Recheck original eight duals with frozen prior source; then full-P model."""
    out=Path(output);prior=Path(weighted_directory);ids=tuple(header['scenario_ids']);nf=header['variable_slices']['first_stage'][1]
    need(weighted_result['kind']=='CASE0_SAVED_DUAL_BOUND_RESULT_V1' and
         weighted_result['status']=='AUDITED_NUMERICAL_AUXILIARY_BOUND' and weighted_result['model_sha256']==header['model_sha256'] and
         weighted_result['plan_sha256']==ub['plan_sha256'] and weighted_result['upper_bound_usd']==ub['upper_bound_usd'] and
         weighted_result['planning_acceptance'] is False and math.isfinite(weighted_result['lower_bound_usd']),
         'Prior audited weighted bound/model/anchor differs')
    rechecked=out/'revalidated_cuts';rechecked.mkdir()
    weighted,_=dual_module.derive_saved_cuts(A,arrays,header,ub,P,probability,output=rechecked,deadline=deadline,binding=binding)
    prior_audit=read(file_reference(prior/'model_audit.json'));prior_construction=read(file_reference(prior/'construction.json'))
    need(coefficient_sha256(weighted)==prior_audit['coefficient_sha256']==prior_construction['bound_problem_coefficient_sha256'],
         'Revalidated original weighted coefficient identity differs')
    cuts={};floors={};slices=header['variable_slices'];m=header['block_map'][0]['row_stop']
    z=np.r_[ub['plan']['x'],np.asarray(ub['plan']['r']).ravel()]
    first=dict(matrix=A[:m,:nf].tocsr(),objective_constant_usd=0.,
        **{k:arrays[k][:nf] for k in ('c','integrality','lower_bounds','upper_bounds')},
        **{k:arrays[k][:m] for k in ('row_lower','row_upper')})
    need(coefficient_sha256(first)==prior_construction['original_first_stage_coefficient_sha256'],
         'Original first-stage domain/cost identity differs')
    for i,sid in enumerate(ids):
        left(deadline);oldref=file_reference(prior/f'cut_{i:03d}.json');old=read(oldref);new=read(file_reference(rechecked/f'cut_{i:03d}.json'))
        oldgrad=file_reference(prior/f'gradient_{i:03d}.npy');verify_reference(oldgrad)
        g=np.load(oldgrad['path'],allow_pickle=False);newg=np.load(rechecked/f'gradient_{i:03d}.npy',allow_pickle=False)
        need(old==new and np.array_equal(g,newg),'Saved original cut or gradient changed during reuse')
        cut=old['cut'];need(cut['scenario_id']==sid and cut['template_sha256']==header['recourse_template_sha256s'][i],
                           'Saved full support cut axis differs')
        cuts[i]=(cut,g);a,b=slices['recourses'][sid]
        rows=[v for v in header['block_map'] if v['scope']==sid and v['kind']=='risk_epigraph']
        need(len(rows)==1 and rows[0]['row_stop']==rows[0]['row_start']+1,'Original risk cost row missing')
        c=A[rows[0]['row_start'],a:b].toarray().ravel()
        floor=certify_box_floor(dict(c=c,lower_bounds=arrays['lower_bounds'][a:b],upper_bounds=arrays['upper_bounds'][a:b],
            scenario_id=sid,template_sha256=cut['template_sha256']),scenario_constant_usd=0.)
        floors[i]=floor;atomic_json(out/f'floor_{i:03d}.json',_json(dict(floor=floor,original_model_sha256=header['model_sha256'],
            original_cost_row=rows[0],original_column_slice=[a,b],saved_cut_reference=oldref,saved_gradient_reference=oldgrad)))
    model,construction=build_epigraph_master(first,ids,cuts,probability_set=P,floors=floors)
    groups=[dict(anchor_first_stage=z,cuts=cuts,source_references=[binding['saved_dual_result'],binding['ub_result']])]
    check=audit_multi_anchor_model(model,first,ids,groups,probability_set=P,floors=floors,deadline=deadline)
    need(not np.any((abs(model['matrix'].data)>0)&(abs(model['matrix'].data)<1e-13)),
         'Native would discard tiny cut coefficients; no unreviewed coefficient repair authorized')
    # Nonbinding old integer start, with only its free epigraph intercept raised
    # enough to satisfy all cuts/floors. No first-stage variable is fixed.
    start=np.r_[z,probability['theta0'],probability['alpha'],probability['beta']]
    activity=model['matrix']@start;tail_error=float(np.max(model['row_lower'][m:]-activity[m:],initial=0.))
    start[nf]+=max(0.,tail_error)+1e-6
    cert=_check_vector(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],start)
    need(cert['passed'],'Original anchor is not a feasible nonbinding compact start')
    atomic_json(out/'construction.json',_json(construction));atomic_json(out/'model_audit.json',_json(check));atomic_json(out/'start_check.json',_json(cert))
    return model,check,first,start


def evaluate_native_result(model,model_audit,first,header,ub,weighted_result,solved,*,output,deadline,binding):
    out=Path(output);expected=coefficient_sha256(model);nf=len(first['c']);H=header['variable_slices']['x'][1]
    result=dict(kind='CASE0_COMPACT_DRO_CANDIDATE_RESULT_V1',status='NO_NEW_CANDIDATE',method_id='EA_DRO',
        model_sha256=header['model_sha256'],auxiliary_model_sha256=expected,source_binding=binding,
        plan=None,plan_sha256=None,first_stage_numeric_passed=False,changed_plan=False,
        prior_plan_sha256=ub['plan_sha256'],prior_upper_bound_usd=ub['upper_bound_usd'],prior_lower_bound_usd=weighted_result['lower_bound_usd'],
        upper_bound_usd=None,lower_bound_usd=None,gap=None,gap_passed=False,planning_acceptance=False,scientific=False,formal_OOS=False,
        auxiliary_objective_usd=None,auxiliary_mip_gap=None if solved.mip_gap is None else float(solved.mip_gap),native_status=int(solved.status),new_candidate_requires_fresh_eight_Q=True,
        full_DRO_upper_bound_not_evaluated=True,actual_solve_type='compact_full_ambiguity_integer_epigraph_master')
    if solved.x is not None:
        v=np.asarray(solved.x,float);np.save(out/'native_vector.npy',v,allow_pickle=False)
        aux=_check_vector(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],v)
        need(aux['passed'] and solved.native_record['valid_incumbent'] is True,'Native compact incumbent violates original auxiliary arrays')
        z=v[:nf];check=_check_vector(first['c'],first['integrality'],first['lower_bounds'],first['upper_bounds'],first['matrix'],first['row_lower'],first['row_upper'],z)
        need(check['passed'],'New candidate violates original first-stage rows/bounds/integrality')
        old=np.r_[ub['plan']['x'],np.asarray(ub['plan']['r']).ravel()];difference=float(np.max(abs(z-old),initial=0.))
        plan=dict(x=z[:H].tolist(),r=z[H:].reshape(np.asarray(ub['plan']['r']).shape).tolist(),
            task_sha256=header['task_sha256'],config_hash=header['config_hash'])
        changed=canonical_hash(plan)!=ub['plan_sha256']
        atomic_json(out/'first_stage_check.json',_json(check));atomic_json(out/'candidate_plan.json',dict(plan=plan,plan_sha256=canonical_hash(plan),source_binding=binding))
        result.update(status='DISTINCT_INTEGER_CANDIDATE' if changed else 'UNCHANGED_INTEGER_CANDIDATE',plan=plan,
            plan_sha256=canonical_hash(plan),first_stage_numeric_passed=True,changed_plan=changed,
            max_first_stage_difference=difference,auxiliary_objective_usd=float(model['c']@v+model['objective_constant_usd']),
            first_stage_check_reference=file_reference(out/'first_stage_check.json'),native_vector_reference=file_reference(out/'native_vector.npy'))
    try:
        # Use the installed R5 signature exactly; no post-R5 relaxation kwargs.
        check=audit_case0_native_bound(model,model_audit,solved.native_record,native_status=solved.status,
            native_bound=solved.mip_dual_bound,incumbent=solved.x,expected_coefficient_sha256=expected,
            fixed_upper_bound=ub['upper_bound_usd'],deadline=deadline)
        atomic_json(out/'native_bound_audit.json',_json(check));result.update(lower_bound_usd=check['lower_bound_usd'],
            native_bound_audit_reference=file_reference(out/'native_bound_audit.json'))
    except ValueError as error:result['native_bound_rejection']=str(error)
    result['retained_original_lower_bound_usd']=max(weighted_result['lower_bound_usd'],result['lower_bound_usd']) if result['lower_bound_usd'] is not None else weighted_result['lower_bound_usd']
    left(deadline);atomic_json(out/'result.json',_json(result));return result


def numeric_run(refs,out,deadline,solver_seconds,hash_checker,dual_module):
    left(deadline);header=read(refs['model']);verify_reference(refs['matrix']);verify_reference(refs['model_arrays'])
    A=load_npz(refs['matrix']['path']).tocsr()
    with np.load(refs['model_arrays']['path'],allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    hash_checker.check_model_hash(A,arrays,header)
    ub=read(refs['ub_result']);weighted=read(refs['saved_dual_result']);identity=read(refs['input_identity']);cal=read(refs['calibration_manifest'])
    need(cal['context_sha256']==canonical_hash(identity),'Original calibration input differs')
    moments=read(cal['moments']);verify_reference(cal['arrays'])
    with np.load(cal['arrays']['path'],allow_pickle=False) as data:features=data['planning'].copy()
    bounds=moments['moments']['bounds_by_rho']['1.0'];P=ProbabilitySet(tuple(header['scenario_ids']),FEATURE_NAMES,features,bounds['lower'],bounds['upper'])
    probability=read(refs['probability'])
    model,model_audit,first,start=prepare_epigraph(A,arrays,header,ub,P,probability,weighted,Path(refs['saved_dual_result']['path']).parent,
        output=out,deadline=deadline,binding=refs,dual_module=dual_module)
    save_npz(out/'matrix.npz',model['matrix']);np.savez_compressed(out/'model_arrays.npz',**{k:model[k] for k in
        ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')})
    del A,arrays;gc.collect();grant=min(solver_seconds,left(deadline)-5.)
    need(grant>0,'No original master allowance after compact preparation')
    atomic_json(out/'native_started.json',dict(grant_seconds=grant,coefficient_sha256=coefficient_sha256(model),
        full_ambiguity_epigraph=True,integer_first_stage=True,nonbinding_old_anchor_start=True,all_preparation_charged_to_master=True))
    solved=solve_milp(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],
        options=dict(time_limit=grant,mip_rel_gap=.005,threads=1,solver_feasibility_tolerance=1e-9),log_path=out/'native.log',start=start)
    atomic_json(out/'native_record.json',_json(solved.native_record))
    return evaluate_native_result(model,model_audit,first,header,ub,weighted,solved,output=out,deadline=deadline,binding=refs)


def run_request(request_path,expected_sha256):
    began=time.monotonic();rr=file_reference(request_path);need(rr['sha256']==expected_sha256,'Saved-dual request changed')
    request=read(rr);refs=request.get('references',{})
    need(set(request)=={'kind','references','output_directory','budget_seconds','solver_seconds'} and
         request['kind']=='CASE0_COMPACT_DRO_CANDIDATE_REQUEST_V1' and set(refs)=={'model','matrix','model_arrays','input_identity','calibration_manifest',
         'config','source_manifest','state','settlement','ub_result','probability','model_hash_checker','review','saved_dual_result','saved_dual_source'},'Saved-dual request schema differs')
    review=read(refs['review']);without=dict(request,references={k:v for k,v in refs.items() if k!='review'})
    need(review['kind']=='CASE0_COMPACT_DRO_CANDIDATE_REVIEW_V1' and review['passed'] is True and review['request_without_review']==without and
         review['auditor_sha256']==file_reference(__file__)['sha256'],'Exact saved-dual source/request review required')
    state=read(refs['state']);settlement=read(refs['settlement']);key='size0/EA_DRO'
    need(settlement['kind']=='CASE0_SAVED_DUAL_BOUND_SETTLEMENT_V1' and settlement['scientific_planning_acceptance'] is False and
         settlement['ub_result_reference']==refs['ub_result'] and settlement['result_reference']==refs['saved_dual_result'] and
         settlement['numerical_auxiliary_bound_accepted'] is True and state['saved_dual_bound_result_reference']==refs['saved_dual_result'] and state['saved_dual_bound_settlement_reference']==refs['settlement'] and
         state['saved_candidate_ub_reference']==refs['ub_result'] and state['active_config_reference']==refs['config'] and
         state['source_release_reference']==refs['source_manifest'] and state['compute_seconds']==settlement['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={key:settlement['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={key:settlement['closing_master_seconds']} and state['active_job_id'] is None and
         state['active_configuration'] is None and state['pending_saved_dual_bound'] is None and state['evaluation_opened'] is False and state['completed']=={},
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
    need(inventory.get('src/evcs_experiments/multi_anchor_bound_audit.py')=='9fd199f0eab88b7b65326875891ebcd415891241b536eb79543f17a5ff9404ea' and
         len(inventory)==len(source['files']) and all(inventory.get(k)==v for k,v in config['source_sha256'].items()),'Producer source inventory differs')
    inputs=root/'planning/size0/EA_DRO/inputs';solve=inputs.parent/'compact-metadata-1/solve'
    for key,name in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]:need(Path(refs[key]['path'])==solve/name,'Original saved array path differs')
    need(Path(refs['input_identity']['path'])==inputs/'identity.json' and Path(refs['calibration_manifest']['path'])==inputs/'calibration_manifest.json','Original input/calibration path differs')
    ub=read(refs['ub_result']);need(ub['kind']=='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' and ub['passed'] is True and
        ub['first_stage_physical_passed'] is True and ub['full_support_fixed_Q_and_KKT_passed'] is True and ub['probability_primal_dual_passed'] is True and
        len(ub['scenario_ids'])==8 and all(ub['source_binding'][k]==refs[k] for k in ('model','input_identity','config','source_manifest')) and
        Path(refs['probability']['path'])==Path(refs['ub_result']['path']).parent/'probability.json','Original audited UB source differs')
    verify_reference(refs['model_hash_checker']);need(refs['model_hash_checker']['sha256']==HASH_CHECKER_SHA,'Original semantic checker source differs')
    checker=load_module(refs['model_hash_checker'],'case0_compact_original_hash',HASH_CHECKER_SHA)
    dual_module=load_module(refs['saved_dual_source'],'case0_compact_original_dual',SAVED_DUAL_SHA)
    weighted=read(refs['saved_dual_result'])
    need(weighted['lower_bound_usd']==settlement['lower_bound_usd'] and all(weighted['source_binding'][k]==refs[k] for k in ('model','matrix','model_arrays','input_identity','calibration_manifest','config','source_manifest','ub_result','probability','model_hash_checker')),
         'Prior saved dual source/model/input binding differs')
    out=Path(request['output_directory']).resolve();need(out.is_relative_to(root/'operations') and out!=root/'operations','New saved-dual output required')
    out.mkdir(parents=True,exist_ok=False)
    try:return numeric_run({k:v for k,v in refs.items() if k!='review'},out,deadline,solver_seconds,checker,dual_module)
    except BaseException as error:
        atomic_json(out/'failure.json',dict(type=type(error).__name__,error=str(error),request_reference=rr,planning_acceptance=False));raise
    finally:
        elapsed=time.monotonic()-began;atomic_json(out/'operation_accounting.json',dict(actual_seconds=elapsed,master_seconds=elapsed,
            request_reference=rr,all_preparation_extraction_assembly_solver_audit_charged_to_master=True,
            scope='Whole isolated parent actual elapsed is authoritative conservative master/configuration/global charge; inner time is nested, never added twice.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();r=run_request(a.request,a.expected_sha256);print(json.dumps({k:r[k] for k in ('status','changed_plan','lower_bound_usd','plan_sha256','planning_acceptance')}))

if __name__=='__main__':main()
