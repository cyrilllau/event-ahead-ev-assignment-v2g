"""One bounded continuous relaxation of the frozen Case0 extensive matrix.

This is an auxiliary numerical lower-bound diagnostic for an already audited
integer plan. It creates no MIP, recourse solve, campaign state, or OOS result.
A stopped native objective is never a lower bound. Saved dual vectors must pass
independent original-unit checks; exact zero elimination is separately replayed.
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
from scipy.sparse import load_npz
from evcs_experiments.campaign_batch import file_reference, verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.native_gurobi import _model_arrays, _row_map, _check_vector, GUROBI_VERSION
from evcs_experiments.runtime.jobctl import atomic_json

ZERO_SHA='cc0f4695194e0978685d4a539273ebc0f5b3bc94ea7908831da9777318fdc2d0'
ARRAY_KEYS={'c','row_lower','row_upper','lower_bounds','upper_bounds','integrality','objective_constant_usd'}


def need(ok,message):
    if not bool(ok):raise ValueError(message)


def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())


def left(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('Saved LP diagnostic allowance exhausted')
    return value


def close(a,b,label):
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),label+' differs')


def maxabs(a):return float(np.max(np.abs(a),initial=0.))


def check_model_hash(matrix,arrays,header):
    need(set(arrays)==ARRAY_KEYS,'Saved model array inventory differs')
    constant=float(arrays['objective_constant_usd']);constants=np.asarray(header['scenario_cost_constants_usd'],float)
    need(matrix.shape==tuple(header['shape']) and matrix.nnz==header['nnz'] and
         header['risk_model']=='DRO' and header['first_stage_fixed'] is False and
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


def check_dual(matrix,c,lower,upper,row_lower,row_upper,witness,constant=0.):
    """Numerical LP dual certificate in original units, without solver flags.

    Row multipliers use Gurobi's minimization convention (<= negative, >=
    positive). Bound multipliers are real saved reduced costs split by sign;
    unsupported infinite-bound multipliers are rejected, never discarded.
    """
    mapping=_row_map(row_lower,row_upper);n=len(c)
    rc=np.asarray(witness['reduced_cost'],float)
    need(rc.shape==(n,) and np.isfinite(rc).all(),'LP dual column axes/nonfinite')
    multipliers={k:np.asarray(witness[k],float) for k in mapping}
    need(all(v.shape==(len(mapping[k]),) and np.isfinite(v).all() for k,v in multipliers.items()),'LP witness row axes/nonfinite')
    d=np.zeros(matrix.shape[0]);dual=float(constant)
    for name,rows in mapping.items():
        d[rows]+=multipliers[name]
        rhs=row_upper if name=='le' else row_lower
        dual+=float(rhs[rows]@multipliers[name])
    lo=np.maximum(rc,0.);hi=np.minimum(rc,0.);fl=np.isfinite(lower);fu=np.isfinite(upper)
    signs=max(float(np.max(multipliers['le'],initial=0.)),float(np.max(-multipliers['ge'],initial=0.)),maxabs(lo[~fl]),maxabs(hi[~fu]))
    residual=c-matrix.T@d-rc
    station=maxabs(residual);free=(~fl)&(~fu)
    need(max(signs,station)<=1e-8,'LP dual signs/stationarity violated')
    dual+=float(lower[fl]@lo[fl]+upper[fu]@hi[fu])
    need(math.isfinite(dual),'Nonfinite independent dual objective')
    # A dual-simplex iterate can certify a bound while its primal is infeasible.
    # The separate, already certified integer incumbent provides our upper bound.
    primal=None;absolute_gap=None;strong=False;primal_check=dict(passed=False,reason='Primal unavailable')
    if 'primal' in witness:
        x=np.asarray(witness['primal'],float)
        primal_check=_check_vector(c,np.zeros(n,dtype=np.int8),lower,upper,matrix,row_lower,row_upper,x)
        if x.shape==(n,) and np.isfinite(x).all():
            value=float(c@x+constant)
            if math.isfinite(value):
                primal=value;absolute_gap=abs(primal-dual)
                strong=absolute_gap<=1e-4+1e-7*max(1.,abs(primal),abs(dual))
    return dict(passed=True,primal_available='primal' in witness,primal_feasible=primal_check['passed'],
        strong_duality_passed=strong,primal_objective_usd=primal,dual_objective_usd=dual,
        max_stationarity_error=station,max_dual_sign_error=signs,primal_check=primal_check,
        max_unsupported_lower_bound_multiplier=maxabs(lo[~fl]),max_unsupported_upper_bound_multiplier=maxabs(hi[~fu]),
        free_column_count=int(free.sum()),max_free_column_stationarity_error=maxabs(residual[free]),
        max_free_column_reduced_cost=maxabs(rc[free]),
        absolute_primal_dual_gap_usd=absolute_gap,original_unit_tolerance=1e-8,
        certificate_scope='Floating-point dual feasibility certificate at the existing original-unit tolerance; not symbolic exact arithmetic. Primal feasibility and strong duality are diagnostic only.')


def native_lp(matrix,arrays,output,deadline):
    """Same validated sparse translation, explicit continuous relaxation only."""
    import gurobipy as gp
    need(tuple(gp.gurobi.version())==GUROBI_VERSION,'Unchanged Gurobi 12.0.3 required')
    began=time.monotonic();left(deadline)
    c,integer,lo,hi,a,rl,ru=_model_arrays(arrays['c'],arrays['integrality'],arrays['lower_bounds'],arrays['upper_bounds'],
        matrix,arrays['row_lower'],arrays['row_upper'])
    mapping=_row_map(rl,ru);env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();model=None
    try:
        model=gp.Model(env=env);model.ModelSense=gp.GRB.MINIMIZE
        log=Path(output)/'lp.log'
        with log.open('x'):pass
        params=dict(OutputFlag=1,LogToConsole=0,Threads=1,Seed=0,FeasibilityTol=1e-9,OptimalityTol=1e-8,
                    IntFeasTol=1e-8,SoftMemLimit=24.,Method=1,PreDual=0)
        for key,value in params.items():model.setParam(key,value)
        model.setParam('LogFile',str(log))
        variables=model.addMVar(len(c),lb=lo,ub=hi,obj=c,vtype=gp.GRB.CONTINUOUS,name='relaxed')
        constraints={}
        for name,sense,rhs in [('eq',gp.GRB.EQUAL,rl),('le',gp.GRB.LESS_EQUAL,ru),('ge',gp.GRB.GREATER_EQUAL,rl)]:
            rows=mapping[name]
            if rows.size:constraints[name]=model.addMConstr(a[rows],variables,sense,rhs[rows],name=name)
        model.update();need(model.IsMIP==0,'Saved LP diagnostic unexpectedly produced a MIP')
        load=time.monotonic()-began;seconds=left(deadline)-20.
        need(seconds>0,'Insufficient original master allowance after LP loading/closing reserve')
        model.setParam('TimeLimit',seconds)
        atomic_json(Path(output)/'native_started.json',dict(kind='CASE0_SAVED_LP_NATIVE_START_V1',parameters=params,
            time_limit_seconds=seconds,load_seconds=load,matrix_shape=list(a.shape),matrix_nnz=a.nnz,
            original_integer_columns=int(integer.sum()),native_integer_columns=0,
            row_map={k:dict(count=len(v),sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for k,v in mapping.items()}))
        model.optimize()
        result=dict(status_code=int(model.Status),runtime_seconds=float(model.Runtime),load_seconds=load,
            native_elapsed_seconds=time.monotonic()-began,sol_count=int(model.SolCount),parameters=params,
            is_mip=bool(model.IsMIP),bound_available=False)
        witness=None
        # A stopped LP may have useful dual values. Admission nevertheless needs
        # independent dual feasibility below. An
        # unavailable/incomplete basis yields NO_VALID_BOUND, not a raw ObjVal.
        try:
            witness=dict(reduced_cost=np.asarray(variables.RC,float),
                **{k:np.asarray(constraints[k].Pi,float) if k in constraints else np.zeros(0) for k in mapping})
            try:witness['primal']=np.asarray(variables.X,float)
            except (gp.GurobiError,AttributeError) as error:result['primal_unavailable']=str(error)
            np.savez_compressed(Path(output)/'lp_witness.npz',**witness)
        except (gp.GurobiError,AttributeError) as error:result['witness_unavailable']=str(error)
        atomic_json(Path(output)/'native_result.json',result)
        return result,witness
    finally:
        if model is not None:model.dispose()
        env.dispose()


def numeric_diagnostic(*,references,output,master_deadline,upper_bound):
    """Read only the saved arrays/proof. Never rebuild templates or run Q/MIP."""
    out=Path(output);left(master_deadline)
    header=read(references['model']);master=read(references['master'])
    verify_reference(references['matrix']);verify_reference(references['model_arrays'])
    matrix=load_npz(references['matrix']['path']).tocsr()
    with np.load(references['model_arrays']['path'],allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    model_hash=check_model_hash(matrix,arrays,header)
    need(master['solution']['model_sha256']==model_hash and master['status']=='TIME_LIMIT' and
         master['lower_bound_usd'] is None and master['mip_gap'] is None,'Original failed-bound producer differs')
    proof=master['raw_solver']['zero_reduction'];path=get_paths().project_root/'scripts/audits/audit_zero_reduction.py'
    checker_ref=file_reference(path);need(checker_ref['sha256']==ZERO_SHA,'Exact-zero independent checker source differs')
    need(proof['original_columns']==matrix.shape[1] and proof['original_constraint_check_required'] is True,'Original reduction scope differs')
    spec=importlib.util.spec_from_file_location('saved_lp_independent_zero',path);checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    checked=checker.audit_zero_reduction(matrix,arrays['row_lower'],arrays['row_upper'],arrays['lower_bounds'],arrays['upper_bounds'],proof)
    cols=np.asarray(proof['retained_columns'],int);rows=np.asarray(proof['retained_rows'],int)
    reduced=matrix[rows][:,cols].tocsr();small={k:arrays[k][rows] for k in ('row_lower','row_upper')}
    small.update({k:arrays[k][cols] for k in ('c','integrality','lower_bounds','upper_bounds')})
    atomic_json(out/'reduction_check.json',dict(checked,checker_reference=checker_ref,original_model_sha256=model_hash))
    del master,proof;gc.collect();left(master_deadline)
    native,witness=native_lp(reduced,small,out,master_deadline)
    result=dict(kind='CASE0_SAVED_LP_BOUND_DIAGNOSTIC_V1',status='NO_VALID_BOUND',bound_available=False,
        model_sha256=model_hash,upper_bound_usd=upper_bound,lower_bound_usd=None,gap=None,gap_passed=False,
        source_binding=references,planning_acceptance=False,formal_OOS=False,Case1_acceptance=False,
        numerical_auxiliary_only=True,diagnostic_gap_within_threshold=False,
        original_model_kind='MILP',diagnostic_model_kind='continuous_relaxation',native_status_code=native['status_code'],
        exact_zero_proof=checked,master_runtime_includes_load_proof_and_audit=True)
    if witness is not None:
        try:
            left(master_deadline)
            dual=check_dual(reduced,small['c'],small['lower_bounds'],small['upper_bounds'],small['row_lower'],small['row_upper'],witness,float(arrays['objective_constant_usd']))
            original=dict(passed=False,reason='Primal unavailable');lift_exact=False
            if 'primal' in witness:
                x=np.asarray(witness['primal'],float)
                if x.shape==(len(cols),) and np.isfinite(x).all():
                    lifted=np.zeros(matrix.shape[1]);lifted[cols]=x;lift_exact=True
                    original=_check_vector(arrays['c'],np.zeros(matrix.shape[1],dtype=np.int8),arrays['lower_bounds'],arrays['upper_bounds'],
                        matrix,arrays['row_lower'],arrays['row_upper'],lifted)
                    # Check the descriptive objective map even when this LP
                    # primal is infeasible; primal feasibility does not gate LB.
                    mapped=float(arrays['c']@lifted+arrays['objective_constant_usd']);value=dual['primal_objective_usd']
                    original['objective_lift_within_tolerance']=(value is not None and math.isfinite(mapped) and
                        abs(mapped-value)<=1e-4+1e-7*max(1.,abs(mapped),abs(value)))
            lb=dual['dual_objective_usd'];need(lb<=upper_bound+1e-4+1e-7*abs(upper_bound),'Independent LP bound exceeds audited integer UB')
            gap=max(0.,upper_bound-lb)/max(1.,abs(upper_bound));left(master_deadline)
            atomic_json(out/'dual_check.json',dict(dual,original_relaxation_check=original,zero_lift_exact=lift_exact,original_primal_feasibility_required_for_dual_bound=False))
            result.update(status='VALID_NUMERICAL_LP_BOUND',bound_available=True,lower_bound_usd=lb,gap=gap,
                          gap_passed=False,diagnostic_gap_within_threshold=gap<=.005,dual_check_reference=file_reference(out/'dual_check.json'),
                          witness_reference=file_reference(out/'lp_witness.npz'))
        except (ValueError,TimeoutError) as error:result['certificate_rejection']=str(error)
    left(master_deadline);atomic_json(out/'result.json',result);return result


def run_request(request_path,expected_sha256):
    began=time.monotonic();request_ref=file_reference(request_path);need(request_ref['sha256']==expected_sha256,'LP request changed')
    request=read(request_ref);refs=request.get('references',{})
    need(set(request)=={'kind','references','output_directory','master_budget_seconds','budget_seconds'} and
         request['kind']=='CASE0_SAVED_LP_BOUND_REQUEST_V1' and set(refs)=={
         'master','model','matrix','model_arrays','input_identity','config','source_manifest','state','settlement','ub_result','review'},'Wrong saved LP request schema')
    review=read(refs['review']);without=dict(request,references={k:v for k,v in refs.items() if k!='review'})
    need(review['kind']=='CASE0_SAVED_LP_BOUND_REVIEW_V1' and review['passed'] is True and
         review['request_without_review']==without and review['auditor_sha256']==file_reference(__file__)['sha256'],'Exact LP source/request review required')
    state=read(refs['state']);settlement=read(refs['settlement']);key='size0/EA_DRO'
    need(settlement['kind']=='CASE0_SAVED_CANDIDATE_AUDIT_SETTLEMENT_V1' and settlement['ub_audit_passed'] is True and
         settlement['scientific_planning_acceptance'] is False and settlement['ub_result_reference']==refs['ub_result'] and
         state['saved_candidate_audit_settlement_reference']==refs['settlement'] and state['saved_candidate_ub_reference']==refs['ub_result'] and
         state['active_config_reference']==refs['config'] and state['source_release_reference']==refs['source_manifest'] and
         state['compute_seconds']==settlement['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={key:settlement['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={key:settlement['closing_master_seconds']} and
         state['active_job_id'] is None and state['active_configuration'] is None and state['pending_saved_candidate_audit'] is None and
         state['evaluation_opened'] is False and state['completed']=={},'Saved LP authority/settled budgets differ')
    read(settlement['terminal_review_reference'])
    cfgleft=1800-settlement['closing_configuration_seconds'];masterleft=1200-settlement['closing_master_seconds']
    close(cfgleft,settlement['remaining_configuration_seconds'],'Remaining configuration');close(masterleft,settlement['remaining_master_seconds'],'Remaining master')
    seconds=request['budget_seconds'];mseconds=request['master_budget_seconds']
    need(all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in (seconds,mseconds)) and
         mseconds<=masterleft and mseconds<seconds<=cfgleft and seconds<=21600-state['compute_seconds'],'LP allowance exceeds original cumulative budgets')
    deadline=began+seconds;paths=get_paths();root=paths.campaign_root
    need(root.name=='ieee33-v54-q1-case0-expanded' and Path(refs['state']['path']).is_relative_to(root/'operations') and
         Path(refs['state']['path'])!=root/'state.json','Expanded Case0 immutable launch snapshot required')
    jobid=os.environ.get('EVCS_JOB_ID');jobdir=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
    need(jobid and jobdir==root/'jobs'/jobid and (jobdir/'manifest.json').is_file(),'Exact jobctl parent required')
    parent=json.loads((jobdir/'manifest.json').read_text())
    guard=review['guard_reference'];verify_reference(guard)
    need(Path(guard['path']).parent==Path(request_ref['path']).parent==Path(__file__).resolve().parent and
         parent['argv']==[str(paths.project_root/'.venv/bin/python'),guard['path']],
         'Exact reviewed guard/actual parent argv differs')
    need(parent['job_id']==jobid and parent['outroot']==str(root/'jobs') and parent['threads']==1 and
         seconds<parent['timeout_seconds'] and parent['timeout_seconds']+parent['grace_seconds']<=cfgleft,'LP actual parent/resource allowance differs')
    from evcs_experiments.case0_isolation import verify_current_cgroup
    verify_current_cgroup()
    source=read(refs['source_manifest']);config=read(refs['config'])
    need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json','Original producer release required')
    from evcs_experiments.case0_campaign import verify_sources
    from evcs_experiments.case0_profiles import validate_config,EXPANDED
    verify_sources(config);need(validate_config(config,root)==EXPANDED,'Original expanded config required')
    inventory={r['path']:r['sha256'] for r in source['files']}
    need(len(inventory)==len(source['files']) and all(inventory.get(k)==v for k,v in config['source_sha256'].items()),'Producer source inventory differs')
    inputs=root/'planning/size0/EA_DRO/inputs';solve=inputs.parent/'compact-metadata-1/solve'
    for key,name in [('master','master.json'),('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]:
        need(Path(refs[key]['path'])==solve/name,'Original saved '+key+' path differs')
    need(Path(refs['input_identity']['path'])==inputs/'identity.json','Original input identity path differs');read(refs['input_identity'])
    ub=read(refs['ub_result']);header=read(refs['model'])
    need(ub['kind']=='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' and ub['passed'] is True and
         ub['first_stage_physical_passed'] is True and ub['full_support_fixed_Q_and_KKT_passed'] is True and
         ub['probability_primal_dual_passed'] is True and ub['model_sha256']==header['model_sha256'] and
         ub['scenario_ids']==header['scenario_ids'] and len(ub['scenario_ids'])==8 and
         all(ub['source_binding'][k]==refs[k] for k in ('master','model','input_identity','config','source_manifest')),
         'Independently audited UB source/model/support differs')
    upper=float(ub['upper_bound_usd']);need(math.isfinite(upper),'Finite audited UB required')
    out=Path(request['output_directory']).resolve();need(out.is_relative_to(root/'operations') and out!=root/'operations','New operation output required')
    out.mkdir(parents=True,exist_ok=False);master_began=time.monotonic();master_deadline=min(master_began+mseconds,deadline)
    try:
        return numeric_diagnostic(references={k:v for k,v in refs.items() if k!='review'},output=out,master_deadline=master_deadline,upper_bound=upper)
    except BaseException as error:
        atomic_json(out/'failure.json',dict(type=type(error).__name__,error=str(error),request_reference=request_ref,
            bound_available=False,planning_acceptance=False,formal_OOS=False));raise
    finally:
        atomic_json(out/'operation_accounting.json',dict(actual_seconds=time.monotonic()-began,
            master_seconds=time.monotonic()-master_began,master_budget_seconds=mseconds,request_reference=request_ref,
            scope='Single LP diagnostic including saved array loading, proof, native load/solve, certificate and output. Entire parent charged configuration/global by owner.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();r=run_request(a.request,a.expected_sha256)
    print(json.dumps({k:r[k] for k in ('status','upper_bound_usd','lower_bound_usd','gap','planning_acceptance')}))


if __name__=='__main__':main()
