"""Assemble an original extensive primal witness from an audited plan and Qs.

No optimization or scenario generation. This is a newly assembled witness,
not recovery of the producer's missing native master vector. All original
matrix rows, bounds, integrality and objective are checked without relaxation.
"""
import argparse,importlib.util,json,math,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.campaign_batch import verify_reference,file_reference
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.native_gurobi import _check_vector
from evcs_experiments.runtime.jobctl import atomic_json

HASH_CHECKER_SHA='7333d02ff1668ad04d75bc73f8b139570401093679a10a6160fd411a0c2c3582'

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)
def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())
def left(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Original extensive closure time exhausted')
def close(a,b,label):
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),label+' differs')

def assemble_witness(header,ub,probability,primal_by_scenario):
    """Every original column filled exactly once; no auxiliary objective used."""
    ids=header['scenario_ids'];sl=header['variable_slices'];plan=ub['plan'];n=header['shape'][1]
    need(header['risk_model']=='DRO' and ub['passed'] is True and ub['scenario_ids']==ids and
         ub['model_sha256']==header['model_sha256'] and canonical_hash(plan)==ub['plan_sha256'] and
         plan['task_sha256']==header['task_sha256'] and plan['config_hash']==header['config_hash'],
         'Original plan/task/model/support identity differs')
    need(set(primal_by_scenario)==set(ids) and len(ids)==len(set(ids)) and probability['support_ids']==ids and
         probability['ambiguity_sha256']==header['ambiguity_sha256'] and probability['certified_optimal'] is True,
         'Original recourse/probability full support differs')
    z=np.full(n,np.nan);covered=np.zeros(n,bool)
    def put(span,values,label):
        a,b=span;v=np.asarray(values,float).ravel()
        need(0<=a<b<=n and v.shape==(b-a,) and np.isfinite(v).all() and not covered[a:b].any(),label+' slice differs')
        z[a:b]=v;covered[a:b]=True
    put(sl['x'],plan['x'],'x');put(sl['r'],plan['r'],'r')
    for sid in ids:put(sl['recourses'][sid],primal_by_scenario[sid],'recourse '+sid)
    put(sl['theta0'],[probability['theta0']],'theta0');put(sl['alpha'],probability['alpha'],'alpha');put(sl['beta'],probability['beta'],'beta')
    need(covered.all() and np.isfinite(z).all(),'Unfilled original extensive columns')
    return z

def check_witness(A,arrays,header,ub,probability,primals):
    z=assemble_witness(header,ub,probability,primals)
    check=_check_vector(arrays['c'],arrays['integrality'],arrays['lower_bounds'],arrays['upper_bounds'],A,arrays['row_lower'],arrays['row_upper'],z)
    need(check['passed'],'Assembled witness violates original extensive rows/bounds/integrality')
    objective=float(arrays['c']@z+float(arrays['objective_constant_usd']))
    close(objective,ub['upper_bound_usd'],'Original full-vector objective and audited UB')
    close(ub['Fpre_usd']+probability['dual_objective_usd'],objective,'Original Fpre plus fullP dual')
    return z,dict(kind='CASE0_ASSEMBLED_ORIGINAL_EXTENSIVE_WITNESS_CHECK_V1',passed=True,
        model_sha256=header['model_sha256'],plan_sha256=ub['plan_sha256'],objective_usd=objective,original_array_check=check,
        original_native_primal_recovered=False,assembled_from_independent_Q_primal_and_P_dual=True,
        source_risk_model='DRO',new_Q_calls=0,new_master_calls=0,planning_acceptance=False,formal_OOS=False)

def numeric_closure(refs,out,deadline):
    left(deadline);header=read(refs['model']);ub=read(refs['ub_result']);probability=read(refs['probability'])
    need(ub['kind']=='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' and ub['first_stage_physical_passed'] and ub['full_support_fixed_Q_and_KKT_passed'] and ub['probability_primal_dual_passed'],
         'Audited original plan/fullQ/P required')
    need(ub['source_binding']['model']==refs['model'] and
         Path(refs['probability']['path'])==Path(refs['ub_result']['path']).parent/'probability.json','Original UB/physical producer pairing differs')
    checkerref=refs['model_hash_checker'];verify_reference(checkerref);need(checkerref['sha256']==HASH_CHECKER_SHA,'Original semantic checker changed')
    spec=importlib.util.spec_from_file_location('case0_closure_original_hash',checkerref['path']);checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    verify_reference(refs['matrix']);verify_reference(refs['model_arrays']);A=load_npz(refs['matrix']['path']).tocsr()
    with np.load(refs['model_arrays']['path'],allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checker.check_model_hash(A,arrays,header);left(deadline)
    completed=ub['completed_scenario_references'];need(len(completed)==len(header['scenario_ids']),'Incomplete fullQ axis')
    primals={}
    for i,entry in enumerate(completed):
        sid=header['scenario_ids'][i]
        need(entry['index']==i and entry['scenario_id']==sid and entry['template_sha256']==header['recourse_template_sha256s'][i],
             'Saved fixedQ scenario/template association differs')
        for ref in entry['files'].values():verify_reference(ref)
        with np.load(entry['files']['primal']['path'],allow_pickle=False) as f:
            need(f.files==['fixed'],'Unexpected original fixedQ primal inventory');primals[sid]=f['fixed'].copy()
    z,check=check_witness(A,arrays,header,ub,probability,primals);left(deadline)
    output=Path(out);output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'assembled_primal.npz',primal=z)
    check.update(source_binding=refs,assembled_primal_reference=file_reference(output/'assembled_primal.npz'))
    atomic_json(output/'result.json',check);left(deadline);return check

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    args=p.parse_args();started=time.monotonic();rr=file_reference(args.request);need(rr['sha256']==args.expected_sha256,'Closure request changed')
    req=read(rr);need(set(req)=={'kind','references','output_directory','budget_seconds','review_reference'} and
        req['kind']=='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_REQUEST_V1' and
        set(req['references'])=={'model','matrix','model_arrays','ub_result','probability','model_hash_checker'},'Closure request type/keys differ')
    seconds=req['budget_seconds'];need(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=20,'Invalid bounded closure time')
    review=read(req['review_reference']);base={k:v for k,v in req.items() if k!='review_reference'}
    need(review['kind']=='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_REVIEW_V1' and review['passed'] is True and review['request_without_review']==base and review['auditor_sha256']==file_reference(__file__)['sha256'],'Exact closure source/request independent review required')
    from evcs_experiments.config import get_paths
    from evcs_experiments.case0_isolation import verify_current_cgroup
    paths=get_paths();verify_current_cgroup();output=Path(req['output_directory']).resolve()
    need(output!=paths.campaign_root/'operations' and paths.campaign_root.name=='ieee33-v54-q1-case0-expanded' and
         output.is_relative_to(paths.campaign_root/'operations'),'Only fresh expanded operations output allowed')
    try:
        r=numeric_closure(req['references'],req['output_directory'],started+req['budget_seconds']);print(json.dumps(r))
    finally:
        atomic_json(Path(req['output_directory']).parent/'closure_accounting.json',dict(actual_seconds=time.monotonic()-started,
            master_seconds=0.,scope='Independent assembled original witness audit; full isolated audit receipt charged config/global separately; no optimization.'))
