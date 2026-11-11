"""One supervised full-support dual-cut epigraph master; no recourse or OOS."""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
import numpy as np
from scipy.sparse import load_npz, save_npz
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation
from evcs_experiments.epigraph_master import build_epigraph_master
from evcs_experiments.native_gurobi import solve_milp
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.first_stage import FirstStagePlan, assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks
from evcs_experiments.relaxation_bound import coefficient_sha256

KIND='S05_FULL_SUPPORT_EPIGRAPH_MASTER_PROBE_V1'

def need(ok,msg):
    if not ok:raise ValueError(msg)

def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))

def ref(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)

def verify(r):
    need(set(r)=={'path','sha256','size_bytes'} and Path(r['path']).is_absolute() and ref(r['path'])==r,'Changed bound evidence')
    return Path(r['path'])

def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
    return ref(p)

def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def candidate_plan(first,case,tasks,vector):
    v=np.asarray(vector);n=len(first.c);h=len(first.task_ids)
    need(v.ndim==1 and len(v)>=n and np.isfinite(v).all(),'Invalid candidate vector')
    plan=FirstStagePlan(v[:h],v[h:n].reshape(h,len(first.periods)),first.task_sha256,first.config_hash)
    check=first.evaluate(plan);physical=check_first_stage(case,tasks,plan)
    need(check.feasible and physical['passed'],'Candidate fails original first-stage physics or integrality')
    return plan,check,physical

def budget_guard(req,s):
    need(req['kind']==KIND and req['plan_id']==s['plan_id'],'Wrong plan/kind')
    need(s['kind']=='S05_DUAL_CUT_BOUND_PARENT_ACCOUNTING_V1' and s['status']=='SETTLED_PENDING_BOUND_REVIEW'
         and s['accounting_PASS'] is True and s['nested_seconds_added'] is False and s['cumulative_seconds_provisional'] is False,'Unsettled predecessor')
    opening=Decimal(str(s['opening_budget']['opening_cumulative_seconds']));duration=Decimal(str(s['terminal_parent_seconds']))
    total=opening+duration
    need(total.is_finite() and total>=0 and float(total)==s['cumulative_accounted_seconds']
         and s['cumulative_cap_seconds']==7200 and float(Decimal(7200)-total)==s['remaining_seconds'],'Wrong cumulative budget')
    need(req['wall_seconds']==590 and req['solver_seconds']==120 and req['parent_job']['timeout_seconds']==600,'Wrong probe limits')
    need(s['remaining_seconds']>=600+1790+10,'Final evaluation reserve unavailable')
    return float(total)

def parent(req,rr):
    paths=get_paths();p=req['parent_job'];d=Path(p['directory']).resolve()
    need(d==paths.campaign_root.resolve()/'jobs'/p['job_id'],'Wrong job directory')
    need(os.environ.get('EVCS_JOB_ID')==p['job_id'] and os.environ.get('EVCS_JOB_DIRECTORY')==str(d),'Not supervised')
    m=read(d/'manifest.json');s=read(d/'status.json')
    need(m['argv']==[sys.executable,'-B',str(Path(__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',rr['sha256']]
         and m['cwd']==str(paths.project_root.resolve()) and m['outroot']==str(d.parent)
         and m['job_id']==p['job_id'] and m['spec_id']=='S05','Wrong parent command')
    need(m['threads']==p['threads']==1 and m['timeout_seconds']==p['timeout_seconds']==600
         and m['grace_seconds']==p['grace_seconds']==5,'Wrong parent resource limits')
    need(verify(p['controller_reference'])==Path(m['controller_path']).resolve()==paths.project_root/'src/evcs_experiments/runtime/jobctl.py','Wrong job controller')
    need(s['state']=='RUNNING' and os.getpid()==s['command_pid'] and os.getsid(0)==s['supervisor_sid']
         and observation(s)['supervisor_alive'],'Wrong live parent')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        need(os.environ.get(k)=='1','Wrong thread environment')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(m['created_at_utc'])).total_seconds()
    need(0<=age<590,'Expired parent')
    return dict(job_id=p['job_id'],manifest_reference=ref(d/'manifest.json'),deadline=time.monotonic()+590-age)

def execute(request_path,expected):
    began=time.monotonic();rr=ref(request_path);need(rr['sha256']==expected,'Changed request')
    req=read(request_path);refs=req['references'];paths=get_paths();pr=parent(req,rr)
    attempt=paths.campaign_root/'epigraph_attempts'/canonical_hash(refs['settlement']);attempt.mkdir(parents=True,exist_ok=False)
    write(attempt/'reservation.json',dict(request_reference=rr,parent=pr,settlement=refs['settlement'],requires_terminal_parent_settlement=True))
    out=Path(req['output_root']).resolve();need(out==paths.campaign_root.resolve()/'epigraph_probes'/pr['job_id'],'Wrong output')
    out.mkdir(parents=True,exist_ok=False)
    try:
        for r in refs.values():verify(r)
        root=paths.project_root.resolve();manifest=read(refs['source_manifest']['path']);need(Path(refs['source_manifest']['path']).parent==root,'Wrong source root')
        files={}
        for row in manifest['files']:
            p=(root/row['path']).resolve();need(p.is_relative_to(root) and str(p) not in files,'Source escape/duplicate')
            r=dict(path=str(p),sha256=row['sha256'],size_bytes=row['size_bytes']);verify(r);files[str(p)]=r
        need(files[str(Path(__file__).resolve())]==refs['script'],'Unbound driver')
        for name,module in tuple(sys.modules.items()):
            if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
                p=getattr(module,'__file__',None)
                if p is not None:need(ref(p)==files.get(str(Path(p).resolve())),'Mixed package source')
        need(Path(os.environ['EVCS_CONFIG']).resolve()==Path(refs['runtime_config']['path']),'Wrong config')
        opening=budget_guard(req,read(refs['settlement']['path']))
        prior_review=read(refs['previous_result_review']['path'])
        need(prior_review['status']=='PASS_INDEPENDENT_NUMERICAL_DUAL_CUT_BOUND_REVIEW' and prior_review['passed'],'Unreviewed predecessor result')
        pack=read(refs['cut_pack']['path']);review=read(refs['cut_pack_review']['path'])
        need(review['status']=='PASS_FULL_SUPPORT_DUAL_CUT_FLOOR_PACK_REVIEW' and review['passed'] is True,'Unaccepted derived cuts')
        need(refs['cut_pack'] in review['references'].values(),'Review does not bind pack')
        for r in review['references'].values():verify(r)
        need(pack['status']=='DERIVED_FULL_SUPPORT_DUAL_CUT_FLOOR_PACK','Wrong cut pack')
        prep_seconds=Decimal(str(review['derivation_elapsed_seconds']))
        need(prep_seconds.is_finite() and prep_seconds>=0 and float(prep_seconds)==req['preprocessing_charge_seconds'],'Unaccounted cut construction')
        opening=float(Decimal(str(opening))+prep_seconds)
        need(7200-opening>=600+1790+10,'Preprocessing spent the final evaluation reserve')
        cp=read(refs['checkpoint']['path']);state=cp['state'];identity=state['identity']
        need(cp['sha256']==canonical_hash(state) and canonical_hash(identity)==state['problem_sha256']==pack['problem_sha256'],'Wrong original scientific identity')
        ids=identity['scenario_ids'];need(len(ids)==160 and ids==pack['support_ids'] and identity['constants_usd']==pack['constants_usd'],'Changed complete support')
        need(pack['source_references']['checkpoint']==refs['checkpoint'] and pack['source_references']['original_first_record']==refs['first_record'],'Wrong first-stage origin')
        verify(pack['gradients']);G=load_npz(pack['gradients']['path']).tocsr();verify(pack['used_original_references'])
        need(G.shape==(160,25200),'Unexpected full-scale cut dimensions')
        floors={i:r for i,r in enumerate(pack['floors'])};cuts={i:(r,G.getrow(i).toarray().ravel()) for i,r in enumerate(pack['cuts'])}
        pi=pack['effective_ambiguity'];P=ProbabilitySet(tuple(ids),tuple(pi['feature_ids']),np.asarray(pi['features']),np.asarray(pi['lower']),np.asarray(pi['upper']))
        need(P.ambiguity_sha256==identity['effective_ambiguity_sha256'],'Changed ambiguity')
        weights=np.asarray(pack['probabilities']);moments=P.features.T@weights
        need(weights.shape==(160,) and np.isfinite(weights).all() and np.all(weights>=0)
             and abs(weights.sum()-1)<=1e-8 and np.max(P.lower-moments,initial=0)<=1e-8
             and np.max(moments-P.upper,initial=0)<=1e-8,'Original ambiguity lacks the accepted feasible witness')
        raw=read(refs['raw_request']['path']);need(raw['problem_identity']==identity and raw['formal_OOS'] is False and raw['domain']=='formal','Wrong physical origin')
        taskref=raw['input_binding']['task_source']
        if set(taskref)=={'path','sha256','bytes'}:
            taskref=dict(path=taskref['path'],sha256=taskref['sha256'],size_bytes=taskref['bytes'])
        verify(taskref)
        case=load_physical_case(paths.data_root);tasks=load_frozen_tasks(taskref['path'],case,taskref['sha256']);first=assemble_first_stage(case,tasks)
        need(first.matrix_sha256==pack['first_stage_sha256']==identity['first_stage_sha256'],'Changed first-stage matrix')
        fr=read(refs['first_record']['path'])
        for r in fr['artifacts']:verify(r)
        fd=Path(refs['first_record']['path']).parent;old=arrays(fd/'arrays.npz');oldA=load_npz(fd/'matrix.npz').tocsr()
        need((first.A!=oldA).nnz==0 and all(np.array_equal(getattr(first,k),old[k]) for k in old),'Changed original first-stage arrays')
        fp=dict(old,matrix=oldA,objective_constant_usd=0.,source_model_sha256=first.matrix_sha256,formal_problem_sha256=state['problem_sha256'])
        model,assembly=build_epigraph_master(fp,ids,cuts,probability_set=P,floors=floors)
        write(out/'assembly.json',assembly);save_npz(out/'matrix.npz',model['matrix']);np.savez_compressed(out/'model_arrays.npz',**{k:model[k] for k in ('c','lower_bounds','upper_bounds','row_lower','row_upper','integrality','objective_constant_usd')})
        z=arrays(refs['start_vector']['path'])['primal'];n=len(first.c);need(z.shape==(n,),'Wrong start vector shape')
        startplan,startcheck,physical=candidate_plan(first,case,tasks,z)
        theta=max(r['intercept_usd']+(G.getrow(i)@z).item() for i,r in enumerate(pack['cuts']))
        theta=max(theta,max(float(r['floor_usd']) for r in pack['floors']))
        start=np.r_[z,theta,np.zeros(2*len(P.feature_ids))]
        write(out/'start_check.json',dict(matrix_check=startcheck.as_dict(),physical=physical,vector_reference=refs['start_vector']))
        np.savez_compressed(out/'start.npz',vector=start)
        grant=min(req['solver_seconds'],pr['deadline']-time.monotonic()-30);need(grant>0,'No solver budget')
        write(out/'execution.json',dict(kind=KIND,request_reference=rr,parent=pr,opening_cumulative_seconds=opening,preprocessing_charge_seconds=float(prep_seconds),verification_and_assembly_seconds=time.monotonic()-began,solver_grant_seconds=grant,first_stage_sha256=first.matrix_sha256,problem_sha256=state['problem_sha256']))
        solved=solve_milp(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],options=dict(time_limit=grant,mip_rel_gap=.001,threads=1),start=start,log_path=out/'native_solver.log')
        write(out/'native_solver.json',solved.native_record)
        result=dict(kind=KIND,request_reference=rr,status=str(solved.message),solver_status=int(solved.status),full_CCG_converged=False,scientific_acceptance=False,formal_OOS=False,lockable=False,original_native_lower_bound_modified=False,first_stage_sha256=first.matrix_sha256,problem_sha256=state['problem_sha256'],auxiliary_coefficient_sha256=coefficient_sha256(model),lower_bound_usd=None if solved.mip_dual_bound is None else float(solved.mip_dual_bound+model['objective_constant_usd']),candidate_available=solved.x is not None,full_support_candidate_UB_usd=None,recourse_solver_calls=0,probability_solver_calls=0,master_milp_calls=1,opening_cumulative_seconds=opening,requires_terminal_parent_settlement=True,cumulative_seconds_provisional=True)
        if solved.x is not None:
            v=np.asarray(solved.x);plan,check,physical=candidate_plan(first,case,tasks,v)
            np.savez_compressed(out/'candidate_vector.npz',vector=v);vr=ref(out/'candidate_vector.npz')
            p=dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash)
            write(out/'candidate_plan.json',p);write(out/'candidate_check.json',dict(matrix_check=check.as_dict(),physical=physical))
            result.update(candidate_plan_sha256=canonical_hash(p),candidate_first_cost_usd=check.objective_usd,candidate_auxiliary_objective_usd=float(solved.fun+model['objective_constant_usd']),candidate_x_count=int(np.count_nonzero(plan.x>.5)),candidate_vector=vr)
            write(out/'candidate_binding.json',dict(first_stage_sha256=first.matrix_sha256,problem_sha256=state['problem_sha256'],vector_reference=vr,vector_key='vector',first_stage_slice=[0,n],plan_sha256=canonical_hash(p),source_request=rr,source_native_result=ref(out/'native_solver.json'),formal_OOS=False,lockable=False))
        result['attempt_elapsed_seconds']=time.monotonic()-began
        write(out/'result.json',result);write(out/'artifact_manifest.json',dict(files=[ref(p) for p in sorted(out.iterdir()) if p.is_file()]))
        print(json.dumps(dict(status=result['status'],bound=result['lower_bound_usd'],candidate=result['candidate_available'])))
    except Exception as exc:
        write(out/'failure.json',dict(error=type(exc).__name__,message=str(exc),request_reference=rr,elapsed_seconds=time.monotonic()-began,requires_terminal_parent_settlement=True,scientific_acceptance=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--expected-request-sha256',required=True);a=p.parse_args();execute(a.request,a.expected_request_sha256)
