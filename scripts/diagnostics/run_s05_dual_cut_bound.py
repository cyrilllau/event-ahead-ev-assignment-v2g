"""Bound the formal problem using saved recourse dual cuts and a feasible law."""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.config import get_paths
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation
from evcs_experiments.relaxation_bound import solve_relaxation_bound
from evcs_experiments.dual_cut_bound import build_dual_cut, build_weighted_dual_problem

KIND='S05_SAVED_DUAL_CUT_BOUND_V1'


def need(ok,message):
    if not ok:raise ValueError(message)


def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def ref(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def verify(r):
    need(set(r)=={'path','sha256','size_bytes'} and Path(r['path']).is_absolute() and ref(r['path'])==r,'Changed evidence bytes')
    return Path(r['path'])


def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
    return ref(p)


def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def budget_guard(request,settlement):
    need(request['kind']==KIND and request['plan_id']==settlement['plan_id'],'Wrong plan')
    need(settlement['kind']=='S05_SAVED_RMP_BOUND_PROBE_PARENT_ACCOUNTING_V1'
         and settlement['status']=='SETTLED_NO_BOUND_PENDING_NEXT_DECISION'
         and settlement['accounting_PASS'] is True and settlement['cumulative_seconds_provisional'] is False
         and settlement['nested_seconds_added'] is False,'Unreviewed predecessor settlement')
    total=Decimal(str(settlement['opening_budget']['opening_cumulative_seconds']))+Decimal(str(settlement['terminal_parent_seconds']))
    need(total.is_finite() and total>=0 and float(total)==settlement['cumulative_accounted_seconds']
         and float(Decimal(7200)-total)==settlement['remaining_seconds'] and settlement['cumulative_cap_seconds']==7200,'Wrong cumulative cost')
    need(request['wall_seconds']==590 and request['solver_seconds']==120 and request['parent_job']['timeout_seconds']==600,'Wrong probe limit')
    need(settlement['remaining_seconds']>=600+1790+10,'Final evidence reserve unavailable')
    return float(total)


def parent(req,rr):
    p=req['parent_job'];paths=get_paths();d=Path(p['directory']).resolve()
    need(d==paths.campaign_root.resolve()/'jobs'/p['job_id'],'Wrong parent path')
    need(os.environ.get('EVCS_JOB_ID')==p['job_id'] and os.environ.get('EVCS_JOB_DIRECTORY')==str(d),'Not supervised')
    mr=ref(d/'manifest.json');m=read(mr['path']);s=read(d/'status.json')
    need(m['argv']==[sys.executable,'-B',str(Path(__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',rr['sha256']]
         and m['cwd']==str(paths.project_root.resolve()) and m['outroot']==str(d.parent)
         and m['job_id']==p['job_id'] and m['spec_id']=='S05','Wrong parent command')
    need(m['threads']==p['threads']==1 and m['timeout_seconds']==p['timeout_seconds']==600
         and m['grace_seconds']==p['grace_seconds']==5,'Wrong parent resources')
    need(verify(p['controller_reference'])==Path(m['controller_path']).resolve()==paths.project_root/'src/evcs_experiments/runtime/jobctl.py','Wrong controller')
    need(s['state']=='RUNNING' and os.getpid()==s['command_pid'] and os.getsid(0)==s['supervisor_sid']
         and observation(s)['supervisor_alive'],'Wrong live parent')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        need(os.environ.get(k)=='1','Wrong thread environment')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(m['created_at_utc'])).total_seconds()
    need(0<=age<590,'Expired parent')
    return dict(manifest_reference=mr,job_id=p['job_id'],deadline=time.monotonic()+590-age)


def first_digest(meta,A,v):
    body={k:meta[k] for k in ('task_sha256','config_hash','task_ids','periods','variable_map','constraint_map','units')}
    h=hashlib.sha256(json.dumps(body,sort_keys=True).encode())
    for x in (A.data,A.indices,A.indptr,v['row_lower'],v['row_upper'],v['lower_bounds'],v['upper_bounds'],v['integrality'],v['c']):
        a=np.asarray(x,dtype='<i8' if np.asarray(x).dtype.kind in 'iu' else '<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def execute(request_path,expected_sha):
    began=time.monotonic();rr=ref(request_path);need(rr['sha256']==expected_sha,'Changed request')
    req=read(request_path);refs=req['references'];paths=get_paths();pr=parent(req,rr)
    attempt=paths.campaign_root/'dual_bound_attempts'/canonical_hash(refs['settlement']);attempt.mkdir(parents=True,exist_ok=False)
    write(attempt/'reservation.json',dict(request_reference=rr,parent=pr,settlement=refs['settlement'],requires_terminal_parent_settlement=True))
    out=Path(req['output_root']).resolve();need(out==paths.campaign_root.resolve()/'dual_bound_probes'/pr['job_id'],'Wrong output')
    out.mkdir(parents=True,exist_ok=False)
    used={}
    def checked(r):verify(r);used[r['path']]=r;return read(r['path'])
    def artifact_record(r):
        value=checked(r)
        for a in value['artifacts']:verify(a);used[a['path']]=a
        return value
    try:
        for r in refs.values():verify(r);used[r['path']]=r
        root=paths.project_root.resolve();manifest=read(refs['source_manifest']['path']);files={}
        need(Path(refs['source_manifest']['path']).parent==root,'Wrong source root')
        for row in manifest['files']:
            p=(root/row['path']).resolve();need(p.is_relative_to(root) and str(p) not in files,'Source path escape or duplicate')
            r=dict(path=str(p),sha256=row['sha256'],size_bytes=row['size_bytes']);verify(r);files[str(p)]=r
        need(files[str(Path(__file__).resolve())]==refs['script'],'Unbound driver')
        for name,module in tuple(sys.modules.items()):
            if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
                p=getattr(module,'__file__',None)
                if p is not None:need(ref(p)==files.get(str(Path(p).resolve())),'Mixed source')
        spent=budget_guard(req,read(refs['settlement']['path']));settled=read(refs['settlement']['path'])
        card=read(refs['probability_acceptance']['path']);saved=read(refs['saved160_acceptance']['path'])
        need(card['status']=='PASS_PROBABILITY_RECOVERY_COMPONENT_REVIEW' and saved['status']=='PASS_SAVED160_COMPONENT_REVIEW'
             and card['references']['saved160_acceptance']==refs['saved160_acceptance'],'Unaccepted original components')
        cp=read(refs['checkpoint']['path']);s=cp['state'];pending=s['pending'];ids=s['identity']['scenario_ids']
        need(cp['sha256']==canonical_hash(s) and s['problem_sha256']==canonical_hash(s['identity'])
             ==card['problem_sha256']==saved['problem_sha256']==settled['original_problem_sha256'],'Wrong formal identity')
        need(card['references']['original_checkpoint']==refs['checkpoint'] and saved['references']['checkpoint']==refs['checkpoint'],'Wrong accepted checkpoint')
        need(s['status']=='PROBABILITY_FAILED' and len(ids)==160 and pending['iteration']==1,'Wrong original state')
        for key in ('history','raw_request','old_source_manifest'):
            need(refs[key] in saved['references'].values(),'Unbound original source or history')
        raw=read(refs['raw_request']['path']);need(raw['domain']=='formal' and raw['formal_OOS'] is False and raw['problem_identity']==s['identity'],'Wrong raw domain')
        history=read(refs['history']['path']);need(len(history)==163,'Incomplete original history')
        pe=checked(history[-1]);need(pe['event']=='probability' and refs['probability_input'] in pe['artifacts'],'Unbound original probability input')
        prob=read(refs['probability_solver']['path']);need(card['references']['probability_solver']==refs['probability_solver'] and prob['support_ids']==ids,'Wrong probability source/axis')
        pi=read(refs['probability_input']['path'])['effective_ambiguity'];P=ProbabilitySet(tuple(ids),tuple(pi['feature_ids']),np.asarray(pi['features']),np.asarray(pi['lower']),np.asarray(pi['upper']))
        need(P.ambiguity_sha256==prob['ambiguity_sha256']==s['identity']['effective_ambiguity_sha256'],'Wrong ambiguity content')
        auditor=verify(refs['digest_script']);need(files[str(auditor)]==refs['digest_script'],'Unbound template digest')
        spec=importlib.util.spec_from_file_location('_dual_bound_template_digest',auditor);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        firstrow=artifact_record(refs['first_record']);fd=Path(refs['first_record']['path']).parent
        need(fd==Path(refs['raw_request']['path']).parent/'first_stage','Wrong first-stage origin')
        fm=read(fd/'identity.json');fa=npz(fd/'arrays.npz');A=load_npz(fd/'matrix.npz').tocsr()
        need(first_digest(fm,A,fa)==fm['matrix_sha256']==s['identity']['first_stage_sha256'],'Wrong first-stage coefficients')
        first=dict(fa,matrix=A,objective_constant_usd=0.,source_model_sha256=fm['matrix_sha256'],formal_problem_sha256=s['problem_sha256'])
        anchor=np.r_[pending['plan']['x'],np.asarray(pending['plan']['r']).ravel()]
        cuts={};vectors={};positive=np.flatnonzero(np.asarray(prob['probabilities'])>0)
        for i in positive:
            i=int(i);er=history[i+2];event=artifact_record(er);ed=Path(er['path']).parent
            need(event['event']=='recourse' and event['scenario_index']==i and event['scenario_id']==ids[i]
                 and event['plan_sha256']==canonical_hash(pending['plan']),'Wrong recourse origin')
            tr=checked(next(x for x in event['artifacts'] if Path(x['path']).name=='template_reference.json'))
            template=artifact_record(tr);td=Path(tr['path']).parent
            meta=audit.metadata_header(td/'identity.json');v=npz(td/'arrays.npz')
            m={k:load_npz(td/(k+'.npz')).tocsr() for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
            need(audit.template_digest(meta,m,v,fm['matrix_sha256'])==template['template_sha256']==meta['template_sha256']
                 ==pending['recourse_certificates'][i]['template_sha256'],'Wrong template coefficients')
            solver=read(ed/'solver.json');need(canonical_hash(solver)==pending['recourse_certificates'][i]['solver_result_sha256'],'Wrong saved dual solver')
            dual=npz(ed/'dual_kkt.npz')
            for key in ('inequality','equality','lower','upper'):
                need(np.array_equal(dual[key],np.asarray(solver['certificate']['dual_'+key+'_marginals'])),'Changed original dual')
            cut,g=build_dual_cut(dict(v,**m,scenario_id=ids[i],template_sha256=template['template_sha256']),dual,scenario_constant_usd=s['identity']['constants_usd'][i],
                                 anchor_first_stage=anchor,anchor_q_usd=pending['q_usd'][i])
            cuts[i]=(cut,g);vectors['gradient_'+str(i)]=g
            write(out/('cut_'+str(i)+'.json'),dict(index=i,scenario_id=ids[i],event=er,template=tr,cut=cut))
        problem,aggregate=build_weighted_dual_problem(first,ids,prob['probabilities'],cuts,probability_set=P)
        write(out/'aggregate.json',aggregate);vectors.update(objective=problem['c'],probabilities=np.asarray(prob['probabilities']))
        np.savez_compressed(out/'cut_vectors.npz',**vectors)
        write(out/'input_references.json',dict(files=list(used.values())))
        remaining=min(req['solver_seconds'],pr['deadline']-time.monotonic()-30);need(remaining>0,'No solver budget')
        write(out/'execution.json',dict(kind=KIND,request_reference=rr,parent=pr,opening_cumulative_seconds=spent,
              original_problem_sha256=s['problem_sha256'],first_stage_sha256=fm['matrix_sha256'],matrix_shape=list(A.shape),
              positive_probability_indices=positive.tolist(),verification_and_cut_seconds=time.monotonic()-began,solver_grant_seconds=remaining))
        result,values=solve_relaxation_bound(problem,time_limit_seconds=remaining,threads=1,log_path=out/'solver_summary.json')
        np.savez_compressed(out/'lp_vectors.npz',**values)
        result.update(bound_kind='NUMERICAL_PROBABILITY_WEIGHTED_RECOURSE_DUAL_BOUND',request_reference=rr,
              original_native_lower_bound_usd=None,full_CCG_converged=False,normal_ccg_checkpoint_compatible=False,
              lockable=False,formal_OOS=False,scientific_acceptance=False,recourse_solver_calls=0,
              master_milp_calls=0,opening_cumulative_seconds=spent,requires_terminal_parent_settlement=True,
              cumulative_seconds_provisional=True,attempt_elapsed_seconds=time.monotonic()-began)
        write(out/'result.json',result);write(out/'artifact_manifest.json',dict(files=[ref(p) for p in sorted(out.iterdir()) if p.is_file()]))
        print(json.dumps(dict(status=result['status'],bound=result['lower_bound_usd'],scientific_acceptance=False)))
    except Exception as exc:
        write(out/'failure.json',dict(error=type(exc).__name__,message=str(exc),request_reference=rr,
              elapsed_seconds=time.monotonic()-began,requires_terminal_parent_settlement=True,scientific_acceptance=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--expected-request-sha256',required=True)
    a=p.parse_args();execute(a.request,a.expected_request_sha256)
