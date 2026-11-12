"""Full-H, full-support development C&CG with bounded template residency."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.sparse import save_npz

from evcs_experiments.ccg import CCGProblem, run_ccg
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256, source_provenance
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks, load_scenario_physical_inputs
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.profile_s03_scale import I03_SHA, TARGET, read_ref, rss_bytes
from evcs_experiments.validation.validate_s03_probability import _physical_acceptance, target_inputs
from evcs_experiments.validation.validate_s03_extensive import prerequisites
from evcs_experiments.validation.validate_s03_ccg import _save_model
from evcs_experiments.validation.s02_toys import _json_file, _save_solution

PROFILE_SHA = '359acc267be815b51918812636834437c060468111a07b88021f87ed2b3d5099'


def execution_budget(run_mode='probe', *, wall_seconds=None, rmp_seconds=None,
                     job_wall_seconds=None, max_completed_iterations=1):
    """Resolve operational limits without changing any solver accuracy setting."""
    if run_mode not in ('probe', 'completion'):
        raise ValueError('Unknown scale run mode')
    job_cap = 600. if run_mode == 'probe' else 7200.
    job_wall_seconds = job_cap if job_wall_seconds is None else job_wall_seconds
    wall_seconds = job_wall_seconds - 10. if wall_seconds is None else wall_seconds
    rmp_seconds = (300. if run_mode == 'probe' else 600.) if rmp_seconds is None else rmp_seconds
    for value in (job_wall_seconds, wall_seconds, rmp_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Scale budgets must be positive finite numbers')
    if not (rmp_seconds <= wall_seconds <= job_wall_seconds - 10. and job_wall_seconds <= job_cap):
        raise ValueError(f'{run_mode} budget requires RMP <= program <= job - 10s and job <= {job_cap:g}s')
    if type(max_completed_iterations) is not int or max_completed_iterations <= 0:
        raise ValueError('Completed iteration budget must be a positive integer')
    return dict(run_mode=run_mode, job_wall_seconds=float(job_wall_seconds),
        program_wall_seconds=float(wall_seconds), rmp_time_limit_seconds=float(rmp_seconds),
        finalization_reserve_seconds=10., max_completed_iterations=max_completed_iterations,
        iteration_limit_scope='this invocation, including a resumed pending iteration',
        checkpoint_kind='algorithm and completed LPs; no solver search-tree warm start')


def execution_boundary(checkpoint_path, *, resume, require_pending, max_completed_iterations):
    """Bind the stopping rule to the actual saved work before loading full data."""
    if require_pending and (not resume or max_completed_iterations != 1):
        raise ValueError('Pending evaluation requires --resume and --max-completed-iterations 1')
    record = dict(require_pending_evaluation=require_pending,
        stop_after_completed_iterations=max_completed_iterations, initial_completed_iterations=0,
        initial_pending_iteration=None, initial_partial_q_count=0, input_checkpoint=None)
    if not resume:
        return record
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError('Resume checkpoint missing')
    envelope = json.loads(path.read_text())
    state = envelope['state']
    if envelope['sha256'] != canonical_hash(state):
        raise ValueError('Checkpoint checksum mismatch')
    pending = state['pending']
    if require_pending and pending is None:
        raise ValueError('Pending evaluation requested but checkpoint has no pending iteration')
    record.update(initial_completed_iterations=len(state['trace']),
        initial_pending_iteration=None if pending is None else pending['iteration'],
        initial_partial_q_count=0 if pending is None else len(pending['q_usd']),
        input_checkpoint=dict(path=str(path.resolve()), sha256=sha256(path)))
    return record


def execution_job_reference(budget):
    """Check the declared job cap against the supervising job when present."""
    directory = os.environ.get('EVCS_JOB_DIRECTORY')
    if directory is None:
        return None
    path = Path(directory) / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['timeout_seconds'] != budget['job_wall_seconds']:
        raise ValueError('Scale declared job budget differs from supervising job timeout')
    return dict(path=str(path.resolve()), sha256=sha256(path),
        job_id=manifest['job_id'], timeout_seconds=manifest['timeout_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-root',type=Path,default=get_paths().data_root)
    p.add_argument('--s03-i03-acceptance',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-mode',choices=('probe','completion'),default='probe',
        help='probe: at most 600s job; completion: explicitly allow at most 7200s job')
    p.add_argument('--job-wall-seconds',type=float,
        help='supervisor timeout declaration; defaults to the selected mode cap')
    p.add_argument('--wall-seconds',type=float,help='program budget; defaults to job budget minus 10s')
    p.add_argument('--rmp-seconds',type=float,help='per-master cap; defaults to 300s probe / 600s completion')
    p.add_argument('--max-completed-iterations',type=int,default=1,
        help='complete at most this many C&CG iterations in this invocation (default: 1)')
    p.add_argument('--require-pending-evaluation',action='store_true',
        help='resume only an existing pending full-support evaluation, then stop before another master')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--master-backend',choices=('highs','gurobi'),default='highs')
    p.add_argument('--optimizer-threads',type=int,default=1)
    p.add_argument('--no-zero-presolve',action='store_true')
    args=p.parse_args();start=time.monotonic();paths=get_paths();out=args.output.resolve()
    budget=execution_budget(args.run_mode,wall_seconds=args.wall_seconds,rmp_seconds=args.rmp_seconds,
        job_wall_seconds=args.job_wall_seconds,max_completed_iterations=args.max_completed_iterations)
    boundary=execution_boundary(out/'checkpoint.json',resume=args.resume,
        require_pending=args.require_pending_evaluation,max_completed_iterations=args.max_completed_iterations)
    job_reference=execution_job_reference(budget)
    driver=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__))
    if sha256(args.profile)!=PROFILE_SHA:raise ValueError('Pinned original scale diagnostic changed')
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    sessions=out/'sessions';sessions.mkdir(exist_ok=True)
    session=sessions/f'{len(list(sessions.iterdir()))+1:04d}';session.mkdir()
    card=_physical_acceptance(args.s03_i03_acceptance,I03_SHA)
    for name,digest in card['reviews'].items():
        if sha256(args.s03_i03_acceptance.parent/name)!=digest:raise ValueError('I03 review changed')
    ref=card['source_refs']['s03_i01_acceptance'];_,inherited=prerequisites(ref['path'],ref['sha256'])
    execution=json.loads((args.profile.parent/'execution.json').read_text())
    artifacts=json.loads((args.profile.parent/'artifact_sha256.json').read_text())
    if sha256(args.profile.parent/'artifact_sha256.json')!='3387d9fe018b2271d0e1e8be048a1832774098f4bbc2aec5d9e2afd15a573592':
        raise ValueError('Profile artifact manifest changed')
    for name,digest in artifacts.items():
        if sha256(args.profile.parent/name)!=digest:raise ValueError('Profile evidence changed')
    taskref=execution['task_source'];bank=execution['bank'];rawtasks=read_ref(taskref)
    ledger=read_ref(bank['scenario_ledger'])
    if len(rawtasks['vehicle_summaries'])!=80 or bank['scenario_count']!=160:raise ValueError('Full fleet/support changed')
    ids,phi,lo,hi=target_inputs(inherited,TARGET,1.)
    if list(ids)!=bank['scenario_ids'] or [x['scenario_id'] for x in ledger['scenarios']]!=list(ids):
        raise ValueError('Full ordered support mismatch')
    case=load_physical_case(args.case_root);tasks=load_frozen_tasks(taskref['path'],case,taskref['sha256'])
    if len(tasks.task_ids)!=1008 or len(tasks.task_ids)!=bank['task_count']:raise ValueError('Full H changed')
    first=assemble_first_stage(case,tasks);scenes={};assembly=[]
    def scene(i):
        if i not in scenes:
            value=load_scenario_physical_inputs(bank['file']['path'],bank['scenario_ledger']['path'],int(i),case,tasks,
                expected_npz_sha256=bank['file']['sha256'],expected_ledger_sha256=bank['scenario_ledger']['sha256'],
                expected_scenario_sha256=ledger['scenarios'][i]['scenario_sha256'])
            if value.scenario_id!=ids[i]:raise ValueError('Loaded atom ID changed')
            if len(scenes)>=2:scenes.pop(next(iter(scenes)))
            scenes[i]=value
        return scenes[i]
    def load(i):
        if rss_bytes()>28*1024**3:raise MemoryError('Stop before another template exceeds campaign budget')
        began=time.monotonic();s=scene(i);loaded=time.monotonic();t=assemble_recourse(case,tasks,s)
        assembly.append(dict(index=i,scenario_id=ids[i],template_sha256=t.template_sha256,
            load_seconds=loaded-began,assembly_seconds=time.monotonic()-loaded,rss_bytes=rss_bytes()))
        d=out/'templates'/f'{i:03d}'
        if d.exists():
            ref=json.loads((d/'identity.json').read_text())
            if ref['template_sha256']!=t.template_sha256:raise ValueError('Saved template identity changed')
            for name,digest in json.loads((d/'hashes.json').read_text()).items():
                if sha256(d/name)!=digest:raise ValueError('Saved template evidence changed')
        else:
            d.mkdir(parents=True)
            for key in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'):save_npz(d/(key+'.npz'),getattr(t,key))
            np.savez_compressed(d/'arrays.npz',b_ub=t.b_ub,b_eq=t.b_eq,c=t.c,lower_bounds=t.lower_bounds,upper_bounds=t.upper_bounds)
            _json_file(d/'identity.json',dict(index=i,scenario_id=ids[i],scenario_sha256=t.scenario_sha256,template_sha256=t.template_sha256,H=len(t.task_ids)))
            _json_file(d/'hashes.json',{v.name:sha256(v) for v in sorted(d.iterdir())})
        _json_file(session/'assembly.json',assembly)
        return t
    binding={'npz':bank['file'],'ledger':bank['scenario_ledger'],'tasks':taskref,
        'physical_case_sha256':case.case_sha256,
        'source_manifest_sha256':sha256(paths.project_root/'RELEASE_MANIFEST.json')}
    cache=LazyRecourses(ids,[x['scenario_sha256'] for x in ledger['scenarios']],first.matrix_sha256,binding,load,capacity=2)
    problem=CCGProblem(first,cache,ProbabilitySet(ids,tuple(inherited['feature_order']),phi,lo,hi))
    events=[]
    def sink(event,iteration,**data):
        d=session/'events'/f'{len(events):06d}_{event}';d.mkdir(parents=True)
        meta=dict(event=event,iteration=iteration,path=str(d.relative_to(out)))
        if event=='rmp':
            model,result=data['model'],data['result'];_save_model(d,model,result)
            if result.solution is not None:
                plan=result.solution.plan;check=check_first_stage(case,tasks,plan);_json_file(d/'first_stage_check.json',check)
                if not check['passed']:raise ValueError('Original first-stage physical failure')
                for sid,y in zip(model.scenario_ids,result.solution.recourse_solutions):
                    i=ids.index(sid);_save_solution(d/f'witness_{i}.npz',y)
                    checked=check_recourse(case,tasks,scene(i),plan,y,check_objective_bound=False)
                    _json_file(d/f'physical_{i}.json',checked)
                    if not checked['passed']:raise ValueError('Original RMP recourse physical failure')
        elif event=='recourse':
            i,result,plan=data['index'],data['result'],data['plan'];meta['scenario_index']=i;meta['scenario_id']=ids[i]
            _json_file(d/'solver.json',result.as_dict())
            _json_file(d/'plan.json',dict(x=plan.x,r=plan.r,task_sha256=plan.task_sha256,config_hash=plan.config_hash))
            if result.solution is not None:
                _save_solution(d/'solution.npz',result.solution)
                checked=check_recourse(case,tasks,scene(i),plan,result.solution,check_objective_bound=result.optimal)
                _json_file(d/'physical_check.json',checked)
                if not checked['passed']:raise ValueError('Fixed-plan recourse physical failure')
        elif event in ('probability','phase1'):_json_file(d/'solver.json',data['result'].as_dict())
        else:_json_file(d/'trace.json',data['record'])
        events.append(meta);_json_file(session/'events.json',events)
        _json_file(out/'progress.json',dict(phase=event,iteration=iteration,event=meta,
            elapsed_seconds=time.monotonic()-start,rss_bytes=rss_bytes(),cache=cache.statistics()))
    _json_file(session/'execution.json',dict(paths=paths.as_dict(),python=sys.version,argv=sys.argv,
        execution_driver=driver,execution_budget=budget,execution_boundary=boundary,supervising_job=job_reference,
        binding=binding,config_sha256=sha256(os.environ['EVCS_CONFIG']),profile_sha256=PROFILE_SHA,
        I03_acceptance_sha256=I03_SHA,H=1008,base_fleet=80,support=160,target=TARGET,rho=1,
        zero_presolve=False if args.no_zero_presolve else 'exact_zero_singleton_v1',
        master_backend=args.master_backend,optimizer_threads=args.optimizer_threads,
        native_start='checked_full_shedding_hint_v1',resume=args.resume,OOS=False,full_S03_pass=False))
    left=budget['program_wall_seconds']-(time.monotonic()-start)-budget['finalization_reserve_seconds']
    if left<=0:raise TimeoutError('Prerequisites consumed scale execution budget')
    result=run_ccg(problem,checkpoint_path=out/'checkpoint.json',resume=args.resume,wall_seconds=left,
        rmp_time_limit_seconds=budget['rmp_time_limit_seconds'],max_iterations=budget['max_completed_iterations'],
        oracle_time_limit_seconds=30,zero_presolve=not args.no_zero_presolve,evidence_sink=sink,
        native_start=True,solver_log_directory=session/'solver_logs',
        master_backend=args.master_backend,optimizer_threads=args.optimizer_threads)
    _json_file(session/'checkpoint_snapshot.json',json.loads((out/'checkpoint.json').read_text()))
    _json_file(session/'result.json',result.as_dict())
    report=dict(status=result.status,lower_bound_usd=result.state['lower_bound_usd'],upper_bound_usd=result.state['upper_bound_usd'],
        execution_driver=driver,execution_budget=budget,execution_boundary=boundary,supervising_job=job_reference,
        completed_iterations_this_session=len(result.state['trace'])-boundary['initial_completed_iterations'],
        full_iterations=len(result.state['trace']),partial_q_count=0 if result.state['pending'] is None else len(result.state['pending']['q_usd']),
        full_support=160,H=1008,base_fleet=80,wall_seconds=time.monotonic()-start,
        cumulative_algorithm_seconds=result.state['elapsed_seconds'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        recovered_failure_overhead_seconds=result.state.get('checkpoint_migration',{}).get('additional_wall_seconds',0.),
        cumulative_accounted_seconds=result.state['elapsed_seconds']+result.state.get('checkpoint_migration',{}).get('additional_wall_seconds',0.),
        previous_profile_wall_seconds=json.loads(args.profile.read_text())['total_wall_seconds'],cache=cache.statistics(),
        full_S03_pass=False,independent_scientific_audit_required=True)
    _json_file(session/'summary.json',report);_json_file(session/'model_code_manifest.json',source_provenance())
    # Each completed session binds its immutable checkpoint snapshot; mutable
    # campaign checkpoint/progress must not invalidate an earlier session.
    files = list(session.rglob('*')) + list((out/'templates').rglob('*'))
    _json_file(session/'artifact_sha256.json',{str(f.relative_to(out)):sha256(f) for f in sorted(files)
        if f.is_file() and f!=session/'artifact_sha256.json'})
    print(json.dumps({k:v for k,v in report.items() if k!='cache'}))
    return 0 if result.status in ('CONVERGED','TIME_LIMIT','ITERATION_LIMIT') else 2

if __name__=='__main__':raise SystemExit(main())
