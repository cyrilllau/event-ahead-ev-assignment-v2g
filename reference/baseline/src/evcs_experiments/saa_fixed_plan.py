"""Serial SAA fixed-plan LP evidence preparation, without formal acceptance.

Checkpoints retain raw primal/dual/RHS and every attempt, including limits.
The caller's queue receipt remains authoritative for post-return accounting.
No adversarial probability or CCG state is constructed.
"""
from pathlib import Path
import json
import math
import time

import numpy as np

from .campaign_batch import file_reference, verify_reference
from .first_stage_check import check_first_stage
from .random_streams import canonical_hash
from . import recourse, recourse_check, first_stage_check
from .recourse import solve_fixed_plan
from .recourse_check import check_recourse
from .runtime.jobctl import atomic_json


def _need(ok, message):
    if not ok: raise ValueError(message)


def _serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, 'items'): return {k:_serial(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_serial(v) for v in value]
    return value


def _read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def _new(path, value):
    with Path(path).open('x') as stream:
        json.dump(_serial(value),stream,indent=2,allow_nan=False);stream.write('\n')
    return file_reference(path)


def complete_saa_fixed_plan_q(*, plan, case, tasks, scenario_ids, scenario_sha256s,
        template_sha256s, scenario_cost_constants_usd, template_loader, scene_loader,
        source_references, input_binding, output_directory, deadline,
        cumulative_budget_seconds, prior_accounted_seconds=0., resume_checkpoint_reference=None,
        per_lp_seconds=60., export_reserve_seconds=2., domain='development'):
    """Prepare complete Q in original order, restoring only pinned completed LPs.

    Every invocation is charged from its actual monotonic start. On resume,
    prior_accounted_seconds must cover the saved provisional cumulative time;
    this permits a larger immutable queue receipt, never a smaller/reset balance.
    A first call may carry earlier master/assembly charges in that prior balance.
    Formal source/whole-record numerical admission is explicitly still external.
    """
    began=time.monotonic()
    _need(domain in ('development','formal'),'Explicit domain required')
    for name,v in [('deadline',deadline),('budget',cumulative_budget_seconds),
                   ('prior',prior_accounted_seconds),('LP limit',per_lp_seconds),('reserve',export_reserve_seconds)]:
        _need(type(v) in (int,float) and math.isfinite(v) and v>=0,'Invalid '+name)
    _need(0<cumulative_budget_seconds<=7200 and 0<per_lp_seconds<=60 and export_reserve_seconds>0,
          'Invalid frozen time controls')
    _need(prior_accounted_seconds<cumulative_budget_seconds and deadline>began,'No remaining budget')
    stop=min(deadline,began+cumulative_budget_seconds-prior_accounted_seconds)
    ids=list(scenario_ids); hashes=list(scenario_sha256s);templates=list(template_sha256s)
    constants=np.asarray(scenario_cost_constants_usd,dtype=float)
    _need(len(ids)>0 and len(ids)==len(hashes)==len(templates)==len(constants) and
          len(set(ids))==len(ids) and constants.shape==(len(ids),) and np.isfinite(constants).all(),
          'Invalid ordered support/constants')
    _need(domain!='formal' or len(ids)==160,'Formal SAA requires160 scenarios')
    _need(source_references and isinstance(input_binding,dict) and input_binding,'Explicit source/input binding required')
    for ref in source_references:verify_reference(ref)
    for module_path in (__file__,recourse.__file__,recourse_check.__file__,first_stage_check.__file__):
        _need(file_reference(module_path) in source_references,'Fixed-Q implementation/checkers must be source-bound')
    plan_dict=_serial(dict(x=plan.x,r=plan.r,task_sha256=plan.task_sha256,config_hash=plan.config_hash))
    _need(check_first_stage(case,tasks,plan)['passed'],'Invalid fixed first-stage plan')
    identity=dict(kind='SAA_SERIAL_FIXED_PLAN_Q_V1',domain=domain,plan=plan_dict,
        plan_sha256=canonical_hash(plan_dict),case_sha256=case.case_sha256,
        task_sha256=tasks.task_sha256,scenario_ids=ids,scenario_sha256s=hashes,
        template_sha256s=templates,constants_usd=constants.tolist(),
        sources=source_references,input_binding=input_binding,cumulative_budget_seconds=cumulative_budget_seconds,
        per_lp_seconds=per_lp_seconds,export_reserve_seconds=export_reserve_seconds)
    root=Path(output_directory).resolve();checkpoint=root/'checkpoint.json'
    if resume_checkpoint_reference is None:
        root.mkdir(parents=True,exist_ok=False);request_ref=_new(root/'request.json',identity)
        rows=[];completed={}
    else:
        previous=_read(resume_checkpoint_reference)
        _need(file_reference(checkpoint)['sha256']==resume_checkpoint_reference['sha256'],
              'Stale checkpoint cannot reset history')
        request_ref=previous['request_reference'];_need(request_ref==file_reference(root/'request.json') and
            _read(request_ref)==identity,'Fixed-Q request/source/plan changed')
        _need(prior_accounted_seconds>=previous['cumulative_provisional_seconds'],'Prior budget reset')
        rows=previous['attempts'];completed={}
        _need({str(p) for p in root.glob('atom_*/record.json')}=={r['path'] for r in rows},'Attempt inventory changed')
        _need({str(p) for p in root.glob('atom_*')}=={str(Path(r['path']).parent) for r in rows},'Partial attempt requires diagnosis')
        for ref in rows:
            row=_read(ref)
            _need(row['request_reference']==request_ref,'Foreign attempt request')
            for artifact in row['artifacts']:verify_reference(artifact)
            if row['completed']:
                i=row['scenario_index'];_need(i not in completed,'Duplicate completed Q')
                _need(row['plan_sha256']==identity['plan_sha256'] and row['scenario_id']==ids[i] and
                      row['scenario_sha256']==hashes[i] and row['template_sha256']==templates[i], 'Foreign saved Q')
                completed[i]=row['q_usd']
        _need(previous['status']!='FAILED','Failed evidence requires diagnosis')
    def save(status):
        elapsed=time.monotonic()-began
        value=dict(status=status,request_reference=request_ref,attempts=rows,
            completed_indices=sorted(completed),q_usd=[completed.get(i) for i in range(len(ids))],
            attempt_provisional_seconds=elapsed,prior_accounted_seconds=prior_accounted_seconds,
            cumulative_provisional_seconds=prior_accounted_seconds+elapsed,
            queue_receipt_still_required=True,full_support_verified=False,
            independent_numerical_audit_required=True,formal_OOS=False,scientific_spec_pass=False)
        atomic_json(checkpoint,value)
        snapshots=root/'checkpoints';snapshots.mkdir(exist_ok=True)
        snapshot=snapshots/f"checkpoint_{len(list(snapshots.glob('checkpoint_*.json'))):06d}.json"
        with snapshot.open('xb') as stream:stream.write(checkpoint.read_bytes())
        return dict(value,checkpoint_reference=file_reference(snapshot))
    save('RUNNING')
    try:
        for i,sid in enumerate(ids):
            if i in completed:continue
            if stop-time.monotonic()<=export_reserve_seconds:return save('TIME_LIMIT')
            template=template_loader(i);scene=scene_loader(i)
            _need(template.template_sha256==templates[i] and template.scenario_sha256==hashes[i] and
                scene.scenario_sha256==hashes[i] and scene.scenario_id==sid and
                scene.physical_case_sha256==case.case_sha256 and
                scene.development is (domain=='development'),'Template/scene physical identity changed')
            remaining=stop-time.monotonic()-export_reserve_seconds
            if remaining<=0:return save('TIME_LIMIT')
            result=solve_fixed_plan(template,plan,time_limit_seconds=min(per_lp_seconds,remaining))
            folder=root/f'atom_{len(rows):05d}';folder.mkdir()
            artifacts=[_new(folder/'solver.json',result.as_dict()),_new(folder/'plan.json',plan_dict)]
            bub,beq=template.fixed_rhs(plan);np.savez_compressed(folder/'fixed_rhs.npz',b_ub=bub,b_eq=beq)
            artifacts.append(file_reference(folder/'fixed_rhs.npz'))
            physical=None
            if result.solution is not None:
                vector=template.pack(result.solution);np.save(folder/'primal.npy',vector,allow_pickle=False)
                artifacts.append(file_reference(folder/'primal.npy'))
                physical=check_recourse(case,tasks,scene,plan,result.solution,check_objective_bound=result.optimal)
                artifacts.append(_new(folder/'physical_check.json',physical))
            cert=result.certificate
            valid=bool(result.status=='OPTIMAL' and result.solver_status==0 and result.optimal and cert.get('optimal_certificate_valid') and result.solution is not None
                       and physical['passed'] and result.objective_usd is not None and np.isfinite(result.objective_usd))
            if valid:
                dual={key:np.asarray(cert['dual_'+key+'_marginals']) for key in ('inequality','equality','lower','upper')}
                np.savez_compressed(folder/'dual_kkt.npz',**dual)
                artifacts.append(file_reference(folder/'dual_kkt.npz'))
            row=dict(request_reference=request_ref,scenario_index=i,scenario_id=sid,
                scenario_sha256=hashes[i],template_sha256=templates[i],plan_sha256=identity['plan_sha256'],
                completed=valid,q_usd=float(result.objective_usd+constants[i]) if valid else None,
                native_status=result.status,artifacts=artifacts,independent_numerical_audit_required=True)
            rows.append(_new(folder/'record.json',row))
            if physical is not None and not physical['passed']:
                raise ValueError('Invalid physical recourse witness')
            if valid:completed[i]=row['q_usd']
            elif not (result.status=='LIMIT_REACHED' and result.solver_status==1 and not result.optimal):
                raise ValueError('Invalid/nonoptimal recourse certificate')
            if time.monotonic()>stop:raise TimeoutError('Fixed-Q shared deadline exceeded during export')
            save('RUNNING')
            if not valid:return save('TIME_LIMIT')
        return save('COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT')
    except Exception:
        save('FAILED')
        raise
