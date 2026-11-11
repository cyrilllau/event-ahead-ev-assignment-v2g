"""Locked fixed-plan evaluation; scenario factories run only after access checks.

A factory is trusted, frozen evaluator code. Its input files must be bound by the
protocol/plan references. It returns the complete physical Q1 case, tasks and
scene; no planning ablation is applied in this evaluator.
"""
from contextlib import nullcontext
import csv
import inspect
import json
from pathlib import Path
import time
import numpy as np
from .evaluation_lock import begin_evaluation, verify_plan_batch
from .evaluation_metrics import metrics_from_solution, plan_identity
from .first_stage import FirstStagePlan
from .recourse import assemble_recourse, solve_fixed_plan
from .evidence import sha256
from .validation.s02_toys import _save_solution,_serial


def read_locked_plan(record):
    ref=record['references']['plan'];path=Path(ref['path'])
    if sha256(path)!=ref['sha256']:raise ValueError('Plan bytes changed')
    p=json.loads(path.read_text());plan=FirstStagePlan(p['x'],p['r'],p['task_sha256'],p['config_hash'])
    if plan_identity(plan)!=record['plan_sha256']:raise ValueError('Plan array semantic hash differs')
    return plan,p


def _write(path,value):
    with Path(path).open('x') as f:json.dump(_serial(value),f,allow_nan=False,indent=2);f.write('\n')


def evaluate_locked_batch(lock_dir,batch_id,scenario_factory,output,*,domain,sample_ids_by_target,
                          s04_acceptance_ref=None,lp_time_limit_seconds=30,wall_seconds=50):
    """One bounded serial batch, with per-item certificates and stop-on-failure.

    The factory receives (plan_record, sample_id, saved_plan_metadata). DSP is
    checked against an explicit EA source before any scenario is accessed.
    Fixed-plan LP timeouts retain their solver record and stop this batch.
    """
    if not (0 < lp_time_limit_seconds <= wall_seconds <= 7190):raise ValueError('Invalid shared evaluation budget')
    verified=verify_plan_batch(lock_dir,batch_id);records=verified['batch']['plans']
    if domain=='formal' and verified['batch_spec'].get('oos_enabled') is False:raise ValueError('Algorithm-only group may not open OOS')
    if domain=='formal':
        cfg=verified['protocol'].get('configuration',{})
        eb=cfg.get('evaluation_budget',{});pb=cfg.get('planning_budget',{})
        if lp_time_limit_seconds!=eb.get('lp_time_limit_seconds') or wall_seconds!=eb.get('job_wall_seconds',0)-eb.get('shutdown_reserve_seconds',0):raise ValueError('Formal evaluation budget differs from frozen protocol')
        for record in records:
            budget=json.loads(Path(record['references']['budget']['path']).read_text())
            numeric=json.loads(Path(record['references']['solver_record']['path']).read_text())
            if budget.get('planning_budget')!=pb:raise ValueError('Plan planning budget differs from common frozen budget')
            actual=numeric.get('cumulative_wall_seconds')
            if isinstance(actual,bool) or not isinstance(actual,(float,int)) or not np.isfinite(actual) or not 0<=actual<=pb.get('cumulative_wall_seconds',0):raise ValueError('Invalid or excessive cumulative plan time')
            if numeric.get('domain')!='formal' or numeric.get('plan_sha256')!=record['plan_sha256'] or numeric.get('full_support_verified') is not True or numeric.get('status') not in ('OPTIMAL','CONVERGED','TIME_LIMIT'):raise ValueError('Formal plan requires its original full-support numerical record')
            lb,ub=numeric.get('lower_bound_usd'),numeric.get('upper_bound_usd')
            if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) for v in (lb,ub)) or lb>ub+1e-4:raise ValueError('Formal plan lacks valid finite bound ordering')
    factory_source=inspect.getsourcefile(scenario_factory)
    allowed=verified['protocol']['references']['evaluator']+verified['protocol']['references']['source']
    if factory_source is None or not any(Path(v['path']).resolve()==Path(factory_source).resolve() and v['sha256']==sha256(factory_source) for v in allowed):raise ValueError('Scenario factory source is not frozen by the evaluator protocol')
    plans={p['plan_id']:read_locked_plan(p) for p in records}
    for record in records:
        plan,meta=plans[record['plan_id']]
        if record['method_id'] in ('DSP','DSP_from_EA') and meta.get('execution_mode')!='DSP':raise ValueError('DSP method requires DSP execution mode')
        if meta.get('execution_mode')=='DSP':
            source=meta.get('ea_source_plan_id')
            if source not in plans:raise ValueError('DSP requires its locked EA source plan')
            other=next(p for p in records if p['plan_id']==source)
            if other['target_id']!=record['target_id'] or plan_identity(plans[source][0])!=plan_identity(plan):raise ValueError('DSP changed EA plan/target')
            for key in ('tasks','routes','initial_state'):
                if record['references'][key]['sha256']!=other['references'][key]['sha256']:raise ValueError('DSP changed EA '+key)
            if meta.get('departure_h')!=0 or plans[source][1].get('execution_mode')!='EA':raise ValueError('Invalid DSP departure/source')
    begin_evaluation(lock_dir,batch_id,domain=domain,sample_ids_by_target=sample_ids_by_target,s04_acceptance_ref=s04_acceptance_ref)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();rows=[]
    _write(out/'lock_binding.json',{k:v for k,v in verified.items() if k.endswith('_reference')})
    for record in records:
        plan,meta=plans[record['plan_id']]
        for sid in sample_ids_by_target[record['target_id']]:
            remaining=wall_seconds-(time.monotonic()-started)
            if remaining<=1:
                _write(out/'checkpoint.json',dict(status='TIME_LIMIT',completed_items=len(rows),next_plan_id=record['plan_id'],next_sample_id=sid,wall_seconds=time.monotonic()-started));return dict(status='TIME_LIMIT',completed_items=len(rows))
            folder=out/f'item_{len(rows):06d}';folder.mkdir();before=plan_identity(plan)
            try:
                verify_plan_batch(lock_dir,batch_id)
                if domain=='formal':
                    from .evaluation_access import locked_oos_access
                    context=locked_oos_access(lock_dir,batch_id,record['target_id'],sid,s04_acceptance_ref)
                else:context=nullcontext()
                with context:case,tasks,scene=scenario_factory(record,sid,meta)
                if scene.metadata.get('departure_h')!=meta.get('departure_h'):raise ValueError('Physical replay changed frozen departure')
                if scene.scenario_id!=sid or scene.target_event!=record['target_id']:raise ValueError('Factory sample/target mismatch')
                if bool(scene.development)!=(domain=='development'):raise ValueError('Factory changed formal/development namespace')
                if case.model_revision!='v54-q1' or case.capability_active_coordinate!='accredited_reserve_r_av':raise ValueError('Common evaluation requires real Q1 physics')
                if tasks.task_sha256!=plan.task_sha256 or tasks.config_hash!=plan.config_hash:raise ValueError('Factory changed H/config')
                _write(folder/'physical_metadata.json',dict(scenario_id=scene.scenario_id,scenario_sha256=scene.scenario_sha256,task_sha256=scene.task_sha256,config_hash=scene.config_hash,metadata=dict(scene.metadata)))
                fields=('task_arrival_h','task_arrival_soc','task_gamma_route','task_varsigma','task_mu_full','task_usable_energy_dc_kwh','load_p_kw','load_q_kvar','line_operating_outages')
                np.savez_compressed(folder/'physical_inputs.npz',**{key:np.asarray(getattr(scene,key)) for key in fields})
                template=assemble_recourse(case,tasks,scene)
                remaining=wall_seconds-(time.monotonic()-started)
                if remaining<=1:raise TimeoutError('Shared budget exhausted during physical generation/assembly')
                solved=solve_fixed_plan(template,plan,time_limit_seconds=min(lp_time_limit_seconds,remaining-1))
                _write(folder/'solver.json',solved.as_dict())
                if solved.solution is not None:_save_solution(folder/'solution.npz',solved.solution)
                if not solved.optimal or not solved.certificate.get('optimal_certificate_valid',False):raise RuntimeError('Uncertified fixed-plan LP: '+solved.status)
                report=metrics_from_solution(case,tasks,scene,plan,solved.solution)
                if before!=plan_identity(plan) or report['plan_sha256_after']!=before:raise ValueError('Plan changed during batch')
                verify_plan_batch(lock_dir,batch_id)
                report.update(plan_id=record['plan_id'],method_id=record['method_id'],target_id=record['target_id'],formal_OOS=domain=='formal',optimality_claim=True,template_sha256=template.template_sha256)
                _write(folder/'metrics.json',report)
                rows.append(dict(plan_id=record['plan_id'],method_id=record['method_id'],target_id=record['target_id'],sample_id=sid,plan_sha256=before,**report['costs'],**{k:v for k,v in report['metrics'].items() if not isinstance(v,list)},metrics_path=str(folder/'metrics.json')))
                _write(folder/'artifact_sha256.json',{p.name:sha256(p) for p in folder.iterdir() if p.is_file()})
            except Exception as exc:
                _write(folder/'failure.json',dict(status='FAILED_VALIDITY_OR_CERTIFICATE',error_type=type(exc).__name__,error=str(exc),plan_id=record['plan_id'],sample_id=sid,completed_items=len(rows)))
                raise
    with (out/'metrics_table.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='PASS_LOCKED_FIXED_PLAN_BATCH',formal_OOS=domain=='formal',item_count=len(rows),wall_seconds=time.monotonic()-started,first_stage_optimization_calls=0)
    _write(out/'summary.json',result);_write(out/'artifact_sha256.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return result
