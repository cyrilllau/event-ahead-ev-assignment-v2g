"""Common v54-Q1 metrics for an unchanged first-stage plan and physical dispatch.

This module does not select tasks, alter routes, or open an OOS bank. Missing
ratios/times are JSON null with a reason. Costs retain USD, power kW, AC energy
kWh, and battery throughput DC kWh as separate quantities.
"""
from __future__ import annotations

import json
import numpy as np

from .first_stage import FirstStagePlan
from .random_streams import canonical_hash
from .recourse import assemble_recourse, solve_fixed_plan
from .recourse_check import check_recourse

METRIC_CONTRACT = {
    'version': 'S04-I01-v1', 'model': 'ieee33-v54-q1',
    'early_hours': 3., 'bridge_threshold': .90, 'reserve_detection_kw': 1e-5,
    'reserve_detection_source': 'existing original-unit power residual tolerance; reporting only',
    'zero_denominator': 'null with reason; no zero imputation',
    'bridge_zero_demand': 'null if undefined CLSR is encountered before the first below-threshold period',
    'arrival_soc_population': 'selected route-completed tasks; failed-route share reported separately',
    'social_cost': 'Fshed+Fshort+Fdeg; shortfall retained as a proxy consequence',
    'selection_reporting': 'membership x > 0.5 after original integrality validation; stored plan unchanged',
    'critical_load_aggregation': 'ratio of sums of critical energy; not mean of period ratios',
}


def plan_identity(plan):
    return canonical_hash(dict(x=np.asarray(plan.x).tolist(), r=np.asarray(plan.r).tolist(),
        task_sha256=plan.task_sha256, config_hash=plan.config_hash))


def _ratio(numerator, denominator, name, reasons):
    if denominator == 0:
        reasons[name] = 'zero denominator'
        return None
    return float(numerator / denominator)


def response_times(certified_kw, critical_load_kw, critical_shed_kw, *, dt_h=1.,
                   reserve_detection_kw=1e-5, bridge_threshold=.90):
    """Hand-checkable time-series aggregation independent of LP assembly."""
    cert, load, shed = [np.asarray(v, dtype=float) for v in
                        (certified_kw, critical_load_kw, critical_shed_kw)]
    if (cert.ndim != 1 or not len(cert) or cert.shape != load.shape or load.shape != shed.shape
            or not all(np.isfinite(v).all() for v in (cert, load, shed))
            or not np.isfinite(dt_h) or dt_h <= 0
            or not np.isfinite(reserve_detection_kw) or reserve_detection_kw < 0
            or not np.isfinite(bridge_threshold) or not 0 <= bridge_threshold <= 1):
        raise ValueError('Invalid response time-series axes or reporting thresholds')
    if np.any(load < 0) or np.any(shed < -1e-5) or np.any(shed-load > 1e-5) or np.any(cert < -1e-5):
        raise ValueError('Invalid reserve or critical demand/shedding time series')
    reasons = {}; starts = np.arange(len(cert))*dt_h
    early_weights = np.maximum(0., np.minimum(dt_h, 3.-starts))
    clsr_t = [_ratio(l-s, l, f'CLSR_t[{i}]', reasons) for i,(l,s) in enumerate(zip(load,shed))]
    clsr = _ratio(float(np.sum(load-shed)*dt_h), float(np.sum(load)*dt_h), 'CLSR', reasons)
    early_clsr = _ratio(float((load-shed)@early_weights),float(load@early_weights),'early_CLSR',reasons)
    first_indices = np.flatnonzero(cert > reserve_detection_kw)
    first = None if not len(first_indices) else float(starts[first_indices[0]])
    if first is None: reasons['first_reserve_h'] = 'no certified reserve above reporting threshold'
    bridge = 0.
    for value in clsr_t:
        if value is None:
            bridge = None; reasons['bridge_h'] = 'undefined critical-load ratio before prefix termination'; break
        if value < bridge_threshold: break
        bridge += dt_h
    return dict(CLSR=clsr, early_CLSR=early_clsr, CLSR_t=clsr_t,
        first_reserve_h=first, no_reserve=first is None, bridge_h=bridge,
        early_certified_kwh=float(cert@early_weights), na_reasons=reasons)


def metrics_from_solution(case, tasks, scene, plan, solution):
    """Reconstruct metrics after direct physical checks; never claim optimality."""
    before = plan_identity(plan)
    check = check_recourse(case,tasks,scene,plan,solution)
    if not check['passed']: raise ValueError('Common evaluator requires an original-physics feasible solution')
    t = len(case.periods); h = len(tasks.task_ids)
    if tuple(case.periods) != tuple(range(1,t+1)):
        raise ValueError('Response metrics require consecutive periods starting at one')
    x, reserve = np.asarray(plan.x),np.asarray(plan.r); dt=float(case.dt_h)
    gamma = np.asarray(scene.task_gamma_route); qualified = np.asarray(scene.task_varsigma)
    arrival = np.asarray(scene.task_arrival_h,dtype=float)
    if (gamma.shape != (h,) or qualified.shape != (h,) or arrival.shape != (h,)
            or not np.all((gamma==0)|(gamma==1)) or not np.all((qualified==0)|(qualified==1))
            or np.any(np.isnan(arrival)) or np.any(np.isneginf(arrival))
            or np.any(qualified.astype(bool) & ~gamma.astype(bool))):
        raise ValueError('Invalid task route/qualification/arrival axes')
    nameplate=np.sum(np.asarray(tasks.p_dis_kw)[:,None]*np.asarray(tasks.chi_res)*x[:,None],axis=0)
    procured=np.sum(reserve,axis=0); certified=np.sum(solution.rav,axis=0); dispatched=np.sum(solution.p,axis=0)
    if np.max(np.stack((-dispatched,dispatched-certified,certified-procured,procured-nameplate)),initial=0)>1e-5:
        raise ValueError('Capacity funnel violates original physical tolerance')
    critical=np.asarray(case.critical_buses,dtype=bool)
    if critical.shape!=(len(case.bus_ids),):raise ValueError('Critical bus axis mismatch')
    response=response_times(certified,np.sum(scene.load_p_kw[:,critical],axis=1),
                            np.sum(solution.shed[:,critical],axis=1),dt_h=dt)
    reasons=response.pop('na_reasons'); selected=x>.5; count=int(selected.sum())
    metrics=dict(response,DRR=_ratio(float(certified.sum()*dt),float(procured.sum()*dt),'DRR',reasons),
        RCR=_ratio(float(selected@gamma.astype(float)),count,'RCR',reasons),
        OTR=_ratio(float(np.sum(selected*qualified*(arrival <= (np.asarray(tasks.start_periods)-1)*dt))),count,'OTR',reasons),
        MBQR=_ratio(float(selected@qualified.astype(float)),count,'MBQR',reasons),
        selected_task_count=int(np.count_nonzero(x>.5)),
        selected_vehicle_count=len(set(np.asarray(tasks.vehicle_ids)[x>.5].tolist())),
        no_selected_tasks=bool(count==0),zero_procurement=bool(reserve.sum()==0),
        shortfall_kwh=float(np.sum(solution.short)*dt),v2g_ac_kwh=float(np.sum(solution.p)*dt),
        battery_throughput_dc_kwh=float(np.sum(solution.p/np.asarray(tasks.eta_dis)[:,None])*dt))
    completed=selected*gamma
    metrics['route_completed_selected_count']=int(completed.sum())
    metrics['route_failed_selected_count']=count-int(completed.sum())
    metrics['mean_arrival_soc_completed_selected']=_ratio(float(completed@scene.task_arrival_soc),float(completed.sum()),'mean_arrival_soc_completed_selected',reasons)
    op=check['costs'];first=check['first_stage']['costs']
    costs=dict(Fact_usd=first['activation_usd'],Fcap_usd=first['reservation_usd'],
        Fshed_usd=op['Fshed'],Fshort_usd=op['Fshort'],FV2G_usd=op['FV2G'],Fdeg_usd=op['Fdeg'])
    total=sum(costs.values());social=op['Fshed']+op['Fshort']+op['Fdeg']
    costs.update(total_DSO_usd=total,social_proxy_usd=social)
    after=plan_identity(plan)
    if before != after:raise ValueError('First-stage plan changed during evaluation')
    report=dict(schema_version=1,status='PHYSICALLY_FEASIBLE_METRICS',optimality_claim=False,
        sample_id=scene.scenario_id,scenario_sha256=scene.scenario_sha256,
        task_sha256=tasks.task_sha256,config_hash=tasks.config_hash,case_sha256=case.case_sha256,
        plan_sha256_before=before,plan_sha256_after=after,metric_contract=dict(METRIC_CONTRACT),
        metrics=metrics,costs=costs,na_reasons=reasons,
        timeseries_kw=dict(nameplate=nameplate.tolist(),procured=procured.tolist(),
            certified=certified.tolist(),dispatched=dispatched.tolist()),
        energy_kwh={name:float(np.sum(v)*dt) for name,v in
            (('nameplate',nameplate),('procured',procured),('certified',certified),('dispatched',dispatched))},
        physical_check=check)
    json.dumps(report,allow_nan=False)
    return report


def evaluate_development_plan(case,tasks,scene,plan,*,time_limit_seconds=30):
    """Fixed-plan physical LP foundation; formal OOS still requires S04 locking."""
    if not getattr(scene,'development',False):
        raise ValueError('Formal OOS requires the separately accepted plan-bank locking gateway')
    locked=FirstStagePlan(plan.x,plan.r,plan.task_sha256,plan.config_hash)
    before=plan_identity(plan)
    template=assemble_recourse(case,tasks,scene)
    result=solve_fixed_plan(template,locked,time_limit_seconds=time_limit_seconds)
    if (not result.optimal or result.solution is None or
            not result.certificate.get('optimal_certificate_valid',False)):
        raise RuntimeError('No certified fixed-plan recourse value: '+result.status)
    report=metrics_from_solution(case,tasks,scene,locked,result.solution)
    if plan_identity(plan)!=before or report['plan_sha256_before']!=before:
        raise ValueError('Original plan changed during fixed-plan evaluation')
    report.update(status='CERTIFIED_DEVELOPMENT_FIXED_PLAN',optimality_claim=True,
                  solver=result.as_dict(),template_sha256=template.template_sha256,formal_OOS=False)
    return report,result.solution


def plan_jaccard(left_ids,left_plan,right_ids,right_plan):
    sets=[]
    for ids,plan in ((left_ids,left_plan),(right_ids,right_plan)):
        ids=tuple(ids);x=np.asarray(plan.x)
        if len(set(ids))!=len(ids) or len(ids)!=len(x) or not np.isfinite(x).all() or np.max(abs(x-np.rint(x)),initial=0)>1e-8 or np.any(x < -1e-8) or np.any(x > 1+1e-8):
            raise ValueError('Invalid task identities or selection for Jaccard')
        sets.append({task for task,selected in zip(ids,x) if selected>.5})
    union=sets[0]|sets[1]
    return dict(jaccard=1. if not union else len(sets[0]&sets[1])/len(union),
                both_empty=not union,left_count=len(sets[0]),right_count=len(sets[1]))
