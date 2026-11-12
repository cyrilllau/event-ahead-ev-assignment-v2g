"""Full-support numerical audit adapter for a selected new raw plan/iteration.

Uses the independently reviewed original-array KKT/physical kernel. This
component does not audit a master bound or probability optimization and cannot
alone accept a formal planning record or unlock OOS.
"""
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import load_npz

from .first_stage_check import check_first_stage
from .random_streams import canonical_hash
from .raw_template_audit import audit_template_envelope, document, need, read, record


def _module(reference, name):
    path=read(reference)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def audit_fixed_plan_recourses(*, snapshot_reference, expected_request_reference,
        plan_dict, iteration, case, tasks, scene_loader,
        helper_source_reference, numerical_kernel_source_reference, deadline=None):
    """Audit every support atom of exactly one supplied plan and iteration.

    Caller must bind the kernel/helper references to independently reviewed
    source and the scene loader to accepted physical inputs before formal use.
    Supplied case/tasks/scene values are checked against the raw request and
    original template identity; the scope intentionally excludes source-card
    issuance, probability, master bounds and cumulative-budget settlement.
    """
    need(type(iteration) is int and iteration>=1,'Explicit positive iteration required')
    started=time.monotonic()
    if deadline is not None:
        need(np.isfinite(deadline) and started<deadline<=started+7190, 'Invalid shared audit deadline')
    def remaining():
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Shared numerical audit deadline reached')
    remaining()
    envelope=audit_template_envelope(snapshot_reference,expected_request_reference=expected_request_reference)
    request=document(expected_request_reference);identity=request['problem_identity'];ids=identity['scenario_ids']
    h=_module(helper_source_reference,'_generic_original_kkt_helper')
    kernel=_module(numerical_kernel_source_reference,'_generic_physical_audit_kernel')
    need(helper_source_reference['sha256']==kernel.HELPER_SHA256,'Unreviewed numerical helper')
    plan_sha=canonical_hash(plan_dict)
    plan=SimpleNamespace(x=np.asarray(plan_dict['x']),r=np.asarray(plan_dict['r']),
        task_sha256=plan_dict['task_sha256'],config_hash=plan_dict['config_hash'])
    need(plan.task_sha256==tasks.task_sha256 and plan.config_hash==tasks.config_hash and
         case.case_sha256==tasks.physical_case_sha256,'Supplied physical plan context differs')
    physical=check_first_stage(case,tasks,plan);need(physical['passed'],'First-stage physical check failed')
    snapshot=document(snapshot_reference);first_record=record(snapshot['first_stage'])
    first_refs={Path(r['path']).name:r for r in first_record['artifacts']}
    fm=document(first_refs['identity.json']);fv=h.npz(read(first_refs['arrays.npz']));A=load_npz(read(first_refs['matrix.npz'])).tocsr()
    first_sha=kernel.first_digest(fm,A,fv)
    need(first_sha==fm['matrix_sha256']==identity['first_stage_sha256'],'First-stage semantic identity differs')
    need(fm['task_sha256']==tasks.task_sha256 and fm['config_hash']==tasks.config_hash and
         tuple(fm['task_ids'])==tasks.task_ids and tuple(fm['periods'])==case.periods,'First-stage physical axes differ')
    h.rmp_residuals(A,fv,np.r_[plan.x,plan.r.ravel()])
    fpre=float(fv['c']@np.r_[plan.x,plan.r.ravel()])
    h.close(fpre,physical['costs']['Fpre_usd'],'First-stage physical cost differs',1e-4+1e-7*abs(fpre))
    selected=[r for r in envelope['recourse_events'] if r['iteration']==iteration and r['plan_sha256']==plan_sha]
    need(len(selected)==len(ids) and {r['scenario_index'] for r in selected}==set(range(len(ids))),
         'Selected plan/iteration lacks unique full support')
    templates={r['scenario_id']:r for r in envelope['templates']}
    constants=np.asarray(identity['constants_usd'],dtype=float)
    need(constants.shape==(len(ids),) and np.isfinite(constants).all(),'Invalid original scenario constants')
    selected=sorted(selected,key=lambda row:row['scenario_index']);result=[]
    for row in selected:
        remaining()
        i=row['scenario_index'];event=record(row['reference']);d=Path(row['reference']['path']).parent
        tr=templates[ids[i]];original=record(tr['original_reference']);td=Path(tr['original_reference']['path']).parent
        meta=h.metadata_header(td/'identity.json');arrays=h.npz(td/'arrays.npz')
        matrices={name:load_npz(td/(name+'.npz')).tocsr() for name in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
        scene=scene_loader(i)
        hashes=identity['lazy_input_binding']['scenario_sha256s'] if 'lazy_input_binding' in identity else identity['scenario_sha256s']
        need(scene.scenario_sha256==hashes[i] and scene.physical_case_sha256==case.case_sha256,'New scene identity differs')
        saved=dict(event=event,plan=h.read(d/'plan.json'),solver=h.read(d/'solver.json'),
            solution=h.npz(d/'solution.npz'),vector=h.npz(d/'primal_vector.npz')['vector'],
            dual=h.npz(d/'dual_kkt.npz'),fixed=h.npz(d/'fixed_rhs.npz'),first_stage_sha256=first_sha)
        result.append(kernel.audit_scenario(h,index=i,scenario_id=ids[i],plan_dict=plan_dict,
            constant=float(constants[i]),expected_template_sha256=original['template_sha256'],
            meta=meta,matrices=matrices,arrays=arrays,saved=saved,case=case,tasks=tasks,scene=scene,
            expected_iteration=iteration))
    remaining()
    return dict(elapsed_seconds=time.monotonic()-started,status='PASS_SELECTED_PLAN_FULL_SUPPORT_RECOURSE_NUMERICS',
        plan_sha256=plan_sha,iteration=iteration,support_ids=ids,q_usd=[r['q_usd'] for r in result],
        first_stage_cost_usd=fpre,rows=result,snapshot_reference=snapshot_reference,
        request_reference=expected_request_reference,
        sources=dict(helper=helper_source_reference,numerical_kernel=numerical_kernel_source_reference),
        full_support_recourse_verified=True,probability_audited=False,master_bound_audited=False,
        scientific_acceptance=False,formal_OOS=False,solver_calls=0)


def audit_selected_probability(*, recourse_audit, snapshot_reference,
        expected_request_reference, plan_dict):
    """Recompute the raw probability witness using the just-audited ordered Q.

    Accept only the in-process result from audit_fixed_plan_recourses or an
    externally sealed equivalent supplied by the audited parent. This function
    alone is not a gateway accepting arbitrary user-supplied audit dictionaries.
    """
    from .probability_check import check_probability_solution
    from .probability import ProbabilitySet
    need(recourse_audit['status']=='PASS_SELECTED_PLAN_FULL_SUPPORT_RECOURSE_NUMERICS'
         and recourse_audit['full_support_recourse_verified'] is True
         and recourse_audit['snapshot_reference']==snapshot_reference
         and recourse_audit['request_reference']==expected_request_reference
         and recourse_audit['plan_sha256']==canonical_hash(plan_dict), 'Foreign recourse audit')
    snapshot=document(snapshot_reference);request=document(expected_request_reference)
    need(snapshot['request']==expected_request_reference,'Foreign probability snapshot')
    identity=request['problem_identity'];ids=identity['scenario_ids'];iteration=recourse_audit['iteration']
    need(recourse_audit['support_ids']==ids,'Probability support differs')
    events=[]
    for ref in snapshot['events']:
        event=document(ref)
        if event['event']=='probability' and event['iteration']==iteration:
            events.append((ref,event))
    need(len(events)==1,'Selected iteration needs one raw probability witness')
    reference,event=events[0];record(reference)
    need(event['status']=='RAW_EVIDENCE_RETAINED_PENDING_AUDIT','Failed probability raw event')
    refs={Path(r['path']).name:r for r in event['artifacts']}
    q_input=document(refs['q_input.json']);prob_input=document(refs['probability_input.json']);solver=document(refs['solver.json'])
    need(q_input['plan']==plan_dict and q_input['support_ids']==ids and
         q_input['q_usd']==recourse_audit['q_usd'],'Probability witness uses another plan or Q')
    need(len(q_input['recourse_evidence'])==len(ids),'Incomplete probability Q references')
    # An explicit Q citation can refer to an earlier identical-plan witness.
    # Current full-support Q was independently recomputed above; an earlier
    # same-run/same-plan citation may retain its true iteration identity.
    for i,ref in enumerate(q_input['recourse_evidence']):
        need(ref in snapshot['events'],'Probability references external LP')
        row=document(ref)
        need(row['event']=='recourse' and row['scenario_index']==i and row['scenario_id']==ids[i]
             and row['plan_sha256']==recourse_audit['plan_sha256']
             and type(row['iteration']) is int and 1<=row['iteration']<=iteration
             and row['q_usd']==recourse_audit['q_usd'][i], 'Probability Q source differs')
    def probability_set(value):
        a=ProbabilitySet(**{k:value[k] for k in ('support_ids','feature_ids','features','lower','upper')})
        need(a.ambiguity_sha256==value['ambiguity_sha256'],'Probability input semantic SHA differs')
        return a
    original=probability_set(prob_input['input_ambiguity']);effective=probability_set(prob_input['effective_ambiguity'])
    need(original.ambiguity_sha256==identity['input_ambiguity_sha256'] and
         effective.ambiguity_sha256==identity['effective_ambiguity_sha256'],'Probability input differs from request')
    if identity['risk_model']=='DRO':
        need(effective.ambiguity_sha256==original.ambiguity_sha256,'DRO ambiguity changed')
    elif identity['risk_model']=='ARO':
        need(effective.support_ids==original.support_ids and not effective.feature_ids,'ARO is not the support simplex')
    else:
        raise ValueError('Unsupported probability risk model')
    check=check_probability_solution(effective,recourse_audit['q_usd'],solver,q_support_ids=ids)
    need(check.passed,'Independent probability witness failed')
    check=check.as_dict()
    return dict(status='PASS_SELECTED_PLAN_PROBABILITY_WITNESS',plan_sha256=recourse_audit['plan_sha256'],
        iteration=iteration,probability_event_reference=reference,check=check,
        first_stage_cost_usd=recourse_audit['first_stage_cost_usd'],
        probability_audited=True,master_bound_audited=False,scientific_acceptance=False,
        formal_OOS=False,solver_calls=0)
