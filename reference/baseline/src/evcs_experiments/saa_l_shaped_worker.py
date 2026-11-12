"""Internal formal SAA numerical worker; no CLI or production dispatch admission.

The original queue/jobctl prerequisites and immutable history must already be
valid. This implements the numerical call chain but deliberately supplies no
parent settlement, release acceptance, or permission to bypass dispatch gates.
"""
import json
from pathlib import Path
import time
import numpy as np
from . import campaign_batch as batch
from .formal_planning_worker import _formal_prerequisites
from .saa_l_shaped_formal import (FORMAL_ALGORITHM_ID, load_formal_l_shaped_problem,
    restore_formal_l_shaped_problem)
from .saa_l_shaped import run_saa_l_shaped
from .saa_l_shaped_audit_bridge import FormalFixedQEvidence
from .saa_l_shaped_master_evidence import NativeDevelopmentMasterEvidence
from .saa_l_shaped_candidate import write_formal_candidate, audit_formal_candidate
from .saa_record_bridge import build_saa_record_bridge
from . import saa_fixed_plan_audit as full_q
from .random_streams import canonical_hash

CHECKPOINT_KIND='SAA_L_SHAPED_QUEUE_CHECKPOINT_FORMAL_V1'


def need(ok,message):
    if not bool(ok):raise ValueError(message)


def read(ref):
    batch.verify_reference(ref);return batch.read_json(ref['path'])


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return batch.file_reference(path)


def _provisional_accounting(invocation,elapsed):
    prior=invocation['cumulative_plan_seconds']
    need(all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in (prior,elapsed))
         and prior+elapsed<=invocation['plan_budget_seconds'], 'Invalid original worker provisional accounting')
    return dict(attempt_accounted_seconds=elapsed,queue_prior_attempt_seconds=prior,
        cumulative_accounted_seconds=prior+elapsed)


def run_attempt(invocation_reference):
    """Real source-bound formal numerical chain, callable only under real jobctl.

    Initial160 input loading, source verification, restoration, original Q,
    master, audit and checkpoint export share the invocation's actual clock.
    The queue owns receipt/finalization/canonical and whole-parent settlement.
    This function has no development fixture or unlicensed solver fallback.
    """
    entered=time.monotonic();inv=read(invocation_reference);reqref=inv['item_request_reference'];req=read(reqref)
    budget,_=_formal_prerequisites(req,inv)
    need(req['model_input_mode']=='l_shaped_formal' and req['algorithm_id']==FORMAL_ALGORITHM_ID,
         'Explicit formal decomposition worker request required')
    need(req['method_id']=='SAA' and req['plan_id']==inv['plan_id'], 'Formal decomposition plan differs')
    need('formal_input_reference' not in req,'Worker request must not hash-reference its future preflight output')
    queue=read(inv['request_reference']);opening=batch._carryovers(queue)
    live=batch.read_json(Path(queue['queue_root'])/'state.json')
    latest,spent=batch._history(live,inv['request_reference'],Path(queue['queue_root']),opening)
    prior=inv['cumulative_plan_seconds'];plan=req['plan_id'];debt=opening.get(plan)
    need(prior==spent.get(plan,0.) and inv['plan_budget_seconds']==7200
         and inv['plan_remaining_seconds']==7200-prior,
         'Formal invocation does not inherit complete immutable queue history')
    need(debt is not None and debt['reference']==req.get('failure_carryover_reference')
         ==inv.get('failure_carryover_reference') and inv['opening_carryover_seconds']==debt['accounted_seconds'],
         'Formal decomposition requires the independently settled failed-parent debt')
    started=inv['attempt_started_monotonic'];stop=inv['worker_deadline_monotonic']
    need(all(type(v) in (int,float) and np.isfinite(v) for v in (started,stop,prior))
         and started<=entered<stop and stop-started<=7200-prior
         and stop-entered<=inv['remaining_seconds'], 'Formal worker deadline/reset differs')
    root=Path(inv['attempt_directory']);shared=Path(req['output_root']);result_path=Path(inv['result_path'])
    need(root.is_absolute() and root.resolve()==root and shared.is_absolute() and shared.resolve()==shared
         and result_path==root/'worker_result.json' and not result_path.exists(),'Formal worker paths differ')
    previous=inv['previous_checkpoint'];old=None;checkpoint_path=root/'kernel_checkpoint.json'
    if previous is not None:
        old=read(previous);last=latest.get(inv['item_id'])
        need(old['kind']==CHECKPOINT_KIND and old['request_reference']==reqref
             and last is not None and last['result']['status']=='TIME_LIMIT'
             and last['result']['checkpoint']==previous and old['worker_provisional_seconds']<=prior,
             'Formal resume is not the exact prior immutable queue attempt')
        read(old['kernel_reference']);read(old['event_inventory_reference'])
        problem=restore_formal_l_shaped_problem(old['formal_input_reference'],evidence_root=shared/'atoms',deadline=stop-10.)
        with checkpoint_path.open('xb') as stream:stream.write(Path(old['kernel_reference']['path']).read_bytes())
    else:
        need(latest.get(inv['item_id']) is None and not shared.exists(),
             'Initial formal worker cannot adopt uncommitted outputs or prior attempt')
        shared.mkdir(parents=True,exist_ok=False)
        problem=load_formal_l_shaped_problem(reqref,evidence_root=shared/'atoms',
            preflight_path=shared/'formal_input_preflight.json',deadline=stop-10.)
    if 'input_binding' in req:
        need(problem.input_binding==req['input_binding'],'Formal worker physical input binding differs')
    sources=req['planning_source_references']
    need([row['original_template_reference'] for row in problem.provider.index['rows']]==req['template_references'],
         'Formal worker complete ordered original template references differ')
    sink=FormalFixedQEvidence(problem=problem,source_references=sources,output_root=shared/'events',
        resume_inventory_reference=None if old is None else old['event_inventory_reference'],
        kernel_checkpoint_reference=None if old is None else old['kernel_reference'])
    master=NativeDevelopmentMasterEvidence(shared/'masters',threads=budget['master_threads'])
    # Reserve the original complete-evidence and seal allowances. Kernel clock
    # remains its unchanged7200 cap; the whole parent/queue grant is the tighter
    # actual wall bound after all inherited failed attempts and setup costs.
    remaining=stop-time.monotonic()-budget['final_certificate_reserve_seconds']-budget['shutdown_reserve_seconds']
    need(remaining>0,'Actual formal remaining budget cannot cover next iteration and original certificate reserve')
    state=run_saa_l_shaped(problem,checkpoint_path=checkpoint_path,resume=old is not None,
        expected_checkpoint_sha256=None if old is None else old['kernel_reference']['sha256'],
        budget_seconds=7200,wall_seconds=remaining,master_time_limit_seconds=600,
        recourse_time_limit_seconds=60,mip_relative_gap=.001,optimality_relative_gap=.005,
        evidence_sink=sink,master_sink=master)
    need(state['status'] in ('CONVERGED','TIME_LIMIT','ITERATION_LIMIT'), 'Failed formal kernel requires diagnosis')
    kref=batch.file_reference(checkpoint_path);best=state['best'];qref=None
    if best is not None and state['lower_bound_usd'] is not None:
        plan_value=problem.plan(best['anchor'])
        digest=canonical_hash(dict(x=plan_value.x.tolist(),r=plan_value.r.tolist(),
            task_sha256=plan_value.task_sha256,config_hash=plan_value.config_hash))
        selected=[i for i,s in sink.sweeps.items() if s['plan_sha256']==digest
                  and [s['q'].get(j) for j in range(160)]==best['q_usd']]
        need(selected,'Best formal candidate lacks matching original full Q events')
        i=selected[0];qref=sink._checkpoints.get(i) or sink.checkpoint(i)
    inventory=sink.snapshot(kref)
    result=dict(invocation_sha256=invocation_reference['sha256'],queue_prior_attempt_seconds=prior,raw_engine_status=state['status'],
        driver_status=state['status'],artifacts=[problem.formal_input_reference,kref,inventory],
        formal_OOS=False,scientific_acceptance=False,whole_jobctl_parent_settled=False)
    if qref is None:
        checkpoint=write(root/'queue_checkpoint.json',dict(kind=CHECKPOINT_KIND,request_reference=reqref,
            formal_input_reference=problem.formal_input_reference,kernel_reference=kref,event_inventory_reference=inventory,
            worker_provisional_seconds=time.monotonic()-started,source_references=sources,native_algorithm_status=state['status']))
        result.update(status='TIME_LIMIT',resumable=True,checkpoint=checkpoint)
        result['artifacts'].append(checkpoint)
    else:
        qrequest=read(qref)['request_reference']
        envelope=full_q.bind_saa_template_envelope(request_reference=qrequest,
            template_references=req['template_references'],first_stage_sha256=problem.first_stage.matrix_sha256,
            source_references=sources,output_path=root/'template_envelope.json')
        evidence=dict(kernel_checkpoint_reference=kref,
            master_execution_references=[r['evidence_reference'] for r in state['master_history']],
            full_q_checkpoint_reference=qref,full_q_request_reference=qrequest,
            template_envelope_reference=envelope,source_references=sources)
        cref=write_formal_candidate(problem,plan_id=req['plan_id'],**evidence,output_path=root/'candidate.json')
        joint=audit_formal_candidate(problem,candidate_reference=cref,expected_plan_id=req['plan_id'],
            expected_evidence=evidence,case=problem.case,tasks=problem.tasks,scene_loader=problem.scene_loader,
            helper_reference=req['helper_reference'],full_q_source_references=sources,
            deadline=stop-10.,output_path=root/'joint_audit.json')
        record=write(root/'candidate_record.json',dict(plan_id=req['plan_id'],method_id='SAA',algorithm_id=FORMAL_ALGORITHM_ID,
            request_reference=reqref,candidate=read(cref),l_shaped_evidence=evidence,
            raw_engine_status=state['status'],driver_status=state['status']))
        numerical=build_saa_record_bridge(reqref,record,joint['audit_reference'],root/'numerical_audit.json')
        result.update(status='COMPLETE',full_support_verified=True,numerical_audit_reference=numerical)
        result['artifacts'].extend([cref,record,joint['audit_reference'],numerical,qref,envelope])
    elapsed=time.monotonic()-started
    need(time.monotonic()<stop-10. and prior+elapsed<=7200,
         'Formal worker evidence export consumed original seal reserve')
    result.update(_provisional_accounting(inv,elapsed))
    write(result_path,result)
    return result
