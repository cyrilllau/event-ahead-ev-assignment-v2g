"""Genuine tiny development worker for the original campaign_batch integration.

This fixture is deliberately not a production worker. Native HiGHS and original
LPs are real. The first call stops after one complete iteration; a second queue
attempt resumes actual event/checkpoint bytes and publishes through the original
receipt/final seal. Test fixture input preparation is outside receipt accounting.
"""
import json
import os
from pathlib import Path
import time

from evcs_experiments import campaign_batch as batch
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_candidate as candidate_module
from evcs_experiments import saa_fixed_plan_audit as full_q
from evcs_experiments.saa_l_shaped_audit_bridge import DevelopmentFixedQEvidence
from evcs_experiments.saa_l_shaped_master_evidence import DevelopmentMasterEvidence
from evcs_experiments.saa_l_shaped_provider import StoredDevelopmentSAAProblem
from evcs_experiments.saa_record_bridge import build_saa_record_bridge
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse_template_index import TemplateProvider
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases


def read(ref):
    batch.verify_reference(ref)
    return batch.read_json(ref['path'])


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return batch.file_reference(path)


def main():
    invocation_path = Path(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
    invocation_ref = batch.file_reference(invocation_path)
    assert invocation_ref['sha256'] == os.environ['EVCS_BATCH_INVOCATION_SHA256']
    invocation = read(invocation_ref)
    request_ref = invocation['item_request_reference']
    request = read(request_ref)
    assert request['domain'] == invocation['domain'] == 'development'
    assert request['model_input_mode'] == 'l_shaped_development'
    assert request['algorithm_id'] == kernel.ALGORITHM_ID
    assert request['budget'] == dict(cumulative_wall_seconds=30., single_master_seconds=5.,
        final_certificate_reserve_seconds=5., shutdown_reserve_seconds=1.)
    assert invocation['plan_budget_seconds'] == 30.
    assert invocation['plan_remaining_seconds'] == 30. - invocation['cumulative_plan_seconds']
    for ref in request['planning_source_references']:
        batch.verify_reference(ref)
    assert batch.file_reference(__file__) in request['planning_source_references']
    root = Path(invocation['attempt_directory'])
    shared = Path(request['output_root'])
    shared.mkdir(parents=True, exist_ok=True)
    toy = build_golden_cases()[0]
    first = assemble_first_stage(toy.case, toy.tasks)
    provider = TemplateProvider(request['template_index_reference'], request['template_association_review_reference'],
        first_stage=first, case=toy.case, tasks=toy.tasks, scenario_ids=toy.scenario_ids,
        scenario_sha256s=[s.scenario_sha256 for s in toy.scenarios], domain='development')
    problem = StoredDevelopmentSAAProblem(toy.case, toy.tasks, toy.scenarios,
        toy.scenario_cost_constants_usd - 100., provider=provider, evidence_root=shared/'atoms', capacity=1)
    identity = problem.identity
    sources = request['planning_source_references']
    sink_options = dict(case=toy.case, tasks=toy.tasks, scenario_ids=identity['support_ids'],
        scenario_sha256s=identity['scenario_sha256s'], template_sha256s=identity['template_sha256s'],
        constants_usd=identity['constants_usd'], scene_loader=toy.scenarios.__getitem__,
        input_binding=request['input_binding'], source_references=sources,
        output_root=shared/'events')
    previous = invocation['previous_checkpoint']
    checkpoint_path = root/'kernel_checkpoint.json'
    if previous is not None:
        old = read(previous)
        assert old['kind'] == 'SAA_L_SHAPED_QUEUE_CHECKPOINT_DEVELOPMENT_V1'
        assert old['request_reference'] == request_ref
        assert invocation['cumulative_plan_seconds'] > old['worker_provisional_seconds'] > 0
        read(old['kernel_reference'])
        checkpoint_path.write_bytes(Path(old['kernel_reference']['path']).read_bytes())
        sink_options.update(resume_inventory_reference=old['event_inventory_reference'],
            kernel_checkpoint_reference=old['kernel_reference'])
    sink = DevelopmentFixedQEvidence(**sink_options)
    master = DevelopmentMasterEvidence(shared/'masters')
    # Parent receipt supplies the hard remaining allowance; the kernel's own
    # fixed cumulative clock is retained, never reset or expanded on resume.
    remaining = invocation['worker_deadline_monotonic'] - time.monotonic() - 6.
    assert 0 < remaining <= invocation['plan_remaining_seconds']
    state = kernel.run_saa_l_shaped(problem, checkpoint_path=checkpoint_path, resume=previous is not None,
        expected_checkpoint_sha256=None if previous is None else old['kernel_reference']['sha256'],
        budget_seconds=30., wall_seconds=remaining, master_time_limit_seconds=5., recourse_time_limit_seconds=5.,
        max_iterations=1 if previous is None else 1000, evidence_sink=sink, master_sink=master)
    kernel_ref = batch.file_reference(checkpoint_path)
    if previous is not None:
        assert state['status'] == 'CONVERGED'
        plan = problem.plan(state['best']['anchor'])
        digest = canonical_hash(dict(x=plan.x.tolist(), r=plan.r.tolist(), task_sha256=plan.task_sha256,
            config_hash=plan.config_hash))
        selected = [i for i, sweep in sink.sweeps.items() if sweep['plan_sha256'] == digest
            and len(sweep['q']) == len(problem.support_ids)]
        assert selected
        # Final inventory must include this checkpoint's new event-root file.
        qref = sink.checkpoint(selected[0])
    inventory_ref = sink.snapshot(kernel_ref)
    trace = write(root/'actual_execution.json', dict(pid=os.getpid(), ppid=os.getppid(),
        invocation_reference=invocation_ref, request_reference=request_ref, algorithm_id=kernel.ALGORITHM_ID,
        source_references=sources, queue_prior_attempt_seconds=invocation['cumulative_plan_seconds'],
        previous_checkpoint=previous, kernel_reference=kernel_ref, event_inventory_reference=inventory_ref,
        algorithm_status=state['status'], master_execution_references=[r['evidence_reference'] for r in state['master_history']],
        kernel_sessions=state['sessions'], kernel_trace=state['trace'], formal_admission=False,
        native_solver_calls_real=True, jobctl_supervision=False, actual_campaign_parent=True,
        whole_jobctl_parent_settled=False))
    result = dict(invocation_sha256=invocation_ref['sha256'], raw_engine_status=state['status'],
        driver_status=state['status'], queue_prior_attempt_seconds=invocation['cumulative_plan_seconds'],
        artifacts=[kernel_ref, inventory_ref, trace], formal_OOS=False, scientific_acceptance=False)
    if previous is None:
        assert state['status'] == 'ITERATION_LIMIT' and state['iterations'] == 1
        elapsed = time.monotonic() - invocation['attempt_started_monotonic']
        checkpoint = write(root/'queue_checkpoint.json', dict(kind='SAA_L_SHAPED_QUEUE_CHECKPOINT_DEVELOPMENT_V1',
            request_reference=request_ref, kernel_reference=kernel_ref, event_inventory_reference=inventory_ref,
            worker_provisional_seconds=elapsed, native_algorithm_status=state['status'], source_references=sources))
        result.update(status='TIME_LIMIT', resumable=True, checkpoint=checkpoint)
        result['artifacts'].append(checkpoint)
    else:
        assert state['status'] == 'CONVERGED'
        qrequest = read(qref)['request_reference']
        envelope = full_q.bind_saa_template_envelope(request_reference=qrequest,
            template_references=request['template_references'], first_stage_sha256=first.matrix_sha256,
            source_references=sources, output_path=root/'template_envelope.json')
        evidence = dict(kernel_checkpoint_reference=kernel_ref,
            master_execution_references=[row['evidence_reference'] for row in state['master_history']],
            full_q_checkpoint_reference=qref, full_q_request_reference=qrequest,
            template_envelope_reference=envelope, source_references=sources)
        cref = candidate_module.write_development_candidate(problem, plan_id=request['plan_id'],
            **evidence, output_path=root/'candidate.json')
        joint = candidate_module.audit_development_candidate(problem, candidate_reference=cref,
            expected_plan_id=request['plan_id'], expected_evidence=evidence, case=toy.case, tasks=toy.tasks,
            scene_loader=toy.scenarios.__getitem__, helper_reference=request['helper_reference'],
            full_q_source_references=sources, deadline=invocation['worker_deadline_monotonic'] - 1.,
            output_path=root/'joint_audit.json')
        record = write(root/'candidate_record.json', dict(plan_id=request['plan_id'], method_id='SAA',
            algorithm_id=kernel.ALGORITHM_ID, request_reference=request_ref, candidate=read(cref),
            l_shaped_evidence=evidence, raw_engine_status=state['status'], driver_status=state['status']))
        numerical = build_saa_record_bridge(request_ref, record, joint['audit_reference'], root/'numerical_audit.json')
        result.update(status='COMPLETE', full_support_verified=True, numerical_audit_reference=numerical)
        result['artifacts'].extend([cref, record, joint['audit_reference'], numerical, qref, envelope])
    elapsed = time.monotonic() - invocation['attempt_started_monotonic']
    result.update(attempt_accounted_seconds=elapsed,
        cumulative_accounted_seconds=invocation['cumulative_plan_seconds'] + elapsed)
    write(invocation['result_path'], result)


if __name__ == '__main__':
    main()
