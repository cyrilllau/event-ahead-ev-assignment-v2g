"""One explicit saved-model SAA memory retry; no assembly or historical relabel.

The expense dispatcher owns supervisor/claim/queue admission. This component
reuses the original physical factory, numerical phases and numerical bridge.
Its separate source reference is never substituted for the old mathematical
source inventory. Formal admission is limited to the first Helene memory failure.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
import inspect
from pathlib import Path
import sys
import time

from . import campaign_batch as batch, native_gurobi, saa_planning_pipeline, saa_planning_worker, saa_record_bridge
from .formal_planning_worker import _formal_prerequisites
from .formal_saa import prepare_formal_saa
from .planning_budget import validate_budget
from .random_streams import canonical_hash
from .saa_planning_worker import _DevelopmentFixture
from . import saa_validator_recovery

KIND = 'SAA_SAVED_COMPLETE_MODEL_MEMORY_POLICY_V1'
MODE = 'recover_failed_complete_model'
EXECUTION = 'SAA_SAVED_COMPLETE_MODEL_EXECUTION_V1'
VALIDATOR_SHA256 = 'f49ac95b15c1d45b58ee51ab852f863beb4c11096848550c6c4f05f8d394b20a'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(reference):
    batch.verify_reference(reference)
    return batch.read_json(reference['path'])


def new(path, value):
    return batch._write_new(Path(path), value)


def _formal(fixture, domain):
    if fixture is None:
        need(domain == 'formal', 'Public saved-model caller accepts formal inputs only')
        return True
    need(isinstance(fixture, _DevelopmentFixture) and domain == 'development',
         'Existing private fixture cannot bypass a formal recovery')
    return False


def verify_policy(policy_reference, worker_request_reference, *, _development_fixture=None):
    """Check exact metadata/source/debt; payload arrays are rehashed in phases."""
    p = read(policy_reference); req = read(worker_request_reference)
    formal = _formal(_development_fixture, req['domain'])
    need(p['kind'] == KIND and p['domain'] == req['domain'] and
         p['adapter_reference'] == batch.file_reference(__file__), 'Saved-model policy/source differs')
    need(type(p['soft_memory_limit_gb']) in (int, float) and p['soft_memory_limit_gb'] == 64.,
         'Only explicit24-to64GB native memory policy is implemented')
    oldref = p['original_worker_reference']; old = read(oldref)
    need(old['domain'] == req['domain'] and old['method_id'] == req['method_id'] == 'SAA' and
         old['model_input_mode'] == 'assemble', 'Original failure must be a SAA assembly-mode request')
    fee = read(p['fee_reference']); decision = read(p['isolation_decision_reference'])
    need(fee['kind'] == 'SAA_ASSEMBLED_MASTER_FAILED_PARENT_EXPENSE_V1' and fee['passed'] is True and
         fee['accounting_PASS'] is True and fee['status'] == 'SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION' and
         fee['plan_id'] == old['plan_id'] and fee['references']['worker_request'] == oldref and
         decision['kind'] == 'SAA_MEMORY_FAILURE_ARO_ISOLATION_DECISION_V1' and
         decision['terminal_fee_settlement_reference'] == p['fee_reference'] and
         decision['old_queue_permanently_quarantined'] is True and decision['original_queue_rearm_prohibited'] is True,
         'Actual failed-parent fee and permanent mixed-queue isolation required')
    for policy_key, fee_key in [('model_manifest_reference', 'model_manifest'),
                               ('model_completion_reference', 'model_completion'),
                               ('failed_master_reference', 'master_result'),
                               ('failed_native_reference', 'native_master')]:
        need(p[policy_key] == fee['references'][fee_key], 'Policy replaces frozen failed-parent artifact: ' + policy_key)
    for key in ('model_manifest_reference', 'model_completion_reference'):
        need(p[key] == decision[key], 'Isolation decision/model source differs: ' + key)
    need(p['failed_native_reference'] == decision['native_failure_reference'] and
         decision['terminal_job_result_reference'] == fee['terminal_job_result_reference'] and
         decision['original_queue_reference'] == fee['original_queue_reference'], 'Isolation failure/parent/queue authority differs')
    live_failed_ref = fee['references']['phase_checkpoint']
    read(live_failed_ref)
    need(all(p['failed_phase_checkpoint_reference'][key] == live_failed_ref[key] for key in ('sha256', 'size_bytes')),
         'Immutable failed-phase snapshot does not match the settled terminal bytes')
    spent = fee['SAA_charged_seconds']; cap = old['budget']['cumulative_wall_seconds']
    need(type(spent) in (int, float) and math.isfinite(spent) and 0 < spent < cap and
         fee['SAA_remaining_seconds'] == cap-spent and decision['SAA_charged_seconds'] == spent,
         'Original complete parent expense/balance differs')
    if formal:
        need(old['plan_id'] == 'risk_model.helene_2024.SAA.baseline' and old['target_id'] == 'helene_2024' and
             cap == 7200 and spent == 2065.692102 and fee['SAA_remaining_seconds'] == 5134.307898,
             'Only this first Helene saved-model memory recovery is admitted')
    modules = dict(native=native_gurobi, pipeline=saa_planning_pipeline,
                   worker=saa_planning_worker, bridge=saa_record_bridge)
    need(set(p['original_module_references']) == set(modules), 'Exact original module scope required')
    for name, module in modules.items():
        r = p['original_module_references'][name]
        need(read_source(r) == batch.file_reference(module.__file__) and r in old['planning_source_references'],
             'Original mathematical module differs: ' + name)
    for r in old['planning_source_references']:
        batch.verify_reference(r)
    validator = p.get('validator_reference')
    if formal or validator is not None:
        batch.verify_reference(validator)
        need(validator['sha256'] == VALIDATOR_SHA256 and
             p['original_validator_installer_reference'] == batch.file_reference(saa_validator_recovery.__file__) and
             p['original_validator_installer_reference'] in old['planning_source_references'],
             'Exact reviewed typed validator and original alias installer required')
    expected = deepcopy(old)
    expected.update(output_root=p['new_output_root'], model_input_mode=MODE,
        model_manifest_reference=p['model_manifest_reference'], original_worker_reference=oldref,
        saved_model_recovery_policy_reference=policy_reference, failure_carryover_reference=p['fee_reference'])
    need(req == expected, 'Recovery worker changed fields outside explicit model/debt/source/output binding')
    oldroot = Path(old['output_root']); root = Path(req['output_root'])
    need(oldroot.is_absolute() and root.is_absolute() and root != oldroot and
         not root.is_relative_to(oldroot) and not oldroot.is_relative_to(root), 'New output must be separate from original failure')
    if formal:
        need(root == oldroot.parent.parent/'p2-saa-memory-recovery-20260912/SAA', 'Wrong first-recovery output root')
    model = read(p['model_manifest_reference']); completion = read(p['model_completion_reference'])
    need(model['domain'] == req['domain'] and model['risk_model'] == 'SAA' and
         model['status'] == 'ASSEMBLED_PENDING_RESOURCE_AND_INDEPENDENT_NUMERICAL_AUDIT' and
         canonical_hash({k:v for k,v in model.items() if k != 'manifest_sha256'}) == model['manifest_sha256'],
         'Complete original model semantic identity differs')
    need(completion['kind'] == 'SAA_COMPLETE_MODEL_CHECKPOINT_V1' and completion['request_reference'] == oldref and
         completion['model_manifest_reference'] == p['model_manifest_reference'] and
         completion['model_input_mode'] == 'assemble' and completion['planning_source_references'] == old['planning_source_references'] and
         completion['input_binding'] == model['input_binding'] and completion['prior_cumulative_seconds'] == 0 and
         0 < completion['preparation_elapsed_seconds'] <= spent, 'Original completion/source/expense differs')
    master = read(p['failed_master_reference']); raw = read(p['failed_native_reference']); cp = read(p['failed_phase_checkpoint_reference'])
    need(master['status'] == 'MEM_LIMIT' and master['primal'] is None and master['lower_bound_usd'] is None and
         master['full_support_verified'] is False and master['model_manifest'] == p['model_manifest_reference'] and
         master['raw_native_result'] == p['failed_native_reference'], 'Only original no-incumbent native memory failure may retry')
    native = raw['native_interface']
    need(native['raw_model_status'] == 17 and native['solution_count'] == 0 and
         native['effective_options']['SoftMemLimit'] == 24. and native['effective_options']['Threads'] == 4,
         'Original native memory failure/options differ')
    need(cp['status'] == 'FAILED' and cp['master_result_reference'] == p['failed_master_reference'] and
         cp['fixed_q_checkpoint_reference'] is None and cp['candidate_audit_reference'] is None,
         'Failed phase contains different or completed numerical work')
    phase = read(cp['request_reference'])
    need(cp['request_reference'] == fee['references']['phase_request'], 'Failed phase request differs from settled authority')
    need(phase['model_manifest_reference'] == p['model_manifest_reference'] and phase['sources'] == old['planning_source_references'] and
         phase['input_binding'] == model['input_binding'] and phase['scenario_ids'] == model['scenario_ids'] and
         phase['scenario_sha256s'] == model['scenario_sha256s'], 'Original phase/model/source/support differs')
    need(Path(p['failed_master_reference']['path']) == oldroot/'phases/master/result.json' and
         Path(p['failed_native_reference']['path']) == oldroot/'phases/master/native_result.json' and
         Path(p['failed_phase_checkpoint_reference']['path']).parent == oldroot/'phases/checkpoints' and
         Path(p['model_manifest_reference']['path']) == oldroot/'model/manifest.json' and
         Path(p['model_completion_reference']['path']) == oldroot/'model_completion.json', 'Original model/failure artifact location differs')
    if formal:
        need(len(model['scenario_ids']) == 160 and len(set(model['scenario_ids'])) == 160 and
             old['budget'] == dict(cumulative_wall_seconds=7200, single_master_seconds=600,
                final_certificate_reserve_seconds=1790, shutdown_reserve_seconds=10, master_threads=4,
                recourse_workers=1, blas_threads=1, normal_gap_relative=.005, rmp_mip_gap=.001),
             'Original formal support/resource policy changed')
        observation = read(p['resource_observation_reference'])
        need(observation['available_memory_kib'] >= 128*1024*1024 and observation['native_soft_limit_gb'] == 64. and
             observation['not_a_process_rss_cap'] is True, 'Reviewed64GB resource headroom missing')
    return dict(kind='SAA_SAVED_MODEL_POLICY_VERIFIED_V1', policy_reference=policy_reference,
        worker_request_reference=worker_request_reference, adapter_reference=p['adapter_reference'],
        original_worker_reference=oldref, original_model_manifest_reference=p['model_manifest_reference'],
        original_model_completion_reference=p['model_completion_reference'], original_fee_reference=p['fee_reference'],
        opening_accounted_seconds=spent, cumulative_budget_seconds=cap, remaining_seconds=cap-spent,
        original_mathematical_source_references=old['planning_source_references'],
        validator_reference=validator, original_validator_installer_reference=p.get('original_validator_installer_reference'),
        soft_memory_limit_gb=64., reassembly_performed=False, original_failure_bytes_unchanged=True,
        metadata_source_checks_only=True, model_payload_rehash_required_inside_numerical_budget=True,
        independent_launch_admission=False, scientific_acceptance=False, formal_OOS=False)


def read_source(reference):
    batch.verify_reference(reference)
    return reference


def verify_validator_installed(policy):
    """Dispatcher owns one restore scope covering child and parent settlement."""
    from . import formal_planning_worker
    module = sys.modules.get('evcs_experiments._explicit_recovery_checkpoint_closure')
    need(module is not None and batch.file_reference(module.__file__) == policy['validator_reference'],
         'Reviewed typed validator must be installed by source-bound dispatcher')
    closure = saa_record_bridge._closure
    need(formal_planning_worker._closure is module.verify_checkpoint_closure and
         saa_planning_worker._closure is module.verify_checkpoint_closure and
         closure.__module__ == saa_validator_recovery.__name__ and
         closure.__qualname__ == 'install_validator.<locals>.value_closure' and
         inspect.getclosurevars(closure).nonlocals.get('module') is module,
         'Original reviewed alias installer is not active for numerical/parent bridge')


def _live_resource_observation():
    """A charged current-host check immediately before formal phase work."""
    rows = [line.split() for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')]
    need(len(rows) == 1 and len(rows[0]) == 3 and rows[0][2] == 'kB',
         'Current Linux MemAvailable observation unavailable')
    available = int(rows[0][1])
    need(available >= 128*1024*1024, 'Current resource headroom below reviewed128GiB minimum')
    return dict(kind='SAA_SAVED_MODEL_CURRENT_RESOURCE_OBSERVATION_V1', observed_at_utc=batch.utcnow(),
        source_path='/proc/meminfo', available_memory_kib=available, native_soft_limit_gb=64.,
        not_a_process_rss_cap=True, accounting_scope='Inside the original attempt start/deadline')


@contextmanager
def _memory_policy():
    """Same native parameter delta as the existing audited memory hook."""
    original = native_gurobi._parameters
    need(original.__module__ == native_gurobi.__name__, 'Native resource hook already installed')
    def parameters(options):
        seconds, result = original(options)
        need(0 < seconds <= 600 and result['SoftMemLimit'] == 24. and result['Threads'] == 4 and
             result['MIPGap'] == .001 and all(result[k] == 1e-8 for k in ('FeasibilityTol', 'IntFeasTol', 'OptimalityTol')),
             'Unexpected original native parameters in memory recovery')
        return seconds, dict(result, SoftMemLimit=64.)
    native_gurobi._parameters = parameters
    try:
        yield
    finally:
        native_gurobi._parameters = original


def run_attempt(invocation_reference, *, policy_reference, reservation_reference,
                expense_module, _development_fixture=None):
    """Original numerical phases and bridge, with explicit failed-model input.

The supplied expense module is source-bound by the independent dispatcher;
formal callers must also supply its validated reservation. No CLI bypass exists.
"""
    entered = time.monotonic(); inv = read(invocation_reference); reqref = inv['item_request_reference']; req = read(reqref)
    need(req['schema_version'] == inv['schema_version'] == 1 and req['domain'] == inv['domain'] and
         req['method_id'] == 'SAA' and req['plan_id'] == inv['plan_id'], 'Saved-model invocation schema/domain/plan differs')
    started = inv['attempt_started_monotonic']; deadline = inv['worker_deadline_monotonic']
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (started, deadline)) and started <= entered < deadline,
         'Original invocation deadline invalid')
    formal = _formal(_development_fixture, req['domain'])
    proof = verify_policy(policy_reference, reqref, _development_fixture=_development_fixture)
    p = read(policy_reference); prior = inv['cumulative_plan_seconds']
    need(inv['previous_checkpoint'] is None and inv['previous_result_reference'] is None and
         prior == proof['opening_accounted_seconds'] and inv['opening_carryover_seconds'] == prior and
         inv['failure_carryover_reference'] == req['failure_carryover_reference'] == p['fee_reference'] and
         inv['plan_budget_seconds'] == proof['cumulative_budget_seconds'] and
         inv['plan_remaining_seconds'] == proof['remaining_seconds'] and
         deadline-started <= inv['plan_remaining_seconds'] and deadline-entered <= inv['remaining_seconds']+1e-8,
         'First saved-model recovery cannot reset expense, reuse a prior attempt or increase its grant')
    reservation = expense_module.verify_reservation(reservation_reference, invocation_reference=invocation_reference,
                                                    policy_reference=policy_reference)
    need(reservation['plan_id'] == req['plan_id'] and reservation['accounted_seconds'] == prior,
         'Source-bound expense reservation does not bind this invocation/opening')
    if formal:
        verify_validator_installed(p)
        need(batch.file_reference(expense_module.__file__) == reservation['module_reference'],
             'Expense reservation module differs from executing source')
        budget, _ = _formal_prerequisites(req, inv)
        prepare = prepare_formal_saa; master = audit = 600.; seal = 10.
    else:
        budget = validate_budget(_development_fixture.budget)
        need(budget == req['budget'], 'Private development budget must still match')
        prepare = _development_fixture.prepare; master = _development_fixture.master_seconds
        audit = _development_fixture.audit_reserve_seconds; seal = _development_fixture.seal_reserve_seconds
    attempt = Path(inv['attempt_directory']); result_path = Path(inv['result_path']); root = Path(req['output_root'])
    need(attempt.is_absolute() and result_path.is_absolute() and result_path == attempt/'worker_result.json' and
         not result_path.exists() and not root.exists(),
         'Recovery result/output already exists; terminal settlement required')
    root.mkdir(parents=True, exist_ok=False)
    def used():
        return time.monotonic()-started
    execution_ref = new(attempt/'saved_model_execution.json', dict(kind=EXECUTION, invocation_reference=invocation_reference,
        reservation_reference=reservation_reference, expense_module_reference=reservation.get('module_reference'),
        parent_execution_reference=reservation.get('parent_execution_reference'),
        original_model_reuse=proof, new_output_root=str(root), started_at_monotonic=started,
        reassembly_performed=False, original_phase_reused=False, mathematical_parameters_changed=False,
        native_resource_delta=dict(SoftMemLimit=dict(old=24., new=64.)), scientific_acceptance=False, formal_OOS=False))
    try:
        prepared = prepare(summary_reference=req['summary_reference'], audit_reference=req['audit_reference'],
            target_id=req['target_id'], template_index_reference=req.get('template_index_reference'),
            template_association_review_reference=req.get('template_association_review_reference'))
        physical = prepared.physical; modelref = p['model_manifest_reference']; model = read(modelref)
        need(model['input_binding'] == prepared.input_binding and model['scenario_ids'] == list(prepared.recourses.scenario_ids) and
             model['scenario_sha256s'] == list(prepared.recourses.scenario_sha256s), 'Actual regenerated physical bindings differ from saved model')
        live_resource_ref = new(attempt/'native_resource_observation.json', _live_resource_observation()) if formal else None
        admission_ref = None
        if formal:
            admission_ref = new(root/'formal_admission.json', dict(kind='SAA_FORMAL_PHASE_ADMISSION_V1', domain='formal',
                request_reference=reqref, model_manifest_reference=modelref, planning_source_references=req['planning_source_references'],
                budget=budget, model_input_mode=MODE, original_model_completion_reference=p['model_completion_reference'],
                original_failed_parent_fee_reference=p['fee_reference'], saved_model_execution_reference=execution_ref))
        # The phase implementation checks the actual remaining grant before any
        # large reload, then rehashes every saved matrix under the same deadline.
        with _memory_policy():
            phase = saa_planning_pipeline.run_saa_numerical_phases(model_manifest_reference=modelref,
                first_stage=physical.first_stage, recourses=prepared.recourses, case=physical.case, tasks=physical.tasks,
                scene_loader=physical.bank.load, input_binding=prepared.input_binding,
                scenario_ids=prepared.recourses.scenario_ids, scenario_sha256s=prepared.recourses.scenario_sha256s,
                template_references=req['template_references'], helper_reference=req['helper_reference'],
                source_references=req['planning_source_references'], output_directory=root/'phases',
                deadline=deadline-seal, cumulative_budget_seconds=budget['cumulative_wall_seconds'],
                prior_accounted_seconds=prior+used(), resume_checkpoint_reference=None,
                master_seconds=master, audit_reserve_seconds=audit, seal_reserve_seconds=seal,
                domain=req['domain'], formal_admission_reference=admission_ref)
        complete = phase['status'] in ('COMPLETE_FORMAL_SAA_NUMERICAL_PHASES', 'COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES')
        need(complete or phase['status'].startswith('TIME_LIMIT_'), 'Saved-model phase failed; preserve partials for settlement')
        joined_ref = phase['candidate_audit_reference']; joined = candidate = rawstatus = None
        if complete:
            joined = read(joined_ref)
            need(joined['status'] == 'PASS_SAA_CANDIDATE_NUMERICAL_AUDIT' and joined['passed'] is True,
                 'Complete original joined numerical audit required')
            qcp = read(phase['fixed_q_checkpoint_reference']); plan = read(qcp['request_reference'])['plan']
            need(canonical_hash(plan) == joined['plan_sha256'], 'Original native/fixed-Q plan identity differs')
            rawstatus = joined['native_status']
            candidate = dict(plan=plan, plan_sha256=joined['plan_sha256'], lower_bound_usd=joined['lower_bound_usd'],
                upper_bound_usd=joined['upper_bound_usd'], gap_relative=joined['gap_relative'], native_master_status=rawstatus,
                native_master_solver_status=joined['native_solver_status'], risk_certificate_kind='fixed_empirical')
        driver = ('CONVERGED' if joined['normal_gap_met'] else 'TIME_LIMIT') if complete else 'TIME_LIMIT'
        candidate_ref = new(attempt/'candidate_solver_record.json', dict(plan_id=req['plan_id'], method_id='SAA',
            driver_status=driver, raw_engine_status=rawstatus, candidate=candidate, request_reference=reqref,
            candidate_audit_reference=joined_ref, saved_model_execution_reference=execution_ref, formal_OOS=False))
        numerical_ref = None
        if complete:
            numerical_ref = saa_record_bridge.build_saa_record_bridge(reqref, candidate_ref, joined_ref,
                                                                       output_path=attempt/'numerical_audit.json')
        checkpoint = new(attempt/'worker_checkpoint.json', dict(schema_version=1, request_reference=reqref, output_root=str(root),
            phase_checkpoint_reference=phase['checkpoint_reference'], phase_status=phase['status'], model_manifest_reference=modelref,
            original_model_completion_reference=p['model_completion_reference'], formal_admission_reference=admission_ref,
            candidate_record_reference=candidate_ref, saved_model_execution_reference=execution_ref,
            current_native_resource_reference=live_resource_ref,
            cumulative_accounted_seconds=prior+used(), requires_explicit_recovery_terminal_settlement=True, formal_OOS=False))
        artifacts = [execution_ref, candidate_ref, checkpoint, modelref, p['model_completion_reference']]
        for reference in (phase['checkpoint_reference'], joined_ref, numerical_ref, admission_ref, live_resource_ref):
            if reference is not None and reference not in artifacts:
                artifacts.append(reference)
        elapsed = used()
        need(time.monotonic() < deadline and prior+elapsed < budget['cumulative_wall_seconds'],
             'Saved-model export exceeded original cumulative/parent grant')
        result = dict(schema_version=1, status='COMPLETE' if complete else 'TIME_LIMIT', invocation_sha256=invocation_reference['sha256'],
            artifacts=artifacts, checkpoint=checkpoint, resumable=not complete, declared_work_complete=complete,
            scientific_acceptance=False, formal_OOS=False, full_support_verified=complete, numerical_audit_reference=numerical_ref,
            driver_status=driver, raw_engine_status=rawstatus, has_candidate=candidate is not None,
            cumulative_accounted_seconds=prior+elapsed, cumulative_seconds_provisional=True, attempt_accounted_seconds=elapsed,
            queue_prior_attempt_seconds=prior, worker_seal_reserve_seconds=seal, attempt_started_monotonic=started,
            worker_deadline_monotonic=deadline, failure_carryover_reference=p['fee_reference'], opening_carryover_seconds=prior,
            saved_model_execution_reference=execution_ref, requires_explicit_recovery_terminal_settlement=True)
        new(result_path, result)
        return result
    except Exception as exc:
        failure = attempt/'worker_failure.json'
        if not failure.exists():
            new(failure, dict(status='FAILED_SAA_SAVED_MODEL_RECOVERY_REQUIRES_SETTLEMENT', error_type=type(exc).__name__,
                error=str(exc), cumulative_accounted_seconds=prior+used(), saved_model_execution_reference=execution_ref,
                partial_outputs_preserved=True, scientific_acceptance=False, formal_OOS=False))
        raise
