"""Explicit source-bound recovery for one pending OR in its original EA/OR queue.

No solver, input, request, checkpoint, completed EA, or original fee is replaced.
The only runtime substitutions are the reviewed checkpoint validator, explicit
child dispatch/result provenance, and one queue start-clock statement. Terminal
job-cost coverage is a separate required gate, never inferred from exit zero.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace

from evcs_experiments import campaign_batch as batch
from evcs_experiments import formal_planning_worker as worker
from evcs_experiments.random_streams import canonical_hash

KIND = 'FORMAL_VALIDATOR_RECOVERY_ADMISSION_V1'
REVIEW = 'FORMAL_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1'
RECEIPT = 'FORMAL_VALIDATOR_RECOVERY_EXECUTION_V1'
CLOCK = 'FORMAL_RECOVERY_SUPERVISOR_BIRTH_ACCOUNTING_V1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(ref):
    batch.verify_reference(ref)
    return batch.read_json(ref['path'])


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return batch.file_reference(path)


def actual_argv():
    path = Path('/proc/self/cmdline')
    return path.read_bytes().decode().rstrip('\0').split('\0') if path.exists() else [sys.executable, *sys.argv]


def parent_argv(value, ref):
    return [value['original_python'], value['adapter_reference']['path'], 'parent',
            '--admission', ref['path'], '--expected-sha256', ref['sha256']]


def _reviewed(value):
    need(value.get('kind') == KIND and value.get('domain') in ('formal', 'development'),
         'Wrong ordinary recovery admission domain')
    need(value['adapter_reference'] == batch.file_reference(__file__), 'Recovery adapter source differs')
    review = read(value['independent_review_reference'])
    need(review.get('kind') == REVIEW and review.get('passed') is True and
         review.get('admission_payload_sha256') == canonical_hash({k: v for k, v in value.items() if k != 'independent_review_reference'}),
         'Recovery admission was not independently reviewed')
    need(review.get('adapter_reference') == value['adapter_reference'] and
         review.get('validator_reference') == value['validator_reference'], 'Recovery reviewed source differs')
    if value['domain'] == 'formal':
        need('development_entry_reference' not in value, 'Formal recovery cannot substitute a development worker')
    elif 'development_entry_reference' in value:
        batch.verify_reference(value['development_entry_reference'])


def admission(ref):
    """Read-only source and frozen-opening admission; does not consume a job."""
    value = read(ref)
    _reviewed(value)
    for key in ('queue_reference', 'worker_request_reference', 'original_release_reference',
                'validator_reference', 'baseline_ledger_reference'):
        batch.verify_reference(value[key])
    queue, queue_ref = batch.checked_request(value['queue_reference']['path'])
    need(queue_ref == value['queue_reference'] and queue['domain'] == value['domain'] and
         queue.get('parent_wall_policy') == 'verified_parent_remaining_v1', 'Original queue domain/accounting differs')
    need(Path(queue['queue_root']).resolve() == Path(value['queue_root']).resolve() and len(queue['items']) == 2,
         'Recovery requires the original two-item EA/OR queue')
    ea, pending = queue['items']
    ea_request, request = read(ea['request_reference']), read(pending['request_reference'])
    need(ea_request['method_id'] == 'EA' and request['method_id'] == 'OR' and
         ea_request['domain'] == request['domain'] == value['domain'], 'Recovery supports ordinary EA then OR only')
    need(pending['request_reference'] == value['worker_request_reference'] and
         pending['plan_id'] == value['plan_id'] == request['plan_id'] and
         pending['item_id'] == value['pending_item_id'] and
         value['completed_item_ids'] == [ea['item_id']], 'Recovery item/plan differs')
    need(ea['plan_id'] != pending['plan_id'], 'EA and OR need separate original plan budgets')
    need(pending['plan_budget_seconds'] == request['budget']['cumulative_wall_seconds'], 'Original OR budget differs')
    release = read(value['original_release_reference'])
    release_root = Path(value['original_release_reference']['path']).parent
    inventory = {row['path']: row['sha256'] for row in release['files']}
    need(len(inventory) == len(release['files']), 'Duplicate original release source')
    modules = dict(campaign_batch=batch, formal_planning_worker=worker)
    need(set(value['original_module_references']) == set(modules), 'Original module source scope differs')
    for name, module in modules.items():
        source = value['original_module_references'][name]
        batch.verify_reference(source)
        need(source == batch.file_reference(module.__file__), 'Installed original package differs')
        relative = Path(source['path']).relative_to(release_root).as_posix()
        need(inventory.get(relative) == source['sha256'], 'Original module absent from frozen release')
    need(value['original_python'] == sys.executable, 'Original interpreter differs')
    if value['domain'] == 'formal':
        need(not Path(value['adapter_reference']['path']).is_relative_to(Path(worker.__file__).parent) and
             not Path(value['validator_reference']['path']).is_relative_to(Path(worker.__file__).parent),
             'Added recovery sources cannot mutate the frozen numerical package')
        refs = request['planning_source_references']
        need(refs == [batch.file_reference(p) for p in sorted(Path(worker.__file__).parent.rglob('*.py'))],
             'Original full numerical package differs')
    opening = read(value['baseline_ledger_reference'])
    need(opening['request_reference'] == queue_ref and not opening.get('active') and not opening.get('halted') and
         opening['attempt_receipts'] == value['baseline_receipt_references'], 'Original baseline state differs')
    baseline = [read(r)['result_reference'] for r in opening['attempt_receipts']]
    need(baseline == value['baseline_result_references'] and len(baseline) == 2,
         'Exactly the two original baseline result exemptions are required')
    latest, spent = batch._history(opening, queue_ref, opening=batch._carryovers(queue))
    need(set(latest) == {ea['item_id'], pending['item_id']} and
         latest[ea['item_id']]['result']['status'] == 'COMPLETE' and
         latest[pending['item_id']]['result']['status'] == 'TIME_LIMIT', 'Original completed EA/pending OR differs')
    need(latest[pending['item_id']]['result'].get('checkpoint') is not None, 'Original OR checkpoint missing')
    prior = value['opening_cumulative_seconds']
    need(type(prior) in (int, float) and math.isfinite(prior) and 0 < prior < pending['plan_budget_seconds'] and
         spent.get(value['plan_id']) == prior, 'Recovery budget opening differs or was reset')
    # This entry is for the observed zero-engine OR interruption, not general recovery.
    checkpoint = read(latest[pending['item_id']]['result']['checkpoint'])
    need(checkpoint.get('resume_without_engine_checkpoint') is True, 'OR opening is not the admitted zero-engine checkpoint')
    return value, queue


def _derived_queue_source(original_source_reference):
    batch.verify_reference(original_source_reference)
    text = Path(original_source_reference['path']).read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'run_batch']
    need(len(nodes) == 1 and not nodes[0].decorator_list, 'Original queue function scope differs')
    node = nodes[0]
    source = ''.join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    old = 'started=time.monotonic();root=Path(outputroot).resolve();root.mkdir(parents=True,exist_ok=True)'
    replacement = 'started=_explicit_recovery_parent_start();root=Path(outputroot).resolve();root.mkdir(parents=True,exist_ok=True)'
    need(source.count(old) == 1, 'Original queue start-clock schema differs')
    derived = source.replace(old, replacement)
    record = dict(kind=CLOCK, original_queue_reference=original_source_reference,
                  adapter_reference=batch.file_reference(__file__), replacement_count=1,
                  original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  derived_function_sha256=hashlib.sha256(derived.encode()).hexdigest(),
                  original_receipts_budgets_and_limits_unchanged=True)
    return source, derived, record


def expected_clock_binding(original_source_reference):
    return _derived_queue_source(original_source_reference)[2]


def bind_original_queue_clock(original_source_reference, clock):
    """Only the first queue clock statement changes; no global clock or fee edits."""
    source, derived, record = _derived_queue_source(original_source_reference)
    original = batch.run_batch
    need(batch.file_reference(batch.__file__) == original_source_reference and
         original.__globals__ is batch.__dict__ and original.__code__.co_filename == batch.__file__ and
         inspect.getsource(original) == source, 'Original queue function changed or is already replaced')
    need(clock.get('clock_kind') == 'linux_supervisor_birth_boottime_v1', 'Verified Linux supervisor birth required')
    start = clock['observed_monotonic'] - clock['elapsed_seconds']
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (start, clock['elapsed_seconds'])) and
         clock['elapsed_seconds'] >= 0 and start <= time.monotonic(), 'Invalid conservative parent start')
    key = '_explicit_recovery_parent_start'
    need(key not in batch.__dict__, 'Recovery queue clock already installed')
    namespace = {}
    exec(compile(derived, __file__ + '::derived_original_run_batch', 'exec'), namespace)
    function = FunctionType(namespace['run_batch'].__code__, batch.__dict__, original.__name__,
                            original.__defaults__, original.__closure__)
    used = False
    def original_parent_start():
        nonlocal used
        need(not used, 'One admitted parent permits one queue invocation')
        used = True
        return start
    batch.__dict__[key] = original_parent_start
    batch.run_batch = function
    def restore():
        batch.run_batch = original
        batch.__dict__.pop(key, None)
    return restore, dict(**record, observed_parent_clock=clock, accounting_start_monotonic=start)


def install_validator(value):
    """Bind only formal_planning_worker._closure to the reviewed added source."""
    batch.verify_reference(value['validator_reference'])
    name = 'evcs_experiments._explicit_formal_recovery_checkpoint_closure'
    need(name not in sys.modules, 'Validator already installed in this process')
    spec = importlib.util.spec_from_file_location(name, value['validator_reference']['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        need(module.file_reference is batch.file_reference and module.verify_reference is batch.verify_reference,
             'Validator did not bind original campaign helpers')
    except BaseException:
        sys.modules.pop(name, None)
        raise
    original = worker._closure
    worker._closure = module.verify_checkpoint_closure
    def restore():
        worker._closure = original
        sys.modules.pop(name, None)
    return restore


def verify_recovery_preflight(ref):
    """Actual checkpoint closure with the admitted helper; no solve or queue write."""
    value, queue = admission(ref)
    opening = read(value['baseline_ledger_reference'])
    checkpoint = read(read(opening['attempt_receipts'][1])['result_reference'])['checkpoint']
    restore = install_validator(value)
    try:
        seen = set()
        worker._closure(checkpoint, seen)
    finally:
        restore()
    return dict(kind='FORMAL_VALIDATOR_RECOVERY_PREFLIGHT_V1', admission_reference=ref,
                checkpoint_reference=checkpoint, verified_reference_count=len(seen), passed=True,
                solver_called=False, production_execution_admitted=False,
                clock_binding=expected_clock_binding(value['original_module_references']['campaign_batch']))


def child_argv(value, ref, parent_ref):
    entry = value.get('development_entry_reference', value['adapter_reference'])
    if 'development_entry_reference' in value:
        need(value['domain'] == 'development', 'Formal recovery cannot substitute a development worker')
        batch.verify_reference(entry)
    return [value['original_python'], entry['path'], 'child', '--admission', ref['path'],
            '--expected-sha256', ref['sha256'], '--parent-execution', parent_ref['path'],
            '--parent-sha256', parent_ref['sha256']]


def _verify_opening_claim(value, ref, parent):
    manifest = read(parent['parent_job']['manifest_reference'])
    claim_ref = parent.get('opening_claim_reference')
    need(claim_ref is not None and claim_ref == batch.file_reference(opening_claim_path(value, manifest)),
         'Missing exact one-use opening claim')
    claim = read(claim_ref)
    need(claim.get('kind') == 'FORMAL_RECOVERY_ONE_USE_OPENING_CLAIM_V1' and
         claim['admission_reference'] == ref and claim['parent_job'] == parent['parent_job'] and
         claim['queue_reference'] == value['queue_reference'] and
         claim['baseline_receipt_references'] == value['baseline_receipt_references'] and
         claim['opening_cumulative_seconds'] == value['opening_cumulative_seconds'], 'Opening claim binding differs')
    return claim_ref


def validate_execution_result(result_ref, invocation_ref, *, baseline_result_references=()):
    result, inv = read(result_ref), read(invocation_ref)
    need(not baseline_result_references, 'Baseline exemptions are internal to the admitted original queue')
    receipt_ref = result.get('validation_recovery_execution_reference')
    need(receipt_ref is not None and receipt_ref in result.get('artifacts', []),
         'New worker omitted recovery execution receipt')
    receipt = read(receipt_ref)
    value, queue = admission(receipt['admission_reference'])
    parent = read(receipt['parent_execution_reference'])
    _verify_opening_claim(value, receipt['admission_reference'], parent)
    need(receipt['kind'] == parent['kind'] == RECEIPT and receipt['role'] == 'child' and parent['role'] == 'parent',
         'Wrong execution receipt role')
    need(parent['admission_reference'] == receipt['admission_reference'] and
         parent['queue_reference'] == inv['request_reference'] == value['queue_reference'] and
         inv['item_request_reference'] == value['worker_request_reference'] and inv['plan_id'] == value['plan_id'],
         'Recovery invocation/queue/plan differs')
    need(receipt['invocation_reference'] == invocation_ref and receipt['parent_job'] == inv['parent_job'] and
         parent['parent_job']['job_id'] == inv['parent_job']['job_id'] and
         parent['parent_job']['manifest_reference'] == inv['parent_job']['manifest_reference'],
         'Recovery parent job differs')
    need(receipt['observed_parent_identity'] == parent['process_identity'] and
         parent['pid'] == parent['process_identity']['pid'] and receipt['ppid'] == parent['pid'],
         'Receipt parent process identity differs')
    need(read(parent['parent_job']['manifest_reference'])['argv'] == parent['actual_argv'] ==
         parent_argv(value, receipt['admission_reference']) and
         receipt['actual_argv'] == child_argv(value, receipt['admission_reference'], receipt['parent_execution_reference']) and
         receipt['declared_worker_argv'] == inv['argv'] == queue['items'][1]['argv'], 'Recovery actual argv differs')
    need(receipt['adapter_reference'] == value['adapter_reference'] and receipt['validator_reference'] == value['validator_reference'] and
         receipt['original_module_references'] == value['original_module_references'], 'Recovery execution source differs')
    binding = parent['queue_clock_execution']
    expected = expected_clock_binding(value['original_module_references']['campaign_batch'])
    need({k: binding[k] for k in expected} == expected and
         binding['accounting_start_monotonic'] == binding['observed_parent_clock']['observed_monotonic'] - binding['observed_parent_clock']['elapsed_seconds'] and
         inv['attempt_started_monotonic'] == binding['accounting_start_monotonic'], 'Missing exact queue clock execution provenance')
    need(inv['cumulative_plan_seconds'] == value['opening_cumulative_seconds'] and
         inv['plan_remaining_seconds'] == inv['plan_budget_seconds'] - value['opening_cumulative_seconds'],
         'Recovery invocation reset original budget')
    return result


def run_child(ref, parent_ref, *, _development_fixture=None):
    value, queue = admission(ref)
    need(value['domain'] == 'development' or _development_fixture is None, 'Formal recovery cannot inject physical fixtures')
    parent = read(parent_ref)
    _verify_opening_claim(value, ref, parent)
    invref = batch.file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
    need(invref['sha256'] == os.environ['EVCS_BATCH_INVOCATION_SHA256'], 'Recovery invocation hash differs')
    inv = read(invref)
    need(parent['admission_reference'] == ref and parent['queue_reference'] == value['queue_reference'] == inv['request_reference'] and
         inv['item_request_reference'] == value['worker_request_reference'] and inv['plan_id'] == value['plan_id'] and
         inv['item_id'] == value['pending_item_id'], 'Wrong recovery invocation/queue')
    need(parent['parent_job']['job_id'] == inv['parent_job']['job_id'] and
         parent['parent_job']['manifest_reference'] == inv['parent_job']['manifest_reference'], 'Wrong recovery parent')
    need(actual_argv() == child_argv(value, ref, parent_ref), 'Child was not started by explicit recovery argv')
    need(os.getppid() == parent['pid'], 'Recovery child parent process differs')
    identity = batch.process_identity(os.getppid())
    need(identity is not None and all(identity[k] == parent['process_identity'][k] for k in ('pid', 'token', 'sid', 'pgid')),
         'Recovery live parent birth identity differs')
    need(inv['cumulative_plan_seconds'] == value['opening_cumulative_seconds'] and
         inv['attempt_started_monotonic'] == parent['queue_clock_execution']['accounting_start_monotonic'],
         'Recovery invocation opening or accounting clock differs')
    restore = install_validator(value)
    original_new = worker._new
    receipt = new(Path(inv['attempt_directory']) / 'validation_recovery_execution.json', dict(
        kind=RECEIPT, role='child', admission_reference=ref, parent_execution_reference=parent_ref,
        invocation_reference=invref, parent_job=inv['parent_job'], declared_worker_argv=inv['argv'], actual_argv=actual_argv(),
        adapter_reference=value['adapter_reference'], validator_reference=value['validator_reference'],
        original_module_references=value['original_module_references'], pid=os.getpid(), ppid=os.getppid(),
        process_identity=batch.process_identity(os.getpid()), observed_parent_identity=parent['process_identity']))
    def first_write(path, data):
        if Path(path).resolve() == Path(inv['result_path']).resolve():
            need('validation_recovery_execution_reference' not in data, 'Worker result provenance already assigned')
            data = dict(data, validation_recovery_execution_reference=receipt, artifacts=[*data['artifacts'], receipt])
        elif Path(path).name in ('candidate_solver_record.json', 'worker_checkpoint.json'):
            data = dict(data, validation_recovery_execution_reference=receipt)
        return original_new(path, data)
    worker._new = first_write
    try:
        worker.run_attempt(invref, _development_fixture=_development_fixture)
        return validate_execution_result(batch.file_reference(inv['result_path']), invref)
    finally:
        worker._new = original_new
        restore()


def opening_claim_path(value, parent_manifest):
    """A fixed opening may be consumed once, independent of admission/job names."""
    original_receipt = read(value['baseline_receipt_references'][1])
    original_invocation = read(original_receipt['invocation_reference'])
    original_parent = original_invocation['parent_job']
    original_manifest = read(original_parent['manifest_reference'])
    if value['domain'] == 'formal':
        need(parent_manifest['outroot'] == original_manifest['outroot'], 'Recovery cannot move original job-control root')
        outroot = Path(original_manifest['outroot'])
    else:
        outroot = Path(parent_manifest.get('outroot', str(Path(value['parent_job_directory']).parent)))
    need(outroot.is_absolute() and outroot.resolve() == outroot and
         Path(value['parent_job_directory']) == outroot / value['parent_job_id'], 'Recovery job-control root differs')
    key = canonical_hash(dict(queue_reference=value['queue_reference'],
                              baseline_receipt_references=value['baseline_receipt_references']))
    return outroot / '.formal-validator-recovery-claims' / (key + '.json')


def run_parent(ref):
    """One job/attempt, with original supervisor-birth-to-seal OR accounting."""
    value = read(ref)
    _reviewed(value)
    queue = read(value['queue_reference'])
    need(actual_argv() == parent_argv(value, ref), 'Parent actual argv differs')
    parent = batch._parent(queue)
    need(parent is not None and parent['job_id'] == value['parent_job_id'] and
         parent['directory'] == value['parent_job_directory'], 'Foreign recovery jobctl parent')
    manifest = read(parent['manifest_reference'])
    need(manifest['argv'] == actual_argv(), 'Actual job manifest argv differs')
    budget = read(value['worker_request_reference'])['budget']['cumulative_wall_seconds']
    need(manifest['timeout_seconds'] + manifest.get('grace_seconds', 5) <= budget - value['opening_cumulative_seconds'],
         'Parent timeout and cleanup exceed original OR remainder')
    claim_path = opening_claim_path(value, manifest)
    claim_path.parent.mkdir(exist_ok=True)
    claim_ref = new(claim_path, dict(kind='FORMAL_RECOVERY_ONE_USE_OPENING_CLAIM_V1',
        admission_reference=ref, parent_job=parent, queue_reference=value['queue_reference'],
        baseline_receipt_references=value['baseline_receipt_references'],
        opening_cumulative_seconds=value['opening_cumulative_seconds'],
        terminal_parent_settlement_required=True, automatic_resume_authorized=False))
    clock_restore, clock_record = bind_original_queue_clock(value['original_module_references']['campaign_batch'], parent['wall_clock'])
    original_checked, original_result, original_subprocess = batch.checked_request, batch._result, batch.subprocess
    holder = {}
    try:
        # Exclusive creation is the one-use reservation, retained even if admission fails.
        parent_ref = new(value['parent_execution_path'], dict(kind=RECEIPT, role='parent', admission_reference=ref,
            queue_reference=value['queue_reference'], parent_job=parent, actual_argv=actual_argv(), pid=os.getpid(),
            process_identity=batch.process_identity(os.getpid()), queue_clock_execution=clock_record, opening_claim_reference=claim_ref,
            opening_cumulative_seconds=value['opening_cumulative_seconds'],
            terminal_parent_settlement_required=True, automatic_resume_authorized=False))
        def checked(path):
            if not holder:
                # Avoid recursion through admission -> checked_request.
                batch.checked_request = original_checked
                try:
                    admitted, frozen_queue = admission(ref)
                finally:
                    batch.checked_request = checked
                need(Path(path).resolve() == Path(admitted['queue_reference']['path']).resolve(), 'Foreign recovery queue')
                state = batch.read_json(Path(frozen_queue['queue_root']) / 'state.json')
                need(state == read(admitted['baseline_ledger_reference']), 'Original queue history changed since admission')
                need(not state.get('active') and not state.get('halted'), 'Prior active/unsettled queue blocks recovery')
                holder.update(value=admitted, restore=install_validator(admitted))
            return original_checked(path)
        def dispatch(argv, *args, **kwargs):
            need(holder and argv == queue['items'][1]['argv'], 'Unexpected original worker dispatch; completed EA cannot rerun')
            return original_subprocess.Popen(child_argv(value, ref, parent_ref), *args, **kwargs)
        def result(result_ref, invocation_ref):
            answer = original_result(result_ref, invocation_ref)
            if holder:
                if result_ref not in value['baseline_result_references']:
                    validate_execution_result(result_ref, invocation_ref)
            return answer
        batch.checked_request, batch._result = checked, result
        batch.subprocess = SimpleNamespace(**{**vars(original_subprocess), 'Popen': dispatch})
        output = batch.run_batch(value['queue_reference']['path'], value['queue_root'])
        increment = output['cumulative_plan_seconds'][value['plan_id']] - value['opening_cumulative_seconds']
        elapsed = time.monotonic() - clock_record['accounting_start_monotonic']
        coverage = new(value['parent_coverage_path'], dict(kind='FORMAL_RECOVERY_PARENT_PRESEAL_COVERAGE_V1',
            admission_reference=ref, parent_execution_reference=parent_ref, elapsed_to_preseal_seconds=elapsed,
            queue_charged_increment_seconds=increment, covered_through_preseal=elapsed <= increment,
            terminal_parent_settlement_required=True, automatic_resume_authorized=False))
        return dict(queue_result=output, parent_coverage_reference=coverage,
                    terminal_parent_settlement_required=True, automatic_resume_authorized=False)
    finally:
        batch.checked_request, batch._result, batch.subprocess = original_checked, original_result, original_subprocess
        if holder:
            holder['restore']()
        clock_restore()


def validate_terminal_coverage(ref, terminal_job_result_reference, closing_queue_snapshot_reference, *, output_path):
    """Require real terminal job wall to fit the unchanged OR charge, or HOLD."""
    value, queue = admission(ref)
    terminal = read(terminal_job_result_reference)
    parent_ref = batch.file_reference(value['parent_execution_path'])
    parent = read(parent_ref)
    closing = read(closing_queue_snapshot_reference)
    need(terminal['job_id'] == value['parent_job_id'] == parent['parent_job']['job_id'], 'Terminal parent job differs')
    need(parent['admission_reference'] == ref and parent['queue_reference'] == value['queue_reference'], 'Terminal admission binding differs')
    need(not closing.get('active') and not closing.get('halted'), 'Unsettled queue cannot be accepted')
    need(closing['attempt_receipts'][:len(value['baseline_receipt_references'])] == value['baseline_receipt_references'],
         'Terminal baseline receipts changed')
    additions = closing['attempt_receipts'][len(value['baseline_receipt_references']):]
    need(len(additions) == 1, 'Exactly one recovered OR receipt is required; missing work needs fee settlement')
    receipt = read(additions[0])
    need(receipt['item_id'] == value['pending_item_id'] and receipt['plan_id'] == value['plan_id'], 'Unexpected terminal item')
    validate_execution_result(receipt['result_reference'], receipt['invocation_reference'])
    need(closing == batch.read_json(Path(value['queue_root']) / 'state.json'), 'Terminal queue snapshot differs from actual state')
    latest, spent = batch._history(closing, value['queue_reference'], Path(value['queue_root']), opening=batch._carryovers(queue))
    charged = spent[value['plan_id']] - value['opening_cumulative_seconds']
    elapsed = terminal['elapsed_seconds']
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'Invalid terminal elapsed cost')
    need(terminal_job_result_reference['path'] == str(Path(value['parent_job_directory']) / 'result.json'), 'Terminal result path differs')
    need(parent['parent_job']['manifest_reference']['path'] == str(Path(value['parent_job_directory']) / 'manifest.json'), 'Terminal manifest path differs')
    manifest = read(parent['parent_job']['manifest_reference'])
    need(manifest['argv'] == parent['actual_argv'] == parent_argv(value, ref), 'Terminal parent manifest argv differs')
    claim_ref = _verify_opening_claim(value, ref, parent)
    original = read(value['baseline_ledger_reference'])
    for key in ('finalization_receipts', 'canonical_publication_receipts'):
        before = original.get(key, [])
        need(closing.get(key, [])[:len(before)] == before, 'Original EA final/canonical evidence changed')
    need(charged > 0 and spent[value['plan_id']] <= queue['items'][1]['plan_budget_seconds'], 'Terminal original cumulative budget exceeded')
    from evcs_experiments.runtime.jobctl import observation
    status_path = Path(value['parent_job_directory']) / 'status.json'
    need(batch.read_json(status_path) == terminal, 'Terminal result and current job status differ')
    observed = observation(terminal)
    quiet = not observed['blocks_new_launch'] and not observed['supervisor_alive'] and not observed['command_alive']
    passed = quiet and elapsed <= charged and terminal.get('complete') is True and terminal.get('returncode') == 0 and terminal.get('state') == 'SUCCEEDED'
    return new(output_path, dict(kind='FORMAL_RECOVERY_TERMINAL_COST_COVERAGE_V1', admission_reference=ref,
        parent_execution_reference=parent_ref, terminal_job_result_reference=terminal_job_result_reference,
        closing_queue_snapshot_reference=closing_queue_snapshot_reference, recovered_receipt_reference=additions[0],
        original_cumulative_opening_seconds=value['opening_cumulative_seconds'], queue_charged_increment_seconds=charged,
        original_cumulative_closing_seconds=spent[value['plan_id']], actual_parent_elapsed_seconds=elapsed,
        uncovered_parent_seconds=max(0., elapsed - charged), passed=passed, terminal_process_observation=observed,
        status='TERMINAL_OR_COST_COVERED' if passed else ('HOLD_UNSETTLED_PARENT_COST' if quiet else 'HOLD_TERMINAL_JOB_NOT_QUIET'),
        automatic_resume_authorized=False, scientific_acceptance=False))


def main(*, _development_fixture=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('parent', 'child', 'preflight'))
    parser.add_argument('--admission', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--parent-execution')
    parser.add_argument('--parent-sha256')
    args = parser.parse_args()
    ref = batch.file_reference(args.admission)
    need(ref['sha256'] == args.expected_sha256, 'Admission hash differs')
    if args.role == 'parent':
        answer = run_parent(ref)
    elif args.role == 'preflight':
        answer = verify_recovery_preflight(ref)
    else:
        parent = batch.file_reference(args.parent_execution)
        need(parent['sha256'] == args.parent_sha256, 'Parent execution hash differs')
        answer = run_child(ref, parent, _development_fixture=_development_fixture)
    print(json.dumps(answer, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
