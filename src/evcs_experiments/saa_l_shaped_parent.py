"""Typed L-shaped parent using the original queue and external wall ledger.

A numerical queue receipt and a terminal jobctl parent are separate evidence.
The latter settles only max(0, parent wall - new queue charges); no queue or
historical failed-parent receipt is edited. Formal dispatch requires its own
source-bound integrated admission, never a development flag.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from evcs_experiments import campaign_batch as batch
from evcs_experiments import saa_recovery_ledger as ledger
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import THREAD_VARS

KIND = 'SAA_L_SHAPED_PARENT_ADMISSION_V1'
REVIEW = 'SAA_L_SHAPED_PARENT_SOURCE_REVIEW_V1'
EXECUTION = 'SAA_L_SHAPED_PARENT_EXECUTION_V1'
TERMINAL_GATE = 'SAA_L_SHAPED_TERMINAL_PARENT_BUDGET_V1'
# The production worker / real Linux terminal verifier are not yet integrated.
# Opening this gate requires a reviewed source change and actual runtime proof.
FORMAL_DISPATCH_IMPLEMENTED = False

need, read, new = ledger.need, ledger.read, ledger.new


def actual_argv():
    path = Path('/proc/self/cmdline')
    return path.read_bytes().decode().rstrip('\0').split('\0') if path.exists() else [sys.executable, *sys.argv]


def parent_argv(reference):
    value = read(reference)
    return [value['original_python'], value['parent_source_reference']['path'],
            '--admission', reference['path'], '--expected-sha256', reference['sha256']]


def verify_admission(reference):
    """Verify only immutable identity; ledger head is checked before dispatch."""
    value = read(reference)
    need(value.get('kind') == KIND and value.get('domain') in ('development', 'formal'),
         'Wrong L-shaped parent admission kind/domain')
    need(value['parent_source_reference'] == batch.file_reference(__file__) and
         value['ledger_source_reference'] == batch.file_reference(ledger.__file__) and
         value['queue_source_reference'] == batch.file_reference(batch.__file__),
         'L-shaped parent/ledger/queue source differs')
    review = read(value['independent_review_reference'])
    need(review.get('kind') == REVIEW and review.get('passed') is True and
         review['admission_payload_sha256'] == canonical_hash({k: v for k, v in value.items()
                                                                if k != 'independent_review_reference'}),
         'L-shaped parent admission lacks exact independent review')
    queue, actual_queue_ref = batch.checked_request(value['queue_reference']['path'])
    need(actual_queue_ref == value['queue_reference'], 'Parent queue reference changed')
    worker = read(value['worker_request_reference'])
    need(queue['domain'] == worker['domain'] == value['domain'] and worker['method_id'] == 'SAA'
         and len(queue['items']) == 1 and queue['items'][0]['request_reference'] == value['worker_request_reference']
         and queue['items'][0]['plan_id'] == worker['plan_id'] == value['plan_id'],
         'Parent requires the same single SAA plan and worker')
    need(queue.get('parent_wall_policy') == 'verified_parent_remaining_v1',
         'L-shaped parent requires original verified wall policy')
    for key in ('parent_source_reference', 'ledger_source_reference', 'queue_source_reference'):
        need(value[key] in queue['source_references'] and value[key] in worker['planning_source_references'],
             'Parent source missing from original queue/worker closure')
    campaign = Path(value['campaign_root'])
    queue_root = Path(queue['queue_root'])
    job = campaign/'jobs'/value['parent_job_id']
    need(campaign.is_absolute() and campaign.resolve() == campaign and
         queue_root.resolve() == queue_root and queue_root.is_relative_to(campaign)
         and Path(value['parent_execution_path']) == job/'l_shaped_parent_execution.json',
         'Parent campaign/queue/execution path differs')
    need(value['original_python'] == sys.executable, 'Parent Python differs from admitted interpreter')
    need(Path(value['parent_job_request_path']).is_absolute() and Path(value['parent_job_request_path']).resolve().is_relative_to(campaign), 'Foreign parent request path')
    anchor = read(value['anchor_reference'])
    need(anchor['queue_reference'] == value['queue_reference'] and anchor['campaign_root'] == str(campaign)
         and anchor['plan_id'] == value['plan_id'] and anchor['ledger_source_reference'] == value['ledger_source_reference'],
         'Parent external ledger differs')
    need(ledger.number(value['opening_effective_seconds'], 'opening effective debt') ==
         ledger.number(value['opening_queue_seconds'], 'opening queue debt') +
         ledger.number(value['opening_extra_seconds'], 'opening external debt'), 'Parent opening arithmetic differs')
    opening = read(value['opening_queue_snapshot_reference'])
    need(opening['request_reference'] == value['queue_reference'] and not opening.get('active')
         and not opening.get('halted'), 'L-shaped successor queue must be ready, never a failed queue rearm')
    if value['domain'] == 'formal':
        need(FORMAL_DISPATCH_IMPLEMENTED, 'Formal L-shaped parent production worker and Linux terminal admission remain unimplemented')
        need(worker.get('model_input_mode') == 'l_shaped_formal', 'Formal decomposition worker required')
        need(anchor['budget_seconds'] == queue['items'][0]['plan_budget_seconds'] == 7200
             and queue['shutdown_reserve_seconds'] == 10, 'Formal original budget/seal differs')
        # This is an external integration decision, not a self-authored runtime flag.
        gate = read(value['formal_integration_reference'])
        need(gate.get('kind') == 'SAA_L_SHAPED_FORMAL_PARENT_INTEGRATION_V1'
             and gate.get('passed') is True and gate.get('actual_gurobi_calls', 0) > 0
             and gate.get('actual_linux_jobctl_parent') is True
             and gate['worker_request_reference'] == value['worker_request_reference']
             and gate['source_references'] == queue['source_references'],
             'Formal whole-parent integration has not been admitted')
        for key in ('native_execution_reference', 'physical_admission_reference', 'terminal_budget_reference'):
            batch.verify_reference(gate[key])
        need(worker.get('failure_carryover_reference') is not None, 'Formal original failure debt is required')
    else:
        need(worker.get('model_input_mode') == 'l_shaped_development' and
             value.get('formal_integration_reference') is None, 'Development parent domain changed')
    if worker.get('failure_carryover_reference') is not None:
        from evcs_experiments.saa_l_shaped_failure_debt import verify_failure_debt
        debt = verify_failure_debt(worker['failure_carryover_reference'], worker, str(queue_root))
        need(debt['plan_id'] == value['plan_id'] and debt['domain'] == value['domain']
             and value['opening_queue_seconds'] >= debt['accounted_seconds'], 'Original failure debt was reset')
    return value, queue, worker


def ledger_parent_contract(anchor, reservation, manifest, admission):
    """Called by the existing immutable ledger for this precise new kind."""
    value, queue, _ = verify_admission(reservation['admission_reference'])
    need(value == admission and value['anchor_reference'] == reservation['anchor_reference']
         and value['queue_reference'] == anchor['queue_reference'] and
         value['parent_job_id'] == reservation['job_id'], 'Foreign L-shaped ledger parent')
    need(value['opening_queue_snapshot_reference'] == reservation['opening_queue_snapshot_reference']
         and value['expected_last_settlement_reference'] == reservation['previous_settlement_reference']
         and value['opening_queue_seconds'] == reservation['opening_queue_seconds']
         and value['opening_extra_seconds'] == reservation['opening_extra_seconds']
         and value['opening_effective_seconds'] == reservation['effective_spent_seconds'],
         'L-shaped admission changed exact ledger head or prior debt')
    need(manifest['argv'] == parent_argv(reservation['admission_reference']), 'L-shaped parent actual argv differs')
    need(manifest['threads'] == 1, 'L-shaped parent outer thread allocation differs')
    timeout = ledger.number(manifest['timeout_seconds'], 'parent timeout')
    grace = ledger.number(manifest['grace_seconds'], 'parent grace')
    need(timeout > 0 and timeout <= queue['job_wall_seconds'] and
         timeout + grace <= anchor['budget_seconds'] - reservation['effective_spent_seconds'],
         'L-shaped parent timeout plus grace exceeds original remaining budget')
    if value['domain'] == 'formal':
        need(timeout <= 7195 and grace == 5, 'Formal parent timeout/grace differs')


def _unreserved_parent_jobs(value):
    """A failed admission still has a jobctl manifest; renaming cannot erase it.

    Such an early failure needs an explicitly reviewed terminal settlement. This
    driver does not fabricate a reservation against a contract that did not pass.
    """
    anchor = read(value['anchor_reference'])
    registered = {read(batch.file_reference(path))['job_id']
                  for path in Path(anchor['ledger_root']).glob('reservation-*.json')}
    missing = []
    for path in (Path(value['campaign_root'])/'jobs').glob('*/manifest.json'):
        manifest = batch.read_json(path)
        argv = manifest.get('argv', [])
        if '--admission' not in argv or '--expected-sha256' not in argv:
            continue
        try:
            candidate_path = Path(argv[argv.index('--admission')+1])
            expected_sha = argv[argv.index('--expected-sha256')+1]
        except IndexError as exc:
            raise ValueError('Malformed earlier parent admission argv') from exc
        if not candidate_path.is_file():
            # Only an exact same driver belongs to this attempt family.
            need(len(argv)<2 or argv[1] != value['parent_source_reference']['path'],
                 'Earlier L-shaped parent admission file is missing')
            continue
        candidate = batch.read_json(candidate_path)
        if candidate.get('kind') != KIND or candidate.get('plan_id') != value['plan_id']:
            continue
        ref = batch.file_reference(candidate_path)
        need(ref['sha256'] == expected_sha and candidate.get('campaign_root') == value['campaign_root'],
             'Earlier L-shaped parent identity was changed')
        if manifest['job_id'] != value['parent_job_id'] and manifest['job_id'] not in registered:
            missing.append(manifest['job_id'])
    need(not missing, 'Earlier unreserved parent expenses require settlement: '+str(sorted(missing)))


def run_parent(reference):
    value, queue, _ = verify_admission(reference)
    need(actual_argv() == parent_argv(reference), 'L-shaped running argv differs')
    need(all(os.environ.get(k) == '1' for k in THREAD_VARS), 'L-shaped parent BLAS environment differs')
    parent = batch._parent(queue)
    need(parent is not None and parent['job_id'] == value['parent_job_id'] and
         Path(parent['directory']) == Path(value['campaign_root'])/'jobs'/value['parent_job_id']
         and parent.get('wall_clock', {}).get('clock_kind') == 'linux_supervisor_birth_boottime_v1',
         'L-shaped parent requires actual original Linux supervisor clock')
    _unreserved_parent_jobs(value)
    anchor, extra, last, pending, _ = ledger._chain(value['anchor_reference'])
    need(pending is None and last == value['expected_last_settlement_reference'] and
         extra == value['opening_extra_seconds'], 'L-shaped parent ledger head differs or remains unsettled')
    reservation = ledger.reserve_recovery_parent(value['anchor_reference'],
        parent_job_request_reference=batch.file_reference(value['parent_job_request_path']),
        parent_manifest_reference=parent['manifest_reference'], admission_reference=reference,
        opening_queue_snapshot_reference=value['opening_queue_snapshot_reference'])
    # Durable reservation precedes queue validation, provider loading and solve.
    execution = new(value['parent_execution_path'], dict(kind=EXECUTION,
        admission_reference=reference, reservation_reference=reservation, parent_job=parent,
        actual_argv=actual_argv(), pid=os.getpid(), process_identity=batch.process_identity(os.getpid()),
        source_references={k: value[k] for k in ('parent_source_reference', 'ledger_source_reference', 'queue_source_reference')},
        formal_OOS=False, scientific_acceptance=False))
    result = batch.run_batch(value['queue_reference']['path'], queue['queue_root'])
    return dict(kind='SAA_L_SHAPED_QUEUE_PARENT_RETURN_V1', status=result['status'],
                queue_result=result, execution_reference=execution, reservation_reference=reservation,
                terminal_parent_settlement_required=True, scientific_acceptance=False)


def settle_parent(admission_reference, reservation_reference, terminal_reference, closing_snapshot_reference,
                  *, output_path):
    """After jobctl termination, settle actual wall and read original publication."""
    value, queue, worker = verify_admission(admission_reference)
    reservation = read(reservation_reference)
    need(reservation['admission_reference'] == admission_reference and
         reservation['anchor_reference'] == value['anchor_reference'], 'Terminal parent reservation differs')
    settlement_ref = ledger.settle_recovery_parent(reservation_reference, terminal_reference,
                                                   closing_snapshot_reference)
    settled = read(settlement_ref)
    anchor, _, last, pending, _ = ledger._chain(value['anchor_reference'])
    need(pending is None and last == settlement_ref, 'Terminal parent requires latest settled ledger')
    closing = read(closing_snapshot_reference)
    from evcs_experiments.canonical_in_attempt import verify_publication
    bindings = []
    for receipt_ref in settled['new_queue_receipt_references']:
        receipt = read(receipt_ref)
        if receipt['status'] != 'COMPLETE':
            continue
        final_refs = [ref for ref in closing.get('finalization_receipts', [])
                      if read(ref)['receipt_reference'] == receipt_ref]
        pub_refs = [ref for ref in closing.get('canonical_publication_receipts', [])
                    if read(ref)['receipt_reference'] == receipt_ref]
        need(len(final_refs) == len(pub_refs) == 1, 'Complete parent candidate lacks original seal/publication')
        verify_publication(pub_refs[0], receipt_reference=receipt_ref,
                           finalization_reference=final_refs[0], worker_request_reference=value['worker_request_reference'])
        bindings.append(dict(receipt_reference=receipt_ref, finalization_reference=final_refs[0],
                             canonical_publication_reference=pub_refs[0]))
    terminal = read(terminal_reference)
    passed = (settled['budget_within_limit'] and not closing.get('active') and not closing.get('halted')
              and terminal['state'] == 'SUCCEEDED' and terminal.get('returncode') == 0)
    card = dict(kind=TERMINAL_GATE, domain=value['domain'], admission_reference=admission_reference,
        settlement_reference=settlement_ref, complete_candidate_bindings=bindings,
        effective_spent_seconds=settled['effective_spent_seconds'],
        remaining_seconds=anchor['budget_seconds']-settled['effective_spent_seconds'],
        passed=passed, status=('SETTLED_COMPLETE_CANDIDATE' if bindings else 'SETTLED_NO_COMPLETE_CANDIDATE')
             if passed else 'HOLD_FAILED_OR_OVER_BUDGET_PARENT', scientific_acceptance=False, formal_OOS=False)
    path = Path(output_path)
    need(path.is_absolute() and path.resolve().is_relative_to(Path(value['campaign_root'])), 'Foreign terminal budget card')
    if path.exists():
        need(read(batch.file_reference(path)) == card, 'Terminal parent card already differs')
        return batch.file_reference(path)
    return new(path, card)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    reference = batch.file_reference(args.admission)
    need(reference['sha256'] == args.expected_sha256, 'Parent admission SHA changed')
    result = run_parent(reference)
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] in ('COMPLETE_BATCH', 'TIME_LIMIT', 'BUDGET_EXHAUSTED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
