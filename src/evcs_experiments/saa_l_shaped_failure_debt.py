"""Exact two-MEM_LIMIT debt carried into one fresh SAA decomposition queue.

This is an accounting adapter, not numerical or launch admission. Old failed
queues and old expense records stay immutable. The predecessor's failed
``passed=False`` expense is deliberately retained, not converted to success.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import campaign_batch as batch
from . import saa_saved_model_expense as original_expense
from .random_streams import canonical_hash

KIND = 'SAA_L_SHAPED_FAILED_PARENTS_CARRYOVER_V1'
REVIEW_KIND = 'SAA_L_SHAPED_FAILED_PARENTS_CARRYOVER_REVIEW_V1'
CLAIM_KIND = 'SAA_L_SHAPED_FAILED_PARENTS_EXCLUSIVE_CLAIM_V1'
HELENE_REVIEW_SHA256 = '384a8a4f49405dabc7b4928db0e0463f41eb14d75c7e9c1e50f0f62079c362c5'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _read(ref):
    batch.verify_reference(ref)
    return batch.read_json(ref['path'])


def _number(value):
    _need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
          'Invalid failed-parent expense')
    return value


def worker_identity(worker):
    """Exclude only the backward pointer to this envelope, avoiding a cycle."""
    return canonical_hash({k: v for k, v in worker.items()
                           if k != 'failure_carryover_reference'})


def verify_predecessor(failure_review_reference, terminal_adoption_reference):
    """Read accepted historical facts and recheck both quarantined live bytes."""
    f = _read(failure_review_reference)
    a = _read(terminal_adoption_reference)
    _need(f['kind'] == 'SAA_SAVED_MODEL_FAILED_PARENT_FEE_AND_CAUSE_REVIEW_V1'
          and f['passed'] is True
          and f['status'] == 'PASS_FAILED_PARENT_EXPENSE_SETTLED_NO_NUMERICAL_RESULT',
          'Failed-parent accounting review is not accepted')
    _need(a['kind'] == 'SAA_SECOND_MEMORY_FAILURE_TERMINAL_ADOPTION_V1'
          and a['failure_accounting_review_reference'] == failure_review_reference
          and a['recovery_queue_rearm_prohibited'] is True
          and a['repeat_same_request_prohibited'] is True
          and a['formal_SAA_algorithm_admitted'] is False,
          'Second failed queue is not explicitly quarantined')
    # These are the recorded accounting dependencies, not a recursive demand to
    # reread every historical numerical matrix or relabel a historical test.
    for ref in f['required_references']:
        batch.verify_reference(ref)
    for value in f.values():
        if isinstance(value, dict) and {'path', 'sha256', 'size_bytes'} <= value.keys():
            batch.verify_reference(value)
    old = _read(f['original_worker_reference'])
    opening = original_expense.verify_opening(
        f['original_failed_fee_reference'], f['original_isolation_decision_reference'],
        f['original_worker_reference'])
    expense = _read(f['terminal_expense_reference'])
    terminal = _read(f['terminal_job_result_reference'])
    reservation = _read(f['reservation_reference'])
    admission = _read(reservation['admission_reference'])
    queue = _read(f['recovery_queue_reference'])
    closing = _read(f['closing_queue_snapshot_reference'])
    original_fee = _read(f['original_failed_fee_reference'])
    original_queue = _read(original_fee['original_queue_reference'])
    original_terminal = _read(original_fee['terminal_job_result_reference'])
    _need(expense['kind'] == 'SAA_SAVED_MODEL_RECOVERY_TERMINAL_EXPENSE_V1'
          and expense['passed'] is False
          and expense['status'] == 'HOLD_FAILED_OR_UNCOVERED_PARENT_EXPENSE_RETAINED'
          and expense['automatic_resume_authorized'] is False,
          'Underlying failed expense must retain its native failure')
    for key, other in [('reservation_reference', 'reservation_reference'),
                       ('terminal_job_result_reference', 'terminal_job_result_reference'),
                       ('closing_snapshot_reference', 'closing_queue_snapshot_reference'),
                       ('original_failed_parent_expense_reference', 'original_failed_fee_reference')]:
        _need(expense[key] == f[other], 'Second expense dependency changed')
    _need(reservation['opening'] == opening
          and admission['queue_reference'] == f['recovery_queue_reference']
          and admission['worker_request_reference'] == f['recovery_worker_reference']
          and reservation['queue_reference'] == f['recovery_queue_reference']
          and reservation['worker_request_reference'] == f['recovery_worker_reference']
          and terminal['job_id'] == reservation['parent_job']['job_id']
          and terminal['job_id'] != original_terminal['job_id']
          and terminal['complete'] is True and terminal['state'] == 'FAILED'
          and terminal['returncode'] == 2,
          'Second parent is not a distinct original failed execution')
    campaign = Path(admission['campaign_root'])
    _need(campaign.is_absolute() and campaign.resolve() == campaign
          and all(Path(p).is_relative_to(campaign) for p in (
              original_queue['queue_root'], queue['queue_root'],
              reservation['parent_job']['directory'])),
          'Predecessor campaign or original queues/job changed')
    _need(len(queue['items']) == 1
          and queue['items'][0]['plan_id'] == opening['plan_id']
          and queue['items'][0]['request_reference'] == f['recovery_worker_reference']
          and closing['request_reference'] == f['recovery_queue_reference']
          and closing['attempt_receipts'] == []
          and closing.get('halted') == 'VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW'
          and closing == batch.read_json(Path(queue['queue_root']) / 'state.json'),
          'Failed recovery queue changed or was rearmed')
    first = _number(opening['accounted_seconds'])
    wall = _number(terminal['elapsed_seconds'])
    total = first + wall
    cap = _number(opening['budget_seconds'])
    _need(expense['original_spent_seconds'] == first
          and expense['new_queue_charge_seconds'] == 0
          and expense['actual_parent_seconds'] == wall
          and expense['new_accounted_seconds'] == wall
          and expense['uncovered_parent_seconds'] == wall
          and math.isclose(expense['cumulative_spent_seconds'], total, rel_tol=0, abs_tol=1e-9)
          and math.isclose(expense['remaining_seconds'], cap-total, rel_tol=0, abs_tol=1e-9)
          and expense['budget_within_limit'] is True and 0 < total < cap,
          'Failed-parent costs reset or double counted')
    _need(f['original_accounted_seconds'] == first
          and f['new_actual_parent_seconds'] == wall
          and f['cumulative_accounted_seconds'] == expense['cumulative_spent_seconds']
          and f['remaining_original_seconds'] == expense['remaining_seconds']
          and f['parent_cost_fully_retained'] is True and f['receipt_count'] == 0
          and f['numerical_acceptance'] is False and f['raw_native_status'] == 17
          and f['native_status'] == 'MEM_LIMIT'
          and f['has_valid_incumbent'] is False and f['has_valid_bound'] is False,
          'Accounting review is inconsistent with the failed result')
    native = _read(f['native_result_reference'])['native_interface']
    _need(native['raw_model_status'] == 17 and native['status'] == 'MEM_LIMIT'
          and native['solution_count'] == 0 and native['valid_incumbent'] is False
          and native['solver_bound_without_constant_usd'] is None,
          'Wrong original native failure')
    _need(a['plan_id'] == opening['plan_id']
          and a['terminal_expense_reference'] == f['terminal_expense_reference']
          and a['cumulative_accounted_seconds'] == f['cumulative_accounted_seconds']
          and a['remaining_original_seconds'] == f['remaining_original_seconds'],
          'Terminal adoption discarded expense')
    if opening['domain'] == 'formal':
        _need(failure_review_reference['sha256'] == HELENE_REVIEW_SHA256
              and first == 2065.692102 and wall == 330.324711 and cap == 7200,
              'Only the exact accepted Helene two-failure debt is supported')
    return dict(reference=failure_review_reference, plan_id=opening['plan_id'],
                domain=opening['domain'], accounted_seconds=f['cumulative_accounted_seconds'],
                budget_seconds=cap, remaining_seconds=f['remaining_original_seconds'],
                campaign_root=str(campaign)), old, queue


def _claim_path(value):
    root = Path(value['campaign_root'])
    failure = _read(value['failure_review_reference'])
    # Renaming a byte-identical review cannot give the same terminal job a new
    # claim address. Its original immutable terminal reference remains stable.
    key = canonical_hash(dict(plan_id=value['plan_id'],
                              terminal_job_result_reference=failure['terminal_job_result_reference']))
    return root / 'operations' / 'saa-l-shaped-debt-claims' / (key + '.json')


def _checked(ref, worker, queue_root, *, require_claim):
    value = _read(ref)
    _need(value['kind'] == KIND and value['domain'] in ('formal', 'development'),
          'Wrong decomposition debt kind/domain')
    _need(value['module_reference'] == batch.file_reference(__file__), 'Debt adapter source changed')
    review = _read(value['independent_review_reference'])
    _need(review['kind'] == REVIEW_KIND and review['passed'] is True
          and review['payload_sha256'] == canonical_hash({k: v for k, v in value.items()
                                                         if k != 'independent_review_reference'}),
          'Decomposition debt successor is not independently reviewed')
    _need(worker.get('failure_carryover_reference') == ref
          and value['successor_worker_identity_sha256'] == worker_identity(worker)
          and worker['method_id'] == 'SAA' and worker['domain'] == value['domain']
          and worker['model_input_mode'] == ('l_shaped_formal' if value['domain'] == 'formal'
                                              else 'l_shaped_development'),
          'Debt successor worker or algorithm mode changed')
    root, campaign = Path(queue_root), Path(value['campaign_root'])
    _need(root.is_absolute() and root.resolve() == root and campaign.resolve() == campaign
          and root.is_relative_to(campaign) and str(root) == value['successor_queue_root'],
          'Debt cannot move to another queue/campaign')
    opening, old, recovery_queue = verify_predecessor(
        value['failure_review_reference'], value['terminal_adoption_reference'])
    _need(str(campaign) == opening['campaign_root'], 'Successor campaign differs from the original admission')
    _need(opening['plan_id'] == value['plan_id'] == worker['plan_id']
          and opening['domain'] == worker['domain']
          and worker['budget'] == old['budget'], 'Original plan/domain/budget changed')
    for key in ('model_id', 'spec_id', 'target_id', 'canonical_identity', 'canonical_batch_id',
                'plan_input_references', 'protocol_reference', 'config_reference',
                'bank_acceptance_reference', 'summary_reference', 'audit_reference',
                'template_references'):
        _need(key in worker and key in old and worker[key] == old[key],
              'Original physical/protocol identity changed: ' + key)
    old_fee = _read(_read(value['failure_review_reference'])['original_failed_fee_reference'])
    old_queue = _read(old_fee['original_queue_reference'])
    _need(str(root) not in (old_queue['queue_root'], recovery_queue['queue_root']),
          'Failed predecessor queues must stay quarantined')
    claim = dict(kind=CLAIM_KIND, failure_review_reference=value['failure_review_reference'],
                 carryover_reference=ref, successor_queue_root=str(root),
                 successor_worker_identity_sha256=value['successor_worker_identity_sha256'])
    path = _claim_path(value)
    _need(path.parent.resolve() == path.parent, 'Aliased failed-parent claim directory')
    if require_claim:
        _need(path.is_file() and not path.is_symlink()
              and _read(batch.file_reference(path)) == claim,
              'Exact exclusive failed-parent claim missing or changed')
    return dict(opening, reference=ref), path, claim


def claim_failure_debt(reference, worker, queue_root):
    """Explicit preparation mutation, idempotent only for the exact successor."""
    opening, path, value = _checked(reference, worker, queue_root, require_claim=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
    except FileExistsError:
        _need(not path.is_symlink() and _read(batch.file_reference(path)) == value,
              'The same failed-parent debt was already claimed by another successor')
    return batch.file_reference(path)


def verify_failure_debt(reference, worker, queue_root):
    return _checked(reference, worker, queue_root, require_claim=True)[0]
