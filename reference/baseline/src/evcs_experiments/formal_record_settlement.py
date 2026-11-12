"""Settle an audited candidate against an immutable campaign attempt receipt.

This accounting adapter does not run numerical checks or confer scientific/OOS
acceptance. It consumes the independent numerical worker's sealed output and
retains every native solver status, including nonconvergence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .evaluation_lock import _verify_reference, file_reference
from .planning_budget import validate_budget
from .random_streams import canonical_hash


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, nonnegative=False):
    _need(type(value) in (int, float) and math.isfinite(value), 'Invalid '+name)
    _need(not nonnegative or value >= 0, 'Negative '+name)
    return value


def _read(reference):
    _verify_reference(reference)
    return json.loads(Path(reference['path']).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _references(value):
    """Validate all nested strong references, without interpreting binary files."""
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'size_bytes'}:
            _verify_reference(value)
        else:
            for child in value.values():
                _references(child)
    elif isinstance(value, list):
        for child in value:
            _references(child)


def settle_plan_record(*, receipt_reference, numerical_audit_reference,
                       prior_cumulative_seconds, planning_budget,
                       output_directory=None):
    """Return ``solver_record`` and ``budget_record``; optionally write once.

    ``prior_cumulative_seconds`` must equal the original invocation balance,
    not a freshly inferred/reset balance. The actual receipt includes worker
    work and queue verification; provisional worker time never replaces it.
    The returned budget record retains the complete original planning policy.
    """
    budget = validate_budget(planning_budget)
    prior = _number(prior_cumulative_seconds, 'prior cumulative seconds', nonnegative=True)
    receipt = _read(receipt_reference)
    _need(receipt.get('returncode') == 0 and type(receipt.get('returncode')) is int,
          'Receipt must record successful process exit')
    invocation_ref = receipt['invocation_reference']
    invocation = _read(invocation_ref)
    result_ref = receipt['result_reference']
    result = _read(result_ref)
    request_ref = invocation['item_request_reference']
    request = _read(request_ref)
    queue = _read(receipt['request_reference'])
    _need(invocation['request_reference'] == receipt['request_reference'], 'Queue binding differs')
    _need(invocation.get('domain') in ('formal', 'development'), 'Missing domain')
    for key in ('item_id', 'plan_id'):
        _need(receipt.get(key) == invocation.get(key) and isinstance(receipt.get(key), str)
              and receipt[key], 'Receipt identity differs: '+key)
    _need(request.get('plan_id') == receipt['plan_id'], 'Worker request plan differs')
    items = [item for item in queue['items'] if item['item_id'] == receipt['item_id']]
    _need(len(items) == 1 and items[0]['plan_id'] == receipt['plan_id']
          and items[0]['request_reference'] == request_ref, 'Frozen queue item differs')
    _need(queue.get('domain') == invocation['domain'], 'Queue domain differs')
    _need(request.get('budget') == budget and
          items[0]['plan_budget_seconds'] == budget['cumulative_wall_seconds'] and
          invocation['plan_budget_seconds'] == budget['cumulative_wall_seconds'], 'Planning budget differs')
    _need(invocation['cumulative_plan_seconds'] == prior and
          invocation['plan_remaining_seconds'] == budget['cumulative_wall_seconds']-prior,
          'Prior balance or remaining budget differs')
    _need(result.get('invocation_sha256') == invocation_ref['sha256'] and
          Path(result_ref['path']) == Path(invocation['result_path']).resolve(), 'Worker result binding differs')
    _need(result.get('status') in ('COMPLETE', 'TIME_LIMIT') and
          receipt.get('status') == result['status'], 'Receipt/result status differs')
    _need(result.get('numerical_audit_reference') == numerical_audit_reference
          and numerical_audit_reference in result.get('artifacts', []), 'Numerical audit not retained by worker')
    _references(result)
    if result['status'] == 'TIME_LIMIT':
        _need(result.get('resumable') is True and result.get('checkpoint') in result['artifacts'],
              'Incomplete worker needs immutable resumable checkpoint')
    elapsed = _number(receipt['elapsed_seconds'], 'receipt elapsed seconds', nonnegative=True)
    worker_elapsed = _number(result['attempt_accounted_seconds'], 'worker elapsed seconds', nonnegative=True)
    _need(worker_elapsed <= elapsed and result['queue_prior_attempt_seconds'] == prior,
          'Worker elapsed or prior balance differs from receipt')
    _need(math.isclose(result['cumulative_accounted_seconds'], prior+worker_elapsed, rel_tol=0, abs_tol=1e-8),
          'Worker provisional cumulative time differs')
    total = prior+elapsed
    _need(total <= budget['cumulative_wall_seconds'], 'Actual cumulative budget exceeded')
    audit = _read(numerical_audit_reference)
    _references(audit)
    is_saa = request.get('method_id') == 'SAA'
    is_l_shaped = is_saa and request.get('model_input_mode') in ('l_shaped_development', 'l_shaped_formal')
    if is_l_shaped:
        expected_domain = 'formal' if request['model_input_mode'] == 'l_shaped_formal' else 'development'
        _need(request.get('domain') == invocation['domain'] == expected_domain,
              'L-shaped settlement requires its exact evidence domain')
    expected_status = ('PASS_SAA_FORMAL_WORKER_NUMERICAL_AUDIT' if is_saa else 'PASS_INDEPENDENT_PLANNING_NUMERICAL_AUDIT')
    _need(audit.get('status') == expected_status
          and audit.get('passed') is True and audit.get('full_support_verified') is True,
          'Complete independent numerical audit required')
    _need(audit.get('plan_id') == receipt['plan_id'] and audit.get('request_reference') == request_ref,
          'Numerical audit request/plan differs')
    candidate_ref = audit['candidate_record_reference']
    _need(candidate_ref in result['artifacts'], 'Candidate was not retained by worker')
    candidate_record = _read(candidate_ref)
    _references(candidate_record)
    candidate = candidate_record['candidate']
    _need(isinstance(candidate, dict) and candidate_record['plan_id'] == receipt['plan_id'], 'Missing or foreign candidate')
    digest = canonical_hash(candidate['plan'])
    _need(digest == candidate['plan_sha256'] == audit['plan_sha256'], 'Candidate semantic identity differs')
    if is_saa:
        _need(request.get('domain') == invocation['domain'], 'SAA invocation/request domain differs')
        from .saa_record_bridge import validate_saa_record_bridge
        validate_saa_record_bridge(audit, request, candidate_record)
    else:
        for name in ('recourse', 'probability', 'master'):
            part = _read(audit[name+'_audit_reference'])
            _references(part)
            _need(isinstance(part.get('status'), str) and part['status'].startswith('PASS_')
                  and part.get('passed', True) is True, 'Failed '+name+' audit')
            _need(part.get('plan_sha256') == digest, 'Foreign '+name+' audit plan')
    lb = _number(audit['lower_bound_usd'], 'lower bound')
    ub = _number(audit['upper_bound_usd'], 'upper bound')
    gap = _number(audit['gap_relative'], 'relative gap', nonnegative=True)
    _need(lb <= ub+1e-4 and math.isclose(gap, max(0., ub-lb)/max(1., abs(ub)), rel_tol=1e-10, abs_tol=1e-12),
          'Invalid bound ordering or relative gap')
    for key in ('lower_bound_usd', 'upper_bound_usd'):
        saved = _number(candidate[key], 'candidate '+key)
        _need(abs(saved-audit[key]) <= 1e-4+1e-7*abs(audit[key]),
              'Candidate/audit monetary field differs: '+key)
    _need(math.isclose(_number(candidate['gap_relative'], 'candidate gap', nonnegative=True),
          gap, rel_tol=1e-10, abs_tol=1e-12), 'Candidate/audit gap differs')
    _need(candidate['native_master_status'] == audit['native_master_status'],
          'Candidate/audit native status differs')
    _need(candidate_record['raw_engine_status'] == audit['raw_engine_status'] == result['raw_engine_status'],
          'Raw engine status differs')
    converged = candidate_record.get('driver_status') == 'CONVERGED'
    raw_converged = (audit['raw_engine_status'] == 'CONVERGED' if is_l_shaped else
                     audit['raw_engine_status'] in ('OPTIMAL', 'TIME_LIMIT') if is_saa else
                     audit['raw_engine_status'] == 'CONVERGED')
    _need(not converged or (gap <= .005 and raw_converged),
          'Convergence exceeds frozen tolerance or raw engine did not converge')
    record = dict(schema_version=1, domain=invocation['domain'], plan_id=receipt['plan_id'],
        target_id=request['target_id'], method_id=request['method_id'],
        plan_sha256=digest, status='CONVERGED' if converged else 'TIME_LIMIT',
        full_support_verified=True, lower_bound_usd=lb, upper_bound_usd=ub, gap_relative=gap,
        native_master_status=candidate['native_master_status'],
        native_master_solver_status=candidate.get('native_master_solver_status'),
        original_rmp=candidate.get('original_rmp'), raw_engine_status=audit['raw_engine_status'],
        driver_status=candidate_record.get('driver_status'), worker_status=result['status'],
        cumulative_wall_seconds=total, prior_cumulative_wall_seconds=prior,
        attempt_wall_seconds=elapsed, worker_provisional_seconds=worker_elapsed,
        queue_verification_overhead_seconds=elapsed-worker_elapsed,
        receipt_reference=receipt_reference, numerical_audit_reference=numerical_audit_reference,
        candidate_record_reference=candidate_ref, request_reference=request_ref,
        accounting_authority='campaign_batch immutable attempt receipt',
        accounting_scope='campaign_attempt_receipt_cutoff',
        post_receipt_administration_included=False, final_budget_acceptance=False,
        scientific_acceptance=False, scientific_spec_pass=False, formal_OOS=False)
    if is_saa:
        record.update(risk_certificate_kind='fixed_empirical',
            candidate_audit_reference=audit['candidate_audit_reference'])
    if is_l_shaped:
        record.update(status=candidate_record['driver_status'],
                      algorithm_id=candidate_record['algorithm_id'],
                      formal_admission=False)
    budget_record = dict(schema_version=1, plan_id=receipt['plan_id'], planning_budget=budget,
        cumulative_wall_seconds=total, receipt_reference=receipt_reference,
        prior_cumulative_wall_seconds=prior, accounting_scope='campaign_attempt_receipt_cutoff',
        post_receipt_administration_included=False, final_budget_acceptance=False,
        formal_OOS=False, scientific_acceptance=False)
    answer = dict(solver_record=record, budget_record=budget_record)
    if output_directory is not None:
        root = Path(output_directory).resolve()
        root.mkdir(parents=True, exist_ok=False)
        refs = {}
        for key, name, value in (('solver_record', 'solver_record.json', record), ('budget', 'budget.json', budget_record)):
            path = root/name
            with path.open('x') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write('\n')
            refs[key] = file_reference(path)
        answer['references'] = refs
    return answer


def finalize_settled_record(answer, *, actual_attempt_seconds_at_cutoff,
                            final_seal_reserve_seconds, output_directory):
    """Export an already validated answer with conservative sealing allocation.

    Only pass the in-process return of ``settle_plan_record``; this is not an
    alternative ingress for arbitrary dictionaries. The queue must confirm its
    final observed time is within the allocated seal reserve and retain a strong
    reference to that confirmation before accepting or using these records.
    The allocation is charged once even if sealing takes less time. It is not
    reported as measured solver time and cannot reset or expand the plan cap.
    """
    import copy
    record = copy.deepcopy(answer['solver_record'])
    budget_record = copy.deepcopy(answer['budget_record'])
    _need(record.get('accounting_scope') == 'campaign_attempt_receipt_cutoff'
          and record.get('final_budget_acceptance') is False,
          'Only an original validated receipt-cutoff answer can be finalized')
    budget = validate_budget(budget_record['planning_budget'])
    prior = _number(record['prior_cumulative_wall_seconds'], 'prior balance', nonnegative=True)
    actual = _number(actual_attempt_seconds_at_cutoff, 'actual attempt cutoff', nonnegative=True)
    reserve = _number(final_seal_reserve_seconds, 'seal reserve', nonnegative=True)
    _need(reserve > 0 and reserve <= budget['shutdown_reserve_seconds'],
          'Seal reserve must be positive and fit declared shutdown reserve')
    if record['domain'] == 'formal':
        _need(reserve == 10, 'Formal final seal allocation must be the declared 10 seconds')
    _need(actual >= record['attempt_wall_seconds'], 'Final cutoff precedes actual receipt')
    _need(budget_record['prior_cumulative_wall_seconds'] == prior
          and budget_record['plan_id'] == record['plan_id'], 'Validated answer accounting differs')
    charged_attempt = actual+reserve
    charged_total = prior+charged_attempt
    _need(charged_total <= budget['cumulative_wall_seconds'],
          'Final actual plus allocated sealing exceeds original cumulative cap')
    fields = dict(accounting_scope='actual_through_final_cutoff_plus_allocated_seal',
        cumulative_wall_seconds=charged_total, cumulative_charged_seconds=charged_total,
        cumulative_charged_seconds_before_final_seal=prior+actual,
        actual_attempt_seconds_at_cutoff=actual, charged_attempt_seconds=charged_attempt,
        final_seal_allocated_seconds=reserve,
        post_receipt_administration_seconds_at_cutoff=actual-record['attempt_wall_seconds'],
        post_receipt_administration_included=True, final_budget_acceptance=True,
        requires_seal_confirmation=True, seal_confirmation_verified=False,
        allocation_policy='charge full predeclared seal reserve once; unused allocation is not refunded',
        scientific_acceptance=False, scientific_spec_pass=False, formal_OOS=False)
    record.update(fields)
    budget_record.update(fields)
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    references = {}
    for key, name, value in (('solver_record', 'solver_record.json', record),
                             ('budget', 'budget.json', budget_record)):
        path = root/name
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        references[key] = file_reference(path)
    return dict(solver_record=record, budget_record=budget_record, references=references)
