"""Campaign reservation and terminal glue for one reviewed SAA bound parent.

The external ledger charges the actual parent once, including failures before
the driver creates its reservation. This module never adopts a numerical bound.
"""
from pathlib import Path
from evcs_experiments import campaign_batch as batch
from evcs_experiments.runtime import jobctl


def need(value, message):
    if not value:
        raise ValueError(message)


def read(ref):
    batch.verify_reference(ref)
    return batch.read_json(ref['path'])


def registration(state, anchor_ref, admission_ref, job_request_ref, *, ledger):
    """Validate before dispatch; caller atomically saves returned authority."""
    adm = read(admission_ref)
    job = read(job_request_ref)
    need(jobctl.checked_request(job) == job, 'Job request must seal all defaults')
    anchor, extra, last, pending, _ = ledger._chain(anchor_ref)
    need(not state.get('active_jobs') and not state.get('saa_bound_only_pending'),
         'Active or pending authority blocks dispatch')
    external = state['saa_recovery_external_ledger']
    need(external['anchor_reference'] == anchor_ref and
         external['latest_settlement_reference'] == last and
         not external.get('pending_job_id') and
         not external.get('unsettled_parent_debt_blocks_relaunch') and pending is None,
         'Authority ledger head or pending debt differs')
    opening = (anchor['initial_queue_snapshot_reference'] if last is None
               else read(last)['closing_queue_snapshot_reference'])
    _, q = ledger._snapshot(opening, anchor['queue_reference'], inventory=True)
    need(external['effective_spent_seconds'] == q + extra == adm['opening_effective_seconds'],
         'Authority effective budget differs')
    need(batch.read_json(Path(adm['queue_root']) / 'state.json') == read(opening),
         'Live queue changed before dispatch')
    need(not read(opening).get('active') and not read(opening).get('halted'),
         'Original queue is active or halted')
    need(job['job_id'] == adm['parent_job_id'] and
         Path(job['outroot']) == Path(anchor['campaign_root']) / 'jobs',
         'Foreign job request')
    # Exercise the same immutable admission contract before a manifest exists.
    candidate = dict(admission_reference=admission_ref,
                     opening_queue_snapshot_reference=opening,
                     opening_queue_seconds=q, effective_spent_seconds=q + extra,
                     previous_settlement_reference=last, job_id=job['job_id'])
    ledger._bound_parent_contract(anchor, candidate, job, adm)
    claim = dict(job_id=job['job_id'], admission_reference=admission_ref,
                 request_reference=adm['request_reference'],
                 expected_last_settlement_reference=last,
                 job_request_reference=job_request_ref, anchor_reference=anchor_ref,
                 opening_queue_snapshot_reference=opening)
    return dict(state, active_jobs=[job['job_id']], saa_bound_only_pending=claim)


def terminal(state, closing_ref, *, ledger):
    """Settle a dead parent, including an exact pre-reservation failure.

    If the ledger commit succeeded but authority writing was interrupted, replay
    joins that same settlement. Missing/ambiguous terminal evidence keeps debt.
    """
    claim = state['saa_bound_only_pending']
    adm = read(claim['admission_reference'])
    need(state['active_jobs'] == [claim['job_id']] and
         adm['parent_job_id'] == claim['job_id'] and
         adm['request_reference'] == claim['request_reference'], 'Foreign pending authority')
    job = Path(adm['parent_job_directory'])
    terminal_ref = batch.file_reference(job / 'result.json')
    result = read(terminal_ref)
    observed = jobctl.observation(batch.read_json(job / 'status.json'))
    need(result.get('complete') is True and result['state'] in ledger.TERMINAL and
         not observed['blocks_new_launch'] and not observed.get('supervisor_alive') and
         not observed.get('command_alive'), 'Parent is not confirmed terminal')
    anchor, _, last, pending, reservation_ref = ledger._chain(claim['anchor_reference'])
    manifest_ref = batch.file_reference(job / 'manifest.json')
    if pending is None and last != claim['expected_last_settlement_reference']:
        settled = read(last)
        reservation_ref = settled['reservation_reference']
        pending = read(reservation_ref)
        need(pending['job_id'] == claim['job_id'], 'Another parent changed ledger head')
    elif pending is None:
        # No computation can be silently lost when the CLI failed before reserve.
        reservation_ref = ledger.reserve_recovery_parent(
            claim['anchor_reference'], parent_job_request_reference=claim['job_request_reference'],
            parent_manifest_reference=manifest_ref, admission_reference=claim['admission_reference'],
            opening_queue_snapshot_reference=claim['opening_queue_snapshot_reference'])
        pending = read(reservation_ref)
    need(pending['job_id'] == claim['job_id'] and
         pending['admission_reference'] == claim['admission_reference'] and
         pending['parent_job_request_reference'] == claim['job_request_reference'] and
         pending['parent_manifest_reference'] == manifest_ref and
         pending['previous_settlement_reference'] == claim['expected_last_settlement_reference'],
         'Reservation differs from authority claim')
    need(read(closing_ref) == read(claim['opening_queue_snapshot_reference']),
         'Bound-only parent changed original queue')
    settlement_ref = ledger.settle_recovery_parent(reservation_ref, terminal_ref, closing_ref)
    settled = read(settlement_ref)
    need(settled['new_queue_charge_seconds'] == 0, 'Unexpected bound-only queue charge')
    external = dict(state['saa_recovery_external_ledger'],
                    latest_settlement_reference=settlement_ref,
                    effective_spent_seconds=settled['effective_spent_seconds'],
                    cumulative_extra_seconds=settled['cumulative_extra_seconds'],
                    queue_cumulative_seconds=settled['closing_queue_seconds'],
                    pending_job_id=None, pending_admission_reference=None,
                    pending_job_request_reference=None, pending_reservation_reference=None,
                    unsettled_parent_debt_blocks_relaunch=False)
    updated = dict(state, active_jobs=[], saa_bound_only_pending=None,
                   saa_recovery_external_ledger=external,
                   saa_bound_only_terminal=dict(settlement_reference=settlement_ref,
                     terminal_job_result_reference=terminal_ref,
                     scientific_acceptance=False, numerical_review_pending=True))
    return updated, settlement_ref
