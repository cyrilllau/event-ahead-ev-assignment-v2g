"""Bounded Case0 planning audit, with explicit terminal parent-wall accounting.

The numerical auditor only reads old evidence. This entry owns an immutable
reservation before importing that auditor. A hard kill leaves the reservation
unsettled; the next visit must settle the exact terminal job before continuing.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from .campaign_batch import file_reference, verify_reference
from .case0_campaign import active_config, campaign_root, read, save_state
from .runtime.jobctl import atomic_json, observation, TERMINAL, utcnow


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def settle_audit_job(root, reservation_reference):
    """Charge the terminal parent duration once, including failed audit work."""
    root = Path(root).resolve()
    verify_reference(reservation_reference)
    reservation = read(reservation_reference['path'])
    _need(reservation['kind'] == 'CASE0_AUDIT_RESERVATION_V1', 'Wrong reservation kind')
    folder = Path(reservation['operation_root']).resolve()
    _need(folder.is_relative_to(root / 'operations'), 'Foreign audit operation')
    job = root / 'jobs' / reservation['job_id']
    verify_reference(reservation['parent_manifest_reference'])
    _need(Path(reservation['parent_manifest_reference']['path']) == job / 'manifest.json', 'Foreign parent manifest')
    terminal = read(job / 'result.json')
    status = read(job / 'status.json')
    _need(terminal.get('complete') is True and terminal.get('state') in TERMINAL,
          'Audit parent is not terminal')
    observed = observation(status)
    _need(status['state'] == terminal['state'] and not observed['blocks_new_launch']
          and not observed.get('supervisor_alive') and not observed.get('command_alive'),
          'Audit parent still has live or unreviewed work')
    seconds = float(terminal['elapsed_seconds'])
    _need(math.isfinite(seconds) and seconds >= 0, 'Invalid actual parent duration')
    receipt_path = folder / 'terminal_accounting.json'
    expected = dict(kind='CASE0_AUDIT_TERMINAL_ACCOUNTING_V1',
        reservation_reference=reservation_reference, parent_result_reference=file_reference(job / 'result.json'),
        opening_compute_seconds=reservation['opening_compute_seconds'], charged_seconds=seconds,
        closing_compute_seconds=reservation['opening_compute_seconds'] + seconds,
        clock_scope='Actual jobctl parent wall time, including isolation startup and shutdown; not child-only timing')
    if receipt_path.exists():
        _need(read(receipt_path) == expected, 'Terminal accounting changed')
    else:
        atomic_json(receipt_path, expected)
    receipt_ref = file_reference(receipt_path)
    state = read(root / 'state.json')
    settled = state.setdefault('audit_terminal_settlements', {})
    if reservation['job_id'] in settled:
        _need(settled[reservation['job_id']] == receipt_ref, 'Settled audit receipt changed')
        return receipt_ref
    _need(state['compute_seconds'] == reservation['opening_compute_seconds'], 'Concurrent Case0 budget mutation')
    _need(state.get('audit_reservation_reference') == reservation_reference, 'Active reservation differs')
    state['compute_seconds'] = expected['closing_compute_seconds']
    state['audit_reservation_reference'] = None
    state['active_job_id'] = None
    settled[reservation['job_id']] = receipt_ref
    state['next_action'] = 'Read independent audit output and terminal receipt; require the corresponding scientific acceptance before downstream evaluation or reporting.'
    save_state(root, state)
    return receipt_ref


def run_audit_job(request_reference):
    began = time.monotonic()
    from .case0_isolation import verify_current_cgroup
    isolation = verify_current_cgroup()
    verify_reference(request_reference)
    request = read(request_reference['path'])
    root = campaign_root()
    folder = Path(request['operation_root']).resolve()
    _need(folder.is_relative_to(root / 'operations') and folder.is_dir(), 'Foreign audit output directory')
    seconds = request['seconds']
    _need(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 7180,
          'Finite bounded audit duration required')
    job_id = os.environ['EVCS_JOB_ID']
    job = root / 'jobs' / job_id
    _need(Path(os.environ['EVCS_JOB_DIRECTORY']).resolve() == job, 'Foreign Case0 parent')
    manifest = read(job / 'manifest.json')
    _need(manifest['job_id'] == job_id and Path(manifest['outroot']) == root / 'jobs', 'Foreign parent manifest')
    with (root / '.audit.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / 'state.json')
        config = active_config(root)
        phase = request.get('audit_phase', 'planning')
        _admit_phase(state, phase)
        _need(not state.get('audit_reservation_reference'), 'Prior audit must be terminally settled')
        _need(not state.get('pending_evaluation_attempt'), 'Prior evaluation parent must be terminally settled')
        _need(manifest['timeout_seconds'] + manifest['grace_seconds'] <= config['total_compute_seconds'] - state['compute_seconds'],
              'Case0 remaining budget cannot cover bounded audit parent')
        reservation = dict(kind='CASE0_AUDIT_RESERVATION_V1', at_utc=utcnow(), job_id=job_id,
            operation_root=str(folder), opening_compute_seconds=state['compute_seconds'],
            parent_manifest_reference=file_reference(job / 'manifest.json'), request_reference=request_reference,
            duration_limit_seconds=seconds, isolation=isolation)
        path = folder / 'reservation.json'
        _need(not path.exists(), 'Audit operation is immutable; use an explicit new attempt')
        atomic_json(path, reservation)
        state['audit_reservation_reference'] = file_reference(path)
        state['active_job_id'] = job_id
        state['next_action'] = f'Case0 independent {phase} audit is running; retain reservation and settle actual parent wall time at the next visit.'
        save_state(root, state)
        try:
            # Charged/reserved before importing numerical libraries and reopening inputs.
            if phase == 'planning':
                from .case0_planning_audit import audit_case0_planning as audit
            else:
                from .case0_evaluation_audit import audit_case0_evaluation as audit
            result = audit(request=request['audit_request'], deadline=began + seconds)
            atomic_json(folder / 'audit_job_output.json', dict(kind='CASE0_AUDIT_JOB_OUTPUT_V1',
                result=result, request_reference=request_reference, child_wall_seconds=time.monotonic() - began))
            return result
        except Exception as error:
            atomic_json(folder / 'failure.json', dict(type=type(error).__name__, error=str(error),
                child_wall_seconds=time.monotonic() - began))
            raise


def _admit_phase(state, phase):
    """The post-OOS audit reads only a completed, parent-settled evaluation."""
    _need(state['active_configuration'] is None, 'Unsettled Case0 configuration')
    _need(phase in ('planning', 'evaluation'), 'Unknown Case0 audit phase')
    if phase == 'planning':
        _need(state['evaluation_opened'] is False and state['status'] == 'PLANNING_REVIEW_REQUIRED',
              'Planning state does not admit this audit')
    else:
        _need(state['evaluation_opened'] is True and state['status'] == 'CASE0_EVALUATION_COMPLETE_PENDING_REVIEW'
              and state.get('last_evaluation_settlement') is not None,
              'Completed settled evaluation required for its numerical audit')
    _need(not state.get('pending_evaluation_attempt') and not state.get('audit_reservation_reference'),
          'Unsettled Case0 parent blocks a new audit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    os.environ['EVCS_CONFIG'] = str(Path(args.config).resolve())
    ref = file_reference(args.request)
    _need(ref['sha256'] == args.sha256, 'Audit request checksum differs')
    print(json.dumps(run_audit_job(ref), allow_nan=False))


if __name__ == '__main__':
    main()
