"""Launch or settle one sealed SAA auxiliary parent; no numerical adoption."""
import argparse
import fcntl
import importlib.util
import json
import sys
from pathlib import Path
from evcs_experiments import campaign_batch as batch
from evcs_experiments.runtime import jobctl


def read(ref):
    batch.verify_reference(ref)
    return batch.read_json(ref['path'])


def load(ref, name):
    batch.verify_reference(ref)
    if name in sys.modules:
        raise ValueError('Overlay already imported: ' + name)
    spec = importlib.util.spec_from_file_location(name, ref['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'settle'])
    parser.add_argument('--control', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    control_ref = batch.file_reference(args.control)
    if control_ref['sha256'] != args.sha256:
        raise ValueError('Control changed')
    c = read(control_ref)
    if batch.file_reference(__file__) != c['operator_reference']:
        raise ValueError('Operator source changed')
    review = read(c['authority_review_reference'])
    if not (review['passed'] is True and review['authority_source_sha256'] == c['authority_source']['sha256']
            and review['operator_source_sha256'] == c['operator_reference']['sha256']):
        raise ValueError('Authority glue review differs')
    helper = load(c['authority_source'], 'saa_bound_authority_overlay')
    ledger = load(c['ledger_source'], 'evcs_experiments.saa_recovery_ledger')
    ledger.SOURCE_UPGRADE_REFERENCE = c['ledger_source_upgrade']
    anchor = read(c['anchor_reference'])
    root = Path(anchor['campaign_root'])
    op = Path(args.control).resolve().parent
    if not op.is_relative_to(root / 'operations'):
        raise ValueError('Control must be within campaign operations')
    with (root / '.saa-bound-authority.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = batch.read_json(root / 'state.json')
        if args.action == 'launch':
            updated = helper.registration(state, c['anchor_reference'], c['admission_reference'],
                                          c['job_request_reference'], ledger=ledger)
            job = read(c['job_request_reference'])
            if (Path(job['outroot']) / job['job_id']).exists():
                raise ValueError('Parent job already exists')
            # Jobctl also enforces live OS identity under its own launch lock.
            jobctl.atomic_json(op / 'authority_before_launch.json', state)
            jobctl.atomic_json(root / 'state.json', updated)
            try:
                result = jobctl.launch(job)
            except BaseException as exc:
                jobctl.atomic_json(op / 'dispatch_failure.json', dict(type=type(exc).__name__,
                    error=str(exc), pending_debt_retained=True))
                raise
            jobctl.atomic_json(op / 'launch.json', result)
            next_action = ('SAA bound-only parent ' + job['job_id'] + ' launched. Read one job status; '
                'after actual termination run this sealed operator settle, then independently audit numerical '
                'result. No new Q, restart, budget reset or OOS. Control: ' + str(args.control))
        else:
            claim = state['saa_bound_only_pending']
            if claim['admission_reference'] != c['admission_reference']:
                raise ValueError('Control does not own pending parent')
            queue = Path(read(c['admission_reference'])['queue_root'])
            closing = op / 'closing_queue.json'
            snapshot = batch.read_json(queue / 'state.json')
            if closing.exists() and batch.read_json(closing) != snapshot:
                raise ValueError('Closing snapshot changed')
            if not closing.exists():
                jobctl.atomic_json(closing, snapshot)
            updated, settlement = helper.terminal(state, batch.file_reference(closing), ledger=ledger)
            result = dict(settlement_reference=settlement, scientific_acceptance=False)
            jobctl.atomic_json(op / 'settlement_reference.json', result)
            next_action = ('SAA bound-only parent settled once. Independently review exact cut/model/native '
                'evidence before adopting auxiliary LB; retain original SAA status/UB and no OOS. '
                'Control: ' + str(args.control))
        updated['next_action'] = next_action
        updated['updated_at'] = jobctl.utcnow()
        jobctl.atomic_json(root / 'state.json', updated)
        (root / 'NEXT_ACTION.md').write_text('# Next action\n\n' + next_action + '\n')
        with (root / 'decisions.jsonl').open('a') as log:
            log.write(json.dumps(dict(at_utc=jobctl.utcnow(), event='SAA_BOUND_' + args.action.upper(),
                                      control_reference=control_ref, result=result)) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
