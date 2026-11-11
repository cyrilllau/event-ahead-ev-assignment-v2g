"""Settle one terminal auxiliary-bound parent once; never modify old evidence."""
from decimal import Decimal
import json
from pathlib import Path

from evcs_experiments.campaign_batch import verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import TERMINAL, observation

OPENING = Decimal('5633.96054736330092')
PLAN_ID = 'mechanism.michael_2018.EA_DRO.baseline'
PLAN_SHA256 = '33179f0050e20f734a9f1706e26a88e393e0c2ba79de75b46a2070d25857c1d7'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def account_parent(previous, terminal, live):
    """Only terminal elapsed contributes; source phase timers are not added."""
    need(terminal['state'] in TERMINAL and terminal['complete'] is True,
         'Parent must be terminal')
    need(live['blocks_new_launch'] is False and live['command_alive'] is False
         and live['supervisor_alive'] is False and not live['live_group_members'],
         'Parent or descendants are still alive')
    need(previous['kind'] == 'S05_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1'
         and previous['status'] == 'SETTLED_BOUND_EVIDENCE_PENDING_FINAL_RESULT_REVIEW'
         and previous['cumulative_cap_seconds'] == 7200
         and previous['parent_job_id'] == 's05-multi-anchor-r22-20260909T053500Z'
         and previous['terminal_state'] == 'FAILED'
         and Decimal(previous['opening_cumulative_seconds_decimal']) == Decimal('5438.82363936330092')
         and Decimal(previous['terminal_parent_seconds_decimal']) == Decimal('195.136908')
         and previous['accounting_PASS'] is True and previous['parent_charged_once'] is True
         and previous['nested_seconds_added'] is False
         and previous['cumulative_seconds_provisional'] is False
         and previous['plan_id'] == PLAN_ID, 'Invalid predecessor settlement')
    prior = Decimal(previous['cumulative_accounted_seconds_decimal'])
    actual = Decimal(str(terminal['elapsed_seconds']))
    need(prior == OPENING and actual.is_finite() and actual >= 0, 'Invalid original or terminal balance')
    need(Decimal(previous['remaining_seconds_decimal']) == Decimal(7200)-prior,
         'Inconsistent predecessor remaining budget')
    total = prior + actual
    need(total <= Decimal(7200), 'Original cumulative budget exceeded; manual accounting review required')
    return dict(kind='S05_CACHED_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1', plan_id=PLAN_ID,
        accounting_PASS=True, opening_cumulative_seconds_decimal=str(prior),
        terminal_parent_seconds_decimal=str(actual), cumulative_accounted_seconds_decimal=str(total),
        remaining_seconds_decimal=str(Decimal(7200)-total), cumulative_cap_seconds=7200,
        parent_charged_once=True, nested_seconds_added=False, cumulative_seconds_provisional=False,
        numerical_acceptance=False, scientific_acceptance=False, formal_OOS=False)


def settle(request_path, expected_sha256):
    request_ref = file_reference(request_path)
    need(request_ref['sha256'] == expected_sha256, 'Changed original request')
    req = read(request_path)
    need(req['kind'] == 'S05_CACHED_MULTI_ANCHOR_BOUND_ONLY_V1' and req['plan_id'] == PLAN_ID,
         'Wrong request identity')
    paths = get_paths()
    campaign = paths.campaign_root.resolve()
    root = paths.project_root.resolve()
    for ref in req['references'].values():
        verify_reference(ref)
    source_ref = req['references']['source_manifest']
    need(Path(source_ref['path']).parent == root, 'Settlement must use the bound release')
    source = read(source_ref['path'])
    source_files = {}
    for row in source['files']:
        path = (root / row['path']).resolve()
        need(path.is_relative_to(root) and str(path) not in source_files, 'Invalid source member')
        ref = dict(path=str(path), sha256=row['sha256'], size_bytes=row['size_bytes'])
        verify_reference(ref)
        source_files[str(path)] = ref
    need(file_reference(__file__) == source_files.get(str(Path(__file__).resolve())), 'Unbound settlement source')
    job = Path(req['parent_job']['directory']).resolve()
    need(job == campaign/'jobs'/req['parent_job']['job_id'], 'Wrong parent path')
    manifest = read(job/'manifest.json')
    terminal = read(job/'result.json')
    need(terminal == read(job/'status.json') and terminal['job_id'] == req['parent_job']['job_id'],
         'Parent status/result identity mismatch')
    need(manifest['job_id'] == terminal['job_id'] and manifest['cwd'] == str(root)
         and manifest['outroot'] == str(campaign/'jobs') and manifest['spec_id'] == 'S05'
         and manifest['threads'] == 1 and manifest['timeout_seconds'] == 1000
         and manifest['grace_seconds'] == 5, 'Wrong parent manifest')
    need(manifest['argv'] == [str(root/'.venv/bin/python'), '-B', req['references']['script']['path'],
         '--request', request_ref['path'], '--expected-request-sha256', expected_sha256],
         'Parent did not execute the exact request')
    controller = req['parent_job']['controller_reference']
    verify_reference(controller)
    need(controller == source_files.get(str(root/'src/evcs_experiments/runtime/jobctl.py'))
         and Path(manifest['controller_path']).resolve() == Path(controller['path']).resolve(),
         'Parent controller source differs')
    attempt = campaign/'multi_anchor_bound_attempts'/canonical_hash(dict(
        settlement_sha256=req['references']['settlement']['sha256'], plan_id=PLAN_ID))
    reservation = read(attempt/'reservation.json')
    need(reservation['plan_id'] == PLAN_ID and reservation['request_reference'] == request_ref
         and reservation['parent_manifest_reference'] == file_reference(job/'manifest.json')
         and reservation['settlement'] == req['references']['settlement'], 'Wrong unique reservation')
    destination = attempt/'settlement.json'
    need(not destination.exists(), 'Parent already settled; never charge twice')
    # Decode the serialized elapsed value exactly, without binary float addition.
    exact_terminal = json.loads((job/'result.json').read_text(), parse_float=Decimal)
    record = account_parent(read(req['references']['settlement']['path']), exact_terminal, observation(terminal))
    record.update(status='SETTLED_BOUND_EVIDENCE_PENDING_FINAL_RESULT_REVIEW',
        parent_job_id=terminal['job_id'], terminal_state=terminal['state'],
        request_reference=request_ref, previous_settlement_reference=req['references']['settlement'],
        parent_result_reference=file_reference(job/'result.json'),
        parent_manifest_reference=file_reference(job/'manifest.json'),
        reservation_reference=file_reference(attempt/'reservation.json'),
        settlement_source_reference=file_reference(__file__), source_manifest_reference=source_ref,
        unique_settlement_path=str(destination), old_records_modified=False)
    output = Path(req['output_root']).resolve()
    need(output == campaign/'multi_anchor_bounds'/terminal['job_id'], 'Wrong worker output')
    for name in ('result.json', 'failure.json', 'artifact_manifest.json'):
        path = output/name
        if path.exists():
            record[name.replace('.json', '')+'_reference'] = file_reference(path)
    # Accounting a failed parent is mandatory too. Numeric bound promotion is a
    # separate review, so a partial worker can never manufacture a new accepted LB.
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--expected-request-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(settle(args.request, args.expected_request_sha256), indent=2))
