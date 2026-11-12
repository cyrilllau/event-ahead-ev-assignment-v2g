"""Append-only protocol and complete paired-plan locks before evaluation.

Every use verifies the referenced bytes, including all frozen evaluation source
files. Future planning implementations are bound separately in each plan record.
This gateway authorizes a declared sample scope; it does not open an OOS bank.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re

from .evidence import sha256
from .random_streams import canonical_hash


PROTOCOL_REFERENCE_KINDS = ('source', 'evaluator', 'config', 'random_streams', 'experiment_matrix', 'budgets')
PLAN_REFERENCE_KINDS = ('plan', 'tasks', 'routes', 'initial_state', 'planning_support', 'solver_record', 'budget')
PLAN_IDENTITY_KEYS = ('plan_id', 'target_id', 'method_id', 'setting_id')


def file_reference(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Evidence reference must name a file')
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


def _verify_reference(reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('A file reference requires path, sha256 and size_bytes')
    if (not isinstance(reference['path'], str) or not Path(reference['path']).is_absolute()
            or not isinstance(reference['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])
            or type(reference['size_bytes']) is not int or reference['size_bytes'] < 0):
        raise ValueError('Malformed evidence reference')
    actual = file_reference(reference['path'])
    if actual != reference:
        raise ValueError(f'Evidence bytes or resolved path changed: {reference["path"]}')
    return actual


def _nonempty_ids(values, label):
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f'{label} must be a nonempty list of unique string IDs')
    return values


def _batch_name(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Unsafe or empty batch ID')
    return value


def _reference_list(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label} requires nonempty file references')
    paths = []
    for value in values:
        paths.append(_verify_reference(value)['path'])
    if len(paths) != len(set(paths)):
        raise ValueError(f'{label} contains duplicate references')


def _validate_protocol(spec):
    if not isinstance(spec, dict) or spec.get('domain') not in ('development', 'formal'):
        raise ValueError('Protocol requires an explicit development/formal domain')
    if not isinstance(spec.get('model_version'), str) or not spec['model_version']:
        raise ValueError('Protocol requires model_version')
    targets = _nonempty_ids(spec.get('target_ids'), 'target_ids')
    methods = _nonempty_ids(spec.get('method_ids'), 'method_ids')
    if spec['domain'] == 'formal' and len(targets) != 8:
        raise ValueError('The formal protocol must freeze all eight target events')
    refs = spec.get('references')
    if not isinstance(refs, dict) or not set(PROTOCOL_REFERENCE_KINDS) <= set(refs):
        raise ValueError('Protocol source/evaluator/config/random-stream/matrix/budget references are required')
    for kind, values in refs.items():
        _reference_list(values, f'protocol {kind}')
    batches = spec.get('batches')
    if not isinstance(batches, list) or not batches:
        raise ValueError('Protocol must predeclare the paired batches')
    batch_ids = []
    global_plan_ids = []
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError('Invalid batch specification')
        batch_ids.append(_batch_name(batch.get('batch_id')))
        plans = batch.get('expected_plans')
        if not isinstance(plans, list) or not plans:
            raise ValueError('Each batch must predeclare all expected plans')
        identities = set()
        for plan in plans:
            if (not isinstance(plan, dict) or set(plan) != set(PLAN_IDENTITY_KEYS)
                    or any(not isinstance(v, str) or not v for v in plan.values())):
                raise ValueError('Expected plans require exactly plan/target/method/setting IDs')
            if plan['target_id'] not in targets or plan['method_id'] not in methods:
                raise ValueError('Batch target/method is outside the frozen protocol')
            identity = tuple(plan[key] for key in PLAN_IDENTITY_KEYS[1:])
            if identity in identities:
                raise ValueError('Duplicate target/method/setting in paired batch')
            identities.add(identity)
            global_plan_ids.append(plan['plan_id'])
        samples = batch.get('sample_ids_by_target')
        used_targets = {plan['target_id'] for plan in plans}
        if not isinstance(samples, dict) or set(samples) != used_targets:
            raise ValueError('Sample-list target scope must exactly match the paired plans')
        all_samples = []
        for target, values in samples.items():
            all_samples.extend(_nonempty_ids(values, f'{target} sample IDs'))
        if len(all_samples) != len(set(all_samples)):
            raise ValueError('Sample IDs must be unique across target lists')
    if len(batch_ids) != len(set(batch_ids)) or len(global_plan_ids) != len(set(global_plan_ids)):
        raise ValueError('Duplicate batch or plan IDs in frozen protocol')


def _write_new(path, kind, payload):
    envelope = dict(schema_version=1, kind=kind,
        created_at_utc=datetime.now(timezone.utc).isoformat(), payload=payload,
        payload_sha256=canonical_hash(payload))
    with path.open('x', encoding='utf-8') as handle:
        json.dump(envelope, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return file_reference(path)


def _read(path, kind):
    envelope = json.loads(path.read_text(encoding='utf-8'))
    if (envelope.get('schema_version') != 1 or envelope.get('kind') != kind
            or envelope.get('payload_sha256') != canonical_hash(envelope.get('payload'))):
        raise ValueError(f'Corrupt {kind} envelope')
    return envelope['payload']


@contextlib.contextmanager
def _transaction(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.append.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield directory
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create_protocol_lock(lock_dir, protocol_spec):
    """Create one frozen protocol; existing IDs/files are never overwritten."""
    with _transaction(lock_dir) as directory:
        if (directory / 'protocol.json').exists():
            raise ValueError('Protocol lock already exists; settings cannot be replaced')
        _validate_protocol(protocol_spec)
        return _write_new(directory / 'protocol.json', 'protocol_lock', protocol_spec)


def _verified_protocol(directory):
    path = directory / 'protocol.json'
    protocol = _read(path, 'protocol_lock')
    _validate_protocol(protocol)
    return protocol, file_reference(path)


def _batch_spec(protocol, batch_id):
    _batch_name(batch_id)
    matches = [batch for batch in protocol['batches'] if batch['batch_id'] == batch_id]
    if len(matches) != 1:
        raise ValueError('Batch ID was not frozen in the protocol')
    return matches[0]


def _validate_plans(batch_spec, plans):
    if not isinstance(plans, list):
        raise ValueError('Plan bank must be a list')
    expected = {value['plan_id']: value for value in batch_spec['expected_plans']}
    seen = set()
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError('Invalid plan record')
        identity = {key: plan.get(key) for key in PLAN_IDENTITY_KEYS}
        plan_id = identity['plan_id']
        if not isinstance(plan_id, str) or plan_id in seen or expected.get(plan_id) != identity:
            raise ValueError('Duplicate, extra or mismatched plan identity')
        seen.add(plan_id)
        digest = plan.get('plan_sha256')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Plan semantic SHA is required in addition to raw file SHA')
        refs = plan.get('references')
        base_keys = set(PLAN_REFERENCE_KINDS) | {'planning_source'}
        composite_keys = {'composite_request', 'composite_accounting', 'adapter_sources'}
        if not isinstance(refs, dict):
            raise ValueError('Plan/task/route/state/support/numerical/budget/planning-source references are required')
        extras = set(refs) - base_keys
        if (not base_keys <= set(refs) or
                (extras not in (set(), {'budget_seal'}) and
                 not (extras == composite_keys and identity['method_id'] == 'SAA'))):
            raise ValueError('Unsupported plan reference contract')
        for kind in PLAN_REFERENCE_KINDS:
            _verify_reference(refs[kind])
        _reference_list(refs['planning_source'], 'plan planning_source')
        if extras == {'budget_seal'}:
            _verify_reference(refs['budget_seal'])
        elif extras == composite_keys:
            _verify_reference(refs['composite_request'])
            _verify_reference(refs['composite_accounting'])
            _reference_list(refs['adapter_sources'], 'composite adapter_sources')
        _publication_envelope(plan)
    if seen != set(expected):
        raise ValueError('Paired batch is missing expected plans; lock all methods/settings together')


def _publication_envelope(record):
    """Bind a new publication confirmation without a circular record hash."""
    names = {'canonical_record_reference', 'publication_completion_reference'}
    if not names.intersection(record):
        return None
    if not names <= set(record):
        raise ValueError('Publication envelope requires both original record and completion')
    for name in names:
        _verify_reference(record[name])
    original = json.loads(Path(record['canonical_record_reference']['path']).read_text())
    if {k: v for k, v in record.items() if k not in names} != original:
        raise ValueError('Publication envelope differs from original canonical record')
    completion = json.loads(Path(record['publication_completion_reference']['path']).read_text())
    if completion.get('canonical_record_reference') != record['canonical_record_reference']:
        raise ValueError('Publication completion binds another canonical record')
    return completion


def lock_plan_batch(lock_dir, batch_id, plans):
    """Atomically bind every plan in a predeclared paired batch, before access."""
    with _transaction(lock_dir) as directory:
        protocol, protocol_ref = _verified_protocol(directory)
        spec = _batch_spec(protocol, batch_id)
        destination = directory / f'batch-{batch_id}.json'
        if destination.exists() or (directory / f'access-{batch_id}.json').exists():
            raise ValueError('Batch already locked or accessed; late plans and overwrites are forbidden')
        _validate_plans(spec, plans)
        ordered = {value['plan_id']: value for value in plans}
        payload = dict(batch_id=batch_id, domain=protocol['domain'], protocol_reference=protocol_ref,
            plans=[ordered[value['plan_id']] for value in spec['expected_plans']])
        return _write_new(destination, 'plan_batch_lock', payload)


def _verified_batch(directory, batch_id):
    protocol, protocol_ref = _verified_protocol(directory)
    spec = _batch_spec(protocol, batch_id)
    path = directory / f'batch-{batch_id}.json'
    batch = _read(path, 'plan_batch_lock')
    _verify_reference(batch['protocol_reference'])
    if batch['protocol_reference'] != protocol_ref or batch.get('domain') != protocol['domain'] or batch.get('batch_id') != batch_id:
        raise ValueError('Batch protocol/domain identity changed')
    _validate_plans(spec, batch['plans'])
    return dict(protocol=protocol, batch=batch, batch_spec=spec,
        protocol_reference=protocol_ref, batch_reference=file_reference(path))


def verify_plan_batch(lock_dir, batch_id):
    """Return locked payloads only after checking all referenced bytes again."""
    with _transaction(lock_dir) as directory:
        return _verified_batch(directory, batch_id)


def _accepted_s04(reference, protocol_reference):
    if reference is None:
        raise ValueError('Formal evaluation requires the future full S04 PASS acceptance reference')
    _verify_reference(reference)
    card = json.loads(Path(reference['path']).read_text())
    if card.get('spec_id') != 'S04' or card.get('status') != 'PASS' or card.get('increment_id'):
        raise ValueError('An increment/other-stage result does not grant full S04 acceptance')
    if card.get('formal_protocol_lock') != protocol_reference:
        raise ValueError('S04 acceptance must bind this exact frozen formal protocol')
    _verify_reference(card['formal_protocol_lock'])


def verify_formal_numerical_records(protocol, batch):
    """Enforce project budget/numerical records at the central OOS gate."""
    if 'configuration' not in protocol:
        return  # Generic synthetic lock tests have no project calculation policy.
    pb=protocol['configuration']['planning_budget']
    for record in batch['plans']:
        budget=json.loads(Path(record['references']['budget']['path']).read_text())
        numeric=json.loads(Path(record['references']['solver_record']['path']).read_text())
        if budget.get('planning_budget')!=pb:raise ValueError('Plan planning budget differs from common frozen budget')
        actual=numeric.get('cumulative_wall_seconds')
        if isinstance(actual,bool) or not isinstance(actual,(float,int)) or not math.isfinite(actual) or not 0<=actual<=pb['cumulative_wall_seconds']:raise ValueError('Invalid cumulative plan budget')
        if numeric.get('domain')!='formal' or numeric.get('plan_sha256')!=record['plan_sha256'] or numeric.get('full_support_verified') is not True or numeric.get('status') not in ('OPTIMAL','CONVERGED','TIME_LIMIT'):raise ValueError('Formal plan lacks original full-support numerical record')
        worker_ref = numeric.get('request_reference')
        policy = None
        if worker_ref is not None:
            _verify_reference(worker_ref)
            policy = json.loads(Path(worker_ref['path']).read_text()).get('canonical_export_policy')
        if policy is not None:
            if policy != 'IN_ATTEMPT_FINAL_SEAL_V1':
                raise ValueError('Unknown canonical publication accounting policy')
            completion = _publication_envelope(record)
            if completion is None:
                raise ValueError('Charged canonical publication completion is required')
            from .canonical_in_attempt import verify_publication
            checked = verify_publication(record['publication_completion_reference'],
                receipt_reference=numeric['receipt_reference'],
                finalization_reference=record['references']['budget_seal'],
                worker_request_reference=worker_ref)
            if checked != completion:
                raise ValueError('Canonical publication confirmation differs')
        composite_kind = 'SAA_ORIGINAL_SEAL_PLUS_EXTERNAL_LEDGER_V1'
        composite_record = 'FORMAL_SAA_SAME_PLAN_AUXILIARY_NUMERICAL_RECORD_V1'
        composite = (numeric.get('accounting_kind') == composite_kind or
                     budget.get('accounting_kind') == composite_kind or
                     numeric.get('kind') == composite_record or
                     'composite_accounting' in record['references'] or
                     'composite_request' in record['references'])
        if record.get('method_id') == 'SAA' and not composite and 'final_budget_acceptance' not in numeric:
            raise ValueError('SAA record requires explicit original or composite accounting')
        if composite:
            if (numeric.get('accounting_kind') != composite_kind or
                    budget.get('accounting_kind') != composite_kind or
                    numeric.get('kind') != composite_record):
                raise ValueError('Incomplete SAA composite accounting identity')
            # This validates the original seal against the ORIGINAL bytes, then
            # the new record against its explicit external-ledger adoption.
            from .saa_composite_record import validate_composite
            validate_composite(record, protocol)
        elif 'final_budget_acceptance' in numeric:
            if numeric.get('final_budget_acceptance') is not True or budget.get('final_budget_acceptance') is not True:
                raise ValueError('Final plan accounting is incomplete')
            if numeric.get('requires_seal_confirmation') is not True:
                raise ValueError('Final accounting seal confirmation required')
            if numeric.get('requires_seal_confirmation'):
                seal_ref=record['references'].get('budget_seal')
                if seal_ref is None:raise ValueError('Final accounting seal confirmation required')
                _verify_reference(seal_ref);seal=json.loads(Path(seal_ref['path']).read_text())
                seal_actual=seal.get('actual_attempt_seconds_before_confirmation_write')
                seal_charged=seal.get('charged_attempt_seconds')
                cutoff=numeric.get('actual_attempt_seconds_at_cutoff')
                if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                        for v in (seal_actual,seal_charged,cutoff)) or
                    seal_charged!=numeric.get('charged_attempt_seconds') or
                    seal_charged!=budget.get('charged_attempt_seconds') or seal_actual<cutoff):
                    raise ValueError('Final accounting seal clock or charge differs')
                if (seal.get('status')!='PASS_FINAL_RECORD_SEAL' or
                    seal.get('references',{}).get('solver_record')!=record['references']['solver_record'] or
                    seal.get('references',{}).get('budget')!=record['references']['budget'] or
                    seal.get('actual_attempt_seconds_before_confirmation_write',math.inf)>seal.get('charged_attempt_seconds',-1)):
                    raise ValueError('Final accounting seal differs')
        lb,ub=numeric.get('lower_bound_usd'),numeric.get('upper_bound_usd')
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in (lb,ub)) or lb>ub+1e-4:raise ValueError('Invalid finite plan bound ordering')


def begin_evaluation(lock_dir, batch_id, *, domain, sample_ids_by_target, s04_acceptance_ref=None):
    """Verify a complete locked scope, then append its first-access marker.

Repeat uses return the same marker only after re-verifying every byte. A formal
S04 card is an external reference, so protocol creation does not depend on a
future acceptance card that in turn must pin the protocol.
"""
    with _transaction(lock_dir) as directory:
        checked = _verified_batch(directory, batch_id)
        if domain != checked['protocol']['domain']:
            raise ValueError('Evaluation domain differs from the frozen protocol')
        if sample_ids_by_target != checked['batch_spec']['sample_ids_by_target']:
            raise ValueError('Evaluation sample IDs/order/target scope differs from the frozen paired batch')
        if domain == 'formal':
            if checked['batch_spec'].get('oos_enabled') is False:
                raise ValueError('Algorithm-only batch may not open OOS')
            _accepted_s04(s04_acceptance_ref, checked['protocol_reference'])
            verify_formal_numerical_records(checked['protocol'], checked['batch'])
        elif s04_acceptance_ref is not None:
            raise ValueError('Development scope cannot borrow a formal acceptance identity')
        payload = dict(batch_id=batch_id, domain=domain,
            protocol_reference=checked['protocol_reference'], batch_reference=checked['batch_reference'],
            sample_ids_by_target=sample_ids_by_target, s04_acceptance_reference=s04_acceptance_ref)
        path = directory / f'access-{batch_id}.json'
        if path.exists():
            existing = _read(path, 'evaluation_access')
            if existing != payload:
                raise ValueError('Evaluation access identity changed after first access')
        else:
            _write_new(path, 'evaluation_access', payload)
        return dict(**checked, access_reference=file_reference(path), access=payload)
