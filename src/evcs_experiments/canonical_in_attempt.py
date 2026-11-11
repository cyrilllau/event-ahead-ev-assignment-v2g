"""Canonical publication inside one existing attempt's charged final seal.

The private preparation token is produced only after the queue settles the
original numerical audit. Expensive identity/source/input checks and plan writes
precede the final cutoff. The small completion receipt points to the standalone
record, which points to the original seal; no reference points backwards.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from .campaign_batch import file_reference, verify_reference, read_json
from .evaluation_lock import PLAN_IDENTITY_KEYS, _validate_protocol
from .random_streams import canonical_hash

POLICY = 'IN_ATTEMPT_FINAL_SEAL_V1'
METHODS = {'EA_DRO': 'EA', 'OR': 'OR', 'SAA': 'SAA', 'ARO': 'ARO'}
INPUTS = {'tasks', 'routes', 'initial_state', 'planning_support'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _read(ref):
    verify_reference(ref)
    return read_json(ref['path'])


def _new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return file_reference(path)


def _normal(ref):
    value = dict(path=ref['path'], sha256=ref['sha256'], size_bytes=ref.get('size_bytes', ref.get('bytes')))
    verify_reference(value)
    return value


@dataclass(frozen=True)
class _PreparedPublication:
    reference: dict
    payload: dict
    root: Path


def prepare_publication(receipt_reference, settled_answer, worker_request_reference, output):
    """Queue-internal preparation; pass its in-process settlement return only."""
    started = time.monotonic()
    worker = _read(worker_request_reference)
    _need(worker.get('canonical_export_policy') == POLICY, 'Explicit canonical export policy required')
    numeric = settled_answer['solver_record']
    _need(numeric['request_reference'] == worker_request_reference
          and numeric['receipt_reference'] == receipt_reference
          and numeric['accounting_scope'] == 'campaign_attempt_receipt_cutoff'
          and numeric['final_budget_acceptance'] is False and numeric['worker_status'] == 'COMPLETE',
          'Original completed in-process settlement required')
    envelope = _read(worker['protocol_reference']); protocol = envelope['payload']
    _need(envelope['kind'] == 'protocol_lock' and envelope['payload_sha256'] == canonical_hash(protocol),
          'Canonical protocol envelope differs')
    _validate_protocol(protocol)
    identity = worker['canonical_identity']; batch_id = worker['canonical_batch_id']
    _need(set(identity) == set(PLAN_IDENTITY_KEYS)
          and identity['method_id'] in METHODS and METHODS[identity['method_id']] == worker['method_id'],
          'Canonical worker method differs')
    matches = [row for batch in protocol['batches'] if batch['batch_id'] == batch_id
               for row in batch['expected_plans'] if row == identity]
    _need(len(matches) == 1 and worker['plan_id'] == identity['plan_id']
          and worker['target_id'] == identity['target_id'] and worker['domain'] == protocol['domain']
          and worker['budget'] == protocol['configuration']['planning_budget'],
          'Canonical identity/domain/protocol budget differs')
    sources = worker['planning_source_references']
    from . import formal_record_settlement, campaign_batch
    _need(isinstance(sources, list) and sources and all(file_reference(path) in sources for path in
          (__file__, formal_record_settlement.__file__, campaign_batch.__file__)),
          'Canonical queue/settlement/publisher source not bound')
    for ref in sources:
        verify_reference(ref)
    inputs = worker['plan_input_references']
    _need(set(inputs) == INPUTS, 'Canonical physical input inventory differs')
    for ref in inputs.values():
        verify_reference(ref)
    candidate_record = _read(numeric['candidate_record_reference']); candidate = candidate_record['candidate']
    plan = candidate['plan']; digest = canonical_hash(plan)
    _need(set(plan) == {'x', 'r', 'task_sha256', 'config_hash'} and digest == numeric['plan_sha256']
          and numeric['domain'] == protocol['domain'] and numeric['method_id'] == worker['method_id']
          and numeric['target_id'] == identity['target_id'] and numeric['plan_id'] == identity['plan_id'],
          'Original numerical plan identity differs')
    tasks, support = _read(inputs['tasks']), _read(inputs['planning_support'])
    _need(tasks.get('task_sha256') == plan['task_sha256']
          and inputs['tasks'] == inputs['routes'] == inputs['initial_state'], 'Original task inputs differ')
    if worker['method_id'] == 'SAA':
        audit = _read(numeric['numerical_audit_reference'])
        joint = _read(audit['candidate_audit_reference'])
        from .saa_record_bridge import joint_physical_binding
        ids, binding = joint_physical_binding(joint, worker, candidate_record)
    else:
        ids, binding = candidate['support_ids'], candidate['input_binding']
    expected_ids = support.get('scenario_ids')
    if expected_ids is None:
        expected_ids = [row['scenario_id'] for row in support['scenarios']]
    _need(isinstance(ids, list) and ids and len(ids) == len(set(ids)) and ids == expected_ids,
          'Full ordered planning support differs')
    if protocol['domain'] == 'formal':
        _need(len(ids) == 160 and _normal(binding['task_source']) == inputs['tasks']
              and _normal(binding['target']['planning_index']) == inputs['planning_support']
              and binding['formal_protocol_lock'] == worker['protocol_reference'], 'Formal physical source differs')
    root = Path(output).resolve()
    # The output is a new child of this original attempt, never an inferred
    # campaign/data destination; ordinary frozen inputs remain untouched.
    _need(root.parent == Path(receipt_reference['path']).parent and root.name == 'canonical_record',
          'Canonical output must be the current attempt child')
    root.mkdir(exist_ok=False)
    execution = dict(execution_mode='OR' if worker['method_id'] == 'OR' else 'EA',
                     departure_h=0 if worker['method_id'] == 'OR' else 1)
    plan_ref = _new(root/'plan.json', dict(plan, **execution))
    base = dict(**identity, plan_sha256=digest,
                references=dict(**inputs, plan=plan_ref, planning_source=sources))
    payload = dict(kind='CANONICAL_PREPARATION_V1', policy=POLICY, receipt_reference=receipt_reference,
                   worker_request_reference=worker_request_reference, base_record=base,
                   candidate_record_reference=numeric['candidate_record_reference'],
                   domain=worker['domain'], batch_id=batch_id, preparation_seconds=time.monotonic()-started,
                   before_final_cutoff=True, formal_OOS=False, scientific_acceptance=False)
    ref = _new(root/'preparation.json', payload)
    return _PreparedPublication(ref, deepcopy(payload), root)


def complete_publication(prepared, finalization_reference, *, attempt_started, charged_attempt_seconds, deadline):
    """Write only small canonical metadata within the already allocated seal."""
    _need(isinstance(prepared, _PreparedPublication), 'Private in-process canonical preparation required')
    _need(_read(prepared.reference) == prepared.payload, 'Canonical preparation changed before final seal')
    seal = _read(finalization_reference)
    _need(seal['status'] == 'PASS_FINAL_RECORD_SEAL'
          and seal['receipt_reference'] == prepared.payload['receipt_reference']
          and seal['charged_attempt_seconds'] == charged_attempt_seconds, 'Canonical original seal differs')
    root = prepared.root
    record = deepcopy(prepared.payload['base_record'])
    record['references'].update(**seal['references'], budget_seal=finalization_reference)
    record_ref = _new(root/'plan_record.json', record)
    manifest_ref = _new(root/'manifest.json', dict(files=[prepared.reference, record['references']['plan'], record_ref]))
    actual = time.monotonic()-attempt_started
    _need(actual <= charged_attempt_seconds and time.monotonic() <= deadline,
          'Canonical publication exceeded original seal allocation')
    completion = dict(kind='CANONICAL_COMPLETION_V1', status='PASS_CANONICAL_IN_ATTEMPT_PUBLICATION', policy=POLICY,
                      preparation_reference=prepared.reference, canonical_record_reference=record_ref,
                      manifest_reference=manifest_ref, receipt_reference=prepared.payload['receipt_reference'],
                      worker_request_reference=prepared.payload['worker_request_reference'],
                      finalization_reference=finalization_reference,
                      actual_attempt_seconds_before_confirmation_write=actual,
                      charged_attempt_seconds=charged_attempt_seconds,
                      allocation_is_measured_solver_time=False, formal_OOS=False, scientific_acceptance=False)
    ref = _new(root/'completion.json', completion)
    _need(time.monotonic()-attempt_started <= charged_attempt_seconds and time.monotonic() <= deadline,
          'Canonical confirmation exceeded original seal allocation')
    return ref


def verify_publication(completion_reference, *, receipt_reference, finalization_reference, worker_request_reference):
    """Verify the immutable publication DAG and original cutoff/charge bindings."""
    completion = _read(completion_reference)
    worker = _read(worker_request_reference); seal = _read(finalization_reference)
    _need(worker.get('canonical_export_policy') == completion.get('policy') == POLICY
          and completion.get('kind') == 'CANONICAL_COMPLETION_V1'
          and completion.get('status') == 'PASS_CANONICAL_IN_ATTEMPT_PUBLICATION'
          and completion.get('receipt_reference') == receipt_reference
          and completion.get('worker_request_reference') == worker_request_reference
          and completion.get('finalization_reference') == finalization_reference,
          'Canonical completion policy or original references differ')
    _need(Path(completion_reference['path']) == Path(receipt_reference['path']).parent/'canonical_record/completion.json',
          'Canonical completion is outside original attempt')
    _need(not (Path(receipt_reference['path']).parent/'failure.json').exists(),
          'Canonical attempt has unresolved failure evidence')
    receipt = _read(receipt_reference)
    result = _read(receipt['result_reference'])
    invocation = _read(receipt['invocation_reference'])
    _need(receipt['status'] == result['status'] == 'COMPLETE' and receipt['returncode'] == 0
          and receipt['plan_id'] == invocation['plan_id'] == worker['plan_id']
          and invocation['domain'] == worker['domain']
          and invocation['item_request_reference'] == worker_request_reference
          and result['invocation_sha256'] == receipt['invocation_reference']['sha256'],
          'Canonical original complete invocation/result differs')
    preparation = _read(completion['preparation_reference'])
    record = _read(completion['canonical_record_reference'])
    manifest = _read(completion['manifest_reference'])
    expected_files = [completion['preparation_reference'], record['references']['plan'], completion['canonical_record_reference']]
    _need(manifest == {'files': expected_files}, 'Canonical manifest inventory differs')
    for ref in expected_files:
        verify_reference(ref)
    _need(preparation.get('kind') == 'CANONICAL_PREPARATION_V1' and preparation.get('policy') == POLICY
          and preparation.get('before_final_cutoff') is True
          and preparation.get('receipt_reference') == receipt_reference
          and preparation.get('worker_request_reference') == worker_request_reference,
          'Canonical original preparation differs')
    expected = deepcopy(preparation['base_record'])
    expected['references'].update(**seal['references'], budget_seal=finalization_reference)
    _need(record == expected and all(record[k] == worker['canonical_identity'][k] for k in PLAN_IDENTITY_KEYS)
          and record['references']['planning_source'] == worker['planning_source_references']
          and all(record['references'][k] == worker['plan_input_references'][k] for k in INPUTS),
          'Canonical record/worker/preparation binding differs')
    for ref in record['references']['planning_source']:
        verify_reference(ref)
    numeric = _read(seal['references']['solver_record'])
    budget = _read(seal['references']['budget'])
    for key in ('receipt_reference', 'request_reference'):
        _need(numeric[key] == (receipt_reference if key == 'receipt_reference' else worker_request_reference),
              'Canonical numerical producer differs')
    _need(numeric['plan_sha256'] == record['plan_sha256'] and budget['planning_budget'] == worker['budget'],
          'Canonical numerical plan or original budget differs')
    actual = completion['actual_attempt_seconds_before_confirmation_write']
    charged = completion['charged_attempt_seconds']; cutoff = numeric['actual_attempt_seconds_at_cutoff']
    _need(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (actual, charged, cutoff))
          and cutoff <= seal['actual_attempt_seconds_before_confirmation_write'] <= actual <= charged
          == seal['charged_attempt_seconds'] == numeric['charged_attempt_seconds']
          == budget['charged_attempt_seconds']
          and numeric['prior_cumulative_wall_seconds']+charged <= worker['budget']['cumulative_wall_seconds']
          and completion.get('allocation_is_measured_solver_time') is False,
          'Canonical publication clock or charge differs')
    _need({p.name for p in Path(completion_reference['path']).parent.iterdir()} ==
          {'plan.json', 'plan_record.json', 'manifest.json', 'preparation.json', 'completion.json'},
          'Canonical publication file inventory differs')
    return completion
