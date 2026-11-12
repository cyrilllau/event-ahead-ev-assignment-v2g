"""Publish a new, ordinary sealed worker result without repeating numerical work.

Historical auxiliary/composite records are deliberately unsupported. Publication
is currently admitted only for development: its separately measured metadata
work happens after the queue seal. Formal admission needs a reviewed integration
of that work into the original accounting boundary, not an uncharged exporter.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

from . import formal_record_settlement
from .campaign_batch import _finalization_charge
from .config import get_paths
from .evaluation_lock import (PLAN_IDENTITY_KEYS, _reference_list, _validate_protocol,
                              _verify_reference, file_reference)
from .random_streams import canonical_hash

KIND = 'SEALED_PLAN_RECORD_REQUEST_V1'
METHODS = {'EA_DRO': 'EA', 'ARO': 'ARO', 'OR': 'OR', 'SAA': 'SAA'}
INPUTS = {'tasks', 'routes', 'initial_state', 'planning_support'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _read(reference):
    _verify_reference(reference)
    return json.loads(Path(reference['path']).read_text(),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return file_reference(path)


def _normal(ref):
    result = dict(path=ref['path'], sha256=ref['sha256'],
                  size_bytes=ref.get('size_bytes', ref.get('bytes')))
    _verify_reference(result)
    return result


def validated_sealed_payload(request):
    """Validate the original seal/receipt/audit and return unchanged references.

    This is metadata reconstruction using the existing settlement validator; it
    does not invoke a numerical solver or perform a fresh scientific acceptance.
    The result is not authorization for formal publication or evaluation.
    """
    _need(set(request) == {'kind', 'identity', 'batch_id', 'references',
                          'plan_input_references', 'planning_source'} and
          request['kind'] == KIND, 'Exact sealed publisher request required')
    refs = request['references']
    _need(set(refs) == {'protocol', 'worker_request', 'budget_seal'},
          'Original protocol, worker request and seal required')
    envelope, worker, seal = (_read(refs[k]) for k in
                              ('protocol', 'worker_request', 'budget_seal'))
    protocol = envelope['payload']
    _need(envelope['kind'] == 'protocol_lock' and
          envelope['payload_sha256'] == canonical_hash(protocol), 'Protocol envelope differs')
    _validate_protocol(protocol)
    identity = request['identity']
    _need(set(identity) == set(PLAN_IDENTITY_KEYS), 'Exact canonical identity required')
    matches = [p for batch in protocol['batches'] if batch['batch_id'] == request['batch_id']
               for p in batch['expected_plans'] if p == identity]
    _need(len(matches) == 1, 'Canonical identity not predeclared in batch')
    _need(identity['method_id'] in METHODS and
          worker.get('method_id') == METHODS[identity['method_id']], 'Worker method differs')
    _need(worker.get('canonical_identity') == identity and
          worker.get('canonical_batch_id') == request['batch_id'] and
          worker.get('plan_id') == identity['plan_id'] and
          worker.get('target_id') == identity['target_id'] and
          worker.get('domain') == protocol['domain'] and
          worker.get('protocol_reference') == refs['protocol'], 'Worker canonical binding differs')
    _need(worker['budget'] == protocol['configuration']['planning_budget'],
          'Frozen protocol budget differs')
    sources = request['planning_source']
    _reference_list(sources, 'sealed plan sources')
    _need(sources == worker.get('planning_source_references') and
          all(file_reference(p) in sources for p in
              (__file__, formal_record_settlement.__file__)), 'Publisher or settlement source differs')
    inputs = request['plan_input_references']
    _need(set(inputs) == INPUTS and inputs == worker.get('plan_input_references'),
          'Worker input reference binding differs')
    for ref in inputs.values():
        _verify_reference(ref)

    _need(seal.get('status') == 'PASS_FINAL_RECORD_SEAL' and
          set(seal.get('references', {})) == {'solver_record', 'budget'}, 'Original seal required')
    numeric, budget = (_read(seal['references'][k]) for k in ('solver_record', 'budget'))
    _need(numeric.get('request_reference') == refs['worker_request'] and
          numeric.get('receipt_reference') == seal.get('receipt_reference'), 'Seal request or receipt differs')
    receipt = _read(seal['receipt_reference'])
    result = _read(receipt['result_reference'])
    _need(receipt.get('status') == result.get('status') == numeric.get('worker_status') == 'COMPLETE',
          'Only complete sealed worker evidence can be published')
    prior = numeric['prior_cumulative_wall_seconds']
    _finalization_charge(seal['receipt_reference'], result, prior, refs['budget_seal'])
    original = formal_record_settlement.settle_plan_record(
        receipt_reference=seal['receipt_reference'],
        numerical_audit_reference=numeric['numerical_audit_reference'],
        prior_cumulative_seconds=prior, planning_budget=worker['budget'])
    # Use the original finalizer itself, writing only disposable reconstruction
    # files. This checks every native/status/accounting field, including new ones.
    with tempfile.TemporaryDirectory(prefix='evcs-seal-reconstruction-') as tmp:
        reconstructed = formal_record_settlement.finalize_settled_record(
            original, actual_attempt_seconds_at_cutoff=numeric['actual_attempt_seconds_at_cutoff'],
            final_seal_reserve_seconds=numeric['final_seal_allocated_seconds'],
            output_directory=Path(tmp)/'record')
    _need(reconstructed['solver_record'] == numeric and reconstructed['budget_record'] == budget,
          'Sealed numerical or budget fields differ from original settlement')
    _need(seal.get('seal_allocation_seconds') == numeric['final_seal_allocated_seconds'] and
          seal.get('allocation_is_measured_solver_time') is False, 'Seal allocation semantics differ')
    candidate_record = _read(numeric['candidate_record_reference'])
    candidate = candidate_record['candidate']
    plan = candidate['plan']
    _need(set(plan) == {'x', 'r', 'task_sha256', 'config_hash'} and
          canonical_hash(plan) == numeric['plan_sha256'], 'Original plan identity differs')
    _need(numeric['domain'] == protocol['domain'] and numeric['method_id'] == worker['method_id'] and
          numeric['target_id'] == identity['target_id'] and numeric['plan_id'] == identity['plan_id'],
          'Original numeric identity differs')
    task_data, support = _read(inputs['tasks']), _read(inputs['planning_support'])
    _need(task_data.get('task_sha256') == plan['task_sha256'], 'Input task semantic identity differs')
    _need(inputs['tasks'] == inputs['routes'] == inputs['initial_state'],
          'Ordinary worker routes and initial state must retain the same task source')
    if worker['method_id'] == 'SAA':
        audit = _read(numeric['numerical_audit_reference'])
        joint = _read(audit['candidate_audit_reference'])
        from .saa_record_bridge import joint_physical_binding
        ids, binding = joint_physical_binding(joint, worker, candidate_record)
    else:
        ids, binding = candidate['support_ids'], candidate['input_binding']
    expected_ids = support.get('scenario_ids')
    if expected_ids is None:
        expected_ids = [s['scenario_id'] for s in support['scenarios']]
    _need(isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids) and ids == expected_ids,
          'Full ordered planning support differs')
    if protocol['domain'] == 'formal':
        _need(len(ids) == 160 and _normal(binding['task_source']) == inputs['tasks'] and
              _normal(binding['target']['planning_index']) == inputs['planning_support'] and
              binding['formal_protocol_lock'] == refs['protocol'], 'Formal original inputs differ')
    execution = dict(execution_mode='OR' if worker['method_id'] == 'OR' else 'EA',
                     departure_h=0 if worker['method_id'] == 'OR' else 1)
    return dict(identity=identity, domain=protocol['domain'], plan=dict(plan, **execution),
                plan_sha256=numeric['plan_sha256'],
                references=dict(**inputs, **seal['references'], budget_seal=refs['budget_seal'],
                                planning_source=sources))


def prepare_sealed_plan_record(request, *, output):
    """Append a canonical development record; exact reuses never overwrite it."""
    started = time.monotonic()
    value = validated_sealed_payload(request)
    _need(value['domain'] == 'development',
          'Formal publication requires reviewed publication accounting integration')
    out = Path(output).resolve()
    paths = get_paths()
    _need(not any(out.is_relative_to(Path(p).resolve()) for p in
                  (paths.data_root, paths.frozen_source_root, Path(__file__).parent)),
          'Output overlaps frozen data or active source')
    all_refs = list(request['references'].values()) + list(request['plan_input_references'].values()) + request['planning_source']
    _need(not any(Path(ref['path']).resolve().is_relative_to(out) for ref in all_refs),
          'Output contains an original input or source')
    reused = out.exists()
    if reused:
        _need(json.loads((out/'request.json').read_text()) == request, 'Existing publication request differs')
        manifest = json.loads((out/'manifest.json').read_text())
        expected_files = [file_reference(out/name) for name in
                          ('request.json', 'plan.json', 'plan_record.json')]
        _need(manifest == {'files': expected_files}, 'Existing publication manifest differs')
        _need({p.name for p in out.iterdir()} == {'request.json', 'plan.json', 'plan_record.json', 'manifest.json'},
              'Existing publication inventory differs')
        _need(json.loads((out/'plan.json').read_text()) == value['plan'], 'Existing published plan differs')
    else:
        out.mkdir(parents=True, exist_ok=False)
        _new(out/'request.json', request)
        _new(out/'plan.json', value['plan'])
    record = dict(**value['identity'], plan_sha256=value['plan_sha256'],
                  references=dict(value['references'], plan=file_reference(out/'plan.json')))
    if reused:
        _need(json.loads((out/'plan_record.json').read_text()) == record, 'Existing canonical record differs')
    else:
        _new(out/'plan_record.json', record)
        _new(out/'manifest.json', dict(files=[file_reference(out/name) for name in
                                            ('request.json', 'plan.json', 'plan_record.json')]))
    return dict(status='DEVELOPMENT_CANONICAL_RECORD_PREPARED',
                plan_record_reference=file_reference(out/'plan_record.json'),
                manifest_reference=file_reference(out/'manifest.json'),
                verified_reuse=reused, metadata_seconds=time.monotonic()-started,
                metadata_accounting='separate post-seal administration; not numerical solver time',
                production_admission=False, solver_calls=0, formal_OOS=False, scientific_acceptance=False)
