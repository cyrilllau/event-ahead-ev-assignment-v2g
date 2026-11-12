"""Source-bound core work inventory; does not load models, launch, or lock OOS."""
from pathlib import Path
import json
from .evaluation_lock import file_reference, _verify_reference, PLAN_IDENTITY_KEYS
from .random_streams import canonical_hash

# Protocol names and worker names intentionally differ for EA only.
CORE = {
    'EA_DRO': ('EA', 'formal_planning_worker', 'formal_planning', 'load_formal_problem', 'mechanism'),
    'OR': ('OR', 'formal_planning_worker', 'formal_onset', 'load_formal_onset_problem', 'mechanism'),
    'SAA': ('SAA', 'saa_planning_worker', 'formal_saa', 'prepare_formal_saa', 'risk_model'),
    'ARO': ('ARO', 'formal_planning_worker', 'formal_planning', 'load_formal_problem', 'risk_model'),
}
SETTINGS = {'mechanism': 'lead_h=baseline', 'physical_chain': 'ablation=baseline',
            'risk_model': 'risk_objective=baseline'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(ref):
    _verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def protocol(ref):
    envelope = read(ref)
    body = envelope['payload']
    need(envelope.get('kind') == 'protocol_lock' and envelope.get('payload_sha256') == canonical_hash(body),
         'Protocol envelope differs')
    targets = body['target_ids']
    need(body['domain'] == 'formal' and len(targets) == len(set(targets)) == 8,
         'Complete formal eight-target scope required')
    return body


def source_references():
    return [file_reference(p) for p in sorted(Path(__file__).parent.rglob('*.py'))]


def prepare_core_request(protocol_reference, identity, planning_source_references):
    """Describe an exact existing factory; returned metadata is not a job request."""
    body = protocol(protocol_reference)
    need(planning_source_references == source_references(), 'Current complete source closure differs')
    return _descriptor(body, protocol_reference, identity, planning_source_references)


def _descriptor(body, protocol_reference, identity, planning_source_references):
    need(set(identity) == set(PLAN_IDENTITY_KEYS), 'Exact plan identity required')
    method = identity['method_id']
    need(method in CORE, 'Unsupported core method; no baseline fallback')
    matches = [(b, p) for b in body['batches'] for p in b['expected_plans'] if p == identity]
    need(len(matches) == 1 and identity['target_id'] in body['target_ids'], 'Undeclared or ambiguous identity')
    batch = matches[0][0]['batch_id']
    need(batch in SETTINGS and identity['setting_id'] == SETTINGS[batch]
         and (method == 'EA_DRO' or batch == CORE[method][4]), 'Unsupported core setting')
    worker, module, factory_module, factory, _ = CORE[method]
    return dict(kind='PREPARED_CORE_METHOD_DESCRIPTOR_V1', identity=dict(identity),
                protocol_reference=protocol_reference, worker_method_id=worker,
                worker_module='evcs_experiments.'+module,
                factory='evcs_experiments.'+factory_module+'.'+factory,
                planning_source_references=planning_source_references,
                deployed=False, queue_ready=False, formal_OOS=False, scientific_acceptance=False)


def build_core_worklist(protocol_reference, planning_source_references, accepted=None):
    """List 32 unique core plans, retaining exact accepted-record reuse references.

    Reuse is a metadata assertion backed by a component acceptance, not a new
    numerical audit. Execution must revalidate records at its normal gateway.
    """
    body = protocol(protocol_reference)
    need(planning_source_references == source_references(), 'Current complete source closure differs')
    accepted = {} if accepted is None else accepted
    items = []
    for target in body['target_ids']:
        for method, entry in CORE.items():
            rows = [p for b in body['batches'] if b['batch_id'] == entry[4]
                    for p in b['expected_plans'] if p['target_id'] == target and p['method_id'] == method]
            need(len(rows) == 1, 'Missing or duplicate core configuration')
            identity = rows[0]
            item = _descriptor(body, protocol_reference, identity, planning_source_references)
            item['action'] = 'PREPARE'
            item['protocol_aliases'] = [dict(p) for b in body['batches'] if b['batch_id'] in SETTINGS
                                       for p in b['expected_plans']
                                       if p['target_id'] == target and p['method_id'] == method]
            for alias in item['protocol_aliases']:
                _descriptor(body, protocol_reference, alias, planning_source_references)
            if identity['plan_id'] in accepted:
                refs = accepted[identity['plan_id']]
                need(set(refs) == {'plan_record_reference', 'acceptance_reference'}, 'Incomplete accepted record binding')
                record, card = read(refs['plan_record_reference']), read(refs['acceptance_reference'])
                need(all(record.get(k) == v for k, v in identity.items()), 'Foreign accepted plan')
                need(card.get('status') == 'PASS_COMPONENT'
                     and card.get('component_id') == 'INDIVIDUAL_FORMAL_NUMERICAL_RECORD'
                     and card.get('plan_record_reference') == refs['plan_record_reference'], 'Unaccepted plan record')
                item.update(action='REUSE_ACCEPTED_RECORD', accepted_references=refs)
            items.append(item)
    need(set(accepted) <= {i['identity']['plan_id'] for i in items}, 'Unknown accepted configuration')
    return dict(kind='PREPARED_CORE_WORKLIST_V1', items=items, configuration_count=32,
                deployed=False, queue_ready=False, formal_OOS=False, scientific_acceptance=False)
