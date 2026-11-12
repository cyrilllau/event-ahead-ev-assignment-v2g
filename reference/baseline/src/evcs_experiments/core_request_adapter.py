"""Build real requests for the existing core workers and serial campaign queue.

This module performs input-contract validation, not numerical acceptance or
job admission. Formal workers retain all of their source, parent, bank, and
resource gates. Tiny tests use the workers' existing private dependency fixtures;
there is deliberately no development CLI.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from . import campaign_batch as queue
from .core_method_registry import CORE, _descriptor, protocol, source_references
from .evaluation_lock import PLAN_IDENTITY_KEYS
from .planning_audit_pipeline import validate_audit_configuration
from .planning_budget import validate_budget
from .random_streams import canonical_hash


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _read(reference):
    queue.verify_reference(reference)
    return queue.read_json(reference['path'])


@dataclass(frozen=True)
class _Context:
    domain: str
    body: dict
    protocol_reference: dict
    config_reference: dict
    planning_source_references: list
    budget: dict


def _context(protocol_reference, config_reference, planning_source_references, *, domain):
    envelope = _read(protocol_reference)
    body = envelope['payload']
    _need(envelope.get('kind') == 'protocol_lock'
          and envelope.get('payload_sha256') == canonical_hash(body)
          and body.get('domain') == domain, 'Protocol domain/envelope differs')
    config = _read(config_reference)
    _need(body['configuration'] == config, 'Frozen config/protocol differs')
    budget = validate_budget(config['planning_budget'])
    _need(planning_source_references == source_references(), 'Current complete source closure differs')
    if domain == 'formal':
        protocol(protocol_reference)
        expected = dict(cumulative_wall_seconds=7200, single_master_seconds=600,
                        final_certificate_reserve_seconds=1790, shutdown_reserve_seconds=10,
                        master_threads=4, recourse_workers=1, blas_threads=1,
                        normal_gap_relative=.005, rmp_mip_gap=.001)
        _need(budget == expected, 'Frozen formal planning resources/gaps differ')
    return _Context(domain, deepcopy(body), deepcopy(protocol_reference),
                    deepcopy(config_reference), deepcopy(planning_source_references), deepcopy(budget))


def prepare_core_context(*, protocol_reference, config_reference, planning_source_references):
    """Validate the common formal metadata once for a fixed request list."""
    return _context(protocol_reference, config_reference, planning_source_references, domain='formal')


def _development_context(*, protocol_reference, config_reference, planning_source_references):
    """Private tiny-test context; cannot accept a formal protocol."""
    return _context(protocol_reference, config_reference, planning_source_references, domain='development')


def _identity(context, identity, batch_id=None):
    _need(isinstance(context, _Context), 'Validated core request context required')
    _need(set(identity) == set(PLAN_IDENTITY_KEYS) and identity['method_id'] in CORE,
          'Exact supported core identity required; no baseline fallback')
    matches = [batch for batch in context.body['batches']
               for row in batch['expected_plans'] if row == identity]
    _need(len(matches) == 1 and identity['target_id'] in context.body['target_ids'],
          'Undeclared or ambiguous core identity')
    if batch_id is not None:
        _need(matches[0]['batch_id'] == batch_id, 'Canonical batch/identity differs')
    if context.domain == 'formal':
        _descriptor(context.body, context.protocol_reference, identity, context.planning_source_references)
    return CORE[identity['method_id']]


def build_core_worker_request(context, *, identity, canonical_batch_id, plan_input_references,
                              bank_acceptance_reference, summary_reference, audit_reference,
                              output_root, numerical_audit=None, saa_options=None,
                              template_index_reference=None, template_association_review_reference=None,
                              onset_acceptance_reference=None):
    """Return the actual ordinary/SAA worker schema, with canonical export inputs.

    Every physical/source/configuration/budget binding is supplied explicitly.
    Existing history and preparation carryovers are intentionally not inferred;
    this API constructs new core configurations, while queue resume reuses the
    resulting immutable request without rebuilding it.
    """
    worker_method, _, _, _, _ = _identity(context, identity, canonical_batch_id)
    _need(set(plan_input_references) == {'tasks', 'routes', 'initial_state', 'planning_support'},
          'Exact canonical physical input references required')
    for ref in plan_input_references.values():
        queue.verify_reference(ref)
    card = _read(bank_acceptance_reference)
    _read(summary_reference)
    _read(audit_reference)
    if context.domain == 'formal':
        if worker_method == 'OR':
            from .formal_onset import validate_onset_acceptance
            _need(onset_acceptance_reference == bank_acceptance_reference,
                  'OR requires its exact onset acceptance, never baseline substitution')
            validate_onset_acceptance(onset_acceptance_reference=onset_acceptance_reference,
                                      summary_reference=summary_reference, audit_reference=audit_reference,
                                      protocol_reference=context.protocol_reference)
        else:
            _need(card.get('status') == 'PASS_COMPONENT'
                  and card.get('summary_reference') == summary_reference
                  and card.get('audit_reference') == audit_reference
                  and card.get('protocol_reference') == context.protocol_reference,
                  'Accepted bank does not bind requested inputs/protocol')
    _need(isinstance(output_root, (str, Path)) and Path(output_root).is_absolute(),
          'Absolute explicit output root required')
    request = dict(schema_version=1, domain=context.domain, spec_id='S05', model_id='IEEE33-v54-Q1',
                   plan_id=identity['plan_id'], target_id=identity['target_id'], method_id=worker_method,
                   canonical_identity=deepcopy(identity), canonical_batch_id=canonical_batch_id,
                   plan_input_references=deepcopy(plan_input_references),
                   protocol_reference=context.protocol_reference, config_reference=context.config_reference,
                   planning_source_references=context.planning_source_references,
                   bank_acceptance_reference=bank_acceptance_reference,
                   summary_reference=summary_reference, audit_reference=audit_reference,
                   budget=context.budget, output_root=str(Path(output_root)))
    _need((template_index_reference is None) == (template_association_review_reference is None),
          'Template index and association review must be supplied together')
    if template_index_reference is not None:
        queue.verify_reference(template_index_reference)
        queue.verify_reference(template_association_review_reference)
        request.update(template_index_reference=template_index_reference,
                       template_association_review_reference=template_association_review_reference)
    if worker_method == 'OR':
        _need(onset_acceptance_reference == bank_acceptance_reference,
              'OR requires its exact onset acceptance, never baseline substitution')
        request['onset_acceptance_reference'] = onset_acceptance_reference
    else:
        _need(onset_acceptance_reference is None, 'Only OR may bind onset acceptance')
    if worker_method == 'SAA':
        _need(not isinstance(saa_options, dict) or saa_options.get('model_input_mode') != 'l_shaped_formal',
              'Formal L-shaped worker and whole-parent admission are not integrated; extensive dispatch prohibited')
        _need(numerical_audit is None and isinstance(saa_options, dict),
              'SAA requires explicit SAA phase options, not ordinary audit configuration')
        if saa_options.get('model_input_mode') == 'l_shaped_development':
            from .saa_l_shaped import ALGORITHM_ID
            _need(context.domain == 'development' and
                  set(saa_options) == {'model_input_mode', 'algorithm_id',
                                       'helper_reference', 'template_references'} and
                  saa_options['algorithm_id'] == ALGORITHM_ID,
                  'L-shaped fixture request is development only with exact options')
            queue.verify_reference(saa_options['helper_reference'])
            refs = saa_options['template_references']
            _need(isinstance(refs, list) and refs, 'L-shaped requires original template inventory')
            for ref in refs:
                queue.verify_reference(ref)
            request.update(saa_options)
            return deepcopy(request)
        required = {'model_input_mode', 'max_coefficient_bytes', 'helper_reference', 'template_references'}
        optional = {'model_checkpoint_reserve_seconds', 'model_reload_reserve_seconds'}
        _need(required <= set(saa_options) <= required | optional
              and saa_options['model_input_mode'] == 'assemble',
              'New SAA request requires explicit fresh assembly; no free preassembled model')
        _need(type(saa_options['max_coefficient_bytes']) is int and saa_options['max_coefficient_bytes'] > 0,
              'SAA assembly requires an explicit positive coefficient ceiling')
        queue.verify_reference(saa_options['helper_reference'])
        refs = saa_options['template_references']
        _need(isinstance(refs, list) and refs, 'SAA requires original template inventory')
        for ref in refs:
            queue.verify_reference(ref)
        for key in optional & set(saa_options):
            queue._positive(saa_options[key], key, maximum=300)
        request.update(saa_options)
    else:
        _need(saa_options is None and numerical_audit is not None,
              'Ordinary worker requires its numerical audit; no SAA options')
        reserve = validate_audit_configuration(numerical_audit, formal=context.domain == 'formal',
                                               planning_sources=context.planning_source_references)
        _need(reserve < context.budget['final_certificate_reserve_seconds'],
              'Audit reserve must fit existing final evidence reserve')
        request['numerical_audit'] = numerical_audit
    return deepcopy(request)


def build_core_queue(context, *, batch_id, queue_root, job_wall_seconds, budget_reference,
                     items, executable, cwd, _development_argv=None):
    """Return an ordinary campaign_batch request; no launch or readiness claim."""
    _need(isinstance(context, _Context), 'Validated core request context required')
    _need(queue._id(batch_id), 'Invalid batch identity')
    _need(Path(queue_root).is_absolute(), 'Absolute queue root required')
    executable, cwd = str(executable), str(cwd)
    _need(Path(executable).is_absolute() and Path(executable).is_file()
          and Path(cwd).is_absolute() and Path(cwd).is_dir(), 'Explicit executable/cwd required')
    queue._positive(job_wall_seconds, 'job_wall_seconds')
    reserve = context.budget['shutdown_reserve_seconds']
    _need(reserve < job_wall_seconds, 'Shutdown reserve consumes job budget')
    _need(_read(budget_reference) == context.budget, 'Queue budget reference differs from worker budget')
    _need(isinstance(items, list) and items, 'Nonempty fixed core item list required')
    if _development_argv is not None:
        _need(context.domain == 'development' and isinstance(_development_argv, list)
              and _development_argv and _development_argv[0] == executable
              and all(isinstance(arg, str) and '\0' not in arg for arg in _development_argv),
              'Private test argv cannot replace a formal worker')
    result_items, seen, plans, outputs = [], set(), set(), set()
    for item in items:
        _need(set(item) == {'item_id', 'identity', 'request_reference'}, 'Exact core queue item fields required')
        identity = item['identity']
        method, module, _, _, _ = _identity(context, identity)
        _need(queue._id(item['item_id']) and item['item_id'] not in seen
              and queue._id(identity['plan_id']) and identity['plan_id'] not in plans,
              'Invalid or duplicate item/plan identity')
        seen.add(item['item_id']); plans.add(identity['plan_id'])
        request = _read(item['request_reference'])
        _need(request.get('model_input_mode') != 'l_shaped_formal',
              'Formal L-shaped worker and whole-parent admission are not integrated; extensive dispatch prohibited')
        for key, expected in [('domain', context.domain), ('schema_version', 1), ('spec_id', 'S05'),
                              ('model_id', 'IEEE33-v54-Q1'), ('canonical_identity', identity),
                              ('method_id', method), ('plan_id', identity['plan_id']),
                              ('target_id', identity['target_id']), ('budget', context.budget),
                              ('planning_source_references', context.planning_source_references),
                              ('protocol_reference', context.protocol_reference),
                              ('config_reference', context.config_reference)]:
            _need(request.get(key) == expected, 'Worker/queue binding differs: ' + key)
        _identity(context, identity, request['canonical_batch_id'])
        root = str(Path(request['output_root']).resolve())
        _need(root not in outputs, 'Distinct plans cannot share an output root')
        outputs.add(root)
        argv = [executable, '-m', 'evcs_experiments.' + module]
        if _development_argv is not None:
            argv = _development_argv
        result_items.append(dict(item_id=item['item_id'], plan_id=identity['plan_id'],
                                 request_reference=item['request_reference'],
                                 plan_budget_seconds=context.budget['cumulative_wall_seconds'],
                                 argv=argv, cwd=cwd))
    return deepcopy(dict(schema_version=1, domain=context.domain, spec_id='S05', model_id='IEEE33-v54-Q1',
                         batch_id=batch_id, queue_root=str(queue_root), job_wall_seconds=job_wall_seconds,
                         shutdown_reserve_seconds=reserve, source_references=context.planning_source_references,
                         budget_reference=budget_reference, items=result_items))
