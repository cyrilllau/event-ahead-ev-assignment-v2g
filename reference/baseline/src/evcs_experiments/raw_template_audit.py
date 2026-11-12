"""Independent admission-envelope checks for original and referenced templates.

This adapter validates source association only. Numerical auditors must still
check the actual new plan's RHS, primal/dual, physical witness and probability.
It imports no exporter and does not assemble or solve an optimization problem.
"""
import hashlib
import json
from pathlib import Path

from .recourse_template_index import ReviewedTemplateIndex, REFERENCE_KIND


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read(reference):
    need(set(reference) == {'path', 'sha256', 'size_bytes'}, 'Strict audit reference required')
    p = Path(reference['path'])
    need(p.is_absolute() and p.resolve() == p and p.is_file(), 'Noncanonical audit reference')
    need(p.stat().st_size == reference['size_bytes'], 'Audit artifact size differs')
    h = hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    need(h.hexdigest() == reference['sha256'], 'Audit artifact hash differs')
    return p


def document(reference):
    return json.loads(read(reference).read_text())


def record(reference):
    p = read(reference)
    need(p.name == 'record.json', 'Unknown raw record name')
    value = json.loads(p.read_text())
    refs = value['artifacts']
    need(len({r['path'] for r in refs}) == len(refs), 'Duplicate raw artifact')
    need(all(Path(r['path']).parent == p.parent for r in refs), 'Foreign raw artifact directory')
    need({Path(r['path']).name for r in refs} ==
         {q.name for q in p.parent.iterdir() if q.name != 'record.json'}, 'Uncommitted raw artifact')
    for ref in refs:
        read(ref)
    return value


def audit_template_envelope(snapshot_reference, *, expected_request_reference):
    """Map local admitted records to numeric originals and validate LP pointers.

    Both snapshot and request are externally pinned by the caller's immutable
    attempt receipt. Empty/partial support can be structurally valid; the
    returned coverage is explicit and is never a full-support numerical PASS.
    """
    snapshot = document(snapshot_reference)
    need(snapshot['request'] == expected_request_reference, 'Foreign envelope request')
    request = document(expected_request_reference)
    need(request['formal_OOS'] is False and snapshot['formal_OOS'] is False and
         request['domain'] in ('development', 'formal'), 'Wrong raw envelope domain')
    root = Path(snapshot['evidence_root'])
    need(root.is_absolute() and root.resolve() == root and
         Path(expected_request_reference['path']) == root/'request.json', 'Foreign envelope root')
    need(Path(snapshot['first_stage']['path']) == root/'first_stage/record.json', 'Foreign first stage')
    first = record(snapshot['first_stage'])
    need(first['kind'] == 'first_stage', 'Wrong first-stage kind')
    identity = request['problem_identity'];ids = identity['scenario_ids']
    need(ids and len(set(ids)) == len(ids), 'Invalid request support')
    hashes = (identity['lazy_input_binding']['scenario_sha256s']
              if 'lazy_input_binding' in identity else identity['scenario_sha256s'])
    need(len(hashes) == len(ids), 'Missing physical scenario identities')
    provider = request['input_binding'].get('template_provider')
    reviewed = None
    if provider is not None:
        need(set(provider) == {'template_index_reference', 'template_association_review_reference'},
             'Incomplete provider admission')
        reviewed = ReviewedTemplateIndex(provider['template_index_reference'],
                                         provider['template_association_review_reference'])
        need(reviewed.index['domain'] == request['domain'] and
             reviewed.index['first_stage_sha256'] == identity['first_stage_sha256'] and
             reviewed.index['scenario_ids'] == ids and reviewed.index['scenario_sha256s'] == hashes,
             'Provider does not cover this physical request')
    templates = snapshot['templates']
    need([r['path'] for r in templates] == [str(p) for p in sorted((root/'templates').glob('*/record.json'))]
         and len(templates) == len(list((root/'templates').iterdir())), 'Template inventory differs')
    mapped = {}
    for reference in templates:
        local = record(reference);i = local['scenario_index']
        need(type(i) is int and 0 <= i < len(ids) and i not in mapped and
             local['scenario_id'] == ids[i] and Path(reference['path']) == root/'templates'/f'{i:03d}'/'record.json',
             'Local template addressing differs')
        if reviewed is None:
            need(local['kind'] == 'recourse_template', 'Pointer lacks run admission')
            original_reference, original = reference, local
        else:
            need(local['kind'] == REFERENCE_KIND and len(local['artifacts']) == 1 and
                 Path(local['artifacts'][0]['path']).name == 'template_source.json', 'Invalid admission pointer')
            pointer = document(local['artifacts'][0])
            need(all(pointer.get(k) == v for k, v in provider.items()) and
                 pointer['scenario_index'] == i, 'Pointer admission differs from run')
            row = reviewed.row(i);original_reference = row['original_template_reference']
            need(pointer['original_template_reference'] == original_reference, 'Pointer selects foreign original')
            original = record(original_reference)
        need(original['kind'] == 'recourse_template' and all(original[k] == local[k] for k in
             ('scenario_index', 'scenario_id', 'template_sha256')), 'Original template identity differs')
        names = {Path(r['path']).name for r in original['artifacts']}
        need(names == {'A_ub.npz', 'A_eq.npz', 'Bx_ub.npz', 'Bx_eq.npz', 'Br_ub.npz', 'Br_eq.npz',
                       'arrays.npz', 'identity.json'}, 'Original numeric inventory differs')
        mapped[i] = dict(local_reference=reference, original_reference=original_reference,
                         template_sha256=original['template_sha256'], scenario_id=ids[i])
    events = snapshot['events']
    need([r['path'] for r in events] == [str(p) for p in sorted((root/'events').glob('*/record.json'))]
         and len(events) == len(list((root/'events').iterdir())), 'Event inventory differs')
    need(json.loads((root/'history.json').read_text()) == events, 'Raw journal differs')
    recourse_events = []
    for sequence, reference in enumerate(events):
        event = record(reference)
        need(Path(reference['path']).parent.name.split('_')[0] == f'{sequence:06d}' and
             event['domain'] == request['domain'], 'Event order/domain differs')
        if event['event'] != 'recourse':
            continue
        i = event['scenario_index']
        need(type(i) is int and i in mapped and event['scenario_id'] == ids[i], 'LP template missing')
        refs = {Path(r['path']).name: r for r in event['artifacts']}
        need('template_reference.json' in refs and
             document(refs['template_reference.json']) == mapped[i]['original_reference'],
             'LP callback selects a template outside run admission')
        recourse_events.append(dict(reference=reference, scenario_index=i,
                                   iteration=event['iteration'], plan_sha256=event['plan_sha256']))
    return dict(status='PASS_TEMPLATE_ENVELOPE_ONLY', domain=request['domain'],
                snapshot_reference=snapshot_reference, request_reference=expected_request_reference,
                template_count=len(mapped), support_count=len(ids),
                templates=[mapped[i] for i in sorted(mapped)], recourse_events=recourse_events,
                full_support_verified=False, numerical_audit_passed=False, scientific_acceptance=False,
                formal_OOS=False, solver_calls=0)
