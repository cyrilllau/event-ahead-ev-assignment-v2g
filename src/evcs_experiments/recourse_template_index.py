"""Ordered original-template references with explicit association-review gates.

Preparing an index never accepts its physical/source association. Production
use requires a separate review bound to the complete index and supporting
proof. No plan, Q, primal, dual or optimizer result is imported here.
"""
from pathlib import Path
import json

from .evaluation_lock import file_reference, _verify_reference
from .config import get_paths
from .recourse_store import KIND as BINDING_KIND, _record, load_template

INDEX_KIND = 'ORIGINAL_RECOURSE_TEMPLATE_INDEX_V1'
REVIEW_STATUS = 'PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW'
REFERENCE_KIND = 'recourse_template_reference_v1'


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def read(reference):
    _verify_reference(reference)
    return json.loads(Path(reference['path']).read_text())


def _axes(request):
    identity = request['problem_identity']
    ids = identity['scenario_ids']
    hashes = (identity['lazy_input_binding']['scenario_sha256s']
              if 'lazy_input_binding' in identity else identity['scenario_sha256s'])
    need(request['domain'] in ('formal', 'development') and request['formal_OOS'] is False,
         'Only explicit original planning/development evidence is supported')
    need(ids and len(ids) == len(hashes) and len(set(ids)) == len(ids), 'Invalid original ordered support')
    if request['domain'] == 'formal':
        need(len(ids) == 160, 'Formal import requires the complete original160 planning support')
    return ids, hashes, identity['first_stage_sha256']


def prepare_template_index(*, original_request_reference, original_snapshot_reference,
                           binding_references, proof_references, output):
    """Prepare metadata from an immutable full snapshot; no numerical acceptance.

    Each binding was already numerically checked by bind_template. Its source
    association must still be reviewed using the original audit/proof refs.
    Every actual load rehashes original numeric artifacts and checks semantics.
    """
    request = read(original_request_reference)
    snapshot = read(original_snapshot_reference)
    ids, hashes, first_sha = _axes(request)
    need(snapshot['request'] == original_request_reference, 'Snapshot/request association differs')
    need(isinstance(proof_references, dict) and proof_references, 'Original independent proof references required')
    for reference in proof_references.values():
        _verify_reference(reference)
    records = snapshot['templates']
    need(len(records) == len(ids) == len(binding_references), 'Incomplete original template support')
    need(len({r['path'] for r in records}) == len(records), 'Duplicate original template record')
    rows = []
    for i, (reference, record_reference) in enumerate(zip(binding_references, records)):
        binding = read(reference)
        record = read(record_reference)
        need(binding['kind'] == BINDING_KIND and binding['scientific_acceptance'] is False
             and binding['formal_OOS'] is False and binding['old_q_reused'] is False,
             'Invalid original numeric binding scope')
        need(binding['original_template_reference'] == record_reference and
             binding['first_stage_sha256'] == first_sha and record['kind'] == 'recourse_template',
             'Binding is outside the original snapshot or first stage')
        need(binding['scenario_index'] == record['scenario_index'] == i and
             binding['scenario_id'] == record['scenario_id'] == ids[i] and
             binding['template_sha256'] == record['template_sha256'], 'Template order/identity differs')
        need(original_request_reference in binding['source_references'].values(),
             'Binding does not cite its original request')
        rows.append(dict(scenario_index=i, scenario_id=ids[i], scenario_sha256=hashes[i],
                         template_sha256=record['template_sha256'], binding_reference=reference,
                         original_template_reference=record_reference))
    value = dict(kind=INDEX_KIND, status='PENDING_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW',
        domain=request['domain'], first_stage_sha256=first_sha, scenario_ids=ids,
        scenario_sha256s=hashes, rows=rows, original_request_reference=original_request_reference,
        original_snapshot_reference=original_snapshot_reference, proof_references=proof_references,
        formal_OOS=False, scientific_acceptance=False, old_q_reused=False)
    path = Path(output).resolve()
    paths=get_paths()
    protected=(Path(paths.data_root).resolve(),Path(paths.frozen_source_root).resolve(),
               Path(original_request_reference['path']).resolve().parent,Path(__file__).resolve().parent)
    need(not any(path.is_relative_to(root) for root in protected), 'Output overlaps frozen/original evidence or code')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return file_reference(path)


class ReviewedTemplateIndex:
    """The review is a required caller-supplied attestation, never auto-created."""
    def __init__(self, index_reference, association_review_reference):
        self.index_reference = dict(index_reference)
        self.review_reference = dict(association_review_reference)
        self.index = self.verify()

    @property
    def identity(self):
        return dict(template_index_reference=dict(self.index_reference),
                    template_association_review_reference=dict(self.review_reference))

    def verify(self):
        index, review = read(self.index_reference), read(self.review_reference)
        need(index['kind'] == INDEX_KIND and index['status'] == 'PENDING_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW'
             and index['scientific_acceptance'] is False and index['formal_OOS'] is False
             and index['old_q_reused'] is False, 'Invalid index scope')
        need(review['status'] == REVIEW_STATUS and review['passed'] is True
             and review['domain'] == index['domain'] and review['index_reference'] == self.index_reference
             and review['proof_references'] == index['proof_references']
             and review['scenario_ids'] == index['scenario_ids']
             and review['full_support_association_verified'] is True
             and review['formal_OOS'] is False, 'Incomplete or foreign template association review')
        import evcs_experiments.recourse_store as reader
        sources = review['implementation_sources']
        need(file_reference(__file__) in sources and file_reference(reader.__file__) in sources,
             'Template index/reader source is not reviewed')
        for reference in [*sources, *index['proof_references'].values()]:
            _verify_reference(reference)
        request, snapshot = read(index['original_request_reference']), read(index['original_snapshot_reference'])
        ids, hashes, first_sha = _axes(request)
        need(index['domain'] == request['domain'] and index['scenario_ids'] == ids
             and index['scenario_sha256s'] == hashes and index['first_stage_sha256'] == first_sha
             and snapshot['request'] == index['original_request_reference']
             and len(index['rows']) == len(ids) == len(snapshot['templates']), 'Original index/support changed')
        for i, row in enumerate(index['rows']):
            need(row['scenario_index'] == i and row['scenario_id'] == ids[i]
                 and row['scenario_sha256'] == hashes[i]
                 and row['original_template_reference'] == snapshot['templates'][i], 'Ordered index row changed')
        return index

    def row(self, index):
        need(type(index) is int and 0 <= index < len(self.index['rows']), 'Invalid explicit template index')
        current = self.verify()
        row = current['rows'][index]
        binding = read(row['binding_reference'])
        need(binding['original_template_reference'] == row['original_template_reference']
             and binding['scenario_index'] == index and binding['scenario_id'] == row['scenario_id']
             and binding['template_sha256'] == row['template_sha256']
             and binding['first_stage_sha256'] == current['first_stage_sha256'], 'Reviewed binding changed')
        return row


class TemplateProvider(ReviewedTemplateIndex):
    def __init__(self, index_reference, association_review_reference, *, first_stage,
                 case, tasks, scenario_ids, scenario_sha256s, domain):
        super().__init__(index_reference, association_review_reference)
        need(self.index['domain'] == domain and self.index['first_stage_sha256'] == first_stage.matrix_sha256
             and tuple(self.index['scenario_ids']) == tuple(scenario_ids)
             and tuple(self.index['scenario_sha256s']) == tuple(scenario_sha256s), 'Provider physical support differs')
        self.first_stage, self.case, self.tasks = first_stage, case, tasks

    def load(self, index, scene):
        row = self.row(index)
        need(scene.scenario_sha256 == row['scenario_sha256'], 'Provider scenario semantic identity differs')
        return load_template(row['binding_reference'], self.first_stage, self.case, self.tasks, scene,
            expected_template_sha256=row['template_sha256'], expected_scenario_index=index,
            expected_scenario_id=row['scenario_id'])

    def evidence_reference(self, index, template):
        row = self.row(index)
        need(template.template_sha256 == row['template_sha256'] and
             template.scenario_sha256 == row['scenario_sha256'] and
             template.first_stage.matrix_sha256 == self.index['first_stage_sha256'], 'Exported template differs')
        return dict(**self.identity, scenario_index=index,
                    original_template_reference=row['original_template_reference'])


def resolve_template_reference(source):
    """Validate one explicit pointer and return the original legacy record.

    Raw callbacks can continue pointing directly to original matrix files;
    the new run's snapshot independently preserves the admission provenance.
    """
    reviewed = ReviewedTemplateIndex(source['template_index_reference'],
                                     source['template_association_review_reference'])
    row = reviewed.row(source['scenario_index'])
    need(row['original_template_reference'] == source['original_template_reference'], 'Foreign template pointer')
    record, unused = _record(row['original_template_reference'])
    return record
