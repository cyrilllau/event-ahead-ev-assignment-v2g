"""Original-array OOS group audit for exact user-accepted Case0.1 plans.

Only the group admission/source wrapper differs from the legacy audit. Every
item uses the unchanged independent coefficient, primal/dual and physics check.
"""
import json
import time
from pathlib import Path

from .campaign_batch import file_reference, verify_reference
from .case0_evaluation import _write
from .case0_evaluation_audit import (audit_case0_evaluation_item,
    OriginalTemplateAuditCache, validate_case0_oos_axis,
    verify_evaluation_parent_settlement, policy_source_correspondence)
from .case0_evaluation_runner import prepare_case0_evaluation_lock
from .case0_accepted_evaluation import evaluation_sources, KIND


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def check_sources(producer_sources, manifest_ref, checker_ref):
    """Numerical checker must be the unchanged inherited manifest-bound file."""
    manifest = read(manifest_ref)
    root = Path(manifest_ref['path']).parent
    inventory = {r['path']: r['sha256'] for r in manifest['files']}
    for ref in producer_sources:
        verify_reference(ref)
        p = Path(ref['path'])
        if p.is_relative_to(root):
            need(inventory.get(str(p.relative_to(root))) == ref['sha256'],
                 'Inherited source differs from immutable manifest')
    verify_reference(checker_ref)
    need(checker_ref in producer_sources and Path(checker_ref['path']).is_relative_to(root)
         and Path(checker_ref['path']).name == 'case0_evaluation_audit.py',
         'Original independent numerical checker is not inherited/bound')


def audit_accepted_oos(request, deadline):
    from . import case0_evaluation_audit as numerical
    root = Path(request['root']).resolve()
    acceptance_ref = request['planning_acceptance_reference']
    acceptance = read(acceptance_ref)
    need(acceptance['kind'] == KIND, 'Exact accepted-comparison card required')
    state = json.loads((root/'state.json').read_text())
    progress = read(request['evaluation_state_reference'])
    need(state['active_configuration'] is None and not state.get('pending_evaluation_attempt'),
         'Evaluation parent not settled')
    settlement = verify_evaluation_parent_settlement(request['evaluation_parent_settlement_reference'], state)
    need(progress['status'] == 'COMPLETE_PENDING_INDEPENDENT_EVALUATION_AUDIT'
         and progress['active_item'] is None and progress.get('active_preparation') is None,
         'All original evaluation items must be complete')
    need(Path(request['evaluation_state_reference']['path']) == root/'evaluation/state.json',
         'Foreign completed evaluation state')
    # This reopens original inputs and validates existing lock equality. No LPs.
    lock_ref, contexts, plans = prepare_case0_evaluation_lock(acceptance_reference=acceptance_ref)
    need(lock_ref == request['lock_reference'] == progress['lock_reference'], 'Original OOS lock changed')
    lock = read(lock_ref)
    items = validate_case0_oos_axis(root, lock, progress)
    sources = evaluation_sources()
    need(sources == acceptance['evaluation_source_references'], 'Evaluator source changed')
    checker_ref = file_reference(numerical.__file__)
    inherited = request['inherited_source_manifest_reference']
    check_sources(sources, inherited, checker_ref)
    audit_sources = [*sources, file_reference(__file__)]
    composite = read(request['auditor_source_manifest_reference'])
    need(composite['kind'] == 'CASE0_ACCEPTED_OOS_COMPOSITE_SOURCE_V1'
         and composite['inherited_manifest_reference'] == inherited
         and composite['auditor_source_references'] == audit_sources,
         'Composite auditor source differs')
    review = read(request['independent_review_reference'])
    need(review['passed'] is True and review['auditor_source_references'] == audit_sources
         and review['original_item_checker_reference'] == checker_ref,
         'Exact audit wrapper/checker review missing')
    policy = policy_source_correspondence(inherited, inherited)
    out = Path(request['output_directory']).resolve()
    need(out.is_relative_to(root/'operations') and not out.exists(), 'New separate audit output required')
    out.mkdir(); (out/'items').mkdir()
    cache = OriginalTemplateAuditCache(capacity=1)
    reports = []
    for item in items:
        if time.monotonic() >= deadline:
            raise TimeoutError('Bounded accepted OOS audit deadline; saved item checks retained')
        method = item['method_id']
        ctx = contexts['OR_DRO' if method == 'OR_DRO' else 'EA_DRO']
        scene = ctx.scenario(item['sample_index'],split='oos',regime=item['regime'],
                            departure_h=0 if method=='DSP' else None)
        report = audit_case0_evaluation_item(record_reference=progress['completed'][item['item_id']],
            item=item,lock_reference=lock_ref,lock=lock,plan=plans[method],case=ctx.case,
            tasks=ctx.tasks,scene=scene,producer_source_references=sources,cache=cache,
            deadline=deadline,policy_sources=policy)
        _write(out/'items'/(item['item_id']+'.json'),report)
        reports.append(report)
        from .runtime.jobctl import atomic_json
        atomic_json(out/'progress.json',dict(checked=len(reports),total=768,solver_calls=0))
    for ref in [lock_ref, acceptance_ref, request['evaluation_state_reference'], *sources]:
        verify_reference(ref)
    for report in reports:
        verify_reference(report['record_reference'])
    for ref in cache.original_references.values():
        verify_reference(ref)
    need(time.monotonic() < deadline, 'Final audit seal deadline exceeded')
    card = dict(kind='CASE0_INDEPENDENT_EVALUATION_NUMERICAL_AUDIT_V1',passed=True,item_count=768,
        lock_reference=lock_ref,evaluation_state_reference=request['evaluation_state_reference'],
        planning_acceptance_reference=acceptance_ref,evaluation_parent_settlement_reference=settlement,
        producer_source_manifest_reference=inherited,producer_overlay_sources=sources,
        auditor_source_references=audit_sources,auditor_source_manifest_reference=request['auditor_source_manifest_reference'],
        independent_review_reference=request['independent_review_reference'],policy_source_correspondence=policy,
        original_coefficients_checked=True,fixed_plan_rhs_checked=True,full_primal_dual_kkt_checked=True,
        physical_metrics_checked=True,paired_axes_checked=True,solver_calls=0,
        template_assemblies=cache.assemblies,template_cache_hits=cache.hits,items=reports,
        planning_acceptance_mode=KIND,normal_planning_gap_passed=False,
        formal_OOS=False,Case1_acceptance=False,
        scope='Fixed Zeta48 development comparison with exact user-accepted plans; full independent OOS numerical checks, no normal planning convergence claim')
    _write(out/'audit.json',card)
    return card
