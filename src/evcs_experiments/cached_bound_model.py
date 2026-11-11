"""Admit the audited partial R22 model after its separately settled failure.

Reuse never grants a bound: the source-bound original numerical audits remain
attached to the original coefficients. All verification belongs inside the new
supervised parent, and no recourse or probability solve is performed here.
"""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz

from .campaign_batch import verify_reference
from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256

KIND = 'S05_CACHED_MULTI_ANCHOR_BOUND_ONLY_V1'
OPENING = Decimal('5633.96054736330092')
REMAINING = Decimal('1566.03945263669908')
OLD_OPENING = Decimal('5438.82363936330092')
FAILED_SECONDS = Decimal('195.136908')
PLAN_ID = 'mechanism.michael_2018.EA_DRO.baseline'
PLAN_SHA = '33179f0050e20f734a9f1706e26a88e393e0c2ba79de75b46a2070d25857c1d7'


def need(value, message):
    if not value:
        raise ValueError(message)


def budget_guard(request, settlement, previous_review):
    need(request['kind'] == KIND and request['plan_id'] == PLAN_ID, 'Wrong cached attempt')
    need(settlement['kind'] == 'S05_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1'
         and settlement['status'] == 'SETTLED_BOUND_EVIDENCE_PENDING_FINAL_RESULT_REVIEW'
         and settlement['terminal_state'] == 'FAILED'
         and settlement['parent_job_id'] == 's05-multi-anchor-r22-20260909T053500Z'
         and settlement['plan_id'] == PLAN_ID and settlement['accounting_PASS'] is True
         and settlement['parent_charged_once'] is True and settlement['nested_seconds_added'] is False
         and settlement['cumulative_seconds_provisional'] is False, 'Unsettled original failed attempt')
    need(Decimal(settlement['opening_cumulative_seconds_decimal']) == OLD_OPENING
         and Decimal(settlement['terminal_parent_seconds_decimal']) == FAILED_SECONDS
         and Decimal(settlement['cumulative_accounted_seconds_decimal']) == OPENING
         and Decimal(settlement['remaining_seconds_decimal']) == REMAINING
         and settlement['cumulative_cap_seconds'] == 7200, 'Original failed time not inherited exactly once')
    need(previous_review['passed'] is True
         and previous_review['status'] == 'PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW'
         and previous_review['plan_id'] == PLAN_ID and previous_review['plan_sha256'] == PLAN_SHA
         and previous_review['upper_bound_usd'] == 227762.10627548306
         and previous_review['lower_bound_usd'] == 24135.73087520793
         and previous_review['references']['settlement'] == settlement['previous_settlement_reference'],
         'Original fixed candidate review changed')
    need(request['parent_job']['timeout_seconds'] == 1000
         and request['parent_job']['grace_seconds'] == 5 and request['parent_job']['threads'] == 1
         and request['wall_seconds'] == 990 and request['solver_seconds'] == 600
         and request['post_master_reserve_seconds'] == 100
         and request['preprocessing_charge_seconds'] == 0 and REMAINING >= Decimal(1015),
         'Changed cached-bound prospective limits')
    return OPENING


def load_cached_model(request, *, deadline):
    """Rehash exact old numerical evidence and original raw provenance, no solve."""
    def alive():
        need(np.isfinite(deadline) and time.monotonic() < deadline, 'Cached evidence admission deadline')
    def read(ref):
        alive(); verify_reference(ref)
        return json.loads(Path(ref['path']).read_text())
    refs = request['references']
    card = read(refs['cached_model_review'])
    need(card['kind'] == 'R22_PARTIAL_CUT_MODEL_REUSE_REVIEW_V1' and card['passed'] is True,
         'Independent partial-model reuse review required')
    cr = card['references']
    old_request = read(cr['original_request'])
    need(old_request['kind'] == 'S05_MULTI_ANCHOR_BOUND_ONLY_V1'
         and old_request['plan_id'] == PLAN_ID
         and old_request['adapter_references'] == request['adapter_references']
         and old_request['references']['previous_result_review'] == refs['previous_result_review'],
         'Cached original problem/source admission differs')
    settlement = read(refs['settlement'])
    need(settlement['request_reference'] == cr['original_request']
         and settlement['failure_reference'] == cr['failure']
         and settlement['parent_result_reference'] == cr['parent_result']
         and settlement['source_manifest_reference'] == cr['source_manifest'], 'Cached attempt/settlement differs')
    failure = read(cr['failure']); terminal = read(cr['parent_result'])
    need(terminal['complete'] is True and terminal['state'] == 'FAILED'
         and terminal['job_id'] == 's05-multi-anchor-r22-20260909T053500Z'
         and terminal['returncode'] == 1 and str(terminal['elapsed_seconds']) == str(FAILED_SECONDS)
         and failure['request_reference'] == cr['original_request']
         and failure['phase'] == 'native_master'
         and failure['message'] == 'Gurobi would discard matrix coefficients smaller than 1e-13',
         'Different failed attempt is not admitted')
    source = read(cr['source_manifest']); root = Path(cr['source_manifest']['path']).parent
    source_paths = set()
    for row in source['files']:
        alive(); path = (root / row['path']).resolve()
        need(path.is_relative_to(root) and str(path) not in source_paths, 'Original source escaped/duplicated release')
        source_paths.add(str(path))
        verify_reference(dict(row, path=str(path)))
    manifest = read(cr['failure_manifest']); admitted = {}
    need(manifest['request_reference'] == cr['original_request'], 'Partial manifest request differs')
    for ref in manifest['files']:
        alive(); verify_reference(ref)
        need(ref['path'] not in admitted, 'Duplicate cached artifact')
        admitted[ref['path']] = ref
    for key in ('failure', 'matrix', 'model_arrays', 'model_audit', 'candidate_cut_audit', 'assembly', 'input_binding'):
        need(admitted.get(cr[key]['path']) == cr[key], 'Unbound cached artifact: ' + key)
    ma = read(cr['model_audit']); axis = read(cr['candidate_cut_audit']); assembly = read(cr['assembly'])
    body = dict(ma); digest = body.pop('audit_sha256')
    need(canonical_hash(body) == digest and ma['passed'] is True
         and ma['kind'] == 'INDEPENDENT_MULTI_ANCHOR_MODEL_AUDIT_V1'
         and ma['anchor_count'] == 2 and len(ma['support_ids']) == 160,
         'Original independent model audit changed')
    need(axis['kind'] == 'INDEPENDENT_CANDIDATE_CUT_AXIS_V1' and axis['passed'] is True
         and axis['complete_cut_count'] == 160 and axis['support_ids'] == ma['support_ids']
         and axis['ambiguity_sha256'] == ma['ambiguity_sha256'] and axis['plan_sha256'] == PLAN_SHA,
         'Original complete candidate numerical audit differs')
    binding = read(cr['input_binding'])
    body = dict(binding); digest = body.pop('binding_sha256')
    need(canonical_hash(body) == digest and binding['references'] == request['adapter_references']
         and binding['plan_sha256'] == PLAN_SHA and binding['support_ids'] == ma['support_ids']
         and binding['ambiguity_sha256'] == ma['ambiguity_sha256'], 'Original input binding differs')
    body = dict(assembly); digest = body.pop('construction_sha256')
    need(canonical_hash(body) == digest and assembly['anchor_count'] == 2
         and assembly['support_ids'] == ma['support_ids']
         and assembly['ambiguity_sha256'] == ma['ambiguity_sha256']
         and assembly['master_coefficient_sha256'] == ma['coefficient_sha256'], 'Original assembly binding differs')
    seen = set()
    def graph(value):
        if isinstance(value, list):
            for item in value: graph(item)
        elif isinstance(value, dict):
            if 'path' in value and 'sha256' in value and Path(value['path']).is_absolute():
                key = (value['path'], value['sha256'])
                if key in seen: return
                seen.add(key); alive(); verify_reference(value)
                path = Path(value['path'])
                # Descriptive identity arrays contain no external references.
                if path.suffix == '.json' and path.name != 'identity.json':
                    graph(json.loads(path.read_text()))
            else:
                for item in value.values(): graph(item)
    out = Path(cr['matrix']['path']).parent
    # The independently reviewed original first-stage and source graph must
    # remain current byte-for-byte as well as the per-atom numerical evidence.
    for ref in binding['references'].values():
        alive(); verify_reference(ref)
    graph(binding['references']['original_first_record'])
    graph(binding['source_references'])
    for i, sid in enumerate(ma['support_ids']):
        ref = admitted.get(str(out / 'atoms' / f'{i:03d}.json'))
        need(ref is not None, 'Missing original audited atom')
        atom = read(ref); audit = atom['audit']
        need(audit['passed'] is True and audit['kind'] == 'INDEPENDENT_CANDIDATE_CUT_ATOM_V1'
             and audit['index'] == i and audit['scenario_id'] == sid and audit['plan_sha256'] == PLAN_SHA
             and audit['cut_sha256'] == ma['cut_sha256s'][160+i], 'Original numerical atom/assembly differs')
        need(set(audit['source_references']) == {'event_reference','template_reference','anchor_plan_reference'}
             and atom['provenance']['context_binding'] == binding
             and atom['provenance']['context_binding_sha256'] == binding['binding_sha256'],
             'Original atom context/raw source binding differs')
        cut = dict(atom['cut']); digest = cut.pop('cut_sha256')
        need(canonical_hash(cut) == digest == audit['cut_sha256'], 'Original atom cut changed')
        gradient_ref = atom['gradient_reference']
        need(admitted.get(gradient_ref['path']) == gradient_ref, 'Unbound cached gradient')
        g = np.load(gradient_ref['path'], allow_pickle=False)
        need(hashlib.sha256(np.asarray(g,dtype='<f8').tobytes()).hexdigest()
             == cut['gradient_sha256'] == audit['gradient_sha256'], 'Original atom gradient changed')
        graph(audit['source_references'])
    with np.load(cr['model_arrays']['path'], allow_pickle=False) as z:
        model = {k:z[k].copy() for k in z.files}
    model['objective_constant_usd'] = float(model['objective_constant_usd'])
    model['matrix'] = load_npz(cr['matrix']['path']).tocsr()
    need(coefficient_sha256(model) == ma['coefficient_sha256'] == card['original_coefficient_sha256'],
         'Cached original coefficients differ')
    alive()
    return model, ma, assembly, dict(kind='CACHED_R22_MODEL_ADMISSION_V1', passed=True,
        review_reference=refs['cached_model_review'], original_coefficient_sha256=ma['coefficient_sha256'],
        raw_references_rehashed=len(seen), numerical_audits_reused=True, solver_calls=0,
        scientific_acceptance=False)
