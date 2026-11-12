"""Reuse audited original recourse primals as a complete, nonbinding SAA hint.

This creates no bound or scientific acceptance. The original native interface
must check the assembled vector against every current model row before use.
"""
import hashlib
import time
from pathlib import Path
import numpy as np
from . import campaign_batch as batch


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(ref):
    batch.verify_reference(ref)
    return batch.read_json(ref['path'])


def one_artifact(record, name):
    refs = [r for r in record['artifacts'] if Path(r['path']).name == name]
    need(len(refs) == 1, 'Missing or duplicate original artifact: ' + name)
    batch.verify_reference(refs[0])
    return refs[0]


def map_primal(vector, mapping, original_columns, width):
    v = np.asarray(vector)
    m = np.asarray(mapping)
    need(v.shape == (original_columns,) and np.isfinite(v).all(), 'Original primal shape/finiteness differs')
    need(m.ndim == 1 and m.dtype.kind in 'iu' and m.size == width, 'Retained-column mapping shape/type differs')
    need(len(np.unique(m)) == width and (m >= 0).all() and (m < original_columns).all(), 'Retained-column mapping duplicates/out of range')
    return np.asarray(v[m], dtype=float)


def build_candidate_warm_start(*, model_manifest_reference, evidence, deadline):
    """Read only small original primals/mappings; caller charges this preparation.

    evidence has candidate_review, candidate_audit, candidate_checkpoint,
    candidate_plan and candidate_snapshot strong references. No solver is called.
    """
    started = time.monotonic()
    need(np.isfinite(deadline) and deadline > started, 'Warm-start deadline expired')
    model = read(model_manifest_reference)
    review = read(evidence['candidate_review'])
    audit = read(evidence['candidate_audit'])
    cp = read(evidence['candidate_checkpoint'])
    snapshot = read(evidence['candidate_snapshot'])
    plan = read(evidence['candidate_plan'])
    need(review['passed'] is True and review['status'] == 'PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW' and review['references']['recourse_audit'] == evidence['candidate_audit'], 'Original independent candidate review differs')
    need(audit['passed'] is True and audit['status'] == 'PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT' and audit['checkpoint_reference'] == evidence['candidate_checkpoint'] and audit['raw_snapshot_reference'] == evidence['candidate_snapshot'], 'Original independent full-Q audit differs')
    ids = model['scenario_ids']
    need(model['domain'] == 'formal' and len(ids) == 160 and len(set(ids)) == 160 and len(model['scenarios']) == 160, 'Original formal 160 support required')
    need(cp['support_ids'] == audit['support_ids'] == ids and cp['completed_count'] == cp['next_index'] == audit['complete_lp_count'] == 160, 'Ordered audited support differs')
    need(cp['plan_reference'] == evidence['candidate_plan'] and cp['plan_sha256'] == audit['plan_sha256'] == review['plan_sha256'], 'Audited plan identity differs')
    need(cp['recourse_event_references'] == snapshot['events'] and len(snapshot['events']) == 160 and snapshot['formal_OOS'] is False, 'Original event snapshot differs')
    need(audit['original_first_stage_sha256'] == model['first_stage_sha256'], 'Original first-stage matrix identity differs')
    binding = model['input_binding']
    need(plan['task_sha256'] == binding['task_sha256'] and plan['config_hash'] == binding['config_hash'], 'Plan task/config differs')
    first = np.concatenate((np.asarray(plan['x'], dtype=float).ravel(), np.asarray(plan['r'], dtype=float).ravel()))
    need(np.isfinite(first).all(), 'Nonfinite first-stage plan')
    out = np.empty(model['shape'][1], dtype=float)
    need(first.size == model['scenarios'][0]['column_slice'][0], 'First-stage width differs')
    out[:first.size] = first
    next_column = first.size
    used = []
    for i, scene in enumerate(model['scenarios']):
        need(time.monotonic() < deadline, 'Warm-start preparation deadline expired')
        need(scene['index'] == i and scene['scenario_id'] == ids[i], 'Model scene order differs')
        lo, hi = scene['column_slice']
        need(type(lo) is int and type(hi) is int and lo == next_column and hi > lo and hi <= out.size, 'Model column coverage differs')
        event_ref = snapshot['events'][i]
        event = read(event_ref)
        need(event['event'] == 'recourse' and event['domain'] == 'formal' and event['scenario_index'] == i and event['scenario_id'] == ids[i] and event['plan_sha256'] == cp['plan_sha256'] and event['optimal_certificate_valid'] is True, 'Original event identity differs')
        pointer = one_artifact(event, 'template_reference.json')
        template_ref = read(pointer)
        template = read(template_ref)
        need(template['kind'] == 'recourse_template' and template['scenario_index'] == i and template['scenario_id'] == ids[i] and template['template_sha256'] == scene['template_sha256'], 'Original template identity differs')
        event_plan = one_artifact(event, 'plan.json')
        need(event_plan['sha256'] == evidence['candidate_plan']['sha256'] and event_plan['size_bytes'] == evidence['candidate_plan']['size_bytes'], 'Original event plan bytes differ')
        primal_ref = one_artifact(event, 'primal_vector.npz')
        mapping_ref = scene['mapping_reference']
        batch.verify_reference(mapping_ref)
        with np.load(primal_ref['path'], allow_pickle=False) as data:
            need(data.files == ['vector'], 'Unexpected primal archive arrays')
            raw = data['vector']
        mapping = np.load(mapping_ref['path'], allow_pickle=False)
        out[lo:hi] = map_primal(raw, mapping, scene['original_recourse_columns'], hi-lo)
        next_column = hi
        used.append(dict(event_reference=event_ref, template_reference=template_ref, primal_reference=primal_ref, mapping_reference=mapping_ref))
    need(next_column == out.size and time.monotonic() < deadline, 'Incomplete warm-start vector or expired deadline')
    return out, dict(kind='SAA_AUDITED_CANDIDATE_WARM_START_V1', model_manifest_reference=model_manifest_reference,
        evidence=evidence, artifact_references=used, plan_sha256=cp['plan_sha256'], scenario_ids=ids,
        vector_sha256=hashlib.sha256(out.astype('<f8').tobytes()).hexdigest(), vector_length=out.size,
        preparation_seconds=time.monotonic()-started, solver_calls=0, new_scientific_acceptance=False,
        original_model_feasibility_check_required=True, variables_fixed=False)
