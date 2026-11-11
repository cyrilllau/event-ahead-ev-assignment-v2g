"""Append certified anchor cuts to one shared DRO epigraph; no solver or resume.

Original coefficient/dual source admission remains the caller's responsibility.
This constructor binds the supplied immutable source references and checks each
anchor against its derived cut records; it does not grant scientific acceptance.
"""
import hashlib
import math

import numpy as np
from scipy.sparse import vstack

from .campaign_batch import verify_reference
from .dual_cut_bound import _need
from .epigraph_master import build_epigraph_master
from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256


def build_multi_anchor_epigraph_master(first_stage_problem, support_ids, cut_sets,
                                      *, probability_set, floors):
    """Keep the first anchor's entire model and append only later cut rows.

    Each cut set has exactly ``anchor_first_stage``, ``cuts`` and nonempty
    ``source_references``. Every set covers the same complete scenario axis.
    References must pin the original derivation evidence, not mutable paths.
    One anchor produces exactly the existing single-anchor coefficients.
    """
    _need(isinstance(cut_sets, (list, tuple)) and len(cut_sets) > 0,
          'At least one complete anchor cut set is required')
    ids = tuple(support_ids)
    models, records, anchors = [], [], set()
    for group in cut_sets:
        _need(isinstance(group, dict) and set(group) ==
              {'anchor_first_stage', 'cuts', 'source_references'},
              'Unknown anchor cut-set schema')
        refs = group['source_references']
        _need(isinstance(refs, list) and len(refs) > 0,
              'Original anchor source references required')
        for ref in refs:
            verify_reference(ref)
        _need(len({ref['path'] for ref in refs}) == len(refs),
              'Duplicate anchor source reference')
        model, construction = build_epigraph_master(
            first_stage_problem, ids, group['cuts'],
            probability_set=probability_set, floors=floors)
        n = construction['original_first_stage_columns']
        anchor = np.asarray(group['anchor_first_stage'], dtype='<f8')
        _need(anchor.shape == (n,) and np.isfinite(anchor).all(),
              'Invalid original first-stage anchor vector')
        digest = hashlib.sha256(anchor.tobytes()).hexdigest()
        _need(digest not in anchors, 'Duplicate first-stage anchor')
        anchors.add(digest)
        for i in range(len(ids)):
            cut, gradient = group['cuts'][i]
            _need(cut['anchor_first_stage_sha256'] == digest,
                  'Cut belongs to a different first-stage anchor')
            value = float(cut['intercept_usd'] + np.asarray(gradient) @ anchor)
            q = cut['anchor_q_usd']
            _need(type(q) in (int, float) and math.isfinite(q),
                  'Nonfinite original anchor Q')
            tol = 1e-4 + 1e-7 * max(1., abs(q), abs(value))
            _need(cut['anchor_gap_tolerance_usd'] == tol and
                  abs(value - cut['anchor_affine_value_usd']) <= tol and
                  abs(q - value) <= tol and
                  abs((q - value) - cut['anchor_gap_usd']) <= tol,
                  'Original cut is not tight at its declared anchor')
        models.append(model)
        records.append(dict(anchor_first_stage_sha256=digest,
                            anchor_first_stage=anchor.tolist(),
                            source_references=refs, construction=construction))
    base = models[0]
    first_rows = records[0]['construction']['first_stage_rows']
    count = len(ids)
    # Keep all original first-stage rows, old cuts and floors byte-for-byte in
    # the matrix prefix. All extra rows use the same existing dual columns.
    result = dict(base)
    if len(models) > 1:
        blocks = [base['matrix']] + [m['matrix'][first_rows:first_rows + count]
                                      for m in models[1:]]
        result['matrix'] = vstack(blocks, format='csr')
        for key in ('row_lower', 'row_upper'):
            result[key] = np.concatenate([base[key]] +
                [m[key][first_rows:first_rows + count] for m in models[1:]])
    for group in cut_sets:
        for ref in group['source_references']:
            verify_reference(ref)
    record = dict(kind='MULTI_ANCHOR_FULL_SUPPORT_EPIGRAPH_V1',
        support_ids=list(ids), support_count=count, anchor_count=len(records),
        ambiguity_sha256=probability_set.ambiguity_sha256,
        anchor_sets=records,
        original_first_stage_coefficient_sha256=records[0]['construction']['original_first_stage_coefficient_sha256'],
        retained_single_anchor_coefficient_sha256=coefficient_sha256(base),
        master_coefficient_sha256=coefficient_sha256(result),
        retained_prefix_rows=base['matrix'].shape[0], appended_cut_rows=(len(records)-1)*count,
        matrix_shape=list(result['matrix'].shape),
        variable_slices=records[0]['construction']['variable_slices'],
        shared_ambiguity_dual_columns=True, old_cuts_and_floors_preserved=True,
        scenario_constants_added_once=True, solver_calls=0,
        bound_scope='Auxiliary optimum <= original DRO optimum; native bound needs separate audit',
        candidate_upper_bound_verified=False, scientific_acceptance=False, formal_OOS=False)
    record['construction_sha256'] = canonical_hash(record)
    return result, record
