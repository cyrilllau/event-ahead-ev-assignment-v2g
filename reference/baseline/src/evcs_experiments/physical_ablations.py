"""Pure pre-OOS physical support transformations; no bank/solver admission.

All mappings and their random stream addresses are explicit inputs. This module
does not allocate a production RNG namespace, alter existing random streams,
regenerate scenario banks, or register new planning methods. Array axis zero is
the ordered sample axis within exactly one target and one split.
"""
from dataclasses import dataclass
import re
from types import MappingProxyType

import numpy as np

from .joint_scenarios import array_hash
from .random_streams import DEVELOPMENT_ROOT, Stream, canonical_hash


BLOCK_FIELDS = {
    'weather_grid': frozenset(('grid_hazard', 'line_wind_vectors_m_s', 'road_wind_vectors_m_s',
        'line_outages', 'line_operating_outages', 'line_standard_normals', 'line_threshold_uniforms')),
    'road_vehicle': frozenset(('road_blocked', 'road_movable', 'road_travel_time_min', 'road_corridor_uniforms',
        'task_arrival_h', 'task_gamma_route', 'task_route_time_h', 'task_travel_energy_kwh',
        'task_arrival_soc', 'task_varsigma', 'task_mu_full', 'task_usable_energy_dc_kwh', 'task_failed_arc_ids')),
    'load': frozenset(('load_p_kw', 'load_q_kvar', 'load_archived_q_kvar', 'load_epsilon', 'load_standard_innovation')),
}
IDENTITY_FIELDS = {'method_id', 'target_id', 'split', 'domain', 'base_support_sha256'}
BINARY_FIELDS = {'line_outages', 'line_operating_outages', 'road_blocked', 'road_movable',
                 'task_gamma_route', 'task_varsigma', 'task_mu_full'}
POSITIVE_INFINITY_FIELDS = {'road_travel_time_min', 'task_arrival_h'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _numeric(value):
    array = np.asarray(value)
    _need(array.dtype.kind in 'biuf' and not array.dtype.hasobject,
          'Only real numeric/bool arrays are supported; object dtype is forbidden')
    return array


def support_digest(value):
    """Hash array names/shapes/dtypes/ordered bytes without interpreting units."""
    def tree(node):
        if isinstance(node, (dict, MappingProxyType)):
            _need(node and all(isinstance(k, str) and k for k in node), 'Named nonempty array inventory required')
            return {key: tree(node[key]) for key in sorted(node)}
        return {'array_sha256': array_hash(_numeric(node))}
    return canonical_hash(tree(value))


def _readonly(value):
    result = np.array(value, copy=True); result.setflags(write=False)
    return result


def _scope(identity, method, digest):
    _need(isinstance(identity, dict) and set(identity) == IDENTITY_FIELDS,
          'Exact method/target/split/domain/base-support identity required')
    _need(identity['method_id'] == method, 'Ablation method identity differs')
    _need(isinstance(identity['target_id'], str) and identity['target_id'], 'Explicit target identity required')
    _need(identity['split'] in ('planning', 'calibration'), 'Ablations must not transform OOS samples')
    _need(identity['domain'] == 'development',
          'Development-only component; formal bank and random-stream admission are not implemented')
    _need(isinstance(identity['base_support_sha256'], str)
          and re.fullmatch('[0-9a-f]{64}', identity['base_support_sha256'])
          and identity['base_support_sha256'] == digest, 'Base support array identity differs')
    return dict(identity)


def _stream(value, identity):
    if isinstance(value, Stream):
        stream = value; descriptor = value.as_dict()
    else:
        _need(isinstance(value, dict)
              and set(value) == {'root_seed', 'spawn_key', 'stream_id', 'bit_generator'},
              'Explicit Stream or complete Stream descriptor required')
        stream = Stream(value['root_seed'], tuple(value['spawn_key']))
        descriptor = stream.as_dict()
        _need(descriptor == value, 'Stream descriptor identity differs')
    _need(type(stream.root_seed) is int and stream.root_seed >= 0
          and isinstance(stream.spawn_key, tuple) and len(stream.spawn_key) == 9
          and all(type(v) is int and v >= 0 for v in stream.spawn_key), 'Invalid explicit stream address')
    _need(stream.root_seed == DEVELOPMENT_ROOT, 'Development component must retain the development random root')
    _need(stream.spawn_key[0] == 1
          and stream.spawn_key[1] == (2 if identity['domain'] == 'development' else 1)
          and stream.spawn_key[2] == (1 if identity['split'] == 'planning' else 2)
          and stream.spawn_key[3] == 0, 'Stream domain/split/shared-target address differs')
    return stream, descriptor


def _permutation(indices, size, stream):
    value = np.asarray(indices)
    _need(value.shape == (size,) and value.dtype.kind in 'iu', 'Integer sample permutation required')
    _need(np.array_equal(np.sort(value), np.arange(size)), 'Each mapping must be a sample-index bijection')
    _need(np.array_equal(value, stream.generator().permutation(size)), 'Indices differ from the explicit random stream')
    return np.array(value, dtype=np.int64, copy=True)


def _provenance(identity, mapping_hashes, descriptors, *, required, forbidden):
    return dict(kind='PHYSICAL_ABLATION_COMPONENT_V1', identity=dict(identity),
        mapping_sha256=mapping_hashes, stream_descriptors=descriptors,
        component_only=True, formal_bank_admitted=False, formal_OOS=False, scientific_acceptance=False,
        formal_random_namespace_registered=False, source_target_split_binding_requires_bank_adapter=True,
        required_downstream_steps=list(required), forbidden_postprocessing=list(forbidden))


@dataclass(frozen=True)
class AblationResult:
    arrays: object
    sample_indices: object
    provenance: dict


def no_line_dependence(line_outages, *, sample_indices, streams, identity):
    """Permute every (replay slot,line) column, preserving its exact marginal.

    Arrays have shape (sample,replay_slot,line). Only the full replay-grid line
    states enter here. Hourly maxima must be derived downstream from this result;
    the lost line persistence must not be repaired with a running maximum.
    """
    source = _numeric(line_outages)
    scope = _scope(identity, 'no_line_dependence', support_digest(source))
    _need(source.ndim == 3 and all(source.shape) and np.isin(source, (0, 1)).all(),
          'Binary nonempty (sample,replay_slot,line) outages required')
    count, slots, lines = source.shape
    supplied = np.asarray(sample_indices)
    _need(supplied.shape == source.shape and supplied.dtype.kind in 'iu', 'Line mapping shape/integer dtype differs')
    _need(isinstance(streams, (list, tuple)) and len(streams) == slots
          and all(isinstance(row, (list, tuple)) and len(row) == lines for row in streams),
          'One independent stream for every replay-slot/line cell is required')
    indices = np.empty(source.shape, dtype=np.int64); descriptors = []; seen = set()
    for slot in range(slots):
        row = []
        for line in range(lines):
            stream, desc = _stream(streams[slot][line], scope)
            _need(desc['stream_id'] not in seen, 'Distinct cells require distinct stream addresses')
            seen.add(desc['stream_id']); row.append(desc)
            indices[:, slot, line] = _permutation(supplied[:, slot, line], count, stream)
        descriptors.append(row)
    transformed = _readonly(np.take_along_axis(source, indices, axis=0))
    proof = _provenance(scope, array_hash(indices), descriptors,
        required=('line_operating_outages_from_changed_full_replay', 'road_replay_same_exogenous_draws',
                  'fixed_route_task_replay', 'all_five_features', 'calibration_bootstrap_and_moment_bounds',
                  'new_scenario_identity_and_donor_ledger'),
        forbidden=('maximum_accumulate_on_lines', 'independent_hourly_line_permutation', 'reuse_baseline_moments'))
    proof.update(output_arrays_sha256=support_digest(transformed), per_cell_empirical_marginals_preserved=True,
                 line_persistence_preserved=False, physical_line_process=False,
                 full_replay_time_axis_requires_bank_validation=True, axis_order=['sample', 'replay_slot', 'line'])
    return AblationResult(transformed, _readonly(indices), proof)


def independent_recombination(blocks, *, sample_indices, streams, identity):
    """Use three independent without-replacement permutations of whole atoms.

    Each block's fields retain the same donor, including its entire time/asset
    structure. Derived road/task arrays are not replayed after recombination:
    doing so would reintroduce the dependence this experiment removes.
    """
    _need(isinstance(blocks, dict) and set(blocks) == set(BLOCK_FIELDS), 'Exact three physical blocks required')
    scope = _scope(identity, 'independent_recombination', support_digest(blocks))
    _need(isinstance(sample_indices, dict) and isinstance(streams, dict)
          and set(sample_indices) == set(streams) == set(BLOCK_FIELDS), 'Exact three block index/stream inventories required')
    checked = {}; count = None
    for block, expected_fields in BLOCK_FIELDS.items():
        _need(isinstance(blocks[block], dict) and set(blocks[block]) == expected_fields,
              'Complete physical field inventory required for '+block)
        checked[block] = {}
        for field, values in blocks[block].items():
            array = _numeric(values)
            _need(array.ndim >= 1 and array.shape[0] > 0,
                  'Every field needs a nonempty sample axis: '+field)
            if count is None: count = array.shape[0]
            _need(array.shape[0] == count, 'Block sample count differs')
            allowed = np.isfinite(array) | (np.isposinf(array) if field in POSITIVE_INFINITY_FIELDS else False)
            _need(allowed.all(), 'NaN/negative infinity or unsupported positive infinity in '+field)
            if field in BINARY_FIELDS:
                _need(np.isin(array, (0, 1)).all(), 'Binary physical field required: '+field)
            if field == 'task_failed_arc_ids':
                _need(array.dtype.kind in 'iu', 'Failed arc identifiers require integer dtype')
            checked[block][field] = array
    transformed, indices, descriptors = {}, {}, {}; seen = set()
    for block in BLOCK_FIELDS:
        stream, desc = _stream(streams[block], scope)
        _need(desc['stream_id'] not in seen, 'Independent blocks require distinct stream addresses')
        seen.add(desc['stream_id']); descriptors[block] = desc
        index = _permutation(sample_indices[block], count, stream); indices[block] = _readonly(index)
        transformed[block] = MappingProxyType({name: _readonly(array[index]) for name, array in checked[block].items()})
    proof = _provenance(scope, {name: array_hash(value) for name, value in indices.items()}, descriptors,
        required=('all_five_features', 'calibration_bootstrap_and_moment_bounds', 'new_scenario_identity_and_donor_ledger'),
        forbidden=('road_replay', 'task_replay', 'line_replay', 'reuse_baseline_moments', 'copy_features_or_old_ledger'))
    proof.update(output_arrays_sha256=support_digest(transformed), block_empirical_marginals_preserved=True,
                 within_block_time_asset_structure_preserved=True, sampling_rule='independent_uniform_permutation_without_replacement',
                 weather_block_contains_road_wind=True)
    return AblationResult(MappingProxyType(transformed), MappingProxyType(indices), proof)


def road_support_digest(model, times_h, vectors, thresholds, line_replay, *, lead_h, regime):
    """Bind the unchanged upstream numerical inputs of a single road replay."""
    return canonical_hash(dict(road_model_sha256=model.model_sha256, lead_h=lead_h, regime=regime,
        model_parameters=dict(arc_ids=list(model.arc_ids), corridor_ids=list(model.corridor_ids),
            line_ids=list(model.line_ids), w_thr_m_s=model.w_thr_m_s, w_sev_m_s=model.w_sev_m_s,
            gamma_cap=model.gamma_cap, r3_multiplier=model.r3_multiplier),
        threshold_identity=dict(base_event=thresholds.base_event, variant_index=thresholds.variant_index,
            split=thresholds.split, development=thresholds.development, stream_ids=list(thresholds.stream_ids)),
        arrays_sha256=support_digest(dict(times_h=np.asarray(times_h), vectors=np.asarray(vectors),
            corridor_uniforms=thresholds.corridor_uniforms, arc_uniforms=thresholds.arc_uniforms,
            line_outages=line_replay.outages, line_times_h=line_replay.times_h, gamma=model.gamma,
            arc_corridor_indices=model.arc_corridor_indices, t0_min=model.t0_min,
            kappa_tt=model.kappa_tt, w_mov_m_s=model.w_mov_m_s, kappa_blk=model.kappa_blk,
            alpha0=model.alpha0, alpha1=model.alpha1))))


@dataclass(frozen=True)
class RoadAblationResult:
    replay: object
    provenance: dict


def no_line_road_coupling(*, model, times_h, vectors, thresholds, line_replay, lead_h, regime, identity):
    """Delegate gamma=0 to the existing road replay using the original draws."""
    from .road_disruptions import RoadModel
    _need(isinstance(model, RoadModel), 'Existing physical RoadModel required')
    digest = road_support_digest(model, times_h, vectors, thresholds, line_replay, lead_h=lead_h, regime=regime)
    scope = _scope(identity, 'no_line_road_coupling', digest)
    _need(thresholds.split == scope['split'] and thresholds.development == (scope['domain'] == 'development'),
          'Road threshold domain/split differs')
    result = model.replay(times_h, vectors, thresholds, line_replay,
                          lead_h=lead_h, regime=regime, no_coupling=True)
    _need(result.no_coupling is True and np.all(result.effective_gamma == 0)
          and result.thresholds is thresholds, 'Existing gamma-zero replay contract differs')
    proof = _provenance(scope, None, list(thresholds.stream_ids),
        required=('fixed_route_task_replay', 'all_five_features', 'calibration_bootstrap_and_moment_bounds',
                  'new_scenario_identity_and_donor_ledger'),
        forbidden=('redraw_thresholds', 'set_all_roads_clear', 'reuse_baseline_moments'))
    proof.update(existing_kernel='RoadModel.replay(no_coupling=True)',
                 upstream_draws_unchanged=True, wind_only_blockage_retained=True,
                 road_persistence_retained=True)
    return RoadAblationResult(result, proof)
