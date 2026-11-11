"""One complete in-memory development bank for independent recombination.

Accept already validated, independently pinned JointScenario atoms. Preserve
whole physical blocks, then recompute features and moment calibration. This is
not a formal bank exporter, a new random namespace, or a planning admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import (case_inputs, forecast_tasks, joint_scenarios, line_outages, moment_calibration,
               physical_ablations, random_streams, scenario_features, task_replay)
from .evidence import sha256
from .joint_scenarios import JointContext, JointScenario, JointTarget, array_hash
from .line_outages import operating_outages
from .moment_calibration import bootstrap_moments, calibrate_moments
from .physical_ablations import BLOCK_FIELDS, independent_recombination, support_digest
from .random_streams import DEVELOPMENT_ROOT, Stream, canonical_hash, event_order
from .scenario_features import compute_features

METHOD = 'independent_recombination'
SPLITS = ('planning', 'calibration')
IDENTITY_KEYS = ('scenario_id', 'target_event', 'base_weather_event', 'variant',
                 'variant_index', 'split', 'development', 'regime', 'lead_h', 'residual_donor_event')
LEDGER_KEYS = frozenset((*IDENTITY_KEYS, 'assignment_state_sha256', 'task_sha256',
    'grid_sha256', 'wind_parameters', 'wind_parameters_sha256', 'line_stream_id',
    'road_stream_ids', 'load_draw', 'task_replay_sha256', 'feature_sha256', 'array_hashes',
    'scenario_sha256'))
RAW_FIELDS = frozenset().union(*BLOCK_FIELDS.values())


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _file(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


def _context(context, prepared):
    _need(isinstance(context, JointContext) and isinstance(prepared, JointTarget),
          'Expected an original JointContext and prepared JointTarget')
    events = event_order(context.case.event_ids)
    _need(tuple(context.events) == events, 'Canonical eight-base context order differs')
    target_id = prepared.target.event_id
    _need(target_id in events, 'Unknown target identity')
    # Rebuild only the immutable task indices; this performs no route search or
    # scenario/task replay, and cannot trust caller-mutated prepared arrays.
    canonical = JointContext.prepare_target(context, target_id, prepared.bank)
    _need(prepared.target.state_sha256 == canonical.target.state_sha256 and
          prepared.target.lead_h == canonical.target.lead_h == 1. and
          np.array_equal(prepared.target.replay_times_h, canonical.target.replay_times_h),
          'Prepared target state/time/lead differs from original inputs')
    _need(np.array_equal(canonical.target.replay_times_h, np.arange(-4, 96) / 4),
          'Only the complete 100-slot lead1 replay grid is supported')
    _need(prepared.tasks.prepared_sha256 == canonical.tasks.prepared_sha256 and
          prepared.tasks.task_sha256 == canonical.tasks.task_sha256 == prepared.bank['task_sha256'] and
          prepared.tasks.task_ids == canonical.tasks.task_ids and
          prepared.tasks.departure_h == canonical.tasks.departure_h == -1. and
          prepared.feature_tasks.prepared_sha256 == canonical.feature_tasks.prepared_sha256 and
          prepared.feature_tasks.task_ids == canonical.feature_tasks.task_ids,
          'Prepared frozen H/task order/assignment epoch differs')
    _need(tuple(context.grid.point_ids) == tuple(range(441)) and
          tuple(context.lines.line_ids) == tuple(range(1, 33)) and
          tuple(context.roads.arc_ids) == tuple(range(1, 89)) and
          tuple(context.roads.corridor_ids) == tuple(range(1, 45)) and
          tuple(context.load.bus_ids) == tuple(range(1, 34)) and
          tuple(arc.a for arc in context.arcs) == context.roads.arc_ids,
          'Canonical grid/line/road/corridor/bus axis order differs')
    sources = []
    _need(bool(context.input_hashes), 'Physical source identities are required')
    root = Path(context.case_root).resolve()
    for name, digest in sorted(context.input_hashes.items()):
        relative = Path(name)
        _need(not relative.is_absolute() and '..' not in relative.parts,
              'Physical source path escapes the case root')
        path = (root / relative).resolve()
        _need(path.is_relative_to(root), 'Physical source resolves outside case root')
        ref = _file(path)
        _need(ref['sha256'] == digest, 'Physical source bytes changed: ' + name)
        sources.append(ref)
    return canonical, sources


def _shapes(task_count):
    shapes = dict(grid_hazard=(100, 441), line_wind_vectors_m_s=(100, 32, 2),
        road_wind_vectors_m_s=(100, 88, 2), line_outages=(100, 32),
        line_operating_outages=(24, 32), line_standard_normals=(32,),
        line_threshold_uniforms=(32,), road_blocked=(100, 88), road_movable=(100, 88),
        road_travel_time_min=(100, 88), road_corridor_uniforms=(44,),
        load_p_kw=(24, 33), load_q_kvar=(24, 33), load_archived_q_kvar=(24, 33),
        load_epsilon=(24,), load_standard_innovation=(24,), features=(5,))
    shapes.update({name: (task_count,) for name in RAW_FIELDS if name.startswith('task_')})
    shapes['task_mu_full'] = (task_count, 24)
    return shapes


def _feature(context, prepared, arrays, identity):
    return compute_features(arrays['grid_hazard'], arrays['line_outages'], arrays['road_blocked'],
        arrays['task_usable_energy_dc_kwh'], arrays['load_p_kw'], context.load.baseline_p_kw,
        prepared_tasks=prepared.feature_tasks, times_h=prepared.target.replay_times_h,
        task_ids=prepared.tasks.task_ids, road_ids=context.roads.arc_ids, line_ids=context.lines.line_ids,
        grid_ids=context.grid.point_ids, bus_ids=context.load.bus_ids,
        grid_sha256=context.grid.grid_sha256, scenario_identity=identity)


def _baseline(context, prepared, samples, expected_sha, split):
    _need(isinstance(samples, (tuple, list)) and isinstance(expected_sha, (tuple, list)) and
          len(samples) == len(expected_sha) and len(samples) > 0 and len(samples) % 8 == 0,
          'Each split requires all eight bases and independent ordered scenario pins')
    count = len(samples) // 8
    _need(count <= {'planning': 20, 'calibration': 10}[split], 'Development variant limit exceeded')
    _need(len(set(expected_sha)) == len(expected_sha), 'Duplicate baseline scenario identity')
    shapes = _shapes(len(prepared.tasks.task_ids))
    checked = []
    for index, (sample, pin) in enumerate(zip(samples, expected_sha)):
        _need(isinstance(sample, JointScenario), 'Expected a validated JointScenario atom')
        ledger = dict(sample.ledger)
        _need(set(ledger) == LEDGER_KEYS and set(sample.arrays) == set(shapes),
              'Exact original JointScenario ledger/26-array inventory required')
        variant, base = divmod(index, 8)
        base = context.events[base]
        sid = f'{prepared.target.event_id}/development/{split}/{base}/v{variant+1:03d}/R1'
        _need(ledger['scenario_id'] == sid and ledger['target_event'] == prepared.target.event_id and
              ledger['base_weather_event'] == base and type(ledger['variant_index']) is int and
              type(ledger['variant']) is int and ledger['variant_index'] == variant and ledger['variant'] == variant + 1 and
              ledger['split'] == split and ledger['development'] is True and ledger['regime'] == 'R1' and
              ledger['lead_h'] == 1. and ledger['residual_donor_event'] in context.events and
              ledger['residual_donor_event'] != prepared.target.event_id,
              'Baseline target/domain/split/variant-major base order or LOEO differs')
        _need(ledger['assignment_state_sha256'] == prepared.target.state_sha256 and
              ledger['task_sha256'] == prepared.tasks.task_sha256 and
              ledger['grid_sha256'] == context.grid.grid_sha256,
              'Baseline target initial state/frozen H/grid identity differs')
        _need(sample.scenario_sha256 == ledger['scenario_sha256'] == pin ==
              canonical_hash({k: v for k, v in ledger.items() if k != 'scenario_sha256'}),
              'Baseline ordered scenario pin or semantic ledger hash differs')
        _need(set(ledger['array_hashes']) == set(shapes), 'Baseline array hash inventory differs')
        arrays = {}
        for name, shape in shapes.items():
            value = np.asarray(sample.arrays[name])
            _need(value.shape == shape and value.dtype.kind in 'biuf', 'Canonical array axes/dtype differ: ' + name)
            _need(array_hash(value) == ledger['array_hashes'][name], 'Baseline source array hash differs: ' + name)
            arrays[name] = value
        _need(np.array_equal(operating_outages(prepared.target.replay_times_h, arrays['line_outages']),
                             arrays['line_operating_outages']), 'Baseline operating outage aggregation differs')
        baseline_feature = _feature(context, prepared, arrays, {key: ledger[key] for key in IDENTITY_KEYS})
        _need(np.array_equal(baseline_feature.values, arrays['features']) and
              baseline_feature.feature_sha256 == ledger['feature_sha256'],
              'Baseline five-feature values/hash differ from their physical fields and H')
        checked.append(sample)
    return tuple(checked)


def _streams(streams):
    _need(isinstance(streams, dict) and set(streams) == set(SPLITS), 'Exact planning/calibration streams required')
    seen = set()
    for split in SPLITS:
        _need(isinstance(streams[split], dict) and set(streams[split]) == set(BLOCK_FIELDS),
              'Exact three block stream inventory required')
        for supplied in streams[split].values():
            stream = supplied if isinstance(supplied, Stream) else Stream(supplied['root_seed'], tuple(supplied['spawn_key']))
            key = stream.spawn_key
            _need(stream.root_seed == DEVELOPMENT_ROOT and len(key) == 9 and
                  all(type(x) is int and x >= 0 for x in key) and key[:4] == (1, 2, 1 if split == 'planning' else 2, 0) and
                  key[4] == key[5] == 0 and key[6] > 10,
                  'Expected unregistered aggregate development block stream; primitive/bootstrap/OOS addresses are forbidden')
            _need(stream.stream_id not in seen, 'Distinct split/block stream addresses required')
            seen.add(stream.stream_id)


@dataclass(frozen=True)
class AblationBank:
    planning: tuple[JointScenario, ...]
    calibration: tuple[JointScenario, ...]
    planning_features: np.ndarray
    calibration_features: np.ndarray
    bootstrap: object
    moments: object
    component_proofs: object
    provenance: object


class AblationCalibrationError(ValueError):
    """Retain the original LP diagnostics without admitting a failed bank."""
    def __init__(self, moments):
        self.moment_diagnostics = moments.as_dict()
        super().__init__('Recombined moment calibration did not pass: ' + moments.status)


def build_independent_recombination_bank(context, prepared, *, planning_samples, calibration_samples,
        expected_scenario_sha256, sample_indices, streams, domain='development',
        method_id=METHOD, bootstrap_replications=2000, bootstrap_alpha=.05):
    """Construct both complete splits and their newly calibrated moment sets.

    ``expected_scenario_sha256`` pins the original ordered source ledger hashes
    independently of mutable supplied atoms. Every split is variant-major, then
    all eight canonical weather bases. Block streams are explicit aggregate
    development streams (base/variant/target zero, private mechanism > 10).
    """
    _need(domain == 'development' and method_id == METHOD,
          'Only development independent_recombination is supported; formal/OOS/other ablations are unsupported')
    _need(type(bootstrap_replications) is int and bootstrap_replications == 2000 and
          type(bootstrap_alpha) in (int, float) and bootstrap_alpha == .05,
          'This protocol-bound development bank requires bootstrap replications=2000 and alpha=0.05')
    _need(isinstance(expected_scenario_sha256, dict) and set(expected_scenario_sha256) == set(SPLITS) and
          isinstance(sample_indices, dict) and set(sample_indices) == set(SPLITS),
          'Exact planning/calibration source pins and mappings required; OOS is forbidden')
    _streams(streams)
    prepared, physical_sources = _context(context, prepared)
    originals = dict(planning=_baseline(context, prepared, planning_samples, expected_scenario_sha256['planning'], 'planning'),
                     calibration=_baseline(context, prepared, calibration_samples, expected_scenario_sha256['calibration'], 'calibration'))
    outputs, proofs, feature_rows = {}, {}, {}
    for split, source in originals.items():
        blocks = {block: {field: np.stack([sample.arrays[field] for sample in source])
                         for field in sorted(fields)} for block, fields in BLOCK_FIELDS.items()}
        identity = dict(method_id=METHOD, target_id=prepared.target.event_id, split=split,
                        domain=domain, base_support_sha256=support_digest(blocks))
        transformed = independent_recombination(blocks, sample_indices=sample_indices[split],
                                                streams=streams[split], identity=identity)
        proofs[split] = transformed.provenance
        mapping_sha = canonical_hash(transformed.provenance)
        rows = []
        for index in range(len(source)):
            arrays = {field: _readonly(transformed.arrays[block][field][index])
                      for block, fields in BLOCK_FIELDS.items() for field in sorted(fields)}
            donors = {block: dict(source_index=int(transformed.sample_indices[block][index]),
                scenario_id=source[int(transformed.sample_indices[block][index])].ledger['scenario_id'],
                scenario_sha256=source[int(transformed.sample_indices[block][index])].scenario_sha256)
                for block in BLOCK_FIELDS}
            new_identity = dict(scenario_id=f'{prepared.target.event_id}/development/{split}/{METHOD}/{mapping_sha[:16]}/s{index+1:04d}/R1',
                method_id=METHOD, target_event=prepared.target.event_id, split=split, development=True,
                regime='R1', lead_h=1., output_sample_index=index, block_donors=donors)
            feature = _feature(context, prepared, arrays, new_identity)
            arrays['features'] = _readonly(feature.values)
            ledger = dict(new_identity, assignment_state_sha256=prepared.target.state_sha256,
                task_sha256=prepared.tasks.task_sha256, task_ids=list(prepared.tasks.task_ids),
                grid_sha256=context.grid.grid_sha256, times_sha256=array_hash(prepared.target.replay_times_h),
                component_proof_sha256=mapping_sha, feature_sha256=feature.feature_sha256,
                array_hashes={name: array_hash(value) for name, value in arrays.items()},
                source_replay_sha256_by_block={block: source[donor['source_index']].ledger['task_replay_sha256']
                    for block, donor in donors.items() if block == 'road_vehicle'},
                replay_performed=False, formal_OOS=False)
            digest = canonical_hash(ledger)
            ledger['scenario_sha256'] = digest
            rows.append(JointScenario(MappingProxyType(arrays), MappingProxyType(ledger),
                MappingProxyType(dict(feature_diagnostics=dict(feature.diagnostics),
                                     diagnostics_recomputed=True, old_diagnostics_copied=False)), digest))
        outputs[split] = tuple(rows)
        feature_rows[split] = _readonly(np.stack([row.arrays['features'] for row in rows]))
    boot = bootstrap_moments(feature_rows['calibration'], development=True,
                             replications=bootstrap_replications, alpha=bootstrap_alpha)
    upper = (1., 1., 1., 0. if prepared.feature_tasks.empty_H else 1., 1.09)
    moments = calibrate_moments(feature_rows['planning'], boot, physical_upper=upper, rhos=(.5, 1.))
    if moments.status != 'PASS_MOMENT_CALIBRATION':
        raise AblationCalibrationError(moments)
    sources = [_file(module.__file__) for module in
               (joint_scenarios, physical_ablations, scenario_features, moment_calibration,
                random_streams, line_outages, task_replay, case_inputs, forecast_tasks)] + [_file(__file__)]
    provenance = dict(kind='DEVELOPMENT_INDEPENDENT_RECOMBINATION_BANK_V1', method_id=METHOD,
        domain=domain, target_event=prepared.target.event_id, regime='R1', lead_h=1.,
        task_sha256=prepared.tasks.task_sha256, assignment_state_sha256=prepared.target.state_sha256,
        source_scenario_sha256={split: [sample.scenario_sha256 for sample in rows] for split, rows in originals.items()},
        output_scenario_sha256={split: [sample.scenario_sha256 for sample in rows] for split, rows in outputs.items()},
        split_sizes={split: len(rows) for split, rows in outputs.items()}, event_order=list(context.events),
        source_sample_order='variant_major_then_all_eight_canonical_base_events',
        physical_source_references=physical_sources, implementation_source_references=sources,
        implementation_source_reference_scope='Direct bank/transform/feature/moment/RNG/aggregation/frozen-task kernels; full runtime release audit remains outside this development component',
        component_proofs=proofs, bootstrap=boot.as_dict(), moments=moments.as_dict(),
        feature_matrix_sha256={split: array_hash(rows) for split, rows in feature_rows.items()},
        physical_array_count=25, all_five_features_recomputed=True, baseline_moments_reused=False,
        source_target_H_time_asset_binding_verified=True, finite_empirical_dependence_may_remain=True,
        input_proof_boundary='Independent ordered source scenario pins plus ledger/array/context/H/feature consistency; original donor physics is not resimulated by this adapter',
        source_physics_replayed_or_independently_audited=False,
        exact_product_support_materialized=False, road_or_task_replay_performed=False,
        component_complete=True, formal_bank_admitted=False, formal_random_namespace_registered=False,
        formal_OOS=False, scientific_acceptance=False)
    provenance['bank_sha256'] = canonical_hash(provenance)
    return AblationBank(outputs['planning'], outputs['calibration'], feature_rows['planning'],
                        feature_rows['calibration'], boot, moments, MappingProxyType(proofs), MappingProxyType(provenance))
