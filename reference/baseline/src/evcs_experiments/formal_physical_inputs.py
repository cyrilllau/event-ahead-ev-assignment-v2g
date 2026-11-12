"""Hash-bound formal planning/calibration atoms in the baseline bank v1 layout.

This adapter reads saved atoms. It checks primitive random identities and
replays saved wind/road inputs through the accepted physical kernels; it never
generates another weather trajectory or opens OOS. The development loader is
unchanged. File relocation is explicit and never changes expected bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .config import get_paths
from .evidence import sha256
from .forecast_routes import ForecastNetwork
from .formal_scenarios import PlanningContext
from .joint_scenarios import array_hash
from .load_scenarios import LoadDraw, draw_load_errors, realize_load
from .physical_inputs import PhysicalCase, FrozenTaskInputs, ScenarioPhysicalInputs
from .random_streams import canonical_hash
from .scenario_features import compute_features, generator_contract
from .task_replay import replay_prepared
from .validation.validate_s01_replay import network_arcs
from .wind_scenarios import draw_parameters


INDEX_KEYS = {'target_event', 'split', 'development', 'regime', 'lead_h', 'task_source',
    'task_sha256', 'task_ids', 'assignment_state_sha256', 'times_h', 'scenarios'}
IDENTITY_KEYS = ('scenario_id', 'target_event', 'base_weather_event', 'variant', 'variant_index',
    'split', 'development', 'regime', 'lead_h', 'residual_donor_event')
LEDGER_KEYS = set(IDENTITY_KEYS) | {'assignment_state_sha256', 'task_sha256', 'grid_sha256',
    'wind_parameters', 'wind_parameters_sha256', 'line_stream_id', 'road_stream_ids', 'load_draw',
    'task_replay_sha256', 'feature_sha256', 'array_hashes', 'scenario_sha256'}
TASK_FIELDS = ('arrival_h', 'gamma_route', 'route_time_h', 'travel_energy_kwh', 'arrival_soc',
    'varsigma', 'mu_full', 'usable_energy_dc_kwh', 'failed_arc_ids')
EXPOSED = ('line_operating_outages', 'load_p_kw', 'load_q_kvar', 'load_archived_q_kvar',
    'task_mu_full', 'task_varsigma', 'task_arrival_soc', 'task_arrival_h', 'task_gamma_route',
    'task_usable_energy_dc_kwh')


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


class FormalPhysicalBank:
    """One fixed target's full ordered planning160/calibration80 support."""

    lead_h = 1.
    ledger_keys = LEDGER_KEYS

    def _make_context(self, root):
        return PlanningContext(root, network_arcs(root))

    def _prepare_target(self, context, target, bank):
        return context.prepare_target(target, bank, lead_h=self.lead_h)

    def _verify_extra_ledger(self, ledger):
        pass

    def __init__(self, index_path, case, tasks, *, expected_index_sha256,
                 case_root=None, path_overrides=None):
        _need(isinstance(case, PhysicalCase) and isinstance(tasks, FrozenTaskInputs), 'Expected immutable physical case/tasks')
        _need(tasks.physical_case_sha256 == case.case_sha256, 'Task/case identity differs')
        self.case, self.tasks = case, tasks
        self.path_overrides = {str(k): Path(v).resolve() for k, v in (path_overrides or {}).items()}
        self.index_path = Path(index_path).resolve()
        self.index_reference = dict(path=str(self.index_path), sha256=expected_index_sha256,
            size_bytes=self.index_path.stat().st_size)
        self.index = self._read(self.index_reference)
        index = self.index
        _need(set(index) == INDEX_KEYS, 'Unknown formal bank index schema')
        _need(index['development'] is False and index['split'] in ('planning', 'calibration')
            and index['regime'] == 'R1' and type(index['lead_h']) in (int, float)
            and index['lead_h'] == self.lead_h, 'Only formal planning/calibration with this reader lead/R1 is supported; OOS is locked')
        root = Path(case_root) if case_root is not None else get_paths().data_root
        self.context = self._make_context(root)
        _need(all(case.input_hashes.get(k) == v for k, v in self.context.input_hashes.items()),
            'Physical case and replay-source bytes differ')
        bank = self._read(index['task_source'])
        _need(index['task_source']['sha256'] == tasks.source_sha256, 'Index task source differs from loaded frozen tasks')
        self.prepared = self._prepare_target(self.context, index['target_event'], bank)
        _need(index['task_sha256'] == tasks.task_sha256 == self.prepared.tasks.task_sha256,
            'Index/frozen task semantic identity differs')
        _need(tuple(index['task_ids']) == tasks.task_ids == self.prepared.tasks.task_ids,
            'Index task order differs from frozen H')
        _need(index['assignment_state_sha256'] == self.prepared.target.state_sha256,
            'Index target initial state differs from actual frozen case')
        self.times = np.asarray(index['times_h'], dtype=float)
        _need(np.array_equal(self.times, np.arange(-int(self.lead_h * 4), 96) / 4.), 'Formal replay time order changed')
        rows = index['scenarios']
        count = {'planning': 20, 'calibration': 10}[index['split']]
        expected_ids = [f'{index["target_event"]}/formal/{index["split"]}/{base}/v{v+1:03d}/R1'
            for v in range(count) for base in self.context.events]
        _need(isinstance(rows, list) and len(rows) == len(expected_ids), 'Full formal support count changed')
        _need(all(set(row) == {'scenario_id', 'scenario_sha256', 'manifest'} for row in rows),
            'Unknown formal scenario-index row schema')
        _need([row['scenario_id'] for row in rows] == expected_ids, 'Formal support ID/order changed')
        self.scenario_ids = tuple(expected_ids)
        self.scenario_sha256s = tuple(row['scenario_sha256'] for row in rows)

    def _path(self, reference):
        _need(isinstance(reference, dict) and set(reference) in
            ({'path', 'sha256', 'size_bytes'}, {'path', 'sha256', 'bytes'}), 'Unknown bound-file reference schema')
        _need(isinstance(reference['path'], str) and Path(reference['path']).is_absolute()
            and isinstance(reference['sha256'], str) and re.fullmatch('[0-9a-f]{64}', reference['sha256']),
            'Malformed bound-file identity')
        path = self.path_overrides.get(reference['path'], Path(reference['path']))
        size = reference.get('size_bytes', reference.get('bytes'))
        _need(type(size) is int and size >= 0 and path.stat().st_size == size
            and sha256(path) == reference['sha256'], 'Formal source bytes/hash changed: ' + str(path))
        return path

    def _read(self, reference):
        return json.loads(self._path(reference).read_text())

    def load(self, scenario_index, *, expected_scenario_sha256=None):
        _need(type(scenario_index) is int and 0 <= scenario_index < len(self.scenario_ids), 'Invalid formal scenario index')
        self._path(self.index_reference)
        self._path(self.index['task_source'])
        row = self.index['scenarios'][scenario_index]
        manifest = self._read(row['manifest'])
        _need(set(manifest) == {'scenario_id', 'scenario_sha256', 'files'} and
            set(manifest['files']) == {'arrays', 'ledger', 'diagnostics'}, 'Unknown formal atom manifest schema')
        _need(manifest['scenario_id'] == row['scenario_id'] and manifest['scenario_sha256'] == row['scenario_sha256'],
            'Atom manifest differs from ordered support identity')
        refs = manifest['files']
        paths = {key: self._path(ref) for key, ref in refs.items()}
        ledger = json.loads(paths['ledger'].read_text())
        _need(set(ledger) == self.ledger_keys, 'Unknown formal atom ledger schema')
        self._verify_extra_ledger(ledger)
        digest = canonical_hash({k: v for k, v in ledger.items() if k != 'scenario_sha256'})
        _need(digest == ledger['scenario_sha256'] == row['scenario_sha256'], 'Formal atom semantic hash differs')
        _need(expected_scenario_sha256 is None or digest == expected_scenario_sha256, 'Wrong requested formal scenario identity')
        index = self.index
        base = self.context.events[scenario_index % 8]
        variant = scenario_index // 8
        expected_identity = dict(scenario_id=row['scenario_id'], target_event=index['target_event'],
            base_weather_event=base, variant=variant+1, variant_index=variant, split=index['split'],
            development=False, regime='R1', lead_h=self.lead_h)
        _need(all(ledger[k] == value for k, value in expected_identity.items()) and ledger['development'] is False,
            'Formal atom coordinates/domain differ')
        _need(ledger['task_sha256'] == self.tasks.task_sha256
            and ledger['assignment_state_sha256'] == self.prepared.target.state_sha256
            and ledger['grid_sha256'] == self.context.grid.grid_sha256, 'Formal task/state/grid identity differs')
        h = len(self.tasks.task_ids)
        shapes = dict(grid_hazard=(len(self.times), 441), line_wind_vectors_m_s=(len(self.times), 32, 2),
            road_wind_vectors_m_s=(len(self.times), 88, 2), line_outages=(len(self.times), 32), line_operating_outages=(24, 32),
            road_blocked=(len(self.times), 88), road_movable=(len(self.times), 88), road_travel_time_min=(len(self.times), 88),
            load_p_kw=(24, 33), load_q_kvar=(24, 33), load_archived_q_kvar=(24, 33),
            load_epsilon=(24,), load_standard_innovation=(24,), line_standard_normals=(32,),
            line_threshold_uniforms=(32,), road_corridor_uniforms=(len(self.context.roads.corridor_ids),), features=(5,))
        shapes.update({'task_' + field: (h, 24) if field == 'mu_full' else (h,) for field in TASK_FIELDS})
        booleans = {'line_outages', 'line_operating_outages', 'road_blocked', 'road_movable',
            'task_gamma_route', 'task_varsigma', 'task_mu_full'}
        with np.load(paths['arrays'], allow_pickle=False) as saved:
            _need(set(saved.files) == set(shapes) == set(ledger['array_hashes']), 'Unknown/missing formal physical array schema')
            arrays = {name: saved[name].copy() for name in saved.files}
        for name, value in arrays.items():
            _need(value.shape == shapes[name] and not value.dtype.hasobject, 'Formal physical shape/type differs: ' + name)
            _need(array_hash(value) == ledger['array_hashes'][name], 'Formal physical array semantic hash differs: ' + name)
            if name in booleans:
                _need(value.dtype == bool, 'Physical mask must be Boolean: ' + name)
            elif name == 'task_failed_arc_ids':
                _need(np.issubdtype(value.dtype, np.integer), 'Failed-arc IDs must be integers')
            if name in ('road_travel_time_min', 'task_arrival_h'):
                _need(not np.isnan(value).any() and not np.isneginf(value).any(), 'Invalid physical infinity: ' + name)
            else:
                _need(np.isfinite(value).all(), 'Nonfinite formal physical coefficient: ' + name)
        diagnostics = json.loads(paths['diagnostics'].read_text())
        self._verify_physics(arrays, ledger, expected_identity, base, variant, diagnostics)
        sources = {str(self.index_path): self.index_reference['sha256'],
            str(self._path(index['task_source'])): index['task_source']['sha256'],
            str(self._path(row['manifest'])): row['manifest']['sha256']}
        sources.update({str(paths[key]): ref['sha256'] for key, ref in refs.items()})
        return ScenarioPhysicalInputs(target_event=index['target_event'], split=index['split'],
            base_weather_event=base, variant=variant+1, regime='R1', development=False,
            scenario_id=row['scenario_id'], scenario_sha256=digest, task_sha256=self.tasks.task_sha256,
            config_hash=self.tasks.config_hash, physical_case_sha256=self.case.case_sha256,
            task_ids=self.tasks.task_ids, bus_ids=self.case.bus_ids, line_ids=self.case.line_ids,
            periods=self.case.periods, times_h=self.times, source_hashes=sources,
            metadata=dict(scenario_provenance=ledger, storage_schema='formal_bank_atom_v1',
                verified_array_count=len(arrays), formal_OOS_locked=True,
                config_hash_semantics='canonical_hash(frozen H config)'), **{key: arrays[key] for key in EXPOSED})

    def _verify_physics(self, arrays, ledger, identity, base, variant, diagnostics):
        def same(name, expected, tolerance=0.):
            value = arrays[name]
            _need(np.array_equal(value, expected) if tolerance == 0. else
                np.allclose(value, expected, rtol=0., atol=tolerance), 'Formal physical replay differs: ' + name)
        context = self.context
        split = self.index['split']
        wind = draw_parameters(context.events, self.index['target_event'], base, variant, split=split, development=False)
        _need(canonical_hash(wind.as_dict()) == canonical_hash(ledger['wind_parameters']) == ledger['wind_parameters_sha256']
            and ledger['residual_donor_event'] == wind.donor_event != self.index['target_event'], 'Formal wind stream/LOEO identity differs')
        lt = context.lines.draw_thresholds(context.events, base, variant, split=split, development=False)
        rt = context.roads.draw_thresholds(context.events, base, variant, split=split, development=False)
        ld = draw_load_errors(context.load, context.events.index(base), variant, split=split, development=False)
        saved_draw = ledger['load_draw']
        expected_draw = ld.as_dict()
        _need(set(saved_draw) == set(expected_draw) and
            all(saved_draw[k] == value for k, value in expected_draw.items()
                if k not in ('epsilon', 'draw_sha256')) and saved_draw['development'] is False and
            ledger['line_stream_id'] == lt.stream_id and ledger['road_stream_ids'] == list(rt.stream_ids),
            'Formal primitive stream identity differs')
        # Preserve the producer's exact draw hash/bytes. The accepted kernel
        # validates its epsilon against the original innovation at 1e-14;
        # BLAS differences across hosts do not create a new scenario identity.
        saved_load_draw = LoadDraw(**{**saved_draw,
            'accepted_standard_innovation': np.asarray(saved_draw['accepted_standard_innovation']),
            'epsilon': np.asarray(saved_draw['epsilon'])})
        load = realize_load(context.load, saved_load_draw)
        same('load_epsilon', saved_load_draw.epsilon)
        same('line_standard_normals', lt.standard_normals); same('line_threshold_uniforms', lt.uniforms, 1e-14)
        same('road_corridor_uniforms', rt.corridor_uniforms); same('load_epsilon', ld.epsilon, 1e-14)
        same('load_standard_innovation', ld.accepted_standard_innovation)
        for name, expected in (('load_p_kw', load.p_kw), ('load_q_kvar', load.q_kvar),
                ('load_archived_q_kvar', load.archived_q_kvar)):
            same(name, expected, 2e-12)
        _need(np.allclose(arrays['load_q_kvar'], arrays['load_p_kw'] * self.case.tan_phi_load,
            rtol=0, atol=2e-12), 'Formal Q is not the exact Q1 rated-PF mapping')
        line = context.lines.replay(self.times, arrays['line_wind_vectors_m_s'], lt, lead_h=self.lead_h)
        road = context.roads.replay(self.times, arrays['road_wind_vectors_m_s'], rt, line, lead_h=self.lead_h, regime='R1')
        same('line_outages', line.outages); same('line_operating_outages', line.operating_outages)
        same('road_blocked', road.blocked); same('road_movable', road.movable)
        same('road_travel_time_min', road.travel_time_min, 1e-12)
        identity = {**identity, 'residual_donor_event': ledger['residual_donor_event']}
        network = ForecastNetwork(context.arcs, self.times, arrays['road_travel_time_min']/60., network_hash=canonical_hash(identity))
        replay = replay_prepared(network, self.prepared.tasks, split=split, development=False, scenario_identity=identity)
        for name in TASK_FIELDS:
            same('task_' + name, getattr(replay, name), 0. if name in ('gamma_route', 'varsigma', 'mu_full', 'failed_arc_ids') else 1e-12)
        feature = compute_features(arrays['grid_hazard'], arrays['line_outages'], arrays['road_blocked'],
            arrays['task_usable_energy_dc_kwh'], arrays['load_p_kw'], self.case.baseline_p_kw,
            prepared_tasks=self.prepared.feature_tasks, times_h=self.times, task_ids=self.tasks.task_ids,
            road_ids=context.roads.arc_ids, line_ids=self.case.line_ids, grid_ids=context.grid.point_ids,
            bus_ids=self.case.bus_ids, grid_sha256=context.grid.grid_sha256, scenario_identity=identity)
        same('features', feature.values, 1e-12)
        _need(replay.replay_sha256 == ledger['task_replay_sha256'], 'Formal task replay semantic identity differs')
        _need(isinstance(diagnostics, dict) and isinstance(diagnostics.get('feature_diagnostics'), dict)
            and canonical_hash(diagnostics['feature_diagnostics']) == canonical_hash(dict(feature.diagnostics)),
            'Formal feature diagnostics differ from original physical inputs')
        # A saved contiguous grid and the generator's strided grid slice can
        # reduce in a different floating-point order. Numerical replay above
        # retains the original 1e-12 tolerance; semantic identity retains the
        # producer's exact stored values and independently verified diagnostics.
        saved_feature_hash = canonical_hash(dict(values=arrays['features'].tolist(),
            diagnostics=dict(feature.diagnostics), contract=generator_contract()))
        _need(saved_feature_hash == ledger['feature_sha256'], 'Formal feature semantic identity differs')


def load_formal_scenario_physical_inputs(index_path, scenario_index, case, tasks, *,
        expected_index_sha256, expected_scenario_sha256=None, case_root=None, path_overrides=None):
    return FormalPhysicalBank(index_path, case, tasks, expected_index_sha256=expected_index_sha256,
        case_root=case_root, path_overrides=path_overrides).load(scenario_index,
        expected_scenario_sha256=expected_scenario_sha256)
