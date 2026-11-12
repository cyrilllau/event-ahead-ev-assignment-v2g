"""Onset reassignment physical factory, without optimizer/OOS admission.

Uses target telematics at t=0 and only the future part of the same known
forecast. The contracted fleet is unchanged; H and fixed routes are rebuilt
before scenario replay. Full planning/calibration banks and their new moment
repair remain required before this component can feed a formal OR solve.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import numpy as np

from .case_inputs import InputValidationError
from .forecast_routes import ForecastNetwork
from .forecast_tasks import load_forecast_task_inputs, build_candidate_tasks, service_periods
from .joint_scenarios import JointContext, JointScenario, JointTarget, array_hash
from .load_scenarios import draw_load_errors, realize_load
from .random_streams import canonical_hash
from .scenario_features import compute_features, prepare_feature_tasks
from .task_replay import prepare_task_bank, replay_prepared
from .wind_scenarios import draw_parameters

class OnsetPlanningContext(JointContext):
    """Lead0/R1 planning and calibration, sharing the accepted primitive keys."""

    def prepare_target(self, event, frozen_bank=None, *, basic_soc=.3, tau_blk_h=1,
                       max_expansions=1000000, time_limit_seconds=30.,
                       overall_time_limit_seconds=540.):
        inputs = load_forecast_task_inputs(self.case_root, event, lead_h=0.)
        if inputs.network.arcs != self.arcs:
            raise InputValidationError('Onset forecast and replay road identities differ')
        if any(self.input_hashes.get(name) != digest for name, digest in inputs.input_hashes.items()):
            raise InputValidationError('Onset forecast inputs changed after context creation')
        if inputs.target.forecast_issued_after_assignment:
            raise InputValidationError('Onset forecast was issued after the assignment epoch')
        if tuple(s.vehicle_id for s in inputs.target.vehicle_states) != self.case.vehicle_ids:
            raise InputValidationError('Onset must retain the entire contracted fleet')
        if frozen_bank is None:
            frozen_bank = build_candidate_tasks(inputs, basic_soc=basic_soc, tau_blk_h=tau_blk_h,
                max_expansions=max_expansions, time_limit_seconds=time_limit_seconds,
                overall_time_limit_seconds=overall_time_limit_seconds)
        bank = frozen_bank.as_dict() if hasattr(frozen_bank, 'as_dict') else deepcopy(frozen_bank)
        if bank.get('dependency_sha256') != inputs.dependency_sha256:
            raise InputValidationError('Onset H must bind its t=0 forecast and target state')
        if bank.get('config', {}).get('basic_soc') != basic_soc or bank.get('config', {}).get('tau_blk_h') != tau_blk_h:
            raise InputValidationError('Onset H settings differ from requested settings')
        expected_pairs = {(v, s) for v in self.case.vehicle_ids for s in self.case.station_ids}
        queries = bank.get('queries', [])
        actual_pairs = {(q['vehicle_id'], q['station_id']) for q in queries}
        if actual_pairs != expected_pairs or len(queries) != len(expected_pairs):
            raise InputValidationError('Onset H requires every contracted vehicle/station query')
        latest = float(max(k - 1 for k in range(1, 25) if service_periods(k, tau_blk_h)[3]))
        if bank['config'].get('nearest_stations') != 3 or bank['config'].get('latest_useful_arrival_h') != latest:
            raise InputValidationError('Onset H station selection/service deadline changed')
        for query in queries:
            args = query['arguments']
            expected_destination = int(self.case.station_parameters[query['station_id']]['road_node_c'])
            if (args.get('destination') != expected_destination or args.get('departure_h') != 0.
                    or args.get('latest_arrival_h') != latest
                    or query.get('query_id') != canonical_hash(dict(
                        dependency_sha256=inputs.dependency_sha256, arguments=args))):
                raise InputValidationError('Onset query destination, epoch, deadline or identity changed')
        tasks = prepare_task_bank(bank, self.case.vehicle_parameters, inputs.target)
        if tasks.departure_h != 0.:
            raise InputValidationError('Onset routes must depart at t=0')
        prepared = JointTarget(inputs.target, tasks, prepare_feature_tasks(bank), bank)
        if not hasattr(self, '_onset_prepared'):
            self._onset_prepared = {}
        self._onset_prepared[id(prepared)] = (prepared, canonical_hash(bank))
        return prepared

    def _verify_prepared(self, prepared):
        entry = getattr(self, '_onset_prepared', {}).get(id(prepared))
        if entry is None or entry[0] is not prepared or canonical_hash(prepared.bank) != entry[1]:
            raise InputValidationError('Onset target must be prepared here with an unchanged frozen H')

    def generate(self, prepared, base_event, variant_index, split, *, domain='formal', regime='R1'):
        if domain not in ('development', 'formal'):
            raise InputValidationError('Planning domain must be explicitly development or formal')
        if split not in ('planning', 'calibration'):
            raise InputValidationError('Planning context accepts planning/calibration only; OOS requires a separate locked factory')
        if type(variant_index) is not int or not 0 <= variant_index < {'planning': 20, 'calibration': 10}[split]:
            raise InputValidationError('Variant exceeds the frozen planning20/calibration10 bank')
        if regime != 'R1':
            raise InputValidationError('Planning adapter currently implements baseline R1 only')
        if not isinstance(prepared, JointTarget):
            raise InputValidationError('Expected a prepared frozen target/task bank')
        if base_event not in self.events or prepared.target.event_id not in self.events:
            raise InputValidationError('Unknown scenario target or base')
        self._verify_prepared(prepared)
        target = prepared.target
        original_target = self.case.target(target.event_id, lead_h=0.)
        if target.state_sha256 != original_target.state_sha256 or prepared.tasks.departure_h != 0.:
            raise InputValidationError('Prepared target state/assignment epoch differs from the frozen baseline')
        times = target.replay_times_h
        if not np.array_equal(times, np.arange(0, 96) / 4):
            raise InputValidationError('Onset planning requires 96 actual lead0 replay slots')
        development = domain == 'development'
        key = (domain, split, base_event, variant_index)
        if key not in self.primitives:
            lt = self.lines.draw_thresholds(self.events, base_event, variant_index,
                split=split, development=development)
            rt = self.roads.draw_thresholds(self.events, base_event, variant_index,
                split=split, development=development)
            ld = draw_load_errors(self.load, self.events.index(base_event), variant_index,
                split=split, development=development)
            self.primitives[key] = (lt, rt, ld, realize_load(self.load, ld))
        lt, rt, ld, load = self.primitives[key]
        parameters = draw_parameters(self.events, target.event_id, base_event, variant_index,
            split=split, development=development)
        scenario_id = f'{target.event_id}/{domain}/{split}/{base_event}/v{variant_index+1:03d}/R1'
        identity = dict(scenario_id=scenario_id, target_event=target.event_id, base_weather_event=base_event,
            variant=variant_index + 1, variant_index=variant_index, split=split, development=development,
            regime='R1', lead_h=0., residual_donor_event=parameters.donor_event)
        weather = self.wind.sample(parameters, times, self.points, regime='R1')
        ng, nl = len(self.grid.point_ids), len(self.lines.line_ids)
        line_wind = weather.vectors[:, ng:ng + nl]
        road_wind = weather.vectors[:, ng + nl:]
        line = self.lines.replay(times, line_wind, lt, lead_h=0.)
        road = self.roads.replay(times, road_wind, rt, line, lead_h=0., regime='R1')
        if not np.array_equal(line.times_h, times) or not np.array_equal(road.times_h, times):
            raise InputValidationError('Joint upstream time axes differ')
        network = ForecastNetwork(self.arcs, times, road.travel_time_min / 60., network_hash=canonical_hash(identity))
        replay = replay_prepared(network, prepared.tasks, split=split, development=development,
            scenario_identity=identity)
        feature = compute_features(weather.hazard[:, :ng], line.outages, road.blocked,
            replay.usable_energy_dc_kwh, load.p_kw, self.load.baseline_p_kw,
            prepared_tasks=prepared.feature_tasks, times_h=times, task_ids=replay.task_ids,
            road_ids=road.arc_ids, line_ids=line.line_ids, grid_ids=self.grid.point_ids,
            bus_ids=self.load.bus_ids, grid_sha256=self.grid.grid_sha256, scenario_identity=identity)
        arrays = dict(grid_hazard=weather.hazard[:, :ng], line_wind_vectors_m_s=line_wind,
            road_wind_vectors_m_s=road_wind, line_outages=line.outages,
            line_operating_outages=line.operating_outages, road_blocked=road.blocked,
            road_movable=road.movable, road_travel_time_min=road.travel_time_min,
            load_p_kw=load.p_kw, load_q_kvar=load.q_kvar, load_archived_q_kvar=load.archived_q_kvar,
            load_epsilon=ld.epsilon, load_standard_innovation=ld.accepted_standard_innovation,
            line_standard_normals=lt.standard_normals, line_threshold_uniforms=lt.uniforms,
            road_corridor_uniforms=rt.corridor_uniforms, features=feature.values)
        for field in ('arrival_h', 'gamma_route', 'route_time_h', 'travel_energy_kwh', 'arrival_soc',
                      'varsigma', 'mu_full', 'usable_energy_dc_kwh', 'failed_arc_ids'):
            arrays['task_' + field] = getattr(replay, field)
        hashes = {name: array_hash(value) for name, value in arrays.items()}
        provenance = dict(identity, information_contract='OR_ONSET_SAME_KNOWN_FORECAST_v1',
            forecast_dependency_sha256=prepared.tasks.dependency_sha256, assignment_state_sha256=target.state_sha256,
            task_sha256=prepared.tasks.task_sha256, grid_sha256=self.grid.grid_sha256,
            wind_parameters=parameters.as_dict(), wind_parameters_sha256=weather.parameters_sha256,
            line_stream_id=lt.stream_id, road_stream_ids=list(rt.stream_ids), load_draw=ld.as_dict(),
            task_replay_sha256=replay.replay_sha256, feature_sha256=feature.feature_sha256, array_hashes=hashes)
        digest = canonical_hash(provenance)
        provenance['scenario_sha256'] = digest
        diagnostics = dict(temporal_projection_slots=weather.temporal_projection_count,
            spatial_projection_points_all_domains=weather.spatial_projection_count,
            sampled_points=len(self.points), weather_grid_points=ng,
            feature_diagnostics=dict(feature.diagnostics), load_rejection_count=ld.rejection_count,
            unique_task_routes=replay.unique_routes_replayed)
        return JointScenario(MappingProxyType(arrays), MappingProxyType(provenance),
            MappingProxyType(diagnostics), digest)
