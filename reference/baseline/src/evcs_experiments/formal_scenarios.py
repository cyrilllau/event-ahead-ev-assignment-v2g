"""Formal planning/calibration adapter over the accepted IEEE33 physical kernels.

Only baseline lead=1/R1 is implemented here. This context never opens OOS or
changes a frozen task/route bank. Development is an explicit regression domain,
not a source of files that can be relabelled as formal samples.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import numpy as np

from .case_inputs import InputValidationError
from .forecast_routes import ForecastNetwork
from .joint_scenarios import JointContext, JointScenario, JointTarget, array_hash
from .load_scenarios import draw_load_errors, realize_load
from .random_streams import canonical_hash
from .scenario_features import compute_features, prepare_feature_tasks
from .task_replay import prepare_task_bank, replay_prepared
from .wind_scenarios import draw_parameters


class PlanningContext(JointContext):
    """Reuse physical models while separating every primitive by random domain."""

    def prepare_target(self, event, frozen_bank, *, lead_h=1.):
        if isinstance(lead_h, bool) or lead_h != 1.:
            raise InputValidationError('Formal planning adapter currently supports baseline lead=1 only')
        bank = frozen_bank.as_dict() if hasattr(frozen_bank, 'as_dict') else deepcopy(frozen_bank)
        target = self.case.target(event, lead_h=1.)
        tasks = prepare_task_bank(bank, self.case.vehicle_parameters, target)
        if tasks.departure_h != -1.:
            raise InputValidationError('Frozen tasks must use the original lead=1 assignment epoch')
        return JointTarget(target, tasks, prepare_feature_tasks(bank), bank)

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
        target = prepared.target
        original_target = self.case.target(target.event_id, lead_h=1.)
        if target.state_sha256 != original_target.state_sha256 or prepared.tasks.departure_h != -1.:
            raise InputValidationError('Prepared target state/assignment epoch differs from the frozen baseline')
        times = target.replay_times_h
        if not np.array_equal(times, np.arange(-4, 96) / 4):
            raise InputValidationError('Baseline planning requires 100 actual lead1 replay slots')
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
            regime='R1', lead_h=1., residual_donor_event=parameters.donor_event)
        weather = self.wind.sample(parameters, times, self.points, regime='R1')
        ng, nl = len(self.grid.point_ids), len(self.lines.line_ids)
        line_wind = weather.vectors[:, ng:ng + nl]
        road_wind = weather.vectors[:, ng + nl:]
        line = self.lines.replay(times, line_wind, lt, lead_h=1.)
        road = self.roads.replay(times, road_wind, rt, line, lead_h=1., regime='R1')
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
        provenance = dict(identity, assignment_state_sha256=target.state_sha256,
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
