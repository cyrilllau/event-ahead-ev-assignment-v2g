"""Development R1 wind/grid/road/task/load assembly on explicit scenario IDs."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np

from .case_inputs import load_case_inputs, InputValidationError
from .wind_scenarios import load_wind_bank, draw_parameters
from .line_outages import load_line_model
from .road_disruptions import load_road_model
from .load_scenarios import load_load_model, draw_load_errors, realize_load
from .forecast_routes import ForecastNetwork
from .task_replay import prepare_task_bank, replay_prepared
from .scenario_features import native_weather_grid, prepare_feature_tasks, compute_features
from .random_streams import canonical_hash, event_order


def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


@dataclass(frozen=True)
class JointTarget:
    target: object
    tasks: object
    feature_tasks: object
    bank: dict


@dataclass(frozen=True)
class JointScenario:
    arrays: object
    ledger: object
    diagnostics: object
    scenario_sha256: str


class JointContext:
    """Immutable physical models plus caches of target-independent primitives.

    The only generated namespace in this increment is development. Creating
    formal banks requires a later explicit orchestration gate.
    """
    def __init__(self, case_root, arcs):
        self.case_root=Path(case_root).resolve()
        self.case=load_case_inputs(self.case_root)
        self.wind=load_wind_bank(self.case_root)
        self.lines=load_line_model(self.case_root)
        self.roads=load_road_model(self.case_root,include_connectors=True)
        self.load=load_load_model(self.case_root)
        self.events=event_order(self.case.event_ids)
        self.grid=native_weather_grid(self.wind)
        self.arcs=tuple(arcs)
        if tuple(a.a for a in self.arcs)!=self.roads.arc_ids:
            raise InputValidationError('Joint physical network must retain canonical arc order')
        self.points=np.concatenate((self.grid.points_xy_km,self.lines.points_xy_km,self.roads.points_xy_km))
        self.points.setflags(write=False)
        self.primitives={}
        hashes={}
        for model in (self.case,self.wind,self.lines,self.roads,self.load):
            for name,digest in model.input_hashes.items():
                if name in hashes and hashes[name]!=digest:
                    raise InputValidationError('Conflicting source identities in joint models')
                hashes[name]=digest
        self.input_hashes=MappingProxyType(hashes)

    def prepare_target(self, event, frozen_bank):
        target=self.case.target(event,lead_h=1.)
        return JointTarget(target,prepare_task_bank(frozen_bank,self.case.vehicle_parameters,target),
                           prepare_feature_tasks(frozen_bank),frozen_bank)

    def generate(self, prepared, base_event, variant_index, split):
        if split not in ('planning','calibration') or type(variant_index) is not int or not 0<=variant_index<{'planning':20,'calibration':10}[split]:
            raise InputValidationError('Development joint bank accepts planning20/calibration10 variants only')
        if base_event not in self.events or prepared.target.event_id not in self.events:
            raise InputValidationError('Unknown joint scenario target or base')
        target=prepared.target;times=target.replay_times_h
        if not np.array_equal(times,np.arange(-4,96)/4):
            raise InputValidationError('I08 joint baseline requires 100 actual lead1 replay slots')
        key=(split,base_event,variant_index)
        if key not in self.primitives:
            lt=self.lines.draw_thresholds(self.events,base_event,variant_index,split=split,development=True)
            rt=self.roads.draw_thresholds(self.events,base_event,variant_index,split=split,development=True)
            ld=draw_load_errors(self.load,self.events.index(base_event),variant_index,split=split,development=True)
            self.primitives[key]=(lt,rt,ld,realize_load(self.load,ld))
        lt,rt,ld,load=self.primitives[key]
        parameters=draw_parameters(self.events,target.event_id,base_event,variant_index,split=split,development=True)
        scenario_id=f'{target.event_id}/development/{split}/{base_event}/v{variant_index+1:03d}/R1'
        identity={'scenario_id':scenario_id,'target_event':target.event_id,'base_weather_event':base_event,
                  'variant':variant_index+1,'variant_index':variant_index,'split':split,'development':True,
                  'regime':'R1','lead_h':1.,'residual_donor_event':parameters.donor_event}
        weather=self.wind.sample(parameters,times,self.points,regime='R1')
        ng=len(self.grid.point_ids);nl=len(self.lines.line_ids)
        line_wind=weather.vectors[:,ng:ng+nl];road_wind=weather.vectors[:,ng+nl:]
        line=self.lines.replay(times,line_wind,lt,lead_h=1.)
        road=self.roads.replay(times,road_wind,rt,line,lead_h=1.,regime='R1')
        if not np.array_equal(line.times_h,times) or not np.array_equal(road.times_h,times):
            raise InputValidationError('Joint upstream time axes differ')
        network=ForecastNetwork(self.arcs,times,road.travel_time_min/60.,network_hash=canonical_hash(identity))
        replay=replay_prepared(network,prepared.tasks,split=split,development=True,scenario_identity=identity)
        feature=compute_features(weather.hazard[:,:ng],line.outages,road.blocked,replay.usable_energy_dc_kwh,
                                 load.p_kw,self.load.baseline_p_kw,prepared_tasks=prepared.feature_tasks,times_h=times,
                                 task_ids=replay.task_ids,road_ids=road.arc_ids,line_ids=line.line_ids,
                                 grid_ids=self.grid.point_ids,bus_ids=self.load.bus_ids,
                                 grid_sha256=self.grid.grid_sha256,scenario_identity=identity)
        arrays={'grid_hazard':weather.hazard[:,:ng], 'line_wind_vectors_m_s':line_wind,
                'road_wind_vectors_m_s':road_wind,'line_outages':line.outages,
                'line_operating_outages':line.operating_outages,'road_blocked':road.blocked,
                'road_movable':road.movable,'road_travel_time_min':road.travel_time_min,
                'load_p_kw':load.p_kw,'load_q_kvar':load.q_kvar,'load_archived_q_kvar':load.archived_q_kvar,
                'load_epsilon':ld.epsilon,'load_standard_innovation':ld.accepted_standard_innovation,
                'line_standard_normals':lt.standard_normals,'line_threshold_uniforms':lt.uniforms,
                'road_corridor_uniforms':rt.corridor_uniforms,'features':feature.values}
        for field in ('arrival_h','gamma_route','route_time_h','travel_energy_kwh','arrival_soc','varsigma','mu_full','usable_energy_dc_kwh','failed_arc_ids'):
            arrays['task_'+field]=getattr(replay,field)
        hashes={name:array_hash(value) for name,value in arrays.items()}
        provenance={**identity,'assignment_state_sha256':target.state_sha256,
                    'task_sha256':prepared.tasks.task_sha256,'grid_sha256':self.grid.grid_sha256,
                    'wind_parameters':parameters.as_dict(),'wind_parameters_sha256':weather.parameters_sha256,
                    'line_stream_id':lt.stream_id,'road_stream_ids':list(rt.stream_ids),
                    'load_draw':ld.as_dict(),'task_replay_sha256':replay.replay_sha256,
                    'feature_sha256':feature.feature_sha256,'array_hashes':hashes}
        digest=canonical_hash(provenance)
        provenance['scenario_sha256']=digest
        diagnostics={'temporal_projection_slots':weather.temporal_projection_count,
                     'spatial_projection_points_all_domains':weather.spatial_projection_count,
                     'sampled_points':len(self.points),'weather_grid_points':ng,
                     'feature_diagnostics':dict(feature.diagnostics),
                     'load_rejection_count':ld.rejection_count,'unique_task_routes':replay.unique_routes_replayed}
        return JointScenario(MappingProxyType(arrays),MappingProxyType(provenance),MappingProxyType(diagnostics),digest)
