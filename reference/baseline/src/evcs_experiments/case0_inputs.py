"""Independent small-fleet IEEE33 Case0; exclusively development random streams.

Keeps the full physical network, 24 hours and per-vehicle task construction.
Case0 addresses begin at variant10000 and never authorize formal OOS access.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
from .config import get_paths
from .evaluation_lock import file_reference
from .evidence import sha256
from .forecast_tasks import load_forecast_task_inputs, build_candidate_tasks
from .forecast_routes import ForecastNetwork
from .joint_scenarios import JointContext
from .task_replay import prepare_task_bank, replay_prepared
from .scenario_features import prepare_feature_tasks, compute_features, FEATURE_NAMES
from .physical_inputs import load_physical_case, load_frozen_tasks
from .evaluation_physical import scenario_from_replay
from .first_stage import assemble_first_stage
from .recourse import assemble_recourse
from .wind_scenarios import draw_parameters
from .load_scenarios import draw_load_errors, realize_load
from .moment_calibration import bootstrap_moments, calibrate_moments
from .probability import ProbabilitySet
from .random_streams import canonical_hash


def _need(value,message):
    if not bool(value):raise ValueError(message)


@dataclass(frozen=True)
class Case0Sample:
    physical: object
    features: object
    ledger: dict
    replay: object


class Case0Context:
    target_id='michael_2018'
    variant_offset=10000

    def __init__(self,output_directory,*,case_root=None,fleet_per_depot=3,planning_variants=2,lead_hours=1,variant_start=10000,profile="legacy"):
        from .case0_profiles import TARGETS, validate_inputs
        validate_inputs(profile,fleet_per_depot,planning_variants,variant_start)
        _need(type(lead_hours) in (int,float) and lead_hours in (0,1),'Case0 assignment lead is zero or one hour')
        self.target_id=TARGETS[profile]
        self.variant_offset=variant_start
        root=Path(case_root or get_paths().data_root).resolve();out=Path(output_directory).resolve()
        _need(not out.is_relative_to(root) and not out.is_relative_to(get_paths().frozen_source_root),'Case0 output must remain outside frozen sources')
        inputs=load_forecast_task_inputs(root,self.target_id,lead_h=float(lead_hours))
        selection_state=inputs.case.target(self.target_id,lead_h=1.)
        depots=sorted({state.depot_id for state in selection_state.vehicle_states})
        _need(len(depots)==4,'Case0 requires original four depots')
        selected=tuple(v for depot in depots for v in sorted(s.vehicle_id for s in selection_state.vehicle_states if s.depot_id==depot)[:fleet_per_depot])
        _need(len(selected)==4*fleet_per_depot,'Insufficient original vehicles in a depot')
        subset_case=replace(inputs.case,vehicle_ids=selected,
            vehicle_parameters=MappingProxyType({v:inputs.case.vehicle_parameters[v] for v in selected}),
            _profiles=MappingProxyType({v:inputs.case._profiles[v] for v in selected}))
        target=subset_case.target(self.target_id,lead_h=float(lead_hours))
        dependency=canonical_hash(dict(kind='CASE0_FORECAST_SUBSET_V1',parent_forecast_dependency=inputs.dependency_sha256,
            vehicle_ids=selected,assignment_state_sha256=target.state_sha256))
        subset=replace(inputs,case=subset_case,target=target,dependency_sha256=dependency)
        path=out/'tasks.json';reopening=out.exists()
        if reopening:
            _need((out/'identity.json').is_file(),'Partial Case0 input directory requires diagnosis')
            saved=json.loads((out/'identity.json').read_text())
            _need(saved['task_file']==file_reference(path),'Saved Case0 task source changed')
            bank=json.loads(path.read_text())
            _need(bank['dependency_sha256']==dependency,'Saved Case0 forecast dependency changed')
        else:
            bank=build_candidate_tasks(subset,overall_time_limit_seconds=60.).as_dict()
            _need(bank['certified'],'Case0 full subset task queries did not finish')
            out.mkdir(parents=True,exist_ok=False)
            path.write_text(json.dumps(bank,indent=2,allow_nan=False)+'\n')
        self.task_reference=file_reference(path);self.bank=json.loads(path.read_text())
        self.case=load_physical_case(root);self.tasks=load_frozen_tasks(path,self.case,self.task_reference['sha256'])
        self.first=assemble_first_stage(self.case,self.tasks)
        self.target=target;self.selected_vehicle_ids=selected;self.lead_hours=float(lead_hours)
        self.output_directory=out;self.context=JointContext(root,inputs.network.arcs);self.events=self.context.events
        self.prepared=prepare_task_bank(self.bank,subset_case.vehicle_parameters,target)
        self.feature_tasks=prepare_feature_tasks(self.bank)
        self.counts={'planning':planning_variants,'calibration':4,'oos':8}
        self.identity=dict(kind='IEEE33_CASE0_DEVELOPMENT_V1',input_adapter_sha256=sha256(__file__),target=self.target_id,lead_h=self.lead_hours,
            vehicle_ids=list(selected),task_sha256=self.tasks.task_sha256,task_file=self.task_reference,
            assignment_state_sha256=target.state_sha256,variant_offset=self.variant_offset,
            variants_per_base=self.counts.copy(),formal_OOS=False,physical_case_sha256=self.case.case_sha256)
        if profile != 'legacy':self.identity['profile']=profile
        if reopening:_need(saved==self.identity,'Case0 configuration/source differs from existing inputs')
        else:(out/'identity.json').write_text(json.dumps(self.identity,indent=2)+'\n')
        self.planning_ids=tuple(self.sample_id(i,'planning') for i in range(8*planning_variants))
        self._samples=OrderedDict();self._probability=None

    def sample_id(self,index,split='planning',regime='R1'):
        _need(split in self.counts and type(index) is int and 0<=index<8*self.counts[split],'Case0 sample outside declared split')
        _need(regime in ('R1','R2','R3') and (split=='oos' or regime=='R1'),'Only OOS has shifted regimes')
        variant,base=divmod(index,8)
        return f'{self.target_id}/case0/development/{split}/{self.events[base]}/v{self.variant_offset+variant+1:05d}/{regime}'

    def sample(self,index,*,split='planning',regime='R1',departure_h=None):
        sid=self.sample_id(index,split,regime);departure=-self.lead_hours if departure_h is None else float(departure_h)
        _need(departure in (-self.lead_hours,0.) and (departure_h is None or type(departure_h) in (int,float)), 'Case0 permits original departure or DSP departure0')
        _need(departure_h is None or split=='oos','DSP departure override only applies to evaluation')
        key=(sid,departure)
        if key in self._samples:return self._samples[key]
        c=self.context;variant,bi=divmod(index,8);variant+=self.variant_offset;base=self.events[bi]
        lt=c.lines.draw_thresholds(self.events,base,variant,split=split,development=True)
        rt=c.roads.draw_thresholds(self.events,base,variant,split=split,development=True)
        ld=draw_load_errors(c.load,bi,variant,split=split,development=True);load=realize_load(c.load,ld)
        parameters=draw_parameters(self.events,self.target_id,base,variant,split=split,development=True)
        times=self.target.replay_times_h;weather=c.wind.sample(parameters,times,c.points,regime=regime)
        ng,nl=len(c.grid.point_ids),len(c.lines.line_ids)
        line=c.lines.replay(times,weather.vectors[:,ng:ng+nl],lt,lead_h=self.lead_hours)
        road=c.roads.replay(times,weather.vectors[:,ng+nl:],rt,line,lead_h=self.lead_hours,regime=regime)
        identity=dict(scenario_id=sid,target_event=self.target_id,base_weather_event=base,variant=variant+1,
            variant_index=variant,split=split,development=True,regime=regime,lead_h=self.lead_hours,
            residual_donor_event=parameters.donor_event,case0_vehicle_ids=list(self.selected_vehicle_ids),
            assignment_state_sha256=self.target.state_sha256)
        network=ForecastNetwork(c.arcs,times,road.travel_time_min/60.,network_hash=canonical_hash(identity))
        replay=replay_prepared(network,self.prepared,departure_h=departure,split=split,development=True,scenario_identity=identity)
        feature=compute_features(weather.hazard[:,:ng],line.outages,road.blocked,replay.usable_energy_dc_kwh,
            load.p_kw,c.load.baseline_p_kw,prepared_tasks=self.feature_tasks,times_h=times,task_ids=replay.task_ids,
            road_ids=road.arc_ids,line_ids=line.line_ids,grid_ids=c.grid.point_ids,bus_ids=c.load.bus_ids,
            grid_sha256=c.grid.grid_sha256,scenario_identity=identity)
        physical=scenario_from_replay(self.case,self.tasks,replay,identity=identity,
            line_operating_outages=line.operating_outages,load_p_kw=load.p_kw,times_h=times,source_hashes=c.input_hashes)
        ledger=dict(identity,wind_parameters=parameters.as_dict(),line_stream_id=lt.stream_id,
            road_stream_ids=list(rt.stream_ids),load_draw=ld.as_dict(),task_sha256=self.tasks.task_sha256,
            departure_h=departure,feature_sha256=feature.feature_sha256,formal_OOS=False)
        answer=Case0Sample(physical,feature.values,ledger,replay)
        self._samples[key]=answer
        while len(self._samples)>2:self._samples.popitem(last=False)
        return answer

    def scenario(self,index,*,split='planning',regime='R1',departure_h=None):
        return self.sample(index,split=split,regime=regime,departure_h=departure_h).physical

    def recourse(self,index):
        return assemble_recourse(self.case,self.tasks,self.scenario(index))

    @property
    def probability_set(self):
        if self._probability is None:
            manifest_path=self.output_directory/'calibration_manifest.json'
            if manifest_path.exists():
                manifest=json.loads(manifest_path.read_text())
                _need(manifest['context_sha256']==canonical_hash(self.identity),'Case0 calibration context differs')
                for ref in (manifest['arrays'],manifest['moments']):_need(file_reference(ref['path'])==ref,'Case0 saved calibration changed')
                saved=json.loads(Path(manifest['moments']['path']).read_text())
                self.moments=saved['moments'];self.bootstrap=saved['bootstrap']
                with np.load(manifest['arrays']['path'],allow_pickle=False) as z:planning=z['planning'].copy()
                bounds=self.moments['bounds_by_rho']['1.0']
                self._probability=ProbabilitySet(self.planning_ids,FEATURE_NAMES,planning,bounds['lower'],bounds['upper'])
                return self._probability
            _need(not (self.output_directory/'feature_calibration.npz').exists() and not (self.output_directory/'moments.json').exists(),
                'Partial Case0 calibration requires diagnosis')
            planning=np.stack([self.sample(i).features for i in range(len(self.planning_ids))])
            calibration=np.stack([self.sample(i,split='calibration').features for i in range(32)])
            bootstrap=bootstrap_moments(calibration,development=True)
            moments=calibrate_moments(planning,bootstrap)
            self.moments=moments.as_dict();self.bootstrap=bootstrap.as_dict()
            np.savez_compressed(self.output_directory/'feature_calibration.npz',planning=planning,calibration=calibration,
                bootstrap_indices=bootstrap.indices,bootstrap_means=bootstrap.bootstrap_means)
            (self.output_directory/'moments.json').write_text(json.dumps(dict(moments=self.moments,bootstrap=self.bootstrap),indent=2)+'\n')
            manifest_path.write_text(json.dumps(dict(context_sha256=canonical_hash(self.identity),
                arrays=file_reference(self.output_directory/'feature_calibration.npz'),moments=file_reference(self.output_directory/'moments.json')),indent=2)+'\n')
            bounds=self.moments['bounds_by_rho']['1.0']
            self._probability=ProbabilitySet(self.planning_ids,FEATURE_NAMES,planning,bounds['lower'],bounds['upper'])
        return self._probability
