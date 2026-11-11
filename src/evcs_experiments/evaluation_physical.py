"""Adapt a freshly replayed physical scenario without changing its data domain."""
import numpy as np
from .physical_inputs import ScenarioPhysicalInputs,_array_hash
from .random_streams import canonical_hash


def scenario_from_replay(case,tasks,replay,*,identity,line_operating_outages,load_p_kw,
                         times_h,source_hashes):
    """Build the common Q1 adapter; formal calls require active locked access.

    This is an adapter, not a scenario generator. The caller supplies the actual
    physical outage/load realization from the frozen generator, never planning
    ablation coefficients. Original Q1 Q is recomputed from rated power factor.
    """
    required={'target_event','base_weather_event','variant','regime','split','development','scenario_id'}
    if not required<=set(identity):raise ValueError('Incomplete physical scenario identity')
    if type(identity['development']) is not bool or identity['split'] not in ('planning','calibration','oos'):raise ValueError('Invalid physical namespace')
    if not identity['development'] and identity['split']=='oos':
        from .evaluation_access import require_oos_access
        require_oos_access(target_event=identity['target_event'],base_event=identity['base_weather_event'],variant_index=identity['variant']-1,sample_id=identity['scenario_id'],regime=identity['regime'])
    if tuple(replay.task_ids)!=tuple(tasks.task_ids) or replay.task_sha256!=tasks.task_sha256:raise ValueError('Physical replay changed frozen task axis')
    if bool(replay.metadata['development'])!=identity['development'] or replay.metadata['split']!=identity['split']:raise ValueError('Replay domain/split differs')
    replay_identity=dict(replay.metadata['scenario_identity'])
    if replay_identity.get('scenario_id')!=identity['scenario_id']:raise ValueError('Replay sample identity differs')
    h,t,n,l=len(tasks.task_ids),len(case.periods),len(case.bus_ids),len(case.line_ids)
    p=np.asarray(load_p_kw,dtype=float);outage=np.asarray(line_operating_outages)
    if p.shape!=(t,n) or not np.isfinite(p).all() or np.any(p<0) or outage.shape!=(t,l) or not np.all((outage==0)|(outage==1)):raise ValueError('Invalid physical load/outage axes')
    if tuple(case.periods)!=tuple(range(1,25)) or case.dt_h!=1.:raise ValueError('Real physical replay uses 24 hourly service slots')
    a=np.asarray(replay.arrival_h);g=np.asarray(replay.gamma_route);soc=np.asarray(replay.arrival_soc)
    if a.shape!=(h,) or soc.shape!=(h,) or g.shape!=(h,) or not np.array_equal(np.isfinite(a),g) or np.isnan(a).any() or np.isneginf(a).any() or not np.isfinite(soc).all() or np.any((soc<0)|(soc>1)):raise ValueError('Invalid arrival/Gamma/SOC')
    start=np.maximum(a,np.asarray(tasks.start_periods)-1)
    qualified=g & (soc>=tasks.qualification_soc) & (24-start>=tasks.tau_blk_h)
    mu=qualified[:,None] & (start[:,None]<=np.arange(24)) & (np.arange(24)+tasks.tau_blk_h<=24)
    energy=qualified*np.maximum(soc-tasks.soc_departure_min,0)*tasks.e_cap_kwh
    if not np.array_equal(qualified,replay.varsigma) or not np.array_equal(mu,replay.mu_full) or not np.allclose(energy,replay.usable_energy_dc_kwh,atol=1e-12,rtol=0):raise ValueError('Replay physical deliverability differs from Q1 definition')
    if not source_hashes:raise ValueError('Physical source identities are required')
    arrays=dict(line_operating_outages=outage.astype(bool),load_p_kw=p,load_q_kvar=p*case.tan_phi_load[None,:],
                task_mu_full=replay.mu_full,task_varsigma=replay.varsigma,task_arrival_soc=soc,task_arrival_h=a,
                task_gamma_route=g,task_usable_energy_dc_kwh=replay.usable_energy_dc_kwh)
    metadata=dict(identity=identity,replay=replay.as_dict(),departure_h=replay.departure_h,source_hashes=dict(source_hashes),array_hashes={k:_array_hash(v) for k,v in arrays.items()},load_Q_mapping='P times exact rated power factor')
    return ScenarioPhysicalInputs(**{k:identity[k] for k in required},scenario_sha256=canonical_hash(metadata),task_sha256=tasks.task_sha256,config_hash=tasks.config_hash,physical_case_sha256=case.case_sha256,task_ids=tasks.task_ids,bus_ids=case.bus_ids,line_ids=case.line_ids,periods=case.periods,times_h=np.asarray(times_h),source_hashes=source_hashes,metadata=metadata,load_archived_q_kvar=arrays['load_q_kvar'],**arrays)
