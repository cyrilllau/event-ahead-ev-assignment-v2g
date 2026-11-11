"""Independent formal input audit; no bank regeneration, solver, or OOS calls.

All 1920 saved atoms receive hash/identity and numerical wind-to-grid/road,
scalar route/task and feature checks. Sixteen predeclared atoms additionally
resample the original weather fields separately in the three physical domains.
Frozen input loaders, task preparation (identity only) and the accepted weather
interpolator are reused; no formal/joint generator or feature evaluator runs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time
import numpy as np

TOL=1e-8
ARRAY_TOL=3e-14
TIMES=np.arange(-4,96,dtype=float)/4
TASK_FIELDS=('arrival_h','gamma_route','route_time_h','travel_energy_kwh','arrival_soc','varsigma','mu_full','usable_energy_dc_kwh','failed_arc_ids')
ARRAY_FIELDS=('grid_hazard','line_wind_vectors_m_s','road_wind_vectors_m_s','line_outages','line_operating_outages','road_blocked','road_movable','road_travel_time_min','load_p_kw','load_q_kvar','load_archived_q_kvar','load_epsilon','load_standard_innovation','line_standard_normals','line_threshold_uniforms','road_corridor_uniforms','features')+tuple('task_'+k for k in TASK_FIELDS)


def check(condition,message):
    if not bool(condition):raise AssertionError(message)


def close(actual,expected,label,tolerance=ARRAY_TOL):
    a,b=np.asarray(actual),np.asarray(expected)
    check(a.shape==b.shape,label+': shape')
    check(np.array_equal(np.isfinite(a),np.isfinite(b)) and not np.isnan(a).any(),label+': nonfinite mask')
    check(np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b)),label+': infinity sign')
    finite=np.isfinite(b);error=float(np.max(abs(a[finite].astype(float)-b[finite].astype(float)),initial=0.))
    check(error<=tolerance,label+': numerical difference '+str(error));return error


def exact(actual,expected,label):
    check(np.asarray(actual).shape==np.asarray(expected).shape and np.array_equal(actual,expected),label)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def ah(value):
    a=np.ascontiguousarray(value);return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def bh(value):
    a=np.ascontiguousarray(value);header=json.dumps({'shape':a.shape,'dtype':str(a.dtype)},sort_keys=True).encode()
    return hashlib.sha256(header+a.tobytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))

def verified(ref):
    path=Path(ref['path']);check(path.is_absolute() and path.is_file(),'Missing absolute evidence '+str(path))
    check(sha(path)==ref['sha256'],'Changed reference '+str(path))
    size=ref.get('size_bytes',ref.get('bytes'))
    if size is not None:check(path.stat().st_size==size,'Reference size '+str(path))
    return path

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def stream(split,base,variant,mechanism,asset=0):
    check(split in ('planning','calibration'),'Audit cannot access OOS')
    root={'planning':20260901,'calibration':20260902}[split]
    key=(1,1,{'planning':1,'calibration':2}[split],0,base+1,variant+1,mechanism,asset,0)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(root,spawn_key=key)))
    return rng,canonical({'root_seed':root,'spawn_key':key})


def wind_parameters(events,target,base,variant,split):
    bi=events.index(base)
    def rng(m):return stream(split,bi,variant,m)[0]
    normal=NormalDist(1.,.08);amplitude=normal.inv_cdf(normal.cdf(.85)+rng(1).random()*(normal.cdf(1.2)-normal.cdf(.85)))
    loc=rng(2)
    for _ in range(100000):
        offset=loc.normal(0,4,size=2)
        if np.linalg.norm(offset)<=10:break
    else:raise AssertionError('Location rejection exhausted')
    donor=next(events[int(i)] for i in rng(4).permutation(len(events)) if events[int(i)]!=target)
    return dict(target_event=target,base_event=base,donor_event=donor,split=split,development=False,variant_index=variant,
        amplitude=float(amplitude),offset_xy_km=offset.tolist(),lag_h=int(rng(3).choice([-2,-1,0,1,2],p=[.1,.2,.4,.2,.1])),
        error_scale=float(rng(6).uniform(.75,1.25)),block_start_indices=rng(5).integers(0,23,size=9).tolist())


def physical_arrays(arrays,ledger,lines,roads,load,events,*,lead_h=1.):
    """Independent source-coefficient equations, without replay implementations."""
    split=ledger['split'];bi=events.index(ledger['base_weather_event']);v=ledger['variant_index']
    rng,sid=stream(split,bi,v,7);z=rng.standard_normal(len(lines.line_ids));exact(arrays['line_standard_normals'],z,'Line formal normal stream')
    uniforms=np.array([.5*math.erfc(-float(value)/math.sqrt(2)) for value in lines.cholesky@z])
    close(arrays['line_threshold_uniforms'],uniforms,'Line correlated CDF');check(ledger['line_stream_id']==sid,'Line stream ID')
    w=arrays['line_wind_vectors_m_s'];angle=np.radians(lines.azimuth_deg_from_north)
    speed=np.sqrt(np.sum(w*w,axis=-1));cross=abs(w[:,:,0]*np.cos(angle)-w[:,:,1]*np.sin(angle))
    def probability(c):return np.exp(-np.logaddexp(0.,-(c[...,0]+c[...,1]*speed+c[...,2]*cross)))
    p=1-(1-probability(lines.pole_coefficients))*(1-probability(lines.wire_coefficients))
    outage=uniforms[None,:]<=np.maximum.accumulate(p,axis=0)
    exact(arrays['line_outages'],outage,'Independent persistent line outages')
    exact(arrays['line_operating_outages'],outage[int(4*lead_h):].reshape(24,4,32).any(axis=1),'Hourly line mapping')
    road_u=[];road_ids=[]
    for cid in roads.corridor_ids:
        generator,identity=stream(split,bi,v,8,int(cid));road_u.append(generator.random());road_ids.append(identity)
    exact(arrays['road_corridor_uniforms'],road_u,'Physical corridor streams');check(ledger['road_stream_ids']==road_ids,'Road stream IDs')
    rw=arrays['road_wind_vectors_m_s'];exact(rw[:,::2],rw[:,1::2],'Reverse corridor local weather')
    rs=np.sqrt(np.sum(rw*rw,axis=-1));hazard=np.clip((rs-roads.w_thr_m_s)/(roads.w_sev_m_s-roads.w_thr_m_s),0,1)
    direct=roads.kappa_blk*np.exp(-np.logaddexp(0.,-(roads.alpha0+roads.alpha1*rs)))
    survival=np.ones_like(direct)
    for i in range(len(lines.line_ids)):survival*=1-outage[:,i,None]*np.minimum(roads.gamma_cap,roads.gamma[i])
    blocked=np.asarray(road_u)[roads.arc_corridor_indices][None,:]<=np.maximum.accumulate(1-(1-direct)*survival,axis=0)
    movable=(~blocked)&(rs<=roads.w_mov_m_s)
    travel=np.where(movable,roads.t0_min*(1+roads.kappa_tt*hazard),np.inf)
    exact(arrays['road_blocked'],blocked,'Independent persistent road blockage');exact(arrays['road_movable'],movable,'Independent wind movement')
    close(arrays['road_travel_time_min'],travel,'Independent road minutes',1e-12)
    generator,identity=stream(split,bi,v,9)
    for count in range(1,100001):
        standard=generator.standard_normal(24);epsilon=load.cholesky@standard
        if np.all((epsilon>=-.09)&(epsilon<=.09)):break
    else:raise AssertionError('Load rejection exhausted')
    exact(arrays['load_standard_innovation'],standard,'Formal full-vector load innovation');close(arrays['load_epsilon'],epsilon,'Load correlation',1e-14)
    d=ledger['load_draw'];check(d['stream_id']==identity and d['development'] is False and d['split']==split and d['base_index']==bi and d['variant_index']==v,'Load stream semantics')
    check(d['proposal_count']==count and d['rejection_count']==count-1,'Whole-vector rejection count')
    close(d['accepted_standard_innovation'],standard,'Load ledger innovations');close(d['epsilon'],epsilon,'Load ledger epsilon')
    check(d['model_sha256']==load.model_sha256,'Load model hash')
    draw=dict(standard_innovation=standard.tolist(),epsilon=epsilon.tolist(),proposal_count=count,stream_id=identity,
        model_sha256=load.model_sha256,split=split,development=False,base_index=bi,variant_index=v)
    check(d['draw_sha256']==canonical(draw),'Load draw hash')
    for name,baseline in [('load_p_kw',load.baseline_p_kw),('load_q_kvar',load.baseline_q_kvar),('load_archived_q_kvar',load.archived_baseline_q_kvar)]:
        close(arrays[name],baseline*(1+epsilon[:,None]),'Independent '+name,1e-12)


def scalar_task_arrays(tasks,states,hardware,arcs,travel,*,tau=1.,basic_soc=.3,times=TIMES,departure_h=-1.):
    """Original-minute half-open entry replay, cached only for identical routes."""
    lookup={int(a['a']):(i,a) for i,a in enumerate(arcs)};cache={};raw=[]
    for task in tasks:
        vehicle=task['vehicle_id'];key=(vehicle,tuple(task['arc_ids']))
        if key not in cache:
            state=states[vehicle];hw=hardware[vehicle];node=state.road_node;visited={node};arrival=float(departure_h);distance=freeflow=0.;failed=-1;reason='COMPLETE'
            for aid in task['arc_ids']:
                i,arc=lookup[aid];check(int(arc['m_road'])==node and int(arc['n_road']) not in visited,'Frozen route geometry')
                node=int(arc['n_road']);visited.add(node)
                if arrival<times[0] or arrival>=24:failed=aid;reason='ENTRY_OUTSIDE_HORIZON';break
                slot=int(np.searchsorted(times,arrival,side='right')-1);minutes=float(travel[slot,i])
                if not math.isfinite(minutes):failed=aid;reason='ENTRY_IMPASSABLE';break
                arrival+=minutes/60.;distance+=float(arc['D_a_km']);freeflow+=float(arc['T0_a_min'])/60.
            if failed!=-1:value=(math.inf,False,0.,0.,0.,reason,failed)
            else:
                check(abs(distance-task['distance_km'])<=1e-10 and abs(freeflow-task['freeflow_h'])<=1e-10,'Route length/freeflow changed')
                elapsed=arrival-departure_h;energy=float(hw['E_km_e_kWh_per_km'])*distance+float(hw['P_aux_e_kW'])*max(elapsed-freeflow,0.)
                soc=max(state.soc-energy/float(hw['E_cap_e_kWh']),0.)
                value=(arrival,True,elapsed,energy,soc,'COMPLETE',-1)
            cache[key]=value
        raw.append(cache[key])
    n=len(tasks)
    fields={name:np.array([row[j] for row in raw],dtype=bool if name=='gamma_route' else np.int64 if name=='failed_arc_ids' else float) for name,j in [('arrival_h',0),('gamma_route',1),('route_time_h',2),('travel_energy_kwh',3),('arrival_soc',4),('failed_arc_ids',6)]}
    capacities=np.array([float(hardware[t['vehicle_id']]['E_cap_e_kWh']) for t in tasks])
    departure_soc=np.array([float(hardware[t['vehicle_id']]['SOC_dep_under_e']) for t in tasks])
    qualification=np.array([max(basic_soc,float(hardware[t['vehicle_id']]['SOC_dep_under_e'])+float(hardware[t['vehicle_id']]['P_blk_under_e_kW'])*tau/(float(hardware[t['vehicle_id']]['eta_dis_e'])*float(hardware[t['vehicle_id']]['E_cap_e_kWh']))) for t in tasks])
    start=np.maximum(fields['arrival_h'],np.array([t['start_period']-1 for t in tasks]))
    success=fields['gamma_route']&(fields['arrival_soc']>=qualification)&(24-start>=tau)
    fields.update(varsigma=success,mu_full=success[:,None]&(start[:,None]<=np.arange(24))&(np.arange(24)+tau<=24),
        usable_energy_dc_kwh=np.where(success,np.maximum(fields['arrival_soc']-departure_soc,0)*capacities,0.))
    statuses=[r[5] if not r[1] else 'TASK_QUALIFIED' if success[i] else 'SOC_UNQUALIFIED' if r[4]<qualification[i] else 'INSUFFICIENT_SERVICE_TIME' for i,r in enumerate(raw)]
    return fields,statuses,len(cache)


def feature_values(arrays,tasks,baseline):
    groups={v:[i for i,t in enumerate(tasks) if t['vehicle_id']==v] for v in sorted({t['vehicle_id'] for t in tasks})}
    denominator=np.array([t['usable_energy_kwh'] for t in tasks]);check(np.all(denominator>0),'Forecast energy denominator')
    energy=arrays['task_usable_energy_dc_kwh']
    delivery=math.fsum(math.fsum(1-min(1,float(energy[i]/denominator[i])) for i in indices)/len(indices) for indices in groups.values())/len(groups) if groups else 0.
    return np.array([math.fsum(arrays['grid_hazard'].ravel())/arrays['grid_hazard'].size,
        np.count_nonzero(arrays['line_outages'])/arrays['line_outages'].size,
        np.count_nonzero(arrays['road_blocked'][:,:80])/(arrays['road_blocked'].shape[0]*80),delivery,
        math.fsum(arrays['load_p_kw'].ravel())/math.fsum(np.asarray(baseline).ravel())])


def network_content_hash(arcs,travel_minutes,identity,*,times=TIMES):
    """Reconstruct physical network identity, including the source context.

    The scenario identity is only the source-context hash. Network identity
    additionally commits to geometry, slots and all realized travel times.
    Positive infinity has the network contract's JSON-null representation.
    """
    travel=np.asarray(travel_minutes,dtype=float)/60.
    check(travel.shape==(len(times),len(arcs)),'Network identity physical axes')
    check(not np.isnan(travel).any() and not np.isneginf(travel).any(),'Network identity invalid time')
    return canonical(dict(arcs=[dict(a=int(a['a']),source=int(a['m_road']),target=int(a['n_road']),
        distance_km=float(a['D_a_km']),freeflow_h=float(a['T0_a_min'])/60.) for a in arcs],
        slot_starts_h=np.asarray(times).tolist(),travel_hours=[[None if math.isinf(v) else float(v) for v in row] for row in travel],
        horizon_end_h=24.,source_context_hash=canonical(identity)))


def replay_semantic_hash(prepared,bank,identity,arrays,statuses,arcs,*,times=TIMES,departure_h=-1.):
    """Bind independently checked saved array bytes to the full physical road network."""
    metadata=dict(prepared_sha256=prepared.prepared_sha256,dependency_sha256=bank['dependency_sha256'],
        split=identity['split'],development=False,scenario_identity=identity,
        array_hashes={k:ah(arrays['task_'+k]) for k in TASK_FIELDS},
        tau_blk_h=float(bank['config']['tau_blk_h']),basic_soc=float(bank['config']['basic_soc']),
        failed_route_accounting='Troute/Etravel/SOC zero is task accounting, not zero partial-trip consumption',
        usable_energy_side='battery DC; discharge efficiency is applied later by the physical power model')
    return canonical(dict(prepared_sha256=prepared.prepared_sha256,
        network_sha256=network_content_hash(arcs,arrays['road_travel_time_min'],identity,times=times),
        departure_h=departure_h,metadata=metadata,status=statuses))


def certificate(record,c,a,b,eq,rhs,label):
    check(record.get('success') is True and record.get('solver_status')==0,label+': solver status')
    x=np.asarray(record['witness'],float);check(x.shape==c.shape and np.isfinite(x).all(),label+': witness')
    slack=b-a@x;equality=eq@x-rhs
    primal=max(0.,float(np.max(-slack)),float(np.max(abs(equality))),float(np.max(-x)))
    check(primal<=TOL,label+': primal violation')
    residuals=record['primal_residuals']
    for key,value in [('inequality_slacks',slack),('equality_residuals',equality),('maximum_violation',primal)]:close(residuals[key],value,label+': '+key)
    close(record['objective'],c@x,label+': objective')
    dual=record['dual_certificate'];yi,ye,zl=[np.asarray(dual[k],float) for k in ('inequality_marginals','equality_marginals','lower_bound_marginals')]
    check(yi.shape==b.shape and ye.shape==rhs.shape and zl.shape==x.shape and all(np.isfinite(v).all() for v in (yi,ye,zl)),label+': dual shape')
    stationarity=c-a.T@yi-eq.T@ye-zl
    error=max(0.,float(np.max(yi)),float(np.max(-zl)),float(np.max(abs(stationarity))))
    pobj=float(c@x);dobj=float(b@yi+rhs@ye);gap=abs(pobj-dobj)
    comp=max(float(np.max(abs(yi*slack))),float(np.max(abs(zl*x))))
    check(max(error,gap,comp)<=TOL,label+': invalid dual/KKT')
    for key,value in [('dual_objective',dobj),('primal_objective',pobj),('absolute_objective_gap',gap),('stationarity_max_abs',np.max(abs(stationarity))),('dual_sign_or_stationarity_violation',error),('complementarity_max_abs',comp)]:
        close(dual[key],value,label+': '+key)
    return dict(primal_violation=primal,dual_violation=error,duality_gap=gap,dual_objective=dobj,complementarity=comp)


def explicit_mean(values):
    return np.array([values[0,j] if np.all(values[:,j]==values[0,j]) else math.fsum(values[:,j])/len(values) for j in range(5)])


def audit_moments(planning,calibration,moment,boot,*,empty_h=False):
    lower=np.array([0.,0.,0.,0.,.91]);upper=np.array([1.,1.,1.,0. if empty_h else 1.,1.09]);scale=upper-lower
    for data in (planning,calibration):check(np.isfinite(data).all() and np.all((data>=lower)&(data<=upper)),'Physical moment range')
    indices=np.asarray(boot['indices']);check(indices.shape==(2000,len(calibration)) and indices.dtype==np.int64,'Bootstrap complete-row shape')
    key=(1,1,2,0,0,0,10,0,0);rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(20260902,spawn_key=key)))
    exact(indices,rng.integers(0,len(calibration),size=indices.shape,dtype=np.int64),'Formal bootstrap address')
    meta=moment['bootstrap'];identity=meta['stream_identity']
    check(identity['root_seed']==20260902 and identity['spawn_key']==list(key) and identity['development'] is False and identity['mechanism_id']==10 and identity['split']=='calibration' and identity['bit_generator']=='PCG64','Bootstrap namespace')
    check(identity['stream_id']==canonical({'root_seed':20260902,'spawn_key':key}),'Bootstrap stream hash')
    check(meta['index_sha256']==bh(indices) and meta['bootstrap_means_sha256']==bh(boot['means']),'Bootstrap payload hashes')
    check(meta['replications']==2000 and meta['alpha']==.05 and meta['quantile_probability']==.99 and meta['quantile_method']=='higher','Bootstrap protocol')
    center=explicit_mean(calibration);means=np.array([explicit_mean(calibration[row]) for row in indices])
    width=np.sort(abs(means-center),axis=0)[math.ceil(.99*(2000-1))]
    for actual,expected,label in [(boot['center'],center,'center'),(boot['means'],means,'means'),(boot['halfwidth'],width,'higher quantile')]:close(actual,expected,'Independent bootstrap '+label)
    center=np.asarray(boot['center']);width=np.asarray(boot['halfwidth'])
    for key,value in [('center',center),('halfwidth',width)]:close(meta[key],value,'Bootstrap metadata '+key)
    close(moment['center'],center,'Unmoved calibration center');close(moment['original_halfwidth'],width,'Original width')
    delta,optimum,margin=[float(moment[k]) for k in ('delta','optimal_delta','numeric_margin_delta')]
    check(all(math.isfinite(v) and v>=0 for v in (delta,optimum,margin)),'Nonnegative repair')
    close(delta,optimum+margin,'Repair decomposition');repaired=width+delta*scale
    close(moment['repaired_halfwidth'],repaired,'Common repaired width')
    repair=moment['repair_certificate'];close(repair['physical_scale'],scale,'Physical repair scale')
    check(repair['minimum_rho']==.5 and repair['center_unchanged'] is True and repair['common_repaired_base_width_for_all_rhos'] is True,'Repair protocol')
    n=len(planning);feas=lambda lo,hi:(np.zeros(n),np.vstack([planning.T,-planning.T]),np.r_[hi,-lo],np.ones((1,n)),np.ones(1))
    rawlo=np.maximum(lower,center-.5*width);rawhi=np.minimum(upper,center+.5*width)
    raw=moment['raw_feasibility'];close(raw['lower'],rawlo,'Raw lower');close(raw['upper'],rawhi,'Raw upper');checks=[]
    if raw['success']:
        check(not repair['required'] and delta==optimum==margin==0.,'Unsolicited repair')
        checks.append(certificate(raw,*feas(rawlo,rawhi),'Raw rho0.5'))
    else:
        check(raw['solver_status']==2 and repair['required'],'Raw unresolved status')
        c=np.r_[np.zeros(n),1.];a=np.vstack([np.c_[planning.T,-.5*scale],np.c_[-planning.T,-.5*scale]])
        b=np.r_[center+.5*width,-center+.5*width];eq=np.array([np.r_[np.ones(n),0.]])
        cert=certificate(repair,c,a,b,eq,np.ones(1),'Minimum delta');checks.append(cert)
        close(repair['witness'][-1],optimum,'Minimal repair witness');close(repair['probability_witness'],repair['witness'][:-1],'Repair p axis')
        check(cert['dual_objective']>0,'Raw infeasibility must follow positive minimum-repair dual bound')
    if margin:
        prob=np.asarray(repair['probability_witness']);positive=scale>0
        required=max(0.,float(np.max((2*abs(planning.T@prob-center)[positive]-width[positive])/scale[positive],initial=0.)))
        close(margin,float(np.nextafter(max(required,optimum),np.inf))-optimum,'Measured numeric margin')
        check(margin<=2*TOL/min(scale[positive]),'Oversized numeric margin');close(repair['numeric_margin_record']['measured_required_delta'],required,'Margin witness')
    check(set(moment['bounds_by_rho'])=={'0.5','1.0','1.5','2.0'},'Complete rho sweep')
    previous=None
    for rho in (.5,1.,1.5,2.):
        row=moment['bounds_by_rho'][str(rho)];lo=np.maximum(lower,center-rho*repaired);hi=np.minimum(upper,center+rho*repaired)
        check(row['rho']==rho,'Rho identity');close(row['lower'],lo,'Final lower');close(row['upper'],hi,'Final upper')
        if previous is not None:check(np.all(lo<=previous[0]) and np.all(hi>=previous[1]),'Nested rho bounds')
        previous=lo,hi;checks.append(certificate(row['feasibility'],*feas(lo,hi),'rho '+str(rho)))
        close(row['feasibility']['implied_feature_means'],planning.T@np.asarray(row['feasibility']['witness']),'Implied feature mean')
    check(moment['status']=='PASS_MOMENT_CALIBRATION','Moment status contradicts evidence')
    return dict(bootstrap_replications=2000,index_sha256=bh(indices),delta=delta,certificate_count=len(checks),
        maximum_primal_violation=max(c['primal_violation'] for c in checks),maximum_dual_violation=max(c['dual_violation'] for c in checks),
        maximum_duality_gap=max(c['duality_gap'] for c in checks),max_bootstrap_mean_difference=float(np.max(abs(means-boot['means']))),solver_calls=0)


def onset_route_oracle(arcs, times, travel, q, deadline):
    outgoing = {}
    for i, a in enumerate(arcs):
        outgoing.setdefault(int(a['m_road']), []).append((i, a))
    # No lower-bound heuristic and no dominance across distinct prefixes.
    origin, destination = q['origin'], q['destination']
    heap = [(q['departure_h'], 0., (), (origin,), 0., ())]
    best, expanded = None, 0
    started = time.monotonic()
    while heap:
        deadline()
        arrival, distance, ids, nodes, ff, entries = heapq.heappop(heap)
        if best is not None and arrival > best[0]:
            break
        if arrival > q['latest_arrival_h']:
            continue
        energy = q['energy_kwh_per_km']*distance + q['auxiliary_kw']*max(0., arrival-q['departure_h']-ff)
        soc = max(0., q['initial_soc']-energy/q['capacity_kwh'])
        # Prefix energy is monotone. Slack here only avoids roundoff pruning.
        if soc < q['qualification_soc']-1e-12:
            continue
        expanded += 1
        if expanded > 2000000 or time.monotonic()-started > 15.:
            raise RuntimeError('Independent oracle cap; cannot certify this query')
        if nodes[-1] == destination:
            if soc >= q['qualification_soc']:
                candidate = (arrival, distance, ids, ff, energy, soc, entries)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            continue
        if arrival < times[0] or arrival >= 24.:
            continue
        slot = int(np.searchsorted(times, arrival, side='right')-1)
        for i, a in outgoing.get(nodes[-1], ()):
            nxt = int(a['n_road'])
            if nxt in nodes or not math.isfinite(travel[slot, i]):
                continue
            heapq.heappush(heap, (arrival+float(travel[slot, i]), distance+float(a['D_a_km']),
                                 ids+(int(a['a']),), nodes+(nxt,), ff+float(a['T0_a_min'])/60., entries+(arrival,)))
    return best, expanded


def audit_onset_task_contract(bank,case_root,case,target,deadline):
    """Reconstruct all onset routes and nearest-three H without the task builder."""
    from evcs_experiments.forecast_tasks import load_forecast_task_inputs
    from evcs_experiments.road_disruptions import load_road_model
    from evcs_experiments.wind_field import forecast_vector_field
    inputs=load_forecast_task_inputs(case_root,target.event_id,lead_h=0.)
    check(not target.forecast_issued_after_assignment and inputs.target.state_sha256==target.state_sha256,'Onset known forecast/target state')
    check(bank['dependency_sha256']==inputs.dependency_sha256,'Onset forecast dependency')
    check(bank['config']==dict(basic_soc=.3,tau_blk_h=1.,nearest_stations=3,latest_useful_arrival_h=23.,route_domain='all node-simple paths',station_order='arrival,distance,station_id'),'Onset frozen H configuration')
    states={state.vehicle_id:state for state in target.vehicle_states};hardware=case.vehicle_parameters
    queries=bank['queries'];pairs={(q['vehicle_id'],q['station_id']) for q in queries}
    check(pairs=={(v,c) for v in case.vehicle_ids for c in target.station_ids} and len(queries)==len(pairs),'Onset full fleet pair coverage')
    arcs=[dict(a=a.a,m_road=a.source,n_road=a.target,D_a_km=a.distance_km,T0_a_min=a.freeflow_h*60.) for a in inputs.network.arcs]
    times=target.replay_times_h;travel=inputs.network.travel_hours
    road=load_road_model(case_root,include_connectors=True)
    speed=np.linalg.norm(forecast_vector_field(target.forecast).sample(times,road.points_xy_km),axis=-1)
    expected=road.t0_min[None,:]/60.*(1+road.kappa_tt[None,:]*np.clip((speed-road.w_thr_m_s)/(road.w_sev_m_s-road.w_thr_m_s),0,1))
    expected=np.where(speed<=road.w_mov_m_s,expected,np.inf)
    close(travel,expected,'Independent onset forecast network',1e-12)
    cache, station_routes, expanded = {}, {}, 0
    for query in bank['queries']:
        assert query['status']=='COMPLETE' and query['certificate']['complete']
        deadline()
        q = query['arguments']
        check(query['query_id']==canonical(dict(dependency_sha256=inputs.dependency_sha256,arguments=q)),'Onset query identity')
        vehicle = query['vehicle_id']; hp = hardware[vehicle]; state = states[vehicle]
        expected_q = {'origin':state.road_node,
                      'destination':int(case.station_parameters[query['station_id']]['road_node_c']),
                      'departure_h':0.,'initial_soc':state.soc,
                      'capacity_kwh':float(hp['E_cap_e_kWh']),
                      'energy_kwh_per_km':float(hp['E_km_e_kWh_per_km']),
                      'auxiliary_kw':float(hp['P_aux_e_kW']),
                      'qualification_soc':max(.3,float(hp['SOC_dep_under_e'])+float(hp['P_blk_under_e_kW'])/(float(hp['eta_dis_e'])*float(hp['E_cap_e_kWh']))),
                      'latest_arrival_h':23.}
        assert q == expected_q
        key = json.dumps(q, sort_keys=True)
        if key not in cache:
            cache[key], n = onset_route_oracle(arcs, times, travel, q, deadline)
            expanded += n
        best = cache[key]
        route = query['route']
        if best is None:
            assert route is None
            continue
        assert route is not None and tuple(route['arc_ids'])==best[2]
        check(route['departure_h']==0.,'Onset route departure')
        np.testing.assert_allclose([route['arrival_h'],route['distance_km'],route['freeflow_h'],
                                  route['travel_energy_kwh'],route['arrival_soc']],
                                 [best[0],best[1],best[3],best[4],best[5]], atol=1e-12,rtol=1e-13)
        np.testing.assert_allclose(route['entry_times_h'], best[6], atol=1e-12,rtol=1e-13)
        station_routes.setdefault(query['vehicle_id'],[]).append((best[0],best[1],query['station_id'],route,q))
    expected_tasks = {}
    tau = int(bank['config']['tau_blk_h'])
    for vehicle, candidates in station_routes.items():
        for _, _, station, route, q in sorted(candidates,key=lambda r:r[:3])[:3]:
            vp=hardware[vehicle]
            qual=max(bank['config']['basic_soc'],float(vp['SOC_dep_under_e'])+float(vp['P_blk_under_e_kW'])*tau/(float(vp['eta_dis_e'])*float(vp['E_cap_e_kWh'])))
            assert qual == q['qualification_soc']
            for k in range(1,25):
                block=list(range(k,k+tau))
                reserve=[t>=k and t-1+tau<=24 for t in range(1,25)]
                if (route['arrival_h']<=k-1 and route['arrival_soc']>=qual and 24-(k-1)>=tau
                    and all(t<=24 and reserve[t-1] for t in block)):
                    expected_tasks[(vehicle,station,k)]=(route,qual,reserve,block)
    assert len(expected_tasks)==len(bank['tasks'])
    actualkeys=set()
    for task in bank['tasks']:
        key=(task['vehicle_id'],task['station_id'],task['start_period'])
        check(task['task_id']==f'{key[0]}__{key[1]}__k{key[2]:02d}','Onset task ID')
        assert key not in actualkeys
        actualkeys.add(key)
        route,qual,reserve,block=expected_tasks[key]
        assert task['arc_ids']==route['arc_ids'] and task['reserve']==reserve and task['block_periods']==block
        assert task['occupancy']==[t>=key[2] for t in range(1,25)]
        assert task['qualification_soc']==qual
        for name in ('arrival_h','distance_km','freeflow_h','travel_energy_kwh','arrival_soc'):
            assert task[name]==route[name]
        usable=max(0,route['arrival_soc']-float(hardware[key[0]]['SOC_dep_under_e']))*float(hardware[key[0]]['E_cap_e_kWh'])
        assert task['usable_energy_kwh']==usable and task['travel_time_h']==route['arrival_h']-route['departure_h']
    assert actualkeys==set(expected_tasks)
    return dict(queries=len(queries),unique_independent_searches=len(cache),oracle_expansions=expanded,tasks=len(bank['tasks']))


def anchor_ids(events):
    """Outcome-independent 16 anchors: all targets/bases, both frozen splits."""
    return {f'{target}/formal/planning/{events[i]}/v001/R1' for i,target in enumerate(events)} | {
        f'{target}/formal/calibration/{events[(i+4)%8]}/v010/R1' for i,target in enumerate(events)}


def audit(bank_root,case_root,source_root,audit_output,*,expected_summary_sha256,expected_source_manifest_sha256,wall_seconds=590,diagnostic_atom_limit=None,bank_kind="baseline_v1"):
    check(__debug__,'Auditor requires enabled Python assertions')
    check(bank_kind in ('baseline_v1','onset_v1'),'Explicit supported bank kind')
    onset=bank_kind=='onset_v1';lead=0. if onset else 1.;departure=0. if onset else -1.;times=np.arange(-int(4*lead),96,dtype=float)/4;nt=len(times)
    started=time.monotonic();bank_root=Path(bank_root).resolve();case_root=Path(case_root).resolve();source_root=Path(source_root).resolve();out=Path(audit_output).resolve()
    check(not out.is_relative_to(bank_root) and out!=bank_root,'Audit writes must remain outside bank')
    check(0<wall_seconds<=7190,'Explicit bounded audit budget')
    check(diagnostic_atom_limit is None or (type(diagnostic_atom_limit) is int and 1<=diagnostic_atom_limit<=16),'Diagnostic limit must be 1..16, never a bank acceptance')
    out.mkdir(parents=True,exist_ok=False)
    completed=[];target_records=[];anchors_done=[];checks=0
    def deadline():
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('Bounded audit incomplete; no independent PASS')
    def verify_imports_and_final_hashes():
        imports={name:str(module.__file__) for name,module in sys.modules.items() if name.startswith('evcs_experiments.') and getattr(module,'__file__',None)}
        check(not any(name in imports for name in ('evcs_experiments.formal_bank','evcs_experiments.formal_scenarios','evcs_experiments.joint_scenarios','evcs_experiments.onset_bank','evcs_experiments.onset_planning')),'Forbidden full-bank generator import')
        for name,path in imports.items():check(str(Path(path).resolve()) in source_hashes and sha(path)==source_hashes[str(Path(path).resolve())],'Auditor kernel source bound '+name)
        check(sha(bank_root/'summary.json')==expected_summary_sha256 and sha(source_root/'RELEASE_MANIFEST.json')==expected_source_manifest_sha256,'Evidence changed during audit')
    try:
        check(sha(bank_root/'summary.json')==expected_summary_sha256,'Externally bound summary hash')
        check(sha(source_root/'RELEASE_MANIFEST.json')==expected_source_manifest_sha256,'Externally bound release hash')
        summary=read(bank_root/'summary.json');request=read(bank_root/'request.json');checkpoint=read(bank_root/'checkpoint.json')
        check(summary['status']==('COMPLETE_FORMAL_ONSET_INPUTS_PENDING_INDEPENDENT_AUDIT' if onset else 'COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT') and summary['atom_count']==1920 and summary['formal_OOS'] is False and summary['formal_plans_solved'] is False,'Complete bank scope')
        check(request['domain']=='formal' and request['regime']=='R1' and request['lead_h']==lead and not isinstance(request['lead_h'],bool) and (not onset or request.get('bank_kind')=='onset_v1') and request['split_sizes']=={'planning':20,'calibration':10},'Request scope')
        check(checkpoint['status']=='COMPLETE' and checkpoint['formal_OOS'] is False and checkpoint['request_sha256']==canonical(request),'Final checkpoint identity')
        check(checkpoint['completed_targets']==summary['targets'],'Checkpoint target/index/moment summary binding')
        check(summary['source_manifest']==request['source_manifest'] and verified(request['source_manifest'])==source_root/'RELEASE_MANIFEST.json','Bank source origin')
        release_manifest=read(source_root/'RELEASE_MANIFEST.json');source_hashes={}
        for row in release_manifest['files']:
            path=(source_root/row['path']).resolve();check(path.is_relative_to(source_root) and sha(path)==row['sha256'],'Release member '+row['path']);source_hashes[str(path)]=row['sha256']
        s01=read(verified(request['s01_acceptance']));s04=read(verified(request['s04_acceptance']))
        check(s01['status']=='PASS' and s04['status']=='PASS' and s04['spec_id']=='S04','Accepted prerequisites')
        protocol_ref=request['formal_protocol_lock'];check(summary['formal_protocol_lock']==protocol_ref and s04['formal_protocol_lock']==protocol_ref,'Accepted protocol reference')
        env=read(verified(protocol_ref));protocol=env['payload'];check(env['kind']=='protocol_lock' and env['payload_sha256']==canonical(protocol) and protocol['domain']=='formal','Formal protocol envelope')
        events=protocol['target_ids'];check(events==sorted(events) and len(events)==8 and len(set(events))==8 and request['target_ids']==events,'Complete eight targets')
        check([r['target_event'] for r in summary['targets']]==events and set(summary['task_sources'])==set(events),'Summary target coverage')
        s01_manifest=read(verified(dict(path=s01['deliverables_manifest'],sha256=s01['deliverables_manifest_sha256'])))
        contract=read(verified(s01_manifest['generator_contract.json']))
        task_refs={r['target_event']:r['frozen_tasks'] for r in contract['frozen_task_sources']}
        if onset:
            task_refs=summary['task_sources']
            check(checkpoint['task_sources']==task_refs,'Onset task sources bound by final checkpoint')
            for ref in task_refs.values():check(verified(ref).is_relative_to(bank_root),'New onset H must originate in this bank')
        else:check(task_refs==summary['task_sources'],'Tasks derive from accepted S01 bytes')
        frozen={}
        for ref in protocol['references']['physical_inputs']:
            old=verified(ref);token='/data/cases/ieee33-v54-q1/'
            check(token in str(old),'Canonical physical case path');relative=str(old).split(token,1)[1]
            check(sha(case_root/relative)==ref['sha256'],'Physical inputs differ from pre-OOS lock: '+relative);frozen[relative]=ref['sha256']
        # Reuse accepted input parsers, not the formal assembly/generator.
        from evcs_experiments.case_inputs import load_case_inputs
        from evcs_experiments.line_outages import load_line_model
        from evcs_experiments.road_disruptions import load_road_model
        from evcs_experiments.load_scenarios import load_load_model
        from evcs_experiments.road_connectors import derive_road_connectors
        from evcs_experiments.wind_scenarios import load_wind_bank,WindParameters
        from evcs_experiments.wind_field import latlon_to_xy
        from evcs_experiments.task_replay import prepare_task_bank
        from evcs_experiments.scenario_features import generator_contract as feature_contract
        case=load_case_inputs(case_root);lines=load_line_model(case_root);roads=load_road_model(case_root,include_connectors=True);load=load_load_model(case_root);wind=load_wind_bank(case_root)
        for model in (case,lines,roads,load,wind):
            for name,digest in model.input_hashes.items():check(frozen.get(name)==digest,'Loaded source binding '+name)
        check(sorted(case.event_ids)==events,'Raw case targets')
        latitude=wind.fields[events[0]].latitude;longitude=wind.fields[events[0]].longitude
        la,lo=np.meshgrid(latitude,longitude,indexing='ij');latlon=np.column_stack([la.ravel(),lo.ravel()]);grid=latlon_to_xy(latlon)
        check(grid.shape==(441,2),'Native 441-point weather domain')
        grid_hash=canonical(dict(latitude=latitude.tolist(),longitude=longitude.tolist(),points_latlon_sha256=ah(latlon),points_xy_sha256=ah(grid),point_order='latitude-major ij meshgrid, longitude varying fastest'))
        with (case_root/'04_transportation/A__directed_road_arcs.csv').open(encoding='utf-8-sig',newline='') as f:arcs=sorted(list(csv.DictReader(f))+list(derive_road_connectors(case_root).arc_rows),key=lambda a:int(a['a']))
        check([int(a['a']) for a in arcs]==list(range(1,89)),'Full physical road axes')
        anchors=anchor_ids(events);write(out/'scope.json',dict(expected_atoms=1920,all_saved_array_fields=list(ARRAY_FIELDS),wind_anchor_ids=sorted(anchors),
            wind_anchor_rule='Per target index i: planning v001/base i; calibration v010/base (i+4)%8. Selected before outcome inspection.',
            expected_summary_sha256=expected_summary_sha256,expected_source_manifest_sha256=expected_source_manifest_sha256,wall_seconds=wall_seconds,diagnostic_atom_limit=diagnostic_atom_limit))
        expected_ids=set();primitive_hashes={};shared_boot=None
        for ti,target_id in enumerate(events):
            deadline();target=case.target(target_id,lead_h=lead);states={s.vehicle_id:s for s in target.vehicle_states}
            bank=read(verified(task_refs[target_id]));tasks=bank['tasks'];task_hash=canonical({k:bank[k] for k in ('dependency_sha256','config','certified','tasks')})
            check(task_hash==bank['task_sha256'] and bank['status']=='COMPLETE' and bank['certified'] is True and not bank.get('tasks_provisional',False),'Frozen H semantic contract')
            onset_h_audit=audit_onset_task_contract(bank,case_root,case,target,deadline) if onset else None
            prepared=prepare_task_bank(bank,case.vehicle_parameters,target);close(target.replay_times_h,times,'Actual assignment time axis')
            check(prepared.departure_h==-lead,'Task departure epoch')
            vehicle_ids=[t['vehicle_id'] for t in tasks];groups={v:[i for i,x in enumerate(vehicle_ids) if x==v] for v in sorted(set(vehicle_ids))}
            weights=[1./(len(groups)*len(groups[v])) for v in vehicle_ids]
            feature_prepared_hash=canonical(dict(task_sha256=task_hash,task_ids=[t['task_id'] for t in tasks],vehicle_ids=vehicle_ids,forecast_usable_dc_kwh=[t['usable_energy_kwh'] for t in tasks],task_weights=weights,contract=feature_contract()))
            record=summary['targets'][ti];check(record['H']==len(tasks) and record['planning_count']==160 and record['calibration_count']==80,'Target count/H')
            features={};max_task_error=max_feature_error=0.;route_count=0
            for split,nvariants in [('planning',20),('calibration',10)]:
                index=read(verified(record[split+'_index']));ids=[t['task_id'] for t in tasks]
                check(index['target_event']==target_id and index['split']==split and index['development'] is False and index['regime']=='R1' and index['lead_h']==lead,'Index identity')
                check(index['task_source']==task_refs[target_id] and index['task_sha256']==task_hash and index['task_ids']==ids and index['assignment_state_sha256']==target.state_sha256,'Index H/target initial state')
                exact(index['times_h'],times,'Index replay slots');rows=index['scenarios'];check(len(rows)==8*nvariants,'Split count')
                values=[]
                for position,row in enumerate(rows):
                    deadline();variant,bi=divmod(position,8);base=events[bi];sid=f'{target_id}/formal/{split}/{base}/v{variant+1:03d}/R1'
                    check(row['scenario_id']==sid and sid not in expected_ids,'Ordered full sample IDs');expected_ids.add(sid)
                    manifest=read(verified(row['manifest']));check(manifest['scenario_id']==sid,'Atom manifest ID')
                    check(checkpoint['completed_atoms'].get(sid)==row['manifest'],'Checkpoint complete atom binding')
                    ledger=read(verified(manifest['files']['ledger']));diagnostics=read(verified(manifest['files']['diagnostics']))
                    with np.load(verified(manifest['files']['arrays']),allow_pickle=False) as z:arrays={name:z[name] for name in z.files}
                    check(set(arrays)==set(ARRAY_FIELDS)==set(ledger['array_hashes']),'All 26 physical arrays required')
                    shapes={'grid_hazard':(nt,441),'line_wind_vectors_m_s':(nt,32,2),'road_wind_vectors_m_s':(nt,88,2),
                        'line_outages':(nt,32),'line_operating_outages':(24,32),'road_blocked':(nt,88),'road_movable':(nt,88),
                        'road_travel_time_min':(nt,88),'load_p_kw':(24,33),'load_q_kvar':(24,33),'load_archived_q_kvar':(24,33),
                        'load_epsilon':(24,),'load_standard_innovation':(24,),'line_standard_normals':(32,),
                        'line_threshold_uniforms':(32,),'road_corridor_uniforms':(len(roads.corridor_ids),),'features':(5,),
                        **{'task_'+k:((len(tasks),24) if k=='mu_full' else (len(tasks),)) for k in TASK_FIELDS}}
                    bool_fields={'line_outages','line_operating_outages','road_blocked','road_movable','task_gamma_route','task_varsigma','task_mu_full'}
                    for name,array in arrays.items():
                        check(array.shape==shapes[name],sid+': explicit physical axes '+name)
                        if name in bool_fields:check(array.dtype==np.dtype(bool),sid+': binary dtype '+name)
                        elif name=='task_failed_arc_ids':check(np.issubdtype(array.dtype,np.integer),sid+': failed arc integer dtype')
                        else:check(np.issubdtype(array.dtype,np.floating),sid+': physical floating dtype '+name)

                    for name,array in arrays.items():
                        check(ah(array)==ledger['array_hashes'][name],sid+': array hash '+name)
                        check(not np.isnan(array).any() and (name in ('task_arrival_h','road_travel_time_min') or np.isfinite(array).all()),sid+': nonfinite '+name)
                    check(canonical({k:v for k,v in ledger.items() if k!='scenario_sha256'})==ledger['scenario_sha256']==row['scenario_sha256']==manifest['scenario_sha256'],'Ledger semantic SHA')
                    params=wind_parameters(events,target_id,base,variant,split)
                    identity=dict(scenario_id=sid,target_event=target_id,base_weather_event=base,variant=variant+1,variant_index=variant,split=split,development=False,regime='R1',lead_h=lead,residual_donor_event=params['donor_event'])
                    for key,value in identity.items():check(ledger[key]==value,'Formal identity '+key)
                    check(params['donor_event']!=target_id and canonical(params)==canonical(ledger['wind_parameters'])==ledger['wind_parameters_sha256'],'Formal wind RNG/LOEO')
                    check(ledger['task_sha256']==task_hash and ledger['assignment_state_sha256']==target.state_sha256 and ledger['grid_sha256']==grid_hash,'Physical H/state/grid identity')
                    check(arrays['grid_hazard'].shape==(nt,441) and np.all((arrays['grid_hazard']>=0)&(arrays['grid_hazard']<=1)),'Full-grid hazard domain')
                    if onset:
                        check(ledger.get('information_contract')=='OR_ONSET_SAME_KNOWN_FORECAST_v1' and ledger.get('forecast_dependency_sha256')==bank['dependency_sha256'],'Onset known forecast information contract')
                    physical_arrays(arrays,ledger,lines,roads,load,events,lead_h=lead)
                    for name in ('line_standard_normals','line_threshold_uniforms','road_corridor_uniforms','load_epsilon','load_standard_innovation','load_p_kw','load_q_kvar'):
                        address=(split,base,variant,name);digest=ah(arrays[name]);check(primitive_hashes.setdefault(address,digest)==digest,'Target-shared primitive pairing')
                    raw,statuses,nroutes=scalar_task_arrays(tasks,states,case.vehicle_parameters,arcs,arrays['road_travel_time_min'],tau=bank['config']['tau_blk_h'],basic_soc=bank['config']['basic_soc'],times=times,departure_h=departure);route_count+=nroutes
                    for name,array in raw.items():max_task_error=max(max_task_error,close(arrays['task_'+name],array,'Independent task '+name,1e-12))
                    # The saved arrays have passed scalar replay. Use their bytes
                    # to reproduce identity serialization without requiring an
                    # irrelevant bitwise equality between two float summations.
                    check(ledger['task_replay_sha256']==replay_semantic_hash(prepared,bank,identity,arrays,statuses,arcs,times=times,departure_h=prepared.departure_h),'Replay semantic hash')
                    fvalue=feature_values(arrays,tasks,load.baseline_p_kw);max_feature_error=max(max_feature_error,close(arrays['features'],fvalue,'Five independent features',1e-12));values.append(arrays['features'])
                    fd=diagnostics['feature_diagnostics'];check(fd['task_sha256']==task_hash and fd['scenario_identity']==identity and fd['task_count']==len(tasks) and fd['road_feature_arc_count']==80 and fd['excluded_connector_count']==8,'Feature domains/identity')
                    components={name:ah(np.asarray(value,dtype=float)) for name,value in (
                        ('grid_hazard',arrays['grid_hazard']),('line_outages',arrays['line_outages']),
                        ('canonical_road_blocked',arrays['road_blocked'][:,:80]),('task_usable_dc',arrays['task_usable_energy_dc_kwh']),
                        ('load_p_kw',arrays['load_p_kw']),('base_p_kw',load.baseline_p_kw),('times_h',times))}
                    check(fd['component_array_hashes']==components and fd['prepared_tasks_sha256']==feature_prepared_hash,'Feature input/prepared-task hashes')
                    check(fd['time_count']==nt and fd['weather_grid_count']==441 and fd['closed_line_count']==32 and fd['road_input_arc_count']==88
                        and fd['candidate_vehicle_count']==len(groups) and fd['empty_H']==(not tasks) and fd['phi_D_structurally_zero']==(not tasks)
                        and fd['phi_load_exceeds_one']==(arrays['features'][4]>1),'Feature diagnostic semantics')
                    check(ledger['feature_sha256']==canonical(dict(values=arrays['features'].tolist(),diagnostics=fd,contract=feature_contract())),'Feature semantic hash')
                    check(diagnostics['sampled_points']==561 and diagnostics['weather_grid_points']==441 and diagnostics['unique_task_routes']==nroutes,'Physical domain/route count diagnostics')
                    if sid in anchors:
                        parameter=WindParameters(**params)
                        for name,points,field in [('grid_hazard',grid,'hazard'),('line_wind_vectors_m_s',lines.points_xy_km,'vectors'),('road_wind_vectors_m_s',roads.points_xy_km,'vectors')]:
                            sampled=wind.sample(parameter,times,points,regime='R1');close(arrays[name],getattr(sampled,field),'Original-field anchor '+name,1e-12)
                        anchors_done.append(sid)
                    completed.append(dict(scenario_id=sid,scenario_sha256=ledger['scenario_sha256'],manifest=row['manifest']));checks+=26
                    if diagnostic_atom_limit is not None and len(completed)>=diagnostic_atom_limit:
                        verify_imports_and_final_hashes();write(out/'scenario_checks.json',completed)
                        result=dict(status='DIAGNOSTIC_ONLY_NOT_BANK_PASS',checked_atom_count=len(completed),array_checks=checks,
                            diagnostic_atom_limit=diagnostic_atom_limit,wind_anchor_ids=sorted(anchors_done),
                            scope='Prefix atoms passed all individual checks; no full-bank coverage or moment acceptance.',
                            max_task_difference=max_task_error,max_feature_difference=max_feature_error,
                            bank_kind=bank_kind,lead_h=lead,source_summary_sha256=expected_summary_sha256,source_manifest_sha256=expected_source_manifest_sha256,
                            audit_source_sha256=sha(__file__),formal_OOS=False,full_S05_pass=False,full_bank_pass=False,solver_calls=0,
                            runtime_seconds=time.monotonic()-started)
                        write(out/'audit.json',result);return result
                features[split]=np.stack(values)
            moment=read(verified(record['moments']))
            with np.load(verified(record['bootstrap']),allow_pickle=False) as z:boot={name:z[name] for name in z.files}
            moments=audit_moments(features['planning'],features['calibration'],moment,boot,empty_h=not tasks)
            if shared_boot is None:shared_boot=moments['index_sha256']
            check(shared_boot==moments['index_sha256'],'Shared bootstrap indices across all targets')
            result=dict(target_event=target_id,H=len(tasks),scenario_count=240,onset_H_audit=onset_h_audit,independent_scalar_routes=route_count,max_task_difference=max_task_error,max_feature_difference=max_feature_error,moments=moments)
            target_records.append(result);write(out/(target_id+'.audit.json'),result);print(json.dumps(result),flush=True)
        check(len(expected_ids)==len(checkpoint['completed_atoms'])==1920 and set(anchors_done)==anchors,'Complete audit and anchor coverage')
        for (split,base,variant,name),digest in primitive_hashes.items():
            if split=='planning' and ('calibration',base,variant,name) in primitive_hashes:check(digest!=primitive_hashes['calibration',base,variant,name],'Independent planning/calibration primitives')
        verify_imports_and_final_hashes()
        write(out/'scenario_checks.json',completed)
        result=dict(status='PASS_INDEPENDENT_FORMAL_BANK_AUDIT',source_output=str(bank_root),bank_kind=bank_kind,lead_h=lead,source_summary_sha256=expected_summary_sha256,source_manifest_sha256=expected_source_manifest_sha256,
            formal_protocol_lock=protocol_ref,s01_acceptance=request['s01_acceptance'],s04_acceptance=request['s04_acceptance'],
            scenario_count=1920,array_checks=checks,targets=target_records,wind_anchor_ids=sorted(anchors_done),
            wind_scope='All atom parameters and stored physical chain audited; original fields resampled only for 16 predeclared anchors.',
            formal_OOS=False,formal_plans_solved=False,solver_calls=0,full_S05_pass=False,audit_source_sha256=sha(__file__),
            runtime_seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(out/'audit.json',result);return result
    except Exception as exc:
        result=dict(status='TIME_LIMIT_INCOMPLETE_AUDIT' if isinstance(exc,TimeoutError) else 'FAIL_INDEPENDENT_FORMAL_BANK_AUDIT',error_type=type(exc).__name__,error=str(exc),
            checked_atom_count=len(completed),completed_targets=target_records,wind_anchors_checked=anchors_done,runtime_seconds=time.monotonic()-started,formal_OOS=False,full_S05_pass=False,
            bank_kind=bank_kind,lead_h=lead,source_summary_sha256=expected_summary_sha256,source_manifest_sha256=expected_source_manifest_sha256,audit_source_sha256=sha(__file__))
        write(out/'audit.json',result);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bank-root','case-root','source-root','audit-output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-summary-sha256',required=True);p.add_argument('--expected-source-manifest-sha256',required=True)
    p.add_argument('--bank-kind',choices=('baseline_v1','onset_v1'),default='baseline_v1')
    p.add_argument('--wall-seconds',type=float,default=590)
    p.add_argument('--diagnostic-atom-limit',type=int,help='Check only a bounded prefix (1..16); exit 2 and never issue bank PASS')
    a=p.parse_args();result=audit(a.bank_root,a.case_root,a.source_root,a.audit_output,expected_summary_sha256=a.expected_summary_sha256,
        expected_source_manifest_sha256=a.expected_source_manifest_sha256,wall_seconds=a.wall_seconds,diagnostic_atom_limit=a.diagnostic_atom_limit,bank_kind=a.bank_kind)
    print(json.dumps(result));return 0 if result['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT' else 2

if __name__=='__main__':raise SystemExit(main())
