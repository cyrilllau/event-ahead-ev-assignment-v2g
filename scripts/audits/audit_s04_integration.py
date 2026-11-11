"""Independent S04-I02 development evidence audit. Never calls an optimizer.

Hand formulas use the declared one-EV/one-bus fixture. LP assembly is reused
only to reconstruct original coefficients for independent primal/dual checks.
This certifies development integration, not formal OOS or whole S04 acceptance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s04_raw_fixture import physical_fixture


def need(value, message):
    if not value:
        raise AssertionError(message)


def close(value, expected, label, tol=1e-7):
    left, right = np.asarray(value, dtype=float), np.asarray(expected, dtype=float)
    need(left.shape == right.shape and np.isfinite(left).all() and
         np.max(np.abs(left-right), initial=0.) <= tol, label)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def references(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'size_bytes'}:
            path = Path(value['path'])
            need(path.is_absolute() and path.is_file(), 'Missing absolute reference: '+str(path))
            need(path.stat().st_size == value['size_bytes'] and digest(path) == value['sha256'], 'Reference changed: '+str(path))
        else:
            for child in value.values():
                references(child)
    elif isinstance(value, list):
        for child in value:
            references(child)


def envelope(path, kind):
    value = read(path)
    need(value['kind'] == kind and value['schema_version'] == 1, 'Lock envelope kind/schema')
    need(value['payload_sha256'] == canonical(value['payload']), 'Lock payload digest')
    references(value['payload'])
    return value['payload']


def kkt(template, plan, arrays, solver):
    need(solver['status'] == 'OPTIMAL' and solver['optimal'] is True, 'Uncertified/limited solver status')
    cert = solver['certificate']
    need(cert['template_sha256'] == template.template_sha256, 'Solver coefficient identity')
    z = np.concatenate([np.asarray(arrays[name]).ravel() for name in template.variable_shapes])
    y, lam, lo, hi = [np.asarray(cert[name], dtype=float) for name in
        ('dual_inequality_marginals','dual_equality_marginals','dual_lower_marginals','dual_upper_marginals')]
    need(all(np.isfinite(v).all() for v in (z,y,lam,lo,hi)), 'Nonfinite primal/dual witness')
    bu = template.b_ub + template.Bx_ub@plan.x + template.Br_ub@plan.r.ravel()
    be = template.b_eq + template.Bx_eq@plan.x + template.Br_eq@plan.r.ravel()
    lb, ub = template.lower_bounds, template.upper_bounds
    max0 = lambda a: float(np.max(a, initial=0.))
    primal = max(max0(template.A_ub@z-bu), max0(abs(template.A_eq@z-be)), max0(lb-z), max0(z-ub))
    stationarity = max0(abs(template.c-template.A_ub.T@y-template.A_eq.T@lam-lo-hi))
    signs = max(max0(y), max0(-lo), max0(hi))
    finite_lo, finite_hi = np.isfinite(lb), np.isfinite(ub)
    infinite = max(max0(abs(lo[~finite_lo])), max0(abs(hi[~finite_hi])))
    objective = float(template.c@z)
    dual = float(bu@y+be@lam+lb[finite_lo]@lo[finite_lo]+ub[finite_hi]@hi[finite_hi])
    gap = abs(objective-dual)
    need(primal <= 1e-7 and max(stationarity,signs,infinite) <= 1e-8 and
         gap <= 1e-4+1e-7*max(abs(objective),abs(dual),1.), 'Independent primal/dual certificate failed')
    close(solver['objective_usd'], objective, 'Reported LP objective')
    for key, expected in [('dual_objective_usd',dual),('dual_bound_usd',dual),('primal_bound_usd',objective),('duality_gap_usd',gap)]:
        close(cert[key],expected,'Reported LP certificate '+key)
    need(cert['optimal_certificate_valid'] is True, 'Claim inconsistent with verified certificate')
    return dict(objective_usd=objective,dual_usd=dual,primal_violation=primal,
                stationarity_violation=stationarity,sign_violation=signs,infinite_marginal_violation=infinite,duality_gap_usd=gap)


def audit(outputroot):
    out = Path(outputroot).resolve()
    manifest = read(out/'artifact_sha256.json')
    current = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p != out/'artifact_sha256.json'}
    need(set(manifest) == current, 'Artifact inventory changed')
    for name, sha in manifest.items():
        need(digest(out/name) == sha, 'Artifact changed: '+name)
    summary = read(out/'summary.json'); references(summary)
    need(summary['status'] == 'PASS_S04_I02_DEVELOPMENT_INTEGRATION' and summary['full_S04_pass'] is False
         and summary['formal_OOS'] is False and summary['item_count'] == 8
         and summary['first_stage_optimization_calls'] == 0, 'Scope/count/first-stage claim')
    prerequisite = read(summary['prerequisite']['path'])
    need(prerequisite['increment_id'] == 'S04-I01' and prerequisite['status'] == 'PASS_INCREMENT', 'Prerequisite acceptance')
    protocol = envelope(out/'locks/protocol.json','protocol_lock')
    batch = envelope(out/'locks/batch-raw-to-table.json','plan_batch_lock')
    access = envelope(out/'locks/access-raw-to-table.json','evaluation_access')
    need(protocol['domain'] == batch['domain'] == access['domain'] == 'development', 'Development domain')
    need(protocol['model_version'] == 'IEEE33-v54-Q1' and protocol['target_ids'] == ['synthetic_fixture']
         and protocol['method_ids'] == ['EA','DSP'], 'Protocol model/target/method')
    raw = read(out/'raw_inputs.json'); samples = raw['samples']; task = raw['bank']['tasks'][0]
    need(raw['scope'] == 'development_hand_fixture' and len(samples) == 4, 'Fixture scope')
    need(raw['arcs'] == [[1,0,1,1.,.25]] and task['arc_ids'] == [1] and raw['arc_to_corridor'] == {'1':'physical_1'}, 'Frozen physical route')
    close(raw['load_p_kw'], [[.5]]*24,'Fixture load'); close(raw['tan_phi'], .5,'Fixture load PF')
    close(raw['states']['EV1']['soc'], .8,'Initial SOC')
    need(protocol['batches'][0]['sample_ids_by_target'] == access['sample_ids_by_target'] == {'synthetic_fixture':list(samples)}, 'Sample lock order/coverage')
    metas = {m:read(out/('fixture-'+m+'.json')) for m in ('EA','DSP')}
    identity = {key:metas['EA'][key] for key in ('x','r','task_sha256','config_hash')}
    for method,meta in metas.items():
        need({key:meta[key] for key in identity} == identity, 'DSP changed first-stage plan')
        close(meta['x'], [1.], 'Fixed selection'); close(meta['r'], [[1.]*24], 'Fixed procurement')
        need(meta['execution_mode'] == method and meta['departure_h'] == (-1. if method == 'EA' else 0.), 'Frozen execution/departure')
    need(metas['DSP']['ea_source_plan_id'] == 'fixture-EA', 'DSP source link')
    plan_hash = canonical(identity)
    need(len(batch['plans']) == 2, 'Complete paired plan bank')
    for record in batch['plans']:
        need(record['plan_sha256'] == plan_hash, 'Plan semantic digest')
        for key in ('tasks','routes','initial_state'):
            need(record['references'][key]['sha256'] == digest(out/'raw_inputs.json'), 'Physical input plan binding')
    reports = []; costs_by_method = {'EA':[], 'DSP':[]}; certificates = []
    folders = sorted((out/'development_end_to_end').glob('item_*'))
    need(len(folders) == 8, 'Exactly eight completed LP items')
    for index, folder in enumerate(folders):
        method = ('EA','DSP')[index//4]; sid = list(samples)[index%4]
        report = read(folder/'metrics.json'); solver = read(folder/'solver.json')
        need(report['method_id'] == method and report['sample_id'] == sid and report['plan_id'] == 'fixture-'+method
             and report['formal_OOS'] is False and report['optimality_claim'] is True, 'Item scope/identity')
        need(report['plan_sha256_before'] == report['plan_sha256_after'] == plan_hash, 'Plan changed in metrics')
        departure = metas[method]['departure_h']; travel = samples[sid]['travel_h']; arrival = departure+travel
        energy_travel = .3+max(0.,travel-.25); soc = .8-energy_travel/80.
        mu = (np.arange(24) >= max(0.,arrival)).astype(float); available = int(mu.sum()); missing = 24-available
        f,replay = physical_fixture(raw,sid,departure)
        close(f.scenario.task_arrival_h,[arrival],'Hand arrival'); close(f.scenario.task_arrival_soc,[soc],'Hand arrival SOC')
        close(f.scenario.task_mu_full,[mu],'Hand availability'); close(f.scenario.task_gamma_route,[1],'Hand route completion')
        close(f.scenario.task_varsigma,[1],'Hand qualification'); close(f.scenario.task_usable_energy_dc_kwh,[(soc-.2)*80],'Hand usable DC energy')
        need(report['scenario_sha256'] == f.scenario.scenario_sha256, 'Saved scenario identity differs from raw replay')
        saved_metadata = read(folder/'physical_metadata.json')
        for key in ('scenario_id','scenario_sha256','task_sha256','config_hash'):
            need(saved_metadata[key] == getattr(f.scenario,key),'Persisted physical identity '+key)
        # Canonical digest normalizes immutable tuple/dict containers to JSON.
        need(canonical(saved_metadata['metadata']) == f.scenario.scenario_sha256,
             'Persisted replay metadata does not yield scenario identity')
        need(saved_metadata['metadata']['departure_h'] == departure,
             'Persisted departure differs from frozen plan')
        with np.load(folder/'physical_inputs.npz',allow_pickle=False) as physical:
            need(set(physical.files) == set(saved_metadata['metadata']['array_hashes']),
                 'Persisted physical array inventory')
            for key in physical.files:
                close(physical[key],getattr(f.scenario,key),'Persisted physical '+key)
            # Direct raw-to-persisted hand checks are separate from the assembler.
            close(physical['task_arrival_h'],[arrival],'Persisted hand arrival')
            close(physical['task_arrival_soc'],[soc],'Persisted hand SOC')
            close(physical['task_mu_full'],[mu],'Persisted hand availability')
            close(physical['load_p_kw'],[[.5]]*24,'Persisted hand active load')
            close(physical['load_q_kvar'],[[.25]]*24,'Persisted hand reactive load')
        with np.load(folder/'solution.npz',allow_pickle=False) as npz:
            arrays = {key:npz[key] for key in npz.files}
        expected = dict(rav=mu[None,:],p=.5*mu[None,:],q=.25*mu[None,:],short=(1-mu)[None,:],
            shed=.5*(1-mu)[:,None],line_p=np.zeros((24,0)),line_q=np.zeros((24,0)),u=np.ones((24,1)),soc_dep=np.array([soc-.5*available/(.95*80)]))
        for key,val in expected.items(): close(arrays[key],val,'Hand optimal primal '+key)
        for key in ('task_sha256','config_hash','scenario_sha256'):
            need(str(arrays[key].item()) == getattr(f.scenario,key),'NPZ identity '+key)
        template = assemble_recourse(f.case,f.tasks,f.scenario)
        need(report['template_sha256'] == template.template_sha256,'Report template binding')
        certificates.append(dict(method=method,sample_id=sid,**kkt(template,f.plan,arrays,solver)))
        costs = dict(Fact_usd=25.,Fcap_usd=.48,Fshed_usd=1.5*missing,Fshort_usd=float(missing),FV2G_usd=.06*available,Fdeg_usd=.025*available/.95)
        costs.update(total_DSO_usd=sum(costs.values()),social_proxy_usd=costs['Fshed_usd']+costs['Fshort_usd']+costs['Fdeg_usd'])
        for key,val in costs.items(): close(report['costs'][key],val,'Hand cost '+key)
        close(solver['objective_usd'],costs['total_DSO_usd']-25.48,'Hand recourse cost')
        series = dict(nameplate=[15.]*24,procured=[1.]*24,certified=mu,dispatched=.5*mu)
        for key,val in series.items():
            close(report['timeseries_kw'][key],val,'Capacity funnel '+key); close(report['energy_kwh'][key],sum(val),'Energy '+key)
        metrics = dict(CLSR=available/24,early_CLSR=max(0,3-missing)/3,CLSR_t=mu,
            first_reserve_h=float(missing),bridge_h=24. if missing == 0 else 0.,early_certified_kwh=float(max(0,3-missing)),
            DRR=available/24,RCR=1.,OTR=float(arrival<=0),MBQR=1.,selected_task_count=1,selected_vehicle_count=1,
            shortfall_kwh=float(missing),v2g_ac_kwh=.5*available,battery_throughput_dc_kwh=.5*available/.95,
            route_completed_selected_count=1,route_failed_selected_count=0,mean_arrival_soc_completed_selected=soc)
        for key,val in metrics.items(): close(report['metrics'][key],val,'Hand metric '+key)
        need(not report['na_reasons'] and not report['metrics']['no_reserve'] and not report['metrics']['no_selected_tasks'] and not report['metrics']['zero_procurement'],'Unexpected NA/empty state')
        costs_by_method[method].append(costs['total_DSO_usd']); reports.append(report)
    rows = list(csv.DictReader((out/'development_end_to_end/metrics_table.csv').open()))
    need(len(rows) == 8,'Raw-to-table coverage')
    for row, report in zip(rows,reports):
        for key in ('method_id','plan_id','sample_id','target_id'):need(row[key] == report[key],'CSV identity '+key)
        need(row['plan_sha256'] == plan_hash,'CSV plan identity')
        for key,val in {**report['costs'],**report['metrics']}.items():
            if isinstance(val,(int,float)) and not isinstance(val,bool):close(float(row[key]),val,'CSV metric '+key)
    stats = read(out/'paired_cost_statistics.json')
    need(stats['statistic'] == 'mean' and stats['reference_method'] == 'EA' and stats['replications'] == 2000,'Statistics scope')
    layout = stats['layout']
    need(layout['shape'] == [1,2,2,2] and layout['target_ids'] == ['synthetic_fixture']
         and layout['base_ids'] == ['b0','b1'] and layout['method_ids'] == ['EA','DSP']
         and layout['variant_ids'] == [['v001','v002'],['v001','v002']], 'Paired statistics layout')
    expected_ids = np.array([[sid,sid] for sid in samples],dtype=object).reshape(1,2,2,2).tolist()
    need(layout['sample_ids'] == expected_ids and stats['layout_sha256'] == canonical(layout), 'Paired sample identities')
    close(stats['target_weights'],[1.],'Target weights')
    key = (1,2,3,0,0,0,11,0,0)
    need(stats['random_stream']['root_seed'] == 20260911 and stats['random_stream']['spawn_key'] == list(key),'Statistics random stream')
    idx = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20260911,spawn_key=key))).integers(0,2,size=(2000,2,2))
    close(stats['bootstrap_indices'],idx,'Shared base/variant draw',0.)
    need(stats['bootstrap_indices_sha256'] == canonical(idx.tolist()),'Bootstrap index hash')
    values = np.array([costs_by_method[m] for m in ('EA','DSP')]).T.reshape(2,2,2)
    draws = np.array([[values[b,idx[r,b],:].sum(axis=0) for b in range(2)] for r in range(2000)]).sum(axis=1)/4
    close(stats['bootstrap_statistics'],draws[:,None,:],'Independent paired bootstrap statistics')
    diffs = draws-draws[:,:1]; close(stats['bootstrap_differences'],diffs[:,None,:],'Independent paired bootstrap differences')
    point = values.mean(axis=(0,1))
    for m,method in enumerate(('EA','DSP')):
        for result in (stats['per_target'][0]['methods'][method],stats['aggregate'][method]):
            close(result['estimate'],point[m],'Paired mean');close(result['ci95'],np.quantile(draws[:,m],[.025,.975]),'Paired CI')
            close(result['difference_from_reference']['estimate'],point[m]-point[0],'Paired difference')
            close(result['difference_from_reference']['ci95'],np.quantile(diffs[:,m],[.025,.975]),'Paired difference CI')
            need(result['status'] == 'DEFINED' and result['na_rate'] == 0.,'Statistics NA preservation')
    mechanisms = read(out/'mechanisms.json')
    for method,mean_cert in [('EA',23.5),('DSP',22.5)]:
        optimism=mechanisms[method]['optimism'];cof=mechanisms[method]['cofailure']
        close(optimism['optimism_ratio'],(24-mean_cert)/24,'Planning optimism')
        close(optimism['r1_mean_certified_kwh'],mean_cert,'Planning optimism empirical energy')
        need(cof['status'] == 'NA' and cof['shared_pair_count'] == 0 and cof['scenario_count'] == 4,'No invented cofailure pair')
        need(cof['joint_failure_probability'] is None and cof['marginal_product'] is None,'No zero-imputed cofailure denominator')
    return dict(status='PASS_INDEPENDENT_S04_I02_DEVELOPMENT_AUDIT',scope='one-EV raw physical replay, two frozen EA/DSP plans, eight LPs and paired statistics',
        formal_OOS=False,full_S04_pass=False,solver_calls=0,artifact_count=len(manifest),audited_item_count=8,
        source_output=str(out),source_manifest_sha256=digest(out/'artifact_sha256.json'),
        hand_costs_usd=costs_by_method,independent_certificates=certificates,
        limitation='Development fixture only. Co-failure NA is tested here; nonzero shared-corridor formulas have separate unit evidence.',
        audit_source_sha256=digest(__file__),audited_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputroot',type=Path,required=True);parser.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args()
    try: result=audit(args.outputroot)
    except Exception as exc:
        result=dict(status='FAIL_INDEPENDENT_S04_I02_DEVELOPMENT_AUDIT',error_type=type(exc).__name__,error=str(exc),solver_calls=0)
        args.audit_output.parent.mkdir(parents=True,exist_ok=True);args.audit_output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');raise
    args.audit_output.parent.mkdir(parents=True,exist_ok=True)
    with args.audit_output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result));return 0

if __name__ == '__main__':raise SystemExit(main())
