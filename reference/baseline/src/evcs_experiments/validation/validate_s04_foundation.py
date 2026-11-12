"""S04-I01 fixed-plan metrics and paired statistics on explicit development toys."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256,source_provenance
from evcs_experiments.evaluation_metrics import METRIC_CONTRACT,evaluate_development_plan,response_times
from evcs_experiments.evaluation_statistics import PairedSamples,empirical_cvar95,worst_decile_clsr,paired_bootstrap_mean
from evcs_experiments.validation.s02_toys import golden_single_bus,make_fixture,_json_file,_save_solution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-root',type=Path,default=get_paths().data_root)
    p.add_argument('--s03-acceptance',type=Path,required=True)
    p.add_argument('--s03-acceptance-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();began=time.monotonic()
    if sha256(args.s03_acceptance)!=args.s03_acceptance_sha256:raise ValueError('S03 acceptance hash mismatch')
    card=json.loads(args.s03_acceptance.read_text())
    if card.get('status')!='PASS' or card.get('spec_id')!='S03':raise ValueError('S03 must be accepted before S04 validation')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checks=[]
    def need(value,label):
        checks.append(dict(check=label,passed=bool(value)))
        if not value:raise ValueError(label)
    physical=[]
    fixtures=(('single_bus_handgolden',golden_single_bus()),
              ('late_eligible_reserve',make_fixture(periods=3,start_periods=[2],reserve=2,load_p=[[0],[1],[1]],tan_phi=0)),
              ('zero_plan',make_fixture(selected=0,tan_phi=0)))
    for name,f in fixtures:
        report,solution=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan,time_limit_seconds=10)
        d=out/'fixed_plan_development'/name;d.mkdir(parents=True)
        _json_file(d/'metrics.json',report);_save_solution(d/'solution.npz',solution)
        need(report['plan_sha256_before']==report['plan_sha256_after'],name+' unchanged plan')
        need(report['solver']['certificate']['optimal_certificate_valid'],name+' fixed-plan LP certificate')
        physical.append(dict(name=name,metrics=report['metrics'],costs=report['costs'],path=str(d)))
    need(abs(physical[0]['costs']['Fshed_usd']+physical[0]['costs']['Fshort_usd']+physical[0]['costs']['FV2G_usd']+physical[0]['costs']['Fdeg_usd']-1.02)<1e-8,'Independent single-bus operating cost 1.02 USD')
    need(physical[1]['metrics']['shortfall_kwh']==0 and physical[1]['metrics']['v2g_ac_kwh']==2,'Procurement-dispatch difference is not reserve shortfall')
    need(all(physical[2]['metrics'][k] is None for k in ('RCR','OTR','MBQR','DRR')),'Zero-plan ratios are NA')
    series=response_times([0,2,3,8],[10,10,20,10],[1,1,10,0])
    need(series['bridge_h']==2 and series['first_reserve_h']==1 and series['early_certified_kwh']==5,'Hand-series first/bridge/early')
    need(abs(series['CLSR']-38/50)<1e-14 and abs(series['early_CLSR']-28/40)<1e-14,'Load-energy weighted CLSR')
    upper=empirical_cvar95(np.arange(1.,31.));lower=worst_decile_clsr(np.arange(15.)/100.)
    need(abs(upper['value']-(30+.5*29)/1.5)<1e-12 and upper['tail_mass_in_samples']==1.5,'Exact fractional upper tail')
    need(abs(lower['value']-.005/1.5)<1e-12 and lower['tail_mass_in_samples']==1.5,'Exact fractional lower tail')
    values=np.empty((8,2,3,2));ids=np.empty(values.shape,dtype=object)
    for t in range(8):
        for b in range(2):
            for v in range(3):
                values[t,b,v]=[100*t+10*b+v,100*t+10*b+v+7]
                ids[t,b,v,:]=f'statistical_handcheck/t{t}/b{b}/v{v}'
    samples=PairedSamples(values,tuple(f't{t}' for t in range(8)),('b0','b1'),[['v0','v1','v2']]*2,('reference','plus7'),ids)
    statistics=paired_bootstrap_mean(samples,seed=20260911,reference_method='reference')
    again=paired_bootstrap_mean(samples,seed=20260911,reference_method='reference')
    need(statistics['bootstrap_indices_sha256']==again['bootstrap_indices_sha256'],'Same-seed resampling identity')
    contrast=statistics['aggregate']['plus7']['difference_from_reference']
    need(abs(contrast['estimate']-7)<1e-12 and np.allclose(contrast['ci95'],[7,7],atol=1e-12,rtol=0),'Paired constant difference hand proof')
    _json_file(out/'statistics_input.json',dict(values=values,layout=samples.layout_identity,scope='synthetic statistical handcheck only'))
    _json_file(out/'paired_bootstrap.json',statistics)
    _json_file(out/'metric_handchecks.json',dict(checks=checks,time_series=series,cvar95=upper,worst_decile=lower,physical=physical))
    _json_file(out/'evaluator_contract.json',dict(metrics=METRIC_CONTRACT,statistics={k:statistics[k] for k in ('replications','confidence_level','interval_method','na_policy','random_stream','resampling_axis','cross_target_estimand','inference_scope')},formal_OOS_gateway_implemented=False,full_S04_pass=False))
    _json_file(out/'source_provenance.json',source_provenance())
    report=dict(status='PASS_S04_I01_DEVELOPMENT_FOUNDATION',full_S04_pass=False,formal_OOS=False,
        first_stage_optimization_calls=0,fixed_plan_LP_count=3,checks_passed=len(checks),wall_seconds=time.monotonic()-began,
        prerequisites=dict(S03=dict(path=str(args.s03_acceptance),sha256=args.s03_acceptance_sha256)),
        remaining_S04=['formal plan-bank locking gateway','planning optimism and route co-failure','raw-to-table development integration','formal protocol/config/evaluator freeze'])
    _json_file(out/'summary.json',report)
    _json_file(out/'artifact_sha256.json',{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps(report));return 0

if __name__=='__main__':raise SystemExit(main())
