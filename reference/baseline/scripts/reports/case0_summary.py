#!/usr/bin/env python3
"""Tables from a hash-pinned independent Case0 OOS PASS, never pending metrics.

Declared statistical streams: legacy seed20260911; expanded seed20260912.
2000 matched within-base variant resamples; conditional finite-template inference.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from evcs_experiments.case0_profiles import LEGACY, EXPANDED, TARGETS, validate_config, lock_target
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.evaluation_statistics import PairedSamples,paired_bootstrap,_summary,_json_numbers

SEED=20260911
REPLICATIONS=2000
METHODS={'R1':('EA_DRO','DSP','OR_DRO','SAA','ARO','ZERO'),
         'R2':('EA_DRO','SAA','ARO'),'R3':('EA_DRO','SAA','ARO')}
SID=re.compile(r'^([^/]+)/case0/development/oos/([^/]+)/v(\d{5})/(R[123])$')


def need(ok,message):
    if not bool(ok):raise ValueError(message)


def study_scope(audit, seed=None):
    """Resolve the exact scientific lock, never infer a study from result labels."""
    def read_ref(ref):
        verify_reference(ref)
        return json.loads(Path(ref['path']).read_text())
    lock=read_ref(audit['lock_reference'])
    need(lock.get('kind')=='CASE0_PRE_OOS_LOCK_V1' and lock.get('formal_OOS') is False,
         'Wrong scientific OOS lock')
    acceptance=read_ref(audit['planning_acceptance_reference'])
    need(lock['acceptance_reference']==audit['planning_acceptance_reference'] and
         lock['config_reference']==acceptance['config_reference'], 'Planning/config lock identity differs')
    config=read_ref(lock['config_reference']);profile=validate_config(config)
    need(lock.get('profile',LEGACY)==profile and lock_target(lock)==config['target_event'],
         'Scientific profile/target differs from OOS lock')
    offset=30000 if profile==EXPANDED else 10000
    need(config['variant_start']==offset,'Scientific stream block differs')
    declared=20260912 if profile==EXPANDED else SEED
    need(config.get('statistics_seed',declared)==declared,'Undeclared statistics seed in configuration')
    need(seed is None or (type(seed) is int and seed==declared),'Undeclared statistics seed')
    return dict(profile=profile,target_event=TARGETS[profile],variant_start=offset,seed=declared,
                lock_reference=audit['lock_reference'],config_reference=lock['config_reference'])


def sample_axis(sample_id,regime,scope=None):
    scope=scope or dict(target_event='michael_2018',variant_start=10000)
    match=SID.fullmatch(sample_id)
    need(match is not None and match[1]==scope['target_event'] and match[4]==regime,
         'Wrong Case0 development sample ID/regime/target')
    variant=int(match[3]);need(scope['variant_start']+1<=variant<=scope['variant_start']+8,'Undeclared Case0 OOS variant')
    return match[2],f'v{variant:05d}'


def scalar_fields(row):
    fields={}
    for section in ('costs','metrics','energy_kwh'):
        for key,value in row[section].items():
            if isinstance(value,(list,dict)):continue
            need(value is None or isinstance(value,(int,float,bool)),'Unsupported scalar metric')
            if value is not None:need(np.isfinite(value),'Nonfinite metric')
            fields[section+'.'+key]=value
    return fields


def paired_samples(rows,regime,field,scope=None):
    scope=scope or dict(target_event="michael_2018",variant_start=10000)
    selected=[row for row in rows if row['item']['regime']==regime]
    axes={sample_axis(row['scenario_id'],regime,scope) for row in selected}
    bases=tuple(sorted({b for b,v in axes}));variants=tuple(f'v{i:05d}' for i in range(scope['variant_start']+1,scope['variant_start']+9))
    need(len(bases)==8 and axes=={(b,v) for b in bases for v in variants},'Exactly8 base by8 variant axes required')
    methods=METHODS[regime];values=np.full((1,8,8,len(methods)),np.nan)
    ids=np.empty(values.shape,dtype=object);reasons=np.full(values.shape,'',dtype=object);seen=set()
    for row in selected:
        method=row['item']['method_id'];need(method in methods,'Unexpected method/regime')
        base,variant=sample_axis(row['scenario_id'],regime,scope);key=(base,variant,method)
        need(key not in seen,'Duplicate paired observation');seen.add(key)
        b,v,m=bases.index(base),variants.index(variant),methods.index(method)
        fields=scalar_fields(row);need(field in fields,'Metric inventory differs across items')
        value=fields[field];ids[0,b,v,m]=row['scenario_id']
        if value is None:
            reason=row['na_reasons'].get(field.split('.',1)[1])
            need(isinstance(reason,str) and bool(reason),'Undefined metric needs original NA reason')
            reasons[0,b,v,m]=reason
        else:values[0,b,v,m]=float(value)
    need(len(seen)==64*len(methods),'Missing paired observation')
    return PairedSamples(values=values,target_ids=(scope['target_event'],),base_ids=bases,
        variant_ids=np.tile(variants,(8,1)),method_ids=methods,sample_ids=ids,na_reasons=reasons)


def bootstrap_references(samples,statistic='mean',*,seed=SEED):
    """One matched draw set/statistic evaluation; subtract each named reference."""
    original=paired_bootstrap(samples,seed=seed,reference_method='EA_DRO',statistic=statistic,
                              replications=REPLICATIONS,development=True)
    methods=samples.method_ids
    points=np.asarray([[target['methods'][method]['estimate'] for method in methods]
                       for target in original['per_target']],dtype=float)
    draws=np.asarray(original['bootstrap_statistics'],dtype=float)
    output=[]
    for reference in ('EA_DRO','SAA'):
        ref=methods.index(reference);differences=points-points[:,ref:ref+1]
        difference_draws=draws-draws[:,:,ref:ref+1]
        result=dict(original,reference_method=reference,
                    bootstrap_differences=_json_numbers(difference_draws))
        result['per_target']=[]
        for t,target in enumerate(original['per_target']):
            result['per_target'].append(dict(target,methods={method:dict(target['methods'][method],
                difference_from_reference=_summary(differences[t,m],difference_draws[:,t,m]))
                for m,method in enumerate(methods)}))
        result['aggregate']={method:dict(original['aggregate'][method],
            difference_from_reference=_summary(np.mean(differences[:,m]),np.mean(difference_draws[:,:,m],axis=1)))
            for m,method in enumerate(methods)}
        output.append(result)
    return output


def load_audit(path,sha256):
    ref=file_reference(path);need(ref['sha256']==sha256,'Independent audit SHA differs')
    audit=json.loads(Path(ref['path']).read_text())
    need(audit.get('kind')=='CASE0_INDEPENDENT_EVALUATION_NUMERICAL_AUDIT_V1' and audit.get('passed') is True,
         'Only independently audited Case0 OOS results are accepted')
    need(audit['item_count']==len(audit['items'])==768 and audit['formal_OOS'] is False and audit['Case1_acceptance'] is False,
         'Wrong audited group/scope')
    for key in ('original_coefficients_checked','fixed_plan_rhs_checked','full_primal_dual_kkt_checked','physical_metrics_checked','paired_axes_checked'):
        need(audit[key] is True,'Independent numerical audit scope incomplete')
    for key in ('lock_reference','evaluation_state_reference','planning_acceptance_reference',
                'evaluation_parent_settlement_reference','producer_source_manifest_reference','independent_review_reference'):
        verify_reference(audit[key])
    for source in audit['auditor_source_references']:verify_reference(source)
    scope=study_scope(audit)
    expected={f'{regime}.{i:03d}.{method}' for regime,methods in METHODS.items() for i in range(64) for method in methods}
    need({row['item']['item_id'] for row in audit['items']}==expected,'Audited item coverage differs')
    for row in audit['items']:
        need(row['passed'] is True,'Failed row in purported audit');verify_reference(row['record_reference'])
        item=row['item'];need(item['item_id']==f"{item['regime']}.{item['sample_index']:03d}.{item['method_id']}",'Item label differs')
        sample_axis(row['scenario_id'],item['regime'],scope)
    return ref,audit


def csv_file(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def summarize(audit_path,expected_sha256,output_directory,*,seed=None):
    reference,audit=load_audit(audit_path,expected_sha256);rows=audit['items'];scope=study_scope(audit,seed)
    output=Path(output_directory).resolve();need(not output.exists(),'Summary output must be a new directory')
    output.mkdir(parents=True)
    unit=[];curves=[];summary=[];bootstrap=[]
    fields=tuple(sorted(scalar_fields(rows[0])))
    need(all(tuple(sorted(scalar_fields(row)))==fields for row in rows),'Scalar reporting schema differs')
    for row in rows:
        regime=row['item']['regime'];base,variant=sample_axis(row['scenario_id'],regime,scope)
        identity=dict(**row['item'],scenario_id=row['scenario_id'],base_id=base,variant_id=variant,
                      plan_sha256=row['plan_sha256'],scenario_sha256=row['scenario_sha256'])
        unit.append(dict(identity,**scalar_fields(row),na_reasons=json.dumps(row['na_reasons'],sort_keys=True)))
        series=dict(row['timeseries_kw']);series['CLSR']=row['metrics']['CLSR_t']
        need(all(len(values)==24 for values in series.values()),'Original24-hour curve axis differs')
        for name,values in series.items():
            for period,value in enumerate(values,1):
                curves.append(dict(identity,quantity=name,period=period,value=value,undefined=value is None))
    for regime in METHODS:
        for field in fields:
            samples=paired_samples(rows,regime,field,scope)
            statistics=['mean']+(['cvar95'] if field.startswith('costs.') else [])
            if field=='metrics.CLSR':statistics.append('worst_decile_clsr')
            for statistic in statistics:
                for result in bootstrap_references(samples,statistic,seed=scope['seed']):
                    reference_method=result['reference_method']
                    for method,data in result['per_target'][0]['methods'].items():
                        difference=data['difference_from_reference']
                        summary.append(dict(regime=regime,metric=field,statistic=statistic,method=method,
                            reference_method=reference_method,estimate=data['estimate'],ci95_lower=data['ci95'][0],ci95_upper=data['ci95'][1],
                            difference=difference['estimate'],difference_ci95_lower=difference['ci95'][0],difference_ci95_upper=difference['ci95'][1],
                            status=data['status'],difference_status=difference['status'],na_count=data['na_count'],na_rate=data['na_rate'],
                            na_reasons=json.dumps(data['na_reasons'],sort_keys=True)))
                    bootstrap.append(dict(regime=regime,metric=field,**{k:v for k,v in result.items() if k not in
                        ('bootstrap_indices','bootstrap_statistics','bootstrap_differences','layout')}))
    means=[]
    for regime,methods in METHODS.items():
        for method in methods:
            selected=[r for r in curves if r['regime']==regime and r['method_id']==method]
            for quantity in sorted({r['quantity'] for r in selected}):
                for period in range(1,25):
                    values=[r['value'] for r in selected if r['quantity']==quantity and r['period']==period]
                    need(len(values)==64,'Curve pairing incomplete')
                    missing=sum(v is None for v in values)
                    means.append(dict(regime=regime,method=method,quantity=quantity,period=period,
                        mean=None if missing else float(np.mean(values)),na_count=missing,na_rate=missing/64))
    csv_file(output/'items.csv',unit);csv_file(output/'summary.csv',summary)
    csv_file(output/'curves_by_item.csv',curves);csv_file(output/'curves_mean.csv',means)
    payload=dict(kind='CASE0_AUDITED_DESCRIPTIVE_STATISTICS_V1',audit_reference=reference,seed=scope['seed'],study_scope=scope,
        replications=REPLICATIONS,statistical_stream='New predeclared Case0 summary stream; original scenario streams unchanged',
        method_difference='method minus named reference; tail differences are differences of CVaRs',
        na_policy='Propagate undefined observations; no zero filling or complete-case deletion',
        inference_scope=f"Conditional on8 fixed base-weather templates and single {scope['target_event']} target; not new-storm population inference",
        bootstrap=bootstrap,script_reference=file_reference(__file__),
        statistics_source_reference=file_reference(__import__('evcs_experiments.evaluation_statistics',fromlist=['x']).__file__),
        outputs=[file_reference(p) for p in sorted(output.iterdir())])
    with (output/'summary.json').open('x') as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();summarize(args.audit,args.sha256,args.output,seed=args.seed)

if __name__=='__main__':main()
