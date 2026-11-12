"""S04-I02 development raw/replay/locked-plan/LP/table/statistics integration."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256,source_provenance
from evcs_experiments.evaluation_lock import file_reference,create_protocol_lock,lock_plan_batch,verify_plan_batch
from evcs_experiments.evaluation_batch import evaluate_locked_batch
from evcs_experiments.evaluation_metrics import plan_identity,METRIC_CONTRACT
from evcs_experiments.evaluation_statistics import PairedSamples,paired_bootstrap_mean
from evcs_experiments.evaluation_mechanisms import planning_optimism,shared_corridor_cofailure
from evcs_experiments.validation.s04_raw_fixture import raw_fixture,frozen_fixture,locked_factory,physical_fixture
from evcs_experiments.validation.s02_toys import _json_file


def run(output,i01_acceptance,i01_sha):
    started=time.monotonic();paths=get_paths();out=Path(output).resolve()
    if sha256(i01_acceptance)!=i01_sha:raise ValueError('S04-I01 prerequisite hash changed')
    card=json.loads(Path(i01_acceptance).read_text())
    if card.get('increment_id')!='S04-I01' or card.get('status')!='PASS_INCREMENT':raise ValueError('S04-I01 must be accepted')
    out.mkdir(parents=True,exist_ok=False);raw=raw_fixture();f=frozen_fixture(raw)
    _json_file(out/'raw_inputs.json',raw);_json_file(out/'budget.json',dict(domain='development',shared_wall_seconds=50,lp_seconds=10))
    _json_file(out/'solver_record.json',dict(scope='hand-authored fixed plan; no first-stage optimization',status='DEVELOPMENT_HAND_INPUT'))
    _json_file(out/'planning_support.json',dict(scope='explicit known forecast',certified_energy_prediction_kwh=24.,rule='forecast fixed-plan certified profile 1 kW for 24 hours'))
    ref=file_reference(out/'raw_inputs.json');source=paths.project_root/'src/evcs_experiments'
    expected=[];plans=[]
    for method in ['EA','DSP']:
        pid='fixture-'+method;expected.append(dict(plan_id=pid,target_id=raw['target_id'],method_id=method,setting_id='baseline'))
        meta=dict(x=f.plan.x.tolist(),r=f.plan.r.tolist(),task_sha256=f.plan.task_sha256,config_hash=f.plan.config_hash,execution_mode=method,departure_h=-1. if method=='EA' else 0.,ea_source_plan_id=None if method=='EA' else 'fixture-EA')
        _json_file(out/(pid+'.json'),meta)
        refs=dict(plan=file_reference(out/(pid+'.json')),tasks=ref,routes=ref,initial_state=ref,planning_support=file_reference(out/'planning_support.json'),solver_record=file_reference(out/'solver_record.json'),budget=file_reference(out/'budget.json'),planning_source=[file_reference(source/'validation/s04_raw_fixture.py')])
        plans.append(dict(**expected[-1],plan_sha256=plan_identity(f.plan),references=refs))
    refs={kind:[file_reference(source/'evaluation_lock.py')] for kind in ['source','evaluator','config','random_streams','experiment_matrix','budgets']}
    refs['source']=[file_reference(p) for p in sorted(source.glob('*.py'))]
    refs['evaluator']=[file_reference(source/'evaluation_batch.py'),file_reference(source/'validation/s04_raw_fixture.py')]
    refs['config']=[file_reference(paths.spec_root/'PROTOCOL.md'),file_reference(paths.spec_root/'runtime_policy.json')]
    refs['random_streams']=[file_reference(source/'random_streams.py')];refs['experiment_matrix']=[file_reference(paths.spec_root/'experiment_matrix.csv')];refs['budgets']=[file_reference(out/'budget.json')]
    samples={raw['target_id']:list(raw['samples'])}
    spec=dict(domain='development',model_version='IEEE33-v54-Q1',target_ids=[raw['target_id']],method_ids=['EA','DSP'],references=refs,batches=[dict(batch_id='raw-to-table',expected_plans=expected,sample_ids_by_target=samples)])
    locks=out/'locks';create_protocol_lock(locks,spec);lock_plan_batch(locks,'raw-to-table',plans)
    result=evaluate_locked_batch(locks,'raw-to-table',locked_factory,out/'development_end_to_end',domain='development',sample_ids_by_target=samples,wall_seconds=50,lp_time_limit_seconds=10)
    if result['status']!='PASS_LOCKED_FIXED_PLAN_BATCH':raise ValueError('Incomplete development integration')
    reports=[json.loads(p.read_text()) for p in sorted((out/'development_end_to_end').glob('item_*/metrics.json'))]
    values=np.zeros((1,2,2,2));ids=np.empty(values.shape,dtype=object)
    for report in reports:
        sample=raw['samples'][report['sample_id']];b=int(sample['base_id'][1:]);v=int(sample['variant_id'][1:])-1;m=['EA','DSP'].index(report['method_id']);values[0,b,v,m]=report['costs']['total_DSO_usd'];ids[0,b,v,m]=report['sample_id']
    paired=PairedSamples(values,(raw['target_id'],),('b0','b1'),[['v001','v002']]*2,('EA','DSP'),ids)
    stats=paired_bootstrap_mean(paired,seed=20260911,reference_method='EA');_json_file(out/'paired_cost_statistics.json',stats)
    mechanisms={}
    for method in ['EA','DSP']:
        rows=[r for r in reports if r['method_id']==method];sids=[r['sample_id'] for r in rows]
        failures=np.array([~physical_fixture(raw,sid,-1. if method=='EA' else 0.)[1].gamma_route for sid in sids])
        mechanisms[method]=dict(optimism=planning_optimism(24.,[r['energy_kwh']['certified'] for r in rows],sample_ids=sids),cofailure=shared_corridor_cofailure(f.tasks.task_ids,f.plan.x,{f.tasks.task_ids[0]:[1]},{1:'physical_1'},failures,sample_ids=sids))
    _json_file(out/'mechanisms.json',mechanisms)
    _json_file(out/'evaluator_contract.json',dict(metrics=METRIC_CONTRACT,fixed_plan=True,first_stage_optimization_calls=0,physical_replay='production task_replay; raw geometry/delays/state; exact Q1 adapter',formal_OOS=False,statistical_scope='single synthetic target; shared two-base/two-variant hand fixture'))
    _json_file(out/'source_provenance.json',source_provenance());verify_plan_batch(locks,'raw-to-table')
    report=dict(status='PASS_S04_I02_DEVELOPMENT_INTEGRATION',full_S04_pass=False,formal_OOS=False,item_count=len(reports),plan_count=2,fixed_plan_LP_count=8,first_stage_optimization_calls=0,paired_bootstrap_replications=2000,wall_seconds=time.monotonic()-started,prerequisite=file_reference(i01_acceptance),protocol_lock=file_reference(locks/'protocol.json'),plan_manifest=file_reference(locks/'batch-raw-to-table.json'))
    _json_file(out/'summary.json',report);_json_file(out/'artifact_sha256.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()});return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-root',type=Path,default=get_paths().data_root);p.add_argument('--s04-i01-acceptance',type=Path,required=True);p.add_argument('--s04-i01-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.output,a.s04_i01_acceptance,a.s04_i01_sha256)));return 0
if __name__=='__main__':raise SystemExit(main())
