"""Keep a real three-plan paired-evaluation bundle for clean-environment QA.

Explicit output must not already exist. Requires a local Gurobi license; never SSH.
"""
import argparse
import io
import json
from pathlib import Path
import time
import numpy as np
from ev_assignment_experiments.config.run import RunPlan
from ev_assignment_experiments.datasets.synthetic import problem
from ev_assignment_experiments.datasets.evaluation import domain_identity
from ev_assignment_experiments.runtime.supervisor import launch, status
from ev_assignment_experiments.tracking.store import Store, Ref, encode, sha256
from ev_assignment_experiments.tracking.decisions import decide
from ev_assignment_experiments.workflows import l08, evaluation
from ev_assignment_experiments.publication.bundle import select, build, verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args()
    root=Path(a.output)
    if not root.is_absolute():raise ValueError('Explicit absolute validation destination required')
    root.mkdir(parents=True,exist_ok=False)
    store=Store(root/'store');attempts=[];decisions=[]
    for method in ['DET','DRO','ARO']:
        recipe=RunPlan.parse(dict(schema_version=2,study='paired-development',title='Complete paired development flow',method=method,
            input_kind='synthetic_l08',budget_group='plan-'+method,budget_seconds=60,solver=dict(seconds=25,absolute_gap=1e-5)))
        r=l08.plan(store,recipe);admission=launch(store,r);deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            done=status(store,admission['attempt'])
            if done['status']=='settled':break
            time.sleep(.1)
        if done['status']!='settled' or done['returncode']!=0:raise RuntimeError(done)
        attempt=Ref(**done['attempt']);attempts.append(attempt)
        decisions.append(decide(store,attempt,'adopted','Development numerical validation only',actor='validation harness'))
    samples=root/'samples';samples.mkdir();rows=[];p=problem('DRO')
    for sid in p.scenario_ids:
        sc=p.load_scenario(sid);name=sid.replace('/','-');b=io.BytesIO();np.savez_compressed(b,**sc.physical)
        physical=b.getvalue();replay=encode(dict(route_ids=[q['route_id'] for q in p.routes],states=list(sc.route_states)))
        (samples/(name+'.npz')).write_bytes(physical);(samples/(name+'.json')).write_bytes(replay)
        rows.append(dict(sample_id='evaluation/'+name,origin='synthetic development fixture; not research OOS',
            physical=dict(path=name+'.npz',sha256=sha256(physical)),route_states=dict(path=name+'.json',sha256=sha256(replay))))
    manifest=dict(schema_version=1,scope='development',domain_sha256=domain_identity(p),protocol='L08_PAIRED_FIXED_PLAN_V1',random_stream='synthetic-fixed-pair-v1',samples=rows)
    run=evaluation.prepare(store,study='paired-development',decisions=decisions,manifest=manifest,budget_group='evaluation',budget_seconds=120)
    first=evaluation.execute(store,run,input_root=str(samples),seconds=30,max_cells=1)
    final=evaluation.execute(store,run,input_root=str(samples),seconds=30)
    assert first['cells']==1 and not first['passed'] and final['passed']
    pub=select(store,'paired-development','Complete paired development package',
        [dict(label=m,attempt=r.as_dict(),decision=d.as_dict(),generator='procurement_profile') for m,r,d in zip(['DET','DRO','ARO'],attempts,decisions)],
        evaluations=[dict(label='Common total cost',run=run.as_dict(),generator='paired_metrics')])
    bundle=build(store,attempts,root/'bundle',publication=pub)
    result=dict(passed=verify(root/'bundle')['passed'],scope='development',formal_experiment_pass=False,
        attempts=[r.as_dict() for r in attempts],evaluation=run.as_dict(),publication=pub.as_dict(),
        source_releases=bundle['source_releases'],first_slice=first,final=final)
    (root/'receipt.json').write_bytes(encode(result));print(json.dumps(dict(passed=result['passed'],evaluation=run.as_dict(),scope='development')))


if __name__=='__main__':main()
