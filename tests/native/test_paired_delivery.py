"""Actual native plans -> interrupted paired LP evaluation -> offline publication."""
import io
import json
import time
import numpy as np
import pytest
from ev_assignment_experiments.config.run import RunPlan
from ev_assignment_experiments.datasets.synthetic import problem
from ev_assignment_experiments.datasets.evaluation import domain_identity
from ev_assignment_experiments.runtime.supervisor import launch, status
from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.tracking.store import Store, Ref, encode, sha256
from ev_assignment_experiments.tracking.decisions import decide
from ev_assignment_experiments.workflows import l08, evaluation
from ev_assignment_experiments.publication.bundle import select, build, verify, render


def pack(folder):
    folder.mkdir()
    p=problem('DRO'); samples=[]
    for sid in p.scenario_ids:
        sc=p.load_scenario(sid); key=sid.replace('/','-')
        b=io.BytesIO(); np.savez_compressed(b,**sc.physical)
        raw=b.getvalue();(folder/f'{key}.npz').write_bytes(raw)
        replay=encode(dict(route_ids=[q['route_id'] for q in p.routes],states=list(sc.route_states)))
        (folder/f'{key}.json').write_bytes(replay)
        samples.append(dict(sample_id='evaluation/'+key,origin='synthetic development fixture (not held-out research OOS)',
            physical=dict(path=f'{key}.npz',sha256=sha256(raw)),route_states=dict(path=f'{key}.json',sha256=sha256(replay))))
    return dict(schema_version=1,scope='development',protocol='L08_PAIRED_FIXED_PLAN_V1',
        domain_sha256=domain_identity(p),random_stream='synthetic-fixed-pair-v1',samples=samples)


def test_actual_plans_paired_resume_offline_figures_and_tamper(tmp_path):
    store=Store(tmp_path/'store'); attempts=[]; decisions=[]
    for method in ['DET','DRO','ARO']:
        recipe=RunPlan.parse(dict(schema_version=2,study='comparison',title='Development paired delivery',method=method,
            input_kind='synthetic_l08',budget_group='comparison-'+method,budget_seconds=60,solver=dict(seconds=25,absolute_gap=1e-5)))
        run=l08.plan(store,recipe); started=launch(store,run); deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            state=status(store,started['attempt'])
            if state['status']=='settled':break
            time.sleep(.05)
        assert state['status']=='settled' and state['returncode']==0,state
        a=Ref(**state['attempt']);attempts.append(a)
        decisions.append(decide(store,a,'adopted','Small direct/native/physical validation',actor='test researcher'))
    root=tmp_path/'samples'; manifest=pack(root)
    run=evaluation.prepare(store,study='comparison',decisions=decisions,manifest=manifest,budget_group='paired-budget',budget_seconds=120)
    assert not list(store.records('events')) == [] # planning events exist; evaluation has none
    assert evaluation.rows_for(store,run)=={}
    partial=evaluation.execute(store,run,input_root=str(root),seconds=30,max_cells=1)
    assert not partial['passed'] and partial['cells']==1 and len(partial['missing'])==5
    rows=evaluation.rows_for(store,run); original={r:store.path(r).read_bytes() for r,p in rows.values()}
    spent=BudgetLedger(store.root,'paired-budget').read()['spent_seconds']
    complete=evaluation.execute(store,run,input_root=str(root),seconds=30)
    assert complete['passed'] and complete['cells']==6 and complete['solver_calls']==0
    assert BudgetLedger(store.root,'paired-budget').read()['spent_seconds']>spent>0
    assert all(store.path(r).read_bytes()==raw for r,raw in original.items())
    for method in ['DET','DRO','ARO']:
        assert complete['metrics']['actual_V2G_ac']['methods'][method]['mean']==pytest.approx(30)
        assert complete['metrics']['critical_ENS']['methods'][method]['mean']==pytest.approx(0)
    # A completed replay does no more solving or accounting.
    charged=BudgetLedger(store.root,'paired-budget').read()['spent_seconds']
    assert evaluation.execute(store,run,input_root=str(root))['passed']
    assert BudgetLedger(store.root,'paired-budget').read()['spent_seconds']==charged
    desc,locked,m=evaluation.describe(store,run)
    p,result=evaluation.planning(store,locked['plans']['DET'])
    row=evaluation.rows_for(store,run)['DET',m['samples'][0]['sample_id']][1]
    changed=dict(row,metrics=dict(row['metrics'],cost=row['metrics']['cost']+1))
    with pytest.raises(ValueError,match='metric differs'):
        evaluation.verify_cell(store,desc,locked,m['samples'][0],changed,p,result)
    for field,value in [('plan_sha256','0'*64),('random_stream','other'),('source',dict(kind='objects',id='0'*64))]:
        with pytest.raises(ValueError,match='identity'):
            evaluation.verify_cell(store,desc,locked,m['samples'][0],dict(row,**{field:value}),p,result)
    items=[dict(label='plan-'+str(i),attempt=a.as_dict(),decision=d.as_dict(),generator='procurement_profile') for i,(a,d) in enumerate(zip(attempts,decisions))]
    pub=select(store,'comparison','Paired development example',items,evaluations=[dict(label='Common paired cost',run=run.as_dict(),generator='paired_metrics')])
    out=tmp_path/'bundle'; result=build(store,attempts,out,publication=pub)
    assert not result['paper_facing'] and verify(out)['passed']
    import shutil
    shutil.rmtree(root);shutil.rmtree(store.root)
    assert verify(out)['passed']
    tables={p.name:p.read_bytes() for p in (out/'tables').glob('*.csv')}
    svgs={p.name:p.read_bytes() for p in (out/'figures').glob('*.svg')}
    render(out)
    assert {p.name:p.read_bytes() for p in (out/'tables').glob('*.csv')}==tables
    assert {p.name:p.read_bytes() for p in (out/'figures').glob('*.svg')}==svgs
    assert len(list((out/'figures').glob('*.pdf')))==4
    assert 'mean' in (out/'tables'/f'evaluation-{run.id}.csv').read_text()
