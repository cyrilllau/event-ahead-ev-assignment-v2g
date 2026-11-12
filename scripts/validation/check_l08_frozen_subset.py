"""Bounded development solve on an explicit subset of the real frozen physical data.

This is not the 30-EV/400-support experiment. It exercises an actual frozen route,
IEEE33 physical network, native Gurobi, and independent explicit-task audit locally.
"""
import argparse,json,hashlib,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from ev_assignment.domain.planning import Scenario,SolverOptions
from ev_assignment.methods.l08 import solve
from ev_assignment_experiments.datasets.l08 import L08Dataset
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
a=p.parse_args();d=L08Dataset(a.inputs,a.manifest);source=d.problem('DET')
chosen=next(i for i,q in enumerate(source.routes) if q['c']==1 and q['arrival']<=13.75)
q=dict(source.routes[chosen]);original_vehicle=q['e'];q['e']=0
scene=source.load_scenario('forecast44')
subset=replace(source,identity=hashlib.sha256((source.identity+str(chosen)+'one-EV-k55-j63').encode()).hexdigest(),
    routes=(q,),fleet_size=1,energy_bounds=np.array([source.energy_bounds[original_vehicle]]),
    starts=(55,),ends=(63,),scenario=lambda sid:Scenario(sid,scene.physical,(scene.route_states[chosen],)),
    scope='development_frozen_one_EV_subset',det_proof=None)
started=time.monotonic();result=solve(subset,'DET',SolverOptions(seconds=60,absolute_gap=1e-5))
assert result.audit and result.audit.passed and result.gap_passed
report=dict(passed=True,scope=subset.scope,full_experiment_pass=False,source_manifest=a.manifest,
    original_route_index=chosen,original_vehicle=original_vehicle,route_id=q['route_id'],
    native_status=result.native_status,lower=result.lower.value,upper=result.upper.value,
    gap=result.upper.value-result.lower.value,physics_passed=result.audit.passed,
    elapsed_seconds=time.monotonic()-started,accepted_for_comparison=result.accepted_for_comparison,
    scenario_metrics=result.audit.checks['scenarios'])
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
