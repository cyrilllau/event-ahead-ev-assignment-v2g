"""Read-only local validation of frozen400 identities and exact native matrices.

Usage: python scripts/validation/check_l08_frozen.py --inputs ABS --manifest SHA --output ABS
No solver optimization and no SSH. Outputs belong outside the code checkout.
"""
import argparse,json,time
from pathlib import Path
import gurobipy as gp
import numpy as np
from scipy import sparse
from ev_assignment.domain.planning import SolverOptions
from ev_assignment.l08 import builders
from ev_assignment_experiments.datasets.l08 import L08Dataset,R41
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
a=p.parse_args();begun=time.monotonic();d=L08Dataset(a.inputs,a.manifest)
report=dict(inputs=d.inputs.verify(),solver_optimizations=0)
for sid in ['forecast44']+[f'train/{i:04d}' for i in range(400)]:d.scene(sid)
report['scenario_identity_count']=401
amb=d.ambiguity();report['probability_original']=amb.check_original(amb.feasible_anchor)
with gp.Env(params={'OutputFlag':0}) as env:
 problem=d.problem('DRO');model,_=builders.first_stage(problem,env=env)
 try:
  actual=builders.arrays(model)
  original=sparse.load_npz(d.inputs.path(R41+'/identity/FIRST_A.npz'))
  z=d.inputs.npz(R41+'/identity/FIRST_ARRAYS.npz')
  assert actual['A'].shape==original.shape and (actual['A']!=original).nnz==0
  for k in ['cost','lo','hi','rhs','sense']:assert np.array_equal(actual[k],z[k]),k
  names=d.inputs.read(R41+'/identity/FIRST_NAMES.json');rows=d.inputs.read(R41+'/identity/FIRST_ROWS.json')
  assert actual['names']==names and actual['row_names']==rows
  report['first_stage']=dict(rows=model.NumConstrs,columns=model.NumVars,binaries=model.NumBinVars,exact_coefficients=True)
 finally:model.dispose()
 model,_,_,_=builders.extensive(d.problem('DET'),'DET',SolverOptions(),env=env)
 try:
  report['det_r38']=dict(rows=model.NumConstrs,columns=model.NumVars,binaries=model.NumBinVars,
    screened_routes=sum(v.VarName.startswith('route[') and v.UB==0 for v in model.getVars()),
    exact_capacity_proof=True)
  assert model.NumVars==99234 and model.NumBinVars==25770
  # Shared strengthening is redundant in the integer model and remains explicit.
 finally:model.dispose()
report.update(passed=True,elapsed_seconds=time.monotonic()-begun,oos_opened=False)
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
