#!/usr/bin/env python3
"""Compare frozen and installed generators at fixed development stream identities.

Only 16 small scenarios (eight targets, one planning and one calibration) are
regenerated per source version. The accepted S01 bank is never changed.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

WORKER = r'''
from pathlib import Path
import importlib,json,sys
import evcs_experiments
from evcs_experiments.joint_scenarios import JointContext,array_hash
case,routes,source=map(Path,sys.argv[1:4])
assert Path(evcs_experiments.__file__).resolve().is_relative_to(source.resolve())
if sys.argv[4]=='frozen':
 from validate_s01_replay import network_arcs
else:
 from evcs_experiments.validation.validate_s01_replay import network_arcs
ctx=JointContext(case,network_arcs(case))
result={'input_hashes':dict(ctx.input_hashes),'grid_sha256':ctx.grid.grid_sha256,'samples':[]}
for event in ctx.events:
 bank=json.loads((routes/f'{event}.tasks.json').read_text())
 target=ctx.prepare_target(event,bank)
 for split in ('planning','calibration'):
  sample=ctx.generate(target,ctx.events[0],0,split)
  result['samples'].append({'target':event,'split':split,'scenario_sha256':sample.scenario_sha256,
      'ledger':dict(sample.ledger),'array_hashes':{name:array_hash(a) for name,a in sample.arrays.items()}})
print(json.dumps(result,sort_keys=True,allow_nan=False))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--routes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.project_root.resolve()
    case = root / 'data/cases/ieee33-v54-q1'
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for label, source in [('frozen', root / 'provenance/legacy_s01/runner'), ('current', root / 'src')]:
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', WORKER, str(case), str(args.routes.resolve()), str(source), label],
                                cwd=args.output.resolve(), env=env, capture_output=True, text=True, timeout=60)
        (args.output / (label + '.stderr.log')).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{label} failed: {result.stderr}')
        versions[label] = json.loads(result.stdout)
        (args.output / (label + '.json')).write_text(json.dumps(versions[label], indent=2) + '\n')
    if versions['frozen'] != versions['current']:
        raise AssertionError('Fixed-stream scientific outputs or provenance differ after relocation')
    summary = {'status': 'PASS', 'targets': 8, 'scenarios_per_version': 16, 'frozen_inputs': len(versions['current']['input_hashes']),
               'compared': 'All scenario arrays, model identities and full scenario ledgers; exact equality',
               'formal_experiments_run': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
