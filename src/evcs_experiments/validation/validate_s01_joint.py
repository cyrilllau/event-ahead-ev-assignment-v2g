#!/usr/bin/env python3
"""Development joint-bank generation and moment preprocessing with checkpoints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_paths, frozen_control_root, source_provenance, acceptance_for, historical_path, verify_scientific_kernel_identity)

from evcs_experiments.joint_scenarios import JointContext,array_hash
from evcs_experiments.moment_calibration import bootstrap_moments,calibrate_moments
from evcs_experiments.random_streams import canonical_hash
from .validate_s01_replay import find_acceptance,network_arcs
from .validate_s01_wind import write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verify_scientific_kernel_identity()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case-root','route-evidence','last-acceptance','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();case=a.case_root.resolve();out=a.output.resolve();routes=a.route_evidence.resolve()
    out.mkdir(parents=True,exist_ok=False);runner=frozen_runner_root()
    previous=json.loads(a.last_acceptance.read_text())
    if previous['increment']!='S01-I07' or previous['status']!='PASS_INCREMENT':raise ValueError('Expected accepted I07 source baseline')
    for name,digest in previous['source_hashes'].items():
        if sha(runner/name)!=digest:raise ValueError(f'Changed accepted source: {name}')
    route_accept_path,route_accept=find_acceptance(routes,'I06')
    for name,entry in route_accept['raw_artifacts'].items():
        if sha(routes/name)!=entry['sha256']:raise ValueError(f'Changed frozen route source: {name}')
    command=[sys.executable,'-m','unittest','discover','-s',str(get_paths().project_root / 'tests'),'-p','test_*.py','-v']
    test=subprocess.run(command,cwd=case,capture_output=True,text=True,timeout=60)
    (out/'tests.stdout.log').write_text(test.stdout);(out/'tests.stderr.log').write_text(test.stderr)
    if test.returncode or re.search(r'skipped=\d+',test.stderr):
        write_json(out/'increment_validation.json',{'increment':'S01-I08','status':'FAIL_TESTS_OR_SKIPS','test_returncode':test.returncode})
        return 2
    tests=int(re.search(r'Ran (\d+) tests',test.stderr).group(1))
    context=JointContext(case,network_arcs(case))
    for name,digest in context.input_hashes.items():
        if sha(case/name)!=digest:raise ValueError(f'Source bytes differ from loaded model: {name}')
    upstream={'last_acceptance':str(a.last_acceptance.resolve()),'last_acceptance_sha256':sha(a.last_acceptance),
              'route_acceptance':str(route_accept_path),'route_acceptance_sha256':sha(route_accept_path),
              'input_hashes':dict(context.input_hashes),'grid_sha256':context.grid.grid_sha256,
              'development':True,'regime':'R1','split_sizes':{'planning':160,'calibration':80},
              'preflight_only':a.preflight_only,'formal_OOS_outcomes_generated':False}
    write_json(out/'upstream_validation.json',upstream)
    grid_arrays={'points_xy_km':context.grid.points_xy_km,'point_ids':np.array(context.grid.point_ids),
                 'line_points_xy_km':context.lines.points_xy_km,'road_points_xy_km':context.roads.points_xy_km,
                 'line_ids':np.array(context.lines.line_ids),'road_ids':np.array(context.roads.arc_ids),
                 'bus_ids':np.array(context.load.bus_ids),'baseline_p_kw':context.load.baseline_p_kw,
                 'baseline_q_kvar':context.load.baseline_q_kvar,'baseline_archived_q_kvar':context.load.archived_baseline_q_kvar}
    np.savez_compressed(out/'static_axes.npz',**grid_arrays)
    events=context.events[:1] if a.preflight_only else context.events
    overview=[];feature_rows=[];moments=[];full_ledger=[];source_payload_hashes={}
    for event in events:
        bank=json.loads((routes/f'{event}.tasks.json').read_text())
        prepared=context.prepare_target(event,bank)
        indices={'target_event':event,'task_sha256':bank['task_sha256'],'task_ids':list(prepared.tasks.task_ids),
                 'task_vehicle_ids':list(prepared.feature_tasks.vehicle_ids),
                 'forecast_usable_dc_kwh':prepared.feature_tasks.forecast_usable_dc_kwh.tolist(),
                 'task_weights':prepared.feature_tasks.task_weights.tolist(),
                 'assignment_state_sha256':prepared.target.state_sha256,'times_h':prepared.target.replay_times_h.tolist()}
        write_json(out/f'{event}.task_index.json',indices)
        split_features={};split_ledgers={}
        for split,formal_variants in (('planning',20),('calibration',10)):
            variants=1 if a.preflight_only else formal_variants
            lists={};ledger=[];diagnostics=[]
            for variant in range(variants):
                for base in context.events:
                    sample=context.generate(prepared,base,variant,split)
                    for name,value in sample.arrays.items():lists.setdefault(name,[]).append(value)
                    ledger.append(dict(sample.ledger));diagnostics.append(dict(sample.diagnostics))
                    feature_rows.append({'target_event':event,'split':split,'base_weather_event':base,'variant':variant+1,
                                         'scenario_id':sample.ledger['scenario_id'],**{name:float(value) for name,value in zip(
                                             ('Phi_W','Phi_L','Phi_R','Phi_D','Phi_load'),sample.arrays['features'])}})
            arrays={name:np.stack(value) for name,value in lists.items()}
            arrays.update({'scenario_ids':np.array([row['scenario_id'] for row in ledger]),
                           'task_ids':np.array(prepared.tasks.task_ids),'times_h':prepared.target.replay_times_h})
            split_features[split]=arrays['features'];split_ledgers[split]=ledger
            payload={name:array_hash(value) for name,value in arrays.items()}
            source_payload_hashes[f'{event}/{split}']=payload
            # Every scenario, every numerical field is regenerated in reverse
            # order. Cached exogenous primitives are immutable and do not draw
            # by call order; later independent audit reconstructs them fresh.
            for index in reversed(range(len(ledger))):
                row=ledger[index]
                repeat=context.generate(prepared,row['base_weather_event'],row['variant_index'],split)
                if repeat.scenario_sha256!=row['scenario_sha256']:raise ValueError('Reverse regeneration provenance differs')
                for name,value in repeat.arrays.items():
                    if not np.array_equal(value,arrays[name][index]):raise ValueError(f'Reverse regeneration differs: {event}/{split}/{index}/{name}')
            np.savez_compressed(out/f'{event}.{split}.npz',**arrays)
            write_json(out/f'{event}.{split}.ledger.json',{'scenarios':ledger,'diagnostics':diagnostics,'array_payload_hashes':payload,
                       'reverse_regeneration_all_fields_identical':True})
            full_ledger.extend(ledger)
            write_json(out/'progress.json',{'completed_targets':overview,'current_target':event,'completed_split':split,
                                          'completed_scenarios':len(full_ledger),'preflight_only':a.preflight_only})
            print(json.dumps({'event':event,'split':split,'scenarios':len(ledger),'task_count':len(prepared.tasks.task_ids)}),flush=True)
            del arrays,lists
        bootstrap=bootstrap_moments(split_features['calibration'],development=True,replications=2000,alpha=.05)
        physical_upper=(1.,1.,1.,0. if prepared.feature_tasks.empty_H else 1.,1.09)
        calibration=calibrate_moments(split_features['planning'],bootstrap,physical_upper=physical_upper)
        moment=calibration.as_dict()
        write_json(out/f'{event}.moments.json',moment)
        np.savez_compressed(out/f'{event}.bootstrap.npz',indices=bootstrap.indices,means=bootstrap.bootstrap_means,
                            center=bootstrap.center,halfwidth=bootstrap.halfwidth)
        bootstrap_meta={'index_sha256':bootstrap.index_sha256,'stream_identity':dict(bootstrap.stream_identity)}
        write_json(out/f'{event}.bootstrap.json',bootstrap_meta)
        if calibration.status!='PASS_MOMENT_CALIBRATION':
            write_json(out/'increment_validation.json',{'increment':'S01-I08','status':'FAIL_MOMENT_CALIBRATION',
                       'target_event':event,'moment_status':calibration.status,'full_S01_pass':False,
                       'completed_scenarios':len(full_ledger),'preflight_only':a.preflight_only})
            return 3
        overview.append({'event':event,'task_count':len(prepared.tasks.task_ids),'planning_scenarios':len(split_features['planning']),
                         'calibration_scenarios':len(split_features['calibration']),'task_sha256':bank['task_sha256'],
                         'delta':float(calibration.delta),'bootstrap_index_sha256':bootstrap.index_sha256})
        moments.append({'event':event,**moment})
        write_json(out/'progress.json',{'completed_targets':overview,'completed_scenarios':len(full_ledger),'preflight_only':a.preflight_only})
        print(json.dumps(overview[-1]),flush=True)
    for filename,rows in [('features.csv',feature_rows),('scenario_ledger.csv',[{key:row[key] for key in
            ('scenario_id','target_event','base_weather_event','residual_donor_event','variant','split','regime','lead_h','scenario_sha256','task_sha256')} for row in full_ledger])]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    contract=frozen_control_root() / 'FEATURE_MOMENT_CONTRACT_CN.md'
    sources=frozen_source_paths()
    summary={'increment':'S01-I08','status':'PASS_PREFLIGHT' if a.preflight_only else 'PASS_INCREMENT_AUTOMATED',
             'full_S01_pass':False,'created_at_utc':datetime.now(timezone.utc).isoformat(),'targets':overview,
             'total_scenarios':len(full_ledger),'test_count':tests,'test_command':command,'test_returncode':test.returncode,
             'source_hashes':{str(path.relative_to(runner)):sha(path) for path in sources},
             'contract_sha256':sha(contract),'upstream_validation':upstream,'array_payload_hashes':source_payload_hashes,
             'all_fields_reverse_regenerated_identically':True,'formal_OOS_outcomes_generated':False,
             'production_experiments_executed':False,'runtime':{'host':platform.node(),'python':platform.python_version(),'numpy':np.__version__},
             'requires_independent_full_bank_audit':True}
    write_json(out/'increment_validation.json',summary)
    write_json(out/'artifact_sha256.json',{path.name:sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    print(json.dumps({'status':summary['status'],'scenarios':len(full_ledger),'output':str(out)}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
