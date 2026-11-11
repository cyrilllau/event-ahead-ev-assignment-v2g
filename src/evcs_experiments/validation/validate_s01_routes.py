#!/usr/bin/env python3
"""Forecast-only exact route/H development validation; formal OOS stays closed."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_paths, frozen_control_root, source_provenance, acceptance_for, historical_path, verify_scientific_kernel_identity)
from evcs_experiments.case_inputs import load_case_inputs
from evcs_experiments.forecast_tasks import load_forecast_task_inputs, build_candidate_tasks
from evcs_experiments.random_streams import event_order, canonical_hash
from .validate_s01_wind import write_json, array_hash


def main():
    verify_scientific_kernel_identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-seconds', type=float, default=5.)
    parser.add_argument('--max-expansions', type=int, default=1000000)
    args = parser.parse_args()
    case, out = args.case_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runner = frozen_runner_root()
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(get_paths().project_root / 'tests'), '-p', 'test_*.py', '-v']
    test = subprocess.run(command, cwd=case, capture_output=True, text=True, timeout=60)
    (out/'tests.stdout.log').write_text(test.stdout)
    (out/'tests.stderr.log').write_text(test.stderr)
    if test.returncode:
        write_json(out/'increment_validation.json', {'increment':'S01-I06', 'status':'FAIL_TESTS', 'test_returncode':test.returncode})
        return test.returncode
    events = event_order(load_case_inputs(case).event_ids)
    counts, dependencies, payloads = [], {}, {}
    all_complete = True
    for event in events:
        started = time.monotonic()
        inputs = load_forecast_task_inputs(case, event, lead_h=1.)
        bank = build_candidate_tasks(inputs, basic_soc=.3, tau_blk_h=1.,
                                     max_expansions=args.max_expansions, time_limit_seconds=args.query_seconds)
        record = bank.as_dict()
        write_json(out/f'{event}.tasks.json', record)
        arrays = {'forecast_speed_m_s':inputs.forecast_speed,
                  'forecast_travel_hours':inputs.forecast_travel_hours,
                  'slot_starts_h':inputs.network.slot_starts_h}
        np.savez_compressed(out/f'{event}.forecast.npz', **arrays)
        payloads[event] = {key:array_hash(value) for key, value in arrays.items()}
        dependencies[event] = dict(inputs.input_hashes)
        if bank.certified:
            repeat = build_candidate_tasks(inputs, basic_soc=.3, tau_blk_h=1.,
                                           max_expansions=args.max_expansions, time_limit_seconds=args.query_seconds)
            assert repeat.certified and repeat.task_sha256 == bank.task_sha256
            assert repeat.as_dict()['tasks'] == record['tasks']
        else:
            all_complete = False
        counts.append({'event':event, 'status':bank.status, 'certified':bank.certified,
                       'tasks':len(bank.tasks), 'queries':len(bank.queries),
                       'vehicles':len(bank.vehicle_summaries), 'task_sha256':bank.task_sha256,
                       'dependency_sha256':bank.dependency_sha256,
                       'elapsed_seconds_including_repeat':time.monotonic()-started})
        # Each target is a persistent checkpoint, even if a later target times out.
        write_json(out/'progress.json', {'completed_targets':counts, 'all_completed_queries_certified':all_complete})
        print(json.dumps(counts[-1]), flush=True)
        if not bank.certified:
            break
    sources = frozen_source_paths()
    contract = frozen_control_root() / 'FORECAST_ROUTE_CONTRACT_CN.md'
    summary = {'increment':'S01-I06', 'status':'PASS_INCREMENT_AUTOMATED' if all_complete and len(counts)==8 else 'SEARCH_INCOMPLETE',
               'full_S01_pass':False, 'created_at_utc':datetime.now(timezone.utc).isoformat(),
               'targets':counts, 'test_command':command, 'test_returncode':test.returncode,
               'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),
               'source_hashes':{str(p.relative_to(runner)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               'forecast_input_hashes':dependencies, 'array_payload_hashes':payloads,
               'repeat_task_payload_identical':all_complete and len(counts)==8,
               'formal_OOS_outcomes_generated':False, 'production_experiments_executed':False,
               'runtime':{'host':platform.node(), 'python':platform.python_version(), 'numpy':np.__version__},
               'remaining_S01':['realized fixed-route replay', 'features/moment feasibility', 'complete scenario assembly']}
    write_json(out/'increment_validation.json', summary)
    write_json(out/'artifact_sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'status':summary['status'], 'targets':len(counts), 'output':str(out)}), flush=True)
    return 0 if summary['status']=='PASS_INCREMENT_AUTOMATED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
