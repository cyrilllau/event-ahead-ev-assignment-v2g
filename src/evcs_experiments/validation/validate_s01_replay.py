#!/usr/bin/env python3
"""Replay accepted development roads over accepted forecast-only frozen H."""
from pathlib import Path
from datetime import datetime, timezone
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

from evcs_experiments.case_inputs import load_case_inputs
from evcs_experiments.forecast_routes import Arc, ForecastNetwork
from evcs_experiments.random_streams import canonical_hash, event_order
from evcs_experiments.road_connectors import derive_road_connectors
from evcs_experiments.task_replay import prepare_task_bank, replay_prepared
from .validate_s01_wind import write_json, array_hash


PHYSICAL_FIELDS = (
    'arrival_h', 'gamma_route', 'route_time_h', 'travel_energy_kwh', 'arrival_soc',
    'varsigma', 'mu_full', 'usable_energy_dc_kwh', 'status', 'failed_arc_ids',
)
METHODS = ('EA', 'DSP')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check_file(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}')
    return actual


def find_acceptance(evidence, increment):
    return acceptance_for(evidence, increment)


def validate_upstream(case, roads, routes, runner):
    road_acceptance_path, road_acceptance = find_acceptance(roads, 'I04')
    route_acceptance_path, route_acceptance = find_acceptance(routes, 'I06')
    for acceptance in (road_acceptance, route_acceptance):
        for relative, expected in acceptance['source_hashes'].items():
            check_file(runner / relative, expected)
    road_manifest, route_manifest = read_json(roads / 'artifact_sha256.json'), read_json(routes / 'artifact_sha256.json')
    road_required = ('context.json', 'development_road_ledger.json', 'array_payload_hashes.json',
                     'development_roads.npz', 'increment_validation.json')
    for filename in road_required:
        check_file(roads / filename, road_manifest[filename])
    check_file(roads / 'development_roads.npz', road_acceptance['raw_npz_sha256'])
    for filename, entry in route_acceptance['raw_artifacts'].items():
        check_file(routes / filename, entry['sha256'])
        if (routes / filename).stat().st_size != entry['bytes']:
            raise ValueError(f'Accepted byte size mismatch: {filename}')
        if route_manifest[filename] != entry['sha256']:
            raise ValueError('I06 manifest disagrees with accepted raw artifact')
    context = read_json(roads / 'context.json')
    # Acceptance context_sha256 is the canonical JSON payload identity.
    if canonical_hash(context) != road_acceptance['context_sha256']:
        raise ValueError('Accepted road context identity differs')
    for group in ('wind_input_hashes', 'line_input_hashes', 'road_input_hashes', 'connector_input_hashes'):
        for relative, expected in context[group].items():
            check_file(case / relative, expected)
    route_summary = read_json(routes / 'increment_validation.json')
    check_file(routes / 'increment_validation.json', route_manifest['increment_validation.json'])
    for dependencies in route_summary['forecast_input_hashes'].values():
        for relative, expected in dependencies.items():
            check_file(case / relative, expected)
    if (context['baseline_lead_h'] != 1 or context['arc_ids'] != list(range(1, 89))
            or context['regimes'] != ['R1', 'R2', 'R3']):
        raise ValueError('Expected accepted lead=1, 88 arcs and three regime labels')
    provenance = {
        'road_acceptance': str(road_acceptance_path), 'road_acceptance_sha256': sha(road_acceptance_path),
        'route_acceptance': str(route_acceptance_path), 'route_acceptance_sha256': sha(route_acceptance_path),
        'road_evidence': str(roads), 'route_evidence': str(routes),
        'accepted_road_npz_sha256': road_acceptance['raw_npz_sha256'],
        'accepted_road_context_sha256': road_acceptance['context_sha256'],
        'road_ledger_sha256': road_manifest['development_road_ledger.json'],
        'route_raw_artifacts': route_acceptance['raw_artifacts'],
        'upstream_sources_and_case_inputs_verified': True,
        'scope': 'Accepted development planning draws only; no new or formal OOS physical draws',
    }
    return context, read_json(roads / 'development_road_ledger.json'), provenance


def network_arcs(case):
    with (case / '04_transportation/A__directed_road_arcs.csv').open(newline='', encoding='utf-8-sig') as source:
        rows = list(csv.DictReader(source))
    rows.extend(derive_road_connectors(case).arc_rows)
    rows.sort(key=lambda row: int(row['a']))
    if [int(row['a']) for row in rows] != list(range(1, 89)):
        raise ValueError('Full routing graph requires canonical arc IDs 1..88')
    return tuple(Arc(int(row['a']), int(row['m_road']), int(row['n_road']),
                     float(row['D_a_km']), float(row['T0_a_min']) / 60.) for row in rows)


def main():
    verify_scientific_kernel_identity()
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('case-root', 'road-evidence', 'route-evidence', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    args = parser.parse_args()
    case, roads, routes, out = (p.resolve() for p in
                                (args.case_root, args.road_evidence, args.route_evidence, args.output))
    out.mkdir(parents=True, exist_ok=False)
    runner = frozen_runner_root()
    context, ledger, upstream = validate_upstream(case, roads, routes, runner)
    write_json(out / 'upstream_validation.json', upstream)
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(get_paths().project_root / 'tests'), '-p', 'test_*.py', '-v']
    tests = subprocess.run(command, cwd=case, capture_output=True, text=True, timeout=60)
    (out / 'tests.stdout.log').write_text(tests.stdout)
    (out / 'tests.stderr.log').write_text(tests.stderr)
    if tests.returncode:
        write_json(out / 'increment_validation.json', {'increment': 'S01-I07', 'status': 'FAIL_TESTS',
                                                       'test_returncode': tests.returncode})
        return tests.returncode
    test_match = re.search(r'Ran (\d+) tests?', tests.stderr)
    inputs = load_case_inputs(case)
    events, arcs = event_order(inputs.event_ids), network_arcs(case)
    with np.load(roads / 'development_roads.npz', allow_pickle=False) as raw:
        payload_hashes = read_json(roads / 'array_payload_hashes.json')
        for key in raw.files:
            if array_hash(raw[key]) != payload_hashes[key]:
                raise ValueError(f'Accepted upstream numeric payload mismatch: {key}')
        travel_min = raw['travel_time_min']
        times = raw['baseline_replay_times_h']
        regime_labels = tuple(str(value) for value in raw['regimes'])
        if (travel_min.shape != (128, 3, 100, 88) or len(ledger) != 128
                or not np.array_equal(times, np.arange(-4, 96) / 4)
                or not np.array_equal(raw['arc_ids'], np.arange(1, 89))
                or regime_labels != tuple(context['regimes'])):
            raise ValueError('Accepted road arrays do not match their labeled axes')
    by_identity = {}
    for index, row in enumerate(ledger):
        key = (row['target_event'], row['base_weather_event'], int(row['variant']))
        if key in by_identity or row['residual_donor_event'] == row['target_event']:
            raise ValueError('Duplicate scenario identity or target residual donor leakage')
        by_identity[key] = index
    expected = {(target, base, variant) for target in events for base in events for variant in (1, 2)}
    if set(by_identity) != expected:
        raise ValueError('Road ledger must exactly cover 8 targets x 8 bases x 2 development variants')
    summaries, target_records, all_payload_hashes = [], [], {}
    for event in events:
        bank_path = routes / f'{event}.tasks.json'
        bank = read_json(bank_path)
        target = inputs.target(event, lead_h=1.)
        prepared = prepare_task_bank(bank, inputs.vehicle_parameters, target)
        task_ids = tuple(task['task_id'] for task in bank['tasks'])
        moving = np.asarray([task['distance_km'] > 0 for task in bank['tasks']])
        scenarios = []
        for variant in (1, 2):
            for base in events:
                road_index = by_identity[(event, base, variant)]
                for regime in regime_labels:
                    regime_index = regime_labels.index(regime)
                    source_row = ledger[road_index]
                    scenarios.append({
                        'scenario_id': f'{event}/development/planning/{base}/v{variant:03d}/{regime}',
                        'target_event': event, 'base_weather_event': base, 'variant': variant,
                        'regime': regime, 'split': 'planning', 'development': True, 'lead_h': 1.,
                        'road_npz_row_index': road_index, 'road_npz_regime_index': regime_index,
                        'road_ledger_index': road_index,
                        'residual_donor_event': source_row['residual_donor_event'],
                        'upstream_realization_sha256': source_row['realization_sha256'],
                        'upstream_wind_parameters_sha256': source_row['wind_parameters_sha256'],
                    })
        arrays = {field: [] for field in PHYSICAL_FIELDS}
        replay_records = []
        for method in METHODS:
            method_arrays = {field: [] for field in PHYSICAL_FIELDS}
            records = []
            for scenario in scenarios:
                ri, gi = scenario['road_npz_row_index'], scenario['road_npz_regime_index']
                network = ForecastNetwork(arcs, times, travel_min[ri, gi] / 60., horizon_end_h=24.,
                                          network_hash=canonical_hash(scenario))
                result = replay_prepared(network, prepared, departure_h=-1. if method == 'EA' else 0.,
                                         split='planning', development=True, scenario_identity=scenario)
                if result.task_sha256 != bank['task_sha256'] or tuple(result.task_ids) != task_ids:
                    raise ValueError('Realized replay changed accepted task identity/order')
                for field in PHYSICAL_FIELDS:
                    method_arrays[field].append(np.asarray(getattr(result, field)))
                records.append(result.as_dict())
            for field in PHYSICAL_FIELDS:
                arrays[field].append(np.stack(method_arrays[field]))
            replay_records.append(records)
        arrays = {key: np.stack(value) for key, value in arrays.items()}
        # Repeat in reversed execution order; compare every physical field.
        for mi in reversed(range(len(METHODS))):
            for si in reversed(range(len(scenarios))):
                scenario = scenarios[si]
                ri, gi = scenario['road_npz_row_index'], scenario['road_npz_regime_index']
                network = ForecastNetwork(arcs, times, travel_min[ri, gi] / 60., horizon_end_h=24.,
                                          network_hash=canonical_hash(scenario))
                repeat = replay_prepared(network, prepared, departure_h=-1. if METHODS[mi] == 'EA' else 0.,
                                         split='planning', development=True, scenario_identity=scenario)
                for field in PHYSICAL_FIELDS:
                    if not np.array_equal(np.asarray(getattr(repeat, field)), arrays[field][mi, si]):
                        raise ValueError(f'Reverse-order replay differs: {event}/{mi}/{si}/{field}')
        arrays.update({'task_ids': np.asarray(task_ids), 'methods': np.asarray(METHODS),
                       'scenario_ids': np.asarray([row['scenario_id'] for row in scenarios]),
                       'road_npz_row_indices': np.asarray([row['road_npz_row_index'] for row in scenarios]),
                       'road_npz_regime_indices': np.asarray([row['road_npz_regime_index'] for row in scenarios]),
                       'task_is_moving': moving, 'departure_hours': np.array([-1., 0.]),
                       'slot_starts_h': times})
        all_payload_hashes[event] = {key: array_hash(value) for key, value in arrays.items()}
        np.savez_compressed(out / f'{event}.replay.npz', **arrays)
        record = {'target_event': event, 'assignment_state_sha256': target.state_sha256,
                  'accepted_task_sha256': bank['task_sha256'], 'accepted_task_file_sha256': sha(bank_path),
                  'task_sha256': bank['task_sha256'],
                  'accepted_dependency_sha256': bank['dependency_sha256'], 'methods': list(METHODS),
                  'method_departure_h': {'EA': -1., 'DSP': 0.}, 'task_ids': list(task_ids),
                  'scenarios': scenarios, 'scenario_ledger': scenarios, 'replay_records': replay_records,
                  'array_payload_hashes': all_payload_hashes[event],
                  'repeat_every_physical_field_reverse_order_identical': True}
        write_json(out / f'{event}.replay.json', record)
        for mi, method in enumerate(METHODS):
            for regime in regime_labels:
                selected = np.array([row['regime'] == regime for row in scenarios])
                gamma, qualified = arrays['gamma_route'][mi, selected], arrays['varsigma'][mi, selected]
                summaries.append({'target_event': event, 'method': method, 'regime': regime,
                                  'scenarios': int(selected.sum()), 'tasks': len(task_ids),
                                  'moving_tasks': int(moving.sum()), 'zero_distance_tasks': int((~moving).sum()),
                                  'route_completion_fraction': float(gamma.mean()) if gamma.size else None,
                                  'qualified_fraction': float(qualified.mean()) if qualified.size else None,
                                  'moving_qualified_fraction': float(qualified[:, moving].mean()) if moving.any() else None,
                                  'mean_usable_dc_kwh': float(arrays['usable_energy_dc_kwh'][mi, selected].mean()) if gamma.size else None})
        target_records.append({'event': event, 'tasks': len(task_ids), 'scenarios': len(scenarios),
                               'methods': len(METHODS), 'task_sha256': bank['task_sha256'],
                               'replay_npz_sha256': sha(out / f'{event}.replay.npz')})
        write_json(out / 'progress.json', {'completed_targets': target_records})
        print(json.dumps(target_records[-1]), flush=True)
    with (out / 'replay_summary.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    sources = frozen_source_paths()
    summary = {'increment': 'S01-I07', 'status': 'PASS_INCREMENT_AUTOMATED', 'full_S01_pass': False,
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'targets': target_records,
               'test_count': int(test_match.group(1)) if test_match else None,
               'contract_sha256': sha(frozen_control_root() / 'TASK_REPLAY_CONTRACT_CN.md'),
               'test_command': command, 'test_returncode': tests.returncode,
               'source_hashes': {str(p.relative_to(runner)): sha(p) for p in sources},
               'upstream_validation': upstream, 'array_payload_hashes': all_payload_hashes,
               'frozen_H_unchanged': True, 'repeat_every_physical_field_reverse_order_identical': True,
               'baseline_road_paths': 384, 'method_replays': 768,
               'formal_OOS_outcomes_generated': False, 'production_experiments_executed': False,
               'runtime': {'host': platform.node(), 'python': platform.python_version(), 'numpy': np.__version__},
               'remaining_S01': ['five features', 'moment feasibility', 'complete scenario assembly']}
    write_json(out / 'increment_validation.json', summary)
    write_json(out / 'artifact_sha256.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'status': summary['status'], 'output': str(out)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
