#!/usr/bin/env python3
"""Bounded development road-layer validation, without formal OOS or routing."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_paths, frozen_control_root, source_provenance, acceptance_for, historical_path, verify_scientific_kernel_identity)
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.wind_scenarios import load_wind_bank, draw_parameters
from evcs_experiments.line_outages import load_line_model
from evcs_experiments.road_disruptions import load_road_model
from evcs_experiments.road_connectors import derive_road_connectors
from .validate_s01_wind import write_json, array_hash


def main():
    verify_scientific_kernel_identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case, out = args.case_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runner = frozen_runner_root()
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(get_paths().project_root / 'tests'), '-p', 'test_*.py', '-v']
    tests = subprocess.run(command, cwd=case, capture_output=True, text=True, timeout=60)
    (out/'tests.stdout.log').write_text(tests.stdout)
    (out/'tests.stderr.log').write_text(tests.stderr)
    if tests.returncode:
        write_json(out/'increment_validation.json', {'increment':'S01-I04', 'status':'FAIL', 'test_returncode':tests.returncode})
        return tests.returncode
    wind, lines = load_wind_bank(case), load_line_model(case)
    roads, canonical = load_road_model(case, include_connectors=True), load_road_model(case)
    connector = derive_road_connectors(case)
    for name, records in [('derived_connector_arcs.csv', connector.arc_rows), ('derived_connector_coupling.csv', connector.coupling_rows)]:
        if records:
            with (out/name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        else:
            with (case/'08_grid_road_coupling/L_A_gamma_blk__line_road_pairs.csv').open() as source:
                columns = next(csv.reader(source))
            with (out/name).open('w', newline='') as f:
                csv.writer(f).writerow(columns)
    write_json(out/'connector_diagnostics.json', dict(connector.diagnostics))
    assert tuple(roads.arc_ids) == tuple(range(1, 89)) and len(roads.corridor_ids) == 44
    assert tuple(canonical.arc_ids) == tuple(range(1, 81)) and len(canonical.corridor_ids) == 40
    times = np.arange(-8, 96)/4
    regimes, leads = ['R1', 'R2', 'R3'], [0., .5, 1., 2.]
    context = {'wind_input_hashes':dict(wind.input_hashes), 'line_input_hashes':dict(lines.input_hashes),
               'road_input_hashes':dict(roads.input_hashes), 'connector_input_hashes':dict(connector.input_hashes),
               'contract_sha256':hashlib.sha256((frozen_control_root() / 'ROAD_GENERATOR_CONTRACT_CN.md').read_bytes()).hexdigest(),
               'arc_ids':list(roads.arc_ids), 'corridor_ids':list(roads.corridor_ids),
               'points_xy_km':roads.points_xy_km.tolist(), 'regimes':regimes, 'leads_h':leads,
               'baseline_lead_h':1., 'canonical_feature_arc_ids':list(canonical.arc_ids)}
    payload = {key:[] for key in ('wind_probability', 'blockage_probability', 'cumulative_blockage_probability',
                                  'blocked', 'movable', 'travel_time_min')}
    rows, wind_payload, thresholds, lead_blocked, lead_movable = [], [], [], [], []
    reverse, shared = [], {}
    for variant in range(2):
        for base in wind.event_ids:
            for target in wind.event_ids:
                params = draw_parameters(wind.event_ids, target, base, variant, development=True)
                line_thresholds = lines.draw_thresholds(wind.event_ids, base, variant, development=True)
                road_thresholds = roads.draw_thresholds(wind.event_ids, base, variant, development=True)
                canonical_thresholds = canonical.draw_thresholds(wind.event_ids, base, variant, development=True)
                np.testing.assert_array_equal(road_thresholds.arc_uniforms[:80], canonical_thresholds.arc_uniforms)
                key = base, variant
                if key in shared:
                    np.testing.assert_array_equal(shared[key], road_thresholds.arc_uniforms)
                shared[key] = road_thresholds.arc_uniforms
                weather = [wind.sample(params, times, roads.points_xy_km, regime=r).vectors for r in ['R1', 'R2']]
                line_wind = [wind.sample(params, times, lines.points_xy_km, regime=r).vectors for r in ['R1', 'R2']]
                line_paths = [lines.replay(times, w, line_thresholds) for w in line_wind]
                paths = [roads.replay(times, weather[ri if ri < 2 else 0], road_thresholds,
                                      line_paths[ri if ri < 2 else 0], regime=regime)
                         for ri, regime in enumerate(regimes)]
                canonical_path = canonical.replay(times, weather[0][:, :80], canonical_thresholds, line_paths[0])
                for key in payload:
                    np.testing.assert_array_equal(getattr(paths[0], key)[:, :80], getattr(canonical_path, key))
                for path in paths:
                    assert np.all(np.diff(path.blocked.astype(int), axis=0) >= 0)
                    assert np.all(np.diff(path.cumulative_blockage_probability, axis=0) >= 0)
                    assert np.array_equal(np.isfinite(path.travel_time_min), path.movable)
                    assert np.all(path.travel_time_min[path.movable] > 0)
                    for key in payload:
                        value = getattr(path, key)
                        np.testing.assert_array_equal(value[:, 0::2], value[:, 1::2])
                for ri in (1, 2):
                    assert np.all(paths[ri].blocked >= paths[0].blocked)
                    assert np.all(paths[ri].movable <= paths[0].movable)
                    assert np.all(paths[ri].travel_time_min >= paths[0].travel_time_min - 1e-12)
                uncoupled = roads.replay(times, weather[0], road_thresholds, line_paths[0], no_coupling=True)
                np.testing.assert_allclose(uncoupled.blockage_probability, uncoupled.wind_probability, rtol=0, atol=2e-16)
                assert np.all(uncoupled.blocked <= paths[0].blocked)
                lead_paths = []
                for lead in leads:
                    lp = lines.replay(times, line_wind[0], line_thresholds, lead_h=lead)
                    rp = roads.replay(times, weather[0], road_thresholds, lp, lead_h=lead)
                    assert rp.times_h[0] == -lead
                    np.testing.assert_array_equal(rp.cumulative_blockage_probability[0], rp.blockage_probability[0])
                    lead_paths.append(rp)
                for shorter, longer in zip(lead_paths, lead_paths[1:]):
                    assert np.all(longer.blocked[longer.times_h >= 0] >= shorter.blocked[shorter.times_h >= 0])
                for key in payload:
                    payload[key].append([getattr(p, key) for p in paths])
                wind_payload.append(weather)
                thresholds.append(road_thresholds.arc_uniforms)
                lead_blocked.append([p.blocked[p.times_h >= 0] for p in lead_paths])
                lead_movable.append([p.movable[p.times_h >= 0] for p in lead_paths])
                row = {'target_event':target, 'base_weather_event':base, 'variant':variant+1,
                       'residual_donor_event':params.donor_event, 'wind_parameters_sha256':canonical_hash(params.as_dict()),
                       'realization_sha256':canonical_hash({'wind':params.as_dict(), 'road_streams':list(road_thresholds.stream_ids), 'context':context}),
                       'R1_canonical_final_blocked_corridors':int(paths[0].blocked[-1, :80:2].sum()),
                       'R1_canonical_movable_fraction':float(paths[0].movable[:, :80].mean()),
                       'R2_canonical_movable_fraction':float(paths[1].movable[:, :80].mean()),
                       'R3_canonical_movable_fraction':float(paths[2].movable[:, :80].mean())}
                rows.append(row)
                reverse.append((params, paths[0].blocked.copy(), paths[0].travel_time_min.copy()))
    for params, blocked, travel in reversed(reverse):
        lt = lines.draw_thresholds(wind.event_ids, params.base_event, params.variant_index, development=True)
        rt = roads.draw_thresholds(wind.event_ids, params.base_event, params.variant_index, development=True)
        lp = lines.replay(times, wind.sample(params, times, lines.points_xy_km).vectors, lt)
        rp = roads.replay(times, wind.sample(params, times, roads.points_xy_km).vectors, rt, lp)
        np.testing.assert_array_equal(blocked, rp.blocked)
        np.testing.assert_array_equal(travel, rp.travel_time_min)
    # Threshold distribution diagnostics; independent physical corridor IDs, not coordinate IDs.
    u = np.stack([roads.draw_thresholds(wind.event_ids, wind.event_ids[i % 8], 3000+i//8, development=True).corridor_uniforms
                  for i in range(4096)])
    levels = [.05, .2, .5, .8, .95]
    cdf_error = max(float(np.max(np.abs((u <= q).mean(axis=0)-q))) for q in levels)
    corr = np.corrcoef(u, rowvar=False)
    correlation_error = float(np.max(np.abs(corr-np.eye(44))))
    assert cdf_error < .04 and correlation_error < .09
    conditional_error = 0.
    for bank_paths in payload['cumulative_blockage_probability']:
        for path in bank_paths:
            for q in (path[0, ::2], path[-1, ::2]):
                conditional_error = max(conditional_error, float(np.max(np.abs((u <= q[None, :]).mean(axis=0)-q))))
    assert conditional_error < .04
    stats = {'n_threshold_vectors':4096, 'corridors':44, 'max_uniform_cdf_abs_error':cdf_error,
             'max_offdiagonal_threshold_correlation':correlation_error,
             'max_fixed_wind_line_conditional_marginal_abs_error':conditional_error,
             'tolerances':{'cdf':.04, 'correlation':.09, 'conditional_marginal':.04},
             'namespace':'development planning road_uniform, variants3000..3511 zero-based,8bases'}
    arrays = {key:np.asarray(values) for key, values in payload.items()}
    arrays.update({'wind_vectors_m_s':np.asarray(wind_payload), 'reference_times_h':times,
                   'baseline_replay_times_h':np.arange(-4, 96)/4, 'arc_ids':np.array(roads.arc_ids),
                   'regimes':np.array(regimes), 'leads_h':np.array(leads), 'arc_uniform_thresholds':np.stack(thresholds),
                   'R1_operating_slot_blocked_by_lead':np.asarray(lead_blocked),
                   'R1_operating_slot_movable_by_lead':np.asarray(lead_movable),
                   'statistical_corridor_thresholds':u, 'gamma_R1':roads.gamma})
    np.savez_compressed(out/'development_roads.npz', **arrays)
    write_json(out/'context.json', context)
    write_json(out/'road_threshold_statistics.json', stats)
    write_json(out/'development_road_ledger.json', rows)
    with (out/'development_road_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    hashes = {key:array_hash(value) for key, value in arrays.items()}
    write_json(out/'array_payload_hashes.json', hashes)
    source = frozen_source_paths()
    summary = {'increment':'S01-I04', 'status':'PASS_INCREMENT_AUTOMATED', 'full_S01_pass':False,
               'created_at_utc':datetime.now(timezone.utc).isoformat(), 'development_weather_realizations':128,
               'regime_road_paths':384, 'lead_checks':512, 'no_coupling_checks':128,
               'canonical_vs_extended_checks':128, 'canonical_directed_arcs':80, 'routing_directed_arcs':88,
               'canonical_physical_corridors':40, 'routing_physical_corridors':44,
               'threshold_statistics':stats, 'regimes':regimes, 'leads_h':leads,
               'reverse_order_scope':'128R1/lead1blocked and travel-time paths',
               'OOS_outcomes_generated':False, 'production_experiments_executed':False,
               'test_returncode':tests.returncode, 'test_command':command, 'context_sha256':canonical_hash(context),
               'source_hashes':{str(p.relative_to(runner)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
               'array_payload_hashes':hashes, 'runtime':{'host':platform.node(), 'python':platform.python_version(), 'numpy':np.__version__},
               'remaining_S01':['load generation', 'forecast routes/H', 'fixed-route replay', 'features/moment feasibility']}
    write_json(out/'increment_validation.json', summary)
    write_json(out/'artifact_sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
