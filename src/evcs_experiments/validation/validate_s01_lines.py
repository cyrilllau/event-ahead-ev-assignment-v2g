#!/usr/bin/env python3
"""Development validation of persistent line outages; not full S01 acceptance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_paths, frozen_control_root, source_provenance, acceptance_for, historical_path, verify_scientific_kernel_identity)
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.wind_scenarios import load_wind_bank, draw_parameters
from evcs_experiments.line_outages import load_line_model
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
    test_command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(get_paths().project_root / 'tests'), '-p', 'test_*.py', '-v']
    test = subprocess.run(test_command, cwd=case, capture_output=True, text=True, timeout=60)
    (out/'tests.stdout.log').write_text(test.stdout)
    (out/'tests.stderr.log').write_text(test.stderr)
    if test.returncode:
        write_json(out/'increment_validation.json', {'increment':'S01-I03', 'status':'FAIL', 'test_returncode':test.returncode})
        return test.returncode
    wind = load_wind_bank(case)
    model = load_line_model(case)
    times = np.arange(-8, 96)/4
    leads = [0., .5, 1., 2.]
    classes = [2, 4, 5]
    regimes = ['R1', 'R2']
    assert model.line_ids == tuple(range(1, 33)) and model.tie_ids == tuple(range(33, 38))
    contract_path = frozen_control_root() / 'LINE_GENERATOR_CONTRACT_CN.md'
    context = {
        'wind_input_hashes':dict(wind.input_hashes), 'line_input_hashes':dict(model.input_hashes),
        'line_ids':model.line_ids, 'tie_ids_excluded':model.tie_ids,
        'points_xy_km':model.points_xy_km.tolist(), 'azimuth_deg_from_north':model.azimuth_deg_from_north.tolist(),
        'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        'regimes':regimes, 'pole_classes':classes, 'lead_hours':leads,
        'baseline_lead_h':1., 'reference_times_h':times.tolist(),
    }
    rows, probabilities, cumulative, outages, hourly, lead_hourly, uniforms, gaussians = [], [], [], [], [], [], [], []
    shared = {}
    reverse_entries = []
    for variant in range(2):
        for base in wind.event_ids:
            for target in wind.event_ids:
                p = draw_parameters(wind.event_ids, target, base, variant, development=True)
                resistance = model.draw_thresholds(wind.event_ids, base, variant, development=True)
                shared_key = (base, variant)
                if shared_key in shared:
                    assert np.array_equal(shared[shared_key], resistance.uniforms)
                shared[shared_key] = resistance.uniforms
                samples = [wind.sample(p, times, model.points_xy_km, regime=r) for r in regimes]
                assert np.array_equal(samples[0].vectors, wind.sample(p, times, model.points_xy_km, regime='R3').vectors)
                trajectories = [[model.replay(times, sample.vectors, resistance, lead_h=1., pole_class=c)
                                 for c in classes] for sample in samples]
                r1 = trajectories[0][1]
                for regime_trajectories in trajectories:
                    for replay in regime_trajectories:
                        assert np.all(np.diff(replay.cumulative_failure_probability, axis=0) >= 0)
                        assert np.all(np.diff(replay.outages.astype(int), axis=0) >= 0)
                        # Independently regroup exactly the operating slots, including late-hour failures.
                        expected_hourly = np.array([replay.outages[(replay.times_h >= h)&(replay.times_h < h+1)].any(axis=0)
                                                    for h in range(24)])
                        assert np.array_equal(replay.operating_outages, expected_hourly)
                for j in range(3):
                    assert np.all(trajectories[1][j].failure_probability + 1e-14 >= trajectories[0][j].failure_probability)
                    assert np.all(trajectories[1][j].outages >= trajectories[0][j].outages)
                # Curves can cross at extreme synthetic winds; actual bank must still be calculated, never relabeled.
                lead_replays = [model.replay(times, samples[0].vectors, resistance, lead_h=lead) for lead in leads]
                for lead, replay in zip(leads, lead_replays):
                    assert replay.times_h[0] == -lead
                    np.testing.assert_array_equal(replay.cumulative_failure_probability[0], replay.failure_probability[0])
                for before, after in zip(lead_replays, lead_replays[1:]):
                    assert np.all(after.operating_outages >= before.operating_outages)
                probabilities.append([[r.failure_probability for r in rs] for rs in trajectories])
                cumulative.append([[r.cumulative_failure_probability for r in rs] for rs in trajectories])
                outages.append([[r.outages for r in rs] for rs in trajectories])
                hourly.append([[r.operating_outages for r in rs] for rs in trajectories])
                lead_hourly.append([r.operating_outages for r in lead_replays])
                uniforms.append(resistance.uniforms)
                gaussians.append(resistance.correlated_normals)
                record = {'target_event':target, 'base_weather_event':base, 'variant':variant+1,
                          'residual_donor_event':p.donor_event, 'line_stream_id':resistance.stream_id,
                          'wind_parameters_sha256':canonical_hash(p.as_dict()),
                          'realization_sha256':canonical_hash({'wind':p.as_dict(), 'line_stream_id':resistance.stream_id, 'context':context}),
                          'R1_class4_failed_lines_final':int(r1.outages[-1].sum()),
                          'R2_class4_failed_lines_final':int(trajectories[1][1].outages[-1].sum()),
                          'R1_class4_mean_final_probability':float(r1.cumulative_failure_probability[-1].mean())}
                rows.append(record)
                reverse_entries.append((p, resistance.uniforms.copy(), r1.outages.copy()))
    for p, u, expected in reversed(reverse_entries):
        resistance = model.draw_thresholds(wind.event_ids, p.base_event, p.variant_index, development=True)
        assert np.array_equal(u, resistance.uniforms)
        w = wind.sample(p, times, model.points_xy_km).vectors
        assert np.array_equal(expected, model.replay(times, w, resistance).outages)
    # Persist empirical evidence for the copula using independent development scenario addresses.
    # This intentionally uses test-only variants outside the 2-variant development weather batch.
    draws = [model.draw_thresholds(wind.event_ids, wind.event_ids[i % 8], 1000+i//8, development=True)
             for i in range(8192)]
    z = np.stack([d.correlated_normals for d in draws])
    u = np.stack([d.uniforms for d in draws])
    empirical_cov = np.cov(z, rowvar=False, ddof=0)
    cov_error = float(np.max(np.abs(empirical_cov-model.covariance)))
    levels = np.array([.05, .2, .5, .8, .95])
    empirical_cdf = np.array([(u <= q).mean(axis=0) for q in levels])
    cdf_error = float(np.max(np.abs(empirical_cdf-levels[:, None])))
    upper = np.triu_indices(32, 1)
    nearest_index = int(np.argmax(model.covariance[upper]))
    farthest_index = int(np.argmin(model.covariance[upper]))
    joint_checks = []
    for k in (nearest_index, farthest_index):
        i, j = int(upper[0][k]), int(upper[1][k])
        rho = float(model.covariance[i, j])
        exact = .25+math.asin(rho)/(2*math.pi)
        observed = float(((u[:, i] <= .5)&(u[:, j] <= .5)).mean())
        assert abs(observed-exact) < .025
        joint_checks.append({'line_ids':[model.line_ids[i], model.line_ids[j]], 'latent_rho':rho,
                             'exact_P_both_thresholds_below_half':exact, 'empirical_probability':observed})
    assert cov_error < .075 and cdf_error < .025
    conditional_error = 0.
    # Fix each generated wind history; over independent damage resistance,
    # each line's failure marginal must equal its running maximum probability.
    for paths in cumulative:
        for regime_paths in paths:
            for q in (regime_paths[1][0], regime_paths[1][-1]):
                empirical_failure = (u <= q[None, :]).mean(axis=0)
                conditional_error = max(conditional_error, float(np.max(np.abs(empirical_failure-q))))
    assert conditional_error < .025
    statistical = {'n':len(draws), 'max_covariance_abs_error':cov_error, 'max_uniform_cdf_abs_error':cdf_error,
                   'max_fixed_weather_conditional_marginal_abs_error':conditional_error,
                   'fixed_weather_checks':'128 weather x 2 regimes x first/final replay slot x 32 lines, class4',
                   'cdf_levels':levels.tolist(), 'empirical_cdf':empirical_cdf.tolist(), 'joint_checks':joint_checks,
                   'tolerances':{'covariance':.075, 'cdf':.025, 'joint':.025},
                   'seed_namespace':'development/planning/line_gaussian; variants 1000..2023 zero-based'}
    arrays = {'failure_probability':np.asarray(probabilities), 'cumulative_failure_probability':np.asarray(cumulative),
              'replay_outages':np.asarray(outages), 'operating_outages':np.asarray(hourly),
              'R1_class4_operating_outages_by_lead':np.asarray(lead_hourly),
              'uniform_thresholds':np.stack(uniforms), 'gaussian_thresholds':np.stack(gaussians),
              'baseline_replay_times_h':np.arange(-4, 96)/4, 'line_ids':np.array(model.line_ids),
              'regimes':np.array(regimes), 'pole_classes':np.array(classes), 'leads_h':np.array(leads),
              'covariance':model.covariance, 'points_xy_km':model.points_xy_km,
              'statistical_gaussians':z, 'statistical_uniforms':u}
    np.savez_compressed(out/'development_lines.npz', **arrays)
    write_json(out/'development_line_ledger.json', rows)
    with (out/'development_line_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out/'context.json', context)
    write_json(out/'copula_statistical_checks.json', statistical)
    payload_hashes = {k:array_hash(v) for k, v in arrays.items()}
    write_json(out/'array_payload_hashes.json', payload_hashes)
    source = frozen_source_paths()
    summary = {'increment':'S01-I03', 'status':'PASS_INCREMENT_AUTOMATED', 'full_S01_pass':False,
               'created_at_utc':datetime.now(timezone.utc).isoformat(), 'development_weather_realizations':len(rows),
               'baseline_lead_line_paths':len(rows)*len(regimes)*len(classes),
               'additional_lead_checks':len(rows)*len(leads), 'baseline_replay_slots':100, 'operating_periods':24,
               'closed_lines':32, 'excluded_fixed_open_ties':5, 'shared_resistance_vectors':len(shared),
               'regimes':regimes, 'pole_classes':classes, 'leads_h':leads, 'reverse_order_reproduction':True,
               'reverse_order_scope':'all128 R1/class4/lead1 replay paths and thresholds',
               'assignment_epoch_reset_verified':True, 'R3_wind_identical_to_R1':True,
               'OOS_outcomes_generated':False, 'production_experiments_executed':False,
               'copula_statistics':{k:v for k,v in statistical.items() if k != 'empirical_cdf'},
               'test_returncode':test.returncode, 'test_command':test_command,
               'source_hashes':{str(p.relative_to(runner)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
               'array_payload_hashes':payload_hashes, 'context_sha256':canonical_hash(context),
               'runtime':{'host':platform.node(), 'python':platform.python_version(), 'numpy':np.__version__},
               'remaining_S01':['road/load generation', 'forecast routes/H', 'fixed-route replay', 'features/moment feasibility']}
    write_json(out/'increment_validation.json', summary)
    write_json(out/'artifact_sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
