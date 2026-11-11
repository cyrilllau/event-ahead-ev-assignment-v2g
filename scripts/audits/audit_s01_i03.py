"""Independent persisted-array audit, using accepted I02 wind and SciPy formulas."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import argparse
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_source_file, historical_path, acceptance_for, legacy_campaign_root, source_provenance)
from scipy.special import expit, ndtr

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, default=get_paths().data_root)
    parser.add_argument('--evidence', type=Path, required=True, help='Accepted I03 increment directory')
    parser.add_argument('--output', type=Path, required=True, help='New independent audit JSON file')
    args = parser.parse_args()
    case = args.case_root.resolve()
    out = historical_path(args.evidence).resolve()
    if args.output.exists():
        raise FileExistsError('Audit output must be a new file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path, own_acceptance = acceptance_for(out, 'I03')
    receipt = {'job_id': own_acceptance['job_id'], 'output': str(out)}
    job = historical_path(legacy_campaign_root() / 'jobs' / receipt['job_id'])
    job_result = json.loads((job / 'result.json').read_text())
    assert job_result['state'] == 'SUCCEEDED' and job_result['returncode'] == 0
    summary = json.loads((out/'increment_validation.json').read_text())
    assert summary['status'] == 'PASS_INCREMENT_AUTOMATED'
    for name, digest in summary['source_hashes'].items():
        assert hashlib.sha256(frozen_source_file(name).read_bytes()).hexdigest() == digest, name
    for name, digest in json.loads((out/'artifact_sha256.json').read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest, name
    prior = historical_path(legacy_campaign_root()/'increments/S01-I02/20260905T052820Z')
    acceptance = json.loads((historical_path(legacy_campaign_root()/'acceptance/S01/I02-20260905T052820Z/acceptance.json')).read_text())
    assert hashlib.sha256((prior/'development_wind.npz').read_bytes()).hexdigest() == acceptance['raw_npz_sha256']
    wind = np.load(prior/'development_wind.npz')
    data = np.load(out/'development_lines.npz')
    ledger = json.loads((out/'development_line_ledger.json').read_text())
    wind_ledger = json.loads((prior/'development_wind_ledger.json').read_text())
    for line, weather in zip(ledger, wind_ledger):
        assert (line['target_event'], line['base_weather_event'], line['variant']) == (weather['target_event'], weather['base_weather_event'], weather['variant'])
        assert line['wind_parameters_sha256'] == weather['parameters_sha256']
    assert len(ledger) == len(wind_ledger) == 128
    assert data['failure_probability'].shape == (128, 2, 3, 100, 32)
    assert data['operating_outages'].shape == (128, 2, 3, 24, 32)
    assert data['R1_class4_operating_outages_by_lead'].shape == (128, 4, 24, 32)
    assert np.array_equal(data['line_ids'], np.arange(1, 33))
    assert np.array_equal(data['pole_classes'], [2, 4, 5])
    assert np.array_equal(data['regimes'], ['R1', 'R2'])
    assert np.array_equal(data['leads_h'], [0, .5, 1, 2])
    with (case/'02_power_network/loc_L__line_geography.csv').open() as f:
        geo = {int(r['ell']):r for r in csv.DictReader(f)}
    azimuth = np.array([float(geo[i]['phi_ell_deg_from_north']) for i in range(1, 33)])
    phi = np.deg2rad(90-azimuth)
    indexes = [list(wind['asset_ids']).index(f'ell:{i}') for i in range(1, 33)]
    times = wind['times_h']
    positions = times >= -1
    assert np.array_equal(times[positions], data['baseline_replay_times_h'])
    coefficients = [((-10.289, .0529, .0677), (-6.843, .0345, .0388)),
                    ((-9.313, .0494, .0776), (-6.037, .0219, .0471)),
                    ((-8.884, .0468, .0821), (-6.688, .0260, .0476))]
    maximum_error = 0.
    for ri, regime in enumerate(['R1', 'R2']):
        vectors = wind[f'wind_{regime}_vectors_m_s'][:, :, indexes, :]
        speed = np.hypot(vectors[..., 0], vectors[..., 1])
        theta = np.arctan2(vectors[..., 1], vectors[..., 0])
        perpendicular = speed*np.abs(np.sin(theta-phi))
        for ci, (pole, wire) in enumerate(coefficients):
            p = expit(pole[0]+pole[1]*speed+pole[2]*perpendicular)
            w = expit(wire[0]+wire[1]*speed+wire[2]*perpendicular)
            q = p+w-p*w
            difference = np.max(np.abs(q[:, positions]-data['failure_probability'][:, ri, ci]))
            maximum_error = max(maximum_error, float(difference))
            assert difference < 1e-14
            cumulative = np.maximum.accumulate(q[:, positions], axis=1)
            np.testing.assert_allclose(cumulative, data['cumulative_failure_probability'][:, ri, ci], rtol=0, atol=1e-14)
            failed = data['uniform_thresholds'][:, None, :] <= cumulative
            assert np.array_equal(failed, data['replay_outages'][:, ri, ci])
            operating = failed[:, 4:].reshape(128, 24, 4, 32).any(axis=2)
            assert np.array_equal(operating, data['operating_outages'][:, ri, ci])
            if ri == 0 and ci == 1:
                for li, lead in enumerate(data['leads_h']):
                    path = np.maximum.accumulate(q[:, times >= -lead], axis=1)
                    state = data['uniform_thresholds'][:, None, :] <= path
                    actual = state[:, int(lead*4):].reshape(128, 24, 4, 32).any(axis=2)
                    assert np.array_equal(actual, data['R1_class4_operating_outages_by_lead'][:, li])
    np.testing.assert_allclose(ndtr(data['gaussian_thresholds']), data['uniform_thresholds'], rtol=0, atol=2e-16)
    points = np.array([[float(geo[i]['loc_ell_x_km']), float(geo[i]['loc_ell_y_km'])] for i in range(1, 33)])
    expected_cov = .05*np.eye(32)+.95*np.exp(-np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)/4)
    np.testing.assert_allclose(data['covariance'], expected_cov, rtol=0, atol=1e-15)
    stats = json.loads((out/'copula_statistical_checks.json').read_text())
    cov_error = float(np.max(np.abs(np.cov(data['statistical_gaussians'], rowvar=False, ddof=0)-expected_cov)))
    assert abs(cov_error-stats['max_covariance_abs_error']) < 1e-14
    result = {
        'status':'PASS_INDEPENDENT_ARRAY_AUDIT', 'job_id':receipt['job_id'],
        'accepted_wind_npz_sha256':acceptance['raw_npz_sha256'],
        'line_npz_sha256':hashlib.sha256((out/'development_lines.npz').read_bytes()).hexdigest(),
        'source_and_artifact_hashes_verified':True, 'paths_recomputed_from_accepted_wind':768,
        'lead_operating_paths_recomputed':512, 'fragility_max_abs_error':maximum_error,
        'alternative_crosswind_formula':'speed*abs(sin(atan2(wy,wx)-(90-A)*pi/180))',
        'probability_library':'scipy.special.expit/ndtr; independent of production logaddexp/erfc',
        'covariance_max_empirical_error':cov_error,
        'failed_lines_final_by_regime_class':data['replay_outages'][:, :, :, -1, :].sum(axis=(0, 3)).tolist(),
        'full_S01_pass':False, 'OOS_outcomes_generated':False,
    }
    result.update(source_provenance())
    destination = args.output
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
