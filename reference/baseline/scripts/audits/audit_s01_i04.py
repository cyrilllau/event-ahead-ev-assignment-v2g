"""Independent road-array audit from accepted I02 wind and I03 line states."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import numpy as np
import argparse
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_source_file, historical_path, acceptance_for, legacy_campaign_root, source_provenance)
from scipy.special import expit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, default=get_paths().data_root)
    parser.add_argument('--evidence', type=Path, required=True, help='Accepted I04 increment directory')
    parser.add_argument('--output', type=Path, required=True, help='New independent audit JSON file')
    args = parser.parse_args()
    case = args.case_root.resolve()
    out = historical_path(args.evidence).resolve()
    if args.output.exists():
        raise FileExistsError('Audit output must be a new file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path, own_acceptance = acceptance_for(out, 'I04')
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
    wind_path = historical_path(legacy_campaign_root()/'increments/S01-I02/20260905T052820Z')
    line_path = historical_path(legacy_campaign_root()/'increments/S01-I03/20260905T064212Z')
    for dirname, path, name in [('I02-20260905T052820Z', wind_path, 'development_wind.npz'),
                                ('I03-20260905T064212Z', line_path, 'development_lines.npz')]:
        accepted = json.loads(historical_path(legacy_campaign_root()/'acceptance/S01'/dirname/'acceptance.json').read_text())
        assert hashlib.sha256((path/name).read_bytes()).hexdigest() == accepted['raw_npz_sha256']
    wind = np.load(wind_path/'development_wind.npz')
    line = np.load(line_path/'development_lines.npz')
    data = np.load(out/'development_roads.npz')
    ledger = json.loads((out/'development_road_ledger.json').read_text())
    prior_ledger = json.loads((wind_path/'development_wind_ledger.json').read_text())
    assert len(ledger) == len(prior_ledger) == 128
    for a, b in zip(ledger, prior_ledger):
        assert (a['target_event'], a['base_weather_event'], a['variant']) == (b['target_event'], b['base_weather_event'], b['variant'])
        assert a['wind_parameters_sha256'] == b['parameters_sha256']
    assert data['blocked'].shape == (128, 3, 100, 88)
    assert data['wind_vectors_m_s'].shape == (128, 2, 104, 88, 2)
    assert data['R1_operating_slot_blocked_by_lead'].shape == (128, 4, 96, 88)
    assert np.array_equal(data['arc_ids'], np.arange(1, 89))
    assert np.array_equal(data['regimes'], ['R1', 'R2', 'R3'])
    times = wind['times_h']
    assert np.array_equal(times, data['reference_times_h'])
    positions = times >= -1
    def rows(relative):
        with (case/relative).open() as f:
            return list(csv.DictReader(f))
    arcs = rows('04_transportation/A__directed_road_arcs.csv')
    with (out/'derived_connector_arcs.csv').open() as f:
        connector = list(csv.DictReader(f))
    assert len(connector) == 8
    aux = {int(a['edge_id']):a for a in rows('04_transportation/transport_edges_with_depots__auxiliary.csv')}
    for a in connector:
        original = aux[int(a['corridor_id'])]
        for k, v in [('T0_a_min', 'freeflow_time_min'), ('D_a_km', 'length_km'), ('kappa_tt_a', 'kappa_tt'),
                     ('W_mov_type_a_m_s', 'movement_limit_m_s'), ('kappa_blk_type_a', 'kappa_block'),
                     ('alpha_R_type_a_0', 'alpha0_R'), ('alpha_R_type_a_1', 'alpha1_R')]:
            assert float(a[k]) == float(original[v])
    arcs += connector
    assert [int(a['a']) for a in arcs] == list(range(1, 89))
    def attr(name):
        return np.array([float(a[name]) for a in arcs])
    kblk, a0, a1, wmov, t0, ktt = [attr(k) for k in ['kappa_blk_type_a', 'alpha_R_type_a_0', 'alpha_R_type_a_1',
                                                   'W_mov_type_a_m_s', 'T0_a_min', 'kappa_tt_a']]
    gamma = np.zeros((32, 88))
    for pair in rows('08_grid_road_coupling/L_A_gamma_blk__line_road_pairs.csv'):
        i, a = int(pair['ell'])-1, int(pair['a'])-1
        assert i < 32 and a < 80
        gamma[i, a] = float(pair['gamma_blk_ell_a'])
    with (out/'derived_connector_coupling.csv').open() as f:
        assert list(csv.DictReader(f)) == []
    np.testing.assert_array_equal(gamma, data['gamma_R1'])
    # Independently confirm no connector crosses the coupling buffer, using all
    # archived line-polyline segments, local endpoints and vectorized projections.
    nodes = {int(a['road_node']):np.array([float(a['x_km']), float(a['y_km'])]) for a in rows('04_transportation/transport_nodes_with_depots__auxiliary.csv')}
    segments = []
    for g in rows('02_power_network/loc_L__line_geography.csv'):
        if int(g['ell']) > 32:
            continue
        numbers = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', g['geometry_ell_local_wkt'])]
        pts = np.array(numbers).reshape(-1, 2)
        segments.extend(zip(pts[:-1], pts[1:]))
    starts, ends = np.array(segments)[:, 0], np.array(segments)[:, 1]
    def point_segment(point, start, end):
        d = end-start
        alpha = np.clip(np.sum((point-start)*d, axis=-1)/np.sum(d*d, axis=-1), 0, 1)
        return np.linalg.norm(point-start-alpha[..., None]*d, axis=-1)
    distances = []
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    for identifier in range(41, 45):
        edge = aux[identifier]
        p, q = nodes[int(edge['from_node'])], nodes[int(edge['to_node'])]
        d, s = q-p, ends-starts
        det = cross(d, s)
        good = np.abs(det) > 1e-14
        ta = np.full(len(starts), np.inf); tb = np.full(len(starts), np.inf)
        ta[good] = cross(starts-p, s)[good]/det[good]
        tb[good] = cross(starts-p, d)[good]/det[good]
        assert not np.any(good & (ta >= 0)&(ta <= 1)&(tb >= 0)&(tb <= 1))
        distance = float(min(point_segment(p, starts, ends).min(), point_segment(q, starts, ends).min(),
                             point_segment(starts, p, q).min(), point_segment(ends, p, q).min()))
        assert distance > .25
        distances.append(distance)
    indexes = [list(wind['asset_ids']).index(f'a:{i}') for i in range(1, 81)]
    errors = {'wind_probability':0., 'blockage_probability':0., 'travel_time_min':0.}
    for ri, regime in enumerate(['R1', 'R2', 'R3']):
        wi = 1 if regime == 'R2' else 0
        vectors = data['wind_vectors_m_s'][:, wi]
        np.testing.assert_array_equal(vectors[:, :, :80], wind[f'wind_R{wi+1}_vectors_m_s'][:, :, indexes])
        selected = vectors[:, positions]
        speed = np.hypot(selected[..., 0], selected[..., 1])
        hazard = np.clip((speed-10)/15, 0, 1)
        wind_probability = kblk*expit(a0+a1*speed)
        effective_gamma = np.minimum(.45, gamma*(1.25 if regime == 'R3' else 1))
        failure = line['replay_outages'][:, wi, 1]
        survival = np.ones_like(wind_probability)
        for i in range(32):
            survival *= np.where(failure[:, :, i, None], 1-effective_gamma[i], 1)
        q = wind_probability + (1-wind_probability)*(1-survival)
        running = np.maximum.accumulate(q, axis=1)
        blocked = data['arc_uniform_thresholds'][:, None, :] <= running
        movable = (~blocked)&(speed <= wmov)
        travel = np.where(movable, t0*(1+ktt*(1.25 if regime == 'R3' else 1)*hazard), np.inf)
        for name, expected in [('wind_probability', wind_probability), ('blockage_probability', q),
                               ('cumulative_blockage_probability', running)]:
            error = float(np.max(np.abs(expected-data[name][:, ri])))
            assert error < 2e-14
            if name in errors:
                errors[name] = max(errors[name], error)
        assert np.array_equal(blocked, data['blocked'][:, ri])
        assert np.array_equal(movable, data['movable'][:, ri])
        actual = data['travel_time_min'][:, ri]
        assert np.array_equal(np.isinf(actual), np.isinf(travel))
        travel_error = float(np.max(np.abs(actual[movable]-travel[movable])))
        assert travel_error < 1e-12
        errors['travel_time_min'] = max(errors['travel_time_min'], travel_error)
        assert np.array_equal(blocked[:, :, ::2], blocked[:, :, 1::2])
    u = data['statistical_corridor_thresholds']
    assert u.shape == (4096, 44)
    assert np.max(np.abs(np.corrcoef(u, rowvar=False)-np.eye(44))) < .09
    result = {'status':'PASS_INDEPENDENT_ARRAY_AUDIT', 'job_id':receipt['job_id'], 'full_S01_pass':False,
              'paths_recomputed':384, 'directed_arcs_per_path':88, 'canonical_wind_matches_accepted_I02':True,
              'line_states_from_accepted_I03':True, 'source_and_artifact_hashes_verified':True,
              'raw_npz_sha256':hashlib.sha256((out/'development_roads.npz').read_bytes()).hexdigest(),
              'maximum_errors':errors, 'connector_minimum_line_distances_km':distances,
              'road_blocked_and_movable_states_exact':True,
              'canonical_mean_movable_fraction_by_regime':data['movable'][:, :, :, :80].mean(axis=(0, 2, 3)).tolist(),
              'OOS_outcomes_generated':False}
    result.update(source_provenance())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
