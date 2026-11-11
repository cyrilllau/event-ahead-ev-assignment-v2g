"""Independent reconstruction of PCG64 load draws via stationary AR(1)."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np
import argparse
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_source_file, historical_path, acceptance_for, legacy_campaign_root, source_provenance)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, default=get_paths().data_root)
    parser.add_argument('--evidence', type=Path, required=True, help='Accepted I05 increment directory')
    parser.add_argument('--output', type=Path, required=True, help='New independent audit JSON file')
    args = parser.parse_args()
    c = args.case_root.resolve()
    out = historical_path(args.evidence).resolve()
    if args.output.exists():
        raise FileExistsError('Audit output must be a new file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path, own_acceptance = acceptance_for(out, 'I05')
    receipt = {'job_id': own_acceptance['job_id'], 'output': str(out)}
    job = historical_path(legacy_campaign_root() / 'jobs' / receipt['job_id'])
    job_result = json.loads((job / 'result.json').read_text())
    assert job_result['state'] == 'SUCCEEDED' and job_result['returncode'] == 0
    summary = json.loads((out/'increment_validation.json').read_text())
    assert summary['status'] == 'PASS_INCREMENT_AUTOMATED'
    assert summary['formal_OOS_outcomes_generated'] is False
    for name, digest in summary['source_hashes'].items():
        assert hashlib.sha256(frozen_source_file(name).read_bytes()).hexdigest() == digest, name
    for name, digest in json.loads((out/'artifact_sha256.json').read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest, name
    context = json.loads((out/'context.json').read_text())
    for name, digest in context['load_input_hashes'].items():
        assert hashlib.sha256((c/name).read_bytes()).hexdigest() == digest, name
    data = np.load(out/'development_load.npz')
    ledger = json.loads((out/'development_load_ledger.json').read_text())
    assert len(ledger) == 240
    assert data['p_kw'].shape == data['q_kvar'].shape == data['archived_q_kvar'].shape == (240, 24, 33)
    assert data['epsilon'].shape == data['accepted_standard_innovations'].shape == (240, 24)
    assert np.array_equal(data['hours_h'], np.arange(24)) and np.array_equal(data['bus_ids'], np.arange(1, 34))
    assert sum(row['split'] == 'planning' for row in ledger) == 160
    assert sum(row['split'] == 'calibration' for row in ledger) == 80
    def csv_rows(path):
        with path.open() as f:
            return list(csv.DictReader(f))
    source = {(int(row['t'])-1, int(row['n'])-1):row for row in csv_rows(c/'03_load/P_base_Q_base__N_T.csv')}
    assert len(source) == 792
    baseline_p = np.array([[float(source[t, n]['P_base_n_t_kW']) for n in range(33)] for t in range(24)])
    archived_q = np.array([[float(source[t, n]['Q_base_n_t_kvar']) for n in range(33)] for t in range(24)])
    spec = importlib.util.spec_from_file_location('independent_q1_adapter', c/'tools/reactive_support_model.py')
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    q1 = adapter.load_case(c)
    q_rows = {(row['t']-1, row['n']-1):row for row in q1['loads']}
    baseline_q = np.array([[q_rows[t, n]['Q_base_n_t_kvar'] for n in range(33)] for t in range(24)])
    ratio = np.array([q_rows[0, n]['tan_phi_load_n'] for n in range(33)])
    for name, expected in [('baseline_p_kw', baseline_p), ('baseline_q_kvar', baseline_q),
                           ('archived_baseline_q_kvar', archived_q), ('tan_phi_load', ratio)]:
        np.testing.assert_array_equal(data[name], expected)
    def ar_proposal(z):
        eps = np.empty(24)
        eps[0] = .03*z[0]
        for t in range(1, 24):
            eps[t] = .8*eps[t-1]+.018*z[t]
        return eps
    def reconstruct(base, variant, split):
        key = (1, 2, {'planning':1, 'calibration':2}[split], 0, base+1, variant+1, 9, 0, 0)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20260911, spawn_key=key)))
        for count in range(1, 100001):
            z = rng.standard_normal(24)
            eps = ar_proposal(z)
            if np.all(np.abs(eps) <= .09):
                return z, eps, count
        raise AssertionError('Independent rejection limit exceeded')
    max_error = {'epsilon':0., 'p_kw':0., 'q_kvar':0., 'archived_q_kvar':0., 'phi_load':0.}
    for i, row in enumerate(ledger):
        z, eps, proposals = reconstruct(row['base_index'], row['variant_index'], row['split'])
        np.testing.assert_array_equal(z, data['accepted_standard_innovations'][i])
        assert proposals == row['proposal_count'] == int(data['proposal_counts'][i])
        assert row['rejection_count'] == proposals-1
        error = float(np.abs(eps-data['epsilon'][i]).max())
        max_error['epsilon'] = max(max_error['epsilon'], error)
        assert error < 1e-15
        factor = 1+eps[:, None]
        for name, baseline in [('p_kw', baseline_p), ('q_kvar', baseline_q), ('archived_q_kvar', archived_q)]:
            error = float(np.abs(baseline*factor-data[name][i]).max())
            max_error[name] = max(max_error[name], error)
            assert error < 1e-10
        actual_phi = float(data['p_kw'][i].sum()/baseline_p.sum())
        error = abs(actual_phi-float(data['phi_load'][i]))
        max_error['phi_load'] = max(max_error['phi_load'], error)
        assert error < 1e-14 and .91 <= actual_phi <= 1.09
        np.testing.assert_allclose(data['q_kvar'][i], data['p_kw'][i]*ratio, rtol=0, atol=1e-12)
    assert np.all(data['p_kw'][:, :, 0] == 0) and np.all(data['q_kvar'][:, :, 0] == 0)
    assert np.any(data['phi_load'] > 1) and np.any(data['phi_load'] < 1)
    reuse = csv_rows(out/'target_load_reuse_ledger.csv')
    assert len(reuse) == 1920
    events = set(context['event_order'])
    for row in ledger:
        matches = [value for value in reuse if value['draw_sha256'] == row['draw_sha256']]
        assert len(matches) == 8 and {v['target_event'] for v in matches} == events
    epsilon = data['statistical_epsilon']
    assert epsilon.shape == (8192, 24)
    stats_error = 0.
    for i in range(8192):
        z, eps, proposals = reconstruct(i % 8, 5000+i//8, 'planning')
        assert proposals == int(data['statistical_proposal_counts'][i])
        error = float(np.abs(eps-epsilon[i]).max())
        stats_error = max(stats_error, error)
        assert error < 1e-15
    assert np.all(np.abs(epsilon) < .09)
    reference = data['independent_AR_reference_epsilon']
    assert reference.shape[1] == 24 and len(reference) > 35000
    cov_difference = float(np.max(np.abs(np.cov(epsilon, rowvar=False, ddof=0)-np.cov(reference, rowvar=False, ddof=0))))
    assert cov_difference < .00007
    cross = float(np.abs(np.corrcoef(np.column_stack((epsilon, data['companion_primitives'])), rowvar=False)[:24, 24:]).max())
    assert cross < .06
    payload_hash = hashlib.sha256((out/'development_load.npz').read_bytes()).hexdigest()
    result = {'status':'PASS_INDEPENDENT_ARRAY_AUDIT', 'job_id':receipt['job_id'], 'full_S01_pass':False,
              'source_and_artifact_hashes_verified':True, 'reconstructed_load_components':240,
              'reconstructed_statistical_draws':8192, 'verified_target_reuse_rows':1920,
              'method':'PCG64 direct frozen keys + stationary AR(1) with sigma0.03 and innovation0.018; whole-proposal rejection',
              'maximum_errors':max_error, 'max_statistical_epsilon_error':stats_error,
              'accepted_standard_innovations_and_rejection_counts_exact':True,
              'q1_model_Q_matches_authorized_adapter':True,
              'max_archived_q_rounding_adjustment_kvar':float(np.abs(baseline_q-archived_q).max()),
              'covariance_difference_from_independent_reference':cov_difference,
              'max_cross_mechanism_correlation':cross, 'raw_npz_sha256':payload_hash,
              'formal_OOS_outcomes_generated':False}
    result.update(source_provenance())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
