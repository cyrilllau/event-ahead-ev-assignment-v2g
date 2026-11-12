#!/usr/bin/env python3
"""Development hourly load validation; no optimization or formal OOS access."""
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
from evcs_experiments.case_inputs import load_case_inputs
from evcs_experiments.random_streams import canonical_hash, event_order, stream
from evcs_experiments.load_scenarios import load_load_model, draw_load_errors, realize_load
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
    test = subprocess.run(command, cwd=case, capture_output=True, text=True, timeout=60)
    (out/'tests.stdout.log').write_text(test.stdout)
    (out/'tests.stderr.log').write_text(test.stderr)
    if test.returncode:
        write_json(out/'increment_validation.json', {'increment':'S01-I05', 'status':'FAIL', 'test_returncode':test.returncode})
        return test.returncode
    model = load_load_model(case)
    events = event_order(load_case_inputs(case).event_ids)
    contract = frozen_control_root() / 'LOAD_GENERATOR_CONTRACT_CN.md'
    context = {'load_input_hashes':dict(model.input_hashes), 'model_sha256':model.model_sha256,
               'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),
               'event_order':events, 'hours_h':model.hours_h.tolist(), 'bus_ids':list(model.bus_ids),
               'split_counts':{'development_planning':160, 'development_calibration':80},
               'target_regime_lead_invariance':'Targets, R1/R2/R3, lead and method reuse the addressed hourly load trajectory',
               'reactive_load':'v54-Q1 exact rated Q/P mapping; archived Q retained separately'}
    draws, samples, rows, target_rows = [], [], [], []
    for split, variants in [('planning', 20), ('calibration', 10)]:
        for variant in range(variants):
            for base_index, base_event in enumerate(events):
                draw = draw_load_errors(model, base_index, variant, split=split, development=True)
                sample = realize_load(model, draw)
                assert sample.p_kw.shape == sample.q_kvar.shape == (24, 33)
                factor = 1+draw.epsilon[:, None]
                np.testing.assert_array_equal(sample.p_kw, model.baseline_p_kw*factor)
                np.testing.assert_array_equal(sample.q_kvar, model.baseline_q_kvar*factor)
                np.testing.assert_allclose(sample.q_kvar, sample.p_kw*model.tan_phi_load[None, :], rtol=1e-14, atol=1e-12)
                assert np.all(sample.p_kw >= 0) and np.all(sample.q_kvar >= 0)
                assert np.all(sample.p_kw[:, 0] == 0) and np.all(sample.q_kvar[:, 0] == 0)
                expected_phi = float((model.baseline_p_kw.sum(axis=1)*(1+draw.epsilon)).sum()/model.baseline_p_kw.sum())
                assert abs(sample.phi_load-expected_phi) < 5e-15 and .91 <= sample.phi_load <= 1.09
                row = {'split':split, 'base_index':base_index, 'base_weather_event':base_event, 'variant_index':variant,
                       'stream_id':draw.stream_id, 'draw_sha256':draw.draw_sha256,
                       'realization_sha256':canonical_hash({'draw_sha256':draw.draw_sha256, 'context':context}),
                       'proposal_count':draw.proposal_count, 'rejection_count':draw.rejection_count,
                       'phi_load':sample.phi_load, 'p_energy_kwh':float(sample.p_kw.sum()),
                       'q_hour_sum_kvar_h':float(sample.q_kvar.sum())}
                rows.append(row)
                for target in events:
                    target_rows.append({'target_event':target, **row})
                draws.append(draw); samples.append(sample)
    for i in reversed(range(len(draws))):
        d = draws[i]
        repeat = draw_load_errors(model, d.base_index, d.variant_index, split=d.split, development=True)
        assert repeat.draw_sha256 == d.draw_sha256
        np.testing.assert_array_equal(repeat.epsilon, d.epsilon)
        regenerated = realize_load(model, repeat)
        np.testing.assert_array_equal(regenerated.p_kw, samples[i].p_kw)
        np.testing.assert_array_equal(regenerated.q_kvar, samples[i].q_kvar)
    # A separate development address range supplies diagnostics, never test-set outcomes.
    count = 8192
    statistical = [draw_load_errors(model, i % 8, 5000+i//8, development=True) for i in range(count)]
    epsilon = np.stack([d.epsilon for d in statistical])
    proposal_counts = np.array([d.proposal_count for d in statistical])
    assert np.all(np.abs(epsilon) <= .09) and not np.any(np.abs(epsilon) == .09)
    mean_error = float(np.max(np.abs(epsilon.mean(axis=0))))
    assert mean_error < .002
    # Independent AR(1) construction, separate documented validation RNG. This
    # conditions whole proposals on the same box, not sequential truncation.
    reference_rng = np.random.default_rng(np.random.SeedSequence(20260911, spawn_key=(9905, 1)))
    normals = reference_rng.standard_normal((40000, 24))
    proposal = np.empty_like(normals)
    proposal[:, 0] = .03*normals[:, 0]
    innovation_sigma = .03*np.sqrt(1-.8**2)
    for t in range(1, 24):
        proposal[:, t] = .8*proposal[:, t-1]+innovation_sigma*normals[:, t]
    keep = (np.abs(proposal) <= .09).all(axis=1)
    reference = proposal[keep]
    assert len(reference) > 35000
    actual_cov = np.cov(epsilon, rowvar=False, ddof=0)
    reference_cov = np.cov(reference, rowvar=False, ddof=0)
    covariance_error = float(np.max(np.abs(actual_cov-reference_cov)))
    assert covariance_error < .00007
    # Truncation changes the covariance, so no equality to latent Sigma is claimed.
    actual_lag1 = float(np.diag(np.corrcoef(epsilon, rowvar=False), 1).mean())
    reference_lag1 = float(np.diag(np.corrcoef(reference, rowvar=False), 1).mean())
    assert abs(actual_lag1-reference_lag1) < .035
    actual_acceptance = count/float(proposal_counts.sum())
    reference_acceptance = float(keep.mean())
    assert abs(actual_acceptance-reference_acceptance) < .025
    companion = []
    for i in range(count):
        base, variant = i % 8, 5000+i//8
        companion.append([
            stream('planning', base, variant, 'wind_amplitude', development=True).generator().random(),
            stream('planning', base, variant, 'line_gaussian', development=True).generator().standard_normal(),
            stream('planning', base, variant, 'road_uniform', development=True, asset_index=1).generator().random(),
        ])
    companion = np.array(companion)
    correlations = np.corrcoef(np.column_stack((epsilon, companion)), rowvar=False)[:24, 24:]
    independence_error = float(np.abs(correlations).max())
    assert independence_error < .06
    diagnostics = {'statistical_load_draws':count, 'reference_proposals':40000, 'reference_accepted':int(keep.sum()),
                   'max_abs_mean_error':mean_error, 'max_covariance_difference_from_independent_AR_reference':covariance_error,
                   'latent_covariance_is_not_the_truncated_covariance':True,
                   'actual_mean_lag1_correlation':actual_lag1, 'reference_mean_lag1_correlation':reference_lag1,
                   'actual_acceptance_rate':actual_acceptance, 'reference_acceptance_rate':reference_acceptance,
                   'max_abs_cross_mechanism_correlation':independence_error,
                   'companion_columns':['wind_amplitude_primitive_uniform', 'line_gaussian_first_standard_normal', 'road_corridor1_uniform'],
                   'tolerances':{'mean':.002, 'covariance_reference':.00007, 'lag1_reference':.035, 'acceptance_reference':.025, 'cross_mechanism':.06},
                   'load_namespace':'development/planning/mechanism9, variants5000..6023 zero-based,8bases',
                   'reference_namespace':'SeedSequence(root20260911,spawn_key=(9905,1)); diagnostics only'}
    arrays = {'p_kw':np.stack([p.p_kw for p in samples]), 'q_kvar':np.stack([p.q_kvar for p in samples]),
              'archived_q_kvar':np.stack([p.archived_q_kvar for p in samples]),
              'epsilon':np.stack([d.epsilon for d in draws]),
              'accepted_standard_innovations':np.stack([d.accepted_standard_innovation for d in draws]),
              'proposal_counts':np.array([d.proposal_count for d in draws]), 'phi_load':np.array([p.phi_load for p in samples]),
              'baseline_p_kw':model.baseline_p_kw, 'baseline_q_kvar':model.baseline_q_kvar,
              'archived_baseline_q_kvar':model.archived_baseline_q_kvar, 'tan_phi_load':model.tan_phi_load,
              'hours_h':model.hours_h, 'bus_ids':np.array(model.bus_ids), 'latent_covariance':model.covariance,
              'statistical_epsilon':epsilon, 'statistical_proposal_counts':proposal_counts,
              'independent_AR_reference_epsilon':reference, 'companion_primitives':companion}
    np.savez_compressed(out/'development_load.npz', **arrays)
    write_json(out/'context.json', context)
    write_json(out/'baseline_diagnostics.json', dict(model.diagnostics))
    write_json(out/'load_statistical_checks.json', diagnostics)
    for name, records in [('development_load_ledger.csv', rows), ('target_load_reuse_ledger.csv', target_rows)]:
        with (out/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_json(out/'development_load_ledger.json', rows)
    hashes = {key:array_hash(value) for key, value in arrays.items()}
    write_json(out/'array_payload_hashes.json', hashes)
    sources = frozen_source_paths()
    summary = {'increment':'S01-I05', 'status':'PASS_INCREMENT_AUTOMATED', 'full_S01_pass':False,
               'created_at_utc':datetime.now(timezone.utc).isoformat(), 'development_load_components':len(rows),
               'planning_components':160, 'calibration_components':80, 'target_reuse_ledger_rows':len(target_rows),
               'hours':24, 'buses':33, 'reverse_order_reproduction':True, 'statistical_checks':diagnostics,
               'formal_OOS_outcomes_generated':False, 'development_OOS_namespace_used_only_in_unit_tests':True,
               'production_experiments_executed':False,
               'test_returncode':test.returncode, 'test_command':command, 'context_sha256':canonical_hash(context),
               'source_hashes':{str(p.relative_to(runner)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               'array_payload_hashes':hashes, 'runtime':{'host':platform.node(), 'python':platform.python_version(), 'numpy':np.__version__},
               'remaining_S01':['forecast routes/H', 'fixed-route replay', 'features/moment feasibility', 'complete scenario assembly']}
    write_json(out/'increment_validation.json', summary)
    write_json(out/'artifact_sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
