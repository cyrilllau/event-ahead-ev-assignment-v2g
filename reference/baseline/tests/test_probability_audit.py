"""Handwritten LP/KKT witnesses for the saved-evidence auditor; no solve/draw."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths

_spec = importlib.util.spec_from_file_location(
    'probability_saved_evidence_audit', get_paths().project_root / 'scripts/audits/audit_s03_i01.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def raw_record(z, objective, inequality, equality, lower):
    return {'status': 0, 'success': True, 'options': {'threads': 1},
            'raw_solution_scaled': list(z), 'raw_objective_scaled': objective,
            'ineqlin_marginals_scaled': list(inequality), 'eqlin_marginals_scaled': list(equality),
            'lower_marginals_scaled': list(lower), 'upper_marginals_scaled': [0.]*len(z)}


def save_identity(phi, lo, hi, q, p, theta, alpha, beta, raw_primal, raw_dual):
    phi, lo, hi, q = (np.asarray(v, dtype=float) for v in (phi, lo, hi, q))
    ids = [f'scene{i}' for i in range(len(q))]
    features = [f'f{i}' for i in range(phi.shape[1])]
    offset = float(min(q)); scale = max(1., float(max(abs(q-offset))))
    original_scale = max(1., float(max(abs(q))))
    saved = {'support_ids': ids, 'feature_ids': features,
             'q_sha256': digest({'support_ids': ids, 'q': q.tolist()}),
             'ambiguity_sha256': digest({'support_ids': ids, 'feature_ids': features, 'features': phi.tolist(),
                 'lower': lo.tolist(), 'upper': hi.tolist(), 'feature_orientation': 'scenario_by_feature',
                 'dual_convention': 'alpha_lower_beta_upper'}),
             'probabilities': list(p), 'theta0': theta, 'alpha': list(alpha), 'beta': list(beta),
             'primal_solver': raw_primal, 'dual_solver': raw_dual,
             'certificate': {'q_offset_usd': offset, 'q_scale_usd': scale,
                 'certification_scale_usd': original_scale, 'separation_scale_usd': original_scale,
                 'separation_tolerance_usd': 1e-4+1e-8*original_scale,
                 'objective_transform': {
                     'name': 'subtract_minimum_then_scale',
                     'scaled_cost': '(q - q_offset_usd) / q_scale_usd',
                     'primal_solver_objective': '-scaled_cost @ p', 'dual_solver_rhs': '-scaled_cost',
                     'primal_objective_recovery': 'q_offset_usd * sum(p) - q_scale_usd * raw_objective_scaled',
                     'dual_theta0_recovery': 'q_offset_usd + q_scale_usd * raw_theta0',
                     'dual_multiplier_recovery': 'q_scale_usd * raw_multiplier',
                     'raw_primal_witness': 'primal_solver.raw_solution_scaled',
                     'raw_dual_witness': 'dual_solver.raw_solution_scaled'}}}
    return phi, lo, hi, q, saved


def constant(value=260000., full=False):
    n = 160 if full else 3
    p = [1.] + [0.]*(n-1)
    phi = np.tile([0.,0.,0.,0.,1.], (n,1)) if full else np.empty((n,0))
    lo = np.array([0.,0.,0.,0.,.9]) if full else np.empty(0)
    hi = np.array([1.,1.,1.,1.,1.1]) if full else np.empty(0)
    f = len(lo)
    # Zero objective: any feasible p is optimal. The dual inequality marginals
    # are -p, and its lower-bound marginals are the two moment slacks.
    primal = raw_record(p, 0., [0.]*(2*f), [0.], [0.]*n)
    dual = raw_record([0.]*(1+2*f), 0., [-x for x in p], [],
                      [0.]+(phi.T@p-lo).tolist()+(hi-phi.T@p).tolist())
    return save_identity(phi, lo, hi, np.full(n,value), p, value, [0.]*f, [0.]*f, primal, dual)


def active_moment(lower=False, offset=100000.):
    # Two atoms at feature 0/1. Upper-active optimum p1=.6; lower-active p1=.3.
    if lower:
        primal = raw_record([.7,.3], -.7, [0.,-1.], [-1.], [0.,0.])
        dual = raw_record([1.,1.,0.], .7, [-.7,-.3], [], [0.,0.,.5])
        return save_identity([[0.],[1.]], [.3], [.8], [offset+10.,offset], [.7,.3], offset+10., [10.], [0.], primal, dual)
    primal = raw_record([.4,.6], -.6, [-1.,0.], [0.], [0.,0.])
    dual = raw_record([0.,0.,1.], .6, [-.4,-.6], [], [0.,.4,0.])
    return save_identity([[0.],[1.]], [.2], [.6], [offset,offset+10.], [.4,.6], offset, [0.], [10.], primal, dual)


class ProbabilityAuditTests(unittest.TestCase):
    def test_constant_260000_has_zero_raw_objectives_and_original_scale(self):
        with patch('scipy.optimize.linprog', side_effect=AssertionError('audit must not solve')), \
             patch('numpy.random.default_rng', side_effect=AssertionError('audit must not draw')):
            result = audit.audit_raw_probability_lp(*constant())
        self.assertEqual(result['q_offset_usd'], 260000.)
        self.assertEqual(result['q_scale_usd'], 1.)
        self.assertEqual(result['certification_scale_usd'], 260000.)
        self.assertEqual(result['raw_primal_LP']['primal_objective_scaled'], 0.)
        self.assertEqual(result['raw_dual_LP']['primal_objective_scaled'], 0.)
        self.assertEqual(result['primal_objective_recovered_usd'], 260000.)
        self.assertEqual(result['dual_objective_recovered_usd'], 260000.)

    def test_free_negative_theta_and_zero_feature_simplex(self):
        result = audit.audit_raw_probability_lp(*constant(-4.))
        self.assertEqual(result['q_offset_usd'], -4.)
        self.assertEqual(result['dual_objective_recovered_usd'], -4.)

    def test_affine_upper_moment_beta_and_translation(self):
        for offset in (100000., -8.):
            with self.subTest(offset=offset):
                result = audit.audit_raw_probability_lp(*active_moment(offset=offset))
                self.assertEqual(result['q_scale_usd'], 10.)
                self.assertAlmostEqual(result['primal_objective_recovered_usd'], offset+6.)
                self.assertAlmostEqual(result['dual_objective_recovered_usd'], offset+6.)

    def test_affine_lower_moment_alpha_and_theta_offset(self):
        result = audit.audit_raw_probability_lp(*active_moment(lower=True))
        self.assertAlmostEqual(result['primal_objective_recovered_usd'], 100007.)
        self.assertAlmostEqual(result['dual_objective_recovered_usd'], 100007.)

    def test_offset_tampering_is_rejected_even_with_valid_p_and_original_dual(self):
        values = active_moment(); values[-1]['certificate']['q_offset_usd'] += 1.
        with self.assertRaisesRegex(ValueError, 'q_offset_usd'):
            audit.audit_raw_probability_lp(*values)

    def test_optimization_and_certification_scales_cannot_be_swapped(self):
        for key in ('q_scale_usd', 'certification_scale_usd', 'separation_scale_usd'):
            with self.subTest(key=key):
                values = constant(); values[-1]['certificate'][key] = 2.
                with self.assertRaisesRegex(ValueError, key):
                    audit.audit_raw_probability_lp(*values)
        values = constant(); values[-1]['certificate']['separation_tolerance_usd'] = .00010001
        with self.assertRaisesRegex(ValueError, 'separation tolerance'):
            audit.audit_raw_probability_lp(*values)

    def test_raw_primal_is_not_replaced_by_another_feasible_distribution(self):
        values = constant(); values[-1]['primal_solver']['raw_solution_scaled'] = [.5,.5,0.]
        with self.assertRaisesRegex(ValueError, 'Published probabilities differ'):
            audit.audit_raw_probability_lp(*values)

    def test_theta_and_both_multiplier_restorations_are_checked(self):
        for index, label in ((0,'theta'), (1,'Alpha'), (2,'Beta')):
            with self.subTest(index=index):
                values = active_moment(); values[-1]['dual_solver']['raw_solution_scaled'][index] += .1
                with self.assertRaisesRegex(ValueError, label):
                    audit.audit_raw_probability_lp(*values)

    def test_raw_marginal_and_scaled_objective_forgery_rejected(self):
        values = active_moment(); values[-1]['primal_solver']['ineqlin_marginals_scaled'][0] += .1
        with self.assertRaisesRegex(ValueError, 'numeric certificate'):
            audit.audit_raw_probability_lp(*values)
        values = active_moment(); values[-1]['dual_solver']['raw_objective_scaled'] += .01
        with self.assertRaisesRegex(ValueError, 'Raw LP objective'):
            audit.audit_raw_probability_lp(*values)

    def test_original_q_hash_is_not_replaced_by_centered_q_hash(self):
        values = active_moment(); saved = values[-1]
        saved['q_sha256'] = digest({'support_ids': saved['support_ids'], 'q': [0.,1.]})
        with self.assertRaisesRegex(ValueError, 'Original q identity'):
            audit.audit_raw_probability_lp(*values)

    def test_primal_recovery_keeps_the_raw_normalization_residual(self):
        values = constant(); saved = values[-1]
        saved['probabilities'][0] += 1e-9
        saved['primal_solver']['raw_solution_scaled'][0] += 1e-9
        result = audit.audit_raw_probability_lp(*values)
        self.assertAlmostEqual(result['primal_objective_recovered_usd'], 260000.*(1.+1e-9), delta=1e-8)
        self.assertGreater(result['primal_objective_recovered_usd']-260000., .0002)
        self.assertEqual(saved['probabilities'][0], 1.+1e-9)

    def test_complete_constant_case_and_frozen_original_usd_gate(self):
        phi, lo, hi, q, saved = constant(full=True)
        saved['feature_ids'] = list(audit.FEATURES)
        saved['ambiguity_sha256'] = digest({'support_ids': saved['support_ids'], 'feature_ids': saved['feature_ids'],
            'features': phi.tolist(), 'lower': lo.tolist(), 'upper': hi.tolist(),
            'feature_orientation': 'scenario_by_feature', 'dual_convention': 'alpha_lower_beta_upper'})
        saved.update(status='OPTIMAL', certified_optimal=True, primal_objective_usd=260000., dual_objective_usd=260000.,
                     lower_bound_usd=260000., upper_bound_usd=260000.)
        saved['certificate'].update(feature_orientation='scenario_by_feature', probabilities_clipped_or_renormalized=False,
                                   moment_bounds_repaired=False, raw_RMP_dual_replaced=False, phase_I_initialization_in_scope=False)
        record = {'target': 'toy', 'rho': .5, 'vector': 'constant_q', 'relative_path': 'toy',
                  'ambiguity_sha256': saved['ambiguity_sha256'], 'q_sha256': saved['q_sha256']}
        checked = {'passed': True, 'primal_feasible': True, 'dual_feasible': True, 'gap_passed': True,
                   'objective_components': {'primal_usd': 260000., 'dual_usd': 260000., 'absolute_gap_usd': 0.},
                   'separation': {'support_ids': saved['support_ids'], 'passed': True, 'tolerance_usd': .0027,
                                  'max_positive_residual_usd': 0., 'residuals_usd': [0.]*160}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.savez(folder/'inputs.npz', features=phi, lower=lo, upper=hi, q=q,
                     support_ids=np.array(saved['support_ids']), feature_ids=np.array(saved['feature_ids']))
            (folder/'solution.json').write_text(json.dumps(saved))
            (folder/'physical_probability_check.json').write_text(json.dumps(checked))
            result = audit.audit_case(folder, record, (phi,lo,hi,saved['support_ids']))
            self.assertEqual(result['raw_residuals']['dual_full_support_usd'], 0.)
            self.assertAlmostEqual(result['separation_tolerance_usd'], .0027, delta=1e-15)
            # 2e-8 USD would pass the separate .0027 separation tolerance, but
            # remains invalid against the unchanged 1e-8 USD feasibility gate.
            bad = copy.deepcopy(saved); bad['theta0'] -= 2e-8
            (folder/'solution.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Original-unit or normalized'):
                audit.audit_case(folder, record, (phi,lo,hi,saved['support_ids']))


if __name__ == '__main__':
    unittest.main()
