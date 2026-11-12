"""Hand-derived primal/dual witnesses; never invoke the probability solver."""
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from evcs_experiments.probability_check import check_probability_solution


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def fixture(q=(10., 0.), features=((0.,), (1.,)), lower=(.3,), upper=(.8,),
            p=(.7, .3), theta=10., alpha=(10.,), beta=(0.,)):
    q = np.array(q, dtype=float)
    phi = np.array(features, dtype=float)
    ids = tuple(f"scene{i}" for i in range(len(q)))
    feature_ids = tuple(f"feature{i}" for i in range(phi.shape[1]))
    ps = SimpleNamespace(support_ids=ids, feature_ids=feature_ids, features=phi,
                         lower=np.array(lower, dtype=float), upper=np.array(upper, dtype=float))
    ps.ambiguity_sha256 = digest({"support_ids": list(ids), "feature_ids": list(feature_ids),
        "features": phi.tolist(), "lower": ps.lower.tolist(), "upper": ps.upper.tolist(),
        "feature_orientation": "scenario_by_feature", "dual_convention": "alpha_lower_beta_upper"})
    p = None if p is None else np.array(p, dtype=float)
    alpha = None if alpha is None else np.array(alpha, dtype=float)
    beta = None if beta is None else np.array(beta, dtype=float)
    pobj = None if p is None else float(q @ p)
    dobj = None if theta is None or alpha is None or beta is None else float(theta-ps.lower@alpha+ps.upper@beta)
    result = SimpleNamespace(support_ids=ids, feature_ids=feature_ids,
        ambiguity_sha256=ps.ambiguity_sha256,
        q_sha256=digest({"support_ids": list(ids), "q": q.tolist()}),
        status="OPTIMAL", certified_optimal=True, probabilities=p, theta0=theta,
        alpha=alpha, beta=beta, primal_objective_usd=pobj, dual_objective_usd=dobj,
        lower_bound_usd=pobj, upper_bound_usd=dobj)
    return ps, q, result


def audit(values):
    ps, q, result = values
    return check_probability_solution(ps, q, result, q_support_ids=ps.support_ids)


class ProbabilityCheckTests(unittest.TestCase):
    def test_lower_moment_hand_dual_uses_alpha(self):
        report = audit(fixture())
        self.assertTrue(report.passed)
        self.assertAlmostEqual(report.objective_components["primal_usd"], 7.)
        self.assertAlmostEqual(report.objective_components["dual_usd"], 7.)
        self.assertEqual(report.objective_components["feature_means"], (.3,))
        self.assertEqual(report.separation["residuals_usd"], (0., 0.))

    def test_upper_moment_hand_dual_uses_beta(self):
        report = audit(fixture(q=(0., 10.), lower=(.2,), upper=(.6,), p=(.4, .6),
                              theta=0., alpha=(0.,), beta=(10.,)))
        self.assertTrue(report.passed)
        self.assertAlmostEqual(report.bounds["certified_lower_bound_usd"], 6.)
        self.assertAlmostEqual(report.bounds["certified_upper_bound_usd"], 6.)

    def test_free_negative_theta_and_empty_feature_simplex(self):
        report = audit(fixture(q=(-4., -2.), features=((), ()), lower=(), upper=(),
                              p=(0., 1.), theta=-2., alpha=(), beta=()))
        self.assertTrue(report.passed)
        self.assertEqual(report.objective_components["primal_usd"], -2.)
        self.assertEqual(report.residuals["moment_lower"]["constraint_count"], 0)
        json.dumps(report.as_dict(), allow_nan=False)
        with self.assertRaises(TypeError):
            report.bounds["certified_lower_bound_usd"] = 0.

    def test_fixed_distribution_hand_dual(self):
        report = audit(fixture(q=(2., 6.), features=((1., 0.), (0., 1.)),
                              lower=(.25, .75), upper=(.25, .75), p=(.25, .75),
                              theta=0., alpha=(0., 0.), beta=(2., 6.)))
        self.assertTrue(report.passed)
        self.assertEqual(report.objective_components["primal_usd"], 5.)

    def test_wrong_lower_multiplier_sign_is_rejected(self):
        values = fixture(alpha=(-10.,))
        report = audit(values)
        self.assertFalse(report.passed)
        self.assertFalse(report.residuals["alpha_nonnegative"]["passed"])
        self.assertFalse(report.gap_passed)
        swapped = audit(fixture(alpha=(0.,), beta=(10.,)))
        self.assertFalse(swapped.passed)
        self.assertEqual(swapped.objective_components["dual_usd"], 18.)

    def test_bad_probabilities_and_moment_rejected_without_clipping(self):
        values = fixture(p=(-.1, 1.1))
        report = audit(values)
        self.assertFalse(report.primal_feasible)
        self.assertFalse(report.residuals["probability_nonnegative"]["passed"])
        self.assertFalse(report.residuals["moment_upper"]["passed"])
        np.testing.assert_array_equal(values[2].probabilities, [-.1, 1.1])
        values = fixture(p=(.8, .3))
        self.assertFalse(audit(values).residuals["probability_normalization"]["passed"])

    def test_support_and_feature_permutations_rejected(self):
        ps, q, result = fixture()
        with self.assertRaisesRegex(ValueError, "q support axis"):
            check_probability_solution(ps, q, result, q_support_ids=ps.support_ids[::-1])
        result.support_ids = result.support_ids[::-1]
        with self.assertRaisesRegex(ValueError, "Result support"):
            check_probability_solution(ps, q, result, q_support_ids=ps.support_ids)
        values = fixture(features=((1., 0.), (0., 1.)), lower=(0., 0.), upper=(1., 1.),
                         alpha=(0., 0.), beta=(0., 0.))
        values[2].feature_ids = values[2].feature_ids[::-1]
        with self.assertRaisesRegex(ValueError, "feature axis"):
            audit(values)

    def test_raw_hashes_not_trusted_or_silently_reused(self):
        values = fixture()
        values[1][0] = 11.
        with self.assertRaisesRegex(ValueError, "q hash"):
            audit(values)
        values = fixture()
        values[0].features[0, 0] = .1
        with self.assertRaisesRegex(ValueError, "Probability-set hash"):
            audit(values)

    def test_infeasible_label_without_witness_is_not_farkas(self):
        values = fixture(p=None, theta=None, alpha=None, beta=None)
        values[2].status = "INFEASIBLE"
        report = audit(values)
        self.assertFalse(report.passed)
        self.assertFalse(report.infeasibility_certified)
        self.assertEqual(report.status, "NEED_WITNESSES")
        self.assertIsNone(report.bounds["certified_lower_bound_usd"])
        json.dumps(report.as_dict(), allow_nan=False)

    def test_reported_objective_and_bound_forgery_rejected(self):
        for field, value in (("primal_objective_usd", 8.), ("dual_objective_usd", 6.),
                             ("lower_bound_usd", 8.), ("upper_bound_usd", 6.)):
            with self.subTest(field=field):
                values = fixture()
                setattr(values[2], field, value)
                report = audit(values)
                self.assertFalse(report.passed)
                self.assertIn("reported_"+field, report.reasons)

    def test_separation_pass_does_not_replace_original_dual_tolerance(self):
        values = fixture(q=(1e8,), features=((),), lower=(), upper=(), p=(1.,),
                         theta=1e8-1e-5, alpha=(), beta=())
        report = audit(values)
        self.assertTrue(report.separation["passed"])
        self.assertTrue(report.gap_passed)
        self.assertLess(report.residuals["dual_full_support"]["max_normalized_violation"], 1e-8)
        self.assertGreater(report.residuals["dual_full_support"]["max_violation"], 1e-8)
        self.assertFalse(report.dual_feasible)
        self.assertFalse(report.passed)

    def test_feasible_primal_only_keeps_bound_but_not_optimal_certificate(self):
        values = fixture(theta=None, alpha=None, beta=None)
        values[2].status = "TIME_LIMIT"
        report = audit(values)
        self.assertTrue(report.primal_feasible)
        self.assertFalse(report.passed)
        self.assertEqual(report.bounds["certified_lower_bound_usd"], 7.)
        self.assertIsNone(report.bounds["certified_upper_bound_usd"])

    def test_nonfinite_and_wrong_shapes_rejected(self):
        for bad in ([.7, np.nan], [[.7, .3]], [.7]):
            with self.subTest(bad=bad):
                values = fixture()
                values[2].probabilities = bad
                with self.assertRaisesRegex(ValueError, "probabilities"):
                    audit(values)

    def test_centered_constant_cost_keeps_original_usd_certificate_gate(self):
        # These are hand witnesses in original units, independent of how a
        # solver obtained them. Centered cost is zero with scale=1 in all cases.
        for cost, phi, lower, upper, alpha, beta in (
                (260000., ((0.,), (1.,)), (.3,), (.8,), (0.,), (0.,)),
                (-260000., ((0.,), (1.,)), (.3,), (.8,), (0.,), (0.,)),
                (0., ((), ()), (), (), (), ())):
            with self.subTest(cost=cost):
                values = fixture(q=(cost, cost), features=phi, lower=lower, upper=upper,
                                 p=(.7, .3), theta=cost, alpha=alpha, beta=beta)
                values[2].certificate = {"q_offset_usd": cost, "q_scale_usd": 1.,
                                         "dual_feasible": True}
                report = audit(values)
                self.assertTrue(report.passed)
                self.assertEqual(report.residuals["dual_full_support"]["max_violation"], 0.)
                self.assertEqual(report.separation["scale_usd"], max(1., abs(cost)))
                self.assertEqual(report.separation["tolerance_usd"], 1e-4+1e-8*max(1., abs(cost)))
                # A tiny positive original-USD residual still fails even if
                # centered metadata claims success and separation permits it.
                values[2].theta0 -= 2e-8
                rejected = audit(values)
                self.assertFalse(rejected.passed)
                self.assertFalse(rejected.dual_feasible)
                self.assertTrue(rejected.separation["passed"])

    def test_centered_negative_cost_dual_requires_original_unit_recovery(self):
        # Original q=(-20,-10), offset=-20, scale=10. The centered upper-bound
        # dual is theta'=0, beta'=1; original theta=-20, beta=10 and value=-14.
        values = fixture(q=(-20., -10.), lower=(.2,), upper=(.6,), p=(.4, .6),
                         theta=-20., alpha=(0.,), beta=(10.,))
        values[2].certificate = {"q_offset_usd": -20., "q_scale_usd": 10.}
        report = audit(values)
        self.assertTrue(report.passed)
        self.assertEqual(report.objective_components["dual_usd"], -14.)
        self.assertEqual(report.separation["scale_usd"], 20.)
        original_hash = values[2].q_sha256
        values[2].theta0 = 0.
        values[2].beta = np.array([1.])
        self.assertFalse(audit(values).passed)
        self.assertEqual(values[2].q_sha256, original_hash)


if __name__ == "__main__":
    unittest.main()
