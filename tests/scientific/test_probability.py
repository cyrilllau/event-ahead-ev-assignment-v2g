"""Hand-solved probability LPs, independent dual signs and immutable identities."""
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ev_assignment.methods.probability import ProbabilitySet, solve_worst_distribution
from ev_assignment.domain.random_streams import canonical_hash


def probability_set(phi,lower,upper,ids=None,feature_ids=None):
    a=np.asarray(phi,dtype=float)
    return ProbabilitySet(tuple(ids or [f's{i}' for i in range(len(a))]),
                          tuple(feature_ids if feature_ids is not None else [f'f{i}' for i in range(a.shape[1])]),
                          a,np.asarray(lower,dtype=float),np.asarray(upper,dtype=float))


class ProbabilityTests(unittest.TestCase):
    def solve(self,p,q):
        r=solve_worst_distribution(p,q,q_support_ids=p.support_ids,time_limit_seconds=5)
        self.assertEqual(r.status,'OPTIMAL',r.as_dict())
        self.assertTrue(r.certified_optimal)
        self.assertAlmostEqual(r.primal_objective_usd,r.dual_objective_usd,places=8)
        self.assertTrue(r.certificate['primal_feasible']);self.assertTrue(r.certificate['dual_feasible'])
        self.assertTrue(r.certificate['independent_primal_and_dual_solved'])
        self.assertLessEqual(abs(sum(r.probabilities)-1),1e-8)
        self.assertGreaterEqual(float(min(r.probabilities)),-1e-8)
        mean=p.features.T@r.probabilities
        self.assertTrue(np.all(mean>=p.lower-1e-8));self.assertTrue(np.all(mean<=p.upper+1e-8))
        lhs=r.theta0+p.features@(r.beta-r.alpha)
        self.assertTrue(np.all(lhs>=np.asarray(q)-1e-8))
        self.assertAlmostEqual(r.theta0-p.lower@r.alpha+p.upper@r.beta,r.primal_objective_usd,places=8)
        json.dumps(r.as_dict(),allow_nan=False)
        return r

    def test_saved_160q_failure_is_certified_without_relaxing_original_usd_gate(self):
        from evcs_experiments.config import get_paths
        from ev_assignment.validation.probability import check_probability_solution
        fixture=json.loads((get_paths().project_root/'tests/fixtures/probability_michael_160q_numerical.json').read_text())
        row=fixture['probability_set'];q=np.array(fixture['q_usd']);old=fixture['original_failed_result']
        a=ProbabilitySet(tuple(row['support_ids']),tuple(row['feature_ids']),row['features'],row['lower'],row['upper'])
        self.assertEqual(a.features.shape,(160,5));self.assertTrue(fixture['checkpoint_q_matched'])
        self.assertEqual(a.ambiguity_sha256,row['ambiguity_sha256'])
        self.assertEqual(old['q_sha256'],canonical_hash({'support_ids':list(a.support_ids),'q':q.tolist()}))
        self.assertEqual(old['primal_solver']['status'],0);self.assertEqual(old['dual_solver']['status'],0)
        old_residual=q-(old['theta0']+a.features@(np.array(old['beta'])-np.array(old['alpha'])))
        self.assertGreater(float(old_residual.max()),1e-8)
        self.assertFalse(old['certified_optimal'])  # Preserve the original rejection.
        import ev_assignment.methods.probability as module
        with patch.object(module,'_solver',wraps=module._solver) as calls:
            result=solve_worst_distribution(a,q,q_support_ids=a.support_ids,time_limit_seconds=5.)
        self.assertEqual(calls.call_count,2)
        self.assertTrue(result.certified_optimal,result.as_dict())
        self.assertEqual(result.certificate['feasibility_tolerance'],1e-8)
        self.assertEqual(result.certificate['dual_original_feasibility_tolerance_usd'],1e-8)
        for record in (result.primal_solver,result.dual_solver):
            self.assertEqual(record['options']['primal_feasibility_tolerance'],1e-10)
            self.assertEqual(record['options']['dual_feasibility_tolerance'],1e-10)
        self.assertLessEqual(float((q-(result.theta0+a.features@(result.beta-result.alpha))).max()),1e-8)
        self.assertEqual(result.q_sha256,old['q_sha256']);self.assertEqual(result.ambiguity_sha256,old['ambiguity_sha256'])
        report=check_probability_solution(a,q,result,q_support_ids=a.support_ids)
        self.assertTrue(report.passed,report.as_dict())
        self.assertFalse(result.certificate['probabilities_clipped_or_renormalized'])
        self.assertFalse(result.certificate['moment_bounds_repaired'])
        self.assertFalse(result.certificate['raw_RMP_dual_replaced'])

    def test_singleton_and_constant_load_above_one(self):
        p=probability_set([[.4,1.04]],[.3,1.04],[.5,1.04])
        r=self.solve(p,[7.]);np.testing.assert_array_equal(r.probabilities,[1.])
        self.assertEqual(r.primal_objective_usd,7.)
        self.assertEqual(p.features[0,1],1.04)

    def test_zero_feature_simplex_and_free_negative_theta(self):
        p=probability_set(np.empty((3,0)),[],[])
        r=self.solve(p,[-4.,-2.,-9.])
        self.assertEqual(r.primal_objective_usd,-2.)
        self.assertEqual(r.theta0,-2.)
        self.assertEqual(r.alpha.shape,(0,));np.testing.assert_array_equal(r.probabilities,[0.,1.,0.])

    def test_upper_moment_dual_beta_sign(self):
        p=probability_set([[0.],[1.]],[.25],[.5])
        r=self.solve(p,[0.,10.])
        np.testing.assert_allclose(r.probabilities,[.5,.5],atol=1e-10)
        self.assertEqual(r.primal_objective_usd,5.)
        np.testing.assert_allclose(r.beta,[10.],atol=1e-10);np.testing.assert_allclose(r.alpha,[0.],atol=1e-10)

    def test_lower_moment_dual_alpha_sign(self):
        p=probability_set([[0.],[1.]],[.25],[.5])
        r=self.solve(p,[10.,0.])
        np.testing.assert_allclose(r.probabilities,[.75,.25],atol=1e-10)
        self.assertEqual(r.primal_objective_usd,7.5)
        self.assertAlmostEqual(r.theta0,10.)
        np.testing.assert_allclose(r.alpha,[10.],atol=1e-10);np.testing.assert_allclose(r.beta,[0.],atol=1e-10)

    def test_two_coupled_moments_golden(self):
        p=probability_set([[0.,0.],[1.,0.],[0.,1.]],[.2,.3],[.4,.5])
        r=self.solve(p,[1.,4.,7.])
        np.testing.assert_allclose(r.probabilities,[.1,.4,.5],atol=1e-10)
        self.assertAlmostEqual(r.primal_objective_usd,5.2)
        self.assertAlmostEqual(r.theta0,1.);np.testing.assert_allclose(r.beta,[3.,6.],atol=1e-10)

    def test_fixed_moments_define_fixed_probability(self):
        p=probability_set([[0.,0.],[1.,0.],[0.,1.]],[.2,.3],[.2,.3])
        r=self.solve(p,[1.,4.,7.])
        np.testing.assert_allclose(r.probabilities,[.5,.2,.3],atol=1e-10)
        self.assertAlmostEqual(r.primal_objective_usd,3.4)

    def test_constant_mismatch_is_not_tolerance_repaired(self):
        for lower in [1e-12,.1]:
            p=probability_set([[0.],[0.]],[lower],[.2])
            r=solve_worst_distribution(p,[1.,2.],q_support_ids=p.support_ids)
            self.assertEqual(r.status,'INFEASIBLE');self.assertFalse(r.certified_optimal)
            self.assertIsNone(r.probabilities);self.assertIsNone(r.lower_bound_usd)
            self.assertEqual(r.certificate['analytic_infeasibility']['kind'],'constant_feature_outside_moment_interval')
            self.assertFalse(r.certificate['moment_bounds_repaired'])

    def test_empty_feature_convex_hull_reports_solver_infeasible(self):
        p=probability_set([[0.,0.],[1.,1.]],[.8,0.],[1.,.2])
        r=solve_worst_distribution(p,[1.,2.],q_support_ids=p.support_ids)
        self.assertEqual(r.status,'INFEASIBLE');self.assertFalse(r.certified_optimal)
        self.assertIsNone(r.probabilities)
        self.assertFalse(r.primal_solver['infeasibility_certificate_available'])

    def test_duplicate_feature_atoms_keep_distinct_ids_and_values(self):
        p=probability_set([[0.],[0.],[1.]],[0.],[1.],ids=['same_weather_a','same_weather_b','third'])
        r=self.solve(p,[2.,5.,3.])
        self.assertEqual(r.primal_objective_usd,5.)
        np.testing.assert_array_equal(r.probabilities,[0.,1.,0.])
        self.assertEqual(r.support_ids,p.support_ids)

    def test_permutation_alignment_and_wrong_q_ids(self):
        p=probability_set([[0.,0.],[1.,0.],[0.,1.]],[.2,.3],[.4,.5])
        original=self.solve(p,[1.,4.,7.]);order=[2,0,1]
        permuted=probability_set(p.features[order],p.lower,p.upper,ids=[p.support_ids[i] for i in order])
        result=self.solve(permuted,[7.,1.,4.])
        self.assertAlmostEqual(result.primal_objective_usd,original.primal_objective_usd)
        np.testing.assert_allclose(result.probabilities,original.probabilities[order],atol=1e-10)
        with self.assertRaisesRegex(ValueError,'order'):
            solve_worst_distribution(permuted,[1.,4.,7.],q_support_ids=p.support_ids)

    def test_translation_scaling_and_tied_costs(self):
        p=probability_set([[0.],[1.]],[.25],[.5])
        base=self.solve(p,[0.,10.])
        shifted=self.solve(p,[-20.,-10.]);self.assertAlmostEqual(shifted.primal_objective_usd,base.primal_objective_usd-20)
        large=self.solve(p,[0.,1e6]);self.assertAlmostEqual(large.primal_objective_usd,5e5)
        tied=self.solve(p,[3.,3.]);self.assertAlmostEqual(tied.primal_objective_usd,3.)

    def test_large_constant_cost_is_solved_as_two_zero_cost_lps(self):
        import ev_assignment.methods.probability as module
        actual=module._solver
        grid=np.linspace(0.,1.,160)
        phi=np.column_stack((grid,grid**2,np.sin(grid),np.cos(grid),1.04+.01*grid))
        mean=phi.mean(axis=0);p=probability_set(phi,mean-.01,mean+.01)
        q=np.full(160,260000.)
        with patch.object(module,'_solver',wraps=actual) as solve:
            r=self.solve(p,q)
        self.assertEqual(solve.call_count,2)
        np.testing.assert_array_equal(solve.call_args_list[0].args[0],np.zeros(160))
        np.testing.assert_array_equal(solve.call_args_list[1].args[2],np.zeros(160))
        np.testing.assert_array_equal(r.dual_solver['raw_solution_scaled'],np.zeros(11))
        self.assertEqual(r.primal_solver['raw_objective_scaled'],0.)
        self.assertEqual(r.dual_solver['raw_objective_scaled'],0.)
        self.assertEqual(r.theta0,260000.)
        self.assertEqual(r.certificate['dual']['max_support_violation_usd'],0.)
        self.assertEqual(r.certificate['q_offset_usd'],260000.)
        self.assertEqual(r.certificate['q_scale_usd'],1.)
        self.assertEqual(r.certificate['certification_scale_usd'],260000.)
        self.assertEqual(r.certificate['separation_scale_usd'],260000.)
        self.assertEqual(r.q_sha256,canonical_hash({'support_ids':list(p.support_ids),'q':q.tolist()}))

    def test_centered_raw_witnesses_replay_under_additive_translation(self):
        p=probability_set([[0.,0.],[1.,0.],[0.,1.]],[.2,.3],[.4,.5])
        q=np.array([1.,4.,7.]);base=self.solve(p,q)
        for translation in (-260000.,260000.):
            with self.subTest(translation=translation):
                original=q+translation;r=self.solve(p,original)
                offset=r.certificate['q_offset_usd'];scale=r.certificate['q_scale_usd']
                self.assertEqual(offset,float(original.min()));self.assertEqual(scale,6.)
                self.assertEqual(r.certificate['certification_scale_usd'],float(np.abs(original).max()))
                np.testing.assert_array_equal(r.probabilities,base.probabilities)
                np.testing.assert_array_equal(r.dual_solver['raw_solution_scaled'],base.dual_solver['raw_solution_scaled'])
                z=np.asarray(r.dual_solver['raw_solution_scaled'])
                self.assertEqual(r.theta0,offset+scale*z[0])
                np.testing.assert_array_equal(r.alpha,scale*z[1:3])
                np.testing.assert_array_equal(r.beta,scale*z[3:])
                np.testing.assert_array_equal(r.alpha,base.alpha)
                np.testing.assert_array_equal(r.beta,base.beta)
                self.assertAlmostEqual(r.theta0,base.theta0+translation,places=8)
                self.assertAlmostEqual(r.primal_objective_usd,base.primal_objective_usd+translation,places=8)
                self.assertAlmostEqual(r.primal_objective_usd,
                    offset*sum(r.probabilities)-scale*r.primal_solver['raw_objective_scaled'],places=8)
                self.assertAlmostEqual(r.dual_objective_usd,
                    offset+scale*r.dual_solver['raw_objective_scaled'],places=8)
                self.assertNotEqual(r.q_sha256,base.q_sha256)
                self.assertEqual(r.ambiguity_sha256,base.ambiguity_sha256)

    def test_simplex_dro_and_empirical_ordering_requires_membership(self):
        phi=[[0.,0.],[1.,0.],[0.,1.]];q=[1.,4.,7.]
        wide=self.solve(probability_set(phi,[.2,.3],[.4,.5]),q)
        narrow=self.solve(probability_set(phi,[.3,.4],[.3,.4]),q)
        self.assertLessEqual(narrow.primal_objective_usd,wide.primal_objective_usd)
        self.assertLessEqual(wide.primal_objective_usd,max(q))
        self.assertLessEqual(np.mean(q),wide.primal_objective_usd)
        excludes_empirical=self.solve(probability_set(phi,[.9,0.],[.9,0.]),q)
        self.assertLess(excludes_empirical.primal_objective_usd,np.mean(q))

    def test_immutable_arrays_and_canonical_identity(self):
        data=np.array([[0.],[1.]]);p=probability_set(data,[.25],[.5])
        data[0,0]=100.;self.assertEqual(p.features[0,0],0.)
        expected=canonical_hash({'support_ids':list(p.support_ids),'feature_ids':list(p.feature_ids),
            'features':p.features.tolist(),'lower':p.lower.tolist(),'upper':p.upper.tolist(),
            'feature_orientation':'scenario_by_feature','dual_convention':'alpha_lower_beta_upper'})
        self.assertEqual(p.ambiguity_sha256,expected)
        with self.assertRaises(ValueError):p.features.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):p.lower=np.zeros(1)
        r=self.solve(p,[0.,10.])
        with self.assertRaises(ValueError):r.probabilities.setflags(write=True)
        with self.assertRaises(TypeError):r.certificate['primal_feasible']=False
        self.assertEqual(r.q_sha256,canonical_hash({'support_ids':list(p.support_ids),'q':[0.,10.]}))

    def test_invalid_data_axes_bounds_ids_and_nan_rejected(self):
        cases=[(('a','a'),('f',),[[0.],[1.]],[0.],[1.]),
               (('a',),('f','f'),[[0.,0.]],[0.,0.],[1.,1.]),
               ((),('f',),np.empty((0,1)),[0.],[1.]),
               (('a','b'),('f',),[[0.,1.]],[0.],[1.]),
               (('a',),('f',),[[np.nan]],[0.],[1.]),
               (('a',),('f',),[[0.]],[1.],[0.])]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ValueError):ProbabilitySet(*args)
        p=probability_set([[0.],[1.]],[0.],[1.])
        for q in ([[1.,2.]],[1.,np.inf]):
            with self.assertRaises(ValueError):solve_worst_distribution(p,q,q_support_ids=p.support_ids)
        for limit in (0,-1,True,np.inf,np.nan):
            with self.assertRaises(ValueError):solve_worst_distribution(p,[1.,2.],q_support_ids=p.support_ids,time_limit_seconds=limit)

    def test_timeout_does_not_claim_optimal_or_invent_witness(self):
        p=probability_set([[0.],[1.]],[0.],[1.])
        raw=SimpleNamespace(status=1,x=None)
        record={'status':1,'success':False,'message':'forced time limit','raw_solution_scaled':None,
                'infeasibility_certificate_available':False}
        with patch('ev_assignment.methods.probability._solver',return_value=(raw,record)) as solve:
            r=solve_worst_distribution(p,[1.,2.],q_support_ids=p.support_ids)
        self.assertEqual(solve.call_count,2)
        self.assertEqual(r.status,'TIME_LIMIT');self.assertFalse(r.certified_optimal)
        self.assertIsNone(r.lower_bound_usd);self.assertIsNone(r.upper_bound_usd)
        self.assertIsNone(r.probabilities)

    def test_time_limited_valid_witnesses_are_bounds_not_optimal(self):
        import ev_assignment.methods.probability as module
        actual=module._solver;p=probability_set([[0.],[1.]],[.25],[.5]);calls=0
        def limited(*args,**kwargs):
            nonlocal calls
            calls+=1;raw,record=actual(*args,**kwargs)
            if calls==2:
                raw.status=1;raw.success=False;record.update(status=1,success=False,message='forced limit with incumbent')
            return raw,record
        with patch.object(module,'_solver',side_effect=limited):
            r=solve_worst_distribution(p,[260000.,260010.],q_support_ids=p.support_ids)
        self.assertEqual(r.status,'TIME_LIMIT');self.assertFalse(r.certified_optimal)
        self.assertAlmostEqual(r.lower_bound_usd,260005.);self.assertAlmostEqual(r.upper_bound_usd,260005.)
        self.assertEqual(r.certificate['q_offset_usd'],260000.)
        self.assertTrue(r.certificate['gap_passed'])

    def test_solver_status_zero_cannot_hide_invalid_dual_certificate(self):
        import ev_assignment.methods.probability as module
        actual=module._solver;p=probability_set(np.empty((2,0)),[],[]);calls=0
        def corrupted(*args,**kwargs):
            nonlocal calls
            calls+=1;raw,record=actual(*args,**kwargs)
            if calls==2:
                raw.x[0]-=1e-6
            return raw,record
        with patch.object(module,'_solver',side_effect=corrupted):
            r=solve_worst_distribution(p,[260000.,260001.],q_support_ids=p.support_ids)
        self.assertEqual(r.status,'NUMERICAL_CERTIFICATE_FAILED');self.assertFalse(r.certified_optimal)
        self.assertTrue(r.certificate['primal_feasible']);self.assertFalse(r.certificate['dual_feasible'])
        self.assertEqual(r.certificate['q_scale_usd'],1.)
        self.assertEqual(r.certificate['certification_scale_usd'],260001.)
        self.assertGreater(r.certificate['dual']['max_support_violation_usd'],1e-8)
        self.assertLess(r.certificate['dual']['max_normalized_support_violation'],1e-8)
        self.assertEqual(r.certificate['separation_tolerance_usd'],1e-4+1e-8*260001.)
        self.assertIsNone(r.upper_bound_usd)

    def test_two_independent_lp_calls_and_dual_theta_free(self):
        import ev_assignment.methods.probability as module
        actual=module._solver;p=probability_set([[0.],[1.]],[.25],[.5])
        with patch.object(module,'_solver',wraps=actual) as solve:
            r=self.solve(p,[10.,0.])
        self.assertEqual(solve.call_count,2)
        primal,dual=solve.call_args_list
        np.testing.assert_array_equal(primal.args[0],[-1.,0.])
        np.testing.assert_array_equal(dual.args[0],[1.,-.25,.5])
        self.assertEqual(dual.args[5][0],(None,None))
        self.assertTrue(r.certified_optimal)


if __name__=='__main__':unittest.main()
