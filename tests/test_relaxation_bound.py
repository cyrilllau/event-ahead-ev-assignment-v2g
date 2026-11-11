"""Hand-sized LP relaxation and original-unit certificate tests; no formal solve."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix

from evcs_experiments import relaxation_bound as r


def model():
    # 2x+y=0, .25 <= x <=1, x binary, y free: LP7.25; MILP8.
    return dict(matrix=csr_matrix([[2.,1.],[1.,0.]]),row_lower=np.array([0.,.25]),row_upper=np.array([0.,1.]),
        c=np.array([3.,1.]),lower_bounds=np.array([0.,-np.inf]),upper_bounds=np.array([1.,np.inf]),
        integrality=np.array([1,0]),objective_constant_usd=7.,source_model_sha256='a'*64,formal_problem_sha256='b'*64)


class RelaxationBoundTests(unittest.TestCase):
    def solve(self,p=None):return r.solve_relaxation_bound(model() if p is None else p,time_limit_seconds=5)

    def test_true_fractional_lower_bound_with_equality_ranged_row_and_constant(self):
        p=model();original=copy.deepcopy(p);record,values=self.solve(p)
        self.assertEqual(record['status'],'OPTIMAL');self.assertEqual(record['bound_kind'],'NUMERICAL_LP_RELAXATION_BOUND')
        self.assertAlmostEqual(record['lower_bound_usd'],7.25);self.assertLess(record['lower_bound_usd'],8.)
        np.testing.assert_allclose(values['primal'],[.25,-.5],atol=1e-12)
        self.assertEqual(record['certificate']['free_column_count'],1)
        self.assertEqual(record['certificate']['max_free_column_stationarity_residual'],0.)
        self.assertFalse(record['exact_arithmetic_proof']);self.assertEqual(record['lp_calls'],1)
        self.assertEqual(record['master_milp_calls'],record['recourse_calls'],0)
        self.assertEqual(record['source_model_sha256'],'a'*64)
        self.assertEqual(record['solver_method'],'highs-ipm');self.assertTrue(record['scipy_version']);self.assertTrue(record['highs_version'])
        for key in ('row_lower','row_upper','c','lower_bounds','upper_bounds','integrality'):np.testing.assert_array_equal(p[key],original[key])
        self.assertEqual((p['matrix']!=original['matrix']).nnz,0)
        self.assertAlmostEqual(values['row_lower_marginals'][1],1.)
        self.assertAlmostEqual(values['row_equality_marginals'][0],1.)

    def test_original_upper_row_and_lower_variable_signs(self):
        p=dict(matrix=csr_matrix([[1.,0.]]),row_lower=[-np.inf],row_upper=[2.],c=[-2.,3.],
               lower_bounds=[0.,1.],upper_bounds=[np.inf,np.inf],integrality=[0,0],objective_constant_usd=-5.)
        record,v=self.solve(p);self.assertEqual(record['status'],'OPTIMAL');self.assertEqual(record['lower_bound_usd'],-6.)
        self.assertEqual(v['row_upper_marginals'][0],-2.);self.assertEqual(v['variable_lower_marginals'][1],3.)

    def test_upper_variable_and_fixed_variable_bound_contributions(self):
        p=dict(matrix=csr_matrix((0,2)),row_lower=[],row_upper=[],c=[-2.,3.],lower_bounds=[-np.inf,2.],upper_bounds=[4.,2.],integrality=[0,1],objective_constant_usd=10.)
        record,v=self.solve(p);self.assertEqual(record['status'],'OPTIMAL');self.assertEqual(record['lower_bound_usd'],8.)
        self.assertEqual(v['variable_upper_marginals'][0],-2.)

    def test_dual_and_primal_tampering_fail_original_tolerance(self):
        p=model();record,v=self.solve(p)
        for field,delta in [('row_equality_marginals',2e-8),('primal',2e-8),('row_upper_marginals',2e-8)]:
            changed={k:a.copy() for k,a in v.items()};changed[field][0]+=delta
            cert=r.check_relaxation_certificate(p,changed,solver_status=0,raw_objective_without_constant_usd=.25)
            with self.subTest(field=field):self.assertFalse(cert['passed']);self.assertIsNone(cert['certified_lower_bound_usd'])

    def test_infinite_bound_multiplier_and_wrong_row_mask_fail(self):
        p=model();_,v=self.solve(p)
        for field,index in [('variable_lower_marginals',1),('variable_upper_marginals',1),('row_lower_marginals',0)]:
            changed={k:a.copy() for k,a in v.items()};changed[field][index]=1.
            cert=r.check_relaxation_certificate(p,changed,solver_status=0,raw_objective_without_constant_usd=.25)
            self.assertFalse(cert['passed']);self.assertGreater(cert['max_inactive_or_infinite_bound_multiplier'],1e-8)

    def test_optimal_label_without_witness_or_objective_is_not_certificate(self):
        p=model();_,v=self.solve(p)
        self.assertFalse(r.check_relaxation_certificate(p,{},solver_status=0)['passed'])
        self.assertFalse(r.check_relaxation_certificate(p,v,solver_status=0)['passed'])
        self.assertFalse(r.check_relaxation_certificate(p,v,solver_status=0,raw_objective_without_constant_usd=999.)['passed'])

    def test_timeout_does_not_promote_even_valid_optimal_witness(self):
        p=model();_,v=self.solve(p)
        cert=r.check_relaxation_certificate(p,v,solver_status=1,raw_objective_without_constant_usd=.25)
        self.assertFalse(cert['passed']);self.assertIsNone(cert['certified_lower_bound_usd'])
        raw=OptimizeResult(status=1,success=False,message='Time limit reached',x=None,fun=None,nit=3)
        with patch.object(r,'linprog',return_value=raw) as solve:record,values=self.solve(p)
        self.assertEqual(record['status'],'TIME_LIMIT');self.assertIsNone(record['lower_bound_usd']);self.assertEqual(values,{})
        self.assertEqual(solve.call_args.kwargs['method'],'highs-ipm')

    def test_infeasible_and_unbounded_report_no_bound(self):
        p=model();p['row_lower'][1]=2.;p['row_upper'][1]=3.
        record,_=self.solve(p);self.assertEqual(record['status'],'INFEASIBLE');self.assertIsNone(record['lower_bound_usd'])
        p=dict(matrix=csr_matrix((0,1)),row_lower=[],row_upper=[],c=[-1.],lower_bounds=[0.],upper_bounds=[np.inf],integrality=[0],objective_constant_usd=0.)
        record,_=self.solve(p);self.assertEqual(record['status'],'UNBOUNDED');self.assertIsNone(record['lower_bound_usd'])

    def test_malformed_solver_dual_is_retained_but_cannot_supply_bound(self):
        raw=OptimizeResult(status=0,success=True,message='claimed optimum',x=np.array([.25,-.5]),fun=.25,nit=1,
            ineqlin=OptimizeResult(marginals=np.array([0.,-1.])),eqlin=OptimizeResult(marginals=np.array([np.nan])),
            lower=OptimizeResult(marginals=np.zeros(2)),upper=OptimizeResult(marginals=np.zeros(2)))
        with patch.object(r,'linprog',return_value=raw):record,values=self.solve()
        self.assertEqual(record['status'],'NUMERICAL_CERTIFICATE_FAILED');self.assertIsNone(record['lower_bound_usd'])
        self.assertTrue(np.isnan(values['row_equality_marginals'][0]))

    def test_integrality_2_3_and_fractional_flags_rejected(self):
        for flag in (2,3,.5):
            p=model();p['integrality']=[flag,0]
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,'Only binary/integer'):self.solve(p)

    def test_bad_bounds_nonfinite_coefficients_and_witness_shapes_rejected(self):
        for key,value in [('c',[np.nan,1.]),('row_lower',[np.inf,.25]),('upper_bounds',[-1.,np.inf])]:
            p=model();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.solve(p)
        _,v=self.solve();v['row_lower_marginals']=np.zeros(3)
        with self.assertRaisesRegex(ValueError,'witness shape'):r.check_relaxation_certificate(model(),v,solver_status=0,raw_objective_without_constant_usd=.25)

    def test_digest_binds_integrality_constant_matrix_and_bounds(self):
        original=r.coefficient_sha256(model())
        for field in ('integrality','objective_constant_usd','matrix','lower_bounds'):
            p=model()
            if field=='integrality':p[field][0]=0
            elif field=='objective_constant_usd':p[field]+=1.
            elif field=='matrix':p[field].data[0]+=1.
            else:p[field][0]=-.5
            self.assertNotEqual(r.coefficient_sha256(p),original)

    def test_preparation_budget_exhaustion_does_not_call_solver(self):
        with patch.object(r.time,'monotonic',side_effect=[0.,2.,2.]),patch.object(r,'linprog',side_effect=AssertionError('no launch')):
            record,values=r.solve_relaxation_bound(model(),time_limit_seconds=1)
        self.assertEqual(record['status'],'TIME_LIMIT');self.assertEqual(record['lp_calls'],0);self.assertIsNone(record['lower_bound_usd'])

    def test_solver_summary_exclusive_and_json_finite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'solver.json';record,_=r.solve_relaxation_bound(model(),time_limit_seconds=5,log_path=path)
            self.assertEqual(json.loads(path.read_text()),record)
            with self.assertRaisesRegex(ValueError,'overwrite'):r.solve_relaxation_bound(model(),time_limit_seconds=5,log_path=path)


if __name__=='__main__':unittest.main()
