from fractions import Fraction
import copy
import math
import time
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.bounded_row_relaxation import relax_bounded_rows


def model(matrix,rl,ru,lower,upper):
    n=len(lower)
    return dict(matrix=csr_matrix(matrix),row_lower=np.array(rl,dtype=float),row_upper=np.array(ru,dtype=float),
                lower_bounds=np.array(lower,dtype=float),upper_bounds=np.array(upper,dtype=float),
                c=np.arange(n,dtype=float),integrality=np.zeros(n),objective_constant_usd=3.)


class BoundedRowRelaxationTests(unittest.TestCase):
    def test_signed_coefficients_ranged_row_exact_compensation_and_no_mutation(self):
        a=model([[2.,2e-14,-3e-14]],[-1.],[1.],[0.,-2.,1.],[1.,4.,5.]);original=a['matrix'].copy()
        b,r=relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(1,3))
        self.assertEqual((a['matrix']!=original).nnz,0);self.assertEqual(b['matrix'].nnz,1)
        for key in ('c','integrality','lower_bounds','upper_bounds'):self.assertIs(b[key],a[key])
        for x in (-2.,0.,4.):
            for y in (1.,3.,5.):
                # Choose original row activity exactly zero.
                z=-(2e-14*x-3e-14*y)/2
                activity=2*z
                self.assertLessEqual(b['row_lower'][0],activity);self.assertGreaterEqual(b['row_upper'][0],activity)
        self.assertEqual(r['removed_entry_count'],2);self.assertFalse(r['exact_original_formulation'])

    def test_zero_rhs_subnormal_outward_rounding(self):
        a=model([[1e-320]], [0.],[0.],[0.1],[0.1])
        b,r=relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(0,1))
        exact=-Fraction.from_float(1e-320)*Fraction.from_float(.1)
        self.assertLessEqual(Fraction.from_float(b['row_lower'][0]),exact)
        self.assertGreaterEqual(Fraction.from_float(b['row_upper'][0]),exact)
        self.assertLess(b['row_lower'][0],b['row_upper'][0])

    def test_nonzero_rhs_tiny_drop_below_ulp_rounds_down(self):
        a=model([[1e-14]], [230000.],[np.inf],[0.],[15.])
        b,r=relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(0,1))
        self.assertEqual(b['row_lower'][0],math.nextafter(230000.,-math.inf))
        self.assertTrue(np.isposinf(b['row_upper'][0]));self.assertIsNone(r['rows'][0]['min_sum_exact'])

    def test_tiny_outside_whitelist_threshold_edge_and_invalid_ranges(self):
        a=model([[1e-14,1e-13]],[-1.],[1.],[0.,0.],[1.,1.])
        with self.assertRaises(ValueError):relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(1,2))
        b,r=relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(0,1))
        self.assertEqual(b['matrix'][0,1],1e-13)
        for rows in ((True,1),(1,0),(-1,1)):
            with self.assertRaises(ValueError):relax_bounded_rows(a,allowed_rows=rows,allowed_columns=(0,2))

    def test_unbounded_endpoint_refused_and_unused_side_allowed(self):
        a=model([[1e-14]],[0.],[np.inf],[0.],[np.inf])
        with self.assertRaisesRegex(ValueError,'endpoint'):relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(0,1))
        a['row_lower'][0]=-np.inf;a['row_upper'][0]=1.
        b,r=relax_bounded_rows(a,allowed_rows=(0,1),allowed_columns=(0,1))
        self.assertTrue(np.isneginf(b['row_lower'][0]));self.assertEqual(b['row_upper'][0],1.)

    def test_original_reserve_constraint_supplies_only_compensation_bound(self):
        a=model([[-15.,1.],[0.,1e-14]],[-np.inf,230000.],[0.,np.inf],[0.,0.],[1.,np.inf])
        kw=dict(allowed_rows=(1,2),allowed_columns=(1,2),witness_rows=(0,1),bound_witnesses={1:0})
        b,r=relax_bounded_rows(a,**kw)
        self.assertEqual(r['bound_witnesses'][0]['upper_bound'],15.)
        self.assertTrue(np.isposinf(b['upper_bounds'][1]));self.assertEqual((b['matrix'][:1]!=a['matrix'][:1]).nnz,0)
        self.assertEqual(b['row_lower'][1],math.nextafter(230000.,-math.inf))
        bad=copy.deepcopy(a);bad['upper_bounds'][0]=np.inf
        with self.assertRaisesRegex(ValueError,'endpoint'):relax_bounded_rows(bad,**kw)
        with self.assertRaises(ValueError):relax_bounded_rows(a,**dict(kw,witness_rows=(1,2),bound_witnesses={1:1}))

    def test_fractional_witness_upper_rounds_outward(self):
        a=model([[-1.,3.],[0.,1e-14]],[-np.inf,1.],[0.,np.inf],[0.,0.],[1.,np.inf])
        b,r=relax_bounded_rows(a,allowed_rows=(1,2),allowed_columns=(1,2),witness_rows=(0,1),bound_witnesses={1:0})
        witness=r['bound_witnesses'][0]
        self.assertEqual(witness['upper_bound_exact'],{'numerator':'1','denominator':'3'})
        self.assertGreaterEqual(Fraction.from_float(witness['upper_bound']),Fraction(1,3))
        exact_rhs=Fraction(1)-Fraction.from_float(1e-14)*Fraction(1,3)
        self.assertLessEqual(Fraction.from_float(b['row_lower'][1]),exact_rhs)

    def test_negative_target_witness_and_expired_deadline_rejected(self):
        a=model([[1.,-1.],[0.,1e-14]],[-np.inf,0.],[0.,np.inf],[0.,0.],[1.,np.inf])
        with self.assertRaisesRegex(ValueError,'positive'):relax_bounded_rows(a,allowed_rows=(1,2),allowed_columns=(1,2),witness_rows=(0,1),bound_witnesses={1:0})
        with self.assertRaises(TimeoutError):relax_bounded_rows(a,allowed_rows=(1,2),allowed_columns=(1,2),deadline=time.monotonic()-1)


if __name__=='__main__':unittest.main()
