"""Optional backend gate; local CI checks the original-matrix witness logic."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix
script_path=os.environ.get('EVCS_GUROBI_DIAGNOSTIC_SCRIPT')
if script_path is None:
    from evcs_experiments.config import get_paths
    script_path=get_paths().project_root/'scripts/diagnostics/profile_gurobi_rmp.py'
sp=importlib.util.spec_from_file_location('gurobi_diagnostic', Path(script_path))
d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)


class OriginalWitnessTests(unittest.TestCase):
    def test_fractional_and_infeasible_witness_rejected(self):
        a=csr_matrix([[2.]])
        b=dict(c=np.array([1.]),integrality=np.array([1]),lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),row_lower=np.array([1.]),row_upper=np.array([np.inf]))
        self.assertTrue(d.original_check(a,b,[1.])['passed'])
        self.assertFalse(d.original_check(a,b,[.5])['passed'])
        self.assertFalse(d.original_check(a,b,[0.])['passed'])


def licensed_backend_gate(output):
    """Explicit server call only; no CI license lookup or skipped tests."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    rows=[]
    fixtures=[
      (csr_matrix([[2.]]), dict(c=np.array([1.]),integrality=np.array([1]),lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),row_lower=np.array([1.]),row_upper=np.array([np.inf])),np.array([1.]),1.),
      (csr_matrix([[1.,1.],[1.,-1.]]),dict(c=np.array([2.,-1.]),integrality=np.array([0,0]),lower_bounds=np.array([-np.inf,0.]),upper_bounds=np.array([np.inf,3.]),row_lower=np.array([1.,-2.]),row_upper=np.array([1.,2.])),None,-2.5),
      (csr_matrix([[1.]]),dict(c=np.array([1.]),integrality=np.array([1]),lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),row_lower=np.array([2.]),row_upper=np.array([np.inf])),None,None)]
    for i,(a,b,hint,expected) in enumerate(fixtures):
        raw=milp(b['c'],integrality=b['integrality'],bounds=Bounds(b['lower_bounds'],b['upper_bounds']),constraints=LinearConstraint(a,b['row_lower'],b['row_upper']),options={'time_limit':5,'mip_rel_gap':1e-9})
        record,values=d.solve_saved(a,b,hint,output/f'case_{i}.log',seconds=5,threads=1)
        if expected is None:
            assert raw.status==2 and record['status']=='INFEASIBLE' and record['solution_count']==0
        else:
            assert raw.status==0 and abs(raw.fun-expected)<1e-8
            assert record['status']=='OPTIMAL' and record['valid_incumbent']
            assert abs(record['incumbent_without_constant_usd']-expected)<1e-8 and abs(record['solver_bound_without_constant_usd']-expected)<1e-8
            assert np.array_equal(b['integrality'],np.array([1]) if i==0 else np.array([0,0]))
        rows.append(record)
    return rows
