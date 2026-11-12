"""License-free adapter tests; no test imports the actual gurobipy package."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.sparse import csr_matrix

from evcs_experiments import native_gurobi as native


class FakeGurobiError(Exception):
    pass


def fake_backend(**changes):
    grb = SimpleNamespace(INFINITY=1e100, MINIMIZE=1, INTEGER='I', CONTINUOUS='C',
                          EQUAL='=', LESS_EQUAL='<', GREATER_EQUAL='>')
    values = dict(Status=2, SolCount=1, IsMIP=1, ObjVal=1., ObjBound=1., MIPGap=0.,
                  NodeCount=0., Runtime=.01, MaxMemUsed=.02, MemUsed=.01,
                  IterCount=0., BarIterCount=0, NumConstrs=1, NumVars=1)
    values.update(changes)
    model = MagicMock()
    for name, value in values.items():
        setattr(model, name, value)
    model.__enter__.return_value = model
    variables = SimpleNamespace(X=np.array([1.]))
    model.addMVar.return_value = variables
    env = MagicMock(); env.__enter__.return_value = env
    gp = SimpleNamespace(GRB=grb, GurobiError=FakeGurobiError,
                         gurobi=SimpleNamespace(version=lambda: (12, 0, 3)),
                         Env=MagicMock(return_value=env), Model=MagicMock(return_value=model))
    return gp, model, variables


def small_arrays(integer=1):
    return native._model_arrays(np.array([1.]), np.array([integer]), np.array([0.]),
        np.array([2.]), csr_matrix([[1.]]), np.array([1.]), np.array([np.inf]))


class NativeGurobiTests(unittest.TestCase):
    def extract(self, **changes):
        gp, model, variables = fake_backend(**changes)
        return native._extract_result(gp, model, variables, small_arrays(), {})

    def test_optimum_keeps_certificate_and_native_status(self):
        r = self.extract()
        self.assertTrue(r.success); self.assertEqual(r.status, 0)
        self.assertEqual(r.fun, 1.); self.assertEqual(r.mip_dual_bound, 1.)
        self.assertTrue(r.native_record['valid_incumbent'])
        self.assertEqual(r.native_record['raw_model_status'], 2)
        self.assertEqual(r.native_record['bound_source'], 'MIP_ObjBound')

    def test_limit_without_incumbent_retains_only_finite_mip_bound(self):
        for status, name in ((9, 'TIME_LIMIT'), (17, 'MEM_LIMIT'), (11, 'INTERRUPTED')):
            with self.subTest(status=name):
                r = self.extract(Status=status, SolCount=0, ObjBound=.5, MIPGap=np.inf)
                self.assertEqual(r.status, 1); self.assertEqual(r.limit_status, name)
                self.assertIsNone(r.x); self.assertIsNone(r.fun); self.assertIsNone(r.mip_gap)
                self.assertEqual(r.mip_dual_bound, .5)

    def test_nonfinite_or_sentinel_bound_is_not_published(self):
        for value in (float('nan'), float('inf'), -1e100, 1e100, None):
            with self.subTest(value=value):
                r = self.extract(Status=9, SolCount=0, ObjBound=value)
                self.assertIsNone(r.mip_dual_bound)

    def test_stopped_continuous_lp_objective_never_becomes_bound(self):
        for status in (9, 17, 11):
            r = self.extract(Status=status, IsMIP=0, ObjBound=.5)
            self.assertEqual(r.fun, 1.)
            self.assertIsNone(r.mip_dual_bound); self.assertIsNone(r.mip_gap)
        r = self.extract(IsMIP=0)
        self.assertEqual(r.mip_dual_bound, 1.); self.assertEqual(r.mip_gap, 0.)
        self.assertEqual(r.native_record['bound_source'], 'optimal_LP_objective')

    def test_numeric_and_ambiguous_native_status_never_claim_bound(self):
        for status, scipy_status in ((12, 4), (4, 4), (3, 2), (5, 3), (13, 4)):
            with self.subTest(status=status):
                r = self.extract(Status=status)
                self.assertEqual(r.status, scipy_status)
                self.assertIsNone(r.mip_dual_bound)
                self.assertEqual(r.native_record['raw_model_status'], status)

    def test_invalid_incumbent_and_inconsistent_bound_are_withheld(self):
        gp, model, variables = fake_backend(Status=9)
        variables.X = np.array([.5])
        r = native._extract_result(gp, model, variables, small_arrays(), {})
        self.assertIsNone(r.x); self.assertIsNone(r.fun)
        self.assertFalse(r.native_record['valid_incumbent'])
        self.assertEqual(r.status, 4)
        self.assertEqual(r.native_record['status'], 'TIME_LIMIT')
        self.assertEqual(r.limit_status, 'TIME_LIMIT')
        self.assertTrue(r.native_record['certificate_failures'])
        r = self.extract(ObjBound=2.)
        self.assertIsNone(r.mip_dual_bound)
        self.assertEqual(r.native_record['raw_solver_bound_without_constant_usd'], 2.)
        self.assertFalse(r.success)
        self.assertEqual(r.status, 4)
        r = self.extract(ObjVal=1.1)
        self.assertIsNone(r.x)
        self.assertFalse(r.native_record['objective_matches'])
        self.assertEqual(r.status, 4)
        r = self.extract(Status=17, ObjBound=2.)
        self.assertEqual(r.status, 4)
        self.assertEqual(r.limit_status, 'MEM_LIMIT')
        self.assertEqual(r.native_record['raw_model_status'], 17)

    def test_row_mapping_preserves_equality_ranged_and_free_rows(self):
        lo = np.array([2., -np.inf, 1., 3., -np.inf])
        hi = np.array([2., 5., np.inf, 7., np.inf])
        rows = native._row_map(lo, hi)
        np.testing.assert_array_equal(rows['eq'], [0])
        np.testing.assert_array_equal(rows['le'], [1, 3])
        np.testing.assert_array_equal(rows['ge'], [2, 3])

    def test_dangerous_inputs_are_rejected_before_gurobi_import(self):
        arrays = list(small_arrays())
        bad = []
        for i, value in ((0, np.array([np.nan])), (0, np.array([1.+1.j])), (1, np.array([2])),
                         (2, np.array([3.])), (3, np.array([1e20])),
                         (4, csr_matrix([[1e-14]])), (5, np.array([np.inf]))):
            changed = arrays.copy(); changed[i] = value; bad.append(changed)
        with patch.object(native.importlib, 'import_module') as importer:
            for values in bad:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    native.solve_milp(*values, options={'time_limit': 1.}, log_path='unused.log')
            with self.assertRaisesRegex(ValueError, 'start violates'):
                native.solve_milp(*arrays, options={'time_limit': 1.}, start=np.array([0.]), log_path='unused.log')
            with self.assertRaisesRegex(ValueError, 'log_path'):
                native.solve_milp(*arrays, options={'time_limit': 1.})
            importer.assert_not_called()

    def test_native_row_translation_preserves_both_sides_of_ranged_rows(self):
        gp, model, variables = fake_backend()
        lo = np.array([2., -np.inf, 1., 3., -np.inf])
        hi = np.array([2., 5., np.inf, 7., np.inf])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(native.importlib, 'import_module', return_value=gp):
            r = native.solve_milp([1.], [1], [0.], [2.], csr_matrix([[1.], [2.], [3.], [4.], [5.]]),
                lo, hi, options={'time_limit': 10.}, log_path=Path(tmp)/'rows.log')
        calls = model.addMConstr.call_args_list
        self.assertEqual([call.kwargs['name'] for call in calls], ['eq', 'le', 'ge'])
        for call, coefficients, sense, rhs in zip(calls, ([1.], [2., 4.], [3., 4.]),
                                                 ('=', '<', '>'), ([2.], [5., 7.], [1., 3.])):
            np.testing.assert_array_equal(call.args[0].toarray().ravel(), coefficients)
            self.assertEqual(call.args[2], sense)
            np.testing.assert_array_equal(call.args[3], rhs)
        self.assertEqual(r.native_record['row_map_summary']['free_rows'], 1)
        self.assertEqual(r.native_record['row_map_summary']['ranged_rows'], 1)
        self.assertEqual(r.native_record['row_map_summary']['translated_rows'], 5)

    def test_parameters_are_bounded_and_cannot_silently_override_tolerances(self):
        for options in ({'time_limit': True}, {'time_limit': float('inf')}, {'time_limit': 0},
                        {'time_limit': 1, 'threads': 5}, {'time_limit': 1, 'threads': 1.5},
                        {'time_limit': 1, 'mip_rel_gap': -1}, {'time_limit': 1, 'seed': 3},
                        {'time_limit': 1, 'primal_feasibility_tolerance': 1e-6}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                native._parameters(options)
        seconds, params = native._parameters({'time_limit': 10., 'threads': 4, 'mip_rel_gap': .001})
        self.assertEqual(seconds, 10.)
        self.assertEqual(params['SoftMemLimit'], 24.); self.assertEqual(params['Seed'], 0)
        for name in ('FeasibilityTol', 'OptimalityTol', 'IntFeasTol'):
            self.assertEqual(params[name], 1e-8)

    def test_explicit_solver_refinement_only_tightens_solver_not_acceptance(self):
        for tol in (1e-9,1e-8):
            _,params=native._parameters({'time_limit':1.,'solver_feasibility_tolerance':tol})
            self.assertEqual(params['FeasibilityTol'],tol)
            self.assertEqual(params['IntFeasTol'],1e-8);self.assertEqual(params['OptimalityTol'],1e-8)
        for tol in (True,'1e-9',None,1e-7,1e-10,0.,float('nan'),float('inf')):
            with self.subTest(tol=tol),self.assertRaises(ValueError):
                native._parameters({'time_limit':1.,'solver_feasibility_tolerance':tol})
        gp,model,variables=fake_backend(ObjVal=1.-1.02076214e-8,ObjBound=1.-1.02076214e-8)
        variables.X=np.array([1.-1.02076214e-8])
        _,options=native._parameters({'time_limit':1.,'solver_feasibility_tolerance':1e-9})
        raw=native._extract_result(gp,model,variables,small_arrays(integer=0),{'effective_options':options})
        self.assertEqual(raw.status,4);self.assertIsNone(raw.x)
        self.assertGreater(raw.native_record['original_certificate']['max_row_violation'],1e-8)
        self.assertEqual(native.TOLERANCE,1e-8)

    def test_fake_backend_translation_start_log_and_load_budget(self):
        gp, model, variables = fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            logfile = Path(tmp)/'run.log'
            with patch.object(native.importlib, 'import_module', return_value=gp), \
                 patch.object(native.time, 'monotonic', side_effect=[100., 102., 103.]):
                r = native.solve_with_start(*small_arrays(), np.array([1.]),
                    {'time_limit': 10., 'threads': 4, 'mip_rel_gap': .001}, logfile)
            self.assertTrue(logfile.exists())
            self.assertEqual(r.native_record['effective_options']['TimeLimit'], 8.)
            self.assertEqual(r.native_record['load_seconds'], 2.)
            self.assertEqual(r.native_record['wall_seconds'], 3.)
            np.testing.assert_array_equal(variables.Start, [1.])
            self.assertTrue(r.native_record['start_is_not_fixed_decision'])
            model.optimize.assert_called_once()
            kwargs = model.addMVar.call_args.kwargs
            np.testing.assert_array_equal(kwargs['lb'], [0.])
            np.testing.assert_array_equal(kwargs['ub'], [2.])
            with patch.object(native.importlib, 'import_module') as importer:
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    native.solve_milp(*small_arrays(), options={'time_limit': 1.}, log_path=logfile)
                importer.assert_not_called()

    def test_model_loading_exhaustion_does_not_optimize_or_claim_certificate(self):
        gp, model, variables = fake_backend(Status=1, SolCount=0)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(native.importlib, 'import_module', return_value=gp), \
             patch.object(native.time, 'monotonic', side_effect=[100., 102.]):
            r = native.solve_milp(*small_arrays(), options={'time_limit': 1.},
                                  log_path=Path(tmp)/'load.log')
        model.optimize.assert_not_called()
        self.assertEqual(r.status, 1); self.assertEqual(r.limit_status, 'TIME_LIMIT')
        self.assertIsNone(r.x); self.assertIsNone(r.mip_dual_bound)
        self.assertFalse(r.native_record['valid_incumbent'])

    def test_wrong_version_or_absent_optional_backend_has_no_fallback(self):
        gp, model, variables = fake_backend()
        gp.gurobi.version = lambda: (13, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            for outcome in (gp, ModuleNotFoundError('gurobipy unavailable')):
                patch_args = ({'side_effect': outcome} if isinstance(outcome, Exception)
                              else {'return_value': outcome})
                expected = ModuleNotFoundError if isinstance(outcome, Exception) else RuntimeError
                with patch.object(native.importlib, 'import_module', **patch_args), self.assertRaises(expected):
                    native.solve_milp(*small_arrays(), options={'time_limit': 1.}, log_path=Path(tmp)/'new.log')
            gp.Env.assert_not_called()


if __name__ == '__main__':
    unittest.main()
