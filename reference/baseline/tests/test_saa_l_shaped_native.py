"""License-free translation guards; mock calls are not native Gurobi admission."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix

from evcs_experiments import saa_l_shaped_native as adapter
from evcs_experiments.campaign_batch import file_reference, verify_reference
from evcs_experiments.relaxation_bound import coefficient_sha256


def small_master():
    return dict(c=np.array([2., .5, .5]), integrality=np.array([1, 0, 0]),
        lower_bounds=np.array([0., -100., -90.]), upper_bounds=np.array([1., np.inf, np.inf]),
        matrix=csr_matrix([[1., 0., 0.], [-2., 1., 0.]]),
        row_lower=np.array([0., -99.]), row_upper=np.array([1., np.inf]),
        first_stage_columns=1, theta_columns=2, recourse_columns=0, cut_rows=1)


def raw_result(*, normalized=1, native=17, native_name='MEM_LIMIT', bound=None, incumbent=False):
    return OptimizeResult(status=normalized, success=normalized == 0, message='Gurobi native: ' + native_name,
        x=np.array([0., -99., -90.]) if incumbent else None,
        fun=-94.5 if incumbent else None, mip_dual_bound=bound,
        mip_gap=None, mip_node_count=0., limit_status=native_name if native in adapter.native_gurobi._LIMIT_STATUSES else None,
        native_record=dict(backend='gurobipy', gurobi_version=[12, 0, 3],
            raw_model_status=native, status=native_name, valid_incumbent=incumbent,
            solver_bound_without_constant_usd=bound,
            incumbent_without_constant_usd=-94.5 if incumbent else None,
            bound_source='MIP_ObjBound' if bound is not None else None,
            certificate_failures=[]))


class NativeSmallMasterAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.env = patch.dict(adapter.os.environ, {key: '1' for key in adapter.BLAS_ENVIRONMENT})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)

    def invoke(self, model=None, raw=None, **options):
        model = small_master() if model is None else model
        raw = raw_result() if raw is None else raw
        def native(*args, **kwargs):
            Path(kwargs['log_path']).write_text('Mock adapter fixture; Gurobi was not executed.\n')
            options = kwargs['options']
            _, parameters = adapter.native_gurobi._parameters(options)
            load_seconds = raw.native_record.get('load_seconds', .001)
            run_limit = max(0., options['time_limit'] - load_seconds)
            parameters.update(LogFile=str(kwargs['log_path']))
            if run_limit > 0:
                parameters['TimeLimit'] = run_limit
            raw.native_record.update(effective_options=parameters,
                requested_time_limit_seconds=options['time_limit'], log_path=str(kwargs['log_path']),
                load_seconds=load_seconds, effective_run_time_limit_seconds=run_limit,
                original_matrix_shape=list(args[4].shape), original_matrix_nnz=int(args[4].nnz))
            return raw
        kwargs = dict(time_limit_seconds=5., mip_relative_gap=.001, output_root=self.base / 'native')
        kwargs.update(options)
        with patch.object(adapter.native_gurobi, 'solve_milp', side_effect=native) as call:
            result = adapter.solve_native_small_master(model, **kwargs)
        return result, call

    def test_exact_arrays_and_uniform_constants_forwarded_once(self):
        model = small_master()
        raw = raw_result(normalized=0, native=2, native_name='OPTIMAL', bound=-94.5, incumbent=True)
        result, call = self.invoke(model, raw, threads=4)
        call.assert_called_once()
        for supplied, key in zip(call.call_args.args,
                ('c', 'integrality', 'lower_bounds', 'upper_bounds', 'matrix', 'row_lower', 'row_upper')):
            self.assertIs(supplied, model[key])
        self.assertIs(result.raw, raw)
        options = call.call_args.kwargs['options']
        self.assertEqual(options, result.effective_options)
        self.assertEqual(options['threads'], 4)
        self.assertEqual(options['mip_rel_gap'], .001)
        self.assertEqual(options['primal_feasibility_tolerance'], 1e-8)
        self.assertEqual(options['dual_feasibility_tolerance'], 1e-8)
        self.assertGreater(options['time_limit'], 0)
        self.assertLess(options['time_limit'], 5)
        verify_reference(result.evidence_reference)
        record = json.loads(Path(result.evidence_reference['path']).read_text())
        self.assertEqual(record['native_record'], raw.native_record)
        self.assertEqual(record['normalized_result']['status'], 0)
        self.assertEqual(record['native_record']['raw_model_status'], 2)
        self.assertEqual(record['submitted_coefficient_sha256'],
                         coefficient_sha256(dict(model, objective_constant_usd=0.)))
        self.assertFalse(record['formal_admission'])
        self.assertFalse(record['scientific_acceptance'])
        self.assertEqual(record['recourse_columns'], 0)
        verify_reference(record['native_log_reference'])
        for source in record['source_references'].values():
            verify_reference(source)
        with patch.object(adapter.native_gurobi, 'solve_milp') as native:
            checked = adapter.verify_native_small_master_evidence(result.evidence_reference,
                expected_coefficient_sha256=record['submitted_coefficient_sha256'])
        native.assert_not_called()
        self.assertEqual(checked, record)

    def test_missing_bound_is_none_and_native_mem_limit_is_preserved(self):
        result, _ = self.invoke()
        record = json.loads(Path(result.evidence_reference['path']).read_text())
        self.assertEqual(record['normalized_result']['status'], 1)
        self.assertEqual(record['normalized_result']['limit_status'], 'MEM_LIMIT')
        self.assertEqual(record['native_record']['raw_model_status'], 17)
        self.assertIsNone(result.raw.mip_dual_bound)
        self.assertIsNone(record['normalized_result']['mip_dual_bound'])
        self.assertIsNone(record['native_record']['bound_source'])
        checked = adapter.verify_native_small_master_evidence(result.evidence_reference,
            expected_coefficient_sha256=record['submitted_coefficient_sha256'])
        self.assertEqual(checked['native_record']['status'], 'MEM_LIMIT')

    def test_original_numeric_failure_record_is_not_reinterpreted(self):
        raw = raw_result(normalized=4, native=9, native_name='TIME_LIMIT')
        raw.native_record.update(certificate_failures=['Rejected original coefficient certificate'],
                                 raw_solver_bound_without_constant_usd=-94.)
        raw.message += '; certificate failed: Rejected original coefficient certificate'
        result, _ = self.invoke(raw=raw)
        record = json.loads(Path(result.evidence_reference['path']).read_text())
        self.assertEqual(record['normalized_result']['status'], 4)
        self.assertEqual(record['native_record'], raw.native_record)
        self.assertIsNone(result.raw.mip_dual_bound)
        adapter.verify_native_small_master_evidence(result.evidence_reference,
            expected_coefficient_sha256=record['submitted_coefficient_sha256'])

    def test_preparation_and_native_loading_share_the_original_grant(self):
        with patch.object(adapter.time, 'monotonic', side_effect=[10., 10.3, 10.4, 11.4, 11.5, 11.7]):
            result, call = self.invoke()
        self.assertAlmostEqual(call.call_args.kwargs['options']['time_limit'], 4.7)
        self.assertAlmostEqual(result.native_elapsed_seconds, 1.)
        self.assertAlmostEqual(result.elapsed_seconds, 1.7)
        record = json.loads(Path(result.evidence_reference['path']).read_text())
        self.assertAlmostEqual(record['elapsed_before_evidence_seconds'], 1.5)

    def test_expired_preparation_never_enters_native_solver(self):
        with patch.object(adapter.time, 'monotonic', side_effect=[10., 16.]), \
                patch.object(adapter.native_gurobi, 'solve_milp') as native, \
                self.assertRaisesRegex(TimeoutError, 'preparation exhausted'):
            adapter.solve_native_small_master(small_master(), time_limit_seconds=5.,
                mip_relative_gap=.001, output_root=self.base / 'expired')
        native.assert_not_called()

    def test_resource_and_environment_rejections_do_not_load_solver(self):
        invalid = [dict(time_limit_seconds=601), dict(time_limit_seconds=True),
            dict(time_limit_seconds=float('nan')), dict(time_limit_seconds=0),
            dict(mip_relative_gap=.0011), dict(mip_relative_gap=-1),
            dict(threads=5), dict(threads=1.), dict(threads=True)]
        for i, changes in enumerate(invalid):
            with self.subTest(changes=changes), patch.object(adapter.native_gurobi, 'solve_milp') as native:
                with self.assertRaises(ValueError):
                    self.invoke(**changes, output_root=self.base / str(i))
                native.assert_not_called()
        with patch.dict(adapter.os.environ, {'OPENBLAS_NUM_THREADS': '2'}), \
                patch.object(adapter.native_gurobi, 'solve_milp') as native, \
                self.assertRaisesRegex(ValueError, 'single-threaded'):
            adapter.solve_native_small_master(small_master(), time_limit_seconds=1.,
                mip_relative_gap=.001, output_root=self.base / 'bad_blas')
        native.assert_not_called()

    def test_non_small_or_non_empirical_model_is_rejected(self):
        for key, value in [('recourse_columns', 1), ('theta_columns', 3),
                           ('cut_rows', 9), ('c', np.array([2., .4, .6])),
                           ('integrality', np.array([1, 1, 0])),
                           ('upper_bounds', np.array([1., 20., np.inf]))]:
            model = small_master(); model[key] = value
            with self.subTest(key=key), patch.object(adapter.native_gurobi, 'solve_milp') as native, \
                    self.assertRaises(ValueError):
                adapter.solve_native_small_master(model, time_limit_seconds=1.,
                    mip_relative_gap=.001, output_root=self.base / ('bad_' + key))
            native.assert_not_called()

    def test_evidence_overwrite_and_source_output_are_rejected(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.invoke()
        with patch.object(adapter.native_gurobi, 'solve_milp') as native, \
                self.assertRaisesRegex(ValueError, 'frozen/source'):
            adapter.solve_native_small_master(small_master(), time_limit_seconds=1.,
                mip_relative_gap=.001, output_root=Path(adapter.__file__).resolve().parent / 'bad_native')
        native.assert_not_called()

    def test_native_license_failure_propagates_without_fallback(self):
        with patch.object(adapter.native_gurobi, 'solve_milp', side_effect=RuntimeError('license unavailable')) as native, \
                self.assertRaisesRegex(RuntimeError, 'license unavailable'):
            adapter.solve_native_small_master(small_master(), time_limit_seconds=1.,
                mip_relative_gap=.001, output_root=self.base / 'failed')
        native.assert_called_once()
        self.assertFalse((self.base / 'failed' / 'native_execution.json').exists())

    def test_evidence_pin_log_source_and_expected_matrix_are_required(self):
        result, _ = self.invoke()
        path = Path(result.evidence_reference['path'])
        record = json.loads(path.read_text())
        coefficient = record['submitted_coefficient_sha256']
        with self.assertRaisesRegex(ValueError, 'coefficients differ'):
            adapter.verify_native_small_master_evidence(result.evidence_reference,
                expected_coefficient_sha256='0' * 64)
        original = path.read_bytes()
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(ValueError):
            adapter.verify_native_small_master_evidence(result.evidence_reference,
                expected_coefficient_sha256=coefficient)
        path.write_bytes(original)
        log = Path(record['native_log_reference']['path'])
        log.write_text(log.read_text() + 'changed')
        with self.assertRaises(ValueError):
            adapter.verify_native_small_master_evidence(result.evidence_reference,
                expected_coefficient_sha256=coefficient)

    def test_rehashed_status_bound_source_and_resource_forgery_is_rejected(self):
        cases = [('status', ('normalized_result', 'status'), 0),
                 ('renamed limit', ('normalized_result', 'limit_status'), 'TIME_LIMIT'),
                 ('invented bound', ('normalized_result', 'mip_dual_bound'), 0.),
                 ('reset loading', ('native_record', 'load_seconds'), 0.),
                 ('threads', ('native_record', 'effective_options', 'Threads'), 5),
                 ('source', ('source_references', 'native_gurobi.py', 'sha256'), '0' * 64),
                 ('matrix', ('native_record', 'original_matrix_shape'), [2, 99]),
                 ('version', ('native_record', 'gurobi_version'), [12, 0, 2])]
        for i, (name, keys, value) in enumerate(cases):
            with self.subTest(name=name):
                result, _ = self.invoke(output_root=self.base / ('forgery_' + str(i)))
                path = Path(result.evidence_reference['path']); body = json.loads(path.read_text())
                target = body
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path.write_text(json.dumps(body))
                with patch.object(adapter.native_gurobi, 'solve_milp') as native, self.assertRaises(ValueError):
                    adapter.verify_native_small_master_evidence(file_reference(path),
                        expected_coefficient_sha256=body['submitted_coefficient_sha256'])
                native.assert_not_called()

    def test_exact_loading_budget_stop_keeps_raw_loaded_status_and_no_bound(self):
        raw = OptimizeResult(status=1, success=False,
            message='Gurobi native: model loading exhausted time budget',
            x=None, fun=None, mip_dual_bound=None, mip_gap=None, mip_node_count=None,
            limit_status='TIME_LIMIT', native_record=dict(backend='gurobipy', gurobi_version=[12, 0, 3],
                raw_model_status=1, status='MODEL_LOADING_TIME_LIMIT', valid_incumbent=False,
                solver_run_seconds=0., load_seconds=10.))
        result, _ = self.invoke(raw=raw)
        path = Path(result.evidence_reference['path']); body = json.loads(path.read_text())
        checked = adapter.verify_native_small_master_evidence(result.evidence_reference,
            expected_coefficient_sha256=body['submitted_coefficient_sha256'])
        self.assertEqual(checked['native_record']['raw_model_status'], 1)
        self.assertNotIn('certificate_failures', checked['native_record'])
        self.assertNotIn('TimeLimit', checked['native_record']['effective_options'])
        for section, key, value in [('native_record', 'raw_model_status', 17),
                                    ('native_record', 'solver_run_seconds', 1.),
                                    ('native_record', 'valid_incumbent', True),
                                    ('normalized_result', 'mip_node_count', 0.)]:
            with self.subTest(key=key):
                forged = json.loads(json.dumps(body)); forged[section][key] = value
                path.write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError, 'Loading-time stop'):
                    adapter.verify_native_small_master_evidence(file_reference(path),
                        expected_coefficient_sha256=body['submitted_coefficient_sha256'])


if __name__ == '__main__':
    unittest.main()
