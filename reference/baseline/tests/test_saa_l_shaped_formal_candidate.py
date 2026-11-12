"""Negative formal ingress checks; no physical/Gurobi/formal success claim.

The existing genuine development numerical regression is run separately because
its shared candidate/readback path changed. These cases ensure the new formal
entry cannot relabel toy data, a solver double, or the old extensive dispatcher.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evcs_experiments import core_request_adapter as adapter
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_candidate as candidate
from evcs_experiments import saa_l_shaped_formal as formal
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.config import get_paths
from evcs_experiments.validation.s03_toys import build_golden_cases
import test_core_request_adapter as request_fixtures


class FormalCandidateGuardTests(unittest.TestCase):
    def test_formal_writer_cannot_relabel_original_synthetic_problem(self):
        toy = build_golden_cases()[0]
        problem = kernel.DevelopmentSAAProblem(toy.case, toy.tasks, toy.scenarios,
                                                toy.scenario_cost_constants_usd)
        with self.assertRaises(ValueError):
            candidate._bound_candidate(problem, 'risk_model.helene_2024.SAA.baseline',
                                       {}, domain='formal')
        self.assertEqual(problem.domain, 'development')
        self.assertEqual(problem.identity['algorithm_id'], kernel.ALGORITHM_ID)

    def test_formal_metadata_requires_formal_request_and_typed_joint(self):
        for request in (
            dict(domain='development', method_id='SAA', model_input_mode='l_shaped_development',
                 algorithm_id=kernel.ALGORITHM_ID),
            dict(domain='formal', method_id='SAA', model_input_mode='assemble',
                 algorithm_id=formal.FORMAL_ALGORITHM_ID),
            dict(domain='formal', method_id='ARO', model_input_mode='l_shaped_formal',
                 algorithm_id=formal.FORMAL_ALGORITHM_ID)):
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, 'Explicit formal'):
                candidate.validate_formal_joint({}, request, {})
        request = dict(domain='formal', method_id='SAA', model_input_mode='l_shaped_formal',
                       algorithm_id=formal.FORMAL_ALGORITHM_ID)
        with self.assertRaisesRegex(ValueError, 'Typed L-shaped'):
            candidate.validate_formal_joint(dict(kind=candidate.AUDIT, passed=True), request, {})

    def test_native_test_double_is_not_a_formal_solver_certificate(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            body = dict(native_record=dict(backend='gurobipy', gurobi_version=[12, 0, 3],
                                           synthetic_backend_test_double=True))
            path = root/'actual_test_double.json'
            path.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, 'test doubles'):
                candidate._formal_native_guard(dict(native_adapter_reference=file_reference(path)))
        with self.assertRaisesRegex(ValueError, 'adapter evidence'):
            candidate._formal_native_guard({})

    def test_preflight_output_is_bound_to_original_request_without_a_hash_cycle(self):
        # This is a narrow API handoff test. The physical preflight verifier is
        # explicitly mocked; no formal physical/numerical success is claimed.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            helper = file_reference(get_paths().project_root/'scripts/audits/audit_s05_pending_recovery.py')
            request = dict(domain='formal', method_id='SAA', plan_id='handoff.only', helper_reference=helper)
            binding = {'binding': 1}
            path = root/'worker_request.json'; path.write_text(json.dumps(request))
            request_ref = file_reference(path)
            preflight = root/'preflight.json'; preflight.write_text(json.dumps({'worker_request_reference': request_ref}))
            pref = file_reference(preflight)
            values = dict(formal_input_reference=pref, input_binding=binding, helper_reference=helper)
            record = dict(request_reference=request_ref)
            checked = dict(worker_request_reference=request_ref, input_binding=binding)
            self.assertNotIn('formal_input_reference', request)
            self.assertNotIn('input_binding', request)
            with patch.object(formal, 'validate_formal_input_reference', return_value=checked) as call:
                candidate._formal_physical_readback(request, record, values, values, values)
                call.assert_called_once_with(pref, expected_request=request_ref, expected_identity=values)
            with patch.object(formal, 'validate_formal_input_reference',
                              return_value=dict(checked, input_binding={'foreign': True})):
                with self.assertRaisesRegex(ValueError, 'physical binding differs'):
                    candidate._formal_physical_readback(request, record, values, values, values)
            explicit = dict(request, input_binding={'foreign': True})
            explicit_path = root/'explicit_request.json'; explicit_path.write_text(json.dumps(explicit))
            explicit_ref = file_reference(explicit_path)
            with patch.object(formal, 'validate_formal_input_reference',
                              return_value=dict(checked, worker_request_reference=explicit_ref)):
                with self.assertRaisesRegex(ValueError, 'Explicit formal request input binding differs'):
                    candidate._formal_physical_readback(explicit, dict(request_reference=explicit_ref), values, values, values)
            with self.assertRaisesRegex(ValueError, 'exact original worker request'):
                candidate._formal_physical_readback(dict(request, plan_id='foreign'), record, values, values, values)
            with self.assertRaisesRegex(ValueError, 'KKT helper binding differs'):
                candidate._formal_physical_readback(request, record, values, dict(values, helper_reference=request_ref), values)
            with self.assertRaisesRegex(ValueError, 'preflight reference differs'):
                candidate._formal_physical_readback(request, record, values, dict(values, formal_input_reference=request_ref), values)

    def test_no_generic_domain_override_of_development_entry(self):
        with self.assertRaises(TypeError):
            candidate.audit_development_candidate(None, domain='formal')
        with self.assertRaises(ValueError):
            candidate._domain_contract('formal-ish')


class FormalRequestDispatchGuardTests(unittest.TestCase):
    def test_new_mode_cannot_be_sent_to_old_extensive_worker(self):
        fixture = request_fixtures.CoreRequestAdapterTests()
        fixture.setUp()
        try:
            options = dict(fixture.saa, model_input_mode='l_shaped_formal',
                           algorithm_id=formal.FORMAL_ALGORITHM_ID)
            with self.assertRaisesRegex(ValueError, 'extensive dispatch prohibited'):
                fixture.worker('SAA', saa_options=options)
            # An externally supplied mode mutation must also fail at queue
            # construction, not silently use the registry's extensive module.
            request = fixture.worker('SAA')
            request.update(model_input_mode='l_shaped_formal', algorithm_id=formal.FORMAL_ALGORITHM_ID)
            item = dict(item_id='SAA', identity=fixture.identity('SAA'),
                        request_reference=fixture.write('blocked-request', request))
            with self.assertRaisesRegex(ValueError, 'extensive dispatch prohibited'):
                fixture.queue([item])
        finally:
            fixture.doCleanups()
