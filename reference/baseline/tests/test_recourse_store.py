"""Real physical compile/export/restore checks for original numeric templates."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import save_npz

import evcs_experiments.recourse as recourse
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.recourse_store import ARRAYS, MATRICES, bind_template, load_template
from recourse_fixtures import golden_single_bus, golden_islands


class RecourseStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def prepare(self, fixture=None):
        fixture = golden_single_bus() if fixture is None else fixture
        self.fixture = fixture
        self.original = recourse.assemble_recourse(fixture.case, fixture.tasks, fixture.scenario)
        first = self.original.first_stage
        problem = SimpleNamespace(first_stage=first, identity={'fixture': 'numeric-store'},
            probability_set=SimpleNamespace(support_ids=(fixture.scenario.scenario_id,)),
            recourses=SimpleNamespace(scenario_sha256s=(fixture.scenario.scenario_sha256,)))
        evidence = RawPlanningEvidence(self.root / 'raw', fixture.case, fixture.tasks, problem,
            lambda index: fixture.scenario, domain='development', input_binding={'fixture': True})
        self.record = evidence.template(0, self.original)
        self.binding = bind_template(self.record, first, fixture.case, fixture.tasks, fixture.scenario,
            source_references={'original_request': file_reference(evidence.root / 'request.json')},
            compiler_reference=file_reference(recourse.__file__), output=self.root / 'binding.json')
        return fixture

    def load(self, *, fixture=None, first=None, **overrides):
        fixture = self.fixture if fixture is None else fixture
        arguments = dict(expected_template_sha256=self.original.template_sha256,
            expected_scenario_index=0, expected_scenario_id=fixture.scenario.scenario_id)
        arguments.update(overrides)
        return load_template(self.binding, self.original.first_stage if first is None else first,
            fixture.case, fixture.tasks, fixture.scenario, **arguments)

    def assert_exact_numerics(self, restored):
        for name in MATRICES:
            old, new = getattr(self.original, name), getattr(restored, name)
            self.assertEqual(old.shape, new.shape)
            for field in ('data', 'indices', 'indptr'):
                np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
                with self.assertRaises(ValueError):
                    getattr(new, field).setflags(write=True)
        for name in (*ARRAYS, 'load_p_kw', 'arrival_soc'):
            np.testing.assert_array_equal(getattr(self.original, name), getattr(restored, name))
            with self.assertRaises(ValueError):
                getattr(restored, name).setflags(write=True)
        for name in ('template_sha256', 'reference_bus_ids_by_period', 'station_task_indices',
                     'cmax_usd', 'u_ref', 'task_ids', 'station_ids', 'bus_ids', 'line_ids',
                     'periods', 'variable_slices', 'variable_shapes', 'units'):
            self.assertEqual(getattr(self.original, name), getattr(restored, name))

    def test_two_physical_goldens_restore_exact_numerics_and_lazy_maps(self):
        for name, fixture in (('single', golden_single_bus()), ('islands', golden_islands())):
            with self.subTest(name=name):
                self.root = self.root / name
                self.root.mkdir()
                self.prepare(fixture)
                restored = self.load()
                self.assertFalse(restored.descriptive_maps_loaded)
                self.assert_exact_numerics(restored)
                self.assertEqual(len(restored.variable_map), len(self.original.variable_map))
                self.assertFalse(restored.descriptive_maps_loaded)
                self.assertEqual(restored.as_dict(), self.original.as_dict())
                self.assertTrue(restored.descriptive_maps_loaded)
                for map_name in ('variable_map', 'constraint_map_ub', 'constraint_map_eq'):
                    self.assertEqual(tuple(getattr(restored, map_name)), getattr(self.original, map_name))
                self.assertEqual(file_reference(restored.original_identity_reference['path']),
                                 restored.original_identity_reference)

    def test_two_distinct_plans_get_new_rhs_lp_and_physical_certificates(self):
        for name, fixture in (('single', golden_single_bus()), ('islands', golden_islands())):
            with self.subTest(name=name):
                self.root = self.root / name
                self.root.mkdir()
                self.prepare(fixture)
                restored = self.load()
                zero = FirstStagePlan(np.zeros_like(fixture.plan.x), np.zeros_like(fixture.plan.r),
                    fixture.tasks.task_sha256, fixture.tasks.config_hash)
                objectives = []
                for plan in (fixture.plan, zero):
                    for old, new in zip(self.original.fixed_rhs(plan), restored.fixed_rhs(plan)):
                        np.testing.assert_array_equal(old, new)
                    original_fallback = recourse.full_shedding_fallback(self.original, plan)
                    fallback = recourse.full_shedding_fallback(restored, plan)
                    np.testing.assert_array_equal(self.original.pack(original_fallback), restored.pack(fallback))
                    np.testing.assert_array_equal(restored.pack(restored.unpack(restored.pack(fallback))),
                                                  restored.pack(fallback))
                    self.assertEqual(self.original.objective_components(original_fallback),
                                     restored.objective_components(fallback))
                    self.assertTrue(check_recourse(fixture.case, fixture.tasks, fixture.scenario, plan, fallback)['passed'])
                    original_result = recourse.solve_fixed_plan(self.original, plan, time_limit_seconds=5)
                    result = recourse.solve_fixed_plan(restored, plan, time_limit_seconds=5)
                    self.assertTrue(original_result.optimal and result.optimal)
                    self.assertTrue(result.certificate['optimal_certificate_valid'])
                    self.assertEqual(result.objective_usd, original_result.objective_usd)
                    self.assertEqual(result.objective_components, original_result.objective_components)
                    np.testing.assert_array_equal(restored.pack(result.solution), self.original.pack(original_result.solution))
                    self.assertTrue(check_recourse(fixture.case, fixture.tasks, fixture.scenario, plan,
                                                  result.solution, check_objective_bound=True)['passed'])
                    objectives.append(result.objective_usd)
                self.assertNotEqual(*objectives)
                self.assertFalse(restored.descriptive_maps_loaded)

    def test_actual_soc_route_reactive_and_initial_state_changes_refuse_stale_identity(self):
        self.prepare()
        changes = (
            lambda f: f.scenario.task_arrival_soc.__setitem__(0, .7),
            lambda f: f.scenario.task_gamma_route.__setitem__(0, False),
            lambda f: f.case.station_rating_kva.__setitem__(0, 374.),
            lambda f: f.tasks.forecast_arrival_soc.__setitem__(0, .7),
            lambda f: f.tasks.config.__setitem__('route', 'another-fixed-route'),
        )
        for change in changes:
            fixture = deepcopy(self.fixture)
            change(fixture)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Actual physical'):
                self.load(fixture=fixture)
        changed_first = replace(self.original.first_stage, c=self.original.first_stage.c + 1)
        with self.assertRaisesRegex(ValueError, 'Actual physical'):
            self.load(first=changed_first)

    def test_wrong_support_or_first_stage_or_template_refused(self):
        self.prepare()
        for args in ({'expected_scenario_index': 1}, {'expected_scenario_index': True},
                     {'expected_scenario_id': 'foreign'}, {'expected_template_sha256': '0' * 64}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'identity differs'):
                self.load(**args)
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.load(first=replace(self.original.first_stage, matrix_sha256='0' * 64))

    def test_same_size_artifact_tampering_is_rejected(self):
        self.prepare()
        path = Path(self.record['path']).parent / 'A_ub.npz'
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.load()

    def test_changed_source_and_late_identity_tampering_are_rejected(self):
        self.prepare()
        restored = self.load()
        identity = Path(restored.original_identity_reference['path'])
        raw = identity.read_text()
        self.assertIn('certification_procurement', raw)
        identity.write_text(raw.replace('certification_procurement', 'xertification_procurement'))
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            restored.as_dict()
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.load()

    def test_uncommitted_extra_and_missing_artifact_are_rejected(self):
        self.prepare()
        extra = Path(self.record['path']).parent / 'partial.json'
        extra.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'uncommitted'):
            self.load()
        extra.unlink()
        (extra.parent / 'A_eq.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'uncommitted'):
            self.load()

    def test_resealed_changed_numerics_cannot_replace_expected_semantic_sha(self):
        self.prepare()
        folder = Path(self.record['path']).parent
        altered = self.original.A_ub.copy()
        altered.data[0] += .125
        save_npz(folder / 'A_ub.npz', altered)
        record = json.loads(Path(self.record['path']).read_text())
        record['artifacts'] = [file_reference(ref['path']) for ref in record['artifacts']]
        Path(self.record['path']).write_text(json.dumps(record))
        binding = json.loads(Path(self.binding['path']).read_text())
        binding['original_template_reference'] = file_reference(self.record['path'])
        Path(self.binding['path']).write_text(json.dumps(binding))
        self.binding = file_reference(self.binding['path'])
        with self.assertRaisesRegex(ValueError, 'semantic SHA differs'):
            self.load()

    def test_resealed_extra_zero_csr_column_is_rejected_despite_old_semantic_digest(self):
        self.prepare()
        folder = Path(self.record['path']).parent
        altered = self.original.A_ub.copy()
        altered.resize((altered.shape[0], altered.shape[1] + 1))
        save_npz(folder / 'A_ub.npz', altered)
        record = json.loads(Path(self.record['path']).read_text())
        record['artifacts'] = [file_reference(ref['path']) for ref in record['artifacts']]
        Path(self.record['path']).write_text(json.dumps(record))
        binding = json.loads(Path(self.binding['path']).read_text())
        binding['original_template_reference'] = file_reference(self.record['path'])
        Path(self.binding['path']).write_text(json.dumps(binding))
        self.binding = file_reference(self.binding['path'])
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            self.load()

    def test_binding_refuses_first_stage_with_stale_semantic_sha(self):
        self.prepare()
        body = json.loads(Path(self.binding['path']).read_text())
        wrong = replace(self.original.first_stage, c=self.original.first_stage.c + 1)
        with self.assertRaisesRegex(ValueError, 'first-stage semantic SHA differs'):
            bind_template(self.record, wrong, self.fixture.case, self.fixture.tasks, self.fixture.scenario,
                source_references=body['source_references'], compiler_reference=body['compiler_reference'],
                output=self.root / 'wrong-binding.json')

    def test_binding_is_append_only_and_is_not_scientific_acceptance(self):
        self.prepare()
        body = json.loads(Path(self.binding['path']).read_text())
        self.assertFalse(body['scientific_acceptance'])
        self.assertFalse(body['formal_OOS'])
        self.assertFalse(body['old_q_reused'])
        with self.assertRaises(FileExistsError):
            bind_template(self.record, self.original.first_stage, self.fixture.case, self.fixture.tasks,
                self.fixture.scenario, source_references=body['source_references'],
                compiler_reference=body['compiler_reference'], output=self.binding['path'])


if __name__ == '__main__':
    unittest.main()
