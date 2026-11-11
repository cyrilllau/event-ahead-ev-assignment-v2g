"""Fail-closed provenance and shared-cost checks for the saved RMP probe."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evcs_experiments.config import get_paths

_path = get_paths().project_root / 'scripts/diagnostics/run_s05_relaxation_bound.py'
_spec = importlib.util.spec_from_file_location('saved_rmp_bound_driver', _path)
d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(d)


class BoundProbeBudgetTest(unittest.TestCase):
    def setUp(self):
        self.req = dict(kind=d.KIND, plan_id='test-plan', wall_seconds=590,
                        solver_seconds=550, parent_job=dict(timeout_seconds=600))
        self.settled = dict(status='SETTLED_UB_ONLY_PENDING_INDEPENDENT_REVIEW',
            plan_id='test-plan', cumulative_seconds_provisional=False, nested_seconds_added=False,
            cumulative_accounted_seconds=3390.457493,
            opening_budget=dict(opening_cumulative_seconds=3368.778731), terminal_parent_seconds=21.678762,
            remaining_seconds=3809.542507, cumulative_cap_seconds=7200)

    def test_actual_settled_parent_preserves_remaining_final_reserve(self):
        self.assertEqual(d.budget_check(self.req, self.settled), 3390.457493)

    def test_nested_clock_cannot_be_charged_twice(self):
        self.settled['nested_seconds_added'] = True
        with self.assertRaises(ValueError): d.budget_check(self.req, self.settled)

    def test_provisional_cost_is_not_a_settled_predecessor(self):
        self.settled['cumulative_seconds_provisional'] = True
        with self.assertRaises(ValueError): d.budget_check(self.req, self.settled)

    def test_different_plan_or_reset_rejected(self):
        for key, value in [('plan_id', 'another-plan'), ('cumulative_accounted_seconds', 0),
                           ('cumulative_cap_seconds', 8000), ('terminal_parent_seconds', float('nan'))]:
            with self.subTest(key=key):
                other = copy.deepcopy(self.settled); other[key] = value
                with self.assertRaises(ValueError): d.budget_check(self.req, other)

    def test_final_evidence_reserve_cannot_be_consumed(self):
        self.settled.update(cumulative_accounted_seconds=6000, remaining_seconds=1200)
        self.settled['opening_budget']['opening_cumulative_seconds'] = 6000 - 21.678762
        with self.assertRaises(ValueError): d.budget_check(self.req, self.settled)

    def test_solver_or_parent_limit_cannot_expand_probe(self):
        for key, value in [('solver_seconds', 590), ('wall_seconds', 7200)]:
            with self.subTest(key=key):
                other = copy.deepcopy(self.req); other[key] = value
                with self.assertRaises(ValueError): d.budget_check(other, self.settled)


class BoundProbeReferenceTest(unittest.TestCase):
    def test_modified_original_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'; path.write_text('{}\n')
            ref = d.reference(path); self.assertEqual(d.verify(ref), path.resolve())
            path.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError): d.verify(ref)

    def test_source_manifest_escape_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'release'; root.mkdir()
            external = Path(directory) / 'outside.py'; external.write_text('pass\n')
            f = root / 'code.py'; f.write_text('pass\n'); fr = d.reference(f)
            manifest = root / 'RELEASE_MANIFEST.json'
            row = dict(path='code.py', sha256=fr['sha256'], size_bytes=fr['size_bytes'])
            for rows in [[dict(row, path='../outside.py')], [row, row]]:
                manifest.write_text(json.dumps(dict(files=rows)))
                with self.assertRaises(ValueError): d.source_check(d.reference(manifest), root)

    def test_ordinary_cli_cannot_launch_unsupervised_scale_solver(self):
        req = dict(parent_job=dict(job_id='not-a-real-job',
                   directory=str(get_paths().campaign_root / 'jobs' / 'not-a-real-job')))
        with self.assertRaises(ValueError): d.bind_parent(req, {})

    def test_evidence_files_are_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'result.json'; d.write(p, dict(value=1))
            with self.assertRaises(FileExistsError): d.write(p, dict(value=2))
            self.assertEqual(d.read(p), dict(value=1))


if __name__ == '__main__': unittest.main()
