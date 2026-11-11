"""Revision records never replace original bytes or exempt arbitrary kernels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evcs_experiments import evidence


class KernelRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.active = self.root / 'current'; self.active.mkdir()
        self.frozen = self.root / 'frozen' / 'evcs_experiments'; self.frozen.mkdir(parents=True)
        for name in ('wind_scenarios.py', 'random_streams.py'):
            (self.frozen / name).write_text(f'original {name}\n')
            (self.active / name).write_bytes((self.frozen / name).read_bytes())
        (self.active / 'wind_scenarios.py').write_text('original physical formulas plus scoped OOS gate\n')
        self.path = self.root / 'provenance/source_revisions/s04_oos_access.json'
        self.path.parent.mkdir(parents=True)
        self.files = {'wind_scenarios.py': self.binding('wind_scenarios.py')}

    def binding(self, name):
        return dict(original_sha256=evidence.sha256(self.frozen / name),
            current_sha256=evidence.sha256(self.active / name))

    def verify(self, files):
        self.path.write_text(json.dumps(dict(files=files)))
        with patch.object(evidence, '__file__', str(self.active / 'evidence.py')):
            with patch.object(evidence, 'get_paths', return_value=SimpleNamespace(project_root=self.root)):
                with patch.object(evidence, 'frozen_runner_root', return_value=self.frozen.parent):
                    return evidence.verify_scientific_kernel_identity()

    def test_exact_gate_revision_keeps_both_sources_and_other_kernel_strict(self):
        original = (self.frozen / 'wind_scenarios.py').read_bytes()
        self.assertEqual(self.verify(self.files), 2)
        self.assertEqual((self.frozen / 'wind_scenarios.py').read_bytes(), original)
        self.assertNotEqual(evidence.sha256(self.frozen / 'wind_scenarios.py'),
            evidence.sha256(self.active / 'wind_scenarios.py'))

    def test_old_or_new_record_digest_tampering_is_rejected(self):
        for key in ('original_sha256', 'current_sha256'):
            files = deepcopy(self.files); files['wind_scenarios.py'][key] = '0' * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(files)

    def test_actual_old_or_new_source_tampering_is_rejected(self):
        for directory in (self.active, self.frozen):
            path = directory / 'wind_scenarios.py'; original = path.read_bytes()
            path.write_bytes(original + b'changed\n')
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                self.verify(self.files)
            path.write_bytes(original)

    def test_missing_registration_never_exempts_changed_gate(self):
        with self.assertRaises(ValueError):
            self.verify({})

    def test_registration_cannot_authorize_non_gate_kernel(self):
        (self.active / 'random_streams.py').write_text('changed random addresses\n')
        files = dict(self.files, **{'random_streams.py': self.binding('random_streams.py')})
        with self.assertRaises(ValueError):
            self.verify(files)

    def test_extra_unchanged_kernel_registration_is_not_ignored(self):
        files = dict(self.files, **{'random_streams.py': self.binding('random_streams.py')})
        with self.assertRaises(ValueError):
            self.verify(files)

    def test_record_is_checked_even_if_gate_has_been_reverted_to_original(self):
        (self.active / 'wind_scenarios.py').write_bytes((self.frozen / 'wind_scenarios.py').read_bytes())
        with self.assertRaises(ValueError):
            self.verify(self.files)


if __name__ == '__main__':
    unittest.main()
