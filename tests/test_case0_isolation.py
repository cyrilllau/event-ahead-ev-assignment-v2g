import unittest
from unittest.mock import patch
from evcs_experiments.case0_isolation import isolated_command, unit_name


class Case0IsolationTests(unittest.TestCase):
    def test_only_case0_child_units_are_controllable(self):
        for value in ("user@1006.service", "case1-job", "case0-/../../x", ""):
            with self.assertRaises(ValueError):
                unit_name(value)
        self.assertEqual(unit_name("case0-test"), "case0-test.service")

    def test_effective_limits_belong_to_child_service(self):
        with patch.dict("os.environ", {"EVCS_CONFIG": "/tmp/case0.toml"}, clear=True):
            cmd = isolated_command("case0-test", ["python", "-m", "example"], 1800, 255)
        for property in ("MemoryMax=8G", "MemorySwapMax=0", "CPUQuota=100%", "CPUAffinity=255",
                         "Nice=10", "KillMode=control-group", "RuntimeMaxSec=1800"):
            self.assertIn(property, cmd)
        self.assertIn("--wait", cmd)
        self.assertIn("EVCS_CONFIG=/tmp/case0.toml", cmd)
        self.assertNotIn("user.slice", cmd)


if __name__ == "__main__":
    unittest.main()
