#!/usr/bin/env python3
"""Second-scale integration tests; each test uses a private temporary campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from evcs_experiments.runtime import jobctl



class JobControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="evcs-jobctl-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, job_id, code, **settings):
        request = {"job_id": job_id, "argv": [sys.executable, "-c", code],
                   "cwd": str(self.root), "outroot": str(self.root / "jobs"),
                   "threads": 2, "timeout_seconds": 4, "grace_seconds": 0.15, **settings}
        return subprocess.run([sys.executable, "-m", "evcs_experiments.runtime.jobctl", "launch", "--request", "-"],
                              input=json.dumps(request), capture_output=True, text=True, timeout=4)

    def finished(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            path = self.root / "jobs" / job_id / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result.get("complete"):
                    return result
            time.sleep(0.05)
        self.fail(f"Job {job_id} did not finish")

    def test_success_detached_environment_and_audit(self):
        launched = self.launch("success", "import os; print(os.environ['OMP_NUM_THREADS']); print('literal;$(echo NO)')")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        result = self.finished("success")
        self.assertEqual(result["state"], "SUCCEEDED")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["supervisor_sid"], result["supervisor_pid"])
        self.assertEqual(result["command_process"]["sid"], result["supervisor_sid"])
        self.assertEqual(result["command_process"]["pgid"], result["command_pid"])
        self.assertNotEqual(result["command_pid"], result["supervisor_pid"])
        jobdir = self.root / "jobs" / "success"
        self.assertEqual((jobdir / "stdout.log").read_text(), "2\nliteral;$(echo NO)\n")
        self.assertTrue(all((jobdir / name).exists() for name in
                            ("manifest.json", "status.json", "stdout.log", "stderr.log", "result.json")))
        observed = jobctl.status(self.root / "jobs", "success")["jobs"][0]["observation"]
        self.assertEqual(observed["health"], "COMPLETE")

    def test_nonzero_exit(self):
        launched = self.launch("nonzero", "import sys; print('expected failure', file=sys.stderr); sys.exit(7)")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        result = self.finished("nonzero")
        self.assertEqual((result["state"], result["returncode"]), ("FAILED", 7))
        self.assertIn("expected failure", (self.root / "jobs" / "nonzero" / "stderr.log").read_text())

    def test_duplicate_and_one_active_job(self):
        first = self.launch("duplicate", "import time; time.sleep(0.8)")
        self.assertEqual(first.returncode, 0, first.stderr)
        observed = jobctl.status(self.root / "jobs", "duplicate")["jobs"][0]["observation"]
        self.assertEqual(observed["health"], "SUPERVISED")
        duplicate = self.launch("duplicate", "print('must not run')")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("already exists", duplicate.stderr)
        second = self.launch("second", "print('must not run')")
        self.assertEqual(second.returncode, 2)
        self.assertIn("One active", second.stderr)
        self.assertFalse((self.root / "jobs" / "second").exists())
        self.assertEqual(self.finished("duplicate")["state"], "SUCCEEDED")
        self.assertEqual(self.launch("duplicate", "print('still must not run')").returncode, 2)

    def test_simultaneous_launches_are_serialized(self):
        request = {"job_id": "race", "argv": [sys.executable, "-c", "import time; time.sleep(0.3)"],
                   "cwd": str(self.root), "outroot": str(self.root / "jobs"), "timeout_seconds": 3}
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(request))
        argv = [sys.executable, "-m", "evcs_experiments.runtime.jobctl", "launch", "--request", str(request_path)]
        processes = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(2)]
        outputs = [process.communicate(timeout=4) for process in processes]
        self.assertEqual(sorted(process.returncode for process in processes), [0, 2], outputs)
        self.assertEqual(self.finished("race")["state"], "SUCCEEDED")

    def test_timeout_kills_owned_process_tree(self):
        child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        parent_code = (
            "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); print(p.pid,flush=True); time.sleep(30)"
        )
        launched = self.launch("timeout", parent_code, timeout_seconds=0.6)
        self.assertEqual(launched.returncode, 0, launched.stderr)
        result = self.finished("timeout")
        self.assertEqual(result["state"], "TIMED_OUT")
        self.assertEqual([entry["signal"] for entry in result["termination_actions"]], ["SIGTERM", "SIGKILL"])
        self.assertLess(result["elapsed_seconds"], 3)
        self.assertFalse(jobctl.identity_alive(result["command_process"]))
        deadline = time.monotonic() + 2
        while jobctl.group_members(result["command_pgid"], result["supervisor_sid"]) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(jobctl.group_members(result["command_pgid"], result["supervisor_sid"]), [])
        self.assertTrue(all(entry["sid"] == result["supervisor_pid"] for entry in result["termination_actions"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
