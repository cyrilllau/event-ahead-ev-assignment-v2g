"""Preparation reservation survives a real process exit; terminal parent charges once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from evcs_experiments import case0_evaluation_runner as m


DEAD=dict(blocks_new_launch=False,supervisor_alive=False,command_alive=False,live_group_members=[])

class Case0EvaluationAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.job=self.root/'jobs/test-parent';self.job.mkdir(parents=True)
        m._write(self.root/'config.lock.json',dict(total_compute_seconds=21600))
        m._write(self.root/'state.json',dict(active_configuration=None,compute_seconds=100.,status='PLANNING_REVIEW_REQUIRED'))
        m._write(self.job/'manifest.json',dict(job_id='test-parent',timeout_seconds=100.,grace_seconds=5.))
        m._write(self.job/'status.json',dict(state='SUCCEEDED'))
        self.result=m._write(self.job/'result.json',dict(job_id='test-parent',state='SUCCEEDED',complete=True,elapsed_seconds=12.))
        self.env=dict(EVCS_JOB_DIRECTORY=str(self.job),EVCS_JOB_ID='test-parent')

    def settle(self):
        with patch('evcs_experiments.runtime.jobctl.observation',return_value=DEAD):
            return m.settle_case0_evaluation_attempt(root=self.root,terminal_result_reference=self.result)

    def test_real_hard_exit_preserves_preparation_reservation_and_charges_parent_once(self):
        script='''import os
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import case0_evaluation_runner as m
with patch.object(m,'campaign_root',return_value=Path(os.environ['TEST_ROOT'])), patch('evcs_experiments.case0_isolation.verify_current_cgroup'), patch.object(m,'prepare_case0_evaluation_lock',side_effect=lambda **kw:os._exit(99)):
 m.run_case0_evaluation(acceptance_reference={},seconds=100)
'''
        child=subprocess.run([sys.executable,'-c',script],env={**os.environ,**self.env,'TEST_ROOT':str(self.root)},capture_output=True,timeout=15)
        self.assertEqual(child.returncode,99,child.stderr.decode())
        state=m.read(self.root/'state.json');self.assertEqual(state['active_configuration'],'CASE0_EVALUATION:PREPARATION')
        self.assertTrue(state['pending_evaluation_attempt']);self.assertEqual(state['compute_seconds'],100.)
        with patch.object(m,'campaign_root',return_value=self.root),patch('evcs_experiments.case0_isolation.verify_current_cgroup'):
            with self.assertRaisesRegex(ValueError,'Unsettled'):m.run_case0_evaluation(acceptance_reference={},seconds=100)
        ref=self.settle();self.assertEqual(m.read(self.root/'state.json')['compute_seconds'],112.)
        self.assertEqual(self.settle(),ref);self.assertEqual(m.read(self.root/'state.json')['compute_seconds'],112.)
        self.assertEqual(m.read(self.root/'state.json')['status'],'CASE0_EVALUATION_INTERRUPTED_REQUIRES_DIAGNOSIS')

    def test_normal_failed_preparation_provisional_time_is_not_double_charged(self):
        with patch.dict(os.environ,self.env),patch.object(m,'campaign_root',return_value=self.root),patch('evcs_experiments.case0_isolation.verify_current_cgroup'),patch.object(m,'prepare_case0_evaluation_lock',side_effect=ValueError('audit rejected')):
            with self.assertRaisesRegex(ValueError,'audit rejected'):m.run_case0_evaluation(acceptance_reference={},seconds=100)
        self.assertGreater(m.read(self.root/'state.json')['compute_seconds'],100.)
        ref=self.settle();receipt=m._read(ref)
        self.assertGreater(receipt['provisional_seconds_already_charged'],0.)
        self.assertAlmostEqual(receipt['additional_seconds']+receipt['provisional_seconds_already_charged'],12.)
        self.assertEqual(m.read(self.root/'state.json')['compute_seconds'],112.)

    def test_parent_grace_cannot_exceed_remaining_total_budget(self):
        config=m.read(self.root/'config.lock.json');config['total_compute_seconds']=204.
        (self.root/'config.lock.json').write_text(json.dumps(config))
        with patch.dict(os.environ,self.env),patch.object(m,'campaign_root',return_value=self.root),patch('evcs_experiments.case0_isolation.verify_current_cgroup'):
            with self.assertRaisesRegex(ValueError,'timeout/grace'):m.run_case0_evaluation(acceptance_reference={},seconds=100)
        self.assertNotIn('pending_evaluation_attempt',m.read(self.root/'state.json'))

    def test_unsettled_audit_reservation_blocks_new_evaluation(self):
        state=m.read(self.root/'state.json');state['audit_reservation_reference']=m._write(self.root/'audit-reservation.json',{'pending':True})
        (self.root/'state.json').write_text(json.dumps(state))
        with patch.object(m,'campaign_root',return_value=self.root),patch('evcs_experiments.case0_isolation.verify_current_cgroup'):
            with self.assertRaisesRegex(ValueError,'Unsettled'):m.run_case0_evaluation(acceptance_reference={},seconds=100)

    def test_live_parent_cannot_settle_or_release_reservation(self):
        with patch('evcs_experiments.runtime.jobctl.observation',return_value=dict(DEAD,blocks_new_launch=True)):
            with self.assertRaisesRegex(ValueError,'terminal/dead'):m.settle_case0_evaluation_attempt(root=self.root,terminal_result_reference=self.result)
        self.assertEqual(m.read(self.root/'state.json')['compute_seconds'],100.)

if __name__=='__main__':unittest.main()
