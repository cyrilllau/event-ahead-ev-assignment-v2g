"""Case0 control isolation and single whole-group fallback; no experiment calls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import subprocess
import signal
from evcs_experiments import case0_campaign as c


class Case0CampaignTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)/c.CAMPAIGN_NAME
        self.root.mkdir();(self.root/'MANIFEST_SHA256.csv').write_text('synthetic-input-manifest')
        self.paths=SimpleNamespace(campaign_root=self.root,data_root=self.root)
        for name,value in [('get_paths',lambda:self.paths),('sources',lambda:{'synthetic-source':'a'*64})]:
            patcher=patch.object(c,name,side_effect=value);patcher.start();self.addCleanup(patcher.stop)

    def test_other_campaign_name_is_rejected_without_writes(self):
        self.paths.campaign_root=self.root.parent/'ieee33-v54-q1'
        with self.assertRaisesRegex(ValueError,'own explicitly'):c.prepare('highs')
        self.assertFalse(self.paths.campaign_root.exists())

    def test_backend_and_sources_remain_locked(self):
        original=c.prepare('highs');self.assertEqual(original['status'],'PREPARED')
        c.prepare('highs')
        with self.assertRaisesRegex(ValueError,'backend'):c.prepare('gurobi')
        with patch.object(c,'sources',return_value={'synthetic-source':'b'*64}):
            with self.assertRaisesRegex(ValueError,'source changed'):c.prepare('highs')

    def test_fallback_is_once_before_OOS_preserves_all_costs_and_original_results(self):
        state=c.prepare('highs');state.update(status='FALLBACK_REQUIRED',compute_seconds=1723.5,
            completed={'size0/EA_DRO':{'sha256':'original'}})
        c.save_state(self.root,state)
        changed=c.fallback('Original resource/gap evidence requires declared smaller entire group')
        self.assertEqual(changed['size_index'],1);self.assertEqual(changed['compute_seconds'],1723.5)
        self.assertEqual(changed['completed'],state['completed'])
        self.assertFalse(changed['evaluation_opened'])
        with self.assertRaisesRegex(ValueError,'not available'):c.fallback('second attempt')
        decisions=[json.loads(line) for line in (self.root/'decisions.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['event']=='DECLARED_GROUP_FALLBACK' for row in decisions),1)

    def test_per_configuration_child_has_literal_module_and_hard_owned_group_kill(self):
        folder=self.root/'bounded';folder.mkdir()
        child=Mock(pid=876543);child.wait.side_effect=[subprocess.TimeoutExpired('child',1790),0]
        with patch.object(c.subprocess,'Popen',return_value=child) as popen, \
             patch.object(c.os,'killpg') as kill,patch.object(c.time,'monotonic',return_value=110.), \
             patch.object(c,'__name__','__main__'):
            with self.assertRaisesRegex(TimeoutError,'1800'):c.bounded_configuration('EA_DRO',100.,folder)
        self.assertEqual(popen.call_args.args[0][1:3],['-m','evcs_experiments.case0_campaign'])
        self.assertEqual(popen.call_args.kwargs['process_group'],0)
        self.assertEqual(child.wait.call_args_list[0].kwargs['timeout'],1790.)
        kill.assert_called_once_with(child.pid,signal.SIGKILL)
        self.assertEqual(child.wait.call_count,2)

    def test_hard_timeout_preserves_cost_and_stops_incomplete_group(self):
        state=c.prepare('highs');state['compute_seconds']=35.;c.save_state(self.root,state)
        clock=[100.]
        def timeout(*args):clock[0]=1900.;raise TimeoutError('hard configuration limit')
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={'synthetic':'owned group'}), \
             patch.object(c,'bounded_configuration',side_effect=timeout), \
             patch.object(c.time,'monotonic',side_effect=lambda:clock[0]):
            result=c.plan_batch()
        self.assertEqual(result['status'],'FAILURE_REQUIRES_REVIEW')
        self.assertEqual(result['compute_seconds'],1835.)
        self.assertIsNone(result['active_configuration']);self.assertEqual(result['completed'],{})
        folder=self.root/'planning/size0/EA_DRO'
        self.assertTrue((folder/'failure.json').exists());self.assertTrue((folder/'accounting.json').exists())
        self.assertFalse((self.root/'planning/size0/OR_DRO').exists())

    def test_global_budget_exhaustion_stops_while_short_job_keeps_admission(self):
        state=c.prepare('highs');state['compute_seconds']=20000.;c.save_state(self.root,state)
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}), \
             patch.object(c,'bounded_configuration',side_effect=AssertionError('No inadmissible solve')), \
             patch.object(c.time,'monotonic',return_value=100.):
            result=c.plan_batch();self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
            self.assertEqual(result['compute_seconds'],20000.)
            state.update(status='PREPARED',compute_seconds=0.);c.save_state(self.root,state)
            result=c.plan_batch(seconds=100.)
            self.assertEqual(result['status'],'PREPARED');self.assertIn('new bounded',result['next_action'])

    def test_fallback_refuses_active_job_OOS_or_empty_reason(self):
        state=c.prepare('highs');state['status']='FALLBACK_REQUIRED'
        for key,value in [('evaluation_opened',True),('active_configuration','size0/EA_DRO')]:
            changed=dict(state);changed[key]=value;c.save_state(self.root,changed)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'not available'):c.fallback('reason')
        c.save_state(self.root,state)
        with self.assertRaisesRegex(ValueError,'evidence-based'):c.fallback(' ')

if __name__=='__main__':unittest.main()
