import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from evcs_experiments import case0_campaign as c
from evcs_experiments.case0_profiles import *


class ProfileTests(unittest.TestCase):
    def test_inputs_disjoint_and_legacy_default(self):
        validate_inputs(LEGACY,3,2,10000)
        validate_inputs(EXPANDED,12,1,40000)
        for args in [(LEGACY,12,1,30000),(EXPANDED,12,1,10000),(EXPANDED,10,1,30000),(EXPANDED,12,2,30000),(EXPANDED,True,1,30000)]:
            with self.assertRaises(ValueError):validate_inputs(*args)
        self.assertEqual(context_options({'variant_start':20000}),{'variant_start':20000})

    def test_expanded_lock_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/CAMPAIGNS[EXPANDED];root.mkdir();(root/'MANIFEST_SHA256.csv').write_text('frozen')
            paths=SimpleNamespace(campaign_root=root,data_root=root)
            with patch.object(c,'get_paths',return_value=paths),patch.object(c,'sources',return_value={'test':'source'}):
                c.prepare('gurobi',EXPANDED)
                config=c.active_config(root)
                self.assertEqual(context_options(config),dict(variant_start=30000,profile=EXPANDED))
                self.assertEqual(config['sizes'],[dict(fleet_per_depot=12,planning_variants=1)])
                state=c.read(root/'state.json');state.update(status='FAILURE_REQUIRES_REVIEW',compute_seconds=123.4);c.save_state(root,state)
                before=(root/'state.json').read_bytes()
                with self.assertRaises(ValueError):c.fallback('no fallback declared')
                self.assertEqual((root/'state.json').read_bytes(),before)
                for field,value in [('target_event','michael_2018'),('variant_start',10000),('solver_feasibility_tolerance',1e-8),('master_seconds',1201),('campaign',c.CAMPAIGN_NAME),('sizes',[dict(fleet_per_depot=2,planning_variants=1)])]:
                    altered=copy.deepcopy(config);altered[field]=value
                    with self.assertRaises(ValueError):validate_config(altered,root)

    def test_wrong_campaign_and_backend_fail_before_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            for name,profile,backend in [('ieee33-v54-q1',EXPANDED,'gurobi'),(CAMPAIGNS[LEGACY],EXPANDED,'gurobi'),(CAMPAIGNS[EXPANDED],EXPANDED,'highs')]:
                root=Path(temp)/name
                with patch.object(c,'get_paths',return_value=SimpleNamespace(campaign_root=root)):
                    with self.assertRaises(ValueError):c.prepare(backend,profile)
                self.assertFalse(root.exists())

    def test_expanded_spec_is_source_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'src/evcs_experiments').mkdir(parents=True);(root/'docs').mkdir()
            for path in (root/'requirements.lock',root/'docs/CASE0_SPEC_CN.md',root/'docs/CASE0_EXPANDED_SPEC_CN.md'):path.write_text('a')
            paths=SimpleNamespace(project_root=root,spec_root=root/'docs',campaign_root=root/CAMPAIGNS[EXPANDED])
            with patch.object(c,'get_paths',return_value=paths):
                before=c.sources();(root/'docs/CASE0_EXPANDED_SPEC_CN.md').write_text('b')
                self.assertNotEqual(before,c.sources())

    def test_expanded_plan_dispatch_uses_locked_stream_and_gap_stops_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/CAMPAIGNS[EXPANDED];root.mkdir();(root/'MANIFEST_SHA256.csv').write_text('frozen')
            with patch.object(c,'get_paths',return_value=SimpleNamespace(campaign_root=root,data_root=root)),patch.object(c,'sources',return_value={'test':'source'}),patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}):
                state=c.prepare('gurobi',EXPANDED)
                state['active_configuration']='size0/EA_DRO';c.save_state(root,state)
                with patch('evcs_experiments.case0_inputs.Case0Context',side_effect=RuntimeError('stop before any numerical work')) as context:
                    with self.assertRaisesRegex(RuntimeError,'stop before'):c.plan_one('EA_DRO',100.)
                    self.assertEqual(context.call_args.kwargs['variant_start'],30000)
                    self.assertEqual(context.call_args.kwargs['profile'],EXPANDED)
                    self.assertEqual(context.call_args.kwargs['fleet_per_depot'],12)
                state['active_configuration']=None;c.save_state(root,state)
                def limited(method,started,folder):
                    result=dict(gap=.1,numerical_valid=True,gap_passed=False)
                    (folder/'solve').mkdir();c.atomic_json(folder/'solve/result.json',result)
                    return result
                with patch.object(c,'bounded_configuration',side_effect=limited):out=c.plan_batch()
                self.assertEqual(out['status'],'DIAGNOSIS_REQUIRED')
                self.assertNotIn('use the single declared smaller',out['next_action'])
                self.assertIn('size0/EA_DRO',out['completed'])
                self.assertFalse((root/'planning/size1').exists())
