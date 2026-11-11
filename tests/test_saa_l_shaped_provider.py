"""Focused eager/provider comparison and current incomplete-Q disk evidence."""
from dataclasses import replace
import hashlib,json,os
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_recourse_template_index as fixture_module
from evcs_experiments.saa_l_shaped import DevelopmentSAAProblem,run_saa_l_shaped
from evcs_experiments.saa_l_shaped_provider import StoredDevelopmentSAAProblem
from evcs_experiments.recourse import solve_fixed_plan

OPTS=dict(mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
class SAAProviderTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.TemplateIndexTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
    def stored(self):
        t=self.fixture;f=t.fixture
        return StoredDevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd,
            provider=t.provider(),evidence_root=t.root/'disk',capacity=1)
    def test_same_physical_input_streaming_matches_eager_full_mean_and_small_checkpoint(self):
        t=self.fixture;f=t.fixture
        eager=DevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd)
        expected=run_saa_l_shaped(eager,**OPTS)
        with patch('evcs_experiments.saa_l_shaped.assemble_recourse',side_effect=AssertionError('no assembly')):
            p=self.stored();result=run_saa_l_shaped(p,checkpoint_path=t.root/'cp.json',**OPTS)
        self.assertEqual(result['status'],'CONVERGED')
        for key in ('upper_bound_usd','lower_bound_usd'):
            self.assertAlmostEqual(result[key],expected[key],places=8)
        self.assertEqual(result['best']['q_usd'],expected['best']['q_usd'])
        self.assertLessEqual(len(p.recourses.statistics()['resident_indices']),1)
        self.assertEqual(p.atom_store.statistics()['resident_witness_arrays'],0)
        self.assertLess((t.root/'cp.json').stat().st_size,len(json.dumps(expected)))
        self.assertTrue(all(set(a)=={'kind','reference','anchor_sha256','q_usd','plane_sha256'} for group in result['cuts'] for a in group))
    def test_one_interrupted_q_is_saved_as_bound_diagnostic_and_tamper_refused(self):
        t=self.fixture;p=self.stored();cp=t.root/'cp.json';calls=[]
        def interrupt(template,plan,**kw):
            r=solve_fixed_plan(template,plan,**kw);calls.append(r)
            return replace(r,status='LIMIT_REACHED',optimal=False,solver_status=1) if len(calls)==2 else r
        with patch('evcs_experiments.saa_l_shaped.solve_fixed_plan',side_effect=interrupt):
            result=run_saa_l_shaped(p,checkpoint_path=cp,**OPTS)
        self.assertEqual(result['status'],'TIME_LIMIT');self.assertIsNone(result['upper_bound_usd'])
        self.assertEqual(set(result['last_incomplete_recourse']),{'kind','reference'})
        original_pin=hashlib.sha256(cp.read_bytes()).hexdigest()
        diag=Path(result['last_incomplete_recourse']['reference']['path']);before=diag.read_bytes();diag.write_bytes(before+b' ')
        with self.assertRaises(ValueError):run_saa_l_shaped(p,checkpoint_path=cp,resume=True,expected_checkpoint_sha256=original_pin,**OPTS)
        diag.write_bytes(before)
        resumed=run_saa_l_shaped(p,checkpoint_path=cp,resume=True,expected_checkpoint_sha256=original_pin,**OPTS)
        self.assertEqual(resumed['status'],'CONVERGED');self.assertAlmostEqual(resumed['upper_bound_usd'],30.266666666666666)
