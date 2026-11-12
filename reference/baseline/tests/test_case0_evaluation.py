"""Tiny physical item preparation; no claim of Case0 pre-OOS lock acceptance."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments import case0_evaluation as m
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s02_toys import golden_single_bus


class Case0EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.f=golden_single_bus();f=self.f;f.scenario.target_event='michael_2018';f.scenario.regime='R1';f.scenario.split='oos'
        f.scenario.metadata={'departure_h':-1}
        zero=FirstStagePlan(np.zeros_like(f.plan.x),np.zeros_like(f.plan.r),f.plan.task_sha256,f.plan.config_hash)
        plans={name:f.plan for name in m.METHODS['R1']};plans['ZERO']=zero
        self.lock=m._write(self.root/'lock.json',{'fixture':True});self.calls=0
        def provider(c,t,s):
            self.calls+=1;template=assemble_recourse(c,t,s)
            return template,m._write(self.root/f'template_{self.calls}.json',{'sha256':template.template_sha256})
        self.ctx=m.Case0EvaluationContext(lock_reference=self.lock,lock_validator=lambda ref:dict(
            method_plan_sha256={k:m.plan_identity(p) for k,p in plans.items()},
            sample_ids_by_regime={r:[f.scenario.scenario_id] for r in m.METHODS}),plans=plans,
            physical_factory=lambda method,index,regime:(f.case,f.tasks,f.scenario),template_provider=provider,
            source_references=[m.file_reference(m.__file__)],samples_per_regime=1)

    def run_item(self,method='EA_DRO',folder='item'):
        item=next(x for x in self.ctx.items.values() if x['regime']=='R1' and x['method_id']==method)
        return m.evaluate_case0_item(self.ctx,item,output_directory=self.root/folder,deadline=time.monotonic()+10)

    def test_exact_768_unique_items(self):
        rows=m.case0_items();self.assertEqual(len(rows),768);self.assertEqual(len({r['item_id'] for r in rows}),768)
        self.assertEqual(sum(r['regime']=='R1' for r in rows),384)

    def test_real_item_resume_no_lp_and_shared_template_for_other_method(self):
        first=self.run_item()
        with patch.object(m,'solve_fixed_plan',side_effect=AssertionError('No repeat LP')):again=self.run_item()
        self.assertTrue(again['reused']);self.assertEqual(first['record_reference'],again['record_reference'])
        self.run_item('SAA','saa');self.assertEqual(self.calls,1);self.assertEqual(self.ctx.cache_hits,1)
        row=m._read(first['record_reference']);self.assertFalse(row['scientific_acceptance'])
        metric=m._read(next(r for r in row['artifacts'] if Path(r['path']).name=='metrics.json'))
        self.assertGreater(metric['costs']['total_DSO_usd'],0)

    def test_tampered_raw_evidence_refuses_resume(self):
        first=self.run_item();row=m._read(first['record_reference'])
        p=Path(next(r['path'] for r in row['artifacts'] if Path(r['path']).name=='primal.npy'))
        b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
        with self.assertRaisesRegex(ValueError,'changed'):self.run_item()

    def test_wrong_locked_sample_or_partial_output_rejected(self):
        self.f.scenario.scenario_id='different'
        with self.assertRaisesRegex(ValueError,'sample'):self.run_item()
        with self.assertRaisesRegex(ValueError,'Partial'):self.run_item()

if __name__=='__main__':unittest.main()
