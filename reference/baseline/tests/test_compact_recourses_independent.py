"""Independent bit identity and original-array audit of combined representations."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.compact_recourses import CompactRows, compact_recourse
from evcs_experiments.case0_planning import run_case0_planning
from evcs_experiments.case0_planning_audit import audit_case0_method
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse, _json
from evcs_experiments.validation.s03_toys import build_golden_cases


class CompactIndependentTests(unittest.TestCase):
    def test_signed_zero_and_scalar_bits_survive_category_encoding(self):
        values=(0.,-0.,np.float32(0),np.float32(-0.),np.float64(0),np.float64(-0.),False,0)
        original=tuple({'index':i,'value':v} for i,v in enumerate(values))
        compact=CompactRows(original)
        self.assertEqual(json.dumps(_json(original)),json.dumps([_json(dict(r)) for r in compact]))
        for i,value in enumerate(values):
            actual=compact[i]['value'];self.assertIs(type(actual),type(value))
            if isinstance(value,(float,np.floating)):
                self.assertEqual(np.signbit(actual),np.signbit(value))
        for bad in (np.complex128(1j),np.datetime64('2026-09-10')):
            with self.assertRaises(ValueError):CompactRows(({'index':0,'value':bad},))

    def test_four_methods_combined_compaction_and_zero_proof_keep_original_results(self):
        f=build_golden_cases()[1]
        for scene in f.scenarios:scene.split='planning'
        first=assemble_first_stage(f.case,f.tasks)
        original=[assemble_recourse(f.case,f.tasks,s) for s in f.scenarios]
        compact=[compact_recourse(t) for t in original]
        with tempfile.TemporaryDirectory() as d:
            for method in ('EA_DRO','OR_DRO','SAA','ARO'):
                records=[]
                for mode,recourses in ((False,original),(True,compact)):
                    out=Path(d)/(method+str(mode));now=time.monotonic()
                    _,r=run_case0_planning(first,recourses,f.scenario_ids,f.probability_set,
                        case=f.case,tasks=f.tasks,scenes=f.scenarios,method_id=method,output_dir=out,
                        backend='highs',deadline=now+1800,configuration_started_at=now,master_seconds=5,
                        empirical_probabilities=np.full(len(recourses),1/len(recourses)),zero_presolve=True,
                        compact_descriptive_metadata=mode)
                    records.append(r)
                    with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No reoptimization')):
                        audit=audit_case0_method(result_reference=file_reference(out/'result.json'),case=f.case,
                            tasks=f.tasks,scenes=f.scenarios,probability_set=f.probability_set,deadline=time.monotonic()+20)
                    self.assertTrue(audit['passed'])
                    self.assertTrue(audit['zero_reduction_audit']['original_zero_lift_checked'])
                a,b=records
                self.assertEqual(a['model_sha256'],b['model_sha256'])
                self.assertEqual(a['scenario_ids'],b['scenario_ids'])
                for key in ('lower_bound_usd','upper_bound_usd','risk_value_usd'):
                    self.assertAlmostEqual(a[key],b[key],places=7)
                self.assertNotIn('descriptive_metadata_representation',a)
                self.assertEqual(b['descriptive_metadata_representation'],'lossless_categorical_rows_v1')


if __name__=='__main__':unittest.main()
