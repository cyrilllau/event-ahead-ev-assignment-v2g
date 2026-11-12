"""Lossless descriptive compaction; no optimizer or production experiment."""
import dataclasses
import gc
import json
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np

from evcs_experiments.compact_recourses import CompactRows,CompactRecourseTemplate,compact_recourse,MAP_FIELDS
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.extensive import assemble_extensive
from evcs_experiments.case0_planning import run_case0_planning
from evcs_experiments.validation.s03_toys import build_golden_cases


class CompactRecourseTests(unittest.TestCase):
    def setUp(self):
        self.f=build_golden_cases()[1]
        self.original=assemble_recourse(self.f.case,self.f.tasks,self.f.scenarios[0])

    def test_every_label_and_serialized_template_remains_exact(self):
        compact=compact_recourse(self.original)
        self.assertIsInstance(compact,CompactRecourseTemplate)
        self.assertIs(compact_recourse(compact),compact)
        for name in MAP_FIELDS:
            self.assertEqual([dict(r) for r in getattr(self.original,name)],[dict(r) for r in getattr(compact,name)])
        self.assertEqual(json.dumps(self.original.as_dict(),sort_keys=True),json.dumps(compact.as_dict(),sort_keys=True))
        for f in dataclasses.fields(self.original):
            if f.name not in MAP_FIELDS:self.assertIs(getattr(self.original,f.name),getattr(compact,f.name))

    def test_original_mapping_objects_are_not_retained(self):
        class WeakDict(dict):pass
        row=WeakDict(self.original.variable_map[0]);reference=weakref.ref(row)
        original=dataclasses.replace(self.original,variable_map=(MappingProxyType(row),)+self.original.variable_map[1:])
        compact=compact_recourse(original);del original,row;gc.collect()
        self.assertIsNone(reference());self.assertEqual(dict(compact.variable_map[0]),dict(self.original.variable_map[0]))

    def test_readonly_index_slicing_and_scalar_type_preservation(self):
        rows=CompactRows(({'index':0,'label':False},{'index':1,'label':0},{'index':2,'label':0.0},{'index':3,'label':'0'}))
        self.assertEqual([type(r['label']) for r in rows],[bool,int,float,str])
        self.assertEqual(rows[-1],rows[3]);self.assertEqual(len(rows[1:3]),2)
        with self.assertRaises(IndexError):rows[4]
        with self.assertRaises(AttributeError):rows._length=9
        with self.assertRaises(TypeError):rows[0]['label']='changed'
        for codes in rows._codes:
            if codes is not None:
                with self.assertRaises(ValueError):codes[0]=99
        self.assertEqual(len(CompactRows(())),0)

    def test_signed_zero_and_numpy_scalar_bits_remain_exact(self):
        values=[0.0,-0.0,np.float32(0.0),np.float32(-0.0),np.float64(0.0),np.float64(-0.0)]
        rows=CompactRows(tuple({'index':i,'value':v} for i,v in enumerate(values)))
        for original,row in zip(values,rows):
            self.assertIs(type(row['value']),type(original))
            self.assertEqual(bool(np.signbit(row['value'])),bool(np.signbit(original)))
        for bad in (np.complex64(1+0j),np.datetime64('2020-01-01'),np.complex128(complex(float('nan'),0))):
            with self.assertRaises(ValueError):CompactRows(({'index':0,'value':bad},))

    def test_invalid_metadata_cannot_be_silently_reordered_or_rounded(self):
        for rows in (({'index':1,'value':'x'},),({'index':0,'value':float('nan')},),
                     ({'index':0,'value':'x'},{'value':'x','index':1})):
            with self.assertRaises(ValueError):CompactRows(rows)

    def test_all_risk_models_keep_original_coefficients_and_identity(self):
        first=assemble_first_stage(self.f.case,self.f.tasks)
        original=[assemble_recourse(self.f.case,self.f.tasks,s) for s in self.f.scenarios]
        compact=[compact_recourse(t) for t in original]
        for risk in ('DRO','SAA','ARO'):
            kw=dict(risk_model=risk,scenario_ids=self.f.scenario_ids,
                probability_set=self.f.probability_set if risk=='DRO' else None,
                empirical_probabilities=self.f.empirical_probabilities if risk=='SAA' else None)
            a=assemble_extensive(first,original,**kw);b=assemble_extensive(first,compact,**kw)
            self.assertEqual(a.model_sha256,b.model_sha256);self.assertEqual((a.A!=b.A).nnz,0)
            for name in ('c','row_lower','row_upper','lower_bounds','upper_bounds','integrality'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))

    def test_explicit_case0_path_checks_original_arrays_then_drops_last_rebuild(self):
        for s in self.f.scenarios:s.split='planning'
        first=assemble_first_stage(self.f.case,self.f.tasks)
        compact=[compact_recourse(assemble_recourse(self.f.case,self.f.tasks,s)) for s in self.f.scenarios]
        rebuilt=[];real=assemble_recourse
        def record(*args):
            t=real(*args);rebuilt.append(weakref.ref(t));return t
        def solver_boundary(*args,**kwargs):
            gc.collect();self.assertTrue(all(r() is None for r in rebuilt))
            self.assertEqual(kwargs['mip_relative_gap'],.005);self.assertEqual(kwargs['optimizer_threads'],1)
            raise RuntimeError('checked boundary')
        with tempfile.TemporaryDirectory() as d:
            now=time.monotonic()
            kw=dict(case=self.f.case,tasks=self.f.tasks,scenes=self.f.scenarios,method_id='EA_DRO',
                output_dir=Path(d)/'solve',backend='highs',deadline=now+1800,configuration_started_at=now,
                master_seconds=5,compact_descriptive_metadata=True)
            with patch('evcs_experiments.case0_planning.assemble_recourse',side_effect=record),patch('evcs_experiments.case0_planning.solve_extensive',side_effect=solver_boundary):
                with self.assertRaisesRegex(RuntimeError,'checked boundary'):
                    run_case0_planning(first,compact,self.f.scenario_ids,self.f.probability_set,**kw)
            kw['output_dir']=Path(d)/'uncompacted';plain=[self.original]*len(compact)
            with self.assertRaisesRegex(ValueError,'Compact each uncached'):
                run_case0_planning(first,plain,self.f.scenario_ids,self.f.probability_set,**kw)


if __name__=='__main__':unittest.main()
