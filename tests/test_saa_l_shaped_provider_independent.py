"""Independent real saved-template residency and disk-witness boundaries."""
from copy import deepcopy
from contextlib import ExitStack
import gc,hashlib,json,tempfile,unittest,weakref
from pathlib import Path
from unittest.mock import patch
from tests import test_recourse_template_index as template_fixtures
from evcs_experiments import recourse,recourse_store,recourse_template_index
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments.evaluation_lock import file_reference

class IndependentSAAProviderTests(unittest.TestCase):
    def setUp(self):
        self.fixture=template_fixtures.TemplateIndexTests(methodName='runTest')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture.fixture;self.provider=self.fixture.provider()
        self.root=self.fixture.root/'new_saa';self.checkpoint=self.fixture.root/'new_saa_checkpoint.json'
        self.options=dict(mip_relative_gap=1e-9,optimality_relative_gap=1e-8)

    def problem(self,**changes):
        from evcs_experiments.saa_l_shaped_provider import StoredDevelopmentSAAProblem
        args=dict(provider=self.provider,evidence_root=self.root,capacity=1)
        args.update(changes)
        return StoredDevelopmentSAAProblem(self.f.case,self.f.tasks,self.f.scenarios,self.f.scenario_cost_constants_usd,**args)

    def guards(self):
        stack=ExitStack()
        for module in (recourse,kernel):stack.enter_context(patch.object(module,'assemble_recourse',side_effect=AssertionError('Reassembly is forbidden during provider work')))
        stack.enter_context(patch.object(recourse_store.StoredRecourseTemplate,'as_dict',side_effect=AssertionError('Giant descriptive JSON export is forbidden')))
        return stack

    @staticmethod
    def pin(path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def refs(self,value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=value.keys():yield value
            else:
                for v in value.values():yield from self.refs(v)
        elif isinstance(value,list):
            for v in value:yield from self.refs(v)

    def assert_compact(self,value):
        if isinstance(value,dict):
            self.assertNotIn('dual',value)
            if 'primal' in value:
                self.assertIn('first_stage_columns',value)
                if value['primal'] is not None:self.assertEqual(len(value['primal']),value['first_stage_columns']+value['theta_columns'])
            for v in value.values():self.assert_compact(v)
        elif isinstance(value,list):
            for v in value:self.assert_compact(v)

    def test_original_templates_are_evicted_and_complete_solve_has_compact_checkpoint(self):
        with self.guards():
            p=self.problem()
            self.assertNotIsInstance(p.recourses,tuple)
            self.assertEqual(p.recourses.statistics()['capacity'],1)
            t=p.recourses[0];weak=weakref.ref(t)
            self.assertFalse(t.descriptive_maps_loaded);del t
            t=p.recourses[1];del t;gc.collect()
            self.assertIsNone(weak())
            self.assertEqual(p.recourses.statistics()['resident_indices'],[1])
            result=kernel.run_saa_l_shaped(p,checkpoint_path=self.checkpoint,**self.options)
        self.assertEqual(result['status'],'CONVERGED')
        self.assertAlmostEqual(result['upper_bound_usd'],self.f.expected['SAA']['objective_usd'],places=7)
        self.assertLessEqual(len(p.recourses.statistics()['resident_indices']),1)
        self.assert_compact(result)
        self.assertGreater(len(list(self.root.rglob('*.npz'))),0)
        with self.guards():
            reread=kernel.run_saa_l_shaped(self.problem(),checkpoint_path=self.checkpoint,resume=True,expected_checkpoint_sha256=self.pin(self.checkpoint),**self.options)
        self.assertEqual(result,reread)
        self.assertFalse(reread['formal_admission']);self.assertFalse(reread['formal_OOS'])

    def test_referenced_disk_witness_tampering_rejected_before_converged_reuse(self):
        with self.guards():
            p=self.problem();result=kernel.run_saa_l_shaped(p,checkpoint_path=self.checkpoint,**self.options)
        trusted=self.pin(self.checkpoint)
        owned=[r for r in self.refs(result) if Path(r['path']).is_relative_to(self.root)]
        self.assertTrue(owned)
        target=None
        for reference in owned:
            path=Path(reference['path'])
            if path.suffix=='.npz':target=path;break
            if path.suffix=='.json':
                nested=[r for r in self.refs(json.loads(path.read_text())) if Path(r['path']).suffix=='.npz']
                if nested:target=Path(nested[0]['path']);break
        self.assertIsNotNone(target)
        target.write_bytes(target.read_bytes()+b'corrupt')
        with self.guards(),patch.object(kernel,'_master',side_effect=AssertionError('Corrupt evidence must not solve')):
            with self.assertRaises(ValueError):kernel.run_saa_l_shaped(p,checkpoint_path=self.checkpoint,resume=True,expected_checkpoint_sha256=trusted,**self.options)

    def test_original_numeric_template_and_scene_binding_not_bypassed(self):
        from evcs_experiments.saa_l_shaped_provider import StoredDevelopmentSAAProblem
        with self.guards(),self.assertRaises(ValueError):
            StoredDevelopmentSAAProblem(self.f.case,self.f.tasks,tuple(reversed(self.f.scenarios)),self.f.scenario_cost_constants_usd,provider=self.provider,evidence_root=self.root,capacity=1)
        with self.guards():p=self.problem()
        original=self.fixture.root/'original/templates/000/arrays.npz';original.write_bytes(original.read_bytes()+b'changed')
        p.recourses[1] # Force the affected index out of the one-entry cache.
        with self.assertRaises(ValueError):p.recourses[0]

    def test_external_checkpoint_pin_and_provider_admission_remain_bound(self):
        with self.guards():
            p=self.problem();kernel.run_saa_l_shaped(p,checkpoint_path=self.checkpoint,max_iterations=1,**self.options)
        trusted=self.pin(self.checkpoint)
        with self.assertRaises(ValueError):kernel.run_saa_l_shaped(p,checkpoint_path=self.checkpoint,resume=True,**self.options)
        with self.assertRaises(ValueError):kernel.run_saa_l_shaped(p,checkpoint_path=self.checkpoint,resume=True,expected_checkpoint_sha256='0'*64,**self.options)
        alternate=self.fixture.prepare_index(output='alternate_index.json')
        review=json.loads(Path(self.fixture.review['path']).read_text());review['index_reference']=alternate
        alternative_review=self.fixture.write('alternate_association.json',review)
        other=recourse_template_index.TemplateProvider(alternate,alternative_review,first_stage=self.fixture.first,case=self.f.case,tasks=self.f.tasks,scenario_ids=self.f.scenario_ids,scenario_sha256s=[s.scenario_sha256 for s in self.f.scenarios],domain='development')
        with self.guards(),self.assertRaises(ValueError):
            other_problem=self.problem(provider=other)
            kernel.run_saa_l_shaped(other_problem,checkpoint_path=self.checkpoint,resume=True,expected_checkpoint_sha256=trusted,**self.options)
