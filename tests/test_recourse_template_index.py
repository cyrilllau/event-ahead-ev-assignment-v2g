"""Reviewed import and reference-only export with real small C&CG certificates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.ccg import CCGProblem, run_ccg
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.formal_planning import RawPlanningEvidence, _verified_record
from evcs_experiments.lazy_recourses import LazyRecourses
import evcs_experiments.recourse as recourse
import evcs_experiments.recourse_store as reader
import evcs_experiments.recourse_template_index as index_module
from evcs_experiments.recourse_store import bind_template
from evcs_experiments.recourse_template_index import prepare_template_index, TemplateProvider
from evcs_experiments.validation.s03_toys import build_golden_cases


class TemplateIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fixture = build_golden_cases()[0]
        f = self.fixture
        self.templates = tuple(recourse.assemble_recourse(f.case, f.tasks, scene) for scene in f.scenarios)
        self.first = self.templates[0].first_stage
        original_cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],
            self.first.matrix_sha256,{'fixture':f.name},self.templates.__getitem__,capacity=2)
        problem = CCGProblem(self.first, original_cache, f.probability_set, f.scenario_cost_constants_usd)
        old = RawPlanningEvidence(self.root/'original', f.case, f.tasks, problem, f.scenarios.__getitem__,
                                  domain='development', input_binding={'fixture':f.name})
        records = [old.template(i,t) for i,t in enumerate(self.templates)]
        self.snapshot = old.snapshot(self.root/'original_snapshot.json')
        self.request = file_reference(old.root/'request.json')
        self.bindings = [bind_template(r, self.first, f.case, f.tasks, f.scenarios[i],
            source_references={'original_request':self.request}, compiler_reference=file_reference(recourse.__file__),
            output=self.root/f'binding_{i}.json') for i,r in enumerate(records)]
        # Test-only association proof. No test fixture authorizes production.
        self.proof = self.write('development_proof.json', {'domain':'development','matrices_independently_compared':True})
        self.index = self.prepare_index()
        self.review = self.make_review()

    def write(self, name, value):
        path=self.root/name;path.write_text(json.dumps(value,indent=2)+'\n');return file_reference(path)

    def prepare_index(self, bindings=None, output='index.json', snapshot=None):
        return prepare_template_index(original_request_reference=self.request,
            original_snapshot_reference=self.snapshot if snapshot is None else snapshot,
            binding_references=self.bindings if bindings is None else bindings,
            proof_references={'original_numeric_proof':self.proof}, output=self.root/output)

    def make_review(self, **changes):
        value=dict(status=index_module.REVIEW_STATUS,passed=True,domain='development',
            index_reference=self.index,proof_references={'original_numeric_proof':self.proof},
            scenario_ids=list(self.fixture.scenario_ids),full_support_association_verified=True,formal_OOS=False,
            implementation_sources=[file_reference(index_module.__file__),file_reference(reader.__file__)])
        value.update(changes)
        return self.write('association_review.json',value)

    def provider(self, review=None, **changes):
        f=self.fixture
        args=dict(first_stage=self.first,case=f.case,tasks=f.tasks,scenario_ids=f.scenario_ids,
                  scenario_sha256s=[s.scenario_sha256 for s in f.scenarios],domain='development')
        args.update(changes)
        return TemplateProvider(self.index,self.review if review is None else review,**args)

    def prepare_run(self, risk, *, resume=False, snapshot=None, evidence_root=None):
        f=self.fixture;provider=self.provider()
        binding=dict(fixture=f.name,template_provider=provider.identity)
        cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],self.first.matrix_sha256,
            binding,lambda i:provider.load(i,f.scenarios[i]),capacity=2)
        p=CCGProblem(self.first,cache,f.probability_set,f.scenario_cost_constants_usd,risk)
        e=RawPlanningEvidence(self.root/risk if evidence_root is None else evidence_root,f.case,f.tasks,p,f.scenarios.__getitem__,domain='development',
            input_binding=binding,template_provider=provider,resume=resume,resume_snapshot_reference=snapshot)
        return p,e

    def test_two_risk_configurations_run_real_ccg_without_reassembly_or_map_exports(self):
        for risk in ('DRO','ARO'):
            with self.subTest(risk=risk):
                p,e=self.prepare_run(risk)
                with (patch('evcs_experiments.formal_planning.assemble_recourse',side_effect=AssertionError('no recompile')),
                      patch.object(reader.StoredRecourseTemplate,'as_dict',side_effect=AssertionError('no giant JSON'))):
                    result=run_ccg(p,evidence_sink=e.sink_factory(self.root/(risk+'_call')),wall_seconds=20,
                                   rmp_time_limit_seconds=3,oracle_time_limit_seconds=3,
                                   mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
                    self.assertEqual(result.status,'CONVERGED')
                    self.assertAlmostEqual(result.state['upper_bound_usd'],self.fixture.expected[risk]['objective_usd'],places=7)
                    snapshot=e.snapshot(self.root/(risk+'_snapshot.json'))
                for folder in (e.root/'templates').iterdir():
                    self.assertEqual({p.name for p in folder.iterdir()},{'record.json','template_source.json'})
                    row=_verified_record(folder/'record.json')
                    self.assertEqual(row['kind'],'recourse_template')
                    self.assertEqual(row['template_sha256'],self.templates[int(folder.name)].template_sha256)
                for reference in e.events:
                    event=json.loads(Path(reference['path']).read_text())
                    if event['event']=='recourse':
                        directory=Path(reference['path']).parent
                        self.assertTrue(json.loads((directory/'physical_check.json').read_text())['passed'])
                        with np.load(directory/'dual_kkt.npz') as proof:
                            self.assertLess(np.max(np.abs(proof['stationarity'])),1e-8)
                        template_ref=json.loads((directory/'template_reference.json').read_text())
                        self.assertTrue(Path(template_ref['path']).is_relative_to(self.root/'original'))
                _,restored=self.prepare_run(risk,resume=True,snapshot=snapshot)
                self.assertEqual(restored.q,e.q)
                self.assertLessEqual(p.recourses.statistics()['capacity'],2)

    def test_pending_index_cannot_authorize_loading_without_correct_review(self):
        for changes in ({'passed':False},{'domain':'formal'},{'full_support_association_verified':False},
                        {'scenario_ids':list(reversed(self.fixture.scenario_ids))},
                        {'implementation_sources':[file_reference(reader.__file__)]},
                        {'proof_references':{}},{'formal_OOS':True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.provider(review=self.make_review(**changes))

    def test_incomplete_reordered_duplicate_or_foreign_bindings_refused(self):
        for i,bindings in enumerate((self.bindings[:-1],list(reversed(self.bindings)),[self.bindings[0]]*3)):
            with self.subTest(i=i),self.assertRaises(ValueError):
                self.prepare_index(bindings,output=f'invalid_{i}.json')
            self.assertFalse((self.root/f'invalid_{i}.json').exists())

    def test_development_review_cannot_enable_formal_provider(self):
        with self.assertRaisesRegex(ValueError,'physical support'):
            self.provider(domain='formal')

    def test_binding_and_review_tampering_after_provider_creation_are_rejected(self):
        provider=self.provider()
        path=Path(self.bindings[0]['path']);before=path.read_bytes();path.write_bytes(before+b' ')
        with self.assertRaises(ValueError):provider.load(0,self.fixture.scenarios[0])
        path.write_bytes(before)
        path=Path(self.review['path']);path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):provider.load(0,self.fixture.scenarios[0])

    def test_original_artifact_change_rejected_by_loading_and_snapshot(self):
        p,e=self.prepare_run('DRO');template=p.recourses[0];e.template(0,template)
        original=self.root/'original/templates/000/arrays.npz'
        original.write_bytes(original.read_bytes()+b'x')
        with self.assertRaises(ValueError):self.provider().load(0,self.fixture.scenarios[0])
        with self.assertRaises(ValueError):e.snapshot(self.root/'bad_snapshot.json')

    def test_resealed_foreign_pointer_cannot_replace_reviewed_template(self):
        p,e=self.prepare_run('DRO');e.template(0,p.recourses[0]);folder=e.root/'templates/000'
        source=json.loads((folder/'template_source.json').read_text())
        source['original_template_reference']=json.loads(Path(self.bindings[1]['path']).read_text())['original_template_reference']
        (folder/'template_source.json').write_text(json.dumps(source))
        record=json.loads((folder/'record.json').read_text());record['artifacts']=[file_reference(folder/'template_source.json')]
        (folder/'record.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Foreign template pointer'):_verified_record(folder/'record.json')

    def test_provider_requires_matching_input_binding_for_export(self):
        f=self.fixture;provider=self.provider()
        problem=CCGProblem(self.first,self.templates,f.probability_set)
        with self.assertRaisesRegex(ValueError,'bind the reviewed'):
            RawPlanningEvidence(self.root/'bad_export',f.case,f.tasks,problem,f.scenarios.__getitem__,
                domain='development',input_binding={},template_provider=provider)
        self.assertFalse((self.root/'bad_export').exists())

    def test_valid_foreign_admission_resealed_pointer_rejected_by_snapshot_and_resume(self):
        p,e=self.prepare_run('DRO');e.template(0,p.recourses[0])
        snapshot=e.snapshot(self.root/'before_snapshot.json')
        alternate_index=self.prepare_index(output='alternate_index.json')
        review=json.loads(Path(self.review['path']).read_text())
        review['index_reference']=alternate_index
        alternate_review=self.write('alternate_review.json',review)
        f=self.fixture
        alternate=TemplateProvider(alternate_index,alternate_review,first_stage=self.first,
            case=f.case,tasks=f.tasks,scenario_ids=f.scenario_ids,
            scenario_sha256s=[s.scenario_sha256 for s in f.scenarios],domain='development')
        # Both admissions are valid and identify the same numerics. Only the
        # original admission is bound into this run's immutable request.
        folder=e.root/'templates/000'
        self.write(str(folder/'template_source.json'),alternate.evidence_reference(0,alternate.load(0,f.scenarios[0])))
        record=json.loads((folder/'record.json').read_text())
        record['artifacts']=[file_reference(folder/'template_source.json')]
        replacement=self.write(str(folder/'record.json'),record)
        self.assertEqual(_verified_record(folder/'record.json')['template_sha256'],self.templates[0].template_sha256)
        with self.assertRaisesRegex(ValueError,'admission differs from this run'):
            e.snapshot(self.root/'invalid_snapshot.json')
        resealed=json.loads(Path(snapshot['path']).read_text());resealed['templates']=[replacement]
        foreign_snapshot=self.write('resealed_snapshot.json',resealed)
        with self.assertRaisesRegex(ValueError,'admission differs from this run'):
            self.prepare_run('DRO',resume=True,snapshot=foreign_snapshot)

    def test_resealed_foreign_artifact_with_same_basename_rejected(self):
        p,e=self.prepare_run('DRO');e.template(0,p.recourses[0]);folder=e.root/'templates/000'
        foreign=self.root/'foreign';foreign.mkdir()
        source=json.loads((folder/'template_source.json').read_text())
        ref=self.write(str(foreign/'template_source.json'),source)
        record=json.loads((folder/'record.json').read_text());record['artifacts']=[ref]
        self.write(str(folder/'record.json'),record)
        with self.assertRaisesRegex(ValueError,'canonical files in their record directory'):
            e.snapshot(self.root/'foreign_artifact_snapshot.json')

    def test_symlink_artifact_inside_record_directory_rejected(self):
        p,e=self.prepare_run('DRO');e.template(0,p.recourses[0]);folder=e.root/'templates/000'
        foreign=self.root/'foreign_source.json'
        foreign.write_bytes((folder/'template_source.json').read_bytes())
        (folder/'template_source.json').unlink()
        (folder/'template_source.json').symlink_to(foreign)
        with self.assertRaisesRegex(ValueError,'canonical files in their record directory'):
            e.snapshot(self.root/'symlink_snapshot.json')

    def test_index_preparation_refuses_original_evidence_output(self):
        with self.assertRaisesRegex(ValueError,'Output overlaps'):
            self.prepare_index(output='original/new_index.json')
        self.assertFalse((self.root/'original/new_index.json').exists())

    def test_worker_forwards_reviewed_provider_and_completes_real_small_candidate(self):
        import test_formal_planning_worker as fixtures
        from types import SimpleNamespace
        from evcs_experiments.formal_planning_worker import run_attempt
        helper=fixtures.FormalPlanningWorkerTests(methodName='runTest')
        helper.setUpClass();helper.setUp();self.addCleanup(helper.doCleanups)
        helper.request.update(template_index_reference=self.index,template_association_review_reference=self.review)
        helper.write(helper.request_path,helper.request)
        def prepare(**kwargs):
            self.assertEqual(kwargs['template_index_reference'],self.index)
            self.assertEqual(kwargs['template_association_review_reference'],self.review)
            p,e=self.prepare_run('DRO',evidence_root=kwargs['evidence_root'],
                resume=kwargs['resume_evidence'],snapshot=kwargs['resume_snapshot_reference'])
            return SimpleNamespace(problem=p,evidence=e,sink_factory=e.sink_factory,
                                   input_binding=json.loads((e.root/'request.json').read_text())['input_binding'])
        result=run_attempt(helper.invocation(),_development_fixture=helper.fixture(prepare))
        self.assertEqual(result['status'],'COMPLETE')
        self.assertTrue(result['has_candidate'])
        self.assertFalse(result['scientific_acceptance'])
        self.assertFalse(result['full_support_verified'])


if __name__=='__main__':unittest.main()
