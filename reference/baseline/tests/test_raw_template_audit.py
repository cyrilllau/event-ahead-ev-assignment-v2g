"""Admission-only audit of genuine small CCG evidence and resealed substitutions."""
import json
from pathlib import Path
import unittest

from evcs_experiments.ccg import CCGProblem, run_ccg
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.raw_template_audit import audit_template_envelope
from evcs_experiments.recourse_template_index import TemplateProvider
import test_recourse_template_index as template_fixtures


class RawTemplateAuditTests(unittest.TestCase):
    def setUp(self):
        self.helper=template_fixtures.TemplateIndexTests(methodName='runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.root=self.helper.root

    def run_evidence(self, provider=True):
        h=self.helper;f=h.fixture
        if provider:
            problem,evidence=h.prepare_run('DRO')
        else:
            cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],
                h.first.matrix_sha256,{'fixture':f.name},h.templates.__getitem__,capacity=2)
            problem=CCGProblem(h.first,cache,f.probability_set,f.scenario_cost_constants_usd)
            evidence=RawPlanningEvidence(self.root/'legacy',f.case,f.tasks,problem,f.scenarios.__getitem__,
                domain='development',input_binding={'fixture':f.name})
        result=run_ccg(problem,evidence_sink=evidence.sink_factory(self.root/('provider_call' if provider else 'legacy_call')),
            wall_seconds=20,rmp_time_limit_seconds=3,oracle_time_limit_seconds=3,
            mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
        self.assertEqual(result.status,'CONVERGED')
        self.assertAlmostEqual(result.state['upper_bound_usd'],f.expected['DRO']['objective_usd'],places=7)
        snapshot=evidence.snapshot(self.root/('provider_snapshot.json' if provider else 'legacy_snapshot.json'))
        return evidence,snapshot

    def reseal_record(self, folder):
        path=folder/'record.json';value=json.loads(path.read_text())
        value['artifacts']=[file_reference(p) for p in sorted(folder.iterdir()) if p.name!='record.json']
        return self.helper.write(str(path),value)

    def reseal_snapshot(self, evidence, snapshot):
        value=json.loads(Path(snapshot['path']).read_text())
        value['templates']=[file_reference(p) for p in sorted((evidence.root/'templates').glob('*/record.json'))]
        value['events']=[file_reference(p) for p in sorted((evidence.root/'events').glob('*/record.json'))]
        self.helper.write(str(evidence.root/'history.json'),value['events'])
        return self.helper.write('resealed_snapshot.json',value)

    def audit(self, evidence, snapshot):
        return audit_template_envelope(snapshot,expected_request_reference=file_reference(evidence.root/'request.json'))

    def test_real_legacy_and_provider_ccg_envelopes_do_not_grant_numerical_acceptance(self):
        for provider in (False,True):
            with self.subTest(provider=provider):
                evidence,snapshot=self.run_evidence(provider)
                result=self.audit(evidence,snapshot)
                self.assertEqual(result['status'],'PASS_TEMPLATE_ENVELOPE_ONLY')
                self.assertEqual(result['template_count'],len(self.helper.fixture.scenario_ids))
                self.assertTrue(result['recourse_events'])
                for key in ('numerical_audit_passed','scientific_acceptance','full_support_verified','formal_OOS'):
                    self.assertIs(result[key],False)
                self.assertEqual(result['solver_calls'],0)

    def test_valid_foreign_admission_resealed_pointer_is_rejected(self):
        evidence,snapshot=self.run_evidence();h=self.helper;f=h.fixture
        alternate_index=h.prepare_index(output='alternate_index.json')
        review=json.loads(Path(h.review['path']).read_text());review['index_reference']=alternate_index
        alternate_review=h.write('alternate_review.json',review)
        alternate=TemplateProvider(alternate_index,alternate_review,first_stage=h.first,case=f.case,
            tasks=f.tasks,scenario_ids=f.scenario_ids,scenario_sha256s=[s.scenario_sha256 for s in f.scenarios],domain='development')
        folder=evidence.root/'templates/000'
        h.write(str(folder/'template_source.json'),alternate.evidence_reference(0,alternate.load(0,f.scenarios[0])))
        self.reseal_record(folder)
        with self.assertRaisesRegex(ValueError,'Pointer admission differs from run'):
            self.audit(evidence,self.reseal_snapshot(evidence,snapshot))

    def test_resealed_lp_callback_foreign_template_is_rejected(self):
        evidence,snapshot=self.run_evidence();h=self.helper
        events=[json.loads(Path(r['path']).read_text()) for r in evidence.events]
        offset=next(i for i,event in enumerate(events) if event['event']=='recourse')
        event=events[offset];folder=Path(evidence.events[offset]['path']).parent
        other=(event['scenario_index']+1)%len(h.fixture.scenario_ids)
        foreign=json.loads(Path(h.bindings[other]['path']).read_text())['original_template_reference']
        h.write(str(folder/'template_reference.json'),foreign);self.reseal_record(folder)
        with self.assertRaisesRegex(ValueError,'LP callback selects a template outside run admission'):
            self.audit(evidence,self.reseal_snapshot(evidence,snapshot))

    def test_resealed_same_basename_artifact_from_foreign_directory_is_rejected(self):
        evidence,snapshot=self.run_evidence();h=self.helper
        folder=evidence.root/'templates/000';foreign=self.root/'foreign';foreign.mkdir()
        ref=h.write(str(foreign/'template_source.json'),json.loads((folder/'template_source.json').read_text()))
        record=json.loads((folder/'record.json').read_text());record['artifacts']=[ref]
        h.write(str(folder/'record.json'),record)
        with self.assertRaisesRegex(ValueError,'Foreign raw artifact directory'):
            self.audit(evidence,self.reseal_snapshot(evidence,snapshot))


if __name__=='__main__':unittest.main()
