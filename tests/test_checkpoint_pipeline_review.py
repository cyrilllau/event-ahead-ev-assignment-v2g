"""Historical P2 proof roles retain strict current runtime dependencies.

Actual producer keysets and complete source names are used with synthetic small
payloads. These are closure tests, not replayed experimental acceptances.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.checkpoint_closure import verify_checkpoint_closure
from evcs_experiments.config import get_paths


class CheckpointPipelineReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True)+'\n');return file_reference(path)

    def payload(self,path,text):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return file_reference(path)

    def fixture(self):
        project=get_paths().project_root
        templates=project/'tests/fixtures/checkpoint_pipeline_proof'
        review=json.loads((templates/'pipeline_review.json').read_text())
        acceptance=json.loads((templates/'acceptance.json').read_text())
        canonical=json.loads((templates/'plan_record.json').read_text())
        evidence=Path(acceptance['tests']['log_reference']['path']).parent
        release=self.root/'release';self.release=release;self.evidence=self.root/'evidence'
        current=[];old=[]
        for row in review['planning_sources']:
            rel='src/evcs_experiments/'+row['path'].split('/src/evcs_experiments/',1)[1]
            ref=self.payload(release/rel,'# reviewed synthetic '+rel+'\n');current.append(ref)
            old.append(dict(ref,path=str(self.root/'absent_workstation'/rel)))
        review['planning_sources']=current;canonical['references']['planning_source']=old
        for item in acceptance['source_and_test_snapshots']:
            rel=item['package_relative_path'].removeprefix('source_snapshots/')
            p=release/rel
            if not p.exists():self.payload(p,'# recorded test '+rel+'\n')
            ref=file_reference(p)
            item['original_reference']=dict(ref,path=str(self.root/'absent_workstation'/rel))
            self.payload(self.evidence/item['package_relative_path'],p.read_text())
        for key,row in review['numerical_sources'].items():
            review['numerical_sources'][key]=self.payload(release/'scripts/audits'/Path(row['path']).name,'# numerical '+key+'\n')
        rows=[dict(file_reference(p),path=p.relative_to(release).as_posix()) for p in release.rglob('*') if p.is_file()]
        manifest=dict(schema_version=1,release_name=release.name,created_at_utc='2026-09-11T21:55:25+00:00',
            git_head='a'*40,working_tree_snapshot=True,files=rows)
        review['release_manifest_reference']=self.write(release/'RELEASE_MANIFEST.json',manifest)
        for field in ('independent_review_reference','publication_script_reference'):
            review[field]=self.payload(self.root/'operation'/Path(review[field]['path']).name,'recorded '+field+'\n')
        def historical_mirror(reference):
            relative=Path(reference['path']).relative_to(evidence)
            payload=self.payload(self.evidence/relative,'{}\n')
            return dict(payload,path=str(self.root/'absent_workstation'/relative))
        for field in ('previous_failed_test_log_reference','positive_summary_reference',
                      'positive_reference_closure_reference','disclosure_reference'):
            acceptance[field]=historical_mirror(acceptance[field])
        for plan in acceptance['plans']:
            for field in ('completion_reference','canonical_record_reference','solver_record_reference','budget_reference'):
                plan[field]=historical_mirror(plan[field])
        for field in canonical['references']:
            if field!='planning_source':canonical['references'][field]=historical_mirror(canonical['references'][field])
        canonical_ref=self.write(self.evidence/'plan_record.json',canonical)
        review['original_p2_canonical_evidence_reference']=canonical_ref
        log=self.payload(self.evidence/'accepted_tests.log','recorded fixture test output\n')
        review['p2_test_log_reference']=log
        acceptance['tests']['log_reference']=dict(log,path=str(self.root/'absent_workstation/accepted_tests.log'))
        acceptance['plans'][0]['canonical_record_reference']=dict(canonical_ref,path=str(self.root/'absent_workstation/plan_record.json'))
        self.review,self.acceptance,self.canonical=review,acceptance,canonical
        return self.reseal()

    def reseal(self):
        self.review['p2_acceptance_reference']=self.write(self.evidence/'acceptance.json',self.acceptance)
        return self.write(self.root/'operation/pipeline_review.json',self.review)

    def test_actual_shape_accepts_missing_historical_workstation_but_preserves_bytes(self):
        ref=self.fixture();before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        verify_checkpoint_closure(ref)
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_original_historical_reference_cannot_poison_direct_dependency_cache(self):
        ref=self.fixture();verify_checkpoint_closure(ref)
        direct=deepcopy(self.acceptance['tests']['log_reference'])
        both=self.write(self.root/'both.json',{'pipeline_review':ref,'current_dependency':direct})
        with self.assertRaises((ValueError,FileNotFoundError)):verify_checkpoint_closure(both)
        direct=self.canonical['references']['planning_source'][0]
        both=self.write(self.root/'direct_source.json',{'pipeline_review':ref,'current_dependency':direct})
        with self.assertRaises((ValueError,FileNotFoundError)):verify_checkpoint_closure(both)

    def test_current_runtime_and_mirrored_proof_bytes_stay_strict(self):
        for field in ('planning_source','numerical_source','acceptance','canonical','log'):
            with self.subTest(field=field):
                ref=self.fixture()
                row=(self.review['planning_sources'][0] if field=='planning_source' else
                     next(iter(self.review['numerical_sources'].values())) if field=='numerical_source' else
                     self.review[{'acceptance':'p2_acceptance_reference','canonical':'original_p2_canonical_evidence_reference','log':'p2_test_log_reference'}[field]])
                Path(row['path']).write_text('tampered bytes')
                with self.assertRaises(ValueError):verify_checkpoint_closure(ref)

    def test_resealed_unknown_schema_scope_budget_and_source_mismatch_refused(self):
        for mode in ('unknown','budget','formal','missing_source','duplicate_source','snapshot_mismatch','canonical_mismatch'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode=='unknown':self.review['unchecked_proof']={}
                if mode=='budget':self.review['cumulative_wall_seconds']=7201
                if mode=='formal':self.acceptance['formal_OOS']=True
                if mode=='missing_source':self.review['planning_sources'].pop()
                if mode=='duplicate_source':self.review['planning_sources'].append(deepcopy(self.review['planning_sources'][0]))
                if mode=='snapshot_mismatch':self.acceptance['source_and_test_snapshots'][0]['original_reference']['sha256']='b'*64
                if mode=='canonical_mismatch':
                    self.canonical['references']['planning_source'][0]['sha256']='b'*64
                    self.review['original_p2_canonical_evidence_reference']=self.write(self.evidence/'plan_record.json',self.canonical)
                    self.acceptance['plans'][0]['canonical_record_reference']['sha256']=self.review['original_p2_canonical_evidence_reference']['sha256']
                    self.acceptance['plans'][0]['canonical_record_reference']['size_bytes']=self.review['original_p2_canonical_evidence_reference']['size_bytes']
                with self.assertRaises(ValueError):verify_checkpoint_closure(self.reseal())


if __name__=='__main__':unittest.main()
