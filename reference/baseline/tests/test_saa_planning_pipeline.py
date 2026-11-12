"""Real tiny SAA numerical phase integration and interrupted-Q resume."""
from dataclasses import replace
import json
from pathlib import Path
import time
from types import MappingProxyType
import unittest
from unittest.mock import patch

from evcs_experiments import saa_planning_pipeline as m
from evcs_experiments import saa_fixed_plan
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.config import get_paths
from evcs_experiments.recourse import solve_fixed_plan
from evcs_experiments.saa_streaming import assemble_streaming_saa
import test_saa_fixed_plan_audit as fixture_module


class SAAPipelineTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture_module.SAAQAuditTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        f=self.f;toy=f.f.toy
        self.model=assemble_streaming_saa(f.first,f.f.templates,scenario_ids=toy.scenario_ids,
            scenario_sha256s=[s.scenario_sha256 for s in toy.scenarios],empirical_probabilities=toy.empirical_probabilities,
            scenario_cost_constants_usd=f.f.kw['scenario_cost_constants_usd'],input_binding={'fixture':'pipeline'},
            output_directory=f.root/'model',max_coefficient_bytes=50_000_000)
        sources=[file_reference(p) for p in sorted((get_paths().project_root/'src/evcs_experiments').rglob('*.py'))]
        self.args=dict(model_manifest_reference=file_reference(self.model.root/'manifest.json'),first_stage=f.first,
            recourses=f.f.templates,case=toy.case,tasks=toy.tasks,scene_loader=toy.scenarios.__getitem__,
            input_binding={'fixture':'pipeline'},scenario_ids=toy.scenario_ids,scenario_sha256s=[s.scenario_sha256 for s in toy.scenarios],
            template_references=f.templates,helper_reference=f.args['helper_reference'],source_references=sources,
            output_directory=f.root/'pipeline',deadline=time.monotonic()+30,cumulative_budget_seconds=30)

    def run_pipeline(self,**kw):return m.run_saa_numerical_phases(**{**self.args,**kw})

    def test_formal_admission_and_exact_budget_cannot_use_development_model(self):
        with self.assertRaisesRegex(ValueError,'admission'):
            self.run_pipeline(domain='formal')
        admission_path=self.f.root/'phase_admission.json'
        admission_path.write_text(json.dumps(dict(kind='SAA_FORMAL_PHASE_ADMISSION_V1',domain='formal',
            model_manifest_reference=self.args['model_manifest_reference'],planning_source_references=self.args['source_references'])))
        ref=file_reference(admission_path)
        with self.assertRaisesRegex(ValueError,'resource policy'):
            self.run_pipeline(domain='formal',formal_admission_reference=ref,master_seconds=599.,
                audit_reserve_seconds=600.,seal_reserve_seconds=10.,cumulative_budget_seconds=7200.)
        with self.assertRaisesRegex(ValueError,'model/phase domain'):
            self.run_pipeline(domain='formal',formal_admission_reference=ref,master_seconds=600.,
                audit_reserve_seconds=600.,seal_reserve_seconds=10.,cumulative_budget_seconds=7200.)
        with self.assertRaisesRegex(ValueError,'Development'):
            self.run_pipeline(formal_admission_reference=ref)

    def test_real_master_q_and_independent_join_complete(self):
        result=self.run_pipeline()
        self.assertEqual(result['status'],'COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES')
        audit=json.loads(Path(result['candidate_audit_reference']['path']).read_text())
        self.assertEqual(audit['status'],'PASS_SAA_CANDIDATE_NUMERICAL_AUDIT')
        self.assertAlmostEqual(audit['upper_bound_usd'],-69.73333333333333)
        self.assertFalse(result['formal_admission']);self.assertFalse(result['formal_OOS'])

    def test_partial_q_resume_never_repeats_master_or_completed_lp(self):
        calls=[]
        def solve(t,p,**kw):
            calls.append(t.scenario_sha256);raw=solve_fixed_plan(t,p,**kw)
            if len(calls)==2:return replace(raw,status='LIMIT_REACHED',solver_status=1,optimal=False,solution=None,objective_usd=None,
                certificate=MappingProxyType({'optimal_certificate_valid':False}))
            return raw
        with patch.object(saa_fixed_plan,'solve_fixed_plan',side_effect=solve):first=self.run_pipeline()
        self.assertEqual(first['status'],'TIME_LIMIT_FIXED_Q');before=len(calls)
        with patch.object(m.saa_streaming_solve,'solve_streaming_saa',side_effect=AssertionError('No master repeat')), \
             patch.object(saa_fixed_plan,'solve_fixed_plan',side_effect=solve):
            result=self.run_pipeline(resume_checkpoint_reference=first['checkpoint_reference'],
                prior_accounted_seconds=first['cumulative_provisional_seconds']+.01)
        self.assertEqual(result['status'],'COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES')
        self.assertEqual(len(calls)-before,2);self.assertEqual(calls.count(calls[0]),1)
        self.assertEqual(file_reference(first['checkpoint_reference']['path']),first['checkpoint_reference'])

    def test_completed_resume_raw_primal_tamper_refused_without_solver(self):
        result=self.run_pipeline();q=json.loads(Path(result['fixed_q_checkpoint_reference']['path']).read_text())
        row=json.loads(Path(q['attempts'][0]['path']).read_text())
        p=Path(next(r['path'] for r in row['artifacts'] if Path(r['path']).name=='primal.npy'))
        b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
        with patch.object(m.saa_streaming_solve,'solve_streaming_saa',side_effect=AssertionError('No master repeat')):
            with self.assertRaisesRegex(ValueError,'changed'):
                self.run_pipeline(resume_checkpoint_reference=result['checkpoint_reference'],
                    prior_accounted_seconds=result['cumulative_provisional_seconds'])

    def test_complete_resume_reload_deadline_overrun_refused(self):
        result=self.run_pipeline();clock=[100.]
        def load(*args,**kwargs):
            clock[0]=121.
            return self.model
        with patch.object(m.time,'monotonic',side_effect=lambda:clock[0]),patch.object(m.saa_streaming_store,'load_streaming_saa',side_effect=load),patch.object(m.saa_streaming_solve,'solve_streaming_saa',side_effect=AssertionError('No solve')):
            with self.assertRaises(TimeoutError):
                self.run_pipeline(resume_checkpoint_reference=result['checkpoint_reference'],
                    prior_accounted_seconds=result['cumulative_provisional_seconds'],deadline=120.)

    def test_master_admission_stop_then_resume(self):
        with patch.object(m.saa_streaming_solve,'solve_streaming_saa',side_effect=AssertionError('No short-budget master')), patch.object(m.saa_streaming_store,'load_streaming_saa',side_effect=AssertionError('No short-budget reload')):
            first=self.run_pipeline(deadline=time.monotonic()+3)
        self.assertEqual(first['status'],'TIME_LIMIT_BEFORE_MASTER')
        result=self.run_pipeline(resume_checkpoint_reference=first['checkpoint_reference'],
            prior_accounted_seconds=first['cumulative_provisional_seconds'])
        self.assertEqual(result['status'],'COMPLETE_DEVELOPMENT_SAA_NUMERICAL_PHASES')

    def test_prior_reset_and_changed_input_refused(self):
        result=self.run_pipeline()
        with self.assertRaisesRegex(ValueError,'reset'):
            self.run_pipeline(resume_checkpoint_reference=result['checkpoint_reference'],prior_accounted_seconds=0)
        with self.assertRaisesRegex(ValueError,'binding'):
            self.run_pipeline(resume_checkpoint_reference=result['checkpoint_reference'],
                prior_accounted_seconds=result['cumulative_provisional_seconds'],input_binding={'fixture':'foreign'})

if __name__=='__main__':unittest.main()
