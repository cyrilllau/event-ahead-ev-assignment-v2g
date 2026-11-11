"""One genuine development recourse solve through the prepared reuse adapter."""
import importlib.util
import os
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.ccg import CCGProblem
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.s03_toys import build_golden_cases

p=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))/'scripts/diagnostics/run_s05_candidate_evaluation.py'
s=importlib.util.spec_from_file_location('candidate_real_callback_component',p)
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)


class RealCallbackTest(unittest.TestCase):
    def test_original_raw_exporter_and_solver_with_reused_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=build_golden_cases()[0];scene=f.scenarios[0]
            first=assemble_first_stage(f.case,f.tasks);template=assemble_recourse(f.case,f.tasks,scene)
            # Only index0 is evaluated. Other explicit development support ids
            # make the component's full-axis guard real without formal data.
            ids=(scene.scenario_id,)+tuple('development/unused/'+str(i) for i in range(1,160))
            P=ProbabilitySet(ids,(),np.zeros((160,0)),np.array([]),np.array([]))
            cache=LazyRecourses(ids,[scene.scenario_sha256]*160,first.matrix_sha256,{'fixture':'real_callback'},lambda i:template)
            problem=CCGProblem(first,cache,P,np.zeros(160),'DRO')
            old=RawPlanningEvidence(root/'old',f.case,f.tasks,problem,lambda i:scene,domain='development',input_binding={'fixture':'real_callback'})
            oldtr=old.template(0,template)
            history=[d.write(root/f'prefix{i}.json',{}) for i in range(2)]
            for i,sid in enumerate(ids):
                # Actual adapter locates the conventional template pointer name.
                folder=root/f'event{i}';folder.mkdir();ptr=d.write(folder/'template_reference.json',oldtr)
                history.append(d.write(folder/'record.json',dict(event='recourse',scenario_index=i,scenario_id=sid,artifacts=[ptr])))
            history.append(d.write(root/'probability.json',{}));hr=d.write(root/'accepted_history.json',history)
            state=dict(identity=problem.identity,problem_sha256=canonical_hash(problem.identity),pending={'recourse_certificates':[{'template_sha256':template.template_sha256}]*160})
            cp=d.write(root/'accepted_checkpoint.json',dict(state=state,sha256=canonical_hash(state)))
            card=d.write(root/'acceptance.json',dict(status='PASS_SAVED160_COMPONENT_REVIEW',complete_lp_count=160,references={'history':hr,'checkpoint':cp},problem_sha256=state['problem_sha256']))
            e=RawPlanningEvidence(root/'new',f.case,f.tasks,problem,lambda i:scene,domain='development',input_binding={'fixture':'real_callback'})
            adapter=d.ReusedTemplateEvidenceAdapter(e,old_history_reference=hr,old_checkpoint_reference=cp,saved160_acceptance_reference=card,output=root/'reuse').install()
            v=np.zeros(first.A.shape[1])
            vector=root/'candidate.npz';np.savez_compressed(vector,vector=v)
            resultref=d.write(root/'native.json',{'development_fixture':True})
            binding=d.write(root/'candidate_binding.json',dict(result_reference=resultref,vector_reference=d.reference(vector),vector_key='vector',first_stage_slice=[0,len(v)],first_stage_sha256=first.matrix_sha256))
            plan=d.extract_candidate(binding,first);self.assertTrue(check_first_stage(f.case,f.tasks,plan)['passed'])
            clock=[0.];calls=[]
            def real_solve(t,p,time_limit_seconds):
                calls.append(t.template_sha256);answer=solve_fixed_plan(t,p,time_limit_seconds=5);clock[0]=60.;return answer
            prepared=SimpleNamespace(case=f.case,tasks=f.tasks,problem=problem,evidence=e)
            with patch.object(d,'time',SimpleNamespace(monotonic=lambda:clock[0])):
                answer=d.evaluate_prepared_candidate(prepared,plan,candidate_binding_reference=binding,output=root/'eval',deadline=60.,
                    _development_hooks={'check_first_stage':check_first_stage,'solve':real_solve})
            self.assertEqual(answer['status'],'TIME_LIMIT');self.assertEqual(answer['completed_count'],1);self.assertEqual(len(calls),1)
            er=answer['recourse_event_references'][0];row=d.read(er);folder=Path(er['path']).parent
            self.assertEqual(row['status'],'RAW_EVIDENCE_RETAINED_PENDING_AUDIT')
            self.assertTrue(json.loads((folder/'physical_check.json').read_text())['passed'])
            self.assertEqual(json.loads((folder/'template_reference.json').read_text()),oldtr)
            with np.load(folder/'dual_kkt.npz') as z:self.assertLess(np.max(abs(z['stationarity'])),1e-8)
            self.assertFalse((e.root/'templates/000').exists());self.assertEqual(len(adapter.used),1)
            self.assertIsNotNone(answer['q_usd'][0]);self.assertTrue(all(x is None for x in answer['q_usd'][1:]))
            self.assertFalse(answer['probability_evaluated']);self.assertFalse(answer['scientific_acceptance'])

if __name__=='__main__':unittest.main()
