from dataclasses import replace
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from evcs_experiments.ccg import CCGProblem,run_ccg
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.validation.validate_s03_scale_ccg import (
    execution_boundary,execution_budget,execution_job_reference)

class ScaleCCGCLITests(unittest.TestCase):
    def checked_cli(self,extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.json';p.write_text('{}')
            args=['validate','scale-ccg','--s03-i03-acceptance',str(Path(tmp)/'card.json'),
                  '--profile',str(p),'--output',str(Path(tmp)/'out'),*extra]
            code='from evcs_experiments.cli import main;raise SystemExit(main('+repr(args)+'))'
            env={k:v for k,v in os.environ.items() if k!='EVCS_JOB_DIRECTORY'}
            r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10,env=env)
            self.assertEqual(r.returncode,2)
            self.assertFalse((Path(tmp)/'out').exists())
            return json.loads(r.stderr)['error']

    def test_common_case_root_reaches_source_validation_without_starting_solver(self):
        self.assertEqual(self.checked_cli(),'Pinned original scale diagnostic changed')

    def test_long_budget_requires_explicit_completion_mode(self):
        self.assertIn('probe budget requires',self.checked_cli(['--wall-seconds','1800']))
        self.assertEqual(self.checked_cli(['--run-mode','completion','--job-wall-seconds','2400',
            '--wall-seconds','2390','--rmp-seconds','1800']),
            'Pinned original scale diagnostic changed')

    def test_pending_only_cannot_start_a_fresh_master(self):
        self.assertIn('Pending evaluation requires --resume',
            self.checked_cli(['--require-pending-evaluation']))


class ScaleExecutionBudgetTests(unittest.TestCase):
    def test_default_probe_preserves_600_second_cap_and_stops_after_one_iteration(self):
        budget=execution_budget()
        self.assertEqual((budget['job_wall_seconds'],budget['program_wall_seconds'],
            budget['rmp_time_limit_seconds'],budget['max_completed_iterations']),(600.,590.,300.,1))

    def test_completion_allows_long_master_without_changing_accuracy_settings(self):
        budget=execution_budget('completion',job_wall_seconds=3600,wall_seconds=3590,rmp_seconds=1800)
        self.assertEqual(budget['rmp_time_limit_seconds'],1800)
        self.assertEqual(execution_budget('completion')['program_wall_seconds'],7190)
        self.assertFalse(any('gap' in key or 'tolerance' in key for key in budget))

    def test_over_caps_bad_reserve_and_invalid_numbers_are_rejected(self):
        cases=[dict(job_wall_seconds=601),dict(wall_seconds=591),dict(rmp_seconds=591),
            dict(run_mode='completion',job_wall_seconds=7201),
            dict(run_mode='completion',wall_seconds=7200),
            dict(run_mode='completion',job_wall_seconds=1800,wall_seconds=1791),
            dict(max_completed_iterations=0),dict(max_completed_iterations=True),
            dict(max_completed_iterations=1.5),dict(run_mode='formal')]
        for key in ('wall_seconds','rmp_seconds','job_wall_seconds'):
            cases.extend({key:value} for value in (0,-1,True,float('inf'),float('nan')))
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                execution_budget(**kwargs)

    def test_supervisor_budget_must_match_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json'
            path.write_text(json.dumps(dict(job_id='test',timeout_seconds=600)))
            with patch.dict(os.environ,EVCS_JOB_DIRECTORY=tmp):
                result=execution_job_reference(execution_budget())
                self.assertEqual(result['timeout_seconds'],600)
                self.assertEqual(result['path'],str(path.resolve()))
                self.assertEqual(len(result['sha256']),64)
                with self.assertRaisesRegex(ValueError,'supervising job timeout'):
                    execution_job_reference(execution_budget('completion'))

    def test_pending_boundary_requires_one_iteration_and_actual_pending_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            for resume,limit in ((False,1),(True,2)):
                with self.assertRaisesRegex(ValueError,'Pending evaluation requires'):
                    execution_boundary(path,resume=resume,require_pending=True,max_completed_iterations=limit)
            with self.assertRaisesRegex(ValueError,'Resume checkpoint missing'):
                execution_boundary(path,resume=True,require_pending=True,max_completed_iterations=1)
            state=dict(pending=None,trace=[])
            path.write_text(json.dumps(dict(state=state,sha256=canonical_hash(state))))
            with self.assertRaisesRegex(ValueError,'no pending iteration'):
                execution_boundary(path,resume=True,require_pending=True,max_completed_iterations=1)
            state['pending']=dict(iteration=2,q_usd=[1.,2.])
            path.write_text(json.dumps(dict(state=state,sha256='wrong')))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                execution_boundary(path,resume=True,require_pending=True,max_completed_iterations=1)

    def test_pending_evaluation_finishes_without_repeating_lp_or_starting_another_master(self):
        fixture=build_golden_cases()[0]
        problem=CCGProblem(assemble_first_stage(fixture.case,fixture.tasks),
            tuple(assemble_recourse(fixture.case,fixture.tasks,s) for s in fixture.scenarios),
            fixture.probability_set,fixture.scenario_cost_constants_usd)
        calls=[]
        def interrupted(template,plan,**kwargs):
            calls.append(template.scenario_sha256)
            result=solve_fixed_plan(template,plan,**kwargs)
            return replace(result,status='LIMIT_REACHED',solver_status=1,optimal=False) if len(calls)==2 else result
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            with patch('evcs_experiments.ccg.solve_fixed_plan',side_effect=interrupted):
                initial=run_ccg(problem,checkpoint_path=path,wall_seconds=10)
            self.assertEqual(initial.status,'TIME_LIMIT')
            budget=execution_budget('completion')
            boundary=execution_boundary(path,resume=True,require_pending=True,
                max_completed_iterations=budget['max_completed_iterations'])
            self.assertEqual(boundary['initial_partial_q_count'],1)
            with patch('evcs_experiments.ccg.solve_extensive',side_effect=AssertionError('Unexpected master')):
                with patch('evcs_experiments.ccg.solve_fixed_plan',wraps=solve_fixed_plan) as oracle:
                    result=run_ccg(problem,checkpoint_path=path,resume=True,wall_seconds=10,
                        max_iterations=budget['max_completed_iterations'])
            self.assertEqual(oracle.call_count,len(problem.recourses)-1)
            self.assertIsNone(result.state['pending'])
            self.assertEqual(len(result.state['trace'])-boundary['initial_completed_iterations'],1)
            self.assertEqual(result.status,'ITERATION_LIMIT')
            self.assertIsNotNone(result.state['upper_bound_usd'])

if __name__=='__main__':unittest.main()
