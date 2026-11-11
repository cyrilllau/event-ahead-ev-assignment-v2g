"""Hand-optimum and interruption tests for cumulative planning scheduling."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evcs_experiments.ccg import CCGProblem,run_ccg
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.extensive import solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.planning_budget import run_budgeted_ccg
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.validation.s02_toys import _serial
from evcs_experiments.validation.s03_toys import build_golden_cases


BUDGET=dict(cumulative_wall_seconds=20.,single_master_seconds=3.,
            final_certificate_reserve_seconds=2.,shutdown_reserve_seconds=.1)
OPTIONS=dict(mip_relative_gap=1e-9,optimality_relative_gap=1e-8,oracle_time_limit_seconds=2.)


def sink_factory(directory):
    counter=[0]
    def sink(event, iteration, **objects):
        counter[0]+=1
        value=dict(event=event,iteration=iteration)
        if 'result' in objects:value['raw_solver_result']=objects['result'].as_dict()
        if 'record' in objects:value['record']=objects['record']
        with (directory/f'event_{counter[0]:03d}.json').open('x') as f:
            json.dump(_serial(value),f,allow_nan=False)
    return sink


class PlanningBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=build_golden_cases()
        cls.problems={}
        for i,f in enumerate(cls.fixtures):
            for risk in ('DRO','ARO'):
                cls.problems[i,risk]=CCGProblem(assemble_first_stage(f.case,f.tasks),
                    tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios),
                    f.probability_set,f.scenario_cost_constants_usd,risk)

    def run_driver(self,problem,root,**kwargs):
        return run_budgeted_ccg(problem,outputroot=root,budget=BUDGET,input_binding={'fixture':'hand'},
            engine_options=OPTIONS,evidence_sink_factory=sink_factory,**kwargs)

    def test_six_known_optima_and_normal_gaps_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            for (i,risk),problem in self.problems.items():
                with self.subTest(i=i,risk=risk):
                    root=Path(tmp)/f'{i}_{risk}'
                    result=self.run_driver(problem,root)
                    self.assertEqual(result['status'],'CONVERGED')
                    state=json.loads((root/'checkpoint.json').read_text())['state']
                    self.assertAlmostEqual(state['upper_bound_usd'],self.fixtures[i].expected[risk]['objective_usd'],places=7)
                    self.assertAlmostEqual(state['lower_bound_usd'],state['upper_bound_usd'],places=7)
                    self.assertTrue(result['independent_scientific_audit_required'])
                    for ref in result['calls']:
                        record=json.loads(Path(ref['path']).read_text())
                        self.assertLessEqual(record['granted_seconds'],record['remaining_before_seconds']-2.-.1+1e-3)
                        self.assertEqual(record['max_completed_iterations'],1)
                        self.assertFalse(record['full_spec_pass'])

    def test_reserve_prevents_new_master_and_parent_budget_also_binds(self):
        with tempfile.TemporaryDirectory() as tmp,patch('evcs_experiments.planning_budget.run_ccg',side_effect=AssertionError('no master budget')):
            a=self.run_driver(self.problems[0,'DRO'],Path(tmp)/'cumulative',spent_before_seconds=16.)
            b=self.run_driver(self.problems[0,'DRO'],Path(tmp)/'parent',available_wall_seconds=4.)
            for result in (a,b):
                self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
                self.assertEqual(result['calls'],[])

    def test_pending_drain_never_starts_new_master_and_preserves_native_status(self):
        calls={'oracle':0,'master':0}
        def oracle(*args,**kw):
            calls['oracle']+=1
            result=solve_fixed_plan(*args,**kw)
            if calls['oracle']==1:
                return replace(result,optimal=False,status='TIME_LIMIT',solver_status=1,solution=None)
            return result
        def master(*args,**kw):
            calls['master']+=1
            if calls['master']>1:raise AssertionError('Pending-only drain started a new master')
            return solve_extensive(*args,**kw)
        with tempfile.TemporaryDirectory() as tmp,patch('evcs_experiments.ccg.solve_fixed_plan',side_effect=oracle),patch('evcs_experiments.ccg.solve_extensive',side_effect=master):
            root=Path(tmp)/'run';result=self.run_driver(self.problems[0,'DRO'],root)
            records=[json.loads(Path(ref['path']).read_text()) for ref in result['calls']]
            self.assertEqual([r['mode'] for r in records],['ordinary','pending_only_final_certificate'])
            self.assertEqual(records[0]['status'],'TIME_LIMIT')
            self.assertEqual(calls['master'],1)
            self.assertEqual(len(records),2)
            state=json.loads((root/'checkpoint.json').read_text())['state']
            self.assertIsNone(state['pending'])
            self.assertEqual(len(state['trace']),1)
            self.assertEqual(len(state['best']['q_usd']),len(self.problems[0,'DRO'].probability_set.support_ids))
            self.assertIn(result['status'],('TIME_LIMIT','CONVERGED'))

    def test_resume_rejects_cumulative_reset_and_modified_raw_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';result=self.run_driver(self.problems[0,'DRO'],root)
            snapshot=root/'independent_driver_snapshot.json';snapshot.write_bytes((root/'driver_state.json').read_bytes())
            ref=file_reference(snapshot)
            with self.assertRaisesRegex(ValueError,'cannot reset'):
                self.run_driver(self.problems[0,'DRO'],root,resume=True,previous_driver_snapshot=ref,spent_before_seconds=0.)
            record=json.loads(Path(result['calls'][0]['path']).read_text())
            Path(record['artifacts'][0]['path']).write_text('changed evidence')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                self.run_driver(self.problems[0,'DRO'],root,resume=True,previous_driver_snapshot=ref,
                                spent_before_seconds=result['cumulative_seconds'])

    def test_converged_resume_verifies_evidence_without_another_solver_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';first=self.run_driver(self.problems[0,'DRO'],root)
            snapshot=root/'driver_snapshot.json';snapshot.write_bytes((root/'driver_state.json').read_bytes())
            with patch('evcs_experiments.planning_budget.run_ccg',side_effect=AssertionError('converged plan must be reused')):
                second=self.run_driver(self.problems[0,'DRO'],root,resume=True,
                    previous_driver_snapshot=file_reference(snapshot),spent_before_seconds=first['cumulative_seconds'])
            self.assertEqual(first['calls'],second['calls'])
            self.assertTrue(second['verified_reuse_without_solver'])
            self.assertGreater(second['cumulative_seconds'],first['cumulative_seconds'])

    def test_nonpreemptible_overrun_is_not_a_successful_driver_status(self):
        clock=[0.]
        def overrun(*args,**kwargs):
            value=run_ccg(*args,**kwargs)
            clock[0]=21.
            return value
        with tempfile.TemporaryDirectory() as tmp,patch('evcs_experiments.planning_budget.time.monotonic',side_effect=lambda:clock[0]),patch('evcs_experiments.planning_budget.run_ccg',side_effect=overrun):
            result=self.run_driver(self.problems[0,'DRO'],Path(tmp)/'run')
            self.assertEqual(result['status'],'BUDGET_OVERRUN')
            self.assertEqual(result['cumulative_seconds'],21.)
            self.assertEqual(len(result['calls']),1)
            self.assertIn(result['raw_engine_status'],('ITERATION_LIMIT','CONVERGED'))

    def test_sink_factory_cannot_silently_disable_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError,'must return a callable'):
                run_budgeted_ccg(self.problems[0,'DRO'],outputroot=root,budget=BUDGET,
                    input_binding={},engine_options=OPTIONS,evidence_sink_factory=lambda directory:None)
            self.assertEqual(json.loads((root/'driver_state.json').read_text())['status'],'FAILED_EXCEPTION')

    def test_engine_options_cannot_bypass_iteration_or_wall_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'cannot override'):
                run_budgeted_ccg(self.problems[0,'DRO'],outputroot=Path(tmp)/'bad',budget=BUDGET,
                    input_binding={},engine_options={'wall_seconds':7200},evidence_sink_factory=sink_factory)


if __name__=='__main__':unittest.main()
