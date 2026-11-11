import json
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from evcs_experiments.evaluation_metrics import (response_times,metrics_from_solution,
    evaluate_development_plan,plan_jaccard,plan_identity)
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.validation.s02_toys import golden_single_bus,make_fixture,manual_solution


class ResponseMetricTests(unittest.TestCase):
    def test_first_three_hours_and_contiguous_bridge_handcheck(self):
        r=response_times([0,2,3,8],[10,10,20,10],[1,1,10,0])
        self.assertEqual(r['first_reserve_h'],1.)
        self.assertEqual(r['bridge_h'],2.)
        self.assertEqual(r['early_certified_kwh'],5.)
        self.assertAlmostEqual(r['CLSR'],38/50)
        self.assertAlmostEqual(r['early_CLSR'],28/40)

    def test_partial_early_slot_uses_energy_not_slot_count(self):
        r=response_times([2,4],[10,20],[0,10],dt_h=2.)
        self.assertEqual(r['early_certified_kwh'],8.)
        self.assertAlmostEqual(r['early_CLSR'],30/40)
        self.assertEqual(r['bridge_h'],2.)

    def test_zero_demand_no_reserve_and_threshold_are_explicit(self):
        r=response_times([0,1e-5,0],[0,0,0],[0,0,0])
        for key in ('first_reserve_h','bridge_h','CLSR','early_CLSR'):self.assertIsNone(r[key])
        self.assertTrue(r['no_reserve']);json.dumps(r,allow_nan=False)
        self.assertEqual(response_times([1.1e-5],[1],[1])['first_reserve_h'],0.)
        self.assertEqual(response_times([0,0],[1,0],[1,0])['bridge_h'],0.)

    def test_invalid_metrics_do_not_become_plausible_numbers(self):
        for values in (([1],[1],[2]),([1],[0],[-1]),([float('nan')],[1],[0])):
            with self.assertRaises(ValueError):response_times(*values)


class CommonEvaluatorTests(unittest.TestCase):
    def test_certified_fixed_plan_costs_units_and_no_reoptimization(self):
        f=golden_single_bus();before=plan_identity(f.plan)
        with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No planning')):
            report,y=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan,time_limit_seconds=5)
        self.assertEqual(before,plan_identity(f.plan));self.assertEqual(report['plan_sha256_after'],before)
        c=report['costs'];self.assertAlmostEqual(sum(c[k] for k in ('Fact_usd','Fcap_usd','Fshed_usd','Fshort_usd','FV2G_usd','Fdeg_usd')),c['total_DSO_usd'])
        self.assertAlmostEqual(c['social_proxy_usd'],c['Fshed_usd']+c['Fshort_usd']+c['Fdeg_usd'])
        self.assertAlmostEqual(report['solver']['objective_usd'],1.02)
        self.assertEqual(report['metrics']['RCR'],1);self.assertEqual(report['metrics']['OTR'],1)
        self.assertEqual(report['metrics']['MBQR'],1);self.assertTrue(report['solver']['certificate']['optimal_certificate_valid'])
        json.dumps(report,allow_nan=False)

    def test_funnel_uses_eligibility_and_shortfall_uses_station_slack(self):
        f=make_fixture(periods=3,start_periods=[2],reserve=2,load_p=[[0],[1],[1]],tan_phi=0)
        report,y=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)
        self.assertEqual(report['timeseries_kw']['nameplate'],[0,15,15])
        self.assertEqual(report['energy_kwh']['procured'],4.)
        self.assertEqual(report['metrics']['shortfall_kwh'],0.)
        self.assertEqual(report['metrics']['v2g_ac_kwh'],2.)
        self.assertEqual(report['metrics']['first_reserve_h'],1.)

    def test_otr_requires_qualification_even_with_route_and_timely_arrival(self):
        f=make_fixture(available=False,reserve=2,arrival_soc=.25,tan_phi=0)
        f.scenario.task_arrival_h=np.array([0.])
        report,y=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)
        self.assertEqual(report['metrics']['RCR'],1.)
        self.assertEqual(report['metrics']['OTR'],0.)
        self.assertEqual(report['metrics']['MBQR'],0.)
        self.assertEqual(report['metrics']['shortfall_kwh'],2.)

    def test_empty_plan_has_null_ratios_and_empty_jaccard(self):
        f=make_fixture(selected=0,tan_phi=0)
        report,y=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)
        for name in ('RCR','OTR','MBQR','DRR'):self.assertIsNone(report['metrics'][name])
        self.assertTrue(plan_jaccard(f.tasks.task_ids,f.plan,f.tasks.task_ids,f.plan)['both_empty'])
        self.assertEqual(plan_jaccard(f.tasks.task_ids,f.plan,f.tasks.task_ids,f.plan)['jaccard'],1.)

    def test_invalid_physics_and_uncertified_value_are_not_reported(self):
        f=golden_single_bus();report,y=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)
        with self.assertRaisesRegex(ValueError,'physical|physics'):
            metrics_from_solution(f.case,f.tasks,f.scenario,f.plan,replace(y,p=y.p+100))
        with patch('evcs_experiments.evaluation_metrics.solve_fixed_plan') as solver:
            solver.return_value.optimal=False;solver.return_value.status='TIME_LIMIT'
            with self.assertRaisesRegex(RuntimeError,'TIME_LIMIT'):
                evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)
        f.scenario.development=False
        with self.assertRaisesRegex(ValueError,'Formal OOS'):
            evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)

    def test_integrality_tolerance_does_not_create_a_selected_population(self):
        f=make_fixture(selected=0,tan_phi=0)
        f.plan=FirstStagePlan([1e-10],f.plan.r,f.plan.task_sha256,f.plan.config_hash)
        before=plan_identity(f.plan)
        report,_=evaluate_development_plan(f.case,f.tasks,f.scenario,f.plan)
        self.assertTrue(report['metrics']['no_selected_tasks'])
        for key in ('RCR','OTR','MBQR','mean_arrival_soc_completed_selected'):
            self.assertIsNone(report['metrics'][key])
        self.assertEqual(plan_identity(f.plan),before)

    def test_jaccard_aligns_ids_and_rejects_fractional_selection(self):
        a=FirstStagePlan([1,0],[[1],[0]],'a','a');b=FirstStagePlan([0,1],[[0],[1]],'b','b')
        self.assertEqual(plan_jaccard(['A','B'],a,['B','A'],b)['jaccard'],1.)
        bad=FirstStagePlan([.5],[[1]],'a','a')
        with self.assertRaises(ValueError):plan_jaccard(['A'],bad,['A'],bad)

if __name__=='__main__':unittest.main()
