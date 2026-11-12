"""Hand-derived planning optimism and physical-corridor paired failures."""
import json
import unittest

import numpy as np

from evcs_experiments.evaluation_mechanisms import planning_optimism, shared_corridor_cofailure


class PlanningOptimismTests(unittest.TestCase):
    def test_units_mean_and_negative_optimism_are_preserved(self):
        r=planning_optimism(10.,[4.,8.],sample_ids=['s0','s1'])
        self.assertEqual(r['r1_mean_certified_kwh'],6.)
        self.assertEqual(r['optimism_ratio'],.4)
        self.assertEqual(r['optimism_percent'],40.)
        self.assertEqual(planning_optimism(10.,[12.,18.],sample_ids=['a','b'])['optimism_ratio'],-.5)

    def test_zero_prediction_and_empty_samples_are_explicit_na(self):
        for prediction,values,ids in ((0.,[0.],['s']), (0.,[2.],['s']), (5.,[],[])):
            r=planning_optimism(prediction,values,sample_ids=ids)
            self.assertEqual(r['status'],'NA');self.assertIsNone(r['optimism_ratio'])
            self.assertIsNone(r['optimism_percent']);self.assertTrue(r['na_reasons'])
            json.dumps(r,allow_nan=False)

    def test_values_and_sample_identity_are_bound_and_invalid_inputs_rejected(self):
        a=planning_optimism(10.,[4.,8.],sample_ids=['a','b'])
        b=planning_optimism(10.,[4.,8.],sample_ids=['b','a'])
        self.assertNotEqual(a['input_sha256'],b['input_sha256'])
        for prediction,values,ids in ((True,[1.],['a']),(-1,[1.],['a']),
                (1,[float('nan')],['a']),(1,[-1.],['a']),(1,[1.,2.],['a']),
                (1,[1.,2.],['a','a'])):
            with self.subTest(prediction=prediction,values=values),self.assertRaises(ValueError):
                planning_optimism(prediction,values,sample_ids=ids)


class CorridorCofailureTests(unittest.TestCase):
    def base(self,**changes):
        args=dict(task_ids=['A','B','C'],selection=[1,1,1],
            routes_by_task={'A':[1,3],'B':[2],'C':[5]},
            arc_to_corridor={1:10,2:10,3:20,5:20},
            failures=[[1,1,1],[1,1,1],[0,0,1],[0,0,1]],
            sample_ids=['s0','s1','s2','s3'])
        args.update(changes)
        return shared_corridor_cofailure(**args)

    def test_pairwise_joint_marginal_product_and_excess_handcheck(self):
        r=self.base()
        self.assertEqual(r['shared_pair_count'],2);self.assertEqual(r['pair_scenario_count'],8)
        self.assertEqual(r['joint_failure_count'],4)
        self.assertEqual(r['joint_failure_probability'],.5)
        self.assertEqual(r['marginal_product'],.375)
        self.assertEqual(r['excess_over_marginal_product'],.125)
        ab,ac=r['pairs']
        self.assertEqual((ab['task_i'],ab['task_j']),('A','B'))
        self.assertEqual(ab['excess_over_marginal_product'],.25)
        self.assertEqual(ac['excess_over_marginal_product'],0.)
        json.dumps(r,allow_nan=False)

    def test_reverse_arcs_and_multiple_shared_corridors_count_pair_once(self):
        r=shared_corridor_cofailure(['A','B'],[1,1],{'A':[1,3],'B':[4,2]},
            {1:10,2:10,3:20,4:20},[[1,1],[0,0]],sample_ids=['a','b'])
        self.assertEqual(r['shared_pair_count'],1)
        self.assertEqual(r['pairs'][0]['shared_physical_corridor_ids'],[10,20])
        self.assertEqual(r['pair_scenario_count'],2)

    def test_unselected_tasks_empty_routes_and_disjoint_corridors_do_not_create_pairs(self):
        for changes in (dict(selection=[0,1,1]),
                dict(routes_by_task={'A':[],'B':[],'C':[]}),
                dict(arc_to_corridor={1:10,2:11,3:20,5:21})):
            r=self.base(**changes)
            self.assertEqual(r['shared_pair_count'],0);self.assertEqual(r['status'],'NA')
            self.assertIsNone(r['joint_failure_probability'])
            self.assertIsNone(r['marginal_product']);self.assertIsNone(r['excess_over_marginal_product'])

    def test_binary_tolerance_does_not_create_selected_population(self):
        r=self.base(selection=[1e-10,0,0])
        self.assertEqual(r['selected_task_count'],0);self.assertEqual(r['shared_pair_count'],0)

    def test_no_scenarios_and_zero_tasks_serialize_as_na_without_inventing_rates(self):
        r=self.base(failures=np.empty((0,3)),sample_ids=[])
        self.assertEqual(r['shared_pair_count'],2);self.assertEqual(r['scenario_count'],0)
        self.assertIsNone(r['pairs'][0]['joint_failure_probability'])
        empty=shared_corridor_cofailure([],[],{}, {},np.empty((2,0)),sample_ids=['a','b'])
        self.assertEqual(empty['shared_pair_count'],0)
        json.dumps(r,allow_nan=False);json.dumps(empty,allow_nan=False)

    def test_negative_excess_is_not_clipped(self):
        r=shared_corridor_cofailure(['A','B'],[1,1],{'A':[1],'B':[2]},
            {1:10,2:10},[[1,0],[0,1]],sample_ids=['a','b'])
        self.assertEqual(r['joint_failure_probability'],0.)
        self.assertEqual(r['excess_over_marginal_product'],-.25)

    def test_invalid_axes_selection_mapping_or_failures_fail_loudly(self):
        cases=[dict(selection=[.5,1,1]),dict(selection=[True,False]),
            dict(task_ids=['A','A','C']),dict(routes_by_task={'A':[1],'B':[2]}),
            dict(arc_to_corridor={1:10,2:10,3:20}),dict(failures=[[1,1,1]]),
            dict(failures=[[1,1,.1]]*4),dict(failures=[[1,1,np.nan]]*4),
            dict(sample_ids=['a','a','c','d']),dict(failure_definition='')]
        for changes in cases:
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.base(**changes)

    def test_input_hash_preserves_routes_samples_selection_and_failure_definition(self):
        original=self.base()
        for changes in (dict(sample_ids=['s1','s0','s2','s3']),dict(selection=[1,1,0]),
                dict(routes_by_task={'A':[3,1],'B':[2],'C':[5]}),
                dict(failure_definition='qualification_failure')):
            self.assertNotEqual(original['input_sha256'],self.base(**changes)['input_sha256'])


if __name__=='__main__':unittest.main()
