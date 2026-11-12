"""Exact original physical SAA equivalence, bounded loading and fail-closed gates."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from evcs_experiments.extensive import assemble_extensive, solve_extensive
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.saa_streaming import assemble_streaming_saa, _reduce_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases


class StreamingSAATests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.toys=build_golden_cases()

    def tearDown(self): self.temp.cleanup()

    def prepare(self, index=0, **kwargs):
        toy=self.toys[index]; first=assemble_first_stage(toy.case,toy.tasks)
        hashes=tuple(s.scenario_sha256 for s in toy.scenarios)
        store=LazyRecourses(toy.scenario_ids,hashes,first.matrix_sha256,{'fixture':'physical'},
            lambda i:assemble_recourse(toy.case,toy.tasks,toy.scenarios[i]),capacity=1)
        args=dict(scenario_ids=toy.scenario_ids,scenario_sha256s=hashes,
            empirical_probabilities=toy.empirical_probabilities,output_directory=self.root/str(index),
            scenario_cost_constants_usd=toy.scenario_cost_constants_usd,max_coefficient_bytes=50_000_000)
        args.update(kwargs)
        return toy,store,assemble_streaming_saa(first,store,**args)

    def test_three_original_physical_extensive_goldens_and_lifted_witnesses(self):
        for i,expected in enumerate((30.266666666666666,39.,35.)):
            toy,store,model=self.prepare(i)
            result=milp(model.c,integrality=model.integrality,bounds=Bounds(model.lower_bounds,model.upper_bounds),
                constraints=LinearConstraint(model.A,model.row_lower,model.row_upper),options={'time_limit':5})
            self.assertEqual(result.status,0,result.message)
            self.assertAlmostEqual(result.fun+model.objective_constant_usd,expected,places=7)
            self.assertAlmostEqual(result.mip_dual_bound+model.objective_constant_usd,expected,places=7)
            plan=model.plan(result.x)
            self.assertTrue(check_first_stage(toy.case,toy.tasks,plan)['passed'])
            for j,scene in enumerate(toy.scenarios):
                template=store[j]; vector=model.original_recourse_vector(j,result.x)
                solution=template.unpack(vector)
                self.assertTrue(check_recourse(toy.case,toy.tasks,scene,plan,solution,
                    check_objective_bound=False)['passed'])
            self.assertEqual(store.statistics()['capacity'],1)
            self.assertEqual(len(store.statistics()['resident_indices']),1)
            self.assertLessEqual(len(model.c),len(model.first_stage.c)+sum(len(store[j].c) for j in range(len(store))))
            self.assertEqual(model.manifest['solver_calls'],0)

    def test_exact_weighted_objective_matches_original_reference_with_negative_constants(self):
        weights=np.array([.1,.2,.7]); constants=np.array([-100.,2.,-3.])
        toy,store,model=self.prepare(empirical_probabilities=weights,scenario_cost_constants_usd=constants)
        reference=assemble_extensive(model.first_stage,[store[i] for i in range(len(store))],risk_model='SAA',
            scenario_ids=toy.scenario_ids,empirical_probabilities=weights,scenario_cost_constants_usd=constants)
        gold=solve_extensive(reference,time_limit_seconds=5)
        result=milp(model.c,integrality=model.integrality,bounds=Bounds(model.lower_bounds,model.upper_bounds),
            constraints=LinearConstraint(model.A,model.row_lower,model.row_upper),options={'time_limit':5})
        self.assertEqual(result.status,0)
        self.assertAlmostEqual(result.fun+model.objective_constant_usd,gold.objective_usd,places=7)
        self.assertAlmostEqual(result.mip_dual_bound+model.objective_constant_usd,gold.lower_bound_usd,places=7)

    def test_explicit_support_weights_identity_and_formal_source_gates(self):
        for override,pattern in (({'empirical_probabilities':[1,1,1]},'weights'),
                                  ({'scenario_sha256s':['bad']*3},'identity'),
                                  ({'domain':'formal'},'source binding')):
            with self.subTest(override=override),self.assertRaisesRegex(ValueError,pattern):
                self.prepare(output_directory=self.root/str(len(list(self.root.iterdir()))),**override)

    def test_coefficient_ceiling_and_expired_shared_deadline_start_no_solver(self):
        with self.assertRaisesRegex(ValueError,'ceiling exceeded'):
            self.prepare(max_coefficient_bytes=1)
        self.assertFalse((self.root/'0'/'manifest.json').exists())
        limited=json.loads((self.root/'0'/'resource_limit.json').read_text())
        self.assertEqual(limited['completed_atoms'],1)
        self.assertEqual(limited['solver_calls'],0)
        with self.assertRaisesRegex(TimeoutError,'deadline exhausted'):
            self.prepare(output_directory=self.root/'timeout',deadline=time.monotonic()-1)
        self.assertFalse((self.root/'timeout').exists())

    def test_zero_proof_protects_shared_decisions_and_keeps_contradictory_rows(self):
        # Local x=0 may not remove shared x; y=0 is exact. Tiny z coefficient is not dropped.
        matrix=csr_matrix([[1.,0.,0.],[0.,1.,0.],[1.,0.,1e-20],[0.,0.,0.]])
        reduced,rows,cols,proof=_reduce_recourse(matrix,np.array([0.,0.,-np.inf,1.]),
            np.array([0.,0.,4.,1.]),np.zeros(3),np.ones(3),1)
        np.testing.assert_array_equal(cols,[0,2])
        np.testing.assert_array_equal(rows,[0,2,3])
        self.assertEqual(reduced[1,1],1e-20)
        self.assertEqual(reduced.shape,(3,2))
        self.assertEqual(proof[-1]['columns'],[1])

    def test_reconstruction_rejects_changed_mapping(self):
        _,_,model=self.prepare()
        mapping=model.root/model.manifest['scenarios'][0]['mapping_file']
        mapping.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'map changed'):
            model.original_recourse_vector(0,np.zeros(len(model.c)))


if __name__=='__main__': unittest.main()
