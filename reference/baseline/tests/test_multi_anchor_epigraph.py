import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.epigraph_master import build_epigraph_master, certify_box_floor
from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.relaxation_bound import coefficient_sha256


class MultiAnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def fixture(self, integer=0, moments=False):
        # Q_s(x)=max(4-4x,4x)+constant_s. Two exact dual anchors.
        constants = [-5., -1.] if moments else [-5.]
        ids = tuple('s'+str(i) for i in range(len(constants)))
        p = dict(matrix=csr_matrix([[1.]]), row_lower=[0.], row_upper=[1.],
                 c=[0.], lower_bounds=[0.], upper_bounds=[1.],
                 integrality=[integer], objective_constant_usd=3.)
        probability = ProbabilitySet(ids, ('f',) if moments else (),
            np.array([[0.], [1.]]) if moments else np.empty((1, 0)),
            [.25] if moments else [], [.25] if moments else [])
        groups, floors = [], {}
        for anchor in (0., 1.):
            cuts = {}
            for i, constant in enumerate(constants):
                template = dict(A_ub=csr_matrix([[-1.], [-1.]]), A_eq=csr_matrix((0,1)),
                    Bx_ub=csr_matrix([[4.], [-4.]]), Bx_eq=csr_matrix((0,1)),
                    Br_ub=csr_matrix((2,0)), Br_eq=csr_matrix((0,0)),
                    b_ub=[-4., 0.], b_eq=[], c=[1.], lower_bounds=[0.], upper_bounds=[10.],
                    scenario_id=ids[i], template_sha256='original-'+ids[i])
                dual = dict(inequality=[-1., 0.] if anchor==0. else [0., -1.],
                            equality=[], lower=[0.], upper=[0.])
                cuts[i] = build_dual_cut(template, dual, scenario_constant_usd=constant,
                                        anchor_first_stage=[anchor], anchor_q_usd=4.+constant)
                floors[i] = certify_box_floor(template, scenario_constant_usd=constant)
            source = Path(self.temp.name)/f'anchor-{anchor}.json'
            source.write_text(json.dumps({'synthetic_anchor': anchor}))
            groups.append(dict(anchor_first_stage=[anchor], cuts=cuts,
                               source_references=[file_reference(source)]))
        return p, probability, groups, floors

    def build(self, p, probability, groups, floors):
        return build_multi_anchor_epigraph_master(p, probability.support_ids, groups,
                                                  probability_set=probability, floors=floors)

    def optimum(self, model):
        r = milp(model['c'], integrality=model['integrality'],
                 bounds=Bounds(model['lower_bounds'], model['upper_bounds']),
                 constraints=LinearConstraint(model['matrix'], model['row_lower'], model['row_upper']),
                 options={'time_limit': 3.})
        self.assertEqual(r.status, 0)
        return r.fun+model['objective_constant_usd']

    def test_single_anchor_is_exact_old_model(self):
        p, a, groups, floors = self.fixture()
        old, _ = build_epigraph_master(p, a.support_ids, groups[0]['cuts'], probability_set=a, floors=floors)
        new, record = self.build(p, a, groups[:1], floors)
        self.assertEqual(coefficient_sha256(old), coefficient_sha256(new))
        self.assertEqual(record['appended_cut_rows'], 0)
        self.assertFalse(record['scientific_acceptance'])

    def test_hand_optimum_and_monotone_bound_integer_and_continuous(self):
        for integer, expected in [(0, 0.), (1, 2.)]:
            p, a, groups, floors = self.fixture(integer=integer)
            one, _ = self.build(p, a, groups[:1], floors)
            two, _ = self.build(p, a, groups, floors)
            self.assertAlmostEqual(self.optimum(one), -2.)
            self.assertAlmostEqual(self.optimum(two), expected)
            self.assertLessEqual(self.optimum(one), self.optimum(two))
            # Direct original recourse minimization, independent of assembled rows.
            grid = [0., 1.] if integer else [0., .5, 1.]
            self.assertEqual(min(3.+max(4.-4*x, 4*x)-5. for x in grid), expected)

    def test_moments_share_columns_and_preserve_entire_old_prefix(self):
        p, a, groups, floors = self.fixture(moments=True)
        one, _ = self.build(p, a, groups[:1], floors)
        two, record = self.build(p, a, groups, floors)
        self.assertEqual(two['matrix'].shape, (7, 4))
        np.testing.assert_array_equal(two['matrix'][:5].toarray(), one['matrix'].toarray())
        for key in ('row_lower', 'row_upper'):
            np.testing.assert_array_equal(two[key][:5], one[key])
        for key in ('c', 'lower_bounds', 'upper_bounds', 'integrality'):
            np.testing.assert_array_equal(two[key], one[key])
        self.assertEqual(record['variable_slices']['theta0'], [1,2])
        self.assertAlmostEqual(self.optimum(two), 1.) # .25 weighted +4 constant
        self.assertEqual(coefficient_sha256(p), record['original_first_stage_coefficient_sha256'])

    def test_duplicate_anchor_and_source_rejected(self):
        p, a, groups, floors = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate first-stage'):
            self.build(p, a, [groups[0], copy.deepcopy(groups[0])], floors)
        groups[0]['source_references'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate anchor source'):
            self.build(p, a, groups, floors)

    def test_source_tamper_rejected(self):
        p, a, groups, floors = self.fixture()
        Path(groups[1]['source_references'][0]['path']).write_text('changed')
        with self.assertRaises(ValueError): self.build(p, a, groups, floors)

    def test_rehashed_wrong_anchor_tightness_rejected(self):
        for kind in ('anchor', 'tightness'):
            p, a, groups, floors = self.fixture()
            cut = groups[1]['cuts'][0][0]
            if kind=='anchor': groups[1]['anchor_first_stage'] = [.5]
            else:
                cut['anchor_q_usd'] += 1.
                cut['cut_sha256'] = canonical_hash({k:v for k,v in cut.items() if k!='cut_sha256'})
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.build(p, a, groups, floors)

    def test_full_support_constants_template_and_gradient_rejected(self):
        for kind in ('missing', 'constant', 'template', 'gradient'):
            p, a, groups, floors = self.fixture(moments=True)
            cut, gradient = groups[1]['cuts'][0]
            if kind=='missing': del groups[1]['cuts'][1]
            elif kind=='gradient': gradient[0] += 1.
            else:
                cut['scenario_constant_usd' if kind=='constant' else 'template_sha256'] = 999 if kind=='constant' else 'different'
                cut['cut_sha256'] = canonical_hash({k:v for k,v in cut.items() if k!='cut_sha256'})
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.build(p, a, groups, floors)
        p, a, groups, floors = self.fixture(moments=True)
        with self.assertRaises(ValueError):
            build_multi_anchor_epigraph_master(p, tuple(reversed(a.support_ids)), groups, probability_set=a, floors=floors)


if __name__ == '__main__': unittest.main()
