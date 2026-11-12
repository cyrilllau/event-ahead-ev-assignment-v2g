"""Tiny coefficient and synthetic native metadata audits; no Gurobi license."""
import copy
import hashlib
import time
import unittest

import numpy as np

from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model,audit_native_auxiliary_bound
from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
import test_multi_anchor_epigraph as fixtures


class MultiAnchorBoundAuditTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.MultiAnchorTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.first,self.p,self.groups,self.floors=self.f.fixture(integer=1)
        self.model,_=build_multi_anchor_epigraph_master(self.first,self.p.support_ids,self.groups,probability_set=self.p,floors=self.floors)

    def audit_model(self,model=None):
        return audit_multi_anchor_model(self.model if model is None else model,self.first,self.p.support_ids,self.groups,
            probability_set=self.p,floors=self.floors,deadline=time.monotonic()+5.)

    def native(self):
        m=self.model;rl,ru=np.asarray(m['row_lower']),np.asarray(m['row_upper']);eq=np.isfinite(rl)&np.isfinite(ru)&(rl==ru)
        maps=dict(eq=np.flatnonzero(eq),le=np.flatnonzero(np.isfinite(ru)&~eq),ge=np.flatnonzero(np.isfinite(rl)&~eq))
        rows={k:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for k,v in maps.items()}
        rows.update(original_rows=len(rl),translated_rows=sum(map(len,maps.values())))
        return dict(backend='gurobipy',gurobi_version=[12,0,3],original_matrix_shape=list(m['matrix'].shape),
            original_matrix_nnz=m['matrix'].nnz,effective_options=dict(Threads=1,MIPGap=.005,Seed=0,
                FeasibilityTol=1e-8,OptimalityTol=1e-8,IntFeasTol=1e-8),requested_time_limit_seconds=600.,
            raw_model_status=2,status='OPTIMAL',certificate_failures=[],is_mip=True,bound_source='MIP_ObjBound',
            solver_bound_consistent=True,raw_solver_bound_without_constant_usd=-1.,solver_bound_without_constant_usd=-1.,
            row_map_summary=rows,model_rows=rows['translated_rows'],model_columns=m['matrix'].shape[1],
            valid_incumbent=True,objective_matches=True,solution_count=1,incumbent_without_constant_usd=-1.)

    def audit_native(self,record=None,**changes):
        ma=self.audit_model()
        kw=dict(native_status=0,native_bound=-1.,incumbent=np.array([0.,-1.]),
            expected_coefficient_sha256=ma['coefficient_sha256'],fixed_upper_bound=2.,deadline=time.monotonic()+5.)
        kw.update(changes)
        return audit_native_auxiliary_bound(self.model,ma,self.native() if record is None else record,**kw)

    def test_original_first_cuts_floors_and_constant_are_exact(self):
        result=self.audit_model();self.assertTrue(result['passed'])
        native=self.audit_native();self.assertEqual(native['lower_bound_usd'],2.)
        self.assertFalse(native['native_branch_and_bound_proof_replayed'])
        self.assertFalse(native['new_candidate_upper_bound_verified'])

    def test_coefficient_domain_floor_and_objective_tamper_rejected(self):
        for kind in ('first','cut','floor','integrality','constant','risk'):
            m=copy.deepcopy(self.model)
            if kind=='first':m['matrix'][0,0]=.5
            elif kind=='cut':m['matrix'][-1,0]+=1.
            elif kind=='floor':m['row_lower'][2]+=1.
            elif kind=='integrality':m['integrality'][0]=0
            elif kind=='constant':m['objective_constant_usd']+=1.
            else:m['c'][-1]=2.
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.audit_model(m)

    def test_nonzero_moment_probability_columns_preserved(self):
        self.first,self.p,self.groups,self.floors=self.f.fixture(integer=1,moments=True)
        self.model,_=build_multi_anchor_epigraph_master(self.first,self.p.support_ids,self.groups,probability_set=self.p,floors=self.floors)
        self.assertTrue(self.audit_model()['passed'])
        self.model['matrix'][-1,-1]*=-1.
        with self.assertRaisesRegex(ValueError,'probability columns'):self.audit_model()

    def test_time_limit_kept_and_missing_bound_rejected(self):
        nr=self.native();nr.update(raw_model_status=9,status='TIME_LIMIT',solver_bound_without_constant_usd=-2.,raw_solver_bound_without_constant_usd=-2.)
        result=self.audit_native(nr,native_status=1,native_bound=-2.)
        self.assertEqual(result['native_status'],'TIME_LIMIT');self.assertEqual(result['lower_bound_usd'],1.)
        nr['solver_bound_without_constant_usd']=None
        with self.assertRaisesRegex(ValueError,'finite native'):self.audit_native(nr,native_status=1,native_bound=None)

    def test_finite_bound_without_incumbent_is_valid(self):
        nr=self.native();nr.update(raw_model_status=9,status='TIME_LIMIT',valid_incumbent=False,
            objective_matches=False,solution_count=0,incumbent_without_constant_usd=None)
        result=self.audit_native(nr,native_status=1,incumbent=None)
        self.assertEqual(result['lower_bound_usd'],2.);self.assertIsNone(result['auxiliary_incumbent_objective_usd'])

    def test_native_rowmap_status_policy_bound_and_primal_tamper_rejected(self):
        for kind in ('rowmap','status','policy','bound','proofsource'):
            nr=self.native()
            if kind=='rowmap':nr['row_map_summary']['ge']['source_rows_sha256']='wrong'
            elif kind=='status':nr['status']='TIME_LIMIT'
            elif kind=='policy':nr['effective_options']['Threads']=4
            elif kind=='bound':nr['raw_solver_bound_without_constant_usd']=0.
            else:nr['bound_source']='stopped_LP_incumbent'
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.audit_native(nr)
        with self.assertRaisesRegex(ValueError,'violates'):self.audit_native(incumbent=np.array([.5,-1.]))
        with self.assertRaisesRegex(ValueError,'contradicts'):self.audit_native(fixed_upper_bound=1.)
        with self.assertRaisesRegex(ValueError,'coefficients changed'):self.audit_native(expected_coefficient_sha256='wrong')
        with self.assertRaises(TimeoutError):self.audit_native(deadline=time.monotonic()-1.)



    def test_optional_independent_relaxation_chain_and_tamper(self):
        from evcs_experiments.bounded_row_relaxation import relax_bounded_rows
        from evcs_experiments.bounded_row_relaxation_audit import audit_bounded_row_relaxation
        from evcs_experiments.random_streams import canonical_hash
        original=copy.deepcopy(self.model)
        # A redundant original first-stage row gets a tiny bounded-x term.
        # The synthetic model audit below isolates native-chain identity only.
        original['matrix']=original['matrix'].copy()
        original['matrix'][0,0]=1e-14
        kw=dict(allowed_rows=(0,original['matrix'].shape[0]),allowed_columns=(0,1))
        relaxed,record=relax_bounded_rows(original,**kw)
        ra=audit_bounded_row_relaxation(original,relaxed,record,deadline=time.monotonic()+5,**kw)
        ma=self.audit_model();ma.pop('audit_sha256')
        ma['coefficient_sha256']=ra['original_coefficient_sha256'];ma['audit_sha256']=canonical_hash(ma)
        self.model=relaxed
        kwargs=dict(native_status=0,native_bound=-1.,incumbent=np.array([0.,-1.]),
            expected_coefficient_sha256=ra['relaxed_coefficient_sha256'],fixed_upper_bound=2.,deadline=time.monotonic()+5)
        result=audit_native_auxiliary_bound(relaxed,ma,self.native(),relaxation_audit=ra,**kwargs)
        self.assertEqual(result['original_coefficient_sha256'],ma['coefficient_sha256'])
        self.assertEqual(result['relaxation_audit_sha256'],ra['audit_sha256'])
        with self.assertRaises(ValueError):audit_native_auxiliary_bound(relaxed,ma,self.native(),**kwargs)
        for field in ('original_coefficient_sha256','relaxed_coefficient_sha256','passed','exact_outward_fraction_proof'):
            bad=copy.deepcopy(ra);bad[field]=False;bad.pop('audit_sha256');bad['audit_sha256']=canonical_hash(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit_native_auxiliary_bound(relaxed,ma,self.native(),relaxation_audit=bad,**kwargs)
