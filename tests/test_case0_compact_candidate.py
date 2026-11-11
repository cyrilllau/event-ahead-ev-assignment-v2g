"""Physical compact full-P integer model and source-bound candidate checks.

Local toy optimization uses SciPy MILP only as a test fixture; production calls
only the unchanged Gurobi adapter. Native records below are explicit test doubles.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments import native_gurobi as native


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

root=get_paths().project_root
base=load('compact_test_saved_dual_fixture',root/'tests/test_case0_saved_dual_bound.py')
audit=load('compact_candidate_test',root/'scripts/audits/audit_case0_compact_candidate.py')


def test_native_result(model,tol=1e-9,vector=None):
    raw=milp(model['c'],integrality=model['integrality'],bounds=Bounds(model['lower_bounds'],model['upper_bounds']),
        constraints=LinearConstraint(model['matrix'],model['row_lower'],model['row_upper']),options={'time_limit':3.,'mip_rel_gap':.005})
    if raw.x is None:raise AssertionError(raw.message)
    v=raw.x if vector is None else vector
    arrays=native._model_arrays(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'])
    mapping=native._row_map(model['row_lower'],model['row_upper'])
    rowmap={k:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for k,v in mapping.items()}
    rowmap.update(original_rows=model['matrix'].shape[0],translated_rows=sum(map(len,mapping.values())),
        free_rows=int(np.sum(np.isneginf(model['row_lower'])&np.isposinf(model['row_upper']))),
        ranged_rows=int(np.sum(np.isfinite(model['row_lower'])&np.isfinite(model['row_upper'])&(model['row_lower']<model['row_upper']))))
    _,params=native._parameters(dict(time_limit=3.,threads=1,mip_rel_gap=.005,solver_feasibility_tolerance=tol))
    record=dict(backend='gurobipy',gurobi_version=[12,0,3],effective_options=params,requested_time_limit_seconds=3.,
        original_matrix_shape=list(model['matrix'].shape),original_matrix_nnz=model['matrix'].nnz,row_map_summary=rowmap)
    fake=SimpleNamespace(Status=2,SolCount=1,IsMIP=1,ObjVal=float(model['c']@v),ObjBound=float(raw.mip_dual_bound),MIPGap=float(raw.mip_gap),
        NodeCount=0,Runtime=.01,NumVars=len(v),NumConstrs=rowmap['translated_rows'])
    gp=SimpleNamespace(GRB=SimpleNamespace(INFINITY=1e100),GurobiError=RuntimeError)
    return native._extract_result(gp,fake,SimpleNamespace(X=v),arrays,record)


class CompactCandidateTests(unittest.TestCase):
    def setUp(self):
        base.SavedDualBoundTests.setUp(self)
        self.weighted=self.root/'weighted';self.weighted.mkdir()
        wm,_=base.audit.derive_saved_cuts(self.A,self.arrays,self.header,self.ub,self.P,self.prob,
            output=self.weighted,deadline=time.monotonic()+5,binding={})
        wr,_=base.audit.solve_relaxation_bound(wm,time_limit_seconds=3,threads=1)
        self.weighted_result=dict(kind='CASE0_SAVED_DUAL_BOUND_RESULT_V1',status='AUDITED_NUMERICAL_AUXILIARY_BOUND',
            model_sha256=self.header['model_sha256'],plan_sha256=self.ub['plan_sha256'],upper_bound_usd=self.ub['upper_bound_usd'],
            lower_bound_usd=wr['lower_bound_usd'],planning_acceptance=False)
        p=self.weighted/'result.json';p.write_text(json.dumps(self.weighted_result))
        self.binding=dict(saved_dual_result=file_reference(p),ub_result=file_reference(self.saved/'result.json'))
    def prepare(self,prob=None):
        out=self.root/'compact';out.mkdir()
        with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q')):
            return audit.prepare_epigraph(self.A,self.arrays,self.header,self.ub,self.P,self.prob if prob is None else prob,
                self.weighted_result,self.weighted,output=out,deadline=time.monotonic()+5,binding=self.binding,dual_module=base.audit)
    def test_full_ambiguity_rows_integer_domain_and_original_box_floors(self):
        model,check,first,start=self.prepare();nf=len(first['c']);S=len(self.P.support_ids);K=len(self.P.feature_ids);m=first['matrix'].shape[0]
        self.assertEqual(model['matrix'].shape,(m+2*S,nf+1+2*K));self.assertTrue(check['passed'])
        np.testing.assert_array_equal(model['integrality'][:nf],first['integrality']);self.assertGreater(int(first['integrality'].sum()),0)
        np.testing.assert_array_equal(model['matrix'][m,nf:].toarray().ravel(),np.r_[1.,-self.P.features[0],self.P.features[0]])
        np.testing.assert_array_equal(model['c'][nf:],np.r_[1.,-self.P.lower,self.P.upper])
        self.assertTrue(audit._check_vector(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],start)['passed'])
        solved=test_native_result(model);out=self.root/'result';out.mkdir()
        r=audit.evaluate_native_result(model,check,first,self.header,self.ub,self.weighted_result,solved,
            output=out,deadline=time.monotonic()+5,binding=self.binding)
        self.assertTrue(r['first_stage_numeric_passed']);self.assertEqual(r['plan_sha256'],audit.canonical_hash(r['plan']))
        self.assertIsNone(r['upper_bound_usd']);self.assertIsNone(r['gap']);self.assertFalse(r['gap_passed'] or r['planning_acceptance'])
        self.assertIsNotNone(r['lower_bound_usd']);self.assertLessEqual(r['lower_bound_usd'],self.ub['upper_bound_usd']+1e-6)
    def test_numeric_entry_uses_refined_nonbinding_integer_solver_call(self):
        identity={'fixture':'original inputs'};ip=self.root/'identity.json';ip.write_text(json.dumps(identity))
        fp=self.root/'features.npz';np.savez_compressed(fp,planning=self.P.features)
        mp=self.root/'moments.json';mp.write_text(json.dumps({'moments':{'bounds_by_rho':{'1.0':{'lower':self.P.lower.tolist(),'upper':self.P.upper.tolist()}}}}))
        cp=self.root/'calibration_manifest.json';cp.write_text(json.dumps({'context_sha256':audit.canonical_hash(identity),'arrays':file_reference(fp),'moments':file_reference(mp)}))
        refs={k:file_reference(self.producer/v) for k,v in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]}
        refs.update(self.binding,probability=file_reference(self.saved/'probability.json'),input_identity=file_reference(ip),calibration_manifest=file_reference(cp))
        def solve(c,integrality,lo,hi,A,rl,ru,*,options,log_path,start):
            self.assertEqual(options['solver_feasibility_tolerance'],1e-9);self.assertEqual(options['threads'],1)
            self.assertIsNotNone(start);self.assertGreater(np.sum(integrality),0)
            return test_native_result(dict(c=c,integrality=integrality,lower_bounds=lo,upper_bounds=hi,matrix=A,row_lower=rl,row_upper=ru,objective_constant_usd=0.))
        out=self.root/'full';out.mkdir()
        with patch.object(audit,'FEATURE_NAMES',self.P.feature_ids),patch.object(audit,'solve_milp',side_effect=solve),patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q')):
            r=audit.numeric_run(refs,out,time.monotonic()+10,3,base.hashcheck,base.audit)
        self.assertTrue(r['first_stage_numeric_passed']);self.assertIsNotNone(r['lower_bound_usd'])
        self.assertIsNone(r['upper_bound_usd']);self.assertFalse(r['planning_acceptance'])
    def test_resealed_saved_cut_or_gradient_cannot_be_reused(self):
        p=self.weighted/'gradient_000.npy';g=np.load(p);g[0]+=1.;np.save(p,g)
        with self.assertRaisesRegex(ValueError,'gradient changed'):self.prepare()
    def test_changed_probability_does_not_replace_full_risk_contract(self):
        p=copy.deepcopy(self.prob);p['probabilities'][0]+=.1
        with self.assertRaisesRegex(ValueError,'probability'):self.prepare(p)
    def test_explicit_refined_native_tolerance_audited_without_rewriting(self):
        model,check,first,_=self.prepare();solved=test_native_result(model)
        before=copy.deepcopy(solved.native_record)
        r=audit.audit_case0_native_bound(model,check,solved.native_record,native_status=solved.status,native_bound=solved.mip_dual_bound,
            incumbent=solved.x,expected_coefficient_sha256=audit.coefficient_sha256(model),fixed_upper_bound=self.ub['upper_bound_usd'],deadline=time.monotonic()+5)
        self.assertTrue(r['passed']);self.assertEqual(r['actual_feasibility_tolerance'],1e-9);self.assertEqual(before,solved.native_record)
        solved.native_record['effective_options']['FeasibilityTol']=1e-8
        with self.assertRaisesRegex(ValueError,'tolerance policy'):
            audit.audit_case0_native_bound(model,check,solved.native_record,native_status=solved.status,native_bound=solved.mip_dual_bound,
                incumbent=solved.x,expected_coefficient_sha256=audit.coefficient_sha256(model),fixed_upper_bound=self.ub['upper_bound_usd'],deadline=time.monotonic()+5)
    def test_invalid_native_integrality_is_not_candidate(self):
        model,check,first,_=self.prepare();solved=test_native_result(model);solved.x[0]=.5
        out=self.root/'bad';out.mkdir()
        with self.assertRaisesRegex(ValueError,'incumbent violates'):
            audit.evaluate_native_result(model,check,first,self.header,self.ub,self.weighted_result,solved,output=out,deadline=time.monotonic()+5,binding=self.binding)
    def test_canonical_plan_change_requires_new_q_even_below_feasibility_tolerance(self):
        model,check,first,start=self.prepare();nf=len(first['c']);v=start.copy()
        positive=np.flatnonzero(v[:nf]>1e-6);self.assertGreater(len(positive),0)
        v[positive[0]]-=1e-12;solved=test_native_result(model,vector=v)
        self.assertIsNotNone(solved.x);out=self.root/'tiny';out.mkdir()
        r=audit.evaluate_native_result(model,check,first,self.header,self.ub,self.weighted_result,solved,output=out,deadline=time.monotonic()+5,binding=self.binding)
        self.assertTrue(r['changed_plan']);self.assertLess(r['max_first_stage_difference'],1e-8)
        self.assertNotEqual(r['plan_sha256'],self.ub['plan_sha256']);self.assertTrue(r['new_candidate_requires_fresh_eight_Q'])

if __name__=='__main__':unittest.main()
