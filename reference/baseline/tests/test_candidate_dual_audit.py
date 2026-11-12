"""Hand-derived original saved LP evidence; no production input or solver call."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from scipy.sparse import csr_matrix,save_npz

from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.candidate_dual_audit import audit_candidate_atom,audit_candidate_axis
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse_store import _semantic_sha


class CandidateDualAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.t=self.root/'template';self.t.mkdir();self.e=self.root/'event';self.e.mkdir()
        self.m=dict(A_ub=csr_matrix([[-1.]]),A_eq=csr_matrix((0,1)),Bx_ub=csr_matrix([[2.]]),Bx_eq=csr_matrix((0,1)),Br_ub=csr_matrix((1,0)),Br_eq=csr_matrix((0,0)))
        self.a=dict(b_ub=np.array([-3.]),b_eq=np.array([]),c=np.array([1.]),lower_bounds=np.array([1.]),upper_bounds=np.array([np.inf]),component_objectives=np.array([[1.]]))
        meta=dict(task_sha256='toy-task',config_hash='toy-config',scenario_sha256='toy-scenario',variable_shapes={'y':[1]},task_ids=['h'],bus_ids=['b'],line_ids=[],station_ids=['s'],periods=[1],reference_bus_ids_by_period=[['b']],cmax_usd=10.)
        self.template_sha=_semantic_sha(meta,self.m,self.a,'first');meta['template_sha256']=self.template_sha
        for k,v in self.m.items():save_npz(self.t/(k+'.npz'),v)
        np.savez_compressed(self.t/'arrays.npz',**self.a);self.write(self.t/'identity.json',meta)
        self.tr=self.reseal(self.t,dict(kind='recourse_template',scenario_index=0,scenario_id='s0',template_sha256=self.template_sha))
        self.plan=dict(x=[1.],r=[[]],task_sha256='toy-task',config_hash='toy-config');self.ph=canonical_hash(self.plan)
        self.write(self.e/'plan.json',self.plan);self.write(self.e/'template_reference.json',self.tr)
        self.dual=dict(inequality=np.array([-1.]),equality=np.array([]),lower=np.array([0.]),upper=np.array([0.]),stationarity=np.array([0.]),inequality_slack=np.array([0.]),equality_residual=np.array([]),inequality_complementarity=np.array([0.]),lower_complementarity=np.array([0.]),upper_complementarity=np.array([]))
        np.savez_compressed(self.e/'dual_kkt.npz',**self.dual);np.savez_compressed(self.e/'fixed_rhs.npz',b_ub=[-1.],b_eq=[]);np.savez_compressed(self.e/'primal_vector.npz',vector=[1.])
        cert={f'dual_{k}_marginals':self.dual[k].tolist() for k in ('inequality','equality','lower','upper')}
        cert.update(dual_objective_usd=1.,dual_bound_usd=1.,primal_bound_usd=1.)
        self.write(self.e/'solver.json',dict(status='OPTIMAL',optimal=True,solver_status=0,objective_usd=1.,certificate=cert))
        self.event=dict(event='recourse',domain='formal',scenario_index=0,scenario_id='s0',plan_sha256=self.ph,q_usd=6.)
        self.er=self.reseal(self.e,self.event)
        self.prov=dict(event_reference=self.er,template_reference=self.tr,anchor_plan_reference=file_reference(self.e/'plan.json'))
        self.cut,self.g=build_dual_cut(dict(**self.m,**self.a,scenario_id='s0',template_sha256=self.template_sha),self.dual,scenario_constant_usd=5.,anchor_first_stage=[1.],anchor_q_usd=6.)

    def write(self,p,v):p.write_text(json.dumps(v)+'\n')
    def reseal(self,folder,body):
        body=copy.deepcopy(body);body['artifacts']=[file_reference(p) for p in sorted(folder.iterdir()) if p.name!='record.json']
        self.write(folder/'record.json',body);return file_reference(folder/'record.json')
    def audit(self,**changes):
        kw=dict(expected_index=0,expected_scenario_id='s0',expected_template_sha256=self.template_sha,
            expected_first_stage_sha256='first',expected_constant=5.,expected_plan_sha256=self.ph,expected_q_usd=6.,deadline=time.monotonic()+10.)
        kw.update(changes);return audit_candidate_atom(self.cut,self.g,self.prov,**kw)
    def reseal_event(self):self.prov['event_reference']=self.reseal(self.e,self.event)

    def test_hand_cut_fixed_rhs_and_nonzero_finite_bound(self):
        r=self.audit();self.assertTrue(r['passed']);self.assertEqual(r['intercept_usd'],8.)
        self.assertEqual(self.g.tolist(),[-2.]);self.assertEqual(r['q_usd'],6.)
        self.assertFalse(r['scientific_acceptance']);self.assertEqual(r['solver_calls'],0)

    def test_resealed_fixed_rhs_rejected(self):
        np.savez_compressed(self.e/'fixed_rhs.npz',b_ub=[-3.],b_eq=[]);self.reseal_event()
        with self.assertRaisesRegex(ValueError,'fixed RHS'):self.audit()

    def test_wrong_base_rhs_semantic_hash_rejected(self):
        self.a['b_ub'][0]=-1.;np.savez_compressed(self.t/'arrays.npz',**self.a)
        self.tr=self.reseal(self.t,dict(kind='recourse_template',scenario_index=0,scenario_id='s0',template_sha256=self.template_sha))
        self.write(self.e/'template_reference.json',self.tr);self.prov['template_reference']=self.tr;self.reseal_event()
        with self.assertRaisesRegex(ValueError,'semantic coefficients'):self.audit()

    def test_cut_intercept_from_fixed_rhs_cannot_double_count_anchor(self):
        self.cut['intercept_usd']=6.;self.cut['cut_sha256']=canonical_hash({k:v for k,v in self.cut.items() if k!='cut_sha256'})
        with self.assertRaisesRegex(ValueError,'base RHS'):self.audit()

    def test_wrong_scenario_anchor_constant_and_deadline(self):
        for kw in ({'expected_index':1},{'expected_scenario_id':'s1'},{'expected_plan_sha256':'wrong'},
                   {'expected_constant':6.},{'expected_q_usd':7.}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.audit(**kw)
        with self.assertRaises(TimeoutError):self.audit(deadline=time.monotonic()-1.)

    def test_coherent_raw_and_solver_dual_sign_tamper_rejected(self):
        self.dual['inequality'][0]=1.;np.savez_compressed(self.e/'dual_kkt.npz',**self.dual)
        solver=json.loads((self.e/'solver.json').read_text());solver['certificate']['dual_inequality_marginals']=[1.]
        self.write(self.e/'solver.json',solver);self.reseal_event()
        with self.assertRaisesRegex(ValueError,'KKT'):self.audit()

    def test_nonzero_original_lower_bound_dual_is_in_intercept(self):
        self.dual['inequality'][0]=0.;self.dual['lower'][0]=1.
        np.savez_compressed(self.e/'dual_kkt.npz',**self.dual)
        solver=json.loads((self.e/'solver.json').read_text())
        solver['certificate']['dual_inequality_marginals']=[0.]
        solver['certificate']['dual_lower_marginals']=[1.]
        self.write(self.e/'solver.json',solver);self.reseal_event()
        self.cut,self.g=build_dual_cut(dict(**self.m,**self.a,scenario_id='s0',template_sha256=self.template_sha),self.dual,scenario_constant_usd=5.,anchor_first_stage=[1.],anchor_q_usd=6.)
        result=self.audit();self.assertEqual(result['intercept_usd'],6.);self.assertEqual(self.g.tolist(),[0.])

    def test_native_bound_infinite_and_arithmetic_rejected(self):
        solver=json.loads((self.e/'solver.json').read_text());solver['certificate']['dual_bound_usd']=float('inf')
        self.write(self.e/'solver.json',solver);self.reseal_event()
        with self.assertRaisesRegex(ValueError,'native dual bound'):self.audit()

    def test_axis_full_order_and_original_ambiguity(self):
        record=self.audit();p=ProbabilitySet(('s0',),(),np.empty((1,0)),[],[])
        kw=dict(support_ids=('s0',),probability_set=p,expected_ambiguity_sha256=p.ambiguity_sha256,expected_plan_sha256=self.ph)
        self.assertTrue(audit_candidate_axis([record],**kw)['passed'])
        for records in ([],[record,record],[dict(record,index=1)],[dict(record,scenario_id='s1')]):
            with self.assertRaises(ValueError):audit_candidate_axis(records,**kw)
        with self.assertRaises(ValueError):audit_candidate_axis([record],**dict(kw,expected_ambiguity_sha256='other'))

if __name__=='__main__':unittest.main()
