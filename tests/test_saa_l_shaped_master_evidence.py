"""Actual small-master call, original-row audit, and resealed mutation negatives."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import importlib.util
from evcs_experiments.config import get_paths
_fixture_spec=importlib.util.spec_from_file_location('master_evidence_physical_fixture',get_paths().project_root/'tests/test_recourse_template_index.py')
fixture_module=importlib.util.module_from_spec(_fixture_spec);_fixture_spec.loader.exec_module(fixture_module)
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments.saa_l_shaped_provider import StoredDevelopmentSAAProblem
from evcs_experiments.saa_l_shaped_master_evidence import DevelopmentMasterEvidence,audit_master_evidence,_load_model,_hash
from evcs_experiments.campaign_batch import file_reference

class SmallMasterEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=fixture_module.TemplateIndexTests();dest=os.environ.get('EVCS_SAA_MASTER_TEST_ARTIFACTS_ROOT')
        if dest:
            root=Path(dest).resolve();root.mkdir(parents=True,exist_ok=True)
            place=Path(tempfile.mkdtemp(prefix='physical_',dir=root))
            with patch.object(fixture_module.tempfile,'TemporaryDirectory',return_value=SimpleNamespace(name=str(place),cleanup=lambda:None)):
                cls.fixture.setUp()
        else:cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups);t=cls.fixture;f=t.fixture
        cls.problem=StoredDevelopmentSAAProblem(f.case,f.tasks,f.scenarios,f.scenario_cost_constants_usd,
            provider=t.provider(),evidence_root=t.root/'atoms',capacity=1)
        cls.sink=DevelopmentMasterEvidence(t.root/'masters');cls.calls=[];actual=kernel.milp
        def observed(c,**kw):
            prep=json.loads((cls.sink.root/f'master_{cls.sink.sequence:06d}'/'preparation.json').read_text())
            saved=_load_model(prep['matrix_reference'])
            assert np.array_equal(c,saved['c'])
            assert (kw['constraints'].A!=saved['matrix']).nnz==0
            assert kw['options']['time_limit']<prep['requested_limit_seconds']
            cls.calls.append(dict(preparation_sha256=file_reference(cls.sink.root/f'master_{cls.sink.sequence:06d}'/'preparation.json')['sha256'],
                effective_options=dict(kw['options'])))
            return actual(c,**kw)
        with patch.object(kernel,'milp',side_effect=observed):
            cls.result=kernel.run_saa_l_shaped(cls.problem,checkpoint_path=t.root/'checkpoint.json',master_sink=cls.sink,
                mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
        cls.audits=[]
        with patch.object(kernel,'build_master',side_effect=AssertionError('auditor cannot call builder')),patch.object(kernel,'milp',side_effect=AssertionError('no native repeat')),patch.object(kernel,'solve_fixed_plan',side_effect=AssertionError('no Q repeat')):
            for ref in cls.sink.records:cls.audits.append(audit_master_evidence(cls.problem,ref,deadline=time.monotonic()+30))
        if dest:
            (root/'actual_component.json').write_text(json.dumps(dict(status=cls.result['status'],lower_bound_usd=cls.result['lower_bound_usd'],
                upper_bound_usd=cls.result['upper_bound_usd'],elapsed_seconds=cls.result['elapsed_seconds'],
                master_count=len(cls.calls),actual_calls=cls.calls,master_audits=cls.audits,
                checkpoint_reference=file_reference(t.root/'checkpoint.json'),formal_admission=False),indent=2)+'\n')
    def test_actual_native_calls_original_rows_bounds_and_full_mean(self):
        self.assertEqual(self.result['status'],'CONVERGED');self.assertEqual(len(self.calls),len(self.sink.records))
        self.assertAlmostEqual(self.result['lower_bound_usd'],30.266666666666666,places=8)
        self.assertAlmostEqual(self.result['upper_bound_usd'],30.266666666666666,places=8)
        self.assertTrue(any(a['cut_rows']>0 for a in self.audits))
        for audit in self.audits:
            self.assertTrue(audit['passed']);self.assertEqual(audit['recourse_columns'],0);self.assertEqual(audit['solver_calls'],0)
            self.assertFalse(audit['formal_admission']);self.assertFalse(audit['native_branch_and_bound_proof_replayed'])
        self.assertLessEqual(max(a['native_lower_bound_usd'] for a in self.audits),self.result['upper_bound_usd']+1e-8)
    @contextmanager
    def altered(self):
        ref=self.sink.records[-1];ep=Path(ref['path']);e=json.loads(ep.read_text());pp=Path(e['preparation_reference']['path'])
        p=json.loads(pp.read_text());mp=Path(p['matrix_reference']['path']);paths=(ep,pp,mp);before=[x.read_bytes() for x in paths]
        with np.load(mp,allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
        def seal():
            with mp.open('wb') as f:np.savez_compressed(f,**m)
            p['matrix_reference']=file_reference(mp);p['coefficient_sha256']=_hash(_load_model(p['matrix_reference']))
            pp.write_text(json.dumps(p,indent=2)+'\n');e['preparation_reference']=file_reference(pp)
            e['submitted_coefficient_sha256']=p['coefficient_sha256'];ep.write_text(json.dumps(e,indent=2)+'\n')
            return file_reference(ep)
        try:yield e,p,m,seal,ref
        finally:
            for path,raw in zip(paths,before):path.write_bytes(raw)
    def test_resealed_original_matrix_weights_constants_and_cut_rows_rejected(self):
        d=len(self.problem.first_stage.c);r=self.problem.first_stage.A.shape[0]
        for mode in ('objective','first_row','constant','integrality','cut_theta','cut_intercept','cut_witness'):
            with self.subTest(mode=mode),self.altered() as (e,p,m,seal,ref):
                if mode=='objective':m['c'][d]=.5
                elif mode=='first_row':m['data'][0]+=1
                elif mode=='constant':m['lower_bounds'][d]+=1
                elif mode=='integrality':m['integrality'][0]=1-m['integrality'][0]
                elif mode=='cut_theta':m['data'][m['indptr'][r+1]-1]=2
                elif mode=='cut_intercept':m['row_lower'][r]+=1
                else:p['cuts'][0][0]['q_usd']+=1
                with self.assertRaises(ValueError):audit_master_evidence(self.problem,seal(),deadline=time.monotonic()+30)
    def test_native_status_bound_source_primal_and_original_external_pin(self):
        for mode in ('missing_bound','status','source','primal','raw_bound','joint_rewrite'):
            with self.subTest(mode=mode),self.altered() as (e,p,m,seal,original):
                if mode=='missing_bound':e['native_result']['mip_dual_bound']=None
                elif mode=='status':e['native_result']['status']=2
                elif mode=='source':e['bound_source']='incumbent_objective'
                elif mode=='primal':e['native_result']['x'][0]+=.5
                elif mode=='raw_bound':e['native_result']['mip_dual_bound']-=1
                else:e['native_result']['mip_dual_bound']=e['master_record']['lower_bound_usd']=0
                fresh=seal()
                with self.assertRaises(ValueError):audit_master_evidence(self.problem,original,deadline=time.monotonic()+30)
                if mode!='joint_rewrite':
                    with self.assertRaises(ValueError):audit_master_evidence(self.problem,fresh,deadline=time.monotonic()+30)
    def test_native_limit_bound_without_incumbent_is_not_promoted_to_optimal(self):
        # Explicit metadata control regression, not a claim that the physical tiny
        # model naturally hit the limit. The positive retained native run is real.
        with self.altered() as (e,p,m,seal,ref):
            e['native_result'].update(status=1,success=False,message='controlled LIMIT',x=None,fun=None)
            e['master_record'].update(solver_status=1,message='controlled LIMIT',primal=None,incumbent_objective_usd=None)
            a=audit_master_evidence(self.problem,seal(),deadline=time.monotonic()+30)
            self.assertEqual(a['native_status'],1);self.assertIsNone(a['incumbent_objective_usd'])
            self.assertFalse(a['fixed_plan_full_Q_audited'])
    def test_deadline_and_formal_scope_refused(self):
        with self.assertRaises(TimeoutError):audit_master_evidence(self.problem,self.sink.records[-1],deadline=time.monotonic()-1)
        with self.assertRaises(ValueError):audit_master_evidence(SimpleNamespace(domain='formal'),self.sink.records[-1],deadline=time.monotonic()+1)
