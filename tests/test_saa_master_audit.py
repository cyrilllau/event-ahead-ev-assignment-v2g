"""Independent saved SAA master audit on a real small physical extensive model."""
from dataclasses import replace
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.saa_streaming import assemble_streaming_saa
from evcs_experiments.saa_streaming_solve import solve_streaming_saa
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments import saa_master_audit as audit, native_gurobi
from evcs_experiments.evaluation_lock import file_reference


class SAAMasterAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.toy=build_golden_cases()[0]; toy=self.toy
        first=assemble_first_stage(toy.case,toy.tasks); hashes=tuple(s.scenario_sha256 for s in toy.scenarios)
        self.store=LazyRecourses(toy.scenario_ids,hashes,first.matrix_sha256,{'fixture':'physical'},
            lambda i:assemble_recourse(toy.case,toy.tasks,toy.scenarios[i]),capacity=1)
        self.model=assemble_streaming_saa(first,self.store,scenario_ids=toy.scenario_ids,scenario_sha256s=hashes,
            empirical_probabilities=toy.empirical_probabilities,output_directory=self.root/'model',
            scenario_cost_constants_usd=toy.scenario_cost_constants_usd-100,max_coefficient_bytes=50_000_000)

    def tearDown(self): self.temp.cleanup()

    def run_model(self,**overrides):
        args=dict(case=self.toy.case,tasks=self.toy.tasks,scene_loader=lambda i:self.toy.scenarios[i],
            output_directory=self.root/'solve',backend='highs',time_limit_seconds=5,
            deadline=time.monotonic()+15,evidence_reserve_seconds=5)
        args.update(overrides)
        return solve_streaming_saa(self.model,self.store,**args)

    def audit_result(self):
        return audit.audit_saa_master(model=self.model,recourses=self.store,case=self.toy.case,tasks=self.toy.tasks,
            scene_loader=self.toy.scenarios.__getitem__,result_reference=file_reference(self.root/'solve/result.json'),
            source_references=[file_reference(p) for p in (audit.__file__,native_gurobi.__file__,audit.saa_streaming.__file__,audit.recourse.__file__,audit.saa_streaming_audit.__file__,
                audit.saa_streaming_solve.__file__,audit.first_stage_check.__file__,audit.recourse_check.__file__)],
            deadline=time.monotonic()+15)

    def rewrite(self,name,mutate):
        path=self.root/'solve'/name;row=json.loads(path.read_text());mutate(row);path.write_text(json.dumps(row)+'\n')

    def test_real_original_primal_weighted_negative_constant_and_native_bound(self):
        self.run_model();result=self.audit_result()
        self.assertEqual(result['status'],'PASS_INDEPENDENT_SAA_MASTER_AUDIT')
        self.assertAlmostEqual(result['lower_bound_usd'],-69.73333333333333,places=7)
        self.assertEqual(result['original_recourse_witness_count'],3)
        self.assertFalse(result['full_support_Q_audited']);self.assertFalse(result['formal_OOS'])
        self.assertTrue(result['incumbent_physical_audited'])

    def test_wrong_offset_or_native_status_is_rejected_after_reseal(self):
        self.run_model();saved=(self.root/'solve/result.json').read_text()
        for key,value in [('lower_bound_usd',123.),('status','TIME_LIMIT'),('objective_constant_usd',0.),('normal_convergence',False)]:
            (self.root/'solve/result.json').write_text(saved)
            self.rewrite('result.json',lambda row:row.update({key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):self.audit_result()

    def test_native_time_limit_without_primal_retains_bound_and_no_plan(self):
        raw=OptimizeResult(status=1,message='limit',x=None,fun=None,mip_dual_bound=-120.,mip_gap=None,limit_status=None)
        with patch('evcs_experiments.saa_streaming_solve.milp',return_value=raw):self.run_model()
        result=self.audit_result();self.assertEqual(result['native_status'],'TIME_LIMIT')
        self.assertIsNone(result['plan_sha256']);self.assertFalse(result['incumbent_physical_audited'])
        self.assertAlmostEqual(result['lower_bound_usd'],-218.)

    def test_resealed_native_primal_violation_and_changed_matrix_fail(self):
        self.run_model();path=self.root/'solve/primal.npy';vector=np.load(path);vector[0]=.5;np.save(path,vector)
        self.rewrite('result.json',lambda row:row.update(primal=file_reference(path)))
        with self.assertRaisesRegex(ValueError,'primal'):self.audit_result()
        self.model.c[0]+=1;self.model.c.flush()
        with self.assertRaisesRegex(ValueError,'numeric bytes'):self.audit_result()

    def test_source_bound_native_gurobi_metadata_not_a_license_run(self):
        self.run_model()
        def convert(native):
            native.update(backend='gurobi',native_interface=dict(status='OPTIMAL',raw_model_status=2,gurobi_version=[12,0,3],certificate_failures=[],valid_incumbent=True,
                is_mip=True,solver_bound_consistent=True,bound_source='MIP_ObjBound',
                solver_bound_without_constant_usd=native['raw_lower_bound_without_constant_usd']))
        self.rewrite('native_result.json',convert)
        self.rewrite('result.json',lambda row:row.update(raw_native_result=file_reference(self.root/'solve/native_result.json')))
        self.assertEqual(self.audit_result()['native_status'],'OPTIMAL')
        self.rewrite('native_result.json',lambda row:row['native_interface'].update(bound_source='optimal_LP_objective'))
        self.rewrite('result.json',lambda row:row.update(raw_native_result=file_reference(self.root/'solve/native_result.json')))
        with self.assertRaisesRegex(ValueError,'native MIP bound'):self.audit_result()
        self.rewrite('native_result.json',lambda row:row['native_interface'].update(bound_source='MIP_ObjBound',gurobi_version=[12,0,2]))
        self.rewrite('result.json',lambda row:row.update(raw_native_result=file_reference(self.root/'solve/native_result.json')))
        with self.assertRaisesRegex(ValueError,'version'):self.audit_result()

    def test_native_loading_timeout_and_failed_certificate_normalization(self):
        raw=OptimizeResult(status=1,message='limit',x=None,fun=None,mip_dual_bound=None,mip_gap=None,limit_status=None)
        with patch('evcs_experiments.saa_streaming_solve.milp',return_value=raw):self.run_model()
        def loading(native):
            native.update(backend='gurobi',native_interface=dict(status='MODEL_LOADING_TIME_LIMIT',raw_model_status=1,
                gurobi_version=[12,0,3],solver_run_seconds=0.,valid_incumbent=False))
        self.rewrite('native_result.json',loading)
        self.rewrite('result.json',lambda row:row.update(raw_native_result=file_reference(self.root/'solve/native_result.json')))
        result=self.audit_result();self.assertEqual(result['raw_native_status'],'MODEL_LOADING_TIME_LIMIT')
        self.assertEqual(result['native_status'],'TIME_LIMIT');self.assertTrue(result['diagnostic_only']);self.assertIsNone(result['lower_bound_usd'])
        def failure(native):
            native.update(status=4,native_interface=dict(status='OPTIMAL',raw_model_status=2,gurobi_version=[12,0,3],
                valid_incumbent=False,certificate_failures=['Solver incumbent fails original feasibility/objective certificate']))
        self.rewrite('native_result.json',failure)
        self.rewrite('result.json',lambda row:row.update(status='SOLVER_ERROR',solver_status=4,
            raw_native_result=file_reference(self.root/'solve/native_result.json')))
        result=self.audit_result();self.assertEqual(result['raw_native_status'],'OPTIMAL')
        self.assertEqual(result['native_status'],'SOLVER_ERROR');self.assertIsNone(result['plan_sha256'])
        self.rewrite('result.json',lambda row:row.update(status='OPTIMAL',solver_status=0))
        with self.assertRaisesRegex(ValueError,'status'):self.audit_result()

    def test_missing_checker_source_cannot_use_prior_pass_flags(self):
        self.run_model()
        with self.assertRaisesRegex(ValueError,'source binding'):
            audit.audit_saa_master(model=self.model,recourses=self.store,case=self.toy.case,tasks=self.toy.tasks,
                scene_loader=self.toy.scenarios.__getitem__,result_reference=file_reference(self.root/'solve/result.json'),
                source_references=[],deadline=time.monotonic()+10)

if __name__=='__main__':unittest.main()
