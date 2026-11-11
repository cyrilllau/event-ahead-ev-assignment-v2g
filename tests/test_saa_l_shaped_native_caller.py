"""License-free native caller contract, not a Gurobi execution certificate.

The explicit test double solves the submitted tiny matrices with HiGHS and
passes those values through the original native extraction code. This checks
caller/evidence/original-row integration; actual Gurobi remains a server gate.
"""
import json
import os
from pathlib import Path
import tempfile
import time
import warnings
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from evcs_experiments import native_gurobi
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_native as adapter
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.saa_l_shaped_master_evidence import (
    DevelopmentMasterEvidence, NativeDevelopmentMasterEvidence, audit_master_evidence)
from evcs_experiments.validation.s03_toys import build_golden_cases


def physical_test_double(c, integrality, lower, upper, matrix, row_lower, row_upper,
                         *, options, log_path):
    """Real tiny matrix solution, explicitly simulated Gurobi API metadata."""
    began=time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=RuntimeWarning,message='Unrecognized options detected.*')
        result=milp(c,integrality=integrality,bounds=Bounds(lower,upper),
            constraints=LinearConstraint(matrix,row_lower,row_upper),
            options=dict(time_limit=options['time_limit'],mip_rel_gap=options['mip_rel_gap'],
                         threads=1,mip_feasibility_tolerance=1e-8))
    if result.status!=0:
        raise AssertionError('Tiny caller fixture must solve; not a production timeout test')
    Path(log_path).write_text('TEST DOUBLE ONLY: actual solver was HiGHS, no Gurobi license used.\n')
    seconds,parameters=native_gurobi._parameters(options)
    parameters.update(LogFile=str(log_path),TimeLimit=seconds)
    arrays=native_gurobi._model_arrays(c,integrality,lower,upper,matrix,row_lower,row_upper)
    mapping=native_gurobi._row_map(row_lower,row_upper)
    model=SimpleNamespace(Status=2,SolCount=1,IsMIP=True,ObjVal=result.fun,
        ObjBound=result.mip_dual_bound,MIPGap=result.mip_gap,NodeCount=result.mip_node_count,
        Runtime=time.monotonic()-began,NumConstrs=sum(len(v) for v in mapping.values()),NumVars=len(c))
    gp=SimpleNamespace(GurobiError=RuntimeError,GRB=SimpleNamespace(INFINITY=1e100))
    record=dict(backend='gurobipy',gurobi_version=[12,0,3],effective_options=parameters,
        requested_time_limit_seconds=seconds,log_path=str(log_path),
        original_matrix_shape=list(matrix.shape),original_matrix_nnz=int(matrix.nnz),
        load_seconds=0.,effective_run_time_limit_seconds=seconds,
        synthetic_backend_test_double=True)
    return native_gurobi._extract_result(gp,model,SimpleNamespace(X=result.x),arrays,record)


class NativeCallerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target=os.environ.get('EVCS_SAA_NATIVE_CALLER_TEST_ARTIFACTS_ROOT')
        if target:
            root=Path(target).resolve();root.mkdir(parents=True,exist_ok=True)
            cls.root=Path(tempfile.mkdtemp(prefix='caller_',dir=root))
        else:
            cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
            cls.addClassCleanup(cls.temp.cleanup)
        toy=build_golden_cases()[0]
        cls.problem=kernel.DevelopmentSAAProblem(toy.case,toy.tasks,toy.scenarios,toy.scenario_cost_constants_usd)

    def test_native_caller_and_original_row_audit_use_the_same_submitted_model(self):
        sink=NativeDevelopmentMasterEvidence(self.root/'masters',threads=1)
        checkpoint=self.root/'kernel.json'
        with patch.dict(os.environ,{k:'1' for k in adapter.BLAS_ENVIRONMENT}), \
                patch.object(native_gurobi,'solve_milp',side_effect=physical_test_double) as calls, \
                patch.object(kernel,'milp',side_effect=AssertionError('No implicit HiGHS fallback in native caller')):
            state=kernel.run_saa_l_shaped(self.problem,checkpoint_path=checkpoint,
                master_sink=sink,budget_seconds=30,wall_seconds=30)
        self.assertEqual(state['status'],'CONVERGED')
        self.assertEqual(calls.call_count,len(sink.records))
        self.assertAlmostEqual(state['lower_bound_usd'],30.266666666666666,places=7)
        self.assertAlmostEqual(state['upper_bound_usd'],30.266666666666666,places=7)
        audits=[]
        with patch.object(native_gurobi,'solve_milp',side_effect=AssertionError('Audit cannot solve')):
            for ref in sink.records:
                audit=audit_master_evidence(self.problem,ref,deadline=time.monotonic()+30)
                self.assertTrue(audit['passed'])
                self.assertEqual(audit['native_provenance'],dict(backend='gurobipy',raw_model_status=2,
                    status_name='OPTIMAL',normalized_status=0))
                self.assertFalse(audit['formal_admission'])
                audits.append(audit)
        (self.root/'caller_contract.json').write_text(json.dumps(dict(
            scope='MOCK_BACKEND_CALLER_CONTRACT_ONLY',gurobi_calls=0,actual_small_model_solver='HiGHS',
            kernel_checkpoint_reference=file_reference(checkpoint),master_audits=audits,
            formal_admission=False),indent=2)+'\n')
        with self.assertRaisesRegex(ValueError,'config changed'):
            kernel.run_saa_l_shaped(self.problem,checkpoint_path=checkpoint,resume=True,
                expected_checkpoint_sha256=file_reference(checkpoint)['sha256'],
                master_sink=DevelopmentMasterEvidence(self.root/'switched_backend'),
                budget_seconds=30,wall_seconds=30)

    def test_native_license_error_is_not_an_implicit_backend_change(self):
        sink=NativeDevelopmentMasterEvidence(self.root/'license_failure')
        with patch.dict(os.environ,{k:'1' for k in adapter.BLAS_ENVIRONMENT}), \
                patch.object(native_gurobi,'solve_milp',side_effect=RuntimeError('license unavailable')) as call, \
                patch.object(kernel,'milp',side_effect=AssertionError('No fallback')) as fallback:
            with self.assertRaisesRegex(RuntimeError,'license unavailable'):
                kernel.run_saa_l_shaped(self.problem,master_sink=sink,budget_seconds=5,wall_seconds=5)
        call.assert_called_once();fallback.assert_not_called()
        self.assertFalse(sink.records)

    def test_native_candidate_full_q_and_canonical_metadata_share_backend_identity(self):
        # Exercise the newly added native branch through the existing actual
        # full-Q/candidate/readback fixture. Its Gurobi metadata is a test double.
        import test_saa_l_shaped_candidate as candidate_fixture
        fixture=type('NativeCandidateFixture',(candidate_fixture.LShapedCandidateTests,),{})
        with patch.dict(os.environ,{k:'1' for k in adapter.BLAS_ENVIRONMENT}), \
                patch.object(native_gurobi,'solve_milp',side_effect=physical_test_double), \
                patch.object(candidate_fixture.masters,'DevelopmentMasterEvidence',NativeDevelopmentMasterEvidence):
            try:
                fixture.setUpClass()
                instance=fixture('test_genuine_two_session_convergence_and_same_plan_original_audits')
                instance.test_genuine_two_session_convergence_and_same_plan_original_audits()
                inner=json.loads(Path(fixture.final_pair[0]['path']).read_text())
                self.assertTrue(inner['master_native_provenance'])
                for item in inner['master_native_provenance']:
                    self.assertEqual(item['backend'],'gurobipy')
                    self.assertEqual(item['raw_model_status'],2)
                    self.assertEqual(item['normalized_status'],0)
                summary=dict(scope='MOCK_NATIVE_CANDIDATE_FULL_Q_READBACK',gurobi_calls=0,
                    algorithm_status=fixture.final['status'],lower_bound_usd=inner['lower_bound_usd'],
                    upper_bound_usd=inner['upper_bound_usd'],formal_admission=False,
                    canonical_metadata_passed=True,full_q_audit_passed=fixture.final_pair[2]['passed'])
                (self.root/'native_candidate_contract.json').write_text(json.dumps(summary,indent=2)+'\n')
            finally:
                fixture.doClassCleanups()


if __name__=='__main__':
    unittest.main()
