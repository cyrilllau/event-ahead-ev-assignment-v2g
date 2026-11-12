"""Saved real toy DRO/ARO masters and resealed numerical tampering."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import load_npz, save_npz

from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.raw_master_audit import audit_selected_master
import test_raw_plan_audit as fixtures


class RawMasterAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RawPlanAuditTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)

    def prepare(self,risk='DRO'):
        e,p = self.fixture.probability_args(risk)
        f = self.fixture.helper.helper.fixture
        args = dict(snapshot_reference=p['snapshot_reference'],expected_request_reference=p['expected_request_reference'],
            plan_dict=p['plan_dict'],iteration=p['recourse_audit']['iteration'],first_stage=e.problem.first_stage,
            recourse_loader=e.problem.recourses.__getitem__,ambiguity=e.problem.ambiguity,case=f.case,tasks=f.tasks,
            scene_loader=f.scenarios.__getitem__,helper_source_reference=self.fixture.sources['helper_source_reference'],
            q_usd=p['recourse_audit']['q_usd'])
        folder = next(Path(ref['path']).parent for ref in e.events if
            (lambda row:row['event']=='rmp' and row['iteration']==args['iteration'])(json.loads(Path(ref['path']).read_text())))
        return e,args,folder

    def test_actual_dro_and_aro_master_certificates(self):
        for risk in ('DRO','ARO'):
            with self.subTest(risk=risk):
                e,args,folder = self.prepare(risk)
                with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('audit cannot solve')):
                    result = audit_selected_master(**args)
                self.assertEqual(result['status'],'PASS_SELECTED_PLAN_RAW_MASTER_AUDIT')
                self.assertEqual(result['native_status'],'OPTIMAL')
                self.assertAlmostEqual(result['objective_usd'],self.fixture.helper.helper.fixture.expected[risk]['objective_usd'],places=7)
                self.assertEqual(result['objective_usd'],result['lower_bound_usd'])
                self.assertTrue(result['solver_bound_usable'])
                self.assertFalse(result['scientific_acceptance'])
                self.assertFalse(result['branch_and_bound_proof_replayed'])
                self.assertEqual(result['solver_calls'],0)

    def reseal(self,e,args,folder):
        self.fixture.helper.reseal_record(folder)
        args['snapshot_reference'] = self.fixture.helper.reseal_snapshot(e,args['snapshot_reference'])

    def test_resealed_matrix_changed_even_with_recomputed_model_digest(self):
        e,args,folder = self.prepare()
        a = load_npz(folder/'matrix.npz').tocsr();a.data[0] *= 2
        save_npz(folder/'matrix.npz',a)
        from evcs_experiments.raw_plan_audit import _module
        h = _module(args['helper_source_reference'],'_master_tamper_helper')
        model = json.loads((folder/'model.json').read_text())
        model['model_sha256'] = h.model_digest(model,a,h.npz(folder/'model_arrays.npz'))
        (folder/'model.json').write_text(json.dumps(model))
        solution = json.loads((folder/'solution.json').read_text());solution['model_sha256']=model['model_sha256']
        (folder/'solution.json').write_text(json.dumps(solution))
        solver = json.loads((folder/'solver.json').read_text());solver['solution']=solution;solver['certificate']['model_sha256']=model['model_sha256']
        (folder/'solver.json').write_text(json.dumps(solver))
        self.reseal(e,args,folder)
        with self.assertRaisesRegex(ValueError,'RMP primal feasibility|block coefficients'):
            audit_selected_master(**args)

    def test_resealed_native_bound_cannot_be_promoted(self):
        e,args,folder = self.prepare()
        path=folder/'solver.json';value=json.loads(path.read_text())
        value['lower_bound_usd'] += 1
        value['certificate']['solver_bound_usable']=True
        path.write_text(json.dumps(value));self.reseal(e,args,folder)
        with self.assertRaisesRegex(ValueError,'Promoted/changed native master bound'):
            audit_selected_master(**args)

    def test_resealed_raw_uncertainty_dual_change_rejected(self):
        e,args,folder = self.prepare()
        path=folder/'solution.json';value=json.loads(path.read_text());value['theta0'] += 1
        path.write_text(json.dumps(value));solver=json.loads((folder/'solver.json').read_text());solver['solution']=value
        (folder/'solver.json').write_text(json.dumps(solver));self.reseal(e,args,folder)
        with self.assertRaisesRegex(ValueError,'uncertainty dual differs'):
            audit_selected_master(**args)

    def test_limited_native_result_retains_real_status_and_nullable_bound(self):
        # A sealed fixture simulates only native terminal status metadata; the
        # incumbent/matrix/physics are the real solved toy evidence unchanged.
        e,args,folder = self.prepare()
        path=folder/'solver.json';value=json.loads(path.read_text())
        value.update(status='TIME_LIMIT',optimal=False,solver_status=1,lower_bound_usd=None)
        value['raw_solver'].update(status=1,raw_lower_bound_without_constant_usd=None,raw_mip_gap=None)
        value['certificate']['solver_bound_usable']=False
        path.write_text(json.dumps(value));self.reseal(e,args,folder)
        result=audit_selected_master(**args)
        self.assertEqual(result['native_status'],'TIME_LIMIT');self.assertIsNone(result['lower_bound_usd'])
        self.assertFalse(result['optimal']);self.assertFalse(result['solver_bound_usable'])

    def test_gurobi_metadata_bound_source_and_original_bound_must_match(self):
        # Exercise native adapter metadata using a real toy incumbent; this
        # does not claim a local Gurobi run or validate a commercial license.
        e,args,folder = self.prepare()
        path=folder/'solver.json';value=json.loads(path.read_text());raw=value['raw_solver']
        raw['solver']='gurobipy/Gurobi'
        raw['native_interface']=dict(raw_model_status=2,status='OPTIMAL',
            incumbent_without_constant_usd=value['objective_usd'],
            solver_bound_without_constant_usd=value['lower_bound_usd'],
            raw_solver_bound_without_constant_usd=value['lower_bound_usd'],
            solution_count=1,is_mip=True,bound_source='MIP_ObjBound')
        path.write_text(json.dumps(value));self.reseal(e,args,folder)
        self.assertTrue(audit_selected_master(**args)['optimal'])
        raw['native_interface']['raw_solver_bound_without_constant_usd'] -= 1.
        path.write_text(json.dumps(value));self.reseal(e,args,folder)
        with self.assertRaisesRegex(ValueError,'original bound was changed'):
            audit_selected_master(**args)

    def test_deadline_prevents_pass(self):
        e,args,folder = self.prepare()
        with patch('evcs_experiments.raw_master_audit.time.monotonic',side_effect=[100.,101.]):
            with self.assertRaisesRegex(TimeoutError,'Shared master audit deadline'):
                audit_selected_master(**args,deadline=100.5)


if __name__=='__main__': unittest.main()
